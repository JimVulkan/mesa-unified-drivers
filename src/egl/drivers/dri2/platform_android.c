/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2010-2011 Chia-I Wu <olvaffe@gmail.com>
 * Copyright (C) 2010-2011 LunarG Inc.
 *
 * Based on platform_x11, which has
 *
 * Copyright © 2011 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <inttypes.h>
#include <stdio.h>
#include "util/libdrm.h"
#include <sys/system_properties.h>
#include <drm-uapi/drm_fourcc.h>
#include <sync/sync.h>
#include <sys/types.h>

#include "util/compiler.h"
#include "util/libsync.h"
#include "util/os_file.h"
#include "util/os_misc.h"

#include "main/glconfig.h"
#include "egl_dri2.h"
#include "eglglobals.h"
#include "loader.h"
#include "loader_dri_helper.h"
#include "platform_android.h"

#include <sys/mman.h>
#include "util/u_gralloc/u_gralloc_xclipse_dump.h"
#include <sys/stat.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>

/* Xclipse bring-up instrument, off by default: MESA_XCLIPSE_PRESENT_PROBE=<n> (or the property
 * debug.mesa_xclipse_present_probe) reads back every n-th presented window buffer on the CPU,
 * after its render fence, and logs its layout and what the GPU left in it. Answers "is the black
 * screen in the buffer, or after it?". debug.mesa_xclipse_hnd_dump 1 logs each gralloc handle. */
static int
droid_present_probe_frame(void)
{
   static int frame = -2;
   if (frame == -2) {
      const char *e = getenv("MESA_XCLIPSE_PRESENT_PROBE");
      char v[PROP_VALUE_MAX] = {0};
      if (!(e && e[0]) && __system_property_get("debug.mesa_xclipse_present_probe", v) > 0)
         e = v;
      frame = e && e[0] ? atoi(e) : -1;
   }
   return frame;
}

static void
droid_present_probe(struct dri2_egl_display *dri2_dpy, struct ANativeWindowBuffer *buf,
                    int fence_fd)
{
   static int presented;
   const int every = droid_present_probe_frame();
   const int want = presented++;
   if (every <= 0 || want % every)
      return;

   struct u_gralloc_buffer_basic_info info;
   struct u_gralloc_buffer_handle h = {
      .handle = buf->handle, .hal_format = buf->format, .pixel_stride = buf->stride,
   };
   if (u_gralloc_get_buffer_basic_info(dri2_dpy->gralloc, &h, &info)) {
      _eglLog(_EGL_WARNING, "[PRESENT_PROBE] no buffer info");
      return;
   }
   int waited = -2;
   if (fence_fd >= 0)
      waited = sync_wait(fence_fd, 2000);

   const size_t size = (size_t)info.offsets[0] + (size_t)info.strides[0] * buf->height;
   uint8_t *map = mmap(NULL, size, PROT_READ, MAP_SHARED, info.fds[0], 0);
   if (map == MAP_FAILED) {
      _eglLog(_EGL_WARNING, "[PRESENT_PROBE] mmap of %zu bytes failed", size);
      return;
   }
   struct dma_buf_sync sync = {.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ};
   ioctl(info.fds[0], DMA_BUF_IOCTL_SYNC, &sync);

   const uint8_t *base = map + info.offsets[0];
   unsigned nonzero = 0, total = 0;
   int fx = -1, fy = -1;
   uint32_t first = 0;
   for (int y = 0; y < buf->height; y += 8) {
      for (int x = 0; x < buf->width; x += 8) {
         const uint32_t p = *(const uint32_t *)(base + (size_t)y * info.strides[0] + x * 4);
         total++;
         if (p & 0x00ffffff) {
            if (!nonzero) {
               fx = x;
               fy = y;
               first = p;
            }
            nonzero++;
         }
      }
   }
   const uint32_t centre =
      *(const uint32_t *)(base + (size_t)(buf->height / 2) * info.strides[0] + (buf->width / 2) * 4);
   _eglLog(_EGL_WARNING,
           "[PRESENT_PROBE] frame %d %dx%d hal_fmt %d buf_stride %d | fourcc 0x%x mod 0x%llx "
           "stride %d offset %d | fence %d wait %d | centre 0x%08x, %u/%u sampled pixels with colour, "
           "first (%d,%d) 0x%08x",
           want, buf->width, buf->height, buf->format, buf->stride, info.drm_fourcc,
           (unsigned long long)info.modifier, info.strides[0], info.offsets[0], fence_fd, waited,
           centre, nonzero, total, fx, fy, first);

   sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
   ioctl(info.fds[0], DMA_BUF_IOCTL_SYNC, &sync);
   munmap(map, size);
}

#include "dri_util.h"

static struct dri_image *
droid_create_image_from_buffer_info(
   struct dri2_egl_display *dri2_dpy, int width, int height,
   struct u_gralloc_buffer_basic_info *buf_info,
   struct u_gralloc_buffer_color_info *color_info, void *priv)
{
   unsigned error = 0;

   struct dri_image *img = dri2_from_dma_bufs(
      dri2_dpy->dri_screen_render_gpu, width, height, buf_info->drm_fourcc,
      buf_info->modifier, buf_info->fds, buf_info->num_planes,
      buf_info->strides, buf_info->offsets, color_info->yuv_color_space,
      color_info->sample_range, color_info->horizontal_siting,
      color_info->vertical_siting, 0, &error, priv);
   if (!img)
      _eglLog(_EGL_WARNING,
              "android: dma-buf import failed (error 0x%x): %dx%d fourcc 0x%x modifier 0x%" PRIx64
              " planes %d fd %d stride %d offset %d",
              error, width, height, buf_info->drm_fourcc, buf_info->modifier,
              buf_info->num_planes, buf_info->fds[0], buf_info->strides[0],
              buf_info->offsets[0]);
   return img;
}

static bool
droid_samsung_gralloc_needs_linear(const char *driver_name)
{
   if (!strcmp(driver_name, "radeonsi"))
      return true;
   if (strcmp(driver_name, "zink"))
      return false;

   static int samsung = -1;
   if (samsung < 0) {
      char v[PROP_VALUE_MAX] = {0};
      samsung = __system_property_get("ro.soc.manufacturer", v) > 0 && !strcasecmp(v, "samsung");
   }
   return samsung;
}

static struct dri_image *
droid_create_image_from_native_buffer(_EGLDisplay *disp,
                                      struct ANativeWindowBuffer *buf,
                                      void *priv)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   struct u_gralloc_buffer_basic_info buf_info;
   struct u_gralloc_buffer_color_info color_info;
   struct u_gralloc_buffer_handle gr_handle = {
      .handle = buf->handle,
      .hal_format = buf->format,
      .pixel_stride = buf->stride,
   };
   struct dri_image *img = NULL;

   u_gralloc_xclipse_dump("egl", buf->handle);

   if (u_gralloc_get_buffer_basic_info(dri2_dpy->gralloc, &gr_handle,
                                       &buf_info))
      return NULL;

   if (u_gralloc_get_buffer_color_info(dri2_dpy->gralloc, &gr_handle,
                                       &color_info))
      return NULL;

   img = droid_create_image_from_buffer_info(dri2_dpy, buf->width, buf->height,
                                             &buf_info, &color_info, priv);

   if (!img) {
      /* If dri driver is gallium virgl, real modifier info queried back from
       * CrOS info (and potentially mapper metadata if integrated later) cannot
       * get resolved and the buffer import will fail. Thus the fallback
       * behavior is preserved so that the buffer can be imported without
       * modifier info as a last resort.
       */
      buf_info.modifier = DRM_FORMAT_MOD_INVALID;
      img = droid_create_image_from_buffer_info(
         dri2_dpy, buf->width, buf->height, &buf_info, &color_info, priv);
   }

   return img;
}

static void
handle_in_fence_fd(struct dri2_egl_surface *dri2_surf, struct dri_image *img)
{
   _EGLDisplay *disp = dri2_surf->base.Resource.Display;
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);

   if (dri2_surf->in_fence_fd < 0)
      return;

   validate_fence_fd(dri2_surf->in_fence_fd);

   if (dri2_dpy->has_native_fence_fd) {
      dri2_set_in_fence_fd(img, dri2_surf->in_fence_fd);
   } else {
      sync_wait(dri2_surf->in_fence_fd, -1);
   }
}

static void
close_in_fence_fd(struct dri2_egl_surface *dri2_surf)
{
   validate_fence_fd(dri2_surf->in_fence_fd);
   if (dri2_surf->in_fence_fd >= 0)
      close(dri2_surf->in_fence_fd);
   dri2_surf->in_fence_fd = -1;
}

/* Window buffers keep their DRI image across frames. Upstream destroyed it after every
 * queueBuffer and imported the gralloc buffer again on the next dequeue: a dma-buf import, a GPU
 * VA map and unmap and a handful of other ioctls per frame, and on Samsung sgpu every VA map is a
 * page-table update the kernel flushes by hand. A buffer is recognised by its ANativeWindowBuffer
 * and the inode of its dma-buf (the cached import holds the dma-buf, so a new buffer can't reuse
 * the inode). */
static uint64_t
droid_buffer_dmabuf_ino(const struct ANativeWindowBuffer *buf)
{
   struct stat st;

   if (!buf || !buf->handle || buf->handle->numFds < 1 || fstat(buf->handle->data[0], &st))
      return 0;
   return st.st_ino;
}

static void
droid_drop_cached_image(struct dri2_egl_surface *dri2_surf, int i)
{
   if (dri2_surf->color_buffers[i].dri_image) {
      if (dri2_surf->dri_image_back == dri2_surf->color_buffers[i].dri_image)
         dri2_surf->dri_image_back = NULL;
      dri2_destroy_image(dri2_surf->color_buffers[i].dri_image);
      dri2_surf->color_buffers[i].dri_image = NULL;
   }
   dri2_surf->color_buffers[i].dmabuf_ino = 0;
}

/* Let go of the back image: destroy it unless the cache owns it. */
static void
droid_release_back_image(struct dri2_egl_surface *dri2_surf)
{
   if (!dri2_surf->dri_image_back)
      return;

   bool cached = false;
   for (int i = 0; i < dri2_surf->color_buffers_count; i++)
      cached |= dri2_surf->color_buffers[i].dri_image == dri2_surf->dri_image_back;
   if (!cached)
      dri2_destroy_image(dri2_surf->dri_image_back);
   dri2_surf->dri_image_back = NULL;
}

/* Async present. The app thread flushes the frame with an asynchronous fence and hands the
 * buffer to this thread, which exports the fence (it waits for the driver thread and the kernel
 * submission) and queues the buffer. Buffers are queued in order; at most two wait here, and
 * dequeueBuffer still paces the app like before. */
static void *
droid_present_thread_main(void *arg)
{
   struct dri2_egl_surface *dri2_surf = arg;
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(dri2_surf->base.Resource.Display);

   pthread_mutex_lock(&dri2_surf->present_mtx);
   for (;;) {
      while (dri2_surf->present_head == dri2_surf->present_tail && !dri2_surf->present_quit)
         pthread_cond_wait(&dri2_surf->present_cond, &dri2_surf->present_mtx);
      if (dri2_surf->present_head == dri2_surf->present_tail)
         break;
      const unsigned i = dri2_surf->present_head % ARRAY_SIZE(dri2_surf->present_q);
      struct ANativeWindowBuffer *buffer = dri2_surf->present_q[i].buffer;
      void *fence = dri2_surf->present_q[i].fence;
      pthread_mutex_unlock(&dri2_surf->present_mtx);

      int fence_fd = -1;
      if (fence) {
         fence_fd = dri_get_fence_fd(dri2_dpy->dri_screen_render_gpu, fence);
         dri_destroy_fence(dri2_dpy->dri_screen_render_gpu, fence);
      }
      droid_present_probe(dri2_dpy, buffer, fence_fd);
      ANativeWindow_queueBuffer(dri2_surf->window, buffer, fence_fd);

      pthread_mutex_lock(&dri2_surf->present_mtx);
      dri2_surf->present_head++;
      pthread_cond_broadcast(&dri2_surf->present_cond);
   }
   pthread_mutex_unlock(&dri2_surf->present_mtx);
   return NULL;
}

static void
droid_present_push(struct dri2_egl_surface *dri2_surf, struct ANativeWindowBuffer *buffer,
                   void *fence)
{
   if (!dri2_surf->present_started) {
      sigset_t all, saved;
      sigfillset(&all);
      pthread_sigmask(SIG_BLOCK, &all, &saved);
      dri2_surf->present_started =
         pthread_create(&dri2_surf->present_thread, NULL, droid_present_thread_main, dri2_surf) == 0;
      pthread_sigmask(SIG_SETMASK, &saved, NULL);
      if (!dri2_surf->present_started) {
         /* No thread: present synchronously, as upstream does. */
         struct dri2_egl_display *dri2_dpy = dri2_egl_display(dri2_surf->base.Resource.Display);
         int fence_fd = -1;
         if (fence) {
            fence_fd = dri_get_fence_fd(dri2_dpy->dri_screen_render_gpu, fence);
            dri_destroy_fence(dri2_dpy->dri_screen_render_gpu, fence);
         }
         ANativeWindow_queueBuffer(dri2_surf->window, buffer, fence_fd);
         dri2_surf->async_present = false;
         return;
      }
   }

   pthread_mutex_lock(&dri2_surf->present_mtx);
   while (dri2_surf->present_tail - dri2_surf->present_head >= 2)
      pthread_cond_wait(&dri2_surf->present_cond, &dri2_surf->present_mtx);
   const unsigned i = dri2_surf->present_tail % ARRAY_SIZE(dri2_surf->present_q);
   dri2_surf->present_q[i].buffer = buffer;
   dri2_surf->present_q[i].fence = fence;
   dri2_surf->present_tail++;
   pthread_cond_broadcast(&dri2_surf->present_cond);
   pthread_mutex_unlock(&dri2_surf->present_mtx);
}

/* Wait until every handed-off buffer is queued. Anything else that talks to the window in an
 * order-dependent way (swap interval, shared-buffer mode, teardown) calls this first. */
static void
droid_present_drain(struct dri2_egl_surface *dri2_surf)
{
   if (!dri2_surf->present_started)
      return;
   pthread_mutex_lock(&dri2_surf->present_mtx);
   while (dri2_surf->present_head != dri2_surf->present_tail)
      pthread_cond_wait(&dri2_surf->present_cond, &dri2_surf->present_mtx);
   pthread_mutex_unlock(&dri2_surf->present_mtx);
}

static void
droid_present_stop(struct dri2_egl_surface *dri2_surf)
{
   if (!dri2_surf->present_started)
      return;
   pthread_mutex_lock(&dri2_surf->present_mtx);
   dri2_surf->present_quit = true;
   pthread_cond_broadcast(&dri2_surf->present_cond);
   pthread_mutex_unlock(&dri2_surf->present_mtx);
   pthread_join(dri2_surf->present_thread, NULL);
   dri2_surf->present_started = false;
}

/* On by default for radeonsi (only the Xclipse runs it on Android);
 * MESA_XCLIPSE_ASYNC_PRESENT=0 or debug.mesa_xclipse_async_present 0 turns it off. */
static bool
droid_want_async_present(struct dri2_egl_display *dri2_dpy)
{
   if (!dri2_dpy->has_native_fence_fd || strcmp(dri2_dpy->driver_name, "radeonsi"))
      return false;
   const char *e = getenv("MESA_XCLIPSE_ASYNC_PRESENT");
   char v[PROP_VALUE_MAX] = {0};
   if (!(e && e[0]) && __system_property_get("debug.mesa_xclipse_async_present", v) > 0)
      e = v;
   return !(e && e[0] == '0');
}

static EGLBoolean
droid_window_dequeue_buffer(struct dri2_egl_surface *dri2_surf)
{
   int fence_fd;

   if (ANativeWindow_dequeueBuffer(dri2_surf->window, &dri2_surf->buffer,
                                   &fence_fd))
      return EGL_FALSE;

   close_in_fence_fd(dri2_surf);

   validate_fence_fd(fence_fd);

   dri2_surf->in_fence_fd = fence_fd;

   /* Record all the buffers created by ANativeWindow and update back buffer
    * for updating buffer's age in swap_buffers.
    */
   EGLBoolean updated = EGL_FALSE;
   for (int i = 0; i < dri2_surf->color_buffers_count; i++) {
      if (!dri2_surf->color_buffers[i].buffer) {
         dri2_surf->color_buffers[i].buffer = dri2_surf->buffer;
      }
      if (dri2_surf->color_buffers[i].buffer == dri2_surf->buffer) {
         dri2_surf->back = &dri2_surf->color_buffers[i];
         updated = EGL_TRUE;
         break;
      }
   }

   if (!updated) {
      /* In case of all the buffers were recreated by ANativeWindow, reset
       * the color_buffers
       */
      for (int i = 0; i < dri2_surf->color_buffers_count; i++) {
         droid_drop_cached_image(dri2_surf, i);
         dri2_surf->color_buffers[i].buffer = NULL;
         dri2_surf->color_buffers[i].age = 0;
      }
      dri2_surf->color_buffers[0].buffer = dri2_surf->buffer;
      dri2_surf->back = &dri2_surf->color_buffers[0];
   }

   return EGL_TRUE;
}

static EGLBoolean
droid_window_enqueue_buffer(_EGLDisplay *disp,
                            struct dri2_egl_surface *dri2_surf)
{
   /* Queue the buffer with stored out fence fd. The ANativeWindow or buffer
    * consumer may choose to wait for the fence to signal before accessing
    * it. If fence fd value is -1, buffer can be accessed by consumer
    * immediately. Consumer or application shouldn't rely on timestamp
    * associated with fence if the fence fd is -1.
    *
    * Ownership of fd is transferred to consumer after queueBuffer and the
    * consumer is responsible for closing it. Caller must not use the fd
    * after passing it to queueBuffer.
    */
   int fence_fd = dri2_surf->out_fence_fd;
   dri2_surf->out_fence_fd = -1;
   droid_present_probe(dri2_egl_display(disp), dri2_surf->buffer, fence_fd);
   ANativeWindow_queueBuffer(dri2_surf->window, dri2_surf->buffer, fence_fd);

   dri2_surf->buffer = NULL;
   dri2_surf->back = NULL;

   droid_release_back_image(dri2_surf);

   return EGL_TRUE;
}

static void
droid_window_cancel_buffer(struct dri2_egl_surface *dri2_surf)
{
   int ret;
   int fence_fd = dri2_surf->out_fence_fd;

   dri2_surf->out_fence_fd = -1;
   ret = ANativeWindow_cancelBuffer(dri2_surf->window, dri2_surf->buffer,
                                    fence_fd);
   dri2_surf->buffer = NULL;
   if (ret < 0) {
      _eglLog(_EGL_WARNING, "ANativeWindow_cancelBuffer failed");
      dri2_surf->base.Lost = EGL_TRUE;
   }

   close_in_fence_fd(dri2_surf);
}

static bool
droid_set_shared_buffer_mode(_EGLDisplay *disp, _EGLSurface *surf, bool mode)
{
#if ANDROID_API_LEVEL >= 24
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(surf);
   struct ANativeWindow *window = dri2_surf->window;

   assert(surf->Type == EGL_WINDOW_BIT);
   assert(_eglSurfaceHasMutableRenderBuffer(&dri2_surf->base));

   _eglLog(_EGL_DEBUG, "%s: mode=%d", __func__, mode);

   /* Shared-buffer mode presents through droid_display_shared_buffer: go synchronous. */
   droid_present_drain(dri2_surf);
   dri2_surf->async_present = false;

   if (ANativeWindow_setSharedBufferMode(window, mode)) {
      _eglLog(_EGL_WARNING,
              "failed ANativeWindow_setSharedBufferMode"
              "(window=%p, mode=%d)",
              window, mode);
      return false;
   }

   if (mode)
      dri2_surf->gralloc_usage |= dri2_dpy->front_rendering_usage;
   else
      dri2_surf->gralloc_usage &= ~dri2_dpy->front_rendering_usage;

   if (ANativeWindow_setUsage(window, dri2_surf->gralloc_usage)) {
      _eglLog(_EGL_WARNING,
              "failed ANativeWindow_setUsage(window=%p, usage=%u)", window,
              dri2_surf->gralloc_usage);
      return false;
   }

   return true;
#else
   _eglLog(_EGL_FATAL, "%s:%d: internal error: unreachable", __FILE__,
           __LINE__);
   return false;
#endif
}

static _EGLSurface *
droid_create_surface(_EGLDisplay *disp, EGLint type, _EGLConfig *conf,
                     void *native_window, const EGLint *attrib_list)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   struct dri2_egl_config *dri2_conf = dri2_egl_config(conf);
   struct dri2_egl_surface *dri2_surf;
   struct ANativeWindow *window = native_window;
   const struct dri_config *config;

   dri2_surf = calloc(1, sizeof *dri2_surf);
   if (!dri2_surf) {
      _eglError(EGL_BAD_ALLOC, "droid_create_surface");
      return NULL;
   }

   dri2_surf->in_fence_fd = -1;

   if (!dri2_init_surface(&dri2_surf->base, disp, type, conf, attrib_list, true,
                          native_window))
      goto cleanup_surface;

   if (type == EGL_WINDOW_BIT && droid_want_async_present(dri2_dpy)) {
      /* The swap makes its own fence; the generic out-fence path stays off. */
      dri2_surf->async_present = true;
      dri2_surf->enable_out_fence = false;
      pthread_mutex_init(&dri2_surf->present_mtx, NULL);
      pthread_cond_init(&dri2_surf->present_cond, NULL);
   }

   if (type == EGL_WINDOW_BIT) {
      int format;
      int buffer_count;
      int min_undequeued_buffers;

      /* Android's libEGL connects the window as its producer before calling the driver. An app
       * that loads this library directly (Minecraft launchers) has no such wrapper, and every
       * dequeueBuffer then fails with "BufferQueue has no connected producer". Connect here;
       * -EINVAL means someone already did. */
      const int cr = droid_native_window_api_connect(window);
      if (cr == 0) {
         dri2_surf->window_connected = true;
      } else if (cr != -EINVAL) {
         _eglError(EGL_BAD_NATIVE_WINDOW, "droid_create_surface: api_connect");
         goto cleanup_surface;
      }

      format = ANativeWindow_getFormat(window);
      if (format < 0) {
         _eglError(EGL_BAD_NATIVE_WINDOW, "droid_create_surface");
         goto cleanup_surface;
      }

      /* Query ANativeWindow for MIN_UNDEQUEUED_BUFFER, minimum amount
       * of undequeued buffers.
       */
      if (ANativeWindow_query(window,
                              ANATIVEWINDOW_QUERY_MIN_UNDEQUEUED_BUFFERS,
                              &min_undequeued_buffers)) {
         _eglError(EGL_BAD_NATIVE_WINDOW, "droid_create_surface");
         goto cleanup_surface;
      }

      /* Required buffer caching slots. */
      buffer_count = min_undequeued_buffers + 2;

      dri2_surf->color_buffers =
         calloc(buffer_count, sizeof(*dri2_surf->color_buffers));
      if (!dri2_surf->color_buffers) {
         _eglError(EGL_BAD_ALLOC, "droid_create_surface");
         goto cleanup_surface;
      }
      dri2_surf->color_buffers_count = buffer_count;

      if (format != dri2_conf->base.NativeVisualID) {
         _eglLog(_EGL_WARNING, "Native format mismatch: 0x%x != 0x%x", format,
                 dri2_conf->base.NativeVisualID);
      }

      ANativeWindow_query(window, ANATIVEWINDOW_QUERY_DEFAULT_WIDTH,
                          &dri2_surf->base.Width);
      ANativeWindow_query(window, ANATIVEWINDOW_QUERY_DEFAULT_HEIGHT,
                          &dri2_surf->base.Height);

      dri2_surf->gralloc_usage =
         ((strcmp(dri2_dpy->driver_name, "kms_swrast") == 0) ||
          (strcmp(dri2_dpy->driver_name, "swrast") == 0))
            ? GRALLOC_USAGE_SW_READ_OFTEN | GRALLOC_USAGE_SW_WRITE_OFTEN
            : GRALLOC_USAGE_HW_RENDER;

      /* Arm's gralloc (kbase Malis) hands a render buffer back AFBC-compressed in a layout no
       * driver can read; MALI_GRALLOC_USAGE_NO_AFBC (private bit 29) and composer usage keep it
       * linear on both the Exynos and the MediaTek allocator (see panvk_android.c). */
      if (loader_fd_is_kbase(dri2_dpy->fd_render_gpu))
         dri2_surf->gralloc_usage |= (1u << 29) | GRALLOC_USAGE_HW_COMPOSER;
      /* Samsung gralloc gives a GPU+composer-only buffer a non-linear layout (a zero byte stride,
       * dimensions padded to 64) that the composer decodes as such, while the drivers here import
       * it as linear rows (the Samsung handle parser in u_gralloc_fallback), so the screen stays
       * black. Any CPU usage bit makes gralloc allocate plain linear rows. radeonsi on Android
       * only exists on Xclipse; zink gets the same buffers on any Samsung phone. */
      if (droid_samsung_gralloc_needs_linear(dri2_dpy->driver_name))
         dri2_surf->gralloc_usage |= GRALLOC_USAGE_SW_READ_RARELY | GRALLOC_USAGE_SW_WRITE_RARELY;

      if (dri2_surf->base.ActiveRenderBuffer == EGL_SINGLE_BUFFER)
         dri2_surf->gralloc_usage |= dri2_dpy->front_rendering_usage;

      if (ANativeWindow_setUsage(window, dri2_surf->gralloc_usage)) {
         _eglError(EGL_BAD_NATIVE_WINDOW, "droid_create_surface");
         goto cleanup_surface;
      }
   }

   config = dri2_get_dri_config(dri2_conf, type, dri2_surf->base.GLColorspace);
   if (!config) {
      _eglError(EGL_BAD_MATCH,
                "Unsupported surfacetype/colorspace configuration");
      goto cleanup_surface;
   }

   if (!dri2_create_drawable(dri2_dpy, config, dri2_surf, dri2_surf))
      goto cleanup_surface;

   if (window) {
      ANativeWindow_acquire(window);
      dri2_surf->window = window;
   }

   return &dri2_surf->base;

cleanup_surface:
   if (dri2_surf->window_connected)
      droid_native_window_api_disconnect(window);
   if (dri2_surf->color_buffers_count)
      free(dri2_surf->color_buffers);
   free(dri2_surf);

   return NULL;
}

static _EGLSurface *
droid_create_window_surface(_EGLDisplay *disp, _EGLConfig *conf,
                            void *native_window, const EGLint *attrib_list)
{
   return droid_create_surface(disp, EGL_WINDOW_BIT, conf, native_window,
                               attrib_list);
}

static _EGLSurface *
droid_create_pbuffer_surface(_EGLDisplay *disp, _EGLConfig *conf,
                             const EGLint *attrib_list)
{
   return droid_create_surface(disp, EGL_PBUFFER_BIT, conf, NULL, attrib_list);
}

static EGLBoolean
droid_destroy_surface(_EGLDisplay *disp, _EGLSurface *surf)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(surf);

   if (dri2_surf->base.Type == EGL_WINDOW_BIT) {
      droid_present_stop(dri2_surf);
      if (dri2_surf->buffer)
         droid_window_cancel_buffer(dri2_surf);

      if (dri2_surf->window_connected)
         droid_native_window_api_disconnect(dri2_surf->window);
      ANativeWindow_release(dri2_surf->window);
   }

   droid_release_back_image(dri2_surf);
   for (int i = 0; i < dri2_surf->color_buffers_count; i++)
      droid_drop_cached_image(dri2_surf, i);

   if (dri2_surf->dri_image_front) {
      _eglLog(_EGL_DEBUG, "%s : %d : destroy dri_image_front", __func__,
              __LINE__);
      dri2_destroy_image(dri2_surf->dri_image_front);
      dri2_surf->dri_image_front = NULL;
   }

   driDestroyDrawable(dri2_surf->dri_drawable);

   close_in_fence_fd(dri2_surf);
   dri2_fini_surface(surf);
   free(dri2_surf->color_buffers);
   free(dri2_surf);

   return EGL_TRUE;
}

static EGLBoolean
droid_swap_interval(_EGLDisplay *disp, _EGLSurface *surf, EGLint interval)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(surf);
   struct ANativeWindow *window = dri2_surf->window;

   droid_present_drain(dri2_surf);
   if (ANativeWindow_setSwapInterval(window, interval))
      return EGL_FALSE;

   surf->SwapInterval = interval;
   return EGL_TRUE;
}

static int
update_buffers(struct dri2_egl_surface *dri2_surf)
{
   if (dri2_surf->base.Lost)
      return -1;

   if (dri2_surf->base.Type != EGL_WINDOW_BIT)
      return 0;

   /* Windows can cap the dequeued-buffer count at one, so the previous frame has to be queued
    * first. Apps dequeue at their first draw into the window (games: the final blit), so the
    * present thread has had most of a frame to finish. */
   if (!dri2_surf->buffer)
      droid_present_drain(dri2_surf);

   /* try to dequeue the next back buffer */
   if (!dri2_surf->buffer && !droid_window_dequeue_buffer(dri2_surf)) {
      _eglLog(_EGL_WARNING, "Could not dequeue buffer from native window");
      dri2_surf->base.Lost = EGL_TRUE;
      return -1;
   }

   /* free outdated buffers and update the surface size */
   if (dri2_surf->base.Width != dri2_surf->buffer->width ||
       dri2_surf->base.Height != dri2_surf->buffer->height) {
      dri2_surf->base.Width = dri2_surf->buffer->width;
      dri2_surf->base.Height = dri2_surf->buffer->height;
   }

   return 0;
}

static int
get_front_bo(struct dri2_egl_surface *dri2_surf, unsigned int format)
{
   struct dri2_egl_display *dri2_dpy =
      dri2_egl_display(dri2_surf->base.Resource.Display);

   if (dri2_surf->dri_image_front)
      return 0;

   if (dri2_surf->base.Type == EGL_WINDOW_BIT) {
      /* According current EGL spec, front buffer rendering
       * for window surface is not supported now.
       * and mesa doesn't have the implementation of this case.
       * Add warning message, but not treat it as error.
       */
      _eglLog(
         _EGL_DEBUG,
         "DRI driver requested unsupported front buffer for window surface");
   } else if (dri2_surf->base.Type == EGL_PBUFFER_BIT) {
      dri2_surf->dri_image_front = dri_create_image(
         dri2_dpy->dri_screen_render_gpu, dri2_surf->base.Width,
         dri2_surf->base.Height, format, NULL, 0, 0, NULL);
      if (!dri2_surf->dri_image_front) {
         _eglLog(_EGL_WARNING, "dri2_image_front allocation failed");
         return -1;
      }
   }

   return 0;
}

static int
get_back_bo(struct dri2_egl_surface *dri2_surf)
{
   _EGLDisplay *disp = dri2_surf->base.Resource.Display;

   if (dri2_surf->dri_image_back)
      return 0;

   if (dri2_surf->base.Type == EGL_WINDOW_BIT) {
      if (!dri2_surf->buffer) {
         _eglLog(_EGL_WARNING, "Could not get native buffer");
         return -1;
      }

      const uint64_t ino = droid_buffer_dmabuf_ino(dri2_surf->buffer);
      if (dri2_surf->back && dri2_surf->back->dri_image && ino &&
          dri2_surf->back->dmabuf_ino == ino) {
         dri2_surf->dri_image_back = dri2_surf->back->dri_image;
      } else {
         if (dri2_surf->back)
            droid_drop_cached_image(dri2_surf, dri2_surf->back - dri2_surf->color_buffers);

         dri2_surf->dri_image_back =
            droid_create_image_from_native_buffer(disp, dri2_surf->buffer, NULL);
         if (!dri2_surf->dri_image_back) {
            _eglLog(_EGL_WARNING, "failed to create DRI image from FD");
            return -1;
         }
         if (dri2_surf->back && ino) {
            dri2_surf->back->dri_image = dri2_surf->dri_image_back;
            dri2_surf->back->dmabuf_ino = ino;
         }
      }

      handle_in_fence_fd(dri2_surf, dri2_surf->dri_image_back);

   } else if (dri2_surf->base.Type == EGL_PBUFFER_BIT) {
      /* The EGL 1.5 spec states that pbuffers are single-buffered.
       * Specifically, the spec states that they have a back buffer but no front
       * buffer, in contrast to pixmaps, which have a front buffer but no back
       * buffer.
       *
       * Single-buffered surfaces with no front buffer confuse Mesa; so we
       * deviate from the spec, following the precedent of Mesa's EGL X11
       * platform. The X11 platform correctly assigns pbuffers to
       * single-buffered configs, but assigns the pbuffer a front buffer instead
       * of a back buffer.
       *
       * Pbuffers in the X11 platform mostly work today, so let's just copy its
       * behavior instead of trying to fix (and hence potentially breaking) the
       * world.
       */
      _eglLog(
         _EGL_DEBUG,
         "DRI driver requested unsupported back buffer for pbuffer surface");
   }

   return 0;
}

/* Some drivers will pass multiple bits in buffer_mask.
 * For such case, will go through all the bits, and
 * will not return error when unsupported buffer is requested, only
 * return error when the allocation for supported buffer failed.
 */
static int
droid_image_get_buffers(struct dri_drawable *driDrawable, unsigned int format,
                        uint32_t *stamp, void *loaderPrivate,
                        uint32_t buffer_mask, struct __DRIimageList *images)
{
   struct dri2_egl_surface *dri2_surf = loaderPrivate;

   images->image_mask = 0;
   images->front = NULL;
   images->back = NULL;

   if (update_buffers(dri2_surf) < 0)
      return 0;

   if (_eglSurfaceInSharedBufferMode(&dri2_surf->base)) {
      if (get_back_bo(dri2_surf) < 0)
         return 0;

      /* We have dri_image_back because this is a window surface and
       * get_back_bo() succeeded.
       */
      assert(dri2_surf->dri_image_back);
      images->back = dri2_surf->dri_image_back;
      images->image_mask |= __DRI_IMAGE_BUFFER_SHARED;

      /* There exists no accompanying back nor front buffer. */
      return 1;
   }

   if (buffer_mask & __DRI_IMAGE_BUFFER_FRONT) {
      if (get_front_bo(dri2_surf, format) < 0)
         return 0;

      if (dri2_surf->dri_image_front) {
         images->front = dri2_surf->dri_image_front;
         images->image_mask |= __DRI_IMAGE_BUFFER_FRONT;
      }
   }

   if (buffer_mask & __DRI_IMAGE_BUFFER_BACK) {
      if (get_back_bo(dri2_surf) < 0)
         return 0;

      if (dri2_surf->dri_image_back) {
         images->back = dri2_surf->dri_image_back;
         images->image_mask |= __DRI_IMAGE_BUFFER_BACK;
      }
   }

   return 1;
}

static EGLint
droid_query_buffer_age(_EGLDisplay *disp, _EGLSurface *surface)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(surface);

   if (update_buffers(dri2_surf) < 0) {
      _eglError(EGL_BAD_ALLOC, "droid_query_buffer_age");
      return -1;
   }

   return dri2_surf->back ? dri2_surf->back->age : 0;
}

static EGLBoolean
droid_swap_buffers(_EGLDisplay *disp, _EGLSurface *draw)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(draw);
   const bool has_mutable_rb = _eglSurfaceHasMutableRenderBuffer(draw);

   /* From the EGL_KHR_mutable_render_buffer spec (v12):
    *
    *    If surface is a single-buffered window, pixmap, or pbuffer surface
    *    for which there is no pending change to the EGL_RENDER_BUFFER
    *    attribute, eglSwapBuffers has no effect.
    */
   if (has_mutable_rb && draw->RequestedRenderBuffer == EGL_SINGLE_BUFFER &&
       draw->ActiveRenderBuffer == EGL_SINGLE_BUFFER) {
      _eglLog(_EGL_DEBUG, "%s: remain in shared buffer mode", __func__);
      return EGL_TRUE;
   }

   for (int i = 0; i < dri2_surf->color_buffers_count; i++) {
      if (dri2_surf->color_buffers[i].age > 0)
         dri2_surf->color_buffers[i].age++;
   }

   /* "XXX: we don't use get_back_bo() since it causes regressions in
    * several dEQP tests.
    */
   if (dri2_surf->back)
      dri2_surf->back->age = 1;

   if (dri2_surf->async_present && !dri2_dpy->pure_swrast) {
      /* One asynchronous flush for the frame and its fence; the present thread exports the fence
       * and queues the buffer. */
      _EGLContext *ctx = _eglGetCurrentContext();
      void *fence = dri_flush_swap_with_fence(dri2_egl_context(ctx)->dri_context,
                                              dri2_surf->dri_drawable,
                                              __DRI2_FLUSH_DRAWABLE |
                                                 __DRI2_FLUSH_INVALIDATE_ANCILLARY,
                                              __DRI2_NOTHROTTLE_SWAPBUFFER);
      if (dri2_surf->buffer) {
         droid_present_push(dri2_surf, dri2_surf->buffer, fence);
         dri2_surf->buffer = NULL;
         dri2_surf->back = NULL;
         droid_release_back_image(dri2_surf);
      } else if (fence) {
         dri_destroy_fence(dri2_dpy->dri_screen_render_gpu, fence);
      }
      dri_invalidate_drawable(dri2_surf->dri_drawable);
      goto swapped;
   }

   dri2_flush_drawable_for_swapbuffers_flags(disp, draw,
                                             __DRI2_NOTHROTTLE_SWAPBUFFER);

   if (dri2_dpy->pure_swrast) {
      driSwapBuffers(dri2_surf->dri_drawable);
      if (dri2_surf->buffer)
         droid_window_enqueue_buffer(disp, dri2_surf);
   } else {
      /* dri2_surf->buffer can be null even when no error has occurred. For
       * example, if the user has called no GL rendering commands since the
       * previous eglSwapBuffers, then the driver may have not triggered
       * a callback to ANativeWindow_dequeueBuffer, in which case
       * dri2_surf->buffer remains null.
       */
      if (dri2_surf->buffer)
         droid_window_enqueue_buffer(disp, dri2_surf);

      dri_invalidate_drawable(dri2_surf->dri_drawable);
   }

swapped:
   /* Update the shared buffer mode */
   if (has_mutable_rb &&
       draw->ActiveRenderBuffer != draw->RequestedRenderBuffer) {
      bool mode = (draw->RequestedRenderBuffer == EGL_SINGLE_BUFFER);
      _eglLog(_EGL_DEBUG, "%s: change to shared buffer mode %d", __func__,
              mode);

      if (!droid_set_shared_buffer_mode(disp, draw, mode))
         return EGL_FALSE;
      draw->ActiveRenderBuffer = draw->RequestedRenderBuffer;
   }

   return EGL_TRUE;
}

static EGLBoolean
droid_query_surface(_EGLDisplay *disp, _EGLSurface *surf, EGLint attribute,
                    EGLint *value)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(surf);
   switch (attribute) {
   case EGL_WIDTH:
      if (dri2_surf->base.Type == EGL_WINDOW_BIT && dri2_surf->window) {
         ANativeWindow_query(dri2_surf->window,
                             ANATIVEWINDOW_QUERY_DEFAULT_WIDTH, value);
         return EGL_TRUE;
      }
      break;
   case EGL_HEIGHT:
      if (dri2_surf->base.Type == EGL_WINDOW_BIT && dri2_surf->window) {
         ANativeWindow_query(dri2_surf->window,
                             ANATIVEWINDOW_QUERY_DEFAULT_HEIGHT, value);
         return EGL_TRUE;
      }
      break;
   default:
      break;
   }
   return _eglQuerySurface(disp, surf, attribute, value);
}

static _EGLImage *
dri2_create_image_android_native_buffer(_EGLDisplay *disp, _EGLContext *ctx,
                                        struct ANativeWindowBuffer *buf)
{
   if (ctx != NULL) {
      /* From the EGL_ANDROID_image_native_buffer spec:
       *
       *     * If <target> is EGL_NATIVE_BUFFER_ANDROID and <ctx> is not
       *       EGL_NO_CONTEXT, the error EGL_BAD_CONTEXT is generated.
       */
      _eglError(EGL_BAD_CONTEXT,
                "eglCreateEGLImageKHR: for "
                "EGL_NATIVE_BUFFER_ANDROID, the context must be "
                "EGL_NO_CONTEXT");
      return NULL;
   }

   if (!buf || buf->common.magic != ANDROID_NATIVE_BUFFER_MAGIC ||
       buf->common.version != sizeof(*buf)) {
      _eglError(EGL_BAD_PARAMETER, "eglCreateEGLImageKHR");
      return NULL;
   }

   struct dri_image *dri_image =
      droid_create_image_from_native_buffer(disp, buf, buf);

   if (dri_image) {
#if ANDROID_API_LEVEL >= 26
      AHardwareBuffer_acquire(ANativeWindowBuffer_getHardwareBuffer(buf));
#endif
      return dri2_create_image_from_dri(disp, dri_image);
   }

   return NULL;
}

static _EGLImage *
droid_create_image_khr(_EGLDisplay *disp, _EGLContext *ctx, EGLenum target,
                       EGLClientBuffer buffer, const EGLint *attr_list)
{
   switch (target) {
   case EGL_NATIVE_BUFFER_ANDROID:
      return dri2_create_image_android_native_buffer(
         disp, ctx, (struct ANativeWindowBuffer *)buffer);
   default:
      return dri2_create_image_khr(disp, ctx, target, buffer, attr_list);
   }
}

static void
droid_flush_front_buffer(struct dri_drawable *driDrawable, void *loaderPrivate)
{
}

static unsigned
droid_get_capability(void *loaderPrivate, enum dri_loader_cap cap)
{
   /* Note: loaderPrivate is _EGLDisplay* */
   switch (cap) {
   case DRI_LOADER_CAP_RGBA_ORDERING:
      return 1;
   default:
      return 0;
   }
}

static void
droid_destroy_loader_image_state(void *loaderPrivate)
{
#if ANDROID_API_LEVEL >= 26
   if (loaderPrivate) {
      AHardwareBuffer_release(
         ANativeWindowBuffer_getHardwareBuffer(loaderPrivate));
   }
#endif
}

static void
droid_add_configs_for_visuals(_EGLDisplay *disp)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   static const struct {
      int hal_format;
      enum pipe_format pipe_format;
   } visuals[] = {
      {HAL_PIXEL_FORMAT_RGBA_8888, PIPE_FORMAT_RGBA8888_UNORM},
      {HAL_PIXEL_FORMAT_RGBX_8888, PIPE_FORMAT_RGBX8888_UNORM},
      {HAL_PIXEL_FORMAT_RGB_565, PIPE_FORMAT_B5G6R5_UNORM},
      /* This must be after HAL_PIXEL_FORMAT_RGBA_8888, we only keep BGRA
       * visual if it turns out RGBA visual is not available.
       */
      {HAL_PIXEL_FORMAT_BGRA_8888, PIPE_FORMAT_BGRA8888_UNORM},
   };

   unsigned int format_count[ARRAY_SIZE(visuals)] = {0};

   /* The nesting of loops is significant here. Also significant is the order
    * of the HAL pixel formats. Many Android apps (such as Google's official
    * NDK GLES2 example app), and even portions the core framework code (such
    * as SystemServiceManager in Nougat), incorrectly choose their EGLConfig.
    * They neglect to match the EGLConfig's EGL_NATIVE_VISUAL_ID against the
    * window's native format, and instead choose the first EGLConfig whose
    * channel sizes match those of the native window format while ignoring the
    * channel *ordering*.
    *
    * We can detect such buggy clients in logcat when they call
    * eglCreateSurface, by detecting the mismatch between the EGLConfig's
    * format and the window's format.
    *
    * As a workaround, we generate EGLConfigs such that all EGLConfigs for HAL
    * pixel format i precede those for HAL pixel format i+1. In my
    * (chadversary) testing on Android Nougat, this was good enough to pacify
    * the buggy clients.
    */
   bool has_rgba = false;
   for (int i = 0; i < ARRAY_SIZE(visuals); i++) {
      /* Only enable BGRA configs when RGBA is not available. BGRA configs are
       * buggy on stock Android.
       */
      if (visuals[i].hal_format == HAL_PIXEL_FORMAT_BGRA_8888 && has_rgba)
         continue;
      for (int j = 0; dri2_dpy->driver_configs[j]; j++) {
         const struct gl_config *gl_config =
            (struct gl_config *) dri2_dpy->driver_configs[j];

         /* Rather than have duplicate table entries for _SRGB formats, just
          * use the linear version of the format for the comparision:
          */
         enum pipe_format linear_format =
            util_format_linear(gl_config->color_format);
         if (linear_format != visuals[i].pipe_format)
            continue;

         const EGLint surface_type = EGL_WINDOW_BIT | EGL_PBUFFER_BIT;
         const EGLint config_attrs[] = {
            EGL_NATIVE_VISUAL_ID,
            visuals[i].hal_format,
            EGL_NATIVE_VISUAL_TYPE,
            visuals[i].hal_format,
            EGL_FRAMEBUFFER_TARGET_ANDROID,
            EGL_TRUE,
            EGL_RECORDABLE_ANDROID,
            EGL_TRUE,
            EGL_NONE,
         };

         struct dri2_egl_config *dri2_conf = dri2_add_config(
            disp, dri2_dpy->driver_configs[j], surface_type, config_attrs);
         if (dri2_conf)
            format_count[i]++;
      }

      if (visuals[i].hal_format == HAL_PIXEL_FORMAT_RGBA_8888 && format_count[i])
         has_rgba = true;
   }

   for (int i = 0; i < ARRAY_SIZE(format_count); i++) {
      if (!format_count[i]) {
         _eglLog(_EGL_DEBUG, "No DRI config supports native format 0x%x",
                 visuals[i].hal_format);
      }
   }
}

static const struct dri2_egl_display_vtbl droid_display_vtbl = {
   .authenticate = NULL,
   .create_window_surface = droid_create_window_surface,
   .create_pbuffer_surface = droid_create_pbuffer_surface,
   .destroy_surface = droid_destroy_surface,
   .create_image = droid_create_image_khr,
   .swap_buffers = droid_swap_buffers,
   .swap_interval = droid_swap_interval,
   .query_buffer_age = droid_query_buffer_age,
   .query_surface = droid_query_surface,
   .get_dri_drawable = dri2_surface_get_dri_drawable,
   .set_shared_buffer_mode = droid_set_shared_buffer_mode,
};

static const __DRIimageLoaderExtension droid_image_loader_extension = {
   .base = {__DRI_IMAGE_LOADER, 4},

   .getBuffers = droid_image_get_buffers,
   .flushFrontBuffer = droid_flush_front_buffer,
   .getCapability = droid_get_capability,
   .flushSwapBuffers = NULL,
   .destroyLoaderImageState = droid_destroy_loader_image_state,
};

static void
droid_display_shared_buffer(struct dri_drawable *driDrawable, int fence_fd,
                            void *loaderPrivate)
{
   struct dri2_egl_surface *dri2_surf = loaderPrivate;
   struct ANativeWindowBuffer *old_buffer UNUSED = dri2_surf->buffer;

   if (!_eglSurfaceInSharedBufferMode(&dri2_surf->base)) {
      _eglLog(_EGL_WARNING, "%s: internal error: buffer is not shared",
              __func__);
      return;
   }

   if (fence_fd >= 0) {
      /* The driver's fence is more recent than the surface's out fence, if it
       * exists at all. So use the driver's fence.
       */
      if (dri2_surf->out_fence_fd >= 0) {
         close(dri2_surf->out_fence_fd);
         dri2_surf->out_fence_fd = -1;
      }
   } else if (dri2_surf->out_fence_fd >= 0) {
      fence_fd = dri2_surf->out_fence_fd;
      dri2_surf->out_fence_fd = -1;
   }

   if (ANativeWindow_queueBuffer(dri2_surf->window, dri2_surf->buffer,
                                 fence_fd)) {
      _eglLog(_EGL_WARNING, "%s: ANativeWindow_queueBuffer failed", __func__);
      close(fence_fd);
      return;
   }

   fence_fd = -1;

   if (ANativeWindow_dequeueBuffer(dri2_surf->window, &dri2_surf->buffer,
                                   &fence_fd)) {
      /* Tear down the surface because it no longer has a back buffer. */
      _eglLog(_EGL_WARNING, "%s: ANativeWindow_dequeueBuffer failed", __func__);

      dri2_surf->base.Lost = true;
      dri2_surf->buffer = NULL;
      dri2_surf->back = NULL;

      droid_release_back_image(dri2_surf);

      dri_invalidate_drawable(dri2_surf->dri_drawable);
      return;
   }

   close_in_fence_fd(dri2_surf);
   validate_fence_fd(fence_fd);
   dri2_surf->in_fence_fd = fence_fd;
   handle_in_fence_fd(dri2_surf, dri2_surf->dri_image_back);
}

static void
droid_swrast_get_drawable_info(struct dri_drawable *drawable,
	        int *x, int *y, int *width, int *height,
	        void *loaderPrivate)
{
   struct dri2_egl_surface *dri2_surf = loaderPrivate;

   update_buffers(dri2_surf);

   *x = 0;
   *y = 0;
   *width = dri2_surf->base.Width;
   *height = dri2_surf->base.Height;
}

static void
droid_swrast_put_image2(struct dri_drawable *draw, int op, int x, int y, int w,
                          int h, int stride, char *data, void *loaderPrivate)
{
   return;
}

static void
droid_swrast_put_image(struct dri_drawable *draw, int op, int x, int y, int w,
                         int h, char *data, void *loaderPrivate)
{
   return;
}

static void
droid_swrast_get_image(struct dri_drawable *read, int x, int y, int w, int h,
                         char *data, void *loaderPrivate)
{
   return;
}

static const __DRIswrastLoaderExtension swrast_loader_extension = {
   .base = {__DRI_SWRAST_LOADER, 2},

   .getDrawableInfo = droid_swrast_get_drawable_info,
   .putImage = droid_swrast_put_image,
   .getImage = droid_swrast_get_image,
   .putImage2 = droid_swrast_put_image2,
};

static const __DRImutableRenderBufferLoaderExtension
   droid_mutable_render_buffer_extension = {
      .base = {__DRI_MUTABLE_RENDER_BUFFER_LOADER, 1},
      .displaySharedBuffer = droid_display_shared_buffer,
};

static const __DRIextension *droid_image_loader_extensions[] = {
   &droid_image_loader_extension.base,
   &image_lookup_extension.base,
   &droid_mutable_render_buffer_extension.base,
   NULL,
};

static const __DRIextension *droid_swrast_image_loader_extensions[] = {
   &droid_image_loader_extension.base,
   &image_lookup_extension.base,
   &droid_mutable_render_buffer_extension.base,
   &swrast_loader_extension.base,
   NULL,
};

static EGLBoolean
droid_load_driver(_EGLDisplay *disp, bool swrast)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);

   if (disp->Options.Zink)
      dri2_dpy->driver_name = strdup("zink");
   else
      dri2_dpy->driver_name = loader_get_driver_for_fd(dri2_dpy->fd_render_gpu);
   if (dri2_dpy->driver_name == NULL)
      return false;

   if (swrast && !disp->Options.Zink) {
      /* Use kms swrast only with vgem / virtio_gpu.
       * virtio-gpu fallbacks to software rendering when 3D features
       * are unavailable since 6c5ab.
       */
      if (strcmp(dri2_dpy->driver_name, "vgem") == 0 ||
          strcmp(dri2_dpy->driver_name, "virtio_gpu") == 0) {
         free(dri2_dpy->driver_name);
         dri2_dpy->driver_name = strdup("kms_swrast");
      } else {
         goto error;
      }
   }

   dri2_dpy->loader_extensions = droid_image_loader_extensions;
   dri2_detect_swrast_kopper(disp);

   return true;

error:
   free(dri2_dpy->driver_name);
   dri2_dpy->driver_name = NULL;
   return false;
}

static void
droid_unload_driver(_EGLDisplay *disp)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);

   free(dri2_dpy->driver_name);
   dri2_dpy->driver_name = NULL;
}

static int
droid_filter_device(_EGLDisplay *disp, int fd, const char *vendor)
{
   drmVersionPtr ver = drmGetVersion(fd);
   if (!ver)
      return -1;

   if (strcmp(vendor, ver->name) != 0) {
      drmFreeVersion(ver);
      return -1;
   }

   drmFreeVersion(ver);
   return 0;
}

static EGLBoolean
droid_probe_device(_EGLDisplay *disp, bool swrast)
{
   /* Check that the device is supported, by attempting to:
    * - load the dri module
    * - and, create a screen
    */
   if (!droid_load_driver(disp, swrast))
      return EGL_FALSE;

   if (!dri2_create_screen(disp)) {
      _eglLog(_EGL_WARNING, "DRI2: failed to create screen");
      droid_unload_driver(disp);
      return EGL_FALSE;
   }
   return EGL_TRUE;
}

static EGLBoolean
droid_open_device(_EGLDisplay *disp, bool swrast)
{
#define MAX_DRM_DEVICES 64
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   _EGLDevice *dev_list = _eglGlobal.DeviceList;
   drmDevicePtr device;

   char *vendor_name = NULL;
   char vendor_buf[PROP_VALUE_MAX];

#ifdef EGL_FORCE_RENDERNODE
   const unsigned node_type = DRM_NODE_RENDER;
#else
   const unsigned node_type = swrast ? DRM_NODE_PRIMARY : DRM_NODE_RENDER;
#endif

   if (__system_property_get("drm.gpu.vendor_name", vendor_buf) > 0)
      vendor_name = vendor_buf;

   while (dev_list) {
      if (!_eglDeviceSupports(dev_list, _EGL_DEVICE_DRM))
         goto next;

      device = _eglDeviceDrm(dev_list);
      assert(device);

      if (!(device->available_nodes & (1 << node_type)))
         goto next;

      dri2_dpy->fd_render_gpu = loader_open_device(device->nodes[node_type]);
      if (dri2_dpy->fd_render_gpu < 0) {
         _eglLog(_EGL_WARNING, "%s() Failed to open DRM device %s", __func__,
                 device->nodes[node_type]);
         goto next;
      }

      /* If a vendor is explicitly provided, we use only that.
       * Otherwise we fall-back the first device that is supported.
       */
      if (vendor_name) {
         if (droid_filter_device(disp, dri2_dpy->fd_render_gpu, vendor_name)) {
            /* Device does not match - try next device */
            close(dri2_dpy->fd_render_gpu);
            dri2_dpy->fd_render_gpu = -1;
            goto next;
         }
         /* If the requested device matches - use it. Regardless if
          * init fails, do not fall-back to any other device.
          */
         if (!droid_probe_device(disp, false)) {
            close(dri2_dpy->fd_render_gpu);
            dri2_dpy->fd_render_gpu = -1;
         }

         break;
      }
      if (droid_probe_device(disp, swrast))
         break;

      /* No explicit request - attempt the next device */
      close(dri2_dpy->fd_render_gpu);
      dri2_dpy->fd_render_gpu = -1;

   next:
      dev_list = _eglDeviceNext(dev_list);
   }

   /* No DRM device: an Arm Mali on the stock kbase kernel driver has /dev/mali0 instead, which
    * gallium panfrost drives through its kbase backend. */
   if (dri2_dpy->fd_render_gpu < 0 && !swrast && !disp->Options.Zink) {
      const char *node = os_get_option("MESA_KBASE_NODE");
      dri2_dpy->fd_render_gpu = loader_open_device(node ? node : "/dev/mali0");
      _eglLog(_EGL_DEBUG, "android: kbase node %s: fd %d", node ? node : "/dev/mali0",
              dri2_dpy->fd_render_gpu);
      if (dri2_dpy->fd_render_gpu >= 0 && !droid_probe_device(disp, false)) {
         close(dri2_dpy->fd_render_gpu);
         dri2_dpy->fd_render_gpu = -1;
      }
   }

   if (dri2_dpy->fd_render_gpu < 0) {
      _eglLog(_EGL_WARNING, "Failed to open %s DRM device",
              vendor_name ? "desired" : "any");
      return EGL_FALSE;
   }

   return EGL_TRUE;
}

EGLBoolean
dri2_initialize_android(_EGLDisplay *disp)
{
   bool device_opened = false;
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   const char *err;

   dri2_dpy->gralloc = u_gralloc_create(U_GRALLOC_TYPE_AUTO);
   if (dri2_dpy->gralloc == NULL) {
      err = "DRI2: failed to get gralloc";
      goto cleanup;
   }

   bool force_pure_swrast = debug_get_bool_option("MESA_ANDROID_NO_KMS_SWRAST", false);

   if (!force_pure_swrast)
      device_opened = droid_open_device(disp, disp->Options.ForceSoftware);

   if ((!device_opened && disp->Options.ForceSoftware) ||
       force_pure_swrast) {
      dri2_dpy->driver_name = strdup("swrast");
      dri2_dpy->loader_extensions = droid_swrast_image_loader_extensions;
      dri2_dpy->fd_render_gpu = -1;
      dri2_dpy->pure_swrast = true;
      dri2_detect_swrast_kopper(disp);

      if (!dri2_create_screen(disp)) {
         err = "DRI2: Failed to create swrast screen";
         goto cleanup;
      }

      device_opened = EGL_TRUE;
   }

   if (!device_opened) {
      err = "DRI2: failed to open device";
      goto cleanup;
   }

   dri2_dpy->fd_display_gpu = dri2_dpy->fd_render_gpu;

   /* A kbase node has no DRM device behind it to describe; it gets the software EGLDevice. */
   if (!dri2_dpy->pure_swrast &&
       !dri2_setup_device(disp, loader_fd_is_kbase(dri2_dpy->fd_render_gpu))) {
      err = "DRI2: failed to setup EGLDevice";
      goto cleanup;
   }

   dri2_setup_screen(disp);

   /* We set the maximum swap interval as 1 for Android platform, since it is
    * the maximum value supported by Android according to the value of
    * ANativeWindow::maxSwapInterval.
    */
   dri2_setup_swap_interval(disp, 1);

   disp->Extensions.ANDROID_framebuffer_target = EGL_TRUE;
   disp->Extensions.ANDROID_image_native_buffer = EGL_TRUE;
   disp->Extensions.ANDROID_recordable = EGL_TRUE;

   /* Querying buffer age requires a buffer to be dequeued.  Without
    * EGL_ANDROID_native_fence_sync, dequeue might call eglClientWaitSync and
    * result in a deadlock (the lock is already held by eglQuerySurface).
    */
   if (disp->Extensions.ANDROID_native_fence_sync) {
      disp->Extensions.EXT_buffer_age = EGL_TRUE;
   } else {
      /* disable KHR_partial_update that might have been enabled in
       * dri2_setup_screen
       */
      disp->Extensions.KHR_partial_update = EGL_FALSE;
   }

   disp->Extensions.KHR_image = EGL_TRUE;

   dri2_dpy->front_rendering_usage = 0;
#if ANDROID_API_LEVEL >= 24
   if (!dri2_dpy->swrast_not_kms &&
       dri2_dpy->loader_extensions == droid_image_loader_extensions &&
       /* In big GL, front rendering is done at the core API level by directly
        * rendering on the front buffer. However, in ES, the front buffer is
        * completely inaccessible through the core ES API.
        *
        * EGL_KHR_mutable_render_buffer is Android's attempt to re-introduce
        * front rendering into ES by squeezing into EGL. Unlike big GL, this
        * extension redirects GL_BACK used by ES for front rendering. Thus we
        * restrict the enabling of this extension to ES only.
        */
       (disp->ClientAPIs & ~(EGL_OPENGL_ES_BIT | EGL_OPENGL_ES2_BIT |
                             EGL_OPENGL_ES3_BIT_KHR)) == 0) {
      /* For cros gralloc, if the front rendering query is supported, then all
       * available window surface configs support front rendering because:
       *
       * 1) EGL queries cros gralloc for the front rendering usage bit here
       * 2) EGL combines the front rendering usage bit with the existing usage
       *    if the window surface requests mutable render buffer
       * 3) EGL sets the combined usage onto the ANativeWindow and the next
       *    dequeueBuffer will ask gralloc for an allocation/re-allocation with
       *    the new combined usage
       * 4) cros gralloc(on top of minigbm) resolves the front rendering usage
       *    bit into either BO_USE_FRONT_RENDERING or BO_USE_LINEAR based on
       *    the format support checking.
       *
       * So at least we can force BO_USE_LINEAR as the fallback.
       */
      uint64_t front_rendering_usage = 0;
      if (!u_gralloc_get_front_rendering_usage(dri2_dpy->gralloc,
                                               &front_rendering_usage)) {
         dri2_dpy->front_rendering_usage = front_rendering_usage;
         disp->Extensions.KHR_mutable_render_buffer = EGL_TRUE;
      }
   }
#endif

   /* Create configs *after* enabling extensions because presence of DRI
    * driver extensions can affect the capabilities of EGLConfigs.
    */
   droid_add_configs_for_visuals(disp);

   /* Fill vtbl last to prevent accidentally calling virtual function during
    * initialization.
    */
   dri2_dpy->vtbl = &droid_display_vtbl;

   return EGL_TRUE;

cleanup:
   return _eglError(EGL_NOT_INITIALIZED, err);
}
