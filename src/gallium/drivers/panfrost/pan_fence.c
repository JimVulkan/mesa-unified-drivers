/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates.
 * Copyright (C) 2008 VMware, Inc.
 * Copyright (C) 2014 Broadcom
 * Copyright (C) 2018 Alyssa Rosenzweig
 * Copyright (C) 2019 Collabora, Ltd.
 * Copyright (C) 2012 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 */

#include "pan_fence.h"
#include "pan_context.h"
#include "pan_screen.h"

#include "util/libsync.h"
#include "util/os_file.h"
#include "util/os_time.h"
#include "util/u_inlines.h"
#include "kmod/kbase_kmod.h"

static bool
is_kbase(struct panfrost_device *dev)
{
   return pan_kmod_dev_is_kbase(dev->kmod.dev);
}

static struct pipe_fence_handle *
kbase_fence_new(uint64_t seq, int sync_fd)
{
   struct pipe_fence_handle *f = calloc(1, sizeof(*f));
   if (!f)
      return NULL;
   pipe_reference_init(&f->reference, 1);
   f->kbase_seq = seq;
   f->sync_fd = sync_fd;
   f->signaled = !seq && sync_fd < 0;
   return f;
}

void
panfrost_fence_reference(struct pipe_screen *pscreen,
                         struct pipe_fence_handle **ptr,
                         struct pipe_fence_handle *fence)
{
   struct panfrost_device *dev = pan_device(pscreen);
   struct pipe_fence_handle *old = *ptr;

   if (pipe_reference(&old->reference, &fence->reference)) {
      if (is_kbase(dev)) {
         if (old->sync_fd >= 0)
            close(old->sync_fd);
      } else {
         dev->sync->destroy(dev->sync, old->syncobj);
      }
      free(old);
   }

   *ptr = fence;
}

bool
panfrost_fence_finish(struct pipe_screen *pscreen, struct pipe_context *ctx,
                      struct pipe_fence_handle *fence, uint64_t timeout)
{
   struct panfrost_device *dev = pan_device(pscreen);
   int ret;

   if (fence->signaled)
      return true;

   uint64_t abs_timeout = os_time_get_absolute_timeout(timeout);
   if (abs_timeout == OS_TIMEOUT_INFINITE)
      abs_timeout = INT64_MAX;

   if (is_kbase(dev)) {
      if (fence->sync_fd >= 0) {
         const int ms = timeout == OS_TIMEOUT_INFINITE ? -1
                        : (int)MIN2(DIV_ROUND_UP(timeout, 1000000), INT32_MAX);
         fence->signaled = sync_wait(fence->sync_fd, ms) == 0;
      } else {
         const int64_t t0 = os_time_get_nano();
         fence->signaled =
            pan_kmod_kbase_wait_seq_timeout(dev->kmod.dev, fence->kbase_seq, abs_timeout);
         const double ms = (os_time_get_nano() - t0) / 1e6;
         if (dev->stall_log && ms > 2.0)
            mesa_logi("pan stall: %.1f ms waiting for a fence", ms);
      }
      return fence->signaled;
   }

   ret = dev->sync->wait(dev->sync, &fence->syncobj, 1,
                        abs_timeout, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);

   fence->signaled = (ret >= 0);
   return fence->signaled;
}

int
panfrost_fence_get_fd(struct pipe_screen *screen, struct pipe_fence_handle *f)
{
   struct panfrost_device *dev = pan_device(screen);
   int fd = -1;

   if (is_kbase(dev)) {
      if (f->sync_fd >= 0)
         return os_dupfd_cloexec(f->sync_fd);
      if (f->signaled || pan_kmod_kbase_wait_seq_timeout(dev->kmod.dev, f->kbase_seq, 0))
         return -1;
      /* A fence-trigger atom behind everything submitted so far, which includes f's work. */
      pan_kmod_kbase_submit_async(dev->kmod.dev, NULL, 0, &fd, NULL, NULL);
      return fd;
   }

   dev->sync->export_sync_file(dev->sync, f->syncobj, &fd);
   return fd;
}

struct pipe_fence_handle *
panfrost_fence_from_fd(struct panfrost_context *ctx, int fd,
                       enum pipe_fd_type type)
{
   struct panfrost_device *dev = pan_device(ctx->base.screen);
   int ret;

   if (is_kbase(dev)) {
      if (type != PIPE_FD_TYPE_NATIVE_SYNC)
         return NULL;
      const int dup = os_dupfd_cloexec(fd);
      return dup >= 0 ? kbase_fence_new(0, dup) : NULL;
   }

   struct pipe_fence_handle *f = calloc(1, sizeof(*f));
   if (!f)
      return NULL;

   if (type == PIPE_FD_TYPE_NATIVE_SYNC) {
      ret = dev->sync->create(dev->sync, 0, &f->syncobj);
      if (ret) {
         mesa_loge("create syncobj failed\n");
         goto err_free_fence;
      }

      ret = dev->sync->import_sync_file(dev->sync, f->syncobj, fd);
      if (ret) {
         mesa_loge("import syncfile failed\n");
         goto err_destroy_syncobj;
      }
   } else {
      assert(type == PIPE_FD_TYPE_SYNCOBJ);
      ret = dev->sync->fd_to_handle(dev->sync, fd, &f->syncobj);
      if (ret) {
         mesa_loge("import syncobj FD failed\n");
         goto err_free_fence;
      }
   }

   pipe_reference_init(&f->reference, 1);

   return f;

err_destroy_syncobj:
   dev->sync->destroy(dev->sync, f->syncobj);
err_free_fence:
   free(f);
   return NULL;
}

struct pipe_fence_handle *
panfrost_fence_create(struct panfrost_context *ctx)
{
   struct panfrost_device *dev = pan_device(ctx->base.screen);
   int fd = -1, ret;

   if (is_kbase(dev))
      return kbase_fence_new(ctx->kbase_seq, -1);

   /* Snapshot the last rendering out fence. We'd rather have another
    * syncobj instead of a sync file, but this is all we get.
    * (HandleToFD/FDToHandle just gives you another syncobj ID for the
    * same syncobj).
    */
   ret = dev->sync->export_sync_file(dev->sync, ctx->syncobj, &fd);
   if (!ret && fd == -1) {
      /* Already signaled: the kbase sync objects export that as no sync file. */
      struct pipe_fence_handle *f = calloc(1, sizeof(*f));
      if (!f || dev->sync->create(dev->sync, DRM_SYNCOBJ_CREATE_SIGNALED, &f->syncobj)) {
         free(f);
         return NULL;
      }
      f->signaled = true;
      pipe_reference_init(&f->reference, 1);
      return f;
   }
   if (ret || fd == -1) {
      mesa_loge("export failed\n");
      return NULL;
   }

   struct pipe_fence_handle *f =
      panfrost_fence_from_fd(ctx, fd, PIPE_FD_TYPE_NATIVE_SYNC);

   close(fd);

   return f;
}
