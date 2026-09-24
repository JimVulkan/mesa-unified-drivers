/*
 * Copyright © 2021 Collabora Ltd.
 *
 * Derived from tu_device.c which is:
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "genxml/gen_macros.h"

#include "decode.h"

#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_event.h"
#include "panvk_image.h"
#include "panvk_image_view.h"
#include "panvk_instance.h"
#include "panvk_physical_device.h"
#include "panvk_priv_bo.h"
#include "panvk_queue.h"

#include "panvk_gs.h"

#include "vk_drm_syncobj.h"
#include "vk_framebuffer.h"

#include "drm-uapi/panfrost_drm.h"

#include "kmod/kbase_kmod.h"
#include "util/u_sync_provider.h"

static void panvk_signal_event_syncobjs(struct panvk_gpu_queue *queue,
                                        struct panvk_batch *batch);

/* What every batch needs before it is (re)submitted: a batch issued before is reset, and the
 * command buffer's descriptors are made GPU-visible. */
static void
panvk_queue_prepare_batch(struct panvk_device *dev, struct panvk_cmd_buffer *cmdbuf,
                          struct panvk_batch *batch)
{
   /* Reset the batch if it's already been issued */
   if (batch->issued) {
      util_dynarray_foreach(&batch->jobs, void *, job)
         memset((*job), 0, 4 * 4);

      /* Reset the tiler before re-issuing the batch */
      if (batch->tiler.ctx_descs.cpu) {
         memcpy(batch->tiler.heap_desc.cpu, &batch->tiler.heap_templ,
                sizeof(batch->tiler.heap_templ));

         struct mali_tiler_context_packed *ctxs = batch->tiler.ctx_descs.cpu;

         for (uint32_t i = 0; i < batch->fb.layer_count; i++)
            memcpy(&ctxs[i], &batch->tiler.ctx_templ, sizeof(*ctxs));
      }

      /* We don't keep track of BO <-> job relationship, so let's just flush the
       * whole desc pool for now. */
      panvk_pool_flush_maps(&cmdbuf->desc_pool);
   }

   /* Flush pending synchronization requests before submitting the job, to
    * make sure things are GPU-visible. */
   pan_kmod_flush_bo_map_syncs(dev->kmod.dev);
}

/* A submission that did not complete: say which job stopped it. The job manager writes each
 * job header's exception status (word 0) and fault pointer (words 2-3) back when the job ends,
 * and leaves 0 in the ones it never ran -- that is how the TLS overrun on the Mali-G52 was found
 * (the first job after the last DONE one was the spilling kernel). */
static void
panvk_kbase_report_faults(struct panvk_device *dev, struct panvk_cmd_buffer *const *cmdbufs,
                          unsigned nr_cmdbufs)
{
   pan_kmod_flush_bo_map_syncs(dev->kmod.dev);
   unsigned b = 0;
   for (uint32_t j = 0; j < nr_cmdbufs; ++j) {
      list_for_each_entry(struct panvk_batch, batch, &cmdbufs[j]->batches, node) {
         unsigned k = 0;
         util_dynarray_foreach(&batch->jobs, void *, job) {
            const uint32_t *w = *job;
            mesa_loge("panvk: batch %u job %u: status 0x%x type %u index %u deps %u,%u "
                      "fault 0x%" PRIx64, b, k, w[0], (w[4] >> 1) & 0x7f, w[4] >> 16,
                      w[5] & 0xffff, w[5] >> 16, (uint64_t)w[2] | ((uint64_t)w[3] << 32));
            k++;
         }
         b++;
      }
   }
}

/* What a kbase submission still has to do once the GPU is done with it. */
struct panvk_kbase_done {
   struct panvk_gpu_queue *queue;
   uint32_t nr_cmdbufs, nr_signals;
   struct panvk_cmd_buffer **cmdbufs;
   uint32_t *signals;
};

/* On the kmod's event thread, when every atom of the submission has reported. The command
 * buffers cannot have been reset or freed yet: they are pending until the signals below. */
static void
panvk_kbase_submit_done(void *data, bool ok)
{
   struct panvk_kbase_done *d = data;
   struct panvk_device *dev = to_panvk_device(d->queue->vk.base.device);

   if (!ok) {
      mesa_loge("panvk: the kbase submission did not complete");
      panvk_kbase_report_faults(dev, d->cmdbufs, d->nr_cmdbufs);
   }

   for (uint32_t j = 0; j < d->nr_cmdbufs; ++j) {
      list_for_each_entry(struct panvk_batch, batch, &d->cmdbufs[j]->batches, node)
         panvk_signal_event_syncobjs(d->queue, batch);
   }

   if (d->nr_signals)
      dev->vk.sync->signal(dev->vk.sync, d->signals, d->nr_signals);

   free(d);
}

/*
 * kbase: every batch of a vkQueueSubmit in one JOB_SUBMIT, each atom waiting on the one before
 * it, and no wait at all. Submitted and waited for batch by batch, a Minecraft frame (about 30
 * render passes, copies and clears) paid a full CPU round trip per batch while the GPU sat idle
 * between them; waited for as a whole, the app still could not record the next frame while the
 * GPU drew this one. Now the signals are marked submitted, the atoms go to the kernel, and the
 * kmod's event thread signals them when the work is done. A fence-trigger atom gives the signals
 * a sync file, which is what the compositor gets for a present.
 */
static void
panvk_queue_submit_kbase_chain(struct panvk_gpu_queue *queue, struct vk_queue_submit *submit)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   const uint32_t nr_cmdbufs = submit->command_buffer_count;
   const uint32_t nr_signals = submit->signal_count;

   struct panvk_kbase_done *d =
      malloc(sizeof(*d) + nr_cmdbufs * sizeof(struct panvk_cmd_buffer *) +
             nr_signals * sizeof(uint32_t));
   if (!d) {
      vk_device_set_lost(&dev->vk, "out of memory submitting");
      return;
   }
   d->queue = queue;
   d->nr_cmdbufs = nr_cmdbufs;
   d->nr_signals = nr_signals;
   d->cmdbufs = (struct panvk_cmd_buffer **)(d + 1);
   d->signals = (uint32_t *)(d->cmdbufs + nr_cmdbufs);

   /* Kept here too: d belongs to the event thread once submitted. */
   uint32_t signals[MAX2(nr_signals, 1)];
   for (uint32_t i = 0; i < nr_signals; i++) {
      assert(vk_sync_type_is_drm_syncobj(submit->signals[i].sync->type));
      signals[i] = d->signals[i] = vk_sync_as_drm_syncobj(submit->signals[i].sync)->syncobj;
   }

   for (uint32_t j = 0; j < nr_cmdbufs; ++j) {
      d->cmdbufs[j] =
         container_of(submit->command_buffers[j], struct panvk_cmd_buffer, vk);
      /* Its batches are reset below if they ran before, which must not happen under a GPU still
       * running the previous submission of it (simultaneous use). */
      if (d->cmdbufs[j]->kbase_seq)
         pan_kmod_kbase_wait_seq(dev->kmod.dev, d->cmdbufs[j]->kbase_seq);
   }

   struct pan_kmod_kbase_atom atoms[PAN_KMOD_KBASE_MAX_ATOMS];
   unsigned nr_atoms = 0;
   uint8_t last = 0;

   for (uint32_t j = 0; j < nr_cmdbufs; ++j) {
      struct panvk_cmd_buffer *cmdbuf = d->cmdbufs[j];

      list_for_each_entry(struct panvk_batch, batch, &cmdbuf->batches, node) {
         panvk_queue_prepare_batch(dev, cmdbuf, batch);

         /* More than one submission takes: send what there is, the rest runs after it. */
         if (nr_atoms + 2 > PAN_KMOD_KBASE_MAX_ATOMS) {
            if (!pan_kmod_kbase_submit_async(dev->kmod.dev, atoms, nr_atoms, NULL, NULL, NULL))
               mesa_loge("panvk: the kbase submission was refused");
            nr_atoms = 0;
            last = 0;
         }

         if (batch->vtc_jc.first_job) {
            atoms[nr_atoms] = (struct pan_kmod_kbase_atom){
               .jc = batch->vtc_jc.first_job,
               .core_req = KBASE_JD_REQ_VERTEX_TILER,
               .atom_number = nr_atoms + 1,
               .dep_atom = last,
            };
            last = ++nr_atoms;
         }

         if (batch->frag_jc.first_job) {
            atoms[nr_atoms] = (struct pan_kmod_kbase_atom){
               .jc = batch->frag_jc.first_job,
               .core_req = KBASE_JD_REQ_FRAGMENT,
               .atom_number = nr_atoms + 1,
               .dep_atom = last,
            };
            last = ++nr_atoms;
         }

         batch->issued = true;
      }
   }

   /* Before submitting: the completion may run before submit_async even returns. */
   pan_kmod_kbase_sync_submitted(signals, nr_signals);

   int fence = -1;
   const uint64_t seq = pan_kmod_kbase_submit_async(
      dev->kmod.dev, atoms, nr_atoms, nr_signals ? &fence : NULL, panvk_kbase_submit_done, d);
   if (!seq) {
      /* Signal anyway, so nothing waits forever on work that was never sent. */
      panvk_kbase_submit_done(d, false);
      return;
   }

   pan_kmod_kbase_sync_attach_fd(signals, nr_signals, fence);
   if (fence >= 0)
      close(fence);

   for (uint32_t j = 0; j < nr_cmdbufs; ++j)
      container_of(submit->command_buffers[j], struct panvk_cmd_buffer, vk)->kbase_seq = seq;

}

static void
panvk_queue_submit_batch(struct panvk_gpu_queue *queue,
                         struct panvk_cmd_buffer *cmdbuf,
                         struct panvk_batch *batch, uint32_t *bos,
                         unsigned nr_bos, uint32_t *in_fences,
                         unsigned nr_in_fences)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);
   ASSERTED int ret;

   panvk_queue_prepare_batch(dev, cmdbuf, batch);

   /* kbase takes the whole frame in one submission: a vertex/tiler atom and a fragment atom that
    * depends on it, which is the shape the vendor driver was captured producing for a triangle.
    * There is no fence to hand back, so pan_kmod_kbase_submit blocks until both atoms report.
    *
    * This branch is not an optimisation. On this kernel DRM_IOCTL_PANFROST_SUBMIT returns
    * SUCCESS and does nothing, so without it every submit below silently no-ops and the driver
    * looks like it works right up until someone reads a pixel back. */
   if (pan_kmod_dev_is_kbase(dev->kmod.dev)) {
      struct pan_kmod_kbase_atom atoms[2];
      unsigned nr_atoms = 0;

      if (batch->vtc_jc.first_job) {
         atoms[nr_atoms++] = (struct pan_kmod_kbase_atom){
            .jc = batch->vtc_jc.first_job,
            .core_req = KBASE_JD_REQ_VERTEX_TILER,
            .atom_number = 1,
         };
      }

      if (batch->frag_jc.first_job) {
         atoms[nr_atoms++] = (struct pan_kmod_kbase_atom){
            .jc = batch->frag_jc.first_job,
            .core_req = KBASE_JD_REQ_FRAGMENT,
            .atom_number = 2,
            .dep_atom = batch->vtc_jc.first_job ? 1 : 0,
         };
      }

      if (nr_atoms && getenv("PANVK_GS_DUMP") && panvk_gs_dbg_pos_cpu) {
         /* kbase_submit below blocks until every atom has reported, so this is the first moment
          * the emulation's output buffer can be read back honestly. */
         const float *p = panvk_gs_dbg_pos_cpu;
         pan_kmod_kbase_submit(dev->kmod.dev, atoms, nr_atoms);
         fprintf(stderr, "[GSDUMP] out_pos cpu=%p gpu=0x%" PRIx64 " verts=%u\n",
                 panvk_gs_dbg_pos_cpu, panvk_gs_dbg_pos_gpu, panvk_gs_dbg_pos_verts);
         for (uint32_t v = 0; v < panvk_gs_dbg_pos_verts && v < 8; v++)
            fprintf(stderr,
                    "[GSDUMP]   v%u = %f %f %f %f   raw %08x %08x %08x %08x\n",
                    v, p[v * 4 + 0], p[v * 4 + 1], p[v * 4 + 2], p[v * 4 + 3],
                    ((const uint32_t *)p)[v * 4 + 0],
                    ((const uint32_t *)p)[v * 4 + 1],
                    ((const uint32_t *)p)[v * 4 + 2],
                    ((const uint32_t *)p)[v * 4 + 3]);
         if (panvk_gs_dbg_in_pos_cpu) {
            const float *ip = panvk_gs_dbg_in_pos_cpu;
            for (uint32_t v = 0; v < 3; v++)
               fprintf(stderr, "[GSDUMP]   VS in v%u = %f %f %f %f\n",
                       v, ip[v * 4 + 0], ip[v * 4 + 1], ip[v * 4 + 2],
                       ip[v * 4 + 3]);
         }
         if (panvk_gs_dbg_idx_cpu) {
            const uint32_t *ix = panvk_gs_dbg_idx_cpu;
            fprintf(stderr, "[GSDUMP]   indices(%u) =", panvk_gs_dbg_idx_count);
            for (uint32_t i = 0; i < panvk_gs_dbg_idx_count && i < 12; i++)
               fprintf(stderr, " %u", ix[i]);
            fprintf(stderr, "\n");
         }
         panvk_gs_dbg_pos_cpu = NULL;
      } else if (nr_atoms && !pan_kmod_kbase_submit(dev->kmod.dev, atoms, nr_atoms))
         mesa_loge("panvk: the kbase submission did not complete");

      /* The GPU is finished by the time we get here, so anything read back on the CPU has to see
       * what it wrote. */
      panvk_pool_invalidate_maps(&cmdbuf->desc_pool);
      pan_kmod_flush_bo_map_syncs(dev->kmod.dev);

      if (PANVK_DEBUG(TRACE)) {
         if (batch->vtc_jc.first_job)
            pandecode_jc(dev->debug.decode_ctx, batch->vtc_jc.first_job,
                         phys_dev->kmod.dev->props.gpu_id);
         if (batch->frag_jc.first_job)
            pandecode_jc(dev->debug.decode_ctx, batch->frag_jc.first_job,
                         phys_dev->kmod.dev->props.gpu_id);
         pandecode_next_frame(dev->debug.decode_ctx);
      }

      if (PANVK_DEBUG(DUMP))
         pandecode_dump_mappings(dev->debug.decode_ctx);

      batch->issued = true;
      return;
   }

   if (batch->vtc_jc.first_job) {
      struct drm_panfrost_submit submit = {
         .bo_handles = (uintptr_t)bos,
         .bo_handle_count = nr_bos,
         .in_syncs = (uintptr_t)in_fences,
         .in_sync_count = nr_in_fences,
         .out_sync = queue->sync,
         .jc = batch->vtc_jc.first_job,
      };

      ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_PANFROST_SUBMIT, &submit);
      assert(!ret);

      if (PANVK_DEBUG(TRACE) || PANVK_DEBUG(SYNC)) {
         ret = drmSyncobjWait(dev->drm_fd, &submit.out_sync, 1, INT64_MAX, 0,
                              NULL);
         assert(!ret);

         /* If we want to read the descriptors back, we need to invalidate the
          * whole desc pool, otherwise we might end up with stale data. */
         panvk_pool_invalidate_maps(&cmdbuf->desc_pool);
         pan_kmod_flush_bo_map_syncs(dev->kmod.dev);
      }

      if (PANVK_DEBUG(TRACE)) {
         pandecode_jc(dev->debug.decode_ctx, batch->vtc_jc.first_job,
                      phys_dev->kmod.dev->props.gpu_id);
      }

      if (PANVK_DEBUG(DUMP))
         pandecode_dump_mappings(dev->debug.decode_ctx);

      if (PANVK_DEBUG(SYNC))
         pandecode_abort_on_fault(dev->debug.decode_ctx, submit.jc,
                                  phys_dev->kmod.dev->props.gpu_id);
   }

   if (batch->frag_jc.first_job) {
      struct drm_panfrost_submit submit = {
         .bo_handles = (uintptr_t)bos,
         .bo_handle_count = nr_bos,
         .out_sync = queue->sync,
         .jc = batch->frag_jc.first_job,
         .requirements = PANFROST_JD_REQ_FS,
      };

      if (batch->vtc_jc.first_job) {
         submit.in_syncs = (uintptr_t)(&queue->sync);
         submit.in_sync_count = 1;
      } else {
         submit.in_syncs = (uintptr_t)in_fences;
         submit.in_sync_count = nr_in_fences;
      }

      ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_PANFROST_SUBMIT, &submit);
      assert(!ret);
      if (PANVK_DEBUG(TRACE) || PANVK_DEBUG(SYNC)) {
         ret = drmSyncobjWait(dev->drm_fd, &submit.out_sync, 1, INT64_MAX, 0,
                              NULL);
         assert(!ret);

         /* If we want to read the descriptors back, we need to invalidate the
          * whole desc pool, otherwise we might end up with stale data. */
         panvk_pool_invalidate_maps(&cmdbuf->desc_pool);
         pan_kmod_flush_bo_map_syncs(dev->kmod.dev);
      }

      if (PANVK_DEBUG(TRACE))
         pandecode_jc(dev->debug.decode_ctx, batch->frag_jc.first_job,
                      phys_dev->kmod.dev->props.gpu_id);

      if (PANVK_DEBUG(DUMP))
         pandecode_dump_mappings(dev->debug.decode_ctx);

      if (PANVK_DEBUG(SYNC))
         pandecode_abort_on_fault(dev->debug.decode_ctx, submit.jc,
                                  phys_dev->kmod.dev->props.gpu_id);
   }

   if (PANVK_DEBUG(TRACE))
      pandecode_next_frame(dev->debug.decode_ctx);

   batch->issued = true;
}

static void
panvk_queue_transfer_sync(struct panvk_gpu_queue *queue, uint32_t syncobj)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   ASSERTED int ret;

   /* kbase: the submission already blocked until the GPU was done, so every signal is due
    * now. (The DRM path below would "succeed" on the Exynos kernel without doing anything, and
    * fail on a MediaTek one.) */
   if (pan_kmod_dev_is_kbase(dev->kmod.dev)) {
      dev->vk.sync->signal(dev->vk.sync, &syncobj, 1);
      return;
   }

   struct drm_syncobj_handle handle = {
      .handle = queue->sync,
      .flags = DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE,
      .fd = -1,
   };

   ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &handle);
   assert(!ret);
   assert(handle.fd >= 0);

   handle.handle = syncobj;
   ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &handle);
   assert(!ret);

   close(handle.fd);
}

static void
panvk_add_wait_event_syncobjs(struct panvk_batch *batch, uint32_t *in_fences,
                              unsigned *nr_in_fences)
{
   util_dynarray_foreach(&batch->event_ops, struct panvk_cmd_event_op, op) {
      switch (op->type) {
      case PANVK_EVENT_OP_SET:
         /* Nothing to do yet */
         break;
      case PANVK_EVENT_OP_RESET:
         /* Nothing to do yet */
         break;
      case PANVK_EVENT_OP_WAIT:
         in_fences[(*nr_in_fences)++] = op->event->syncobj;
         break;
      default:
         UNREACHABLE("bad panvk_cmd_event_op type\n");
      }
   }
}

static void
panvk_signal_event_syncobjs(struct panvk_gpu_queue *queue,
                            struct panvk_batch *batch)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   util_dynarray_foreach(&batch->event_ops, struct panvk_cmd_event_op, op) {
      switch (op->type) {
      case PANVK_EVENT_OP_SET: {
         panvk_queue_transfer_sync(queue, op->event->syncobj);
         break;
      }
      case PANVK_EVENT_OP_RESET: {
         struct panvk_event *event = op->event;

         ASSERTED int ret = dev->vk.sync->reset(dev->vk.sync, &event->syncobj, 1);
         assert(!ret);
         break;
      }
      case PANVK_EVENT_OP_WAIT:
         /* Nothing left to do */
         break;
      default:
         UNREACHABLE("bad panvk_cmd_event_op type\n");
      }
   }
}

VkResult
panvk_per_arch(gpu_queue_submit)(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct panvk_gpu_queue *queue = container_of(vk_queue, struct panvk_gpu_queue, vk);
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   unsigned nr_semaphores = submit->wait_count + 1;
   uint32_t semaphores[nr_semaphores];

   semaphores[0] = queue->sync;
   for (unsigned i = 0; i < submit->wait_count; i++) {
      assert(vk_sync_type_is_drm_syncobj(submit->waits[i].sync->type));
      struct vk_drm_syncobj *syncobj =
         vk_sync_as_drm_syncobj(submit->waits[i].sync);

      semaphores[i + 1] = syncobj->syncobj;
   }

   /* kbase has no GPU-side wait. A wait on an object whose signal is already submitted needs
    * nothing: every kbase submission runs after the ones before it. Anything else (signaled by
    * the host, or not submitted yet) is waited for here, on the CPU. */
   if (pan_kmod_dev_is_kbase(dev->kmod.dev) && submit->wait_count)
      dev->vk.sync->wait(dev->vk.sync, semaphores + 1, submit->wait_count, INT64_MAX,
                         DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL |
                            DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE,
                         NULL);

   if (pan_kmod_dev_is_kbase(dev->kmod.dev) && !PANVK_DEBUG(TRACE) && !PANVK_DEBUG(DUMP) &&
       !getenv("PANVK_GS_DUMP")) {
      panvk_queue_submit_kbase_chain(queue, submit);
      return vk_device_check_status(&dev->vk);
   }

   for (uint32_t j = 0; j < submit->command_buffer_count; ++j) {
      struct panvk_cmd_buffer *cmdbuf =
         container_of(submit->command_buffers[j], struct panvk_cmd_buffer, vk);

      list_for_each_entry(struct panvk_batch, batch, &cmdbuf->batches, node) {
         /* FIXME: should be done at the batch level */
         unsigned nr_bos = panvk_pool_num_bos(&cmdbuf->desc_pool) +
                           panvk_pool_num_bos(&cmdbuf->varying_pool) +
                           panvk_pool_num_bos(&cmdbuf->tls_pool) +
                           batch->fb.bo_count + (batch->blit.src ? 1 : 0) +
                           (batch->blit.dst ? 1 : 0) +
                           (batch->vtc_jc.first_tiler ? 1 : 0) + 1;
         unsigned bo_idx = 0;
         uint32_t bos[nr_bos];

         panvk_pool_get_bo_handles(&cmdbuf->desc_pool, &bos[bo_idx]);
         bo_idx += panvk_pool_num_bos(&cmdbuf->desc_pool);

         panvk_pool_get_bo_handles(&cmdbuf->varying_pool, &bos[bo_idx]);
         bo_idx += panvk_pool_num_bos(&cmdbuf->varying_pool);

         panvk_pool_get_bo_handles(&cmdbuf->tls_pool, &bos[bo_idx]);
         bo_idx += panvk_pool_num_bos(&cmdbuf->tls_pool);

         for (unsigned i = 0; i < batch->fb.bo_count; i++)
            bos[bo_idx++] = pan_kmod_bo_handle(batch->fb.bos[i]);

         if (batch->blit.src)
            bos[bo_idx++] = pan_kmod_bo_handle(batch->blit.src);

         if (batch->blit.dst)
            bos[bo_idx++] = pan_kmod_bo_handle(batch->blit.dst);

         if (batch->vtc_jc.first_tiler)
            bos[bo_idx++] = pan_kmod_bo_handle(dev->tiler_heap->bo);

         bos[bo_idx++] = pan_kmod_bo_handle(dev->sample_positions->bo);
         assert(bo_idx == nr_bos);

         /* Merge identical BO entries. */
         for (unsigned x = 0; x < nr_bos; x++) {
            for (unsigned y = x + 1; y < nr_bos;) {
               if (bos[x] == bos[y])
                  bos[y] = bos[--nr_bos];
               else
                  y++;
            }
         }

         unsigned nr_in_fences = 0;
         unsigned max_wait_event_syncobjs = util_dynarray_num_elements(
            &batch->event_ops, struct panvk_cmd_event_op);
         uint32_t in_fences[nr_semaphores + max_wait_event_syncobjs];
         memcpy(in_fences, semaphores, nr_semaphores * sizeof(*in_fences));
         nr_in_fences += nr_semaphores;

         panvk_add_wait_event_syncobjs(batch, in_fences, &nr_in_fences);

         panvk_queue_submit_batch(queue, cmdbuf, batch, bos, nr_bos, in_fences,
                                  nr_in_fences);

         panvk_signal_event_syncobjs(queue, batch);
      }
   }

   /* Transfer the out fence to signal semaphores */
   for (unsigned i = 0; i < submit->signal_count; i++) {
      assert(vk_sync_type_is_drm_syncobj(submit->signals[i].sync->type));
      struct vk_drm_syncobj *syncobj =
         vk_sync_as_drm_syncobj(submit->signals[i].sync);

      panvk_queue_transfer_sync(queue, syncobj->syncobj);
   }

   return VK_SUCCESS;
}

VkResult
panvk_per_arch(create_gpu_queue)(struct panvk_device *device,
                                 const VkDeviceQueueCreateInfo *create_info,
                                 uint32_t queue_idx,
                                 struct vk_queue **out_queue)
{
   ASSERTED const VkDeviceQueueGlobalPriorityCreateInfoKHR *priority_info =
      vk_find_struct_const(create_info->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR);
   ASSERTED const VkQueueGlobalPriorityKHR priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;

   /* XXX: Panfrost kernel module doesn't support priorities so far */
   assert(priority == VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR);

   struct panvk_gpu_queue *queue = vk_zalloc(&device->vk.alloc, sizeof(*queue), 8,
                                         VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_queue_init(&queue->vk, &device->vk, create_info, queue_idx);
   if (result != VK_SUCCESS)
      goto err_free_queue;

   int ret = device->vk.sync->create(device->vk.sync, DRM_SYNCOBJ_CREATE_SIGNALED,
                                     &queue->sync);
   if (ret) {
      result = panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto err_finish_queue;
   }

   queue->vk.driver_submit = panvk_per_arch(gpu_queue_submit);
   *out_queue = &queue->vk;
   return VK_SUCCESS;

err_finish_queue:
   vk_queue_finish(&queue->vk);

err_free_queue:
   vk_free(&device->vk.alloc, queue);
   return result;
}

void panvk_per_arch(destroy_gpu_queue)(struct vk_queue *vk_queue)
{
   struct panvk_gpu_queue *queue = container_of(vk_queue, struct panvk_gpu_queue, vk);
   struct panvk_device *dev = to_panvk_device(vk_queue->base.device);

   if (pan_kmod_dev_is_kbase(dev->kmod.dev))
      pan_kmod_kbase_wait_seq(dev->kmod.dev, 0);
   vk_queue_finish(&queue->vk);
   dev->vk.sync->destroy(dev->vk.sync, queue->sync);
   vk_free(&dev->vk.alloc, queue);
}

VkResult
panvk_per_arch(gpu_queue_check_status)(struct vk_queue *vk_queue)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(QueueWaitIdle)(VkQueue _queue)
{
   VK_FROM_HANDLE(panvk_gpu_queue, queue, _queue);
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   /* we need to use vk_common_QueueWaitIdle if we ever go threaded */
   assert(queue->vk.submit.mode != VK_QUEUE_SUBMIT_MODE_THREADED);

   if (vk_device_is_lost(&dev->vk)) {
      /* Check printf buffer one more time before exiting */
      u_printf_with_ctx(stdout, &dev->printf.ctx);
      return VK_ERROR_DEVICE_LOST;
   }

   /* kbase: everything submitted so far has completed and signaled. */
   if (pan_kmod_dev_is_kbase(dev->kmod.dev)) {
      pan_kmod_kbase_wait_seq(dev->kmod.dev, 0);
      return vk_device_check_status(&dev->vk);
   }

   ASSERTED int ret = dev->vk.sync->wait(dev->vk.sync, &queue->sync, 1, INT64_MAX,
                                         DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
   assert(!ret);

   return VK_SUCCESS;
}
