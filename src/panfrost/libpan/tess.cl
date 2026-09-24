/*
 * Copyright 2026 XclipseTools
 * SPDX-License-Identifier: MIT
 */

/*
 * Tessellation on the job manager, which has no tessellation stage: the fixed-function
 * tessellator from src/poly runs here as three precompiled kernels between the tess control
 * compute job and the tess eval compute job (see panvk_vX_cmd_draw.c).
 *
 *   panlib_tess_*       (COUNT)        one thread per patch: how many indices it produces
 *   panlib_tess_prefix_sum             one thread: prefix sum, index buffer allocation
 *   panlib_tess_*       (WITH_COUNTS)  one thread per patch: domain points and indices
 *
 * An indirect draw runs panlib_tess_setup_indirect first. Its patch count is only known once
 * that has run, so the tessellator's grid is a fixed size and each thread walks the patches at
 * a stride of the grid.
 */

#include "compiler/libcl/libcl.h"
#include "genxml/gen_macros.h"
#include "lib/pan_encoder.h"
#include "poly/cl/tessellator.h"

#if PAN_ARCH >= 6 && PAN_ARCH <= 7

KERNEL(1)
panlib_tess_isoline(constant struct poly_tess_params *p, uint32_t mode)
{
   for (uint patch = cl_global_id.x; patch < p->nr_patches; patch += cl_global_size.x)
      poly_tess_isoline_process(p, patch, (enum poly_tess_mode)mode);
}

KERNEL(1)
panlib_tess_tri(constant struct poly_tess_params *p, uint32_t mode)
{
   for (uint patch = cl_global_id.x; patch < p->nr_patches; patch += cl_global_size.x)
      poly_tess_tri_process(p, patch, (enum poly_tess_mode)mode);
}

KERNEL(1)
panlib_tess_quad(constant struct poly_tess_params *p, uint32_t mode)
{
   for (uint patch = cl_global_id.x; patch < p->nr_patches; patch += cl_global_size.x)
      poly_tess_quad_process(p, patch, (enum poly_tess_mode)mode);
}

/*
 * An indirect tessellated draw: the vertex and instance counts are in the argument buffer, so
 * this sizes on the GPU what emit_tess_jobs sizes on the CPU for a direct draw. The patch counts
 * go in *p, the per-patch arrays come from the heap, and the tess control job gets its grid.
 * The job was queued as NOT_STARTED and becomes COMPUTE here, or NULL when there is no patch or
 * no room in the heap; nr_patches is then 0, which the tessellator and the tess eval job read as
 * nothing to draw.
 *
 * reset_heap: as in panlib_tess_prefix_sum, which does not rewind for an indirect draw because
 * the arrays allocated here already live in the heap.
 *
 * num_wg_x/y: the tess control shader's num_work_groups sysvals, or the memory sink.
 */
KERNEL(1)
panlib_tess_setup_indirect(global struct poly_tess_params *p, constant uint32_t *cmd,
                           uint32_t reset_heap, global uint8_t *tcs_job, uint32_t tcs_local,
                           global uint32_t *num_wg_x, global uint32_t *num_wg_y)
{
   global struct poly_heap *heap = p->heap;
   if (reset_heap)
      heap->bottom = 0;

   uint ppi = cmd[0] / p->input_patch_size;
   uint instances = cmd[1];
   ulong n = (ulong)ppi * instances;

   /* tcs_buffer, coord_allocs, counts, each 16-byte aligned. */
   const ulong stride_B = p->tcs_stride_el * 4;
   const ulong size_B = n * (stride_B + 8) + 32;
   const uint bottom = heap->bottom;
   if (n == 0 || n > 0xffffffu || bottom + size_B + 16 >= heap->size) {
      ppi = instances = 0;
      n = 0;
   }

   global uchar *blob = heap->base + bottom;
   const uint coord_off = align((uint)(n * stride_B), 16);
   const uint counts_off = coord_off + align((uint)n * 4, 16);
   p->tcs_buffer = (global float *)blob;
   p->coord_allocs = (global uint *)(blob + coord_off);
   p->counts = (global uint32_t *)(blob + counts_off);
   p->patches_per_instance = ppi;
   p->nr_patches = n;
   heap->bottom = bottom + align(counts_off + (uint)n * 4, 16);

   struct mali_job_header_packed hdr;
   pan_pack(&hdr, JOB_HEADER, cfg) {
      cfg.type = n ? MALI_JOB_TYPE_COMPUTE : MALI_JOB_TYPE_NULL;
   }
   pan_merge((global struct mali_job_header_packed *)tcs_job, &hdr, JOB_HEADER);
   if (!n)
      return;

   pan_pack_work_groups_compute(
      (global struct mali_invocation_packed *)(tcs_job +
                                               pan_section_offset(COMPUTE_JOB, INVOCATION)),
      ppi, instances, 1, tcs_local, 1, 1, false, false);
   *num_wg_x = ppi;
   *num_wg_y = instances;
}

/*
 * Serial on purpose: one thread, one pass over the patch counts. The draws this serves have
 * hundreds of patches, not millions, and a single thread needs no local memory or barriers.
 *
 * reset_heap: the first tessellated draw of a batch rewinds the heap. kbase runs a batch to
 * completion before the next one starts, so nothing earlier can still be reading it, and a
 * command buffer submitted twice starts from the same place both times.
 *
 * out: { total indices, index buffer offset in elements from the heap base, total primitives }.
 * total is 0 when the heap cannot take the index buffer, which drops the draw instead of
 * overrunning memory. A geometry shader after the tess eval shader runs on the primitive count.
 */
KERNEL(1)
panlib_tess_prefix_sum(global struct poly_tess_params *p, uint32_t reset_heap,
                       global uint32_t *out)
{
   if (reset_heap)
      p->heap->bottom = 0;

   uint total = 0;
   for (uint i = 0; i < p->nr_patches; i++) {
      total += p->counts[i];
      p->counts[i] = total;
   }

   uint size_B = align(total * 4, 16);
   uint bottom = p->heap->bottom;
   if (bottom + size_B + 16 >= p->heap->size) {
      /* Every patch gets zero indices; WITH_COUNTS then writes nothing. */
      for (uint i = 0; i < p->nr_patches; i++)
         p->counts[i] = 0;
      total = 0;
      size_B = 0;
   }

   p->heap->bottom = bottom + size_B;
   p->index_buffer = (global uint32_t *)(p->heap->base + bottom);

   out[0] = total;
   out[1] = bottom / 4;
   out[2] = total / (p->points_mode ? 1 : p->isolines ? 2 : 3);
}

#endif
