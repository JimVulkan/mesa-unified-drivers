/*
 * Copyright 2025 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "compiler/libcl/libcl.h"
#include "compiler/libcl/libcl_vk.h"
#include "compiler/shader_enums.h"
#include "genxml/gen_macros.h"
#include "lib/pan_encoder.h"
#include "poly/cl/restart.h"
#include "draw_helper.h"

#if PAN_ARCH >= 10

/* Subgroup size is always 16 on v10+ */
#define SUBGROUP_SIZE 16

KERNEL(SUBGROUP_SIZE * SUBGROUP_SIZE)
panlib_update_prims_generated_query_restart(
   global atomic_uint *prims_generated, uint64_t index_buffer,
   uint32_t index_buffer_size_el, uint32_t cmd_stride,
   constant VkDrawIndexedIndirectCommand *cmd, uint32_t view_count,
   uint32_t compact_prim__11, uint32_t index_bytes_log2__3)
{
   assert(get_sub_group_size() == SUBGROUP_SIZE);

   enum mesa_prim prim = poly_uncompact_prim(compact_prim__11);
   uint32_t index_bytes = 1 << index_bytes_log2__3;
   uint32_t restart_index = (1 << (index_bytes * 8)) - 1;

   uint32_t tid = cl_local_id.x;
   uint32_t draw_id = cl_group_id.x;

   cmd = (constant VkDrawIndexedIndirectCommand *)
      ((uintptr_t) cmd + cmd_stride * draw_id);

   POLY_DECL_UNROLL_RESTART_SCRATCH(scratch, SUBGROUP_SIZE * SUBGROUP_SIZE);
   uint32_t prims_per_instance = poly_count_restart_prims(
      (constant uint32_t *) cmd, index_buffer, index_buffer_size_el,
      index_bytes, restart_index, prim, scratch);

   if (tid == 0)
      atomic_fetch_add(prims_generated,
                       prims_per_instance * cmd->instanceCount * view_count);
}

KERNEL(1)
panlib_update_prims_generated_query_indirect(
   global uint32_t *prims_generated, global uint32_t *draw_count_buffer,
   uint32_t max_draw_count, uint32_t cmd_stride, constant uint32_t *cmd,
   uint32_t view_count, uint32_t compact_prim__11)
{
   enum mesa_prim prim = poly_uncompact_prim(compact_prim__11);
   uint32_t draw_count = draw_count_buffer ?
      min(*draw_count_buffer, max_draw_count) : max_draw_count;

   for (uint32_t draw_id = 0; draw_id < draw_count; draw_id++) {
      /* cmd may be either VkDrawnIndirectCommand or
       * VkDrawIndexedIndirectCommand. In both cases the vertex/index count is
       * the first field, and the instance count is the second */
      uint32_t vertex_count = cmd[0];
      uint32_t instance_count = cmd[1];

      uint32_t prims_per_instance =
         u_decomposed_prims_for_vertices(prim, vertex_count);
      *prims_generated += prims_per_instance * instance_count;

      cmd = (constant uint32_t *) ((uintptr_t) cmd + cmd_stride);
   }
}
#endif

#if (PAN_ARCH == 6 || PAN_ARCH == 7)
struct panlib_draw_info {
   struct {
      uint32_t size;
      uint32_t offset;
   } index;

   struct {
      int32_t raw_offset;
      int32_t base;
      uint32_t count;
   } vertex;

   struct {
      int32_t base;
      uint32_t count;
   } instance;

   struct {
      global struct mali_attribute_buffer_packed *descs;
      global struct libpan_draw_helper_varying_buf_info *info;
   } varying_bufs;

   struct {
      global struct mali_attribute_buffer_packed *descs;
      global struct libpan_draw_helper_attrib_buf_info *infos;
      uint32_t valid;
   } attrib_bufs;

   struct {
      global struct mali_attribute_packed *descs;
      global struct libpan_draw_helper_attrib_info *infos;
      uint32_t valid;
   } attribs;

   global uint8_t *idvs_job;
   global uint8_t *vertex_job;
   global uint8_t *tiler_job;

   uint32_t vertex_range;
   uint32_t padded_vertex_count;
   uint8_t primitive_vertex_count;
   global uint8_t *indices;

   uint64_t position;
   uint64_t psiz;
};

static void
panlib_patch_draw_vertex_dcd(struct panlib_draw_info *draw)
{
   global struct mali_invocation_packed *vertex_invocation;
   global struct mali_draw_packed *vertex_dcd;

   if (draw->idvs_job != NULL) {
      vertex_invocation =
         (global struct mali_invocation_packed *)(draw->idvs_job +
                                                  pan_section_offset(
                                                     INDEXED_VERTEX_JOB,
                                                     INVOCATION));
      vertex_dcd = (global struct mali_draw_packed *)(draw->idvs_job +
                                                      pan_section_offset(
                                                         INDEXED_VERTEX_JOB,
                                                         VERTEX_DRAW));
   } else {
      vertex_invocation =
         (global struct mali_invocation_packed *)(draw->vertex_job +
                                                  pan_section_offset(
                                                     COMPUTE_JOB, INVOCATION));
      vertex_dcd = (global struct mali_draw_packed *)(draw->vertex_job +
                                                      pan_section_offset(
                                                         COMPUTE_JOB, DRAW));
   }

   /* Patch the number of invocations on the vertex job */
   pan_pack_work_groups_compute(vertex_invocation, 1, draw->vertex_range,
                                draw->instance.count, 1, 1, 1, true, false);

   /* Patch the draw descriptor */
   pan_unpack(vertex_dcd, DRAW, unpacked_vertex_dcd)
      ;
   pan_pack(vertex_dcd, DRAW, cfg) {
      memcpy(&cfg, &unpacked_vertex_dcd, sizeof(cfg));
      cfg.offset_start = draw->vertex.raw_offset;
      cfg.instance_size =
         draw->instance.count > 1 ? draw->padded_vertex_count : 1;
   }
}

static void
panlib_patch_draw_tiler_dcd(struct panlib_draw_info *draw)
{
   /* No tiler job, nothing to do here */
   if (draw->idvs_job == NULL && draw->tiler_job == NULL)
      return;

   global struct mali_draw_packed *tiler_dcd;
   global struct mali_primitive_packed *tiler_primitive;
   global struct mali_primitive_size_packed *tiler_primsz;

   if (draw->idvs_job != NULL) {
      tiler_dcd = (global struct mali_draw_packed *)(draw->idvs_job +
                                                     pan_section_offset(
                                                        INDEXED_VERTEX_JOB,
                                                        FRAGMENT_DRAW));
      tiler_primitive =
         (global struct mali_primitive_packed *)(draw->idvs_job +
                                                 pan_section_offset(
                                                    INDEXED_VERTEX_JOB,
                                                    PRIMITIVE));
      tiler_primsz =
         (global struct mali_primitive_size_packed *)(draw->idvs_job +
                                                      pan_section_offset(
                                                         INDEXED_VERTEX_JOB,
                                                         PRIMITIVE_SIZE));
   } else {
      global struct mali_invocation_packed *vertex_invocation =
         (global struct mali_invocation_packed *)(draw->vertex_job +
                                                  pan_section_offset(
                                                     COMPUTE_JOB, INVOCATION));
      global struct mali_invocation_packed *tiler_invocation =
         (global struct mali_invocation_packed *)(draw->tiler_job +
                                                  pan_section_offset(
                                                     TILER_JOB, INVOCATION));
      /* Patch the tiler job invocation if present (identical to the vertex
       * invocations) */
      memcpy(tiler_invocation, vertex_invocation, pan_size(INVOCATION));

      tiler_dcd = (global struct mali_draw_packed *)(draw->tiler_job +
                                                     pan_section_offset(
                                                        TILER_JOB, DRAW));
      tiler_primitive =
         (global struct mali_primitive_packed *)(draw->tiler_job +
                                                 pan_section_offset(TILER_JOB,
                                                                    PRIMITIVE));
      tiler_primsz =
         (global struct mali_primitive_size_packed *)(draw->tiler_job +
                                                      pan_section_offset(
                                                         TILER_JOB,
                                                         PRIMITIVE_SIZE));
   }

   /* Patch the attribute start offset */
   pan_unpack(tiler_dcd, DRAW, unpacked_tiler_dcd)
      ;
   pan_pack(tiler_dcd, DRAW, cfg) {
      memcpy(&cfg, &unpacked_tiler_dcd, sizeof(cfg));
      cfg.position = draw->position;
      cfg.offset_start = draw->vertex.raw_offset;
      cfg.instance_size =
         draw->instance.count > 1 ? draw->padded_vertex_count : 1;
      uint32_t primitives_per_instance =
         DIV_ROUND_UP(draw->padded_vertex_count, draw->primitive_vertex_count);
      cfg.instance_primitive_size =
         pan_padded_vertex_count(primitives_per_instance);
   }

   /* Patch the primitive vertex offset to take offset_start into account in
    * case of indexed draw */
   pan_unpack(tiler_primitive, PRIMITIVE, unpacked_tiler_primitive)
      ;

   pan_pack(tiler_primitive, PRIMITIVE, cfg) {
      memcpy(&cfg, &unpacked_tiler_primitive, sizeof(cfg));
      cfg.index_count = draw->vertex.count;

      if (draw->index.size) {
         cfg.indices =
            (uint64_t)draw->indices + draw->index.offset * draw->index.size;
         cfg.base_vertex_offset =
            (int64_t)draw->vertex.base - draw->vertex.raw_offset;
      }
   }

   /* In case we have a point size, we need to patch the descriptor there */
   if (unpacked_tiler_primitive.point_size_array_format !=
       MALI_POINT_SIZE_ARRAY_FORMAT_NONE) {
      pan_pack(tiler_primsz, PRIMITIVE_SIZE, cfg) {
         cfg.size_array = draw->psiz;
      }
   }
}

/* Returns false when the draw's varyings do not fit in what is left of the shared buffer. The
 * space is taken anyway (the offset is shared), so every later draw of the batch fails too, and
 * each is dropped rather than written over the varyings of a draw the fragment job still reads. */
static bool
panlib_patch_varying_bufs(struct panlib_draw_info *draw)
{
   bool fits = true;
   uint32_t vertex_count = draw->padded_vertex_count * draw->instance.count;

   for (uint32_t i = 0; i < PANLIB_VARY_BUF_MAX; i++) {
      global struct libpan_draw_helper_varying_buf_info *info =
         &draw->varying_bufs.info[0];

      global struct mali_attribute_buffer_packed *desc =
         &draw->varying_bufs.descs[i];
      pan_unpack(desc, ATTRIBUTE_BUFFER, unpacked_desc)
         ;

      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         memcpy(&cfg, &unpacked_desc, sizeof(cfg));
         cfg.size = vertex_count * cfg.stride;

         /* Ensure we are aligned properly */
         uint32_t offset_add = ALIGN_POT(cfg.size, 64);

         /* Allocate some space for ourself */
         uint32_t current_offset = atomic_fetch_add_explicit(
            &info->offset, offset_add, memory_order_relaxed);

         if (current_offset + cfg.size > info->size)
            fits = false;

         cfg.pointer = info->address + current_offset;

         if (i == PANLIB_VARY_BUF_POSITION)
            draw->position = cfg.pointer;
         else if (i == PANLIB_VARY_BUF_PSIZ)
            draw->psiz = cfg.pointer;
      }
   }
   return fits;
}

static void
panlib_patch_attrib_buf(struct panlib_draw_info *draw,
                        global struct mali_attribute_buffer_packed *desc,
                        struct libpan_draw_helper_attrib_buf_info info)
{
   uint32_t divisor = draw->padded_vertex_count * info.divisor;
   global struct mali_attribute_buffer_continuation_npot_packed *desc_ext =
      (global struct mali_attribute_buffer_continuation_npot_packed *)&desc[1];

   /* We are effectively recreating the attribute buffer descriptors. The only
    * trustworthy infos in the descriptor are the pointer and size */
   pan_unpack(desc, ATTRIBUTE_BUFFER, unpacked_desc)
      ;

   if (draw->instance.count <= 1) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = info.per_instance ? 0 : info.stride;
         cfg.pointer = unpacked_desc.pointer;
         cfg.size = unpacked_desc.size;
      }
   } else if (!info.per_instance) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_MODULUS;
         cfg.divisor = draw->padded_vertex_count;
         cfg.stride = info.stride;
         cfg.pointer = unpacked_desc.pointer;
         cfg.size = unpacked_desc.size;
      }
   } else if (!divisor) {
      /* instance_divisor == 0 means all instances share the same value.
       * Make it a 1D array with a zero stride.
       */
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = 0;
         cfg.pointer = unpacked_desc.pointer;
         cfg.size = unpacked_desc.size;
      }
   } else if (util_is_power_of_two_or_zero(divisor)) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.stride = info.stride;
         cfg.pointer = unpacked_desc.pointer;
         cfg.size = unpacked_desc.size;
         cfg.divisor_r = __builtin_ctz(divisor);
      }
   } else {
      uint32_t divisor_r = 0, divisor_e = 0;
      uint32_t divisor_d =
         pan_compute_npot_divisor(divisor, &divisor_r, &divisor_e);
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_NPOT_DIVISOR;
         cfg.stride = info.stride;
         cfg.pointer = unpacked_desc.pointer;
         cfg.size = unpacked_desc.size;
         cfg.divisor_r = divisor_r;
         cfg.divisor_e = divisor_e;
      }

      pan_pack(desc_ext, ATTRIBUTE_BUFFER_CONTINUATION_NPOT, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_CONTINUATION;
         cfg.divisor_numerator = divisor_d;
         cfg.divisor = info.divisor;
      }

      return;
   }

   /* If the buffer extension wasn't used, memset(0) */
   for (uint32_t i = 0; i < pan_size(ATTRIBUTE_BUFFER) / 4; i++)
      desc_ext->opaque[i] = 0;
}

static void
panlib_patch_attrib(struct panlib_draw_info *draw,
                    global struct mali_attribute_packed *desc,
                    struct libpan_draw_helper_attrib_info info)
{
   /* We are effectively recreating the attribute descriptors. The only
    * trustworthy infos in the descriptor are the buffer index and format */
   pan_unpack(desc, ATTRIBUTE, unpacked_desc)
      ;

   pan_pack(desc, ATTRIBUTE, cfg) {
      cfg.buffer_index = unpacked_desc.buffer_index;
      cfg.offset = info.base_offset + draw->instance.base * info.stride;
      cfg.offset_enable = true;
      cfg.format = unpacked_desc.format;
   }
}

static void
panlib_patch_job_type_header(global struct mali_job_header_packed *desc,
                             enum mali_job_type job_type)
{
   if (desc == NULL)
      return;

   pan_unpack(desc, JOB_HEADER, unpacked_desc)
      ;

   pan_pack(desc, JOB_HEADER, cfg) {
      memcpy(&cfg, &unpacked_desc, sizeof(cfg));
      cfg.type = job_type;
   }
}

static void
panlib_patch_draw(struct panlib_draw_info *draw)
{
   /* First of all, we need to ensure each jobs have the proper header */
   bool is_null_job = draw->vertex_range == 0 || draw->instance.count == 0;

   /* The varyings first: a draw they do not fit for is not run at all. */
   if (!is_null_job && !panlib_patch_varying_bufs(draw))
      is_null_job = true;

   if (draw->idvs_job != NULL) {
      panlib_patch_job_type_header(
         (global struct mali_job_header_packed *)draw->idvs_job,
         is_null_job ? MALI_JOB_TYPE_NULL : MALI_JOB_TYPE_INDEXED_VERTEX);
   } else {
      panlib_patch_job_type_header(
         (global struct mali_job_header_packed *)draw->vertex_job,
         is_null_job ? MALI_JOB_TYPE_NULL : MALI_JOB_TYPE_VERTEX);
      panlib_patch_job_type_header(
         (global struct mali_job_header_packed *)draw->tiler_job,
         is_null_job ? MALI_JOB_TYPE_NULL : MALI_JOB_TYPE_TILER);
   }

   /* In case there is nothing to draw, let's just bail out. */
   if (is_null_job)
      return;

   /* Patch the vertex draw and invocation descriptors */
   panlib_patch_draw_vertex_dcd(draw);

   /* Patch the tiler descriptors if present */
   panlib_patch_draw_tiler_dcd(draw);

   /* Finally, patch attribute descriptors */
   uint32_t num_vbs = util_last_bit(draw->attrib_bufs.valid);
   for (uint32_t i = 0; i < num_vbs; i++) {
      if (draw->attrib_bufs.valid & BITFIELD_BIT(i))
         panlib_patch_attrib_buf(draw, &draw->attrib_bufs.descs[i * 2],
                                 draw->attrib_bufs.infos[i]);
   }

   uint32_t num_vs_attribs = util_last_bit(draw->attribs.valid);
   for (uint32_t i = 0; i < num_vs_attribs; i++) {
      if (draw->attribs.valid & BITFIELD_BIT(i))
         panlib_patch_attrib(draw, &draw->attribs.descs[i],
                             draw->attribs.infos[i]);
   }
}

static uint32_t
padded_vertex_count(uint32_t vertex_count, uint32_t instance_count, bool idvs)
{
   if (instance_count == 1)
      return vertex_count;

   /* Index-Driven Vertex Shading requires different instances to
    * have different cache lines for position results. Each vertex
    * position is 16 bytes and the Mali cache line is 64 bytes, so
    * the instance count must be aligned to 4 vertices.
    */
   if (idvs)
      vertex_count = ALIGN_POT(vertex_count, 4);

   return pan_padded_vertex_count(vertex_count);
}

/* Argument ordering choosen to avoid any padding */

KERNEL(1)
panlib_draw_indirect_helper(
   global VkDrawIndirectCommand *cmd,
   global struct mali_attribute_buffer_packed *varying_bufs_descs,
   global struct libpan_draw_helper_varying_buf_info *varying_bufs_info,
   global struct mali_attribute_buffer_packed *attrib_bufs_descs,
   global struct libpan_draw_helper_attrib_buf_info *attrib_bufs_infos,
   uint32_t attrib_bufs_valid, uint32_t attribs_valid,
   global struct mali_attribute_packed *attribs_descs,
   global struct libpan_draw_helper_attrib_info *attribs_infos,
   global uint32_t *first_vertex_sysval, global uint32_t *first_instance_sysval,
   global uint32_t *raw_vertex_offset_sysval, global uint8_t *idvs_job,
   global uint8_t *vertex_job, global uint8_t *tiler_job,
   uint32_t primitive_vertex_count)
{
   const uint32_t vertex_count = cmd->vertexCount;
   const uint32_t instance_count = cmd->instanceCount;
   const uint32_t first_vertex = cmd->firstVertex;
   const uint32_t first_instance = cmd->firstInstance;

   struct panlib_draw_info draw = {
      .idvs_job = idvs_job,
      .vertex_job = vertex_job,
      .tiler_job = tiler_job,

      .vertex.base = first_vertex,
      .vertex.raw_offset = first_vertex,
      .vertex.count = vertex_count,
      .instance.base = first_instance,
      .instance.count = instance_count,

      .varying_bufs.descs = varying_bufs_descs,
      .varying_bufs.info = varying_bufs_info,

      .attrib_bufs.descs = attrib_bufs_descs,
      .attrib_bufs.infos = attrib_bufs_infos,
      .attrib_bufs.valid = attrib_bufs_valid,

      .attribs.descs = attribs_descs,
      .attribs.infos = attribs_infos,
      .attribs.valid = attribs_valid,

      .vertex_range = vertex_count,
      .padded_vertex_count =
         padded_vertex_count(vertex_count, instance_count, idvs_job != NULL),
      .primitive_vertex_count = primitive_vertex_count,
   };

   panlib_patch_draw(&draw);

   *first_vertex_sysval = draw.vertex.base;
   *first_instance_sysval = draw.instance.base;
   *raw_vertex_offset_sysval = draw.vertex.raw_offset;
}

KERNEL(1)
panlib_draw_indexed_indirect_helper(
   global VkDrawIndexedIndirectCommand *cmd, global uint8_t *index_buffer_ptr,
   global struct libpan_draw_helper_index_min_max_result *index_min_max_res,
   uint32_t index_size, uint32_t primitive_vertex_count,
   uint32_t attrib_bufs_valid, uint32_t attribs_valid,
   global struct mali_attribute_buffer_packed *varying_bufs_descs,
   global struct libpan_draw_helper_varying_buf_info *varying_bufs_info,
   global struct mali_attribute_buffer_packed *attrib_bufs_descs,
   global struct libpan_draw_helper_attrib_buf_info *attrib_bufs_infos,
   global struct mali_attribute_packed *attribs_descs,
   global struct libpan_draw_helper_attrib_info *attribs_infos,
   global uint32_t *first_vertex_sysval, global uint32_t *first_instance_sysval,
   global uint32_t *raw_vertex_offset_sysval, global uint8_t *idvs_job,
   global uint8_t *vertex_job, global uint8_t *tiler_job)
{
   const uint32_t index_count = cmd->indexCount;
   const uint32_t first_index = cmd->firstIndex;
   const uint32_t first_instance = cmd->firstInstance;
   const uint32_t instance_count = cmd->instanceCount;
   const int32_t vertex_offset = cmd->vertexOffset;
   const uint32_t min_vertex = index_min_max_res->min;
   const uint32_t max_vertex = index_min_max_res->max;
   const uint32_t vertex_range = max_vertex - min_vertex + 1;

   struct panlib_draw_info draw = {
      .idvs_job = idvs_job,
      .vertex_job = vertex_job,
      .tiler_job = tiler_job,

      .index.size = index_size,
      .index.offset = first_index,
      .vertex.base = vertex_offset,
      .vertex.raw_offset = min_vertex + vertex_offset,
      .vertex.count = index_count,
      .instance.base = first_instance,
      .instance.count = instance_count,

      .varying_bufs.descs = varying_bufs_descs,
      .varying_bufs.info = varying_bufs_info,

      .attrib_bufs.descs = attrib_bufs_descs,
      .attrib_bufs.infos = attrib_bufs_infos,
      .attrib_bufs.valid = attrib_bufs_valid,

      .attribs.descs = attribs_descs,
      .attribs.infos = attribs_infos,
      .attribs.valid = attribs_valid,

      .vertex_range = vertex_range,
      .padded_vertex_count =
         padded_vertex_count(vertex_range, instance_count, idvs_job != NULL),
      .primitive_vertex_count = primitive_vertex_count,
      .indices = index_buffer_ptr,
   };

   panlib_patch_draw(&draw);

   *first_vertex_sysval = draw.vertex.base;
   *first_instance_sysval = draw.instance.base;
   *raw_vertex_offset_sysval = draw.vertex.raw_offset;
}

KERNEL(64)
panlib_draw_index_minmax_search_helper(global uint8_t *index_buffer_ptr,
                                       global VkDrawIndexedIndirectCommand *cmd,
                                       global atomic_uint *min_ptr,
                                       global atomic_uint *max_ptr,
                                       uint32_t index_bytes_log2__3,
                                       uint8_t primitive_restart__2)
{
   /* Max count of values to process per thread */
   const uint32_t max_count_per_thread = 1024;

   const uint32_t index_bit_size = (1 << index_bytes_log2__3) * 8;
   const uint32_t start = cmd->firstIndex;
   const uint32_t index_count = cmd->indexCount;

   uint32_t base_idx = cl_global_id.x * max_count_per_thread;

   /* If the thread is out of range, bail out */
   if (base_idx >= index_count)
      return;

   /* Compute expected max iteration to do in this thread */
   uint32_t count = MIN2(max_count_per_thread, index_count - base_idx);

   /* Sanity check so nothing weird will happen */
   assert(base_idx + count <= index_count);

   uint32_t local_min = ((uint64_t)1 << index_bit_size) - 1;
   uint32_t local_max = 0;

   switch (index_bit_size) {
#define MINMAX_SEARCH_CASE(sz)                                                 \
   case sz: {                                                                  \
      global uint##sz##_t *indices = (global uint##sz##_t *)index_buffer_ptr;  \
      for (uint32_t i = 0; i < count; i++) {                                   \
         uint32_t val = (uint32_t)indices[start + base_idx + i];               \
         if (primitive_restart__2 && val == UINT##sz##_MAX)                    \
            continue;                                                          \
         local_min = min(local_min, val);                                      \
         local_max = max(local_max, val);                                      \
      }                                                                        \
      break;                                                                   \
   }
      MINMAX_SEARCH_CASE(32)
      MINMAX_SEARCH_CASE(16)
      MINMAX_SEARCH_CASE(8)
#undef MINMAX_SEARCH_CASE
   default:
      assert(0 && "Invalid index size");
   }

   atomic_fetch_min(min_ptr, local_min);
   atomic_fetch_max(max_ptr, local_max);
}

/* The batch-wide helpers. Per draw, the scan and the patch were a chain of four jobs of one or a
 * few threads each (write the min/max seed, scan with one thread per 1024 indices, patch with one
 * thread), run back to back before the draw's vertex job: measured on the S10e, 0.38 ms of GPU
 * time per draw whatever its size, 63 ms for 150 draws the CPU path draws in 4.7. The work is
 * tiny; the latency of each job and each serial thread was the cost. Here one scan job and one
 * patch job serve every draw of the batch, with the draws spread over workgroups and the indices
 * of a draw over threads. */

static void
panlib_draw_batch_scan_entry(global struct libpan_draw_batch_entry *e, uint32_t tid,
                             uint32_t stride)
{
   if (!e->index_size)
      return;

   global VkDrawIndexedIndirectCommand *cmd = (global VkDrawIndexedIndirectCommand *)e->cmd;
   const uint32_t start = cmd->firstIndex;
   const uint32_t count = cmd->indexCount;
   const bool restart = e->primitive_restart != 0;
   uint32_t lo = UINT32_MAX, hi = 0;

   switch (e->index_size) {
#define BATCH_SCAN_CASE(sz)                                                    \
   case sz / 8: {                                                              \
      global uint##sz##_t *ib = (global uint##sz##_t *)e->index_buffer_ptr + start; \
      for (uint32_t i = tid; i < count; i += stride) {                         \
         const uint32_t v = ib[i];                                             \
         if (restart && v == UINT##sz##_MAX)                                   \
            continue;                                                          \
         lo = min(lo, v);                                                      \
         hi = max(hi, v);                                                      \
      }                                                                        \
      break;                                                                   \
   }
      BATCH_SCAN_CASE(32)
      BATCH_SCAN_CASE(16)
      BATCH_SCAN_CASE(8)
#undef BATCH_SCAN_CASE
   default:
      return;
   }

   if (lo <= hi) {
      atomic_fetch_min((global atomic_uint *)&e->index_min, lo);
      atomic_fetch_max((global atomic_uint *)&e->index_max, hi);
   }
}

/* The [min, max] index of every indexed draw of the batch. Fewer draws than workgroups: each draw
 * is split between groups / count of them; more: each workgroup takes every groups-th draw. */
KERNEL(64)
panlib_draw_batch_minmax(global struct libpan_draw_batch *batch)
{
   global struct libpan_draw_batch_entry *entries =
      (global struct libpan_draw_batch_entry *)batch->entries;
   const uint32_t n = batch->count;
   const uint32_t groups = cl_global_size.x / 64;
   const uint32_t g = cl_global_id.x / 64;
   const uint32_t lid = cl_global_id.x % 64;

   if (n == 0)
      return;

   if (n < groups) {
      const uint32_t slices = groups / n;
      if (g / slices < n)
         panlib_draw_batch_scan_entry(&entries[g / slices], (g % slices) * 64 + lid, slices * 64);
   } else {
      for (uint32_t e = g; e < n; e += groups)
         panlib_draw_batch_scan_entry(&entries[e], lid, 64);
   }
}

/* Every draw's descriptors and sysvals, one thread per draw. */
KERNEL(64)
panlib_draw_batch_patch(global struct libpan_draw_batch *batch)
{
   global struct libpan_draw_batch_entry *entries =
      (global struct libpan_draw_batch_entry *)batch->entries;

   for (uint32_t i = cl_global_id.x; i < batch->count; i += cl_global_size.x) {
      global struct libpan_draw_batch_entry *e = &entries[i];

      struct panlib_draw_info draw = {
         .idvs_job = (global uint8_t *)e->idvs_job,
         .vertex_job = (global uint8_t *)e->vertex_job,
         .tiler_job = (global uint8_t *)e->tiler_job,

         .varying_bufs.descs = (global struct mali_attribute_buffer_packed *)e->varying_bufs_descs,
         .varying_bufs.info =
            (global struct libpan_draw_helper_varying_buf_info *)e->varying_bufs_info,

         .attrib_bufs.descs = (global struct mali_attribute_buffer_packed *)e->attrib_bufs_descs,
         .attrib_bufs.infos =
            (global struct libpan_draw_helper_attrib_buf_info *)e->attrib_bufs_infos,
         .attrib_bufs.valid = e->attrib_bufs_valid,

         .attribs.descs = (global struct mali_attribute_packed *)e->attribs_descs,
         .attribs.infos = (global struct libpan_draw_helper_attrib_info *)e->attribs_infos,
         .attribs.valid = e->attribs_valid,

         .primitive_vertex_count = e->primitive_vertex_count,
      };

      if (e->index_size) {
         global VkDrawIndexedIndirectCommand *cmd = (global VkDrawIndexedIndirectCommand *)e->cmd;
         const uint32_t lo = e->index_min, hi = e->index_max;

         draw.index.size = e->index_size;
         draw.index.offset = cmd->firstIndex;
         draw.vertex.base = cmd->vertexOffset;
         draw.vertex.raw_offset = lo + cmd->vertexOffset;
         draw.vertex.count = cmd->indexCount;
         draw.instance.base = cmd->firstInstance;
         draw.instance.count = cmd->instanceCount;
         /* No index but restarts: nothing to draw. */
         draw.vertex_range = cmd->indexCount && lo <= hi ? hi - lo + 1 : 0;
         draw.indices = (global uint8_t *)e->index_buffer_ptr;

         if (e->index_min_out)
            *(global uint32_t *)e->index_min_out = lo;
         e->index_min = UINT32_MAX;
         e->index_max = 0;
      } else {
         global VkDrawIndirectCommand *cmd = (global VkDrawIndirectCommand *)e->cmd;

         draw.vertex.base = cmd->firstVertex;
         draw.vertex.raw_offset = cmd->firstVertex;
         draw.vertex.count = cmd->vertexCount;
         draw.instance.base = cmd->firstInstance;
         draw.instance.count = cmd->instanceCount;
         draw.vertex_range = cmd->vertexCount;
      }

      draw.padded_vertex_count =
         padded_vertex_count(draw.vertex_range, draw.instance.count, draw.idvs_job != NULL);

      panlib_patch_draw(&draw);

      if (e->first_vertex_sysval)
         *(global uint32_t *)e->first_vertex_sysval = draw.vertex.base;
      if (e->first_instance_sysval)
         *(global uint32_t *)e->first_instance_sysval = draw.instance.base;
      if (e->raw_vertex_offset_sysval)
         *(global uint32_t *)e->raw_vertex_offset_sysval = draw.vertex.raw_offset;
   }
}

/* Geometry shader + primitive restart. Which vertices input primitive p takes depends on every
 * restart index before it, so one thread walks the index buffer, splits it into segments at the
 * restart value, and writes each segment's primitives out as a list of 32-bit indices in the
 * geometry shader's input order, plus a draw command naming how many there are. The emulation
 * then reads that as an indexed LIST. Topology numbers are enum panvk_gs_topology. */
static uint32_t
panlib_gs_read_index(global uint8_t *ib, uint32_t size, uint32_t i)
{
   if (size == 1)
      return ib[i];
   if (size == 2)
      return ((global uint16_t *)ib)[i];
   return ((global uint32_t *)ib)[i];
}

static uint32_t
panlib_gs_seg_prims(uint32_t topology, uint32_t vpp, uint32_t len)
{
   if (topology == 1)
      return len >= vpp ? len - (vpp - 1) : 0;
   if (topology == 2)
      return len >= 3 ? len - 2 : 0;
   if (topology == 3)
      return len >= 6 ? (len - 4) / 2 : 0;
   return len / vpp;
}

/* Position inside the segment of vertex k of primitive p; the same rules as gs_input_record(). */
static uint32_t
panlib_gs_seg_vertex(uint32_t topology, uint32_t vpp, uint32_t p, uint32_t k, uint32_t nprims)
{
   if (topology == 1) {
      if (vpp == 3 && (p & 1))
         return p + (k == 1 ? 2 : k == 2 ? 1 : 0);
      return p + k;
   }
   if (topology == 2)
      return k == 2 ? 0 : p + k + 1;
   if (topology == 3) {
      const uint32_t p2 = p * 2;
      const uint32_t far = p == nprims - 1 ? p2 + 5 : p2 + 6;
      if (p & 1)
         return k == 0 ? p2 + 2 : k == 1 ? p2 - 2 : k == 2 ? p2 : k == 3 ? p2 + 3
              : k == 4 ? p2 + 4 : far;
      return k == 0 ? p2 : k == 1 ? (p == 0 ? 1 : p2 - 2) : k == 2 ? p2 + 2 : k == 3 ? far
           : k == 4 ? p2 + 4 : p2 + 3;
   }
   return p * vpp + k;
}

KERNEL(1)
panlib_gs_restart_unroll(global VkDrawIndexedIndirectCommand *cmd, global uint8_t *ib,
                         global uint32_t *out, global VkDrawIndexedIndirectCommand *out_cmd,
                         uint32_t index_size, uint32_t topology, uint32_t vpp,
                         uint32_t max_prims)
{
   const uint32_t n = cmd->indexCount;
   global uint8_t *base = ib + (uint64_t)cmd->firstIndex * index_size;
   const uint32_t restart =
      index_size == 1 ? 0xffu : index_size == 2 ? 0xffffu : 0xffffffffu;

   uint32_t prims = 0, seg = 0;
   for (uint32_t i = 0; i <= n && prims < max_prims; i++) {
      if (i < n && panlib_gs_read_index(base, index_size, i) != restart)
         continue;

      const uint32_t np = panlib_gs_seg_prims(topology, vpp, i - seg);
      for (uint32_t p = 0; p < np && prims < max_prims; p++, prims++) {
         for (uint32_t k = 0; k < vpp; k++)
            out[prims * vpp + k] = panlib_gs_read_index(
               base, index_size, seg + panlib_gs_seg_vertex(topology, vpp, p, k, np));
      }
      seg = i + 1;
   }

   out_cmd->indexCount = prims * vpp;
   out_cmd->instanceCount = cmd->instanceCount;
   out_cmd->firstIndex = 0;
   out_cmd->vertexOffset = cmd->vertexOffset;
   out_cmd->firstInstance = cmd->firstInstance;
}

/* Transform feedback. The geometry shader emulation left, per invocation, how many complete
 * primitives it emitted and a staging record for every vertex slot. Capture order is draw order,
 * which is invocation order, so one thread turns the counts into each invocation's first
 * primitive, fits the draw into the buffers, and advances the write offsets. Every primitive of a
 * draw is the same size, so the ones that fit are a prefix. */
KERNEL(1)
panlib_xfb_scan(global struct libpan_xfb_draw *d)
{
   global uint32_t *counts = (global uint32_t *)d->counts;
   global uint32_t *first = (global uint32_t *)d->first;
   global uint32_t *off_in = (global uint32_t *)d->offsets_in;
   global uint32_t *off_out = (global uint32_t *)d->offsets_out;

   uint32_t total = 0;
   for (uint32_t i = 0; i < d->invocations; i++) {
      first[i] = total;
      total += counts[i];
   }

   uint32_t fit = total;
   for (uint32_t b = 0; b < 4; b++) {
      const uint32_t o = off_in[b];
      d->start[b] = o;
      if (!d->stride[b])
         continue;
      const uint32_t room = d->size[b] > o ? d->size[b] - o : 0;
      fit = min(fit, room / (d->stride[b] * d->vpp));
   }

   for (uint32_t b = 0; b < 4; b++)
      off_out[b] = off_in[b] + fit * d->stride[b] * d->vpp;

   d->written = fit;
   d->needed = total;
}

/* One thread per emulation invocation: its primitives' vertices, record dword by record dword,
 * to where the table says. Only the captured dwords are written; gaps keep what was there. */
KERNEL(64)
panlib_xfb_copy(global struct libpan_xfb_draw *d)
{
   const uint32_t inv = cl_global_id.x;
   if (inv >= d->invocations)
      return;

   const uint32_t p0 = ((global uint32_t *)d->first)[inv];
   const uint32_t written = d->written;
   if (p0 >= written)
      return;

   const uint32_t n = min(((global uint32_t *)d->counts)[inv], written - p0);
   const uint32_t vpp = d->vpp;
   const uint32_t k_dw = d->record_dwords;
   global uint32_t *idx = (global uint32_t *)d->indices + (uint64_t)inv * d->idx_per_invocation;
   global uint32_t *staging = (global uint32_t *)d->staging;
   global uint32_t *table = (global uint32_t *)d->table;

   for (uint32_t p = 0; p < n; p++) {
      for (uint32_t k = 0; k < vpp; k++) {
         global uint32_t *rec = staging + (uint64_t)idx[p * vpp + k] * k_dw;
         const uint64_t v = (uint64_t)(p0 + p) * vpp + k;

         for (uint32_t j = 0; j < k_dw; j++) {
            const uint32_t e = table[j];
            const uint32_t b = e >> 16;
            /* Another stream's buffer (geometryStreams): not this pass's to write. */
            if (!d->stride[b])
               continue;
            global uint32_t *dst =
               (global uint32_t *)(d->buffer[b] + d->start[b] + v * d->stride[b]);
            dst[e & 0xffff] = rec[j];
         }
      }
   }
}

/* vkCmdBegin/EndTransformFeedback with counter buffers: load the write offsets from them
 * (load != 0, a missing counter starts at 0) or store the offsets into them. */
KERNEL(1)
panlib_xfb_counters(global uint32_t *offsets, uint64_t c0, uint64_t c1, uint64_t c2,
                    uint64_t c3, uint32_t load)
{
   const uint64_t c[4] = {c0, c1, c2, c3};

   for (uint32_t b = 0; b < 4; b++) {
      global uint32_t *counter = (global uint32_t *)c[b];
      if (load)
         offsets[b] = counter ? *counter : 0;
      else if (counter)
         *counter = offsets[b];
   }
}
#endif
