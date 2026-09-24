/*
 * Copyright © 2024 Collabora Ltd.
 * Copyright © 2026 NXP
 *
 * Derived from tu_cmd_buffer.c which is:
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "genxml/gen_macros.h"

#include "panvk_buffer.h"
#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_desc_state.h"
#include "panvk_cmd_draw.h"
#include "panvk_cmd_meta.h"
#include "panvk_cmd_precomp.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_gs.h"
#include "panvk_image.h"
#include "panvk_image_view.h"
#include "panvk_instance.h"
#include "panvk_meta.h"
#include "panvk_priv_bo.h"
#include "panvk_shader.h"

#include "draw_helper.h"
#include "poly/geometry.h"
#include "poly/tessellator.h"
#include "pan_desc.h"
#include "pan_earlyzs.h"
#include "pan_encoder.h"
#include "pan_format.h"
#include "pan_jc.h"
#include "pan_props.h"
#include "pan_shader.h"

#include "vk_format.h"
#include "vk_meta.h"
#include "vk_pipeline_layout.h"

struct panvk_draw_data {
   struct panvk_draw_info info;
   /* Indexed indirect draws: where the index search helper leaves the smallest index. */
   uint64_t index_min_addr;
   /* prepare_vs_attribs allocated new image tables for the vertex shader. */
   bool vs_img_tables_new;
   unsigned vertex_range;
   unsigned padded_vertex_count;
   struct mali_invocation_packed invocation;
   struct {
      uint64_t varyings;
      uint64_t attributes;
      uint64_t attribute_bufs;
   } vs;
   struct {
      uint64_t rsd;
      uint64_t varyings;
   } fs;
   uint64_t varying_bufs;
   /* With a geometry shader the tiler reads a DIFFERENT set of varying buffers from the ones the
    * vertex job wrote: the emulation job in between produces them. Zero when there is no GS. */
   uint64_t gs_varying_bufs;
   uint64_t position;
   union {
      uint64_t psiz;
      float line_width;
   };
   uint64_t tls;
   uint64_t fb;
   const struct pan_tiler_context *tiler_ctx;
   uint64_t viewport;
   struct {
      struct pan_ptr vertex_copy_desc;
      struct pan_ptr frag_copy_desc;
      union {
         struct {
            struct pan_ptr vertex;
            struct pan_ptr tiler;
         };
         struct pan_ptr idvs;
      };
   } jobs;
   struct {
      uint64_t attribs;
      uint64_t attrib_bufs;
      uint64_t varying_bufs;
   } indirect_info;
};

/* Geometry shader input assembly: how the draw's topology groups vertices into the shader's
 * input primitives, and how many primitives one instance makes. */
static enum panvk_gs_topology
gs_topology(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_LINE_STRIP:
   case MESA_PRIM_TRIANGLE_STRIP:
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      return PANVK_GS_TOPO_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return PANVK_GS_TOPO_FAN;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      return PANVK_GS_TOPO_TRI_STRIP_ADJ;
   default:
      return PANVK_GS_TOPO_LIST;
   }
}

static uint32_t
gs_prims_per_instance(const struct panvk_draw_data *draw,
                      const struct panvk_shader_variant *gs)
{
   const uint32_t n = draw->info.vertex.count;
   const uint32_t vpp = MAX2(gs->gs.input_verts_per_prim, 1);

   switch (gs_topology(draw->info.prim)) {
   case PANVK_GS_TOPO_STRIP:
      return n >= vpp ? n - (vpp - 1) : 0;
   case PANVK_GS_TOPO_FAN:
      return n >= 3 ? n - 2 : 0;
   case PANVK_GS_TOPO_TRI_STRIP_ADJ:
      return n >= 6 ? (n - 4) / 2 : 0;
   default:
      return n / vpp;
   }
}

/* Tessellation: the patches a draw has, and the indices its tess eval job is sized for. 64 is
 * maxTessellationGenerationLevel, where a quad domain is 64x64 quads of two triangles. */
static uint32_t
tess_patches(const struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_data *draw)
{
   const uint32_t pcp = MAX2(cmdbuf->vk.dynamic_graphics_state.ts.patch_control_points, 1);
   return (draw->info.vertex.count / pcp) * draw->info.instance.count;
}

static uint32_t
tess_index_cap(const struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_data *draw)
{
   /* An indirect draw's patch count is only on the GPU. */
   if (draw->info.indirect.buffer_dev_addr)
      return PANVK_TESS_MAX_INDICES;
   const uint64_t n = (uint64_t)tess_patches(cmdbuf, draw) * 64 * 64 * 6;
   return MIN2(n, PANVK_TESS_MAX_INDICES);
}

/* The tess mode: either stage may declare it, the tess eval shader wins. */
static void
tess_mode(const struct panvk_cmd_buffer *cmdbuf, enum tess_primitive_mode *prim,
          enum gl_tess_spacing *spacing, bool *ccw, bool *points)
{
   const struct panvk_shader_variant *tcs =
      panvk_shader_only_variant(cmdbuf->state.gfx.tess.tcs);
   const struct panvk_shader_variant *tes =
      panvk_shader_only_variant(cmdbuf->state.gfx.tess.tes);
   *prim = tes->gs.tess_prim ? tes->gs.tess_prim : tcs->gs.tess_prim;
   *spacing = tes->gs.tess_spacing ? tes->gs.tess_spacing : tcs->gs.tess_spacing;
   *ccw = tes->gs.tess_ccw || tcs->gs.tess_ccw;
   *points = tes->gs.tess_point_mode || tcs->gs.tess_point_mode;
}

/* An indirect draw is one whose parameters live in GPU memory and are therefore not knowable
 * while the command buffer is recorded. Being INDEXED is a different property entirely.
 *
 * This used to answer true for any indexed draw, which forced every vkCmdDrawIndexed through
 * the indirect machinery: the descriptors were emitted as placeholders (index_count = 1,
 * offset_start / instance_size / instance_primitive_size left at zero, vertex_range zero) for two
 * precompiled OpenCL kernels to patch on the GPU before the vertex job read them. On the Mali job
 * manager that patch never lands -- measured identically on Mali-G76/arch7/r32p1 and
 * Mali-G72/arch6/r38p1, and independent of the argument buffer's contents:
 *
 *   vkCmdDraw                  triangle rasterised
 *   vkCmdDrawIndirect          no fault, nothing drawn  (the jobs stay MALI_JOB_TYPE_NULL)
 *   vkCmdDrawIndexed           event 88 DATA_INVALID on both atoms, even with indexCount = 0
 *
 * A placeholder primitive descriptor is precisely what DATA_INVALID describes. The vendor driver
 * does not use a GPU helper for indexed draws at all -- captured on both devices, its indexed
 * chain is byte-identical in shape to its array chain (VERTEX -> TILER -> FRAGMENT) with the
 * descriptor filled in by the CPU -- and so does gallium panfrost, which renders correctly on
 * this same hardware. So a direct indexed draw takes the CPU path here too, and the helper is
 * kept for draws that genuinely need it. */
static bool
is_indirect_draw(const struct panvk_draw_data *draw)
{
   return draw->info.indirect.buffer_dev_addr != 0;
}

static bool
has_depth_att(struct panvk_cmd_buffer *cmdbuf)
{
   return (cmdbuf->state.gfx.render.bound_attachments &
           MESA_VK_RP_ATTACHMENT_DEPTH_BIT) != 0;
}

static bool
has_stencil_att(struct panvk_cmd_buffer *cmdbuf)
{
   return (cmdbuf->state.gfx.render.bound_attachments &
           MESA_VK_RP_ATTACHMENT_STENCIL_BIT) != 0;
}

static bool
writes_depth(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   return has_depth_att(cmdbuf) && ds->depth.test_enable &&
          ds->depth.write_enable && ds->depth.compare_op != VK_COMPARE_OP_NEVER;
}

static bool
writes_stencil(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   return has_stencil_att(cmdbuf) && ds->stencil.test_enable &&
          ((ds->stencil.front.write_mask &&
            (ds->stencil.front.op.fail != VK_STENCIL_OP_KEEP ||
             ds->stencil.front.op.pass != VK_STENCIL_OP_KEEP ||
             ds->stencil.front.op.depth_fail != VK_STENCIL_OP_KEEP)) ||
           (ds->stencil.back.write_mask &&
            (ds->stencil.back.op.fail != VK_STENCIL_OP_KEEP ||
             ds->stencil.back.op.pass != VK_STENCIL_OP_KEEP ||
             ds->stencil.back.op.depth_fail != VK_STENCIL_OP_KEEP)));
}

static bool
ds_test_always_passes(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   if (!has_depth_att(cmdbuf))
      return true;

   if (ds->depth.test_enable && ds->depth.compare_op != VK_COMPARE_OP_ALWAYS)
      return false;

   if (ds->stencil.test_enable &&
       (ds->stencil.front.op.compare != VK_COMPARE_OP_ALWAYS ||
        ds->stencil.back.op.compare != VK_COMPARE_OP_ALWAYS))
      return false;

   return true;
}

static inline enum mali_func
translate_compare_func(VkCompareOp comp)
{
   STATIC_ASSERT(VK_COMPARE_OP_NEVER == (VkCompareOp)MALI_FUNC_NEVER);
   STATIC_ASSERT(VK_COMPARE_OP_LESS == (VkCompareOp)MALI_FUNC_LESS);
   STATIC_ASSERT(VK_COMPARE_OP_EQUAL == (VkCompareOp)MALI_FUNC_EQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_LESS_OR_EQUAL == (VkCompareOp)MALI_FUNC_LEQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_GREATER == (VkCompareOp)MALI_FUNC_GREATER);
   STATIC_ASSERT(VK_COMPARE_OP_NOT_EQUAL == (VkCompareOp)MALI_FUNC_NOT_EQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_GREATER_OR_EQUAL ==
                 (VkCompareOp)MALI_FUNC_GEQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_ALWAYS == (VkCompareOp)MALI_FUNC_ALWAYS);

   return (enum mali_func)comp;
}

static enum mali_stencil_op
translate_stencil_op(VkStencilOp in)
{
   switch (in) {
   case VK_STENCIL_OP_KEEP:
      return MALI_STENCIL_OP_KEEP;
   case VK_STENCIL_OP_ZERO:
      return MALI_STENCIL_OP_ZERO;
   case VK_STENCIL_OP_REPLACE:
      return MALI_STENCIL_OP_REPLACE;
   case VK_STENCIL_OP_INCREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_INCR_SAT;
   case VK_STENCIL_OP_DECREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_DECR_SAT;
   case VK_STENCIL_OP_INCREMENT_AND_WRAP:
      return MALI_STENCIL_OP_INCR_WRAP;
   case VK_STENCIL_OP_DECREMENT_AND_WRAP:
      return MALI_STENCIL_OP_DECR_WRAP;
   case VK_STENCIL_OP_INVERT:
      return MALI_STENCIL_OP_INVERT;
   default:
      UNREACHABLE("Invalid stencil op");
   }
}

static VkResult
panvk_draw_prepare_fs_rsd(struct panvk_cmd_buffer *cmdbuf,
                          struct panvk_draw_data *draw)
{
   bool dirty = dyn_gfx_state_dirty(cmdbuf, RS_RASTERIZER_DISCARD_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLAMP_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLIP_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_BIAS_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_BIAS_FACTORS) ||
                dyn_gfx_state_dirty(cmdbuf, RS_LINE_MODE) ||
                /* line mode needs primitive topology */
                dyn_gfx_state_dirty(cmdbuf, IA_PRIMITIVE_TOPOLOGY) ||
                dyn_gfx_state_dirty(cmdbuf, CB_LOGIC_OP_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, CB_LOGIC_OP) ||
                dyn_gfx_state_dirty(cmdbuf, CB_ATTACHMENT_COUNT) ||
                dyn_gfx_state_dirty(cmdbuf, CB_COLOR_WRITE_ENABLES) ||
                dyn_gfx_state_dirty(cmdbuf, CB_BLEND_ENABLES) ||
                dyn_gfx_state_dirty(cmdbuf, CB_BLEND_EQUATIONS) ||
                dyn_gfx_state_dirty(cmdbuf, CB_WRITE_MASKS) ||
                dyn_gfx_state_dirty(cmdbuf, CB_BLEND_CONSTANTS) ||
                dyn_gfx_state_dirty(cmdbuf, COLOR_ATTACHMENT_MAP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_TEST_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_WRITE_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_COMPARE_OP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_COMPARE_OP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_TEST_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_OP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_COMPARE_MASK) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_WRITE_MASK) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_REFERENCE) ||
                dyn_gfx_state_dirty(cmdbuf, MS_RASTERIZATION_SAMPLES) ||
                dyn_gfx_state_dirty(cmdbuf, MS_SAMPLE_MASK) ||
                dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_COVERAGE_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_ONE_ENABLE) ||
                gfx_state_dirty(cmdbuf, FS) || gfx_state_dirty(cmdbuf, OQ) ||
                gfx_state_dirty(cmdbuf, RENDER_STATE);

   if (!dirty) {
      draw->fs.rsd = cmdbuf->state.gfx.fs.rsd;
      return VK_SUCCESS;
   }

   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_rasterization_state *rs = &dyns->rs;
   const struct vk_depth_stencil_state *ds = &dyns->ds;
   const struct vk_input_assembly_state *ia = &dyns->ia;
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct pan_shader_info *fs_info = fs ? &fs->info : NULL;
   uint32_t bd_count = cmdbuf->state.gfx.render.fb.layout.rt_count;
   bool test_s = has_stencil_att(cmdbuf) && ds->stencil.test_enable;
   bool test_z = has_depth_att(cmdbuf) && ds->depth.test_enable;
   bool writes_z = writes_depth(cmdbuf);
   bool writes_s = writes_stencil(cmdbuf);

   bool msaa = dyns->ms.rasterization_samples > 1;
   if ((ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
        ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP) &&
       rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM) {
      /* we need to disable MSAA when rendering bresenham lines.
       *
       * From the Vulkan spec:
       *   "When Bresenham lines are being rasterized, sample locations may
       *    all be treated as being at the pixel center (this may affect
       *    attribute and depth interpolation).""
       */
      msaa = false;
   }

   struct pan_ptr ptr = panvk_cmd_alloc_desc_aggregate(
      cmdbuf, PAN_DESC(RENDERER_STATE), PAN_DESC_ARRAY(bd_count, BLEND));
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct mali_renderer_state_packed *rsd = ptr.cpu;
   struct mali_blend_packed *bds = ptr.cpu + pan_size(RENDERER_STATE);
   struct panvk_blend_info *binfo = &cmdbuf->state.gfx.cb.info;

   uint64_t fs_code = panvk_shader_variant_get_dev_addr(fs);

   if (fs_info != NULL) {
      panvk_per_arch(blend_emit_descs)(cmdbuf, bds);
   } else {
      for (unsigned i = 0; i < bd_count; i++) {
         pan_pack(&bds[i], BLEND, cfg) {
            cfg.enable = false;
            cfg.internal.mode = MALI_BLEND_MODE_OFF;
         }
      }
   }

   pan_pack(rsd, RENDERER_STATE, cfg) {
      bool alpha_to_coverage = dyns->ms.alpha_to_coverage_enable;

      if (fs) {
         pan_shader_prepare_rsd(fs_info, fs_code, &cfg);

         uint8_t rt_mask = cmdbuf->state.gfx.render.bound_attachments &
                           MESA_VK_RP_ATTACHMENT_ANY_COLOR_BITS;
         uint8_t rt_written = color_attachment_written_mask(
            fs, &cmdbuf->vk.dynamic_graphics_state.cal);
         uint8_t rt_read = color_attachment_read_mask(fs, &dyns->ial, rt_mask);
         enum pan_earlyzs_zs_tilebuf_read zs_read =
            (z_attachment_read(fs, &dyns->ial) ||
             s_attachment_read(fs, &dyns->ial))
               ? PAN_EARLYZS_ZS_TILEBUF_READ_NO_OPT
               : PAN_EARLYZS_ZS_TILEBUF_NOT_READ;

         cfg.properties.allow_forward_pixel_to_kill =
            fs_info->fs.can_fpk && !(rt_mask & ~rt_written) &&
            !(rt_read & rt_written) && !alpha_to_coverage &&
            !binfo->any_dest_read;

         bool writes_zs = writes_z || writes_s;
         bool zs_always_passes = ds_test_always_passes(cmdbuf);
         bool oq = cmdbuf->state.gfx.occlusion_query.mode !=
                   MALI_OCCLUSION_MODE_DISABLED;

         struct pan_earlyzs_state earlyzs =
            pan_earlyzs_get(fs->fs.earlyzs_lut, writes_zs || oq,
                            alpha_to_coverage, zs_always_passes, zs_read);

         /* early ZS check for FPK is performed by HW on v7+ */
         cfg.properties.allow_forward_pixel_to_be_killed =
            !fs->info.writes_global &&
            ((PAN_ARCH > 6) || earlyzs.kill != MALI_PIXEL_KILL_FORCE_LATE);

         cfg.properties.pixel_kill_operation = earlyzs.kill;
         cfg.properties.zs_update_operation = earlyzs.update;
         cfg.multisample_misc.evaluate_per_sample =
            (fs->info.fs.sample_shading && dyns->ms.rasterization_samples > 1);
      } else {
         cfg.properties.depth_source = MALI_DEPTH_SOURCE_FIXED_FUNCTION;
         cfg.properties.allow_forward_pixel_to_kill = true;
         cfg.properties.allow_forward_pixel_to_be_killed = true;
         cfg.properties.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
      }

      cfg.multisample_misc.multisample_enable = msaa;
      cfg.multisample_misc.sample_mask = dyns->ms.sample_mask;

      cfg.multisample_misc.depth_function =
         test_z ? translate_compare_func(ds->depth.compare_op)
                : MALI_FUNC_ALWAYS;

      cfg.multisample_misc.depth_write_mask = writes_z;
      cfg.multisample_misc.fixed_function_near_discard =
      cfg.multisample_misc.fixed_function_far_discard =
         vk_rasterization_state_depth_clip_enable(rs);
      cfg.multisample_misc.fixed_function_depth_range_fixed =
         !rs->depth_clamp_enable;
      cfg.multisample_misc.shader_depth_range_fixed = true;

      cfg.stencil_mask_misc.stencil_enable = test_s;
      cfg.stencil_mask_misc.alpha_to_coverage = alpha_to_coverage;
      cfg.stencil_mask_misc.alpha_test_compare_function = MALI_FUNC_ALWAYS;
      cfg.stencil_mask_misc.front_facing_depth_bias = rs->depth_bias.enable;
      cfg.stencil_mask_misc.back_facing_depth_bias = rs->depth_bias.enable;

      if (rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM)
         cfg.stencil_mask_misc.aligned_line_ends = true;

      cfg.depth_units = rs->depth_bias.constant_factor;
      cfg.depth_factor = rs->depth_bias.slope_factor;
      cfg.depth_bias_clamp = rs->depth_bias.clamp;

      cfg.stencil_front.mask = ds->stencil.front.compare_mask;
      cfg.stencil_back.mask = ds->stencil.back.compare_mask;

      cfg.stencil_mask_misc.stencil_mask_front = ds->stencil.front.write_mask;
      cfg.stencil_mask_misc.stencil_mask_back = ds->stencil.back.write_mask;

      cfg.stencil_front.reference_value = ds->stencil.front.reference;
      cfg.stencil_back.reference_value = ds->stencil.back.reference;

      if (test_s) {
         cfg.stencil_front.compare_function =
            translate_compare_func(ds->stencil.front.op.compare);
         cfg.stencil_front.stencil_fail =
            translate_stencil_op(ds->stencil.front.op.fail);
         cfg.stencil_front.depth_fail =
            translate_stencil_op(ds->stencil.front.op.depth_fail);
         cfg.stencil_front.depth_pass =
            translate_stencil_op(ds->stencil.front.op.pass);
         cfg.stencil_back.compare_function =
            translate_compare_func(ds->stencil.back.op.compare);
         cfg.stencil_back.stencil_fail =
            translate_stencil_op(ds->stencil.back.op.fail);
         cfg.stencil_back.depth_fail =
            translate_stencil_op(ds->stencil.back.op.depth_fail);
         cfg.stencil_back.depth_pass =
            translate_stencil_op(ds->stencil.back.op.pass);
      }
   }

   cmdbuf->state.gfx.fs.rsd = ptr.gpu;
   draw->fs.rsd = cmdbuf->state.gfx.fs.rsd;
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_tiler_context(struct panvk_cmd_buffer *cmdbuf,
                                 struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   VkResult result =
      panvk_per_arch(cmd_prepare_tiler_context)(cmdbuf, draw->info.layer_id);
   if (result != VK_SUCCESS)
      return result;

   draw->tiler_ctx = &batch->tiler.ctx;
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_varyings(struct panvk_cmd_buffer *cmdbuf,
                            struct panvk_draw_data *draw)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_link *link = &cmdbuf->state.gfx.link;
   struct pan_ptr bufs = panvk_cmd_alloc_desc_array(
      cmdbuf, PANVK_VARY_BUF_MAX + 1, ATTRIBUTE_BUFFER);
   if (!bufs.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct mali_attribute_buffer_packed *buf_descs = bufs.cpu;
   const struct vk_input_assembly_state *ia =
      &cmdbuf->vk.dynamic_graphics_state.ia;
   bool writes_point_size =
      vs->info.vs.writes_point_size &&
      ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   uint64_t psiz_buf = 0;

   if (is_indirect_draw(draw) &&
       !cmdbuf->state.gfx.vs.indirect_varying_bufs_infos) {
      struct pan_ptr bufs_info_storage = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, sizeof(struct libpan_draw_helper_varying_buf_info), 8);

      if (!bufs_info_storage.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      cmdbuf->state.gfx.vs.indirect_varying_bufs_infos = bufs_info_storage.gpu;

      struct libpan_draw_helper_varying_buf_info *vary_bufs_info =
         bufs_info_storage.cpu;
      vary_bufs_info->address = dev->indirect_varying_buffer->addr.dev;
      vary_bufs_info->size = PANVK_JM_MAX_PER_VTX_ATTRIBUTES_INDIRECT_SIZE *
                             PANVK_JM_MAX_VERTICES_INDIRECT;
      vary_bufs_info->offset = 0;
   }

   uint64_t buf_addrs_general = 0;
   for (unsigned i = 0; i < PANVK_VARY_BUF_MAX; i++) {
      uint32_t buf_size;
      uint64_t buf_addr;
      if (is_indirect_draw(draw)) {
         buf_addr = dev->indirect_varying_buffer->addr.dev;
         buf_size = 0;
      } else {
         buf_size = draw->padded_vertex_count * draw->info.instance.count *
                    link->buf_strides[i];
         struct pan_ptr vb =
            buf_size ? panvk_cmd_alloc_dev_mem(cmdbuf, varying, buf_size, 64)
                     : (struct pan_ptr){0};
         buf_addr = vb.gpu;

         if (buf_size && !buf_addr)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;

         if (i == PANVK_VARY_BUF_POSITION)
            panvk_gs_dbg_in_pos_cpu = vb.cpu;
      }

      pan_pack(&buf_descs[i], ATTRIBUTE_BUFFER, cfg) {
         cfg.stride = link->buf_strides[i];
         cfg.size = buf_size;
         cfg.pointer = buf_addr;
      }

      if (i == PANVK_VARY_BUF_POSITION)
         draw->position = buf_addr;

      if (i == PANVK_VARY_BUF_PSIZ)
         psiz_buf = buf_addr;

      if (i == PANVK_VARY_BUF_GENERAL)
         buf_addrs_general = buf_addr;
   }

   /* We need an empty entry to stop prefetching on Bifrost */
   memset(bufs.cpu + (pan_size(ATTRIBUTE_BUFFER) * PANVK_VARY_BUF_MAX), 0,
          pan_size(ATTRIBUTE_BUFFER));

   if (writes_point_size)
      draw->psiz = psiz_buf;
   else if (ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
            ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP ||
            (cmdbuf->state.gfx.gs.shader &&
             cmdbuf->state.gfx.gs.shader != cmdbuf->state.gfx.gs.app_shader &&
             cmdbuf->vk.dynamic_graphics_state.rs.polygon_mode == VK_POLYGON_MODE_LINE))
      draw->line_width = cmdbuf->vk.dynamic_graphics_state.rs.line.width;
   else
      draw->line_width = 1.0f;

   draw->varying_bufs = bufs.gpu;

   /* --- geometry shader emulation: the second set of varying buffers ----------------------
    *
    * The vertex job writes the buffers built above. With a geometry shader bound, the tiler must
    * instead read what the emulation compute job produces, so allocate a parallel set here, sized
    * for the worst case (every input primitive emits its full declared maximum) because the real
    * count is not known until the shader runs and this driver cannot have a compute job patch the
    * tiler descriptor. See panvk_nir_lower_gs.c.
    */
   const struct panvk_shader *gs_shader = cmdbuf->state.gfx.gs.shader;
   if (gs_shader) {
      const struct panvk_shader_variant *gs = panvk_shader_only_variant(gs_shader);
      /* Every (input primitive, gl_InvocationID) pair over every instance is one invocation of
       * the emulation, with its own worst-case slice of the output. An indirect draw's count is
       * only on the GPU, so it gets the cap. */
      const uint32_t in_prims =
         (cmdbuf->state.gfx.tess.tes ? tess_index_cap(cmdbuf, draw) /
                                          MAX2(gs->gs.input_verts_per_prim, 1)
          : is_indirect_draw(draw)   ? PANVK_GS_INDIRECT_MAX_PRIMS
                                     : gs_prims_per_instance(draw, gs) * draw->info.instance.count) *
         gs->gs.invocations;
      const uint32_t out_verts = in_prims * gs->gs.max_output_verts;
      const uint32_t out_indices =
         in_prims * gs->gs.max_output_prims * gs->gs.output_verts_per_prim;

      struct pan_ptr gs_bufs = panvk_cmd_alloc_desc_array(
         cmdbuf, PANVK_VARY_BUF_MAX + 1, ATTRIBUTE_BUFFER);
      if (!gs_bufs.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      struct mali_attribute_buffer_packed *gs_descs = gs_bufs.cpu;
      uint64_t gs_addrs[PANVK_VARY_BUF_MAX] = {0};

      void *gs_pos_cpu = NULL;
      for (unsigned i = 0; i < PANVK_VARY_BUF_MAX; i++) {
         /* The generic buffer is the emulation's own layout (link->gs_out_stride). */
         const uint32_t buf_stride =
            i == PANVK_VARY_BUF_GENERAL ? link->gs_out_stride : link->buf_strides[i];
         const uint32_t sz = out_verts * buf_stride;
         /* NOT the varying pool. The vertex shader reaches that through the varying unit,
          * and a plain compute store into it does not land -- measured: the same store to the
          * descriptor pool appeared and this one did not. The tiler only needs a pointer and a
          * stride, so ordinary GPU memory serves. */
         struct pan_ptr vb =
            sz ? panvk_cmd_alloc_dev_mem(cmdbuf, desc, sz, 64)
               : (struct pan_ptr){0};
         gs_addrs[i] = vb.gpu;
         if (sz && !gs_addrs[i])
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
         if (i == PANVK_VARY_BUF_POSITION)
            gs_pos_cpu = vb.cpu;

         pan_pack(&gs_descs[i], ATTRIBUTE_BUFFER, cfg) {
            cfg.stride = buf_stride;
            cfg.size = sz;
            cfg.pointer = gs_addrs[i];
         }
      }
      memset(gs_bufs.cpu + (pan_size(ATTRIBUTE_BUFFER) * PANVK_VARY_BUF_MAX), 0,
             pan_size(ATTRIBUTE_BUFFER));

      struct pan_ptr idx =
         panvk_cmd_alloc_dev_mem(cmdbuf, desc, MAX2(out_indices, 1) * 4, 4);
      if (!idx.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      draw->gs_varying_bufs = gs_bufs.gpu;
      cmdbuf->state.gfx.gs.out_pos = gs_addrs[PANVK_VARY_BUF_POSITION];
      cmdbuf->state.gfx.gs.out_general = gs_addrs[PANVK_VARY_BUF_GENERAL];
      cmdbuf->state.gfx.gs.out_index = idx.gpu;
      cmdbuf->state.gfx.gs.in_general = buf_addrs_general;
      cmdbuf->state.gfx.gs.in_general_stride =
         link->buf_strides[PANVK_VARY_BUF_GENERAL];
      cmdbuf->state.gfx.gs.out_general_stride = link->gs_out_stride;
      cmdbuf->state.gfx.gs.out_vertex_count = out_verts;
      cmdbuf->state.gfx.gs.out_index_count = out_indices;
      /* The emulation writes an index buffer describing INDEPENDENT primitives, so whatever
       * strip the shader declared, the tiler draws a list. */
      cmdbuf->state.gfx.gs.out_prim =
         gs->gs.output_verts_per_prim == 3   ? MESA_PRIM_TRIANGLES
         : gs->gs.output_verts_per_prim == 2 ? MESA_PRIM_LINES
                                             : MESA_PRIM_POINTS;
      /* The tess eval job writes one vertex per tessellator index, in index order, so the
       * identity index buffer it leaves draws the tessellator's own primitives. */
      if (gs_shader == cmdbuf->state.gfx.tess.tes) {
         enum tess_primitive_mode prim;
         enum gl_tess_spacing spacing;
         bool ccw, points;
         tess_mode(cmdbuf, &prim, &spacing, &ccw, &points);
         cmdbuf->state.gfx.gs.out_prim = points ? MESA_PRIM_POINTS
                                         : prim == TESS_PRIMITIVE_ISOLINES ? MESA_PRIM_LINES
                                                                          : MESA_PRIM_TRIANGLES;
      }

      /* The slot -> byte offset tables the lowered shader reads: the vertex shader's generic
       * layout on the way in, the emulation's own on the way out. */
      struct pan_ptr offs =
         panvk_cmd_alloc_dev_mem(cmdbuf, desc, 2 * VARYING_SLOT_MAX * 4, 4);
      if (!offs.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      uint32_t *offs_cpu = offs.cpu;
      memcpy(offs_cpu, link->buf_offsets, VARYING_SLOT_MAX * 4);
      memcpy(offs_cpu + VARYING_SLOT_MAX, link->gs_out_offsets, VARYING_SLOT_MAX * 4);
      cmdbuf->state.gfx.gs.in_offsets = offs.gpu;
      cmdbuf->state.gfx.gs.out_offsets = offs.gpu + VARYING_SLOT_MAX * 4;

      /* Transform feedback: a primitive count per emulation invocation and a capture record per
       * output vertex slot. Captured once per draw, on its first layer. */
      cmdbuf->state.gfx.gs.xfb_counts = 0;
      cmdbuf->state.gfx.gs.xfb_staging = 0;
      cmdbuf->state.gfx.gs.xfb_invocations = 0;
      if (cmdbuf->state.gfx.xfb.active && gs->gs.xfb_dwords && draw->info.layer_id == 0 &&
          in_prims) {
         struct pan_ptr counts = panvk_cmd_alloc_dev_mem(cmdbuf, desc, in_prims * 4, 4);
         struct pan_ptr staging = panvk_cmd_alloc_dev_mem(
            cmdbuf, desc, MAX2(out_verts, 1) * gs->gs.xfb_dwords * 4, 4);
         if (!counts.gpu || !staging.gpu)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
         cmdbuf->state.gfx.gs.xfb_counts = counts.gpu;
         cmdbuf->state.gfx.gs.xfb_staging = staging.gpu;
         cmdbuf->state.gfx.gs.xfb_invocations = in_prims;
      }

      /* Bisect switches, read once so they cannot cost anything on the default path. */
      static int cpufill_mode = -1;
      static int in_from_cpu = -1;
      if (cpufill_mode < 0) {
         const char *e = getenv("PANVK_GS_CPUFILL");
         cpufill_mode = (e && e[0] >= '1' && e[0] <= '3') ? e[0] - '0' : 0;
         in_from_cpu = getenv("PANVK_GS_INFROMCPU") != NULL;
      }

      cmdbuf->state.gfx.gs.in_pos_override = 0;
      if (in_from_cpu) {
         struct pan_ptr ip = panvk_cmd_alloc_dev_mem(cmdbuf, desc, 48, 64);
         if (ip.gpu) {
            /* CLIP space: the emulation's input is (vs_keep_clip_space). On the probe's 64x64
             * target this is (4,60) (60,60) (32,4) in pixels. */
            static const float tri[3][4] = {
               {-0.875f, 0.875f, 0.5f, 1.0f},
               {0.875f, 0.875f, 0.5f, 1.0f},
               {0.0f, -0.875f, 0.5f, 1.0f},
            };
            memcpy(ip.cpu, tri, sizeof(tri));
            cmdbuf->state.gfx.gs.in_pos_override = ip.gpu;
         }
      }

      panvk_gs_dbg_pos_cpu = gs_pos_cpu;
      panvk_gs_dbg_idx_cpu = idx.cpu;
      panvk_gs_dbg_idx_count = out_indices;
      panvk_gs_dbg_pos_gpu = gs_addrs[PANVK_VARY_BUF_POSITION];
      panvk_gs_dbg_pos_verts = out_verts;

      /* -- BISECT: PANVK_GS_CPUFILL --------------------------------------------------------
       * Every attempt so far judged the emulation job's position stores from a rendered image,
       * which cannot tell "the store did not land" apart from "the tiler was told something
       * wrong". A CPU write demonstrably lands. So author the tiler's whole input here, skip
       * the emulation job entirely, and turn culling off: if THIS triangle does not appear,
       * the shader stores are innocent and the fault is in the tiler descriptors. */
      cmdbuf->state.gfx.gs.cpufill = false;
      cmdbuf->state.gfx.gs.posonly = false;
      cmdbuf->state.gfx.gs.keepjob = false;
      if (gs_pos_cpu && cpufill_mode && out_verts >= 3) {
         /* SCREEN SPACE, not NDC: nir_lower_viewport_transform (bifrost_nir.c) rewrites the
          * vertex shader's gl_Position into screen-space xyz with w = 1/w, so that is what the
          * tiler expects to find here. The probe's attachment is 64x64. */
         static const float tri[3][4] = {
            {4.0f, 60.0f, 0.5f, 1.0f},
            {60.0f, 60.0f, 0.5f, 1.0f},
            {32.0f, 4.0f, 0.5f, 1.0f},
         };
         memcpy(gs_pos_cpu, tri, sizeof(tri));
         uint32_t *idx_cpu = idx.cpu;
         for (unsigned i = 0; i < out_indices; i++)
            idx_cpu[i] = i < 3 ? i : 0;
         cmdbuf->state.gfx.gs.out_index_count = 3;
         cmdbuf->state.gfx.gs.cpufill = true;
         cmdbuf->state.gfx.gs.posonly = cpufill_mode == 2;
         cmdbuf->state.gfx.gs.keepjob = cpufill_mode == 3;
         fprintf(stderr,
                 "[GSFILL] pos=%p/0x%" PRIx64 " idx=0x%" PRIx64
                 " out_verts=%u pos_stride=%u\n",
                 gs_pos_cpu, gs_addrs[PANVK_VARY_BUF_POSITION], idx.gpu,
                 out_verts, link->buf_strides[PANVK_VARY_BUF_POSITION]);
      }
   } else {
      draw->gs_varying_bufs = 0;
   }

   draw->indirect_info.varying_bufs =
      cmdbuf->state.gfx.vs.indirect_varying_bufs_infos;
   draw->vs.varyings = panvk_priv_mem_dev_addr(link->vs.attribs);
   draw->fs.varyings = panvk_priv_mem_dev_addr(link->fs.attribs);
   return VK_SUCCESS;
}

static void
panvk_draw_emit_attrib_buf(
   const struct panvk_draw_data *draw,
   const struct vk_vertex_binding_state *buf_info, uint32_t stride,
   const struct panvk_attrib_buf *buf,
   struct mali_attribute_buffer_packed *desc,
   struct libpan_draw_helper_attrib_buf_info *helper_buf_info)
{
   uint64_t addr = buf->address & ~63ULL;
   unsigned size = buf->size + (buf->address & 63);
   unsigned divisor = draw->padded_vertex_count * buf_info->divisor;
   bool per_instance = buf_info->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   struct mali_attribute_buffer_packed *buf_ext = &desc[1];

   /* In case of indirect draw, the descriptor will be patched at runtime */
   if (helper_buf_info != NULL) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.pointer = addr;
         cfg.size = size;
      }

      helper_buf_info->divisor = buf_info->divisor;
      helper_buf_info->stride = stride;
      helper_buf_info->per_instance = per_instance;
   } else if (draw->info.instance.count <= 1) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = per_instance ? 0 : stride;
         cfg.pointer = addr;
         cfg.size = size;
      }
   } else if (!per_instance) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_MODULUS;
         cfg.divisor = draw->padded_vertex_count;
         cfg.stride = stride;
         cfg.pointer = addr;
         cfg.size = size;
      }
   } else if (!divisor) {
      /* instance_divisor == 0 means all instances share the same value.
       * Make it a 1D array with a zero stride.
       */
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = 0;
         cfg.pointer = addr;
         cfg.size = size;
      }
   } else if (util_is_power_of_two_or_zero(divisor)) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.stride = stride;
         cfg.pointer = addr;
         cfg.size = size;
         cfg.divisor_r = __builtin_ctz(divisor);
      }
   } else {
      unsigned divisor_r = 0, divisor_e = 0;
      unsigned divisor_d =
         pan_compute_npot_divisor(divisor, &divisor_r, &divisor_e);
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_NPOT_DIVISOR;
         cfg.stride = stride;
         cfg.pointer = addr;
         cfg.size = size;
         cfg.divisor_r = divisor_r;
         cfg.divisor_e = divisor_e;
      }

      pan_cast_and_pack(buf_ext, ATTRIBUTE_BUFFER_CONTINUATION_NPOT, cfg) {
         cfg.divisor_numerator = divisor_d;
         cfg.divisor = buf_info->divisor;
      }

      buf_ext = NULL;
   }

   /* If the buffer extension wasn't used, memset(0) */
   if (buf_ext)
      memset(buf_ext, 0, pan_size(ATTRIBUTE_BUFFER));
}

static void
panvk_draw_emit_attrib(const struct panvk_draw_data *draw,
                       const struct vk_vertex_attribute_state *attrib_info,
                       const struct vk_vertex_binding_state *buf_info,
                       const struct panvk_attrib_buf *buf,
                       struct mali_attribute_packed *desc,
                       struct libpan_draw_helper_attrib_info *helper_attrib_info)
{
   bool per_instance = buf_info->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   enum pipe_format f = vk_format_to_pipe_format(attrib_info->format);
   unsigned buf_idx = attrib_info->binding;

   pan_pack(desc, ATTRIBUTE, cfg) {
      cfg.buffer_index = buf_idx * 2;
      cfg.offset_enable = true;
      cfg.format = GENX(pan_format_from_pipe_format)(f)->hw;

      uint32_t offset = attrib_info->offset + (buf->address & 63);

      /* In case of indirect draw, the descriptor will be patched at runtime */
      if (helper_attrib_info != NULL) {
         helper_attrib_info->base_offset = offset;
         helper_attrib_info->stride = per_instance ? buf_info->stride : 0;
      } else {
         cfg.offset = offset;
         if (per_instance)
            cfg.offset += draw->info.instance.base * buf_info->stride;
      }
   }
}

static VkResult
panvk_draw_prepare_vs_attribs(struct panvk_cmd_buffer *cmdbuf,
                              struct panvk_draw_data *draw)
{
   const struct panvk_shader *vs = cmdbuf->state.gfx.vs.shader;
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;
   unsigned num_imgs = vs->desc_info.others.count[PANVK_BIFROST_DESC_TABLE_IMG];
   unsigned num_vs_attribs = util_last_bit(vi->attributes_valid);
   unsigned num_vbs = util_last_bit(vi->bindings_valid);
   unsigned attrib_count =
      num_imgs ? MAX_VS_ATTRIBS + num_imgs : num_vs_attribs;
   bool dirty =
      dyn_gfx_state_dirty(cmdbuf, VI) ||
      dyn_gfx_state_dirty(cmdbuf, VI_BINDINGS_VALID) ||
      dyn_gfx_state_dirty(cmdbuf, VI_BINDING_STRIDES) ||
      gfx_state_dirty(cmdbuf, VB) || gfx_state_dirty(cmdbuf, DESC_STATE) ||
      is_indirect_draw(draw) != cmdbuf->state.gfx.vs.previous_draw_was_indirect ||
      /* An indirect draw's helper PATCHES these descriptors on the GPU, so they cannot be
       * shared with the next draw: measured, two indirect draws in a row both fetched the
       * second one's vertices (multiDrawIndirect drew RIGHT twice on Mali-G76, nothing on
       * Mali-G72). */
      is_indirect_draw(draw);

   if (!dirty)
      return VK_SUCCESS;

   unsigned attrib_buf_count = (num_vbs + num_imgs) * 2;
   struct pan_ptr bufs = panvk_cmd_alloc_desc_array(
      cmdbuf, attrib_buf_count + 1, ATTRIBUTE_BUFFER);
   struct mali_attribute_buffer_packed *attrib_buf_descs = bufs.cpu;
   struct pan_ptr attribs =
      panvk_cmd_alloc_desc_array(cmdbuf, attrib_count, ATTRIBUTE);
   struct mali_attribute_packed *attrib_descs = attribs.cpu;

   if (!bufs.gpu || (attrib_count && !attribs.gpu))
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct libpan_draw_helper_attrib_buf_info *bufs_infos = NULL;
   struct libpan_draw_helper_attrib_info *attribs_infos = NULL;

   if (is_indirect_draw(draw)) {
      struct pan_ptr bufs_infos_storage = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc,
         num_vbs * sizeof(struct libpan_draw_helper_attrib_buf_info), 8);
      struct pan_ptr attribs_infos_storage = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc,
         num_vs_attribs * sizeof(struct libpan_draw_helper_attrib_info), 8);

      if (!bufs_infos_storage.gpu ||
          (num_vs_attribs && !attribs_infos_storage.gpu))
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      cmdbuf->state.gfx.vs.indirect_attrib_bufs_infos = bufs_infos_storage.gpu;
      cmdbuf->state.gfx.vs.indirect_attribs_infos = attribs_infos_storage.gpu;
      bufs_infos = bufs_infos_storage.cpu;
      attribs_infos = attribs_infos_storage.cpu;
   }

   for (unsigned i = 0; i < num_vbs; i++) {
      if (vi->bindings_valid & BITFIELD_BIT(i)) {
         struct libpan_draw_helper_attrib_buf_info *helper_buf_info =
            bufs_infos ? &bufs_infos[i] : NULL;
         panvk_draw_emit_attrib_buf(draw, &vi->bindings[i],
                                    dyns->vi_binding_strides[i],
                                    &cmdbuf->state.gfx.vb.bufs[i],
                                    &attrib_buf_descs[i * 2], helper_buf_info);
      } else {
         memset(&attrib_buf_descs[i * 2], 0, sizeof(*attrib_buf_descs) * 2);
      }
   }

   for (unsigned i = 0; i < num_vs_attribs; i++) {
      if (vi->attributes_valid & BITFIELD_BIT(i)) {
         unsigned buf_idx = vi->attributes[i].binding;
         struct libpan_draw_helper_attrib_info *helper_attrib_info =
            attribs_infos ? &attribs_infos[i] : NULL;
         panvk_draw_emit_attrib(draw, &vi->attributes[i],
                                &vi->bindings[buf_idx],
                                &cmdbuf->state.gfx.vb.bufs[buf_idx],
                                &attrib_descs[i], helper_attrib_info);
      } else {
         memset(&attrib_descs[i], 0, sizeof(attrib_descs[0]));
      }
   }

   /* A NULL entry is needed to stop prefecting on Bifrost */
   memset(bufs.cpu + (pan_size(ATTRIBUTE_BUFFER) * attrib_buf_count), 0,
          pan_size(ATTRIBUTE_BUFFER));

   cmdbuf->state.gfx.vs.attrib_bufs = bufs.gpu;
   cmdbuf->state.gfx.vs.attribs = attribs.gpu;

   if (num_imgs) {
      cmdbuf->state.gfx.vs.desc.img_attrib_table =
         attribs.gpu + (MAX_VS_ATTRIBS * pan_size(ATTRIBUTE));
      cmdbuf->state.gfx.vs.desc.tables[PANVK_BIFROST_DESC_TABLE_IMG] =
         bufs.gpu + (num_vbs * pan_size(ATTRIBUTE_BUFFER) * 2);
      cmdbuf->state.gfx.vs.desc.img_attrib_table_cpu =
         (uint8_t *)attribs.cpu + (MAX_VS_ATTRIBS * pan_size(ATTRIBUTE));
      cmdbuf->state.gfx.vs.desc.tables_cpu[PANVK_BIFROST_DESC_TABLE_IMG] =
         (uint8_t *)bufs.cpu + (num_vbs * pan_size(ATTRIBUTE_BUFFER) * 2);
      /* Fresh image tables: the vertex shader's image descriptors have to be copied into them,
       * even when the descriptor state itself did not change. */
      draw->vs_img_tables_new = true;
   }

   return VK_SUCCESS;
}

static void
panvk_draw_prepare_attributes(struct panvk_cmd_buffer *cmdbuf,
                              struct panvk_draw_data *draw)
{
   panvk_draw_prepare_vs_attribs(cmdbuf, draw);
   draw->vs.attributes = cmdbuf->state.gfx.vs.attribs;
   draw->vs.attribute_bufs = cmdbuf->state.gfx.vs.attrib_bufs;
   draw->indirect_info.attribs = cmdbuf->state.gfx.vs.indirect_attribs_infos;
   draw->indirect_info.attrib_bufs =
      cmdbuf->state.gfx.vs.indirect_attrib_bufs_infos;
}

static void
panvk_emit_viewport(struct panvk_cmd_buffer *cmdbuf,
                    struct mali_viewport_packed *vpd)
{
   const struct vk_viewport_state *vp = &cmdbuf->vk.dynamic_graphics_state.vp;

   if (vp->viewport_count < 1)
      return;

   float minz, maxz;
   panvk_depth_range(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state.vp,
                     &minz, &maxz);

   /* A geometry shader writing gl_ViewportIndex can put a primitive in any viewport: the tiler's
    * one scissor is the union of them all, and the fragment shader keeps each primitive inside
    * its own (panvk_nir_lower_viewport_scissor_fs). */
   const bool multi = cmdbuf->state.gfx.gs.shader &&
                      panvk_shader_only_variant(cmdbuf->state.gfx.gs.shader)->gs.writes_viewport;
   const unsigned count = multi ? vp->viewport_count : 1;

   int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;
   for (unsigned i = 0; i < count; i++) {
      const VkViewport *viewport = &vp->viewports[i];
      const VkRect2D *scissor = &vp->scissors[i];

      /* The spec says "width must be greater than 0.0" */
      assert(viewport->width >= 0);
      int vminx = (int)viewport->x;
      int vmaxx = (int)(viewport->x + viewport->width);

      /* Viewport height can be negative */
      int vminy = MIN2((int)viewport->y, (int)(viewport->y + viewport->height));
      int vmaxy = MAX2((int)viewport->y, (int)(viewport->y + viewport->height));

      assert(scissor->offset.x >= 0 && scissor->offset.y >= 0);
      vminx = MAX2(scissor->offset.x, vminx);
      vminy = MAX2(scissor->offset.y, vminy);
      vmaxx = MIN2(scissor->offset.x + scissor->extent.width, vmaxx);
      vmaxy = MIN2(scissor->offset.y + scissor->extent.height, vmaxy);

      minx = MIN2(minx, vminx);
      miny = MIN2(miny, vminy);
      maxx = MAX2(maxx, vmaxx);
      maxy = MAX2(maxy, vmaxy);
   }

   /* Make sure we don't end up with a max < min when width/height is 0 */
   maxx = maxx > minx ? maxx - 1 : maxx;
   maxy = maxy > miny ? maxy - 1 : maxy;

   /* Clamp viewport scissor to valid range */
   minx = CLAMP(minx, 0, UINT16_MAX);
   maxx = CLAMP(maxx, 0, UINT16_MAX);
   miny = CLAMP(miny, 0, UINT16_MAX);
   maxy = CLAMP(maxy, 0, UINT16_MAX);

   pan_pack(vpd, VIEWPORT, cfg) {
      cfg.scissor_minimum_x = minx;
      cfg.scissor_minimum_y = miny;
      cfg.scissor_maximum_x = maxx;
      cfg.scissor_maximum_y = maxy;
      cfg.minimum_z = minz;
      cfg.maximum_z = maxz;
   }
}

static VkResult
panvk_draw_prepare_viewport(struct panvk_cmd_buffer *cmdbuf,
                            struct panvk_draw_data *draw)
{
   /* When rasterizerDiscardEnable is active, it is allowed to have viewport and
    * scissor disabled.
    * As a result, we define an empty one.
    */
   if (!cmdbuf->state.gfx.vpd || dyn_gfx_state_dirty(cmdbuf, VP_VIEWPORTS) ||
       gfx_state_dirty(cmdbuf, GS) ||
       dyn_gfx_state_dirty(cmdbuf, VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
       dyn_gfx_state_dirty(cmdbuf, VP_SCISSORS) ||
       dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLIP_ENABLE) ||
       dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLAMP_ENABLE) ||
       dyn_gfx_state_dirty(cmdbuf, VP_DEPTH_CLAMP_RANGE)) {
      struct pan_ptr vp = panvk_cmd_alloc_desc(cmdbuf, VIEWPORT);
      if (!vp.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      panvk_emit_viewport(cmdbuf, vp.cpu);
      cmdbuf->state.gfx.vpd = vp.gpu;
   }

   draw->viewport = cmdbuf->state.gfx.vpd;
   return VK_SUCCESS;
}

static void
panvk_emit_vertex_dcd(struct panvk_cmd_buffer *cmdbuf,
                      const struct panvk_draw_data *draw,
                      struct mali_draw_packed *dcd)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_desc_state *vs_desc_state =
      &cmdbuf->state.gfx.vs.desc;

   pan_pack(dcd, DRAW, cfg) {
      cfg.state = panvk_priv_mem_dev_addr(vs->rsd);
      cfg.attributes = draw->vs.attributes;
      cfg.attribute_buffers = draw->vs.attribute_bufs;
      cfg.varyings = draw->vs.varyings;
      cfg.varying_buffers = draw->varying_bufs;
      cfg.thread_storage = draw->tls;

      /* In case of indirect draw, the descriptor will be patched at runtime */
      if (!is_indirect_draw(draw)) {
         cfg.offset_start = draw->info.vertex.raw_offset;
         cfg.instance_size =
            draw->info.instance.count > 1 ? draw->padded_vertex_count : 1;
      }

      cfg.uniform_buffers = vs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_UBO];
      cfg.push_uniforms = cmdbuf->state.gfx.vs.push_uniforms;
      cfg.textures = vs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_TEXTURE];
      cfg.samplers = vs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_SAMPLER];
   }
}

static VkResult
panvk_draw_prepare_vertex_job(struct panvk_cmd_buffer *cmdbuf,
                              struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   util_dynarray_append(&batch->jobs, ptr.cpu);
   draw->jobs.vertex = ptr;

   memcpy(pan_section_ptr(ptr.cpu, COMPUTE_JOB, INVOCATION), &draw->invocation,
          pan_size(INVOCATION));

   pan_section_pack(ptr.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = 5;
   }

   panvk_emit_vertex_dcd(cmdbuf, draw,
                         pan_section_ptr(ptr.cpu, COMPUTE_JOB, DRAW));
   return VK_SUCCESS;
}

static enum mali_draw_mode
translate_prim(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return MALI_DRAW_MODE_POINTS;
   case MESA_PRIM_LINES:
      return MALI_DRAW_MODE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return MALI_DRAW_MODE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return MALI_DRAW_MODE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return MALI_DRAW_MODE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return MALI_DRAW_MODE_TRIANGLE_FAN;
#if PAN_ARCH >= 9
   case MESA_PRIM_LINES_ADJACENCY:
      return MALI_DRAW_MODE_LINES_ADJACENCY;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      return MALI_DRAW_MODE_LINE_STRIP_ADJACENCY;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      return MALI_DRAW_MODE_TRIANGLES_ADJACENCY;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      return MALI_DRAW_MODE_TRIANGLE_STRIP_ADJACENCY;
#endif
   default:
      UNREACHABLE("Invalid primitive type");
   }
}

static void
panvk_emit_tiler_primitive(struct panvk_cmd_buffer *cmdbuf,
                           const struct panvk_draw_data *draw,
                           struct mali_primitive_packed *prim)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_input_assembly_state *ia = &dyns->ia;
   const struct vk_rasterization_state *rs = &dyns->rs;
   bool writes_point_size =
      vs->info.vs.writes_point_size &&
      ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   bool secondary_shader = vs->info.vs.secondary_enable && fs != NULL;
   assert(!(vs->info.outputs_written & VARYING_BIT_PRIMITIVE_ID));
   bool fs_reads_primitive_id = fs ? fs->info.fs.reads_primitive_id : false;

   pan_pack(prim, PRIMITIVE, cfg) {
      cfg.draw_mode = translate_prim(draw->gs_varying_bufs &&
                                        !cmdbuf->state.gfx.gs.posonly
                                        ? cmdbuf->state.gfx.gs.out_prim
                                        : draw->info.prim);
      if (writes_point_size)
         cfg.point_size_array_format = MALI_POINT_SIZE_ARRAY_FORMAT_FP16;
      cfg.primitive_index_enable = fs_reads_primitive_id;
      cfg.primitive_index_writeback = fs_reads_primitive_id;

      cfg.first_provoking_vertex =
         cmdbuf->state.gfx.render.first_provoking_vertex != U_TRISTATE_NO;

      if (draw->info.index.restart_enable)
         cfg.primitive_restart = MALI_PRIMITIVE_RESTART_IMPLICIT;
      cfg.job_task_split = 6;

      if (draw->info.index.index_size) {
         switch (draw->info.index.index_size) {
         case 4:
            cfg.index_type = MALI_INDEX_TYPE_UINT32;
            break;
         case 2:
            cfg.index_type = MALI_INDEX_TYPE_UINT16;
            break;
         case 1:
            cfg.index_type = MALI_INDEX_TYPE_UINT8;
            break;
         default:
            UNREACHABLE("Invalid index size");
         }
      }

      /* In case of indirect draw, the descriptor will be patched at runtime */
      cfg.index_count = is_indirect_draw(draw) ? 1 : draw->info.vertex.count;

      /* The index buffer address and the base vertex offset were previously written ONLY by
       * panlib_patch_draw_tiler_dcd, because every indexed draw was classified as indirect. A
       * direct indexed draw has to carry them itself, with the same arithmetic the helper uses:
       * indices are biased by the first index, and base_vertex_offset cancels the offset_start
       * that the vertex job already applied, so the tiler resolves index i to the varying slot
       * the vertex shader actually wrote. */
      /* With a geometry shader the tiler does not draw the application's primitives at all: it
       * draws what the emulation job produced, as an indexed list over the index buffer that job
       * filled in. The counts are the worst-case ones fixed at record time. */
      if (draw->gs_varying_bufs && !cmdbuf->state.gfx.gs.posonly) {
         cfg.index_type = MALI_INDEX_TYPE_UINT32;
         cfg.index_count = cmdbuf->state.gfx.gs.out_index_count;
         cfg.indices = cmdbuf->state.gfx.gs.out_index;
         cfg.base_vertex_offset = 0;
      } else if (!is_indirect_draw(draw) && draw->info.index.index_size) {
         cfg.indices = draw->info.index.buffer_dev_addr +
                       (uint64_t)draw->info.index.offset *
                          draw->info.index.index_size;
         cfg.base_vertex_offset =
            (int64_t)draw->info.vertex.base - draw->info.vertex.raw_offset;
      }

      cfg.low_depth_cull = cfg.high_depth_cull =
         vk_rasterization_state_depth_clip_enable(rs);

      cfg.secondary_shader = secondary_shader;
   }
}

static void
panvk_emit_tiler_primitive_size(struct panvk_cmd_buffer *cmdbuf,
                                const struct panvk_draw_data *draw,
                                struct mali_primitive_size_packed *primsz)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const bool writes_point_size =
      vs->info.vs.writes_point_size && draw->info.prim == MESA_PRIM_POINTS;

   pan_pack(primsz, PRIMITIVE_SIZE, cfg) {
      if (writes_point_size) {
         cfg.size_array = draw->psiz;
      } else {
         cfg.fixed_sized = draw->line_width;
      }
   }
}

static void
panvk_emit_tiler_dcd(struct panvk_cmd_buffer *cmdbuf,
                     const struct panvk_draw_data *draw,
                     struct mali_draw_packed *dcd)
{
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;

   /* What the tiler draws: a geometry shader's output replaces the application's primitives. */
   enum mesa_prim reduced_prim = u_reduced_prim(
      draw->gs_varying_bufs && !cmdbuf->state.gfx.gs.posonly ? cmdbuf->state.gfx.gs.out_prim
                                                             : draw->info.prim);
   const bool non_polygon = reduced_prim != MESA_PRIM_TRIANGLES;

   pan_pack(dcd, DRAW, cfg) {
      cfg.front_face_ccw = rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
      cfg.cull_front_face =
         !non_polygon && !cmdbuf->state.gfx.gs.cpufill &&
         (rs->cull_mode & VK_CULL_MODE_FRONT_BIT) != 0;
      cfg.cull_back_face =
         !non_polygon && !cmdbuf->state.gfx.gs.cpufill &&
         (rs->cull_mode & VK_CULL_MODE_BACK_BIT) != 0;

      /* The tiler reads positions from what the geometry job produced, not from what the vertex
       * job wrote. */
      cfg.position = draw->gs_varying_bufs ? cmdbuf->state.gfx.gs.out_pos
                                           : draw->position;
      cfg.state = draw->fs.rsd;
      cfg.attributes = fs_desc_state->img_attrib_table;
      cfg.attribute_buffers =
         fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_IMG];
      cfg.viewport = draw->viewport;
      cfg.varyings = draw->fs.varyings;
      cfg.varying_buffers =
         cfg.varyings ? (draw->gs_varying_bufs ?: draw->varying_bufs) : 0;
      cfg.thread_storage = draw->tls;

      /* For all primitives but lines DRAW.flat_shading_vertex must
       * be set to 0 and the provoking vertex is selected with the
       * PRIMITIVE.first_provoking_vertex field.
       */
      if (reduced_prim == MESA_PRIM_LINES)
         cfg.flat_shading_vertex = true;

      /* In case of indirect draw, the descriptor will be patched at runtime */
      if (draw->gs_varying_bufs && !cmdbuf->state.gfx.gs.posonly) {
         cfg.offset_start = 0;
         cfg.instance_size = 1;
         cfg.instance_primitive_size = 0;
      } else if (!is_indirect_draw(draw)) {
         cfg.offset_start = draw->info.vertex.raw_offset;
         cfg.instance_size =
            draw->info.instance.count > 1 ? draw->padded_vertex_count : 1;
         uint32_t primitives_per_instance =
            DIV_ROUND_UP(draw->padded_vertex_count,
                         mesa_vertices_per_prim(draw->info.prim));
         /* instance_primitive_size has the same restrictions as
          * padded_vertex_count, so we can use pan_padded_vertex_count here. */
         cfg.instance_primitive_size =
            pan_padded_vertex_count(primitives_per_instance);
      }

      cfg.uniform_buffers = fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_UBO];
      cfg.push_uniforms = cmdbuf->state.gfx.fs.push_uniforms;
      cfg.textures = fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_TEXTURE];
      cfg.samplers = fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_SAMPLER];

      cfg.occlusion_query = cmdbuf->state.gfx.occlusion_query.mode;
      cfg.occlusion = cmdbuf->state.gfx.occlusion_query.ptr;
#if PAN_ARCH == 9
      cfg.scissor_to_bounding_box = true;
#endif
   }
}

static void
set_provoking_vertex_mode(struct panvk_cmd_buffer *cmdbuf,
                          enum u_tristate first_provoking_vertex)
{
   struct panvk_cmd_graphics_state *state = &cmdbuf->state.gfx;

   if (first_provoking_vertex != U_TRISTATE_UNSET) {
      /* If this is not the first draw, first_provoking_vertex should match
       * the one from the previous draws. Unfortunately, we can't check it
       * when the render pass is inherited. */
      assert(state->render.first_provoking_vertex == U_TRISTATE_UNSET ||
             state->render.first_provoking_vertex == first_provoking_vertex);
      state->render.first_provoking_vertex = first_provoking_vertex;
   }

   /* Once we emit the first FBDs/TDs, we need to commit to a state. If we
    * choose the wrong one, we will fail the assert when the next application
    * draw happens (with a different state). Use PROVOKING_VERTEX_MODE_FIRST
    * because it's the vulkan default, and so likely to be right more often.
    *
    * TODO: handle this case better */
   if (state->render.first_provoking_vertex == U_TRISTATE_UNSET)
      state->render.first_provoking_vertex = U_TRISTATE_YES;
}

static VkResult
panvk_draw_prepare_tiler_job(struct panvk_cmd_buffer *cmdbuf,
                             struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr;

   if (cmdbuf->state.gfx.fs.required) {
      const struct panvk_shader_desc_info *fs_desc_info =
         &cmdbuf->state.gfx.fs.shader->desc_info;
      struct panvk_shader_desc_state *fs_desc_state =
         &cmdbuf->state.gfx.fs.desc;
      VkResult result = panvk_per_arch(meta_get_copy_desc_job)(
         cmdbuf, fs_desc_info, &cmdbuf->state.gfx.desc_state,
         fs_desc_state, 0, &ptr);
      if (result != VK_SUCCESS)
         return result;
   }

   if (ptr.cpu)
      util_dynarray_append(&batch->jobs, ptr.cpu);

   draw->jobs.frag_copy_desc = ptr;

   ptr = panvk_cmd_alloc_desc(cmdbuf, TILER_JOB);
   util_dynarray_append(&batch->jobs, ptr.cpu);
   draw->jobs.tiler = ptr;

   if (draw->gs_varying_bufs && !cmdbuf->state.gfx.gs.posonly) {
      /* The tiler iterates the geometry shader's output, not the vertex shader's. */
      pan_pack_work_groups_compute(
         pan_section_ptr(ptr.cpu, TILER_JOB, INVOCATION), 1,
         cmdbuf->state.gfx.gs.out_vertex_count, 1, 1, 1, 1, true, false);
   } else {
      memcpy(pan_section_ptr(ptr.cpu, TILER_JOB, INVOCATION), &draw->invocation,
             pan_size(INVOCATION));
   }

   panvk_emit_tiler_primitive(cmdbuf, draw,
                              pan_section_ptr(ptr.cpu, TILER_JOB, PRIMITIVE));

   panvk_emit_tiler_primitive_size(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, TILER_JOB, PRIMITIVE_SIZE));

   panvk_emit_tiler_dcd(cmdbuf, draw,
                        pan_section_ptr(ptr.cpu, TILER_JOB, DRAW));

   pan_section_pack(ptr.cpu, TILER_JOB, TILER, cfg) {
      cfg.address = PAN_ARCH >= 9 ? draw->tiler_ctx->valhall.desc
                                  : draw->tiler_ctx->bifrost.desc;
   }

   pan_section_pack(ptr.cpu, TILER_JOB, PADDING, padding)
      ;

   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_idvs_job(struct panvk_cmd_buffer *cmdbuf,
                            struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr = panvk_cmd_alloc_desc(cmdbuf, INDEXED_VERTEX_JOB);
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   util_dynarray_append(&batch->jobs, ptr.cpu);
   draw->jobs.idvs = ptr;

   memcpy(pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, INVOCATION),
          &draw->invocation, pan_size(INVOCATION));

   panvk_emit_tiler_primitive(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, PRIMITIVE));

   panvk_emit_tiler_primitive_size(
      cmdbuf, draw,
      pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, PRIMITIVE_SIZE));

   pan_section_pack(ptr.cpu, INDEXED_VERTEX_JOB, TILER, cfg) {
      cfg.address = PAN_ARCH >= 9 ? draw->tiler_ctx->valhall.desc
                                  : draw->tiler_ctx->bifrost.desc;
   }

   pan_section_pack(ptr.cpu, INDEXED_VERTEX_JOB, PADDING, _) {
   }

   panvk_emit_tiler_dcd(
      cmdbuf, draw,
      pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, FRAGMENT_DRAW));

   panvk_emit_vertex_dcd(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, VERTEX_DRAW));
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_vs_copy_desc_job(struct panvk_cmd_buffer *cmdbuf,
                                    struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_desc_info *vs_desc_info =
      &cmdbuf->state.gfx.vs.shader->desc_info;
   const struct panvk_shader_desc_state *vs_desc_state =
      &cmdbuf->state.gfx.vs.desc;
   const struct vk_vertex_input_state *vi =
      cmdbuf->vk.dynamic_graphics_state.vi;
   unsigned num_vbs = util_last_bit(vi->bindings_valid);
   struct pan_ptr ptr;

   VkResult result = panvk_per_arch(meta_get_copy_desc_job)(
      cmdbuf, vs_desc_info, &cmdbuf->state.gfx.desc_state, vs_desc_state,
      num_vbs * pan_size(ATTRIBUTE_BUFFER) * 2, &ptr);
   if (result != VK_SUCCESS)
      return result;

   if (ptr.cpu) {
      util_dynarray_append(&batch->jobs, ptr.cpu);
   }

   draw->jobs.vertex_copy_desc = ptr;
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_fs_copy_desc_job(struct panvk_cmd_buffer *cmdbuf,
                                    struct panvk_draw_data *draw)
{
   const struct panvk_shader_desc_info *fs_desc_info =
      &cmdbuf->state.gfx.fs.shader->desc_info;
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr;

   VkResult result = panvk_per_arch(meta_get_copy_desc_job)(
      cmdbuf, fs_desc_info, &cmdbuf->state.gfx.desc_state,
      fs_desc_state, 0, &ptr);
   if (result != VK_SUCCESS)
      return result;

   if (ptr.cpu) {
      util_dynarray_append(&batch->jobs, ptr.cpu);
   }

   draw->jobs.frag_copy_desc = ptr;
   return VK_SUCCESS;
}

static VkResult
panvk_cmd_prepare_draw_link_shaders(struct panvk_cmd_buffer *cmd)
{
   struct panvk_cmd_graphics_state *gfx = &cmd->state.gfx;

   if (!gfx_state_dirty(cmd, VS) && !gfx_state_dirty(cmd, FS) && !gfx_state_dirty(cmd, GS))
      return VK_SUCCESS;

   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmd->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmd));
   const struct panvk_shader_variant *gs =
      cmd->state.gfx.gs.shader ? panvk_shader_only_variant(cmd->state.gfx.gs.shader) : NULL;

   VkResult result =
      panvk_per_arch(link_shaders)(&cmd->desc_pool, vs, gs, fs, &gfx->link);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return result;
   }

   return VK_SUCCESS;
}

static VkResult
prepare_draw(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   struct panvk_shader_desc_state *vs_desc_state = &cmdbuf->state.gfx.vs.desc;
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   struct panvk_descriptor_state *desc_state = &cmdbuf->state.gfx.desc_state;
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;
   VkResult result;
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));

   /* There are only 16 bits in the descriptor for the job ID. Each job has a
    * pilot shader dealing with descriptor copies, and we need one
    * <vertex,tiler> pair per draw.
    */
   if (batch->vtc_jc.job_index + (4 * cmdbuf->state.gfx.render.layer_count) >=
       UINT16_MAX) {
      panvk_per_arch(cmd_close_batch)(cmdbuf);
      batch = panvk_per_arch(cmd_open_batch)(cmdbuf);
   }

   if (fs_user_dirty(cmdbuf) || gfx_state_dirty(cmdbuf, VS) || gfx_state_dirty(cmdbuf, GS)) {
      result = panvk_cmd_prepare_draw_link_shaders(cmdbuf);
      if (result != VK_SUCCESS)
         return result;
   }

   if (cmdbuf->state.gfx.vk_meta) {
      /* vk_meta doesn't care about the provoking vertex mode, we should use
       * the same mode that the application uses. */
      set_provoking_vertex_mode(cmdbuf, U_TRISTATE_UNSET);
   } else {
      enum u_tristate first_provoking_vertex = u_tristate_make(
         cmdbuf->vk.dynamic_graphics_state.rs.provoking_vertex ==
         VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT);
      set_provoking_vertex_mode(cmdbuf, first_provoking_vertex);
   }

   if (!rs->rasterizer_discard_enable) {
      ASSERTED const struct pan_fb_layout *fb =
         &cmdbuf->state.gfx.render.fb.layout;
      uint32_t *nr_samples = &cmdbuf->state.gfx.render.fb.nr_samples;
      uint32_t rasterization_samples =
         cmdbuf->vk.dynamic_graphics_state.ms.rasterization_samples;

      /* If there's no attachment, and the FB descriptor hasn't been allocated
       * yet, we patch nr_samples to match rasterization_samples, otherwise, we
       * make sure those two numbers match. */
      if (!batch->fb.desc.gpu && !cmdbuf->state.gfx.render.bound_attachments) {
         assert(rasterization_samples > 0);
         *nr_samples = rasterization_samples;
      } else {
         assert(rasterization_samples == *nr_samples);
      }

      /* In case we already emitted tiler/framebuffer descriptors, we ensure
       * that the sample count didn't change
       * XXX: This currently can happen in case we resume a render pass with no
       * attachements and without any draw as the FBD is emitted when suspending.
       */
      assert(fb->sample_count == 0 ||
             fb->sample_count == cmdbuf->state.gfx.render.fb.nr_samples);

      result = panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);
      if (result != VK_SUCCESS)
         return result;
   }

   panvk_per_arch(cmd_select_tile_size)(cmdbuf);

   result = panvk_per_arch(cmd_alloc_tls_desc)(cmdbuf, true);
   if (result != VK_SUCCESS)
      return result;

   const struct panvk_shader_desc_info *vs_desc_info =
      &cmdbuf->state.gfx.vs.shader->desc_info;
   const struct panvk_shader_desc_info *fs_desc_info =
      fs ? &cmdbuf->state.gfx.fs.shader->desc_info : NULL;

   uint32_t used_set_mask =
      vs_desc_info->used_set_mask | (fs ? fs_desc_info->used_set_mask : 0);

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, VS) ||
       gfx_state_dirty(cmdbuf, FS)) {
      result = panvk_per_arch(cmd_prepare_push_descs)(cmdbuf, desc_state,
                                                      used_set_mask);
      if (result != VK_SUCCESS)
         return result;
   }

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, VS)) {
      result = panvk_per_arch(cmd_prepare_shader_desc_tables)(
         cmdbuf, desc_state, vs_desc_info, false, vs_desc_state);
      if (result != VK_SUCCESS)
         return result;

      result = panvk_per_arch(cmd_prepare_dyn_ssbos)(
         cmdbuf, desc_state, vs_desc_info, vs_desc_state);
      if (result != VK_SUCCESS)
         return result;
   }

   /* This allocates and initializes the image table, which we need before we
    * can copy descriptors.
    */
   panvk_draw_prepare_attributes(cmdbuf, draw);

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, VS) ||
       draw->vs_img_tables_new) {
      result = panvk_draw_prepare_vs_copy_desc_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return result;
   }

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, FS)) {
      if (fs == NULL) {
         /* No need to setup the FS desc tables if the FS is not executed. */
         memset(fs_desc_state, 0, sizeof(*fs_desc_state));
      } else {
         result = panvk_per_arch(cmd_prepare_shader_desc_tables)(
            cmdbuf, desc_state, fs_desc_info, true, fs_desc_state);
         if (result != VK_SUCCESS)
            return result;

         result = panvk_per_arch(cmd_prepare_dyn_ssbos)(
            cmdbuf, desc_state, fs_desc_info, fs_desc_state);
         if (result != VK_SUCCESS)
            return result;

         result = panvk_draw_prepare_fs_copy_desc_job(cmdbuf, draw);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   draw->tls = batch->tls.gpu;
   draw->fb = batch->fb.desc.gpu;

   result = panvk_draw_prepare_fs_rsd(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   batch->tlsinfo.tls.size = MAX3(vs->info.tls_size, fs ? fs->info.tls_size : 0,
                                  batch->tlsinfo.tls.size);

   panvk_per_arch(cmd_prepare_draw_sysvals)(cmdbuf, &draw->info, fs);

   /* Viewport emission requires up-to-date {scale,offset}.z for min/max Z,
    * so we need to call it after calling cmd_prepare_draw_sysvals(), but
    * viewports are the same for all layers, so we only emit when layer_id=0.
    */
   result = panvk_draw_prepare_viewport(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   return VK_SUCCESS;
}

static VkResult
prepare_draw_layer(struct panvk_cmd_buffer *cmdbuf,
                   struct panvk_draw_data *draw, uint32_t layer)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   VkResult result;

   /* Before the varyings: transform feedback captures on the first layer only. */
   draw->info.layer_id = layer;
   result = panvk_draw_prepare_varyings(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   draw->info.layer_id = layer;
   if (draw->info.layer_id > 0) {
      cmdbuf->state.gfx.sysvals.layer_id = draw->info.layer_id;
      gfx_state_set_dirty(cmdbuf, FS_PUSH_UNIFORMS);
   }

   struct pan_ptr vs_push_uniforms;
   result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
      cmdbuf, vs, &vs_push_uniforms, 1);
   if (result != VK_SUCCESS)
      return result;
   cmdbuf->state.gfx.vs.push_uniforms = vs_push_uniforms.gpu;

   if (fs) {
      struct pan_ptr fs_push_uniforms;
      result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
         cmdbuf, fs, &fs_push_uniforms, 1);
      if (result != VK_SUCCESS)
         return result;
      cmdbuf->state.gfx.fs.push_uniforms = fs_push_uniforms.gpu;
   }

   result = panvk_draw_prepare_tiler_context(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   if (vs->info.vs.idvs) {
      result = panvk_draw_prepare_idvs_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return result;
   } else {
      result = panvk_draw_prepare_vertex_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return result;

      bool needs_tiling =
         !cmdbuf->vk.dynamic_graphics_state.rs.rasterizer_discard_enable ||
         cmdbuf->state.gfx.occlusion_query.mode !=
            MALI_OCCLUSION_MODE_DISABLED;

      if (needs_tiling) {
         result = panvk_draw_prepare_tiler_job(cmdbuf, draw);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   return VK_SUCCESS;
}

/*
 * A CACHE_FLUSH job between a job that writes memory and a later job in the same chain that reads
 * it: the vertex job and the geometry shader emulation, and each indirect draw helper and what it
 * patches (see panvk_cmd_draw_indirect).
 *
 * NOT EMITTED BY DEFAULT ANY MORE. Every one of these was added against a measured failure --
 * the emulation read zeros from the vertex shader's varyings, a vertex job fetched descriptors
 * as the CPU wrote them instead of as the helper patched them, the tiler drew only the indices
 * that had landed -- and every one of those failures came from the same place: memory allocated
 * without BASE_MEM_COHERENT_LOCAL, on which the GPU's own units do not see one another's writes
 * (kbase_kmod.c). With that bit set, every matrix in probe/mali passes on Mali-G76 and Mali-G72
 * with no flush job at all, and a job dependency orders memory as it does for the vendor.
 * PANVK_FLUSH_JOBS=1 puts them back, for A/B only.
 */
static unsigned
emit_cache_flush_job(struct panvk_cmd_buffer *cmdbuf, unsigned dep_job_id)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("PANVK_FLUSH_JOBS", false);
   if (!enabled)
      return dep_job_id;

   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr job = panvk_cmd_alloc_desc(cmdbuf, CACHE_FLUSH_JOB);

   if (!job.gpu)
      return dep_job_id;

   pan_section_pack(job.cpu, CACHE_FLUSH_JOB, PAYLOAD, cfg) {
      cfg.clean_shader_core_ls = true;
      cfg.invalidate_shader_core_ls = true;
      cfg.invalidate_shader_core_other = true;
      cfg.job_manager_clean = true;
      cfg.job_manager_invalidate = true;
      /* Clean the tiler, never invalidate it: these flushes sit between the draws of a batch,
       * where the tiler holds the polygon lists of every draw tiled so far. */
      cfg.tiler_clean = true;
      cfg.tiler_invalidate = false;
      cfg.l2_clean = true;
      cfg.l2_invalidate = true;
   }

   util_dynarray_append(&batch->jobs, job.cpu);

   return pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_CACHE_FLUSH, true, true,
                         dep_job_id, 0, &job, false);
}

/*
 * Queue the geometry shader as a COMPUTE job in the vertex/tiler chain. Returns the job index the
 * tiler should depend on, or 0 on failure.
 */
/*
 * Tessellation followed by a geometry shader: the tess eval job writes its vertices here instead of
 * the geometry shader's buffers, one per tessellator index in index order and positions in clip
 * space, and the geometry shader job reads them as a list of the tessellator's primitives. offsets is the slot -> byte offset
 * table both use, prims the primitive count the tessellator left.
 */
struct tess_gs_io {
   uint64_t pos, general, index, offsets, prims;
   uint32_t stride;
};

static bool
alloc_tess_gs_io(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_data *draw,
                 struct tess_gs_io *io)
{
   const struct panvk_shader_variant *tes =
      panvk_shader_only_variant(cmdbuf->state.gfx.tess.tes);
   const uint32_t cap = tess_index_cap(cmdbuf, draw);

   /* Every generic output of the tess eval shader, 16 bytes each, in slot order. */
   uint32_t offs[VARYING_SLOT_MAX];
   uint32_t n = 0;
   for (unsigned slot = 0; slot < VARYING_SLOT_MAX; slot++)
      offs[slot] = slot >= VARYING_SLOT_VAR0 && slot < 64 &&
                         (tes->gs.outputs_written & BITFIELD64_BIT(slot))
                      ? 16 * n++
                      : PANVK_GS_SLOT_UNUSED;
   io->stride = MAX2(n, 1) * 16;

   struct pan_ptr pos = panvk_cmd_alloc_dev_mem(cmdbuf, desc, cap * 16, 64);
   struct pan_ptr gen = panvk_cmd_alloc_dev_mem(cmdbuf, desc, cap * io->stride, 64);
   struct pan_ptr idx = panvk_cmd_alloc_dev_mem(cmdbuf, desc, cap * 4, 4);
   struct pan_ptr tab = panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(offs), 4);
   if (!pos.gpu || !gen.gpu || !idx.gpu || !tab.gpu)
      return false;
   memcpy(tab.cpu, offs, sizeof(offs));

   io->pos = pos.gpu;
   io->general = gen.gpu;
   io->index = idx.gpu;
   io->offsets = tab.gpu;
   return true;
}

/*
 * The descriptors of a shader the emulation runs as a compute job (geometry, tess control, tess
 * eval). It was compiled as a compute shader, so it reads the bound GRAPHICS sets the way a
 * compute shader reads its own: tables filled by a copy job, and set addresses in its compute
 * sysvals. The job running the shader waits on *copy_job (0 when there is nothing to copy) and
 * points its DRAW section at *ds. Returns false when an allocation fails.
 */
static bool
emu_prepare_descs(struct panvk_cmd_buffer *cmdbuf, const struct panvk_shader *shader,
                  struct panvk_compute_sysvals *sysvals, struct panvk_shader_desc_state *ds,
                  unsigned *copy_job)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct panvk_descriptor_state *desc_state = &cmdbuf->state.gfx.desc_state;
   const struct panvk_shader_desc_info *info = &shader->desc_info;

   *ds = (struct panvk_shader_desc_state){0};
   *copy_job = 0;

   if (panvk_per_arch(cmd_prepare_push_descs)(cmdbuf, desc_state, info->used_set_mask) !=
          VK_SUCCESS ||
       panvk_per_arch(cmd_prepare_shader_desc_tables)(cmdbuf, desc_state, info, true, ds) !=
          VK_SUCCESS ||
       panvk_per_arch(cmd_prepare_dyn_ssbos)(cmdbuf, desc_state, info, ds) != VK_SUCCESS)
      return false;

   u_foreach_bit(i, info->used_set_mask) {
      if (i < MAX_SETS && desc_state->sets[i])
         sysvals->desc.sets[i] = desc_state->sets[i]->descs.dev;
   }
   sysvals->desc.sets[PANVK_DESC_TABLE_CS_DYN_SSBOS] = ds->dyn_ssbos;

   struct pan_ptr job;
   if (panvk_per_arch(meta_get_copy_desc_job)(cmdbuf, info, desc_state, ds, 0, &job) !=
       VK_SUCCESS)
      return false;
   if (job.cpu) {
      util_dynarray_append(&batch->jobs, job.cpu);
      *copy_job =
         pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false, 0, 0, &job, false);
   }
   return true;
}

static unsigned
emit_gs_job(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw,
            unsigned dep_job_id, const struct panvk_shader *gs_shader, uint64_t tess_params,
            uint64_t tess_count, const struct tess_gs_io *tio)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *gs = panvk_shader_only_variant(gs_shader);

   if (!panvk_priv_mem_check_alloc(gs->rsd))
      return 0;

   /* After tessellation: one invocation per tessellator primitive (per index for the tess eval
    * shader), the count at tess_count. With a geometry shader too, tio is where the tess eval
    * job writes (tes_out) and the geometry shader job reads (tes_in). */
   const bool tess = tess_count != 0;
   const bool tes_out = tio && gs_shader == cmdbuf->state.gfx.tess.tes;
   const bool tes_in = tio && !tes_out;
   const bool indirect = is_indirect_draw(draw);
   const uint32_t prims_per_instance = tess       ? tess_index_cap(cmdbuf, draw) /
                                                       MAX2(gs->gs.input_verts_per_prim, 1)
                                       : indirect ? 0
                                                  : gs_prims_per_instance(draw, gs);
   const uint32_t in_prims = tess       ? prims_per_instance
                             : indirect ? PANVK_GS_INDIRECT_MAX_PRIMS
                                        : prims_per_instance * draw->info.instance.count;
   if (!in_prims)
      return dep_job_id;

   /* Primitive restart splits strips and fans at index values only the GPU can see: have
    * panlib_gs_restart_unroll write the primitives out as a list first, and read that the way an
    * indirect draw is read. prims_per_instance above is the no-restart count, an upper bound. */
   const bool restart = !tess && draw->info.index.index_size && draw->info.index.restart_enable &&
                        gs_topology(draw->info.prim) != PANVK_GS_TOPO_LIST;
   uint64_t unrolled = 0, unrolled_cmd = 0, index_min = draw->index_min_addr;
   if (restart) {
      const uint32_t vpp = MAX2(gs->gs.input_verts_per_prim, 1);
      const uint32_t cap = indirect ? PANVK_GS_INDIRECT_MAX_PRIMS : prims_per_instance;
      struct pan_ptr scratch =
         panvk_cmd_alloc_dev_mem(cmdbuf, desc, cap * vpp * 4 + 64, 16);
      if (!scratch.gpu)
         return 0;

      /* [0, 20): the draw command in, [32, 52): the one out, [60, 64): a zero minimum for a
       * direct draw, [64, ...): the list. */
      uint64_t in_cmd = draw->info.indirect.buffer_dev_addr;
      if (!indirect) {
         const VkDrawIndexedIndirectCommand c = {
            .indexCount = draw->info.vertex.count,
            .instanceCount = draw->info.instance.count,
            .firstIndex = draw->info.index.offset,
            .vertexOffset = draw->info.vertex.base,
            .firstInstance = draw->info.instance.base,
         };
         memcpy(scratch.cpu, &c, sizeof(c));
         in_cmd = scratch.gpu;
         index_min = scratch.gpu + 60;
      }
      memset((uint8_t *)scratch.cpu + 60, 0, 4);
      unrolled_cmd = scratch.gpu + 32;
      unrolled = scratch.gpu + 64;

      struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
      const struct panlib_gs_restart_unroll_args args = {
         .cmd = in_cmd,
         .ib = draw->info.index.buffer_dev_addr,
         .out = unrolled,
         .out_cmd = unrolled_cmd,
         .index_size = draw->info.index.index_size,
         .topology = gs_topology(draw->info.prim),
         .vpp = vpp,
         .max_prims = cap,
      };
      panlib_gs_restart_unroll_struct(&precomp_ctx, panlib_1d_with_jm_deps(1, 0, dep_job_id),
                                      PANLIB_BARRIER_NONE, args);
      dep_job_id = emit_cache_flush_job(cmdbuf, batch->vtc_jc.job_index);
   }

   /* One invocation per (input primitive, gl_InvocationID) pair. */
   const uint32_t wg =
      DIV_ROUND_UP(in_prims * gs->gs.invocations, PANVK_GS_WORKGROUP_SIZE);

   struct panvk_compute_sysvals sysvals = {0};
   sysvals.num_work_groups.x = wg;
   sysvals.local_group_size.x = PANVK_GS_WORKGROUP_SIZE;
   sysvals.gs = (struct panvk_gs_push){
      .in_pos = tes_in ? tio->pos : cmdbuf->state.gfx.gs.in_pos_override ?: draw->position,
      .in_general = tes_in ? tio->general : cmdbuf->state.gfx.gs.in_general,
      .out_pos = tes_out ? tio->pos : cmdbuf->state.gfx.gs.out_pos,
      .out_general = tes_out ? tio->general : cmdbuf->state.gfx.gs.out_general,
      .out_index = tes_out ? tio->index : cmdbuf->state.gfx.gs.out_index,
      .in_offsets = tes_in ? tio->offsets : cmdbuf->state.gfx.gs.in_offsets,
      .out_offsets = tes_out ? tio->offsets : cmdbuf->state.gfx.gs.out_offsets,
      .in_general_stride = tes_in ? tio->stride : cmdbuf->state.gfx.gs.in_general_stride,
      .out_general_stride = tes_out ? tio->stride : cmdbuf->state.gfx.gs.out_general_stride,
      .num_input_prims = in_prims,
      .index_buf = restart ? unrolled
                   : tess || !draw->info.index.index_size ? 0
                   : indirect ? draw->info.index.buffer_dev_addr
                              : draw->info.index.buffer_dev_addr +
                                   (uint64_t)draw->info.index.offset * draw->info.index.index_size,
      .index_size = restart ? 4 : tess ? 0 : draw->info.index.index_size,
      .topology = restart || tess ? PANVK_GS_TOPO_LIST : gs_topology(draw->info.prim),
      .prims_per_instance = prims_per_instance,
      .instance_stride = !tess && draw->info.instance.count > 1 ? draw->padded_vertex_count : 0,
      /* The tess eval shader reads the tessellator's output, never the draw's vertices. */
      .indirect_cmd = restart              ? unrolled_cmd
                      : indirect && !tess ? draw->info.indirect.buffer_dev_addr
                                          : 0,
      .in_pos_desc = indirect || restart
                        ? draw->varying_bufs + PANVK_VARY_BUF_POSITION * pan_size(ATTRIBUTE_BUFFER)
                        : 0,
      .in_general_desc = indirect || restart
                            ? draw->varying_bufs +
                                 PANVK_VARY_BUF_GENERAL * pan_size(ATTRIBUTE_BUFFER)
                            : 0,
      .vertex_dcd = indirect || restart
                       ? draw->jobs.vertex.gpu + pan_section_offset(COMPUTE_JOB, DRAW)
                       : 0,
      .index_min = index_min,
      .tiler_index_count = (indirect || restart || tess) && !tes_out && draw->jobs.tiler.gpu
                              ? draw->jobs.tiler.gpu + pan_section_offset(TILER_JOB, PRIMITIVE) + 12
                              : 0,
      .xfb_counts = tes_out ? 0 : cmdbuf->state.gfx.gs.xfb_counts,
      .xfb_staging = tes_out ? 0 : cmdbuf->state.gfx.gs.xfb_staging,
      /* The tess eval job leaves clip-space positions for the geometry shader job. */
      .in_screen_space =
         !tes_in && !panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader)->gs.vs_clip_space,
      .out_clip_space = tes_out,
      .viewports = cmdbuf->state.gfx.gs.viewports,
      .tess_params = tess_params,
      .num_prims_addr = tess_count,
      .polygon_cull =
         ((cmdbuf->vk.dynamic_graphics_state.rs.cull_mode & VK_CULL_MODE_FRONT_BIT)
             ? PANVK_GS_CULL_FRONT : 0) |
         ((cmdbuf->vk.dynamic_graphics_state.rs.cull_mode & VK_CULL_MODE_BACK_BIT)
             ? PANVK_GS_CULL_BACK : 0) |
         (cmdbuf->vk.dynamic_graphics_state.rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE
             ? PANVK_GS_FRONT_CCW : 0),
      /* The vertex shader's viewport transform, so the emulation can undo it on the way in and
       * re-apply it on the way out. Set by prepare_draw, which has already run. */
      .viewport_scale = {cmdbuf->state.gfx.sysvals.viewport.scale.x,
                         cmdbuf->state.gfx.sysvals.viewport.scale.y,
                         cmdbuf->state.gfx.sysvals.viewport.scale.z},
      .viewport_offset = {cmdbuf->state.gfx.sysvals.viewport.offset.x,
                          cmdbuf->state.gfx.sysvals.viewport.offset.y,
                          cmdbuf->state.gfx.sysvals.viewport.offset.z},
   };

   struct panvk_shader_desc_state ds;
   unsigned copy_job;
   if (!emu_prepare_descs(cmdbuf, gs_shader, &sysvals, &ds, &copy_job))
      return 0;

   struct pan_ptr push_uniforms;
   if (panvk_per_arch(cmd_prepare_gs_push_uniforms)(cmdbuf, gs, &sysvals,
                                                    &push_uniforms) != VK_SUCCESS)
      return 0;

   struct pan_compute_dim dim = {wg, 1, 1};
   uint64_t tls = panvk_per_arch(cmd_dispatch_prepare_tls)(cmdbuf, gs, &dim, false);
   if (!tls)
      return 0;

   struct pan_ptr job = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!job.gpu)
      return 0;

   pan_pack_work_groups_compute(
      pan_section_ptr(job.cpu, COMPUTE_JOB, INVOCATION), wg, 1, 1,
      PANVK_GS_WORKGROUP_SIZE, 1, 1, false, false);

   pan_section_pack(job.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = util_logbase2_ceil(PANVK_GS_WORKGROUP_SIZE + 1) + 2;
   }

   pan_section_pack(job.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = panvk_priv_mem_dev_addr(gs->rsd);
      cfg.attributes = ds.img_attrib_table;
      cfg.attribute_buffers = ds.tables[PANVK_BIFROST_DESC_TABLE_IMG];
      cfg.uniform_buffers = ds.tables[PANVK_BIFROST_DESC_TABLE_UBO];
      cfg.textures = ds.tables[PANVK_BIFROST_DESC_TABLE_TEXTURE];
      cfg.samplers = ds.tables[PANVK_BIFROST_DESC_TABLE_SAMPLER];
      cfg.push_uniforms = push_uniforms.gpu;
      cfg.thread_storage = tls;
   }

   util_dynarray_append(&batch->jobs, job.cpu);

   /* barrier, because a job dependency orders execution but does not make the vertex job's
    * varying writes VISIBLE to this one: the vertex shader reaches the varying buffers through
    * the varying unit, and reading them from a compute job without a barrier returned zeros
    * even though the address arithmetic was verified correct. suppress_prefetch, because the
    * tiler descriptor after it is only valid once this has run. */
   return pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, true, true,
                         dep_job_id, copy_job, &job, false);
}

/*
 * Tessellation, between the vertex job (dep_job_id) and the tess eval job:
 *
 *   tess control      compute, one workgroup per patch, one invocation per output vertex
 *   panlib_tess_*     COUNT: indices per patch
 *   prefix_sum        offsets, the index buffer, the total at *count_out
 *   panlib_tess_*     WITH_COUNTS: domain points and indices
 *
 * Returns the job the tess eval job waits on, 0 on failure.
 */
static unsigned
emit_tess_jobs(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw,
               unsigned dep_job_id, uint64_t *params_out, uint64_t *count_out)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *tcs =
      panvk_shader_only_variant(cmdbuf->state.gfx.tess.tcs);
   const struct vk_dynamic_graphics_state *dyn = &cmdbuf->vk.dynamic_graphics_state;

   if (!panvk_priv_mem_check_alloc(tcs->rsd) || !dev->tess_heap)
      return 0;

   /* An indirect draw sizes the patch arrays and the tess control grid on the GPU
    * (panlib_tess_setup_indirect): nothing below that depends on the patch count is used. */
   const bool indirect = is_indirect_draw(draw);
   const uint32_t pcp = MAX2(dyn->ts.patch_control_points, 1);
   const uint32_t ppi = indirect ? 1 : draw->info.vertex.count / pcp;
   const uint32_t instances = indirect ? 1 : draw->info.instance.count;
   const uint32_t nr_patches = indirect ? 0 : ppi * instances;
   const uint32_t local = MAX2(tcs->gs.tcs_vertices_out, 1);
   const uint32_t tcs_stride_el = tcs->gs.tcs_output_stride / 4;

   /* The batch's heap header. The first tessellated draw of every batch rewinds it on the GPU
    * (prefix_sum), which also covers a command buffer submitted again. */
   const bool reset_heap = !batch->tess_heap;
   if (!batch->tess_heap) {
      struct pan_ptr h = panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(struct poly_heap), 16);
      if (!h.gpu)
         return 0;
      struct poly_heap *heap = h.cpu;
      heap->base = dev->tess_heap->addr.dev;
      heap->bottom = 0;
      heap->size = PANVK_TESS_HEAP_SIZE;
      batch->tess_heap = h.gpu;
   }

   uint32_t size = nr_patches * tcs_stride_el * 4;
   const uint32_t coord_off = ALIGN_POT(size, 16);
   size = coord_off + nr_patches * 4;
   const uint32_t counts_off = ALIGN_POT(size, 16);
   size = counts_off + nr_patches * 4;
   const uint32_t draws_off = ALIGN_POT(size, 16);
   size = draws_off + 32;
   const uint32_t out_off = size;
   size += 16;
   const uint32_t params_off = ALIGN_POT(size, 16);
   size = params_off + sizeof(struct poly_tess_params);

   struct pan_ptr blob = panvk_cmd_alloc_dev_mem(cmdbuf, desc, size, 64);
   if (!blob.gpu)
      return 0;

   enum tess_primitive_mode prim;
   enum gl_tess_spacing spacing;
   bool ccw, points;
   tess_mode(cmdbuf, &prim, &spacing, &ccw, &points);

   struct poly_tess_params *p = (struct poly_tess_params *)((uint8_t *)blob.cpu + params_off);
   *p = (struct poly_tess_params){
      .heap = batch->tess_heap,
      .patch_coord_buffer = dev->tess_heap->addr.dev,
      .coord_allocs = blob.gpu + coord_off,
      .out_draws = blob.gpu + draws_off,
      .tcs_buffer = blob.gpu,
      .counts = blob.gpu + counts_off,
      .tcs_per_vertex_outputs = tcs->gs.tcs_per_vertex_outputs,
      .input_patch_size = pcp,
      .output_patch_size = tcs->gs.tcs_vertices_out,
      .tcs_patch_constants = tcs->gs.tcs_nr_patch_outputs,
      .patches_per_instance = ppi,
      .tcs_stride_el = tcs_stride_el,
      .nr_patches = nr_patches,
      .partitioning = spacing == TESS_SPACING_EQUAL ? POLY_TESS_PARTITIONING_INTEGER
                      : spacing == TESS_SPACING_FRACTIONAL_ODD
                         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
                         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN,
      .points_mode = points,
      .isolines = prim == TESS_PRIMITIVE_ISOLINES,
   };
   if (!points && prim != TESS_PRIMITIVE_ISOLINES)
      p->ccw = ccw ^ (dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT);

   const uint64_t params = blob.gpu + params_off;
   const uint64_t out = blob.gpu + out_off;

   /* The tess control job. */
   struct panvk_compute_sysvals sysvals = {0};
   sysvals.num_work_groups.x = ppi;
   sysvals.num_work_groups.y = instances;
   sysvals.num_work_groups.z = 1;
   sysvals.local_group_size.x = local;
   sysvals.local_group_size.y = 1;
   sysvals.local_group_size.z = 1;
   sysvals.gs = (struct panvk_gs_push){
      .in_pos = draw->position,
      .in_general = cmdbuf->state.gfx.gs.in_general,
      .in_offsets = cmdbuf->state.gfx.gs.in_offsets,
      .in_general_stride = cmdbuf->state.gfx.gs.in_general_stride,
      .index_buf = !draw->info.index.index_size ? 0
                   : indirect                   ? draw->info.index.buffer_dev_addr
                                                : draw->info.index.buffer_dev_addr +
                                                     (uint64_t)draw->info.index.offset *
                                                        draw->info.index.index_size,
      .index_size = draw->info.index.index_size,
      .indirect_cmd = indirect ? draw->info.indirect.buffer_dev_addr : 0,
      .in_pos_desc =
         indirect ? draw->varying_bufs + PANVK_VARY_BUF_POSITION * pan_size(ATTRIBUTE_BUFFER) : 0,
      .in_general_desc =
         indirect ? draw->varying_bufs + PANVK_VARY_BUF_GENERAL * pan_size(ATTRIBUTE_BUFFER) : 0,
      .vertex_dcd = indirect ? draw->jobs.vertex.gpu + pan_section_offset(COMPUTE_JOB, DRAW) : 0,
      .index_min = indirect ? draw->index_min_addr : 0,
      .prims_per_instance = ppi,
      .instance_stride = instances > 1 ? draw->padded_vertex_count : 0,
      .in_screen_space =
         !panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader)->gs.vs_clip_space,
      .viewport_scale = {cmdbuf->state.gfx.sysvals.viewport.scale.x,
                         cmdbuf->state.gfx.sysvals.viewport.scale.y,
                         cmdbuf->state.gfx.sysvals.viewport.scale.z},
      .viewport_offset = {cmdbuf->state.gfx.sysvals.viewport.offset.x,
                          cmdbuf->state.gfx.sysvals.viewport.offset.y,
                          cmdbuf->state.gfx.sysvals.viewport.offset.z},
      .tess_params = params,
      .patch_size = pcp,
   };

   struct panvk_shader_desc_state ds;
   unsigned copy_job;
   if (!emu_prepare_descs(cmdbuf, cmdbuf->state.gfx.tess.tcs, &sysvals, &ds, &copy_job))
      return 0;

   struct pan_ptr push_uniforms;
   if (panvk_per_arch(cmd_prepare_gs_push_uniforms)(cmdbuf, tcs, &sysvals,
                                                    &push_uniforms) != VK_SUCCESS)
      return 0;

   struct pan_compute_dim dim = {ppi, instances, 1};
   uint64_t tls = panvk_per_arch(cmd_dispatch_prepare_tls)(cmdbuf, tcs, &dim, indirect);
   if (!tls)
      return 0;

   struct pan_ptr job = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!job.gpu)
      return 0;

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   if (indirect) {
      /* The sysvals the grid feeds, or the memory sink. */
      uint64_t nwg[2] = {0x8ull << 60, 0x8ull << 60};
      if (shader_uses_sysval(tcs, compute, num_work_groups.x))
         nwg[0] = push_uniforms.gpu +
                  shader_remapped_sysval_offset(tcs, sysval_offset(compute, num_work_groups.x));
      if (shader_uses_sysval(tcs, compute, num_work_groups.y))
         nwg[1] = push_uniforms.gpu +
                  shader_remapped_sysval_offset(tcs, sysval_offset(compute, num_work_groups.y));

      const struct panlib_tess_setup_indirect_args a = {
         .p = blob.gpu + params_off,
         .cmd = draw->info.indirect.buffer_dev_addr,
         .reset_heap = reset_heap,
         .tcs_job = job.gpu,
         .tcs_local = local,
         .num_wg_x = nwg[0],
         .num_wg_y = nwg[1],
      };
      /* suppress_prefetch: the tess control job after it is only valid once this has run. */
      panlib_tess_setup_indirect_struct(
         &precomp_ctx, panlib_1d_with_jm_deps(1, 0, dep_job_id),
         PANLIB_BARRIER_JM_BARRIER | PANLIB_BARRIER_JM_SUPPRESS_PREFETCH, a);
      dep_job_id = emit_cache_flush_job(cmdbuf, batch->vtc_jc.job_index);
   }

   pan_pack_work_groups_compute(pan_section_ptr(job.cpu, COMPUTE_JOB, INVOCATION), ppi,
                                instances, 1, local, 1, 1, false, false);

   pan_section_pack(job.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = util_logbase2_ceil(local + 1) + util_logbase2_ceil(1 + 1) +
                           util_logbase2_ceil(1 + 1);
   }

   pan_section_pack(job.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = panvk_priv_mem_dev_addr(tcs->rsd);
      cfg.attributes = ds.img_attrib_table;
      cfg.attribute_buffers = ds.tables[PANVK_BIFROST_DESC_TABLE_IMG];
      cfg.uniform_buffers = ds.tables[PANVK_BIFROST_DESC_TABLE_UBO];
      cfg.textures = ds.tables[PANVK_BIFROST_DESC_TABLE_TEXTURE];
      cfg.samplers = ds.tables[PANVK_BIFROST_DESC_TABLE_SAMPLER];
      cfg.push_uniforms = push_uniforms.gpu;
      cfg.thread_storage = tls;
   }

   util_dynarray_append(&batch->jobs, job.cpu);
   /* NOT_STARTED for an indirect draw: panlib_tess_setup_indirect merges the real type in. */
   unsigned dep = pan_jc_add_job(&batch->vtc_jc,
                                 indirect ? MALI_JOB_TYPE_NOT_STARTED : MALI_JOB_TYPE_COMPUTE,
                                 true, indirect, dep_job_id, copy_job, &job, false);
   dep = emit_cache_flush_job(cmdbuf, dep);

   /* The tessellator. */
   const uint32_t tess_grid = indirect ? PANVK_TESS_INDIRECT_GRID : nr_patches;
   for (unsigned pass = 0; pass < 2; pass++) {
      const uint32_t mode = pass == 0 ? POLY_TESS_MODE_COUNT : POLY_TESS_MODE_WITH_COUNTS;
      const struct panlib_precomp_grid grid = panlib_1d_with_jm_deps(tess_grid, 0, dep);
      if (prim == TESS_PRIMITIVE_ISOLINES) {
         const struct panlib_tess_isoline_args a = {.p = params, .mode = mode};
         panlib_tess_isoline_struct(&precomp_ctx, grid, PANLIB_BARRIER_JM_BARRIER, a);
      } else if (prim == TESS_PRIMITIVE_QUADS) {
         const struct panlib_tess_quad_args a = {.p = params, .mode = mode};
         panlib_tess_quad_struct(&precomp_ctx, grid, PANLIB_BARRIER_JM_BARRIER, a);
      } else {
         const struct panlib_tess_tri_args a = {.p = params, .mode = mode};
         panlib_tess_tri_struct(&precomp_ctx, grid, PANLIB_BARRIER_JM_BARRIER, a);
      }
      dep = emit_cache_flush_job(cmdbuf, batch->vtc_jc.job_index);

      if (pass == 0) {
         const struct panlib_tess_prefix_sum_args a = {
            .p = params,
            .reset_heap = reset_heap && !indirect,
            .out = out,
         };
         panlib_tess_prefix_sum_struct(&precomp_ctx, panlib_1d_with_jm_deps(1, 0, dep),
                                       PANLIB_BARRIER_JM_BARRIER, a);
         dep = emit_cache_flush_job(cmdbuf, batch->vtc_jc.job_index);
      }
   }

   *params_out = params;
   *count_out = out;
   return dep;
}

/*
 * The compute jobs between the vertex job (dep_job_id) and the tiler: tessellation, then the tess
 * eval shader and the geometry shader through the emulation. Returns the job the tiler waits on,
 * 0 on failure.
 */
static unsigned
emit_emulation_jobs(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw,
                    unsigned dep_job_id)
{
   const struct panvk_shader *tes = cmdbuf->state.gfx.tess.tes;
   const struct panvk_shader *gs = cmdbuf->state.gfx.gs.shader;
   unsigned dep = emit_cache_flush_job(cmdbuf, dep_job_id);
   uint64_t tess_params = 0, tess_count = 0;

   if (tes) {
      dep = emit_tess_jobs(cmdbuf, draw, dep, &tess_params, &tess_count);
      if (!dep)
         return 0;
   }

   if (tes && gs != tes) {
      struct tess_gs_io io;
      if (!alloc_tess_gs_io(cmdbuf, draw, &io))
         return 0;
      /* The tessellator's primitive count, next to its index count (panlib_tess_prefix_sum). */
      io.prims = tess_count + 8;
      dep = emit_gs_job(cmdbuf, draw, dep, tes, tess_params, tess_count, &io);
      if (!dep)
         return 0;
      dep = emit_cache_flush_job(cmdbuf, dep);
      return emit_gs_job(cmdbuf, draw, dep, gs, 0, io.prims, &io);
   }

   return emit_gs_job(cmdbuf, draw, dep, gs, tess_params, tess_count, NULL);
}

/*
 * Transform feedback for a draw whose geometry shader emulation just ran (dep_job_id: the flush
 * after it). panlib_xfb_scan puts the primitives in draw order and fits them into the buffers,
 * panlib_xfb_copy moves the records. The capture jobs of a batch are chained through
 * xfb.last_job, and each draw reads the write offsets the previous one left in its own slot, so no
 * job ever reads memory a later one writes.
 */
static void
emit_xfb_jobs(struct panvk_cmd_buffer *cmdbuf, unsigned dep_job_id)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *gs =
      panvk_shader_only_variant(cmdbuf->state.gfx.gs.shader);
   const uint32_t n = cmdbuf->state.gfx.gs.xfb_invocations;
   const uint32_t k_dw = gs->gs.xfb_dwords;

   /* [draw block][16: offsets after this draw][k_dw: table][n: first primitive per invocation] */
   struct pan_ptr mem = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(struct libpan_xfb_draw) + 16 + (k_dw + n) * 4, 16);
   if (!mem.gpu)
      return;

   const uint64_t slot = mem.gpu + sizeof(struct libpan_xfb_draw);
   const uint64_t table = slot + 16;
   struct libpan_xfb_draw *d = mem.cpu;
   *d = (struct libpan_xfb_draw){
      .offsets_in = cmdbuf->state.gfx.xfb.offsets,
      .offsets_out = slot,
      .counts = cmdbuf->state.gfx.gs.xfb_counts,
      .first = table + k_dw * 4,
      .indices = cmdbuf->state.gfx.gs.out_index,
      .staging = cmdbuf->state.gfx.gs.xfb_staging,
      .table = table,
      .invocations = n,
      .idx_per_invocation = gs->gs.max_output_prims * gs->gs.output_verts_per_prim,
      .vpp = gs->gs.output_verts_per_prim,
      .record_dwords = k_dw,
   };
   for (unsigned b = 0; b < 4; b++) {
      d->buffer[b] = cmdbuf->state.gfx.xfb.addr[b];
      d->size[b] = cmdbuf->state.gfx.xfb.size[b];
      d->stride[b] = gs->gs.xfb_strides[b];
   }
   memcpy((uint8_t *)mem.cpu + sizeof(struct libpan_xfb_draw) + 16, gs->gs.xfb_table, k_dw * 4);
   cmdbuf->state.gfx.xfb.offsets = slot;

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   const struct panlib_xfb_scan_args scan = {.d = mem.gpu};
   panlib_xfb_scan_struct(&precomp_ctx,
                          panlib_1d_with_jm_deps(1, dep_job_id, cmdbuf->state.gfx.xfb.last_job),
                          PANLIB_BARRIER_JM_BARRIER, scan);
   const unsigned scanned = emit_cache_flush_job(cmdbuf, batch->vtc_jc.job_index);

   const struct panlib_xfb_copy_args copy = {.d = mem.gpu};
   panlib_xfb_copy_struct(&precomp_ctx, panlib_1d_with_jm_deps(DIV_ROUND_UP(n, 64), scanned, 0),
                          PANLIB_BARRIER_JM_BARRIER, copy);
   cmdbuf->state.gfx.xfb.last_job = emit_cache_flush_job(cmdbuf, batch->vtc_jc.job_index);
}

/*
 * The geometry shader this draw runs: the application's, or, with none bound, the internal one
 * that draws triangles as lines or points for VK_POLYGON_MODE_LINE/POINT (the tiler only fills).
 * Switching marks the shader dirty, which relinks the varyings against it.
 */
static void
panvk_select_gs(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_data *draw)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct vk_rasterization_state *rs = &cmdbuf->vk.dynamic_graphics_state.rs;
   /* The shader whose output the tiler draws: the application's geometry shader, else the tess
    * eval shader, which runs through the same emulation. */
   const struct panvk_shader *gs =
      cmdbuf->state.gfx.gs.app_shader ? cmdbuf->state.gfx.gs.app_shader : cmdbuf->state.gfx.tess.tes;

   if (!gs && rs->polygon_mode != VK_POLYGON_MODE_FILL && !rs->rasterizer_discard_enable &&
       (draw->info.prim == MESA_PRIM_TRIANGLES || draw->info.prim == MESA_PRIM_TRIANGLE_STRIP ||
        draw->info.prim == MESA_PRIM_TRIANGLE_FAN))
      gs = panvk_per_arch(polygon_mode_gs)(dev, rs->polygon_mode == VK_POLYGON_MODE_POINT);

   if (cmdbuf->state.gfx.gs.shader != gs) {
      cmdbuf->state.gfx.gs.shader = gs;
      gfx_state_set_dirty(cmdbuf, GS);
      gfx_state_set_dirty(cmdbuf, GS_PUSH_UNIFORMS);
   }
}

static void
panvk_cmd_draw(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   const struct panvk_shader_variant *vs = panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   VkResult result;

   /* If there's no vertex shader, we can skip the draw. */
   if (!panvk_priv_mem_check_alloc(vs->rsd))
      return;

   /* Tessellation needs both stages, and a draw with no whole patch draws nothing. */
   if (cmdbuf->state.gfx.tess.tes &&
       (!cmdbuf->state.gfx.tess.tcs || !tess_patches(cmdbuf, draw)))
      return;

   /* Needs to be done before get_fs() is called because it depends on
    * fs.required being initialized. */
   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   panvk_select_gs(cmdbuf, draw);

   result = prepare_draw(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return;

   pan_pack_work_groups_compute(&draw->invocation, 1, draw->vertex_range,
                                draw->info.instance.count, 1, 1, 1, true,
                                false);

   struct panvk_batch *batch = cmdbuf->cur_batch;

   unsigned copy_desc_job_id =
      draw->jobs.vertex_copy_desc.gpu
         ? pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false,
                          0, 0, &draw->jobs.vertex_copy_desc, false)
         : 0;

   if (draw->jobs.frag_copy_desc.gpu) {
      /* We don't need to add frag_copy_desc as a dependency because the
       * tiler job doesn't execute the fragment shader, the fragment job
       * will, and the tiler/fragment synchronization happens at the batch
       * level. */
      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false, 0, 0,
                     &draw->jobs.frag_copy_desc, false);
   }

   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   assert(view_mask == 0 || util_bitcount(view_mask) <= batch->fb.layer_count);
   uint32_t enabled_layer_count = view_mask
                                     ? util_bitcount(view_mask)
                                     : cmdbuf->state.gfx.render.layer_count;

   for (uint32_t i = 0; i < enabled_layer_count; i++) {
      const uint32_t layer = (view_mask != 0) ? u_bit_scan(&view_mask) : i;
      result = prepare_draw_layer(cmdbuf, draw, layer);
      if (result != VK_SUCCESS)
         return;

      if (vs->info.vs.idvs) {
         pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_INDEXED_VERTEX, false,
                        false, 0, copy_desc_job_id, &draw->jobs.idvs, false);
      } else {
         unsigned vjob_id =
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_VERTEX, false, false,
                           0, copy_desc_job_id, &draw->jobs.vertex, false);

         /* Geometry shader emulation: a compute job between the vertex job and the tiler job,
          * turning the vertices the vertex shader just wrote into the ones the tiler will read.
          * The tiler waits on it rather than on the vertex job. */
         unsigned gs_job_id = vjob_id;
         if (draw->gs_varying_bufs && (!cmdbuf->state.gfx.gs.cpufill ||
                                       cmdbuf->state.gfx.gs.keepjob)) {
            gs_job_id = emit_emulation_jobs(cmdbuf, draw, vjob_id);
            if (!gs_job_id)
               return;
            /* The tiler reads the emulation's output. Measured on Mali-G72: without this the
             * tiler read an index buffer the GS job had written but not yet published, and drew
             * only the primitives whose indices happened to have landed. */
            gs_job_id = emit_cache_flush_job(cmdbuf, gs_job_id);
            if (cmdbuf->state.gfx.gs.xfb_counts)
               emit_xfb_jobs(cmdbuf, gs_job_id);
         }

         if (draw->jobs.tiler.gpu != 0) {
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_TILER, false, false,
                           gs_job_id, 0, &draw->jobs.tiler, false);
         }
      }
   }

   clear_dirty_after_draw(cmdbuf);
   cmdbuf->state.gfx.vs.previous_draw_was_indirect = false;
}

/* The batch's index scan and patch jobs (see panlib_draw_batch_patch), queued at its first
 * indirect draw so they sit ahead of every draw that waits on them in the job chain; the draws
 * are handed over at the batch's close. Returns the job the draws' vertex jobs wait on. */
#define PANVK_DRAW_BATCH_SCAN_GROUPS 64
#define PANVK_DRAW_BATCH_PATCH_GROUPS 4

static unsigned
draw_helper_ready_job(struct panvk_cmd_buffer *cmdbuf)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;

   if (batch->draw_helper.hdr.gpu)
      return batch->draw_helper.ready_job;

   struct pan_ptr hdr =
      panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(struct libpan_draw_batch), 8);
   if (!hdr.gpu)
      return 0;
   memset(hdr.cpu, 0, sizeof(struct libpan_draw_batch));

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   const struct panlib_draw_batch_minmax_args scan = {.batch = hdr.gpu};
   const struct panlib_draw_batch_patch_args patch = {.batch = hdr.gpu};

   panlib_draw_batch_minmax_struct(&precomp_ctx,
                                   panlib_1d_with_jm_deps(PANVK_DRAW_BATCH_SCAN_GROUPS, 0, 0),
                                   PANLIB_BARRIER_NONE, scan);
   /* The patch reads what the scan wrote. */
   const unsigned scan_job = emit_cache_flush_job(cmdbuf, batch->vtc_jc.job_index);
   panlib_draw_batch_patch_struct(&precomp_ctx,
                                  panlib_1d_with_jm_deps(PANVK_DRAW_BATCH_PATCH_GROUPS, 0,
                                                         scan_job),
                                  PANLIB_BARRIER_JM_SUPPRESS_PREFETCH, patch);

   /* The helper patches the vertex and tiler descriptors on the GPU, and a job dependency alone
    * does not make its stores visible to the jobs that read them: measured on Mali-G72, the
    * vertex job fetched the descriptors as the CPU had written them (varying buffers of size
    * 0, so every position store was dropped), and the patched values only reached memory
    * later. Clean and invalidate between the helper and the draws. */
   batch->draw_helper.ready_job = emit_cache_flush_job(cmdbuf, batch->vtc_jc.job_index);
   batch->draw_helper.hdr = hdr;
   return batch->draw_helper.ready_job;
}

void
panvk_per_arch(cmd_finish_draw_helper)(struct panvk_cmd_buffer *cmdbuf,
                                       struct panvk_batch *batch)
{
   struct libpan_draw_batch *hdr = batch->draw_helper.hdr.cpu;
   const unsigned n = util_dynarray_num_elements(&batch->draw_helper.entries,
                                                 struct libpan_draw_batch_entry);

   if (hdr && n) {
      const size_t size = n * sizeof(struct libpan_draw_batch_entry);
      struct pan_ptr entries = panvk_cmd_alloc_dev_mem(cmdbuf, desc, size, 8);
      if (entries.gpu) {
         memcpy(entries.cpu, util_dynarray_begin(&batch->draw_helper.entries), size);
         hdr->entries = entries.gpu;
         hdr->count = n;
      }
   }

   util_dynarray_fini(&batch->draw_helper.entries);
   memset(&batch->draw_helper, 0, sizeof(batch->draw_helper));
}

static void
panvk_cmd_draw_indirect(struct panvk_cmd_buffer *cmdbuf,
                        struct panvk_draw_data *draw)
{
   const struct panvk_shader_variant *vs = panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   VkResult result;

   /* If there's no vertex shader, we can skip the draw. */
   if (!panvk_priv_mem_check_alloc(vs->rsd))
      return;

   /* Tessellation needs both stages. */
   if (cmdbuf->state.gfx.tess.tes && !cmdbuf->state.gfx.tess.tcs)
      return;

   /* Needs to be done before get_fs() is called because it depends on
    * fs.required being initialized. */
   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   panvk_select_gs(cmdbuf, draw);

   result = prepare_draw(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return;

   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct vk_vertex_input_state *vi =
      cmdbuf->vk.dynamic_graphics_state.vi;

   unsigned copy_desc_job_id =
      draw->jobs.vertex_copy_desc.gpu
         ? pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false,
                          0, 0, &draw->jobs.vertex_copy_desc, false)
         : 0;

   if (draw->jobs.frag_copy_desc.gpu) {
      /* We don't need to add frag_copy_desc as a dependency because the
       * tiler job doesn't execute the fragment shader, the fragment job
       * will, and the tiler/fragment synchronization happens at the batch
       * level. */
      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false, 0, 0,
                     &draw->jobs.frag_copy_desc, false);
   }

   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   assert(view_mask == 0 || util_bitcount(view_mask) <= batch->fb.layer_count);
   uint32_t enabled_layer_count = view_mask
                                     ? util_bitcount(view_mask)
                                     : cmdbuf->state.gfx.render.layer_count;

   const unsigned ready_job = draw_helper_ready_job(cmdbuf);
   if (!ready_job)
      return;

   for (uint32_t i = 0; i < enabled_layer_count; i++) {
      const uint32_t layer = (view_mask != 0) ? u_bit_scan(&view_mask) : i;

      /* Force a new push uniform block to be allocated */
      gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);

      result = prepare_draw_layer(cmdbuf, draw, layer);
      if (result != VK_SUCCESS)
         return;

      assert(draw->info.indirect.buffer_dev_addr != 0);

      struct libpan_draw_batch_entry e = {
         .cmd = draw->info.indirect.buffer_dev_addr,
         .index_buffer_ptr = draw->info.index.buffer_dev_addr,
         .varying_bufs_descs = draw->varying_bufs,
         .varying_bufs_info = draw->indirect_info.varying_bufs,
         .attrib_bufs_descs = draw->vs.attribute_bufs,
         .attrib_bufs_infos = draw->indirect_info.attrib_bufs,
         .attribs_descs = draw->vs.attributes,
         .attribs_infos = draw->indirect_info.attribs,
         .idvs_job = vs->info.vs.idvs ? draw->jobs.idvs.gpu : 0,
         .vertex_job = draw->jobs.vertex.gpu,
         /* With a geometry shader the tiler draws what the emulation wrote, set up on the CPU. */
         .tiler_job = draw->gs_varying_bufs ? 0 : draw->jobs.tiler.gpu,
         .index_size = draw->info.index.index_size,
         .primitive_restart = draw->info.index.restart_enable,
         .primitive_vertex_count = mesa_vertices_per_prim(draw->info.prim),
         .attrib_bufs_valid = vi->bindings_valid,
         .attribs_valid = vi->attributes_valid,
         .index_min = UINT32_MAX,
         .index_max = 0,
      };

      if (shader_uses_sysval(vs, graphics, vs.first_vertex)) {
         e.first_vertex_sysval = cmdbuf->state.gfx.vs.push_uniforms +
                                 shader_remapped_sysval_offset(
                                    vs, sysval_offset(graphics, vs.first_vertex));
      }

      if (shader_uses_sysval(vs, graphics, vs.base_instance)) {
         e.first_instance_sysval = cmdbuf->state.gfx.vs.push_uniforms +
                                   shader_remapped_sysval_offset(
                                      vs, sysval_offset(graphics, vs.base_instance));
      }

      if (shader_uses_sysval(vs, graphics, vs.raw_vertex_offset)) {
         e.raw_vertex_offset_sysval = cmdbuf->state.gfx.vs.push_uniforms +
                                      shader_remapped_sysval_offset(
                                         vs, sysval_offset(graphics, vs.raw_vertex_offset));
      }

      if (draw->info.index.index_size) {
         struct pan_ptr m = panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(uint32_t), 4);
         if (!m.gpu)
            return;
         e.index_min_out = m.gpu;
         draw->index_min_addr = m.gpu;
      }

      util_dynarray_append(&batch->draw_helper.entries, e);

      /* The vertex job waits for the patch job and for this draw's descriptor copy. */
      uint32_t prev_job = ready_job;

      if (vs->info.vs.idvs) {
         pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_INDEXED_VERTEX, false,
                        false, copy_desc_job_id, prev_job, &draw->jobs.idvs, false);
      } else {
         unsigned vjob_id =
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_VERTEX, false, true,
                           copy_desc_job_id, prev_job, &draw->jobs.vertex, false);

         unsigned tiler_dep = vjob_id;
         if (draw->gs_varying_bufs) {
            tiler_dep = emit_emulation_jobs(cmdbuf, draw, vjob_id);
            if (!tiler_dep)
               return;
            /* The tiler reads what the emulation wrote, including its own index count. */
            tiler_dep = emit_cache_flush_job(cmdbuf, tiler_dep);
            if (cmdbuf->state.gfx.gs.xfb_counts)
               emit_xfb_jobs(cmdbuf, tiler_dep);
         }

         /* suppress_prefetch: the helper (or the emulation) patches this descriptor on the GPU,
          * exactly like the vertex job's, so it must not be fetched ahead of the patch. */
         if (draw->jobs.tiler.gpu != 0) {
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_TILER, false, true,
                           tiler_dep, 0, &draw->jobs.tiler, false);
         }
      }
   }

   /*
    * We split every ~1024 indirect draw.
    * This is here for multiple reasons:
    * - The indirect varying buffer offset need to be reset at some point to
    * avoid going outside of bounds.
    * - It is possible to always end up with timeouts for batches with 4k draws
    * (see "dEQP-VK.api.command_buffers.many_indirect_draws_on_secondary") At
    * the same time, because of how TLS works on Mali, we should not split too
    * much as this will cause the TLS budget to go crazy.
    */
   if (batch->vtc_jc.job_index > (5 * 1024)) {
      panvk_per_arch(cmd_close_batch)(cmdbuf);
      batch = panvk_per_arch(cmd_open_batch)(cmdbuf);
      cmdbuf->state.gfx.vs.indirect_varying_bufs_infos = 0;
   }

   clear_dirty_after_draw(cmdbuf);
   cmdbuf->state.gfx.vs.previous_draw_was_indirect = true;
}

static unsigned
padded_vertex_count(struct panvk_cmd_buffer *cmdbuf, uint32_t vertex_count,
                    uint32_t instance_count)
{
   if (instance_count == 1)
      return vertex_count;

   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   bool idvs = vs->info.vs.idvs;

   /* Index-Driven Vertex Shading requires different instances to
    * have different cache lines for position results. Each vertex
    * position is 16 bytes and the Mali cache line is 64 bytes, so
    * the instance count must be aligned to 4 vertices.
    */
   if (idvs)
      vertex_count = ALIGN_POT(vertex_count, 4);

   return pan_padded_vertex_count(vertex_count);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDraw)(VkCommandBuffer commandBuffer, uint32_t vertexCount,
                        uint32_t instanceCount, uint32_t firstVertex,
                        uint32_t firstInstance)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (instanceCount == 0 || vertexCount == 0)
      return;

   /* gl_BaseVertexARB is a signed integer, and it should expose the value of
    * firstVertex in a non-indexed draw. */
   assert(firstVertex < INT32_MAX);

   /* gl_BaseInstance is a signed integer, and it should expose the value of
    * firstInstnace. */
   assert(firstInstance < INT32_MAX);

   struct panvk_draw_data draw = {
      .info = {
         .vertex.base = firstVertex,
         .vertex.raw_offset = firstVertex,
         .vertex.count = vertexCount,
         .instance.base = firstInstance,
         .instance.count = instanceCount,
         .prim = panvk_get_client_prim(cmdbuf),
      },
      .vertex_range = vertexCount,
      .padded_vertex_count =
         padded_vertex_count(cmdbuf, vertexCount, instanceCount),
   };

   panvk_cmd_draw(cmdbuf, &draw);
}

/* How many vertices a direct indexed draw has to shade.
 *
 * The vertex job shades a CONTIGUOUS RANGE and the tiler then resolves index i to varying slot
 * i + base_vertex_offset, so the range has to cover every index the draw can reference. The GPU
 * helper narrows that range to [min_index, max_index] by scanning the index buffer first, which
 * is why upstream needs the search kernel at all; it is an optimisation, not a requirement.
 *
 * Without reading the indices -- which we cannot do at record time, since the buffer's contents
 * are only defined at submit time -- the sound bound is every vertex the bound attribute buffers
 * can address. Indices past that are already undefined behaviour in Vulkan, and Mali clamps the
 * fetch against ATTRIBUTE_BUFFER.size regardless, so nothing reads out of bounds.
 *
 * THE COST IS VERTEX SHADER INVOCATIONS, and it is paid by a draw that indexes a small part of
 * a large shared vertex buffer. The fix for that is gallium panfrost's: a min/max scan of the
 * index range with a per-resource cache (pan_minmax_cache_get in pan_helpers.c). That is a
 * bounded optimisation to add on top of this; it is NOT what makes indexed draws work. */
#define PANVK_DIRECT_INDEXED_MAX_RANGE (64u * 1024)

static uint32_t
direct_indexed_vertex_range(struct panvk_cmd_buffer *cmdbuf,
                            uint32_t index_count, int32_t vertex_offset)
{
   const struct vk_vertex_input_state *vi =
      cmdbuf->vk.dynamic_graphics_state.vi;
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   uint32_t range = 0;
   bool bounded = false;

   u_foreach_bit(i, vi->bindings_valid) {
      const struct vk_vertex_binding_state *binding = &vi->bindings[i];
      uint32_t stride = dyns->vi_binding_strides[i];

      /* Per-instance data is indexed by instance, not by vertex, so it says nothing about how
       * far the index buffer can reach. A zero stride is the same story. */
      if (binding->input_rate != VK_VERTEX_INPUT_RATE_VERTEX || !stride)
         continue;

      uint32_t verts = cmdbuf->state.gfx.vb.bufs[i].size / stride;

      /* offset_start starts the fetch at vertexOffset, so that many vertices are already spent. */
      if (vertex_offset > 0)
         verts = verts > (uint32_t)vertex_offset ? verts - vertex_offset : 0;

      range = bounded ? MIN2(range, verts) : verts;
      bounded = true;
   }

   /* No per-vertex attribute bound: nothing constrains the indices, and the only per-vertex input
    * left is gl_VertexIndex. index_count is the best available guess and is exact whenever the
    * indices are a permutation of [0, index_count), which is the shape this case takes in
    * practice (fullscreen passes and generated geometry). */
   if (!bounded)
      return index_count;

   /* An index buffer can always name at least the vertices the draw asks for. */
   return MAX2(range, index_count);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexed)(VkCommandBuffer commandBuffer,
                               uint32_t indexCount, uint32_t instanceCount,
                               uint32_t firstIndex, int32_t vertexOffset,
                               uint32_t firstInstance)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (instanceCount == 0 || indexCount == 0)
      return;

   /* gl_BaseInstance is a signed integer, and it should expose the value of
    * firstInstnace. */
   assert(firstInstance < INT32_MAX);

   /* A direct indexed draw knows all of its own parameters, so it is emitted by the CPU like
    * vkCmdDraw is, rather than being turned into a synthetic indirect draw for a GPU helper to
    * resolve. See is_indirect_draw() for what that helper does not manage to do here. */
   const uint32_t vertex_range =
      direct_indexed_vertex_range(cmdbuf, indexCount, vertexOffset);

   /* That bound is every vertex the bound buffers can address, which is a disaster for a draw
    * that indexes a small part of a large shared vertex buffer: measured in Minecraft (every chunk
    * section drawn out of one 256 MB block), 4.8 million vertices shaded and 255 MB of varyings
    * allocated PER DRAW, until the process was killed for memory within seconds. Past
    * PANVK_DIRECT_INDEXED_MAX_RANGE, and when the draw plainly uses far less, the GPU finds the
    * real [min, max] index first, through the same scan and helper an indexed indirect draw uses
    * (which is what upstream does for every indexed draw), from a command written here. */
   if (vertex_range > PANVK_DIRECT_INDEXED_MAX_RANGE && vertex_range / 4 > indexCount) {
      struct pan_ptr c =
         panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(VkDrawIndexedIndirectCommand), 4);
      if (c.gpu) {
         *(VkDrawIndexedIndirectCommand *)c.cpu = (VkDrawIndexedIndirectCommand){
            .indexCount = indexCount,
            .instanceCount = instanceCount,
            .firstIndex = firstIndex,
            .vertexOffset = vertexOffset,
            .firstInstance = firstInstance,
         };
         struct panvk_draw_data idraw = {
            .info = {
               .index = panvk_draw_info_index(cmdbuf, 0),
               .indirect.buffer_dev_addr = c.gpu,
               .indirect.draw_count = 1,
               .indirect.stride = sizeof(VkDrawIndexedIndirectCommand),
               .prim = panvk_get_client_prim(cmdbuf),
            },
         };
         panvk_cmd_draw_indirect(cmdbuf, &idraw);
         return;
      }
   }

   struct panvk_draw_data draw = {
      .info = {
         .index = panvk_draw_info_index(cmdbuf, firstIndex),
         /* min_index is taken as 0, so raw_offset is just the application's vertexOffset and
          * base_vertex_offset (base - raw_offset) comes out zero. */
         .vertex.base = vertexOffset,
         .vertex.raw_offset = vertexOffset,
         .vertex.count = indexCount,
         .instance.base = firstInstance,
         .instance.count = instanceCount,
         .prim = panvk_get_client_prim(cmdbuf),
      },
      .vertex_range = vertex_range,
      .padded_vertex_count =
         padded_vertex_count(cmdbuf, vertex_range, instanceCount),
   };

   panvk_cmd_draw(cmdbuf, &draw);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndirect)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                                VkDeviceSize offset, uint32_t drawCount,
                                uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);

   /* The job manager takes one indirect draw at a time: multiDrawIndirect is drawCount of
    * them, each reading its own command. */
   for (uint32_t i = 0; i < drawCount; i++) {
      struct panvk_draw_data draw = {
         .info = {
            .indirect.buffer_dev_addr =
               panvk_buffer_gpu_ptr(buffer, offset + (VkDeviceSize)i * stride),
            .indirect.draw_count = 1,
            .indirect.stride = stride,
            .prim = panvk_get_client_prim(cmdbuf),
         },
      };

      panvk_cmd_draw_indirect(cmdbuf, &draw);
   }
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexedIndirect)(VkCommandBuffer commandBuffer,
                                       VkBuffer _buffer, VkDeviceSize offset,
                                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);

   /* Because we don't currently advertise nullDescriptor for JM, it is only
    * valid to draw with a null index buffer if the draw accesses 0 indices.
    * For direct draws, this is covered by checks on instancedCount and
    * indexCount. For indirect draws we need to add an additional check, under
    * the assumption that if the index buffer is null, the draw must be empty.
    */
   if (drawCount == 0 || cmdbuf->state.gfx.ib.size == 0)
      return;

   /* One indirect draw at a time, as in CmdDrawIndirect. */
   for (uint32_t i = 0; i < drawCount; i++) {
      struct panvk_draw_data draw = {
         .info = {
            .index = panvk_draw_info_index(cmdbuf, 0),
            .indirect.buffer_dev_addr =
               panvk_buffer_gpu_ptr(buffer, offset + (VkDeviceSize)i * stride),
            .indirect.draw_count = 1,
            .indirect.stride = stride,
            .prim = panvk_get_client_prim(cmdbuf),
         },
      };

      panvk_cmd_draw_indirect(cmdbuf, &draw);
   }
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginRendering)(VkCommandBuffer commandBuffer,
                                  const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct panvk_cmd_graphics_state *state = &cmdbuf->state.gfx;
   bool resuming = pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT;

   /* When resuming from a suspended pass, the state should be unchanged. */
   if (resuming && cmdbuf->cur_batch) {
      state->render.flags = pRenderingInfo->flags;
   } else {
      /* If we're not resuming, cur_batch should be NULL.  However, this
       * currently isn't true because of how events are implemented.
       *
       * XXX: Rewrite events to not close and open batch and add an assert here.
       */
      if (cmdbuf->cur_batch)
         panvk_per_arch(cmd_close_batch)(cmdbuf);

      panvk_per_arch(cmd_init_render_state)(cmdbuf, pRenderingInfo);
      cmdbuf->state.gfx.render.fb.needs_load = !resuming;
   }

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndRendering)(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (!(cmdbuf->state.gfx.render.flags & VK_RENDERING_SUSPENDING_BIT)) {
      const struct pan_fb_load *fb_load = &cmdbuf->state.gfx.render.fb.load;
      bool always_load = fb_load->z.always || fb_load->s.always;
      for (unsigned rt = 0; rt < PAN_MAX_RTS; rt++) {
         if (fb_load->rts[rt].always)
            always_load = true;
      }

      if (always_load)
         panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);

      cmdbuf->state.gfx.render.fb.needs_store = true;

      panvk_per_arch(cmd_close_batch)(cmdbuf);
      cmdbuf->cur_batch = NULL;
      panvk_per_arch(cmd_meta_resolve_attachments)(cmdbuf);
   }
}

/* --- VK_EXT_transform_feedback -------------------------------------------------------------- */

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBindTransformFeedbackBuffers2EXT)(
   VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount,
   const VkBindTransformFeedbackBuffer2InfoEXT *pBindingInfos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   for (uint32_t i = 0; i < bindingCount && firstBinding + i < 4; i++) {
      cmdbuf->state.gfx.xfb.addr[firstBinding + i] = pBindingInfos[i].addressRange.address;
      cmdbuf->state.gfx.xfb.size[firstBinding + i] =
         MIN2(pBindingInfos[i].addressRange.size, UINT32_MAX);
   }
}

/* Load the write offsets from the counter buffers (Begin) or store them there (End). Without a
 * counter, Begin starts at 0, which the CPU can write itself. */
static void
xfb_counters(struct panvk_cmd_buffer *cmdbuf, uint32_t first, uint32_t count,
             const VkBindTransformFeedbackBuffer2InfoEXT *infos, bool load)
{
   uint64_t c[4] = {0};
   for (uint32_t i = 0; infos && i < count && first + i < 4; i++)
      c[first + i] = infos[i].addressRange.address;
   const bool any = c[0] || c[1] || c[2] || c[3];

   if (load) {
      struct pan_ptr slot = panvk_cmd_alloc_dev_mem(cmdbuf, desc, 16, 16);
      if (!slot.gpu)
         return;
      memset(slot.cpu, 0, 16);
      cmdbuf->state.gfx.xfb.offsets = slot.gpu;
   }
   if (!any)
      return;

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   const struct panlib_xfb_counters_args args = {
      .offsets = cmdbuf->state.gfx.xfb.offsets,
      .c0 = c[0],
      .c1 = c[1],
      .c2 = c[2],
      .c3 = c[3],
      .load = load,
   };
   panlib_xfb_counters_struct(&precomp_ctx,
                              panlib_1d_with_jm_deps(1, 0, cmdbuf->state.gfx.xfb.last_job),
                              PANLIB_BARRIER_JM_BARRIER, args);
   cmdbuf->state.gfx.xfb.last_job =
      emit_cache_flush_job(cmdbuf, cmdbuf->cur_batch->vtc_jc.job_index);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginTransformFeedback2EXT)(
   VkCommandBuffer commandBuffer, uint32_t firstCounterRange, uint32_t counterRangeCount,
   const VkBindTransformFeedbackBuffer2InfoEXT *pCounterInfos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   xfb_counters(cmdbuf, firstCounterRange, counterRangeCount, pCounterInfos, true);
   cmdbuf->state.gfx.xfb.active = true;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndTransformFeedback2EXT)(
   VkCommandBuffer commandBuffer, uint32_t firstCounterRange, uint32_t counterRangeCount,
   const VkBindTransformFeedbackBuffer2InfoEXT *pCounterInfos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   xfb_counters(cmdbuf, firstCounterRange, counterRangeCount, pCounterInfos, false);
   cmdbuf->state.gfx.xfb.active = false;
}
