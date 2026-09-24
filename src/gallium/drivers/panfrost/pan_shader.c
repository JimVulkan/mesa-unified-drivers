/*
 * Copyright (C) 2025 Arm Ltd.
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates.
 * Copyright (C) 2019-2022 Collabora, Ltd.
 * Copyright (C) 2019 Red Hat Inc.
 * Copyright (C) 2018 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "util/u_debug.h"
#include "util/os_time.h"
#include "pan_shader.h"
#include "nir/tgsi_to_nir.h"
#include "util/u_memory.h"
#include "util/u_prim.h"
#include "nir_builder.h"
#include "nir_serialize.h"
#include "pan_bo.h"
#include "pan_context.h"
#include "poly/nir/poly_nir.h"
#include "pan_compiler.h"
#include "pan_nir.h"
#include "pan_trace.h"
#include "compiler/bifrost/bifrost_compile.h"
#include "shader_enums.h"

static struct panfrost_uncompiled_shader *
panfrost_alloc_shader(const nir_shader *nir)
{
   struct panfrost_uncompiled_shader *so =
      rzalloc(NULL, struct panfrost_uncompiled_shader);

   simple_mtx_init(&so->lock, mtx_plain);
   util_dynarray_init(&so->variants, so);

   so->nir = nir;

   /* Serialize the NIR to a binary blob that we can hash for the disk
    * cache. Drop unnecessary information (like variable names) so the
    * serialized NIR is smaller, and also to let us detect more isomorphic
    * shaders when hashing, increasing cache hits.
    */
   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, nir, true);
   _mesa_blake3_compute(blob.data, blob.size, so->nir_blake3);
   blob_finish(&blob);

   return so;
}

static struct panfrost_compiled_shader *
panfrost_alloc_variant(struct panfrost_uncompiled_shader *so)
{
   return util_dynarray_grow(&so->variants, struct panfrost_compiled_shader, 1);
}

static bool
lower_load_poly_line_smooth_enabled(nir_builder *b, nir_intrinsic_instr *intrin,
                                    void *data)
{
   if (intrin->intrinsic != nir_intrinsic_load_poly_line_smooth_enabled)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def_replace(&intrin->def, nir_imm_true(b));
   return true;
}

/* From the OpenGL 4.6 spec 14.3.1:
 *
 *    If MULTISAMPLE is disabled, multisample rasterization of all primitives
 *    is equivalent to single-sample (fragment-center) rasterization, except
 *    that the fragment coverage value is set to full coverage.
 *
 * So always use the original sample mask when multisample is disabled */
static bool
lower_sample_mask_writes(nir_builder *b, nir_intrinsic_instr *intrin,
                         void *data)
{
   if (intrin->intrinsic != nir_intrinsic_store_output)
      return false;

   if (nir_intrinsic_io_semantics(intrin).location != FRAG_RESULT_SAMPLE_MASK)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *orig = nir_load_sample_mask(b);
   nir_def *new =
      nir_bcsel_pan(b, nir_load_multisampled_pan(b), intrin->src[0].ssa, orig);
   nir_src_rewrite(&intrin->src[0], new);

   return true;
}

/*
 * gl_ViewportIndex from a geometry shader: the tiler has one scissor for every viewport, so the
 * fragment shader discards outside the rectangle of the viewport its primitive picked. The index
 * arrives as a flat varying, the rectangles ({min x, min y, max x, max y} in pixels, max
 * exclusive) from a table the draw path points the GS_PARAMS sysval at.
 */
/* The fragment shader's gl_ViewportIndex and gl_Layer reads, from the slots the emulation
 * writes them in (PAN_GS_VIEWPORT_SLOT). */
static bool
panfrost_move_emulated_inputs(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input &&
       intr->intrinsic != nir_intrinsic_load_interpolated_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location == VARYING_SLOT_VIEWPORT)
      sem.location = PAN_GS_VIEWPORT_SLOT;
   else if (sem.location == VARYING_SLOT_LAYER)
      sem.location = PAN_GS_LAYER_SLOT;
   else
      return false;

   nir_intrinsic_set_io_semantics(intr, sem);
   b->shader->info.inputs_read |= BITFIELD64_BIT(sem.location);
   return true;
}

/* The input index a load of this slot already uses, or a new one: two indices for one slot
 * would leave one of them without a descriptor. */
static unsigned
panfrost_input_base(nir_shader *nir, gl_varying_slot slot)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if ((intr->intrinsic == nir_intrinsic_load_input ||
                 intr->intrinsic == nir_intrinsic_load_interpolated_input) &&
                nir_intrinsic_io_semantics(intr).location == slot)
               return nir_intrinsic_base(intr);
         }
      }
   }

   return nir->num_inputs++;
}

static bool
panfrost_lower_viewport_clip_fs(nir_shader *nir)
{
   const unsigned base = panfrost_input_base(nir, PAN_GS_VIEWPORT_SLOT);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *idx = nir_load_input(&b, 1, 32, nir_imm_int(&b, 0), .base = base,
                                 .dest_type = nir_type_int32,
                                 .io_semantics.location = PAN_GS_VIEWPORT_SLOT,
                                 .io_semantics.num_slots = 1);
   nir->info.inputs_read |= BITFIELD64_BIT(PAN_GS_VIEWPORT_SLOT);
   idx = nir_umin(&b, idx, nir_imm_int(&b, PAN_GS_MAX_VIEWPORTS - 1));

   nir_def *rect = nir_iadd(&b, nir_load_geometry_param_buffer_poly(&b),
                            nir_u2u64(&b, nir_imul_imm(&b, idx, 16)));
   nir_def *r[4];
   for (unsigned i = 0; i < 4; i++)
      r[i] = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, rect, i * 4), .align_mul = 4);

   nir_def *pos = nir_load_frag_coord(&b);
   nir_def *x = nir_channel(&b, pos, 0), *y = nir_channel(&b, pos, 1);
   nir_def *out = nir_ior(&b, nir_ior(&b, nir_flt(&b, x, r[0]), nir_flt(&b, y, r[1])),
                          nir_ior(&b, nir_fge(&b, x, r[2]), nir_fge(&b, y, r[3])));
   nir_discard_if(&b, out);

   /* Or forward pixel kill lets a later primitive remove this one's pixels before the discard
    * runs. */
   nir->info.fs.uses_discard = true;

   return nir_progress(true, impl, nir_metadata_none);
}

/*
 * Cull distances, per fragment like the clip distances (nir_lower_clip_fs). An interpolated cull
 * distance is negative near a single negative vertex too, but the primitive goes only when EVERY
 * vertex is negative. So the vertex shader writes each cull distance as a marker instead, -1/1024
 * for negative and 1 otherwise: interpolated with weights that sum to one, it is negative only
 * where the non-negative vertices weigh (almost) nothing, which is everywhere exactly when all
 * of them are negative. The fragment shader then tests the markers as clip distances.
 */
static bool
panfrost_lower_cull_vs(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const unsigned clip_count = *(const unsigned *)data;

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_CLIP_DIST0 &&
       sem.location != VARYING_SLOT_CLIP_DIST1)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *v = intr->src[0].ssa;
   nir_def *slot = nir_iadd_imm(b, intr->src[1].ssa, sem.location - VARYING_SLOT_CLIP_DIST0);
   nir_def *first = nir_iadd_imm(b, nir_imul_imm(b, slot, 4), nir_intrinsic_component(intr));

   nir_def *comps[4];
   for (unsigned i = 0; i < v->num_components; i++) {
      nir_def *d = nir_channel(b, v, i);
      nir_def *marker = nir_bcsel(b, nir_flt_imm(b, d, 0.0), nir_imm_float(b, -1.0 / 1024.0),
                                  nir_imm_float(b, 1.0));
      nir_def *is_cull = nir_uge_imm(b, nir_iadd_imm(b, first, i), clip_count);
      comps[i] = nir_bcsel(b, is_cull, marker, d);
   }

   nir_src_rewrite(&intr->src[0], nir_vec(b, comps, v->num_components));
   return true;
}

/* Local variables still in memory (arrays indexed dynamically) belong to one
 * invocation, but after nir_lower_workgroup_size several invocations share a
 * hardware thread and its memory. Give each of them its own copy, picked by
 * which of the thread's invocations it is: the local index, which the pass
 * makes per invocation, over the hardware workgroup size.
 */
static bool
panfrost_replicate_temps(nir_shader *nir, unsigned copies, unsigned hw_size)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool any = false;

   nir_foreach_function_temp_variable(var, impl) {
      var->type = glsl_array_type(var->type, copies, 0);
      any = true;
   }

   if (!any)
      return nir_no_progress(impl);

   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var ||
             deref->var->data.mode != nir_var_function_temp)
            continue;

         deref->type = deref->var->type;
         b.cursor = nir_after_instr(instr);
         nir_def *slot =
            nir_udiv_imm(&b, nir_load_local_invocation_index(&b), hw_size);
         nir_deref_instr *elem = nir_build_deref_array(&b, deref, slot);
         nir_def_rewrite_uses_after(&deref->def, &elem->def);
      }
   }

   return nir_progress(true, impl, nir_metadata_control_flow);
}

static void
panfrost_shader_compile(struct panfrost_screen *screen, const nir_shader *ir,
                        struct util_debug_callback *dbg,
                        const struct pan_varying_layout *varying_layout,
                        struct panfrost_shader_key *key, unsigned req_local_mem,
                        struct panfrost_shader_binary *out)
{
   PAN_TRACE_FUNC(PAN_TRACE_GL_SHADER);

   struct panfrost_device *dev = pan_device(&screen->base);

   nir_shader *s = nir_shader_clone(NULL, ir);

   /* The tess control shader runs as a compute job, one workgroup per patch
    * and one invocation per output vertex (pan_gs.h). */
   if (s->info.stage == MESA_SHADER_TESS_CTRL) {
      const unsigned local_size = MAX2(s->info.tess.tcs_vertices_out, 1);
      NIR_PASS(_, s, pan_nir_lower_tcs_inputs);
      NIR_PASS(_, s, poly_nir_lower_tcs, false);
      s->info.stage = MESA_SHADER_COMPUTE;
      memset(&s->info.cs, 0, sizeof(s->info.cs));
      s->info.workgroup_size[0] = local_size;
      s->info.workgroup_size[1] = 1;
      s->info.workgroup_size[2] = 1;
      s->info.workgroup_size_variable = false;
      s->info.outputs_written = 0;
      s->info.inputs_read = 0;
   }

   /* The tess eval shader runs once per tessellator index, as a point-in,
    * point-out geometry shader. */
   if (s->info.stage == MESA_SHADER_TESS_EVAL) {
      NIR_PASS(_, s, poly_nir_lower_tes, false);
      pan_nir_tes_to_gs(s);
   }

   /* A geometry shader runs as a compute job between the vertex and tiler
    * jobs (pan_gs.h), and is compiled as the compute shader it becomes. */
   if (s->info.stage == MESA_SHADER_GEOMETRY) {
      struct pan_gs_info gs;
      pan_gs_gather_info(s, &gs);
      NIR_PASS(_, s, pan_nir_lower_gs, &gs, key->gs.noperspective_varyings);
   }

   /* While graphics shaders are preprocessed at CSO create time, compute
    * kernels are not preprocessed until they're cloned since the driver does
    * not get ownership of the NIR from compute CSOs. Do this preprocessing now.
    * Compute CSOs call this function during create time, so preprocessing
    * happens at CSO create time regardless.
    */
   if (mesa_shader_stage_is_compute(s->info.stage)) {
      pan_preprocess_nir(s, panfrost_device_gpu_id(dev));
   }

   /* Bifrost runs at most 256 invocations per workgroup and GL needs 1024: a
    * larger workgroup runs as 256 invocations that each do the work of
    * several, and is dispatched at that size (panfrost_launch_grid_on_batch).
    */
   unsigned real_workgroup_size = 0;
   if (mesa_shader_stage_is_compute(s->info.stage) && dev->arch >= 6 &&
       !s->info.workgroup_size_variable &&
       s->info.workgroup_size[0] * s->info.workgroup_size[1] *
             s->info.workgroup_size[2] > PAN_MAX_HW_WORKGROUP_SIZE) {
      const unsigned wg_size = s->info.workgroup_size[0] *
                               s->info.workgroup_size[1] *
                               s->info.workgroup_size[2];
      NIR_PASS(_, s, panfrost_replicate_temps,
               DIV_ROUND_UP(wg_size, PAN_MAX_HW_WORKGROUP_SIZE),
               PAN_MAX_HW_WORKGROUP_SIZE);

      /* The pass rewrites the local ID in terms of the local index, and the
       * backend's options ask for the opposite: with both, the sysval lowering
       * inside the pass would trade one for the other forever. The backend
       * gets its local ID back later, from the 1D index. The global ID has to
       * come from the logical local ID too, not from the hardware.
       */
      const nir_shader_compiler_options *options = s->options;
      nir_shader_compiler_options wg_options = *options;
      wg_options.lower_cs_local_index_to_id = false;
      wg_options.has_cs_global_id = false;
      s->options = &wg_options;
      NIR_PASS(_, s, nir_lower_workgroup_size, PAN_MAX_HW_WORKGROUP_SIZE);
      s->options = options;
      NIR_PASS(_, s, nir_lower_compute_system_values, NULL);
      nir_shader_gather_info(s, nir_shader_get_entrypoint(s));
      real_workgroup_size = PAN_MAX_HW_WORKGROUP_SIZE;
   }

   struct pan_compile_inputs inputs = {
      .gpu_id = panfrost_device_gpu_id(dev),
      .gpu_variant = dev->kmod.dev->props.gpu_variant,
   };

   /* Lower this early so the backends don't have to worry about it */
   if (s->info.stage == MESA_SHADER_VERTEX) {
      /* The capture, and a geometry shader, keep the cull distances as they
       * are. */
      unsigned clip_count = s->info.clip_distance_array_size;
      inputs.vs_keep_clip_space = key->vs.keep_clip_space;
      if (s->info.cull_distance_array_size && !key->vs.is_xfb &&
          !key->vs.keep_clip_space) {
         NIR_PASS(_, s, nir_shader_intrinsics_pass, panfrost_lower_cull_vs,
                  nir_metadata_control_flow, &clip_count);
      }

      /* No IDVS for internal XFB shaders */
      inputs.no_idvs = s->info.has_transform_feedback_varyings;

      /* And no IDVS at all on Bifrost. An INDEXED_VERTEX job on a Mali-G76 completes
       * successfully having shaded nothing: the position buffer it should fill stays zero and
       * the tiler bins no primitives. Measured here, in this driver, as a straight A/B on one
       * device -- IDVS on gives an empty frame, BIFROST_MESA_DEBUG=noidvs gives the triangle,
       * pixel-for-pixel what ARM's own driver produces for the same scene.
       *
       * This is not a panvk problem and not a descriptor problem. panvk fails identically, its
       * IDVS descriptors were verified correct field by field, and ARM's driver never emits an
       * INDEXED_VERTEX job on this part at all. Reproducing it in a second, independent driver
       * is what moved it out of "our emission is wrong" and into the hardware/stack.
       *
       * Set PAN_FORCE_IDVS=1 to put it back for anyone chasing the real cause; the failure is
       * silent, so that knob is the only way to reproduce it. Valhall is untouched -- it uses
       * memory-allocated IDVS, a different mechanism. */
      if (pan_arch(panfrost_device_gpu_id(dev)) < 9 &&
          !debug_get_bool_option("PAN_FORCE_IDVS", false))
         inputs.no_idvs = true;

      if (s->info.has_transform_feedback_varyings) {
         static const nir_lower_xfb_to_stores_options xfb_options = {
            .address_format = nir_address_format_64bit_global,
         };

         NIR_PASS(_, s, nir_opt_constant_folding);
         NIR_PASS(_, s, nir_io_add_intrinsic_xfb_info);
         NIR_PASS(_, s, nir_lower_xfb_to_stores, &xfb_options);
      }
   }

   inputs.varying_layout = varying_layout;

   if (s->info.stage == MESA_SHADER_FRAGMENT) {
      inputs.fragcolor_nr_cbufs = key->fs.nr_cbufs_for_fragcolor;

      if (key->fs.sprite_coord_enable) {
         NIR_PASS(_, s, nir_lower_texcoord_replace_late,
                  key->fs.sprite_coord_enable,
                  true /* point coord is sysval */);
         /* Lower load_point_coord if present */
         NIR_PASS(_, s, pan_nir_lower_var_special_pan);
      }

      if (key->fs.clip_plane_enable) {
         NIR_PASS(_, s, nir_lower_clip_fs, key->fs.clip_plane_enable,
                  false, true);
      }

      if (key->fs.emulated_producer) {
         NIR_PASS(_, s, nir_shader_intrinsics_pass, panfrost_move_emulated_inputs,
                  nir_metadata_control_flow, NULL);
      }

      if (key->fs.viewport_clip)
         NIR_PASS(_, s, panfrost_lower_viewport_clip_fs);

      if (key->fs.line_smooth) {
         NIR_PASS(_, s, nir_lower_poly_line_smooth, 16);
         NIR_PASS(_, s, nir_shader_intrinsics_pass,
                  lower_load_poly_line_smooth_enabled,
                  nir_metadata_control_flow, key);
         NIR_PASS(_, s, nir_lower_alu);
      }

      NIR_PASS(_, s, nir_shader_intrinsics_pass,
               lower_sample_mask_writes, nir_metadata_control_flow, NULL);

      if (s->info.fs.accesses_pixel_local_storage)
         NIR_PASS(_, s, panfrost_nir_lower_pls, screen);
   }

   if (dev->arch <= 5 && s->info.stage == MESA_SHADER_FRAGMENT) {
      if (key->fs.nr_cbufs_for_fragcolor) {
         NIR_PASS(_, s, panfrost_nir_remove_fragcolor_stores,
                  key->fs.nr_cbufs_for_fragcolor);
      }

      NIR_PASS(_, s, pan_nir_lower_framebuffer, key->fs.rt_formats,
               pan_raw_format_mask_midgard(key->fs.rt_formats), 0,
               panfrost_device_gpu_prod_id(dev) < 0x700);
   }

   /* Lower resource indices */
   NIR_PASS(_, s, panfrost_nir_lower_res_indices, inputs.gpu_id);

   pan_postprocess_nir(s, &inputs, &out->info);

   if (s->info.stage == MESA_SHADER_VERTEX) {
      NIR_PASS(_, s, nir_inline_sysval,
               nir_intrinsic_load_noperspective_varyings_pan,
               key->vs.noperspective_varyings);
   }

   if (dev->arch >= 9 && mesa_shader_stage_is_compute(s->info.stage)) {
      out->info.cs.allow_merging_workgroups = valhall_can_merge_workgroups(s);
   }

   NIR_PASS(_, s, panfrost_nir_lower_sysvals, dev->arch, &out->sysvals);

   /* For now, we only allow pushing the default UBO 0, and the sysval UBO (if
    * present). Both of these are mapped on the CPU, but other UBOs are not.
    * When we switch to pushing UBOs with a compute kernel (or CSF instructions)
    * we can relax this. */
   assert(s->info.first_ubo_is_default_ubo);
   inputs.fau.pushable_ubos = BITFIELD_BIT(0);

   if (out->sysvals.sysval_count != 0) {
      inputs.fau.pushable_ubos |= BITFIELD_BIT(PAN_UBO_SYSVALS);
   }

   inputs.fau.promote_immediates = true;

   if (dev->arch >= 9) {
      /* Always enable this for GL, it avoids crashes when using unbound
       * resources. */
      inputs.robust_descriptors = true;
   }

   out->binary = UTIL_DYNARRAY_INIT;
   screen->vtbl.compile_shader(s, &inputs, &out->binary, &out->info);

   if (mesa_shader_stage_is_compute(s->info.stage))
      out->info.cs.real_workgroup_size = real_workgroup_size;

   /* Report stats only if we really got the shader compiled */
   if (out->binary.size > 0) {
      if (s->info.stage == MESA_SHADER_VERTEX &&
          out->info.vs.secondary_offset) {
         pan_stats_util_debug(dbg, "MESA_SHADER_POSITION",
                              &out->info.stats);
         pan_stats_util_debug(dbg, "MESA_SHADER_VERTEX",
                              &out->info.stats_idvs_varying);
      } else {
         pan_stats_util_debug(dbg, mesa_shader_stage_name(s->info.stage),
                              &out->info.stats);
      }
   }

   assert(req_local_mem >= out->info.wls_size);
   out->info.wls_size = req_local_mem;

   /* In both clone and tgsi_to_nir paths, the shader is ralloc'd against
    * a NULL context
    */
   ralloc_free(s);
}

static void
panfrost_shader_get(struct pipe_screen *pscreen,
                    struct panfrost_pool *shader_pool,
                    struct panfrost_pool *desc_pool,
                    struct panfrost_uncompiled_shader *uncompiled,
                    struct util_debug_callback *dbg,
                    struct panfrost_compiled_shader *state,
                    unsigned req_local_mem)
{
   struct panfrost_screen *screen = pan_screen(pscreen);
   struct panfrost_device *dev = pan_device(pscreen);

   struct panfrost_shader_binary res = {0};

   /* Try to retrieve the variant from the disk cache. If that fails,
    * compile a new variant and store in the disk cache for later reuse.
    */
   if (!panfrost_disk_cache_retrieve(screen->disk_cache, uncompiled,
                                     &state->key, &res)) {

      /* Only use the varying_layout for FS if the key agrees */
      bool use_layout = uncompiled->nir->info.stage != MESA_SHADER_FRAGMENT ||
                        state->key.fs.vs_varying_layout.known != 0;
      const struct pan_varying_layout *varying_layout =
         use_layout ? &uncompiled->vs_varying_layout : NULL;
      const int64_t t0 = os_time_get_nano();
      panfrost_shader_compile(screen, uncompiled->nir, dbg, varying_layout,
                              &state->key, req_local_mem, &res);
      const double ms = (os_time_get_nano() - t0) / 1e6;
      if (dev->stall_log && ms > 5.0)
         mesa_logi("pan stall: %.1f ms compiling a %s shader variant", ms,
                   mesa_shader_stage_name(uncompiled->nir->info.stage));

      panfrost_disk_cache_store(screen->disk_cache, uncompiled, &state->key,
                                &res);
   }

   state->info = res.info;
   state->sysvals = res.sysvals;

   if (res.binary.size) {
      state->bin = panfrost_pool_take_ref(
         shader_pool,
         pan_pool_upload_aligned(&shader_pool->base, res.binary.data,
                                 res.binary.size, 128));
   }

   util_dynarray_fini(&res.binary);

   /* Don't upload RSD for fragment shaders since they need draw-time
    * merging for e.g. depth/stencil/alpha. RSDs are replaced by simpler
    * shader program descriptors on Valhall, which can be preuploaded even
    * for fragment shaders. */
   bool upload =
      !(uncompiled->nir->info.stage == MESA_SHADER_FRAGMENT && dev->arch <= 7);
   screen->vtbl.prepare_shader(state, desc_pool, upload);

   panfrost_analyze_sysvals(state);
}

static void
panfrost_build_vs_key(struct panfrost_context *ctx,
                      struct panfrost_vs_key *key,
                      struct panfrost_uncompiled_shader *uncompiled)
{
   struct panfrost_uncompiled_shader *fs = ctx->uncompiled[MESA_SHADER_FRAGMENT];

   assert(fs != NULL && "too early");

   /* With a geometry shader, the vertex shader feeds the emulation, which
    * applies the viewport transform and noperspective itself. */
   if (ctx->uncompiled[MESA_SHADER_GEOMETRY] ||
       ctx->uncompiled[MESA_SHADER_TESS_EVAL] ||
       uncompiled->nir->info.next_stage == MESA_SHADER_GEOMETRY ||
       uncompiled->nir->info.next_stage == MESA_SHADER_TESS_CTRL) {
      key->keep_clip_space = true;
      return;
   }

   key->noperspective_varyings = fs->noperspective_varyings;
}

static void
panfrost_build_gs_key(struct panfrost_context *ctx,
                      struct panfrost_gs_key *key)
{
   struct panfrost_uncompiled_shader *fs = ctx->uncompiled[MESA_SHADER_FRAGMENT];

   if (fs)
      key->noperspective_varyings = fs->noperspective_varyings;
}

static void
panfrost_build_fs_key(struct panfrost_context *ctx,
                      struct panfrost_fs_key *key,
                      struct panfrost_uncompiled_shader *uncompiled)
{
   const nir_shader *nir = uncompiled->nir;

   struct panfrost_device *dev = pan_device(ctx->base.screen);
   struct pipe_framebuffer_state *fb = &ctx->pipe_framebuffer;
   struct pipe_rasterizer_state *rast = (void *)ctx->rasterizer;

   /* gl_FragColor lowering needs the number of colour buffers */
   if (uncompiled->fragcolor_lowered) {
      key->nr_cbufs_for_fragcolor = fb->nr_cbufs;
   }

   /* Point sprite lowering needed on Bifrost and newer */
   if (dev->arch >= 6 && rast && ctx->active_prim == MESA_PRIM_POINTS) {
      key->sprite_coord_enable = rast->sprite_coord_enable;
   }

   /* User clip plane lowering needed everywhere */
   if (rast) {
      key->clip_plane_enable = rast->clip_plane_enable;

      /* The last vertex stage's cull distances follow its clip distances, as markers tested
       * like clip distances (panfrost_lower_cull_vs). They have no enables. */
      struct panfrost_uncompiled_shader *vs = ctx->uncompiled[MESA_SHADER_VERTEX];
      struct panfrost_uncompiled_shader *gs = ctx->uncompiled[MESA_SHADER_GEOMETRY];
      if (!gs)
         gs = ctx->uncompiled[MESA_SHADER_TESS_EVAL];
      unsigned clip = 0, cull = 0;
      if (gs) {
         clip = gs->gs.clip_count;
         cull = gs->gs.cull_count;
      } else if (vs) {
         clip = vs->nir->info.clip_distance_array_size;
         cull = vs->nir->info.cull_distance_array_size;
      }
      if (cull) {
         key->clip_plane_enable &= BITFIELD_MASK(clip);
         key->clip_plane_enable |= BITFIELD_RANGE(clip, cull);
      }

      key->viewport_clip = gs && gs->gs.writes_viewport;
      key->emulated_producer = gs != NULL;

      if (u_reduced_prim(ctx->active_prim) == MESA_PRIM_LINES)
         key->line_smooth = rast->line_smooth;
   }

   /* Behind a geometry shader the varyings come from the emulation's own
    * layout, which the fragment shader reads through attribute descriptors. */
   if (!key->clip_plane_enable && !ctx->uncompiled[MESA_SHADER_GEOMETRY] &&
       !ctx->uncompiled[MESA_SHADER_TESS_EVAL])
      key->vs_varying_layout = uncompiled->vs_varying_layout;

   if (dev->arch <= 5) {
      u_foreach_bit(i, (nir->info.outputs_read >> FRAG_RESULT_DATA0)) {
         enum pipe_format fmt = PIPE_FORMAT_R8G8B8A8_UNORM;

         if ((fb->nr_cbufs > i) && fb->cbufs[i].texture)
            fmt = fb->cbufs[i].format;

         if (pan_blendable_formats_v6[fmt].internal)
            fmt = PIPE_FORMAT_NONE;

         key->rt_formats[i] = fmt;
      }
   }
}

static void
panfrost_build_key(struct panfrost_context *ctx,
                   struct panfrost_shader_key *key,
                   struct panfrost_uncompiled_shader *uncompiled)
{
   const nir_shader *nir = uncompiled->nir;

   switch (nir->info.stage) {
   case MESA_SHADER_VERTEX:
      panfrost_build_vs_key(ctx, &key->vs, uncompiled);
      break;
   case MESA_SHADER_FRAGMENT:
      panfrost_build_fs_key(ctx, &key->fs, uncompiled);
      break;
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_TESS_EVAL:
      panfrost_build_gs_key(ctx, &key->gs);
      break;
   default:
      break;
   }
}

static struct panfrost_compiled_shader *
panfrost_new_variant_locked(struct panfrost_context *ctx,
                            struct panfrost_uncompiled_shader *uncompiled,
                            struct panfrost_shader_key *key)
{
   struct panfrost_device *dev = pan_device(ctx->base.screen);
   struct panfrost_compiled_shader *prog = panfrost_alloc_variant(uncompiled);

   *prog = (struct panfrost_compiled_shader){
      .key = *key,
      .stream_output = uncompiled->stream_output,
   };

   panfrost_shader_get(ctx->base.screen, &ctx->shaders, &ctx->descs, uncompiled,
                       &ctx->base.debug, prog, 0);

   prog->earlyzs = pan_earlyzs_analyze(&prog->info, dev->arch);

   return prog;
}

static void
panfrost_bind_shader_state(struct pipe_context *pctx, void *hwcso,
                           mesa_shader_stage type)
{
   struct panfrost_context *ctx = pan_context(pctx);
   ctx->uncompiled[type] = hwcso;
   ctx->prog[type] = NULL;

   ctx->dirty |= PAN_DIRTY_TLS_SIZE;
   ctx->dirty_shader[type] |= PAN_DIRTY_STAGE_SHADER;

   if (hwcso)
      panfrost_update_shader_variant(ctx, type);
}

void
panfrost_update_shader_variant(struct panfrost_context *ctx,
                               mesa_shader_stage type)
{
   /* No shader variants for compute */
   if (type == MESA_SHADER_COMPUTE)
      return;

   /* We need linking information, defer this */
   if ((type == MESA_SHADER_FRAGMENT && !ctx->uncompiled[MESA_SHADER_VERTEX]) ||
       (type == MESA_SHADER_VERTEX && !ctx->uncompiled[MESA_SHADER_FRAGMENT]))
      return;

   /* Also defer, happens with GALLIUM_HUD */
   if (!ctx->uncompiled[type])
      return;

   /* Match the appropriate variant */
   struct panfrost_uncompiled_shader *uncompiled = ctx->uncompiled[type];
   struct panfrost_compiled_shader *compiled = NULL;

   simple_mtx_lock(&uncompiled->lock);

   struct panfrost_shader_key key = {0};
   panfrost_build_key(ctx, &key, uncompiled);

   util_dynarray_foreach(&uncompiled->variants, struct panfrost_compiled_shader,
                         so) {
      if (memcmp(&key, &so->key, sizeof(key)) == 0) {
         compiled = so;
         break;
      }
   }

   if (compiled == NULL)
      compiled = panfrost_new_variant_locked(ctx, uncompiled, &key);

   ctx->prog[type] = compiled;

   simple_mtx_unlock(&uncompiled->lock);
}

static void
panfrost_bind_vs_state(struct pipe_context *pctx, void *hwcso)
{
   panfrost_bind_shader_state(pctx, hwcso, MESA_SHADER_VERTEX);

   /* Fragment shaders are linked with vertex shaders */
   struct panfrost_context *ctx = pan_context(pctx);
   panfrost_update_shader_variant(ctx, MESA_SHADER_FRAGMENT);
}

static void
panfrost_bind_fs_state(struct pipe_context *pctx, void *hwcso)
{
   panfrost_bind_shader_state(pctx, hwcso, MESA_SHADER_FRAGMENT);

   /* Vertex shaders are linked with fragment shaders */
   struct panfrost_context *ctx = pan_context(pctx);
   panfrost_update_shader_variant(ctx, MESA_SHADER_VERTEX);
   panfrost_update_shader_variant(ctx, MESA_SHADER_GEOMETRY);
   panfrost_update_shader_variant(ctx, MESA_SHADER_TESS_EVAL);
}

/* A stage the vertex shader feeds, or the last one before the fragment
 * shader: both of those re-key. */
static void
panfrost_bind_emulated_stage(struct pipe_context *pctx, void *hwcso,
                             mesa_shader_stage stage)
{
   panfrost_bind_shader_state(pctx, hwcso, stage);

   struct panfrost_context *ctx = pan_context(pctx);
   ctx->prog[MESA_SHADER_VERTEX] = NULL;
   ctx->dirty_shader[MESA_SHADER_VERTEX] |= PAN_DIRTY_STAGE_SHADER;
   panfrost_update_shader_variant(ctx, MESA_SHADER_VERTEX);
   ctx->prog[MESA_SHADER_FRAGMENT] = NULL;
   ctx->dirty_shader[MESA_SHADER_FRAGMENT] |= PAN_DIRTY_STAGE_SHADER;
   panfrost_update_shader_variant(ctx, MESA_SHADER_FRAGMENT);
}

static void
panfrost_bind_tcs_state(struct pipe_context *pctx, void *hwcso)
{
   panfrost_bind_emulated_stage(pctx, hwcso, MESA_SHADER_TESS_CTRL);
}

static void
panfrost_bind_tes_state(struct pipe_context *pctx, void *hwcso)
{
   panfrost_bind_emulated_stage(pctx, hwcso, MESA_SHADER_TESS_EVAL);
}

static void
panfrost_bind_gs_state(struct pipe_context *pctx, void *hwcso)
{
   panfrost_bind_shader_state(pctx, hwcso, MESA_SHADER_GEOMETRY);

   /* The vertex shader keeps clip space for it, and the fragment shader
    * reads its layout. */
   struct panfrost_context *ctx = pan_context(pctx);
   ctx->prog[MESA_SHADER_VERTEX] = NULL;
   ctx->dirty_shader[MESA_SHADER_VERTEX] |= PAN_DIRTY_STAGE_SHADER;
   panfrost_update_shader_variant(ctx, MESA_SHADER_VERTEX);
   ctx->prog[MESA_SHADER_FRAGMENT] = NULL;
   ctx->dirty_shader[MESA_SHADER_FRAGMENT] |= PAN_DIRTY_STAGE_SHADER;
   panfrost_update_shader_variant(ctx, MESA_SHADER_FRAGMENT);
}

static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static struct panfrost_shader_key
panfrost_default_shader_key(struct panfrost_uncompiled_shader *so)
{
   struct panfrost_shader_key key = {0};

   /* A vertex shader linked to a geometry shader only ever feeds the
    * emulation, and the linker may have trimmed gl_Position to what the
    * geometry shader reads: it is never viewport-transformed. */
   if (so->nir->info.stage == MESA_SHADER_VERTEX &&
       (so->nir->info.next_stage == MESA_SHADER_GEOMETRY ||
        so->nir->info.next_stage == MESA_SHADER_TESS_CTRL))
      key.vs.keep_clip_space = true;

   if (so->nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* gl_FragColor lowering needs the number of colour buffers on desktop
      * GL, where it acts as an implicit broadcast to all colour buffers.
      *
      * However, gl_FragColor is a legacy feature, so assume that if
      * gl_FragColor is used, there is only a single render target. The
      * implicit broadcast is neither especially useful nor required by GLES.
      */
      if (so->fragcolor_lowered)
         key.fs.nr_cbufs_for_fragcolor = 1;

      key.fs.vs_varying_layout = so->vs_varying_layout;
   }

   return key;
}

static void *
panfrost_create_shader_state(struct pipe_context *pctx,
                             const struct pipe_shader_state *cso)
{
   PAN_TRACE_FUNC(PAN_TRACE_GL_SHADER);

   nir_shader *nir = (cso->type == PIPE_SHADER_IR_TGSI)
                        ? tgsi_to_nir(cso->tokens, pctx->screen, false)
                        : cso->ir.nir;

   struct panfrost_uncompiled_shader *so = panfrost_alloc_shader(nir);

   /* The driver gets ownership of the nir_shader for graphics. The NIR is
    * ralloc'd. Free the NIR when we free the uncompiled shader.
    */
   ralloc_steal(so, nir);

   so->stream_output = cso->stream_output;
   so->nir = nir;

   /* PLS lowering is not taken care of by glsl_to_nir(), so do it here. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       nir->info.fs.accesses_pixel_local_storage) {
      /* Try to optimize the case where inout PLS vars are never
       * read/written to. Needs to be called before
       * nir_lower_io_vars_to_temporaries() because the copy_derefs
       * inserted there prevent us from detecting PLS usage.
       */
      NIR_PASS(_, nir, nir_downgrade_pls_vars);

      /* Lower PLS vars to temporaries before we lower IOs. */
      NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
               nir_shader_get_entrypoint(nir), nir_var_any_pixel_local);

      /* We need to lower all the copy_deref's introduced by lower_io_to-
       * _temporaries before calling nir_lower_io.
       */
      NIR_PASS(_, nir, nir_split_var_copies);
      NIR_PASS(_, nir, nir_lower_var_copies);
      NIR_PASS(_, nir, nir_lower_global_vars_to_local);

      /* Lower all PLS IOs. */
      NIR_PASS(_, nir, nir_lower_io, nir_var_any_pixel_local, glsl_type_size,
               0);

      /* Lower and remove dead derefs and variables to clean up the IR. */
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);

      /* Re-run gather_info() to get the latest accesses_pixel_local_storage
       * state.
       */
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   }

   struct panfrost_device *dev = pan_device(pctx->screen);

   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       nir->info.outputs_written & BITFIELD_BIT(FRAG_RESULT_COLOR)) {
      /* Bifrost and newer handle gl_FragColor in the backend, but Midgard
       * still needs the frontend lowering.  Must happen before IO lowering
       */
      if (dev->arch <= 5) {
         assert(!nir->info.io_lowered);
         NIR_PASS(_, nir, nir_lower_fragcolor,
                  nir->info.fs.color_is_dual_source ? 1 : 8);
      }
      so->fragcolor_lowered = true;
   }

   /* Then run the suite of lowering and optimization, including I/O lowering */
   pan_preprocess_nir(nir, panfrost_device_gpu_id(dev));

   /* Usually IO is lowered by gallium, but TTN doesn't */
   if (!nir->info.io_lowered) {
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
               nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
      NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
               glsl_type_size, nir_lower_io_use_interpolated_input_intrinsics);
      /* nir_lower_io just computes offsets based on the original deref and
       * lower_indirect_derefs ensures that the array derefs have a constant
       * index.  Constant-fold to get us actual constants in in load/store
       * instructions.
       */
      NIR_PASS(_, nir, nir_opt_constant_folding);
   }

   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      so->noperspective_varyings =
         pan_nir_collect_noperspective_varyings_fs(nir);

   if (nir->info.stage == MESA_SHADER_GEOMETRY ||
       nir->info.stage == MESA_SHADER_TESS_EVAL)
      pan_gs_gather_info(nir, &so->gs);

   if (nir->info.stage == MESA_SHADER_TESS_CTRL ||
       nir->info.stage == MESA_SHADER_TESS_EVAL) {
      so->tess.prim = nir->info.tess._primitive_mode;
      so->tess.spacing = nir->info.tess.spacing;
      so->tess.ccw = nir->info.tess.ccw;
      so->tess.point_mode = nir->info.tess.point_mode;
   }

   if (nir->info.stage == MESA_SHADER_TESS_CTRL) {
      so->tess.vertices_out = nir->info.tess.tcs_vertices_out;
      so->tess.per_vertex_outputs = poly_tcs_per_vertex_outputs(nir);
      so->tess.nr_patch_outputs = util_last_bit(nir->info.patch_outputs_written);
      so->tess.output_stride = poly_tcs_output_stride(nir);
   }

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      struct pan_varying_layout *varying_layout = &so->vs_varying_layout;
      pan_varying_collect_formats(varying_layout, nir,
                                  panfrost_device_gpu_id(dev));
      pan_build_varying_layout_compact(varying_layout, nir,
                                       panfrost_device_gpu_id(dev));
   }

   /* If this shader uses transform feedback, compile the transform
    * feedback program. This is a special shader variant.
    */
   struct panfrost_context *ctx = pan_context(pctx);

   if (so->nir->xfb_info && nir->info.stage == MESA_SHADER_VERTEX) {
      so->xfb = calloc(1, sizeof(struct panfrost_compiled_shader));
      so->xfb->key.vs.is_xfb = true;

      panfrost_shader_get(ctx->base.screen, &ctx->shaders, &ctx->descs, so,
                          &ctx->base.debug, so->xfb, 0);

      /* Since transform feedback is handled via the transform
       * feedback program, the original program no longer uses XFB
       */
      pan_nir_remove_xfb(nir);
   }

   /* If we're not using separate shaders, the FS can use VS varying_layout to
    * optimize loads (LD_VAR_BUF instead of LD_VAR).  Gallium won't provide us
    * with the VS directly, so we need to delay the default variant compilation
    * until link time
    */
   if (nir->info.stage == MESA_SHADER_FRAGMENT && !nir->info.separate_shader)
      return so;

   /* Compile the program. We don't use vertex shader keys, so there will
    * be no further vertex shader variants. We do have fragment shader
    * keys, but we can still compile with a default key that will work most
    * of the time.
    */
   struct panfrost_shader_key key = panfrost_default_shader_key(so);

   /* Creating a CSO is single-threaded, so it's ok to use the
    * locked function without explicitly taking the lock. Creating a
    * default variant acts as a precompile.
    */
   panfrost_new_variant_locked(ctx, so, &key);

   return so;
}

static void
panfrost_delete_shader_state(struct pipe_context *pctx, void *so)
{
   struct panfrost_uncompiled_shader *cso =
      (struct panfrost_uncompiled_shader *)so;

   util_dynarray_foreach(&cso->variants, struct panfrost_compiled_shader, so) {
      panfrost_bo_unreference(so->bin.bo);
      panfrost_bo_unreference(so->state.bo);
      panfrost_bo_unreference(so->linkage.bo);
   }

   if (cso->xfb) {
      panfrost_bo_unreference(cso->xfb->bin.bo);
      panfrost_bo_unreference(cso->xfb->state.bo);
      panfrost_bo_unreference(cso->xfb->linkage.bo);
      free(cso->xfb);
   }

   simple_mtx_destroy(&cso->lock);

   ralloc_free(so);
}

static void
panfrost_link_shader(struct pipe_context *pctx, void** handles)
{
   struct panfrost_context *ctx = pan_context(pctx);
   struct panfrost_uncompiled_shader *vs = handles[MESA_SHADER_VERTEX];
   struct panfrost_uncompiled_shader *fs = handles[MESA_SHADER_FRAGMENT];

   if (!fs || fs->nir->info.separate_shader)
      return;

   /* Behind a geometry shader the fragment shader reads the emulation's
    * layout through attribute descriptors: nothing to link here. */
   if (handles[MESA_SHADER_GEOMETRY] || handles[MESA_SHADER_TESS_EVAL])
      return;

   /* We only handle VS and FS for now, it's not clear how varying layout will
    * fit when more shader types are supported.  So assert those are the only
    * shaders present.
    */
   for (unsigned i = 0; i < MESA_SHADER_MESH_STAGES; i++) {
      if (i != MESA_SHADER_VERTEX && i != MESA_SHADER_FRAGMENT)
         assert(handles[i] == NULL);
   }

   /* Only copy the varying layout if we have a VS, sometimes we don't have one
    * (e.g. fixed-function VS), in those cases we just compile a default FS.
    */
   if (vs)
      fs->vs_varying_layout = vs->vs_varying_layout;

   simple_mtx_lock(&fs->lock);

   struct panfrost_shader_key key = panfrost_default_shader_key(fs);
   panfrost_new_variant_locked(ctx, fs, &key);

   simple_mtx_unlock(&fs->lock);
}

/*
 * Create a compute CSO. As compute kernels do not require variants, they are
 * precompiled, creating both the uncompiled and compiled shaders now.
 */
static void *
panfrost_create_compute_state(struct pipe_context *pctx,
                              const struct pipe_compute_state *cso)
{
   struct panfrost_context *ctx = pan_context(pctx);
   struct panfrost_uncompiled_shader *so = panfrost_alloc_shader(cso->prog);
   struct panfrost_compiled_shader *v = panfrost_alloc_variant(so);
   memset(v, 0, sizeof *v);

   assert(cso->ir_type == PIPE_SHADER_IR_NIR && "TGSI kernels unsupported");

   panfrost_shader_get(pctx->screen, &ctx->shaders, &ctx->descs, so,
                       &ctx->base.debug, v, cso->static_shared_mem);

   /* The NIR becomes invalid after this. For compute kernels, we never
    * need to access it again. Don't keep a dangling pointer around.
    */
   ralloc_free((void *)so->nir);
   so->nir = NULL;

   return so;
}

static void
panfrost_bind_compute_state(struct pipe_context *pipe, void *cso)
{
   struct panfrost_context *ctx = pan_context(pipe);
   struct panfrost_uncompiled_shader *uncompiled = cso;

   ctx->uncompiled[MESA_SHADER_COMPUTE] = uncompiled;

   ctx->prog[MESA_SHADER_COMPUTE] =
      uncompiled ? util_dynarray_begin(&uncompiled->variants) : NULL;
}

static void
panfrost_get_compute_state_info(struct pipe_context *pipe, void *cso,
                                struct pipe_compute_state_object_info *info)
{
   struct panfrost_device *dev = pan_device(pipe->screen);
   struct panfrost_uncompiled_shader *uncompiled = cso;
   struct panfrost_compiled_shader *cs =
      util_dynarray_begin(&uncompiled->variants);

   info->max_threads = pan_compute_max_thread_count(&dev->kmod.dev->props,
                                                    cs->info.work_reg_count);
   info->private_memory = cs->info.tls_size;
   info->simd_sizes = pan_subgroup_size(dev->arch);
   info->preferred_simd_size = info->simd_sizes;
}

void
panfrost_shader_context_init(struct pipe_context *pctx)
{
   pctx->create_vs_state = panfrost_create_shader_state;
   pctx->delete_vs_state = panfrost_delete_shader_state;
   pctx->bind_vs_state = panfrost_bind_vs_state;

   pctx->create_tcs_state = panfrost_create_shader_state;
   pctx->delete_tcs_state = panfrost_delete_shader_state;
   pctx->bind_tcs_state = panfrost_bind_tcs_state;

   pctx->create_tes_state = panfrost_create_shader_state;
   pctx->delete_tes_state = panfrost_delete_shader_state;
   pctx->bind_tes_state = panfrost_bind_tes_state;

   pctx->create_gs_state = panfrost_create_shader_state;
   pctx->delete_gs_state = panfrost_delete_shader_state;
   pctx->bind_gs_state = panfrost_bind_gs_state;

   pctx->create_fs_state = panfrost_create_shader_state;
   pctx->delete_fs_state = panfrost_delete_shader_state;
   pctx->bind_fs_state = panfrost_bind_fs_state;

   pctx->link_shader = panfrost_link_shader;

   pctx->create_compute_state = panfrost_create_compute_state;
   pctx->bind_compute_state = panfrost_bind_compute_state;
   pctx->get_compute_state_info = panfrost_get_compute_state_info;
   pctx->delete_compute_state = panfrost_delete_shader_state;
}
