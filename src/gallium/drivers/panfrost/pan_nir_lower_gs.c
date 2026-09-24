/*
 * Copyright 2026 XclipseTools
 * SPDX-License-Identifier: MIT
 */

/*
 * Geometry shaders as a compute job, see pan_gs.h. This is panvk's emulation
 * (src/panfrost/vulkan/panvk_nir_lower_gs.c) for gallium: the IO arrives lowered, so the outputs
 * are store_output intrinsics that this pass collects into per-slot variables, which EmitVertex
 * then writes out; and the per-draw parameters come from memory behind a sysval rather than from
 * push constants.
 *
 * Two hardware facts carried over from panvk: multi-component global loads and stores do not
 * reach memory correctly here, so every access is one 32-bit component at a time; and a vertex
 * shader's position reaches memory through the SNAP_4 varying format, so the positions written
 * here are snapped to 1/256 px the same way.
 */

#include "pan_gs.h"

#include "nir.h"
#include "nir_builder.h"

void
pan_gs_gather_info(const nir_shader *nir, struct pan_gs_info *info)
{
   memset(info, 0, sizeof(*info));

   if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
      /* Run as a point-in, point-out geometry shader (pan_nir_tes_to_gs). */
      info->input_prim = info->output_prim = MESA_PRIM_POINTS;
      info->input_verts = info->output_verts = 1;
      info->max_verts = info->max_prims = info->invocations = 1;
   } else {
      info->input_prim = nir->info.gs.input_primitive;
      info->output_prim = nir->info.gs.output_primitive;
      info->input_verts = mesa_vertices_per_prim(nir->info.gs.input_primitive);
      info->output_verts = mesa_vertices_per_prim(nir->info.gs.output_primitive);
      info->max_verts = nir->info.gs.vertices_out;
      info->max_prims = info->max_verts >= info->output_verts
                           ? info->max_verts - (info->output_verts - 1)
                           : 0;
      info->invocations = MAX2(nir->info.gs.invocations, 1);
   }

   const uint64_t written = nir->info.outputs_written;
   info->writes_psiz = written & VARYING_BIT_PSIZ;
   info->writes_viewport = written & VARYING_BIT_VIEWPORT;

   /* Everything the fragment shader may read as an ordinary varying. Position and point size
    * have buffers of their own, gl_ViewportIndex and gl_Layer travel in slots of their own
    * (PAN_GS_VIEWPORT_SLOT), and the primitive ID has no path to the fragment shader here. */
   info->generic_mask =
      written & ~(VARYING_BIT_POS | VARYING_BIT_PSIZ | VARYING_BIT_PRIMITIVE_ID |
                  VARYING_BIT_EDGE | VARYING_BIT_CLIP_VERTEX | VARYING_BIT_VIEWPORT |
                  VARYING_BIT_LAYER | BITFIELD64_BIT(PAN_GS_VIEWPORT_SLOT) |
                  BITFIELD64_BIT(PAN_GS_LAYER_SLOT));
   if (written & VARYING_BIT_VIEWPORT)
      info->generic_mask |= BITFIELD64_BIT(PAN_GS_VIEWPORT_SLOT);
   if (written & VARYING_BIT_LAYER)
      info->generic_mask |= BITFIELD64_BIT(PAN_GS_LAYER_SLOT);
   info->out_stride = 16 * MAX2(util_bitcount64(info->generic_mask), 1);

   info->clip_count = nir->info.clip_distance_array_size;
   info->cull_count = nir->info.cull_distance_array_size;
}

/* One 32-bit component at a time, see the top of the file. */
static void
store_components(nir_builder *b, nir_def *base, nir_def *byte_off, nir_def *val)
{
   for (unsigned c = 0; c < val->num_components; c++) {
      nir_def *addr = nir_iadd(b, base, nir_u2u64(b, nir_iadd_imm(b, byte_off, c * 4)));
      nir_store_global(b, nir_channel(b, val, c), addr, .align_mul = 4, .write_mask = 0x1);
   }
}

static nir_def *
load_components(nir_builder *b, nir_def *base, nir_def *byte_off, unsigned num_components,
                unsigned bit_size)
{
   const unsigned sz = bit_size / 8;
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];

   for (unsigned c = 0; c < num_components; c++) {
      nir_def *addr = nir_iadd(b, base, nir_u2u64(b, nir_iadd_imm(b, byte_off, c * sz)));
      comps[c] = nir_load_global(b, 1, bit_size, addr, .align_mul = sz);
   }

   return nir_vec(b, comps, num_components);
}

struct lower_gs_state {
   const struct pan_gs_info *info;
   uint32_t noperspective;

   nir_def *params;
   nir_def *in_pos, *in_psiz, *in_general, *in_offsets, *in_general_stride;
   nir_def *out_pos, *out_psiz, *out_general, *out_index;
   nir_def *index_buf, *index_size, *topology, *instance_stride, *prims_per_instance;
   nir_def *index_bias, *vertex_count, *flatshade_first, *dead_vertex;
   /* Vertices per input primitive when info->input_verts is 0 (a tess control patch). */
   nir_def *vpp_dyn;

   nir_def *prim_id, *invocation_id, *instance;
   nir_def *out_vtx_base, *out_idx_base;
   nir_variable *prims_written;

   /* The current value of every output slot, 4 x 32-bit. */
   nir_variable *outputs[64];
};

static nir_def *
param_u32(nir_builder *b, struct lower_gs_state *st, unsigned offset)
{
   return nir_load_global(b, 1, 32, nir_iadd_imm(b, st->params, offset), .align_mul = 4);
}

static nir_def *
param_u64(nir_builder *b, struct lower_gs_state *st, unsigned offset)
{
   return nir_pack_64_2x32_split(b, param_u32(b, st, offset), param_u32(b, st, offset + 4));
}

#define PARAM32(b, st, field) param_u32(b, st, offsetof(struct pan_gs_params, field))
#define PARAM64(b, st, field) param_u64(b, st, offsetof(struct pan_gs_params, field))

/* The input side of the parameters, shared by the geometry and tess control lowerings. */
static void
load_input_params(nir_builder *b, struct lower_gs_state *st)
{
   st->params = nir_load_geometry_param_buffer_poly(b);
   st->in_pos = PARAM64(b, st, in_pos);
   st->in_psiz = PARAM64(b, st, in_psiz);
   st->in_general = PARAM64(b, st, in_general);
   st->in_offsets = PARAM64(b, st, in_offsets);
   st->index_buf = PARAM64(b, st, index_buf);
   st->in_general_stride = PARAM32(b, st, in_general_stride);
   st->index_size = PARAM32(b, st, index_size);
   st->topology = PARAM32(b, st, topology);
   st->prims_per_instance = PARAM32(b, st, prims_per_instance);
   st->instance_stride = PARAM32(b, st, instance_stride);
   st->index_bias = PARAM32(b, st, index_bias);
   st->vertex_count = PARAM32(b, st, vertex_count);
   st->vpp_dyn = PARAM32(b, st, patch_size);
}

/* Clip space to what the tiler takes: screen-space xyz and 1/w, the arithmetic and clamp of
 * nir_lower_viewport_transform, snapped like SNAP_4. */
static nir_def *
clip_to_screen(nir_builder *b, nir_def *p, nir_def *scale, nir_def *offset)
{
   nir_def *w_recip = nir_frcp(b, nir_channel(b, p, 3));
   w_recip = nir_fclamp(b, w_recip, nir_imm_float(b, -32768.0f), nir_imm_float(b, 32768.0f));

   nir_def *ndc = nir_fmul(b, nir_trim_vector(b, p, 3), nir_replicate(b, w_recip, 3));
   nir_def *screen = nir_fadd(b, nir_fmul(b, ndc, scale), offset);

   nir_def *snap_x =
      nir_fmul_imm(b, nir_fround_even(b, nir_fmul_imm(b, nir_channel(b, screen, 0), 256.0)),
                   1.0 / 256.0);
   nir_def *snap_y =
      nir_fmul_imm(b, nir_fround_even(b, nir_fmul_imm(b, nir_channel(b, screen, 1), 256.0)),
                   1.0 / 256.0);

   return nir_vec4(b, snap_x, snap_y, nir_channel(b, screen, 2), w_recip);
}

/* Which of the vertex shader's varying records is vertex i of this invocation's input
 * primitive: primitive assembly, then the index buffer, then the instance. */
static nir_def *
gs_input_record(nir_builder *b, struct lower_gs_state *st, nir_def *i)
{
   const unsigned vpp = st->info->input_verts;
   nir_def *p = st->prim_id;

   /* LIST: p * vpp + i. A patch is only ever a list. */
   nir_def *list = vpp ? nir_iadd(b, nir_imul_imm(b, p, vpp), i)
                       : nir_iadd(b, nir_imul(b, p, st->vpp_dyn), i);

   /* STRIP: p + i. An odd triangle of a strip takes p, p + 2, p + 1: GL's order for it keeps the
    * winding reversed, which is all the geometry shader can see. */
   nir_def *strip_i = i;
   if (vpp == 3) {
      nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, p, 1), 0);
      nir_def *swapped = nir_bcsel(b, nir_ieq_imm(b, i, 1), nir_imm_int(b, 2),
                                   nir_bcsel(b, nir_ieq_imm(b, i, 2), nir_imm_int(b, 1), i));
      strip_i = nir_bcsel(b, odd, swapped, i);
   }
   nir_def *strip = nir_iadd(b, p, strip_i);

   /* FAN: p + 1, p + 2, 0 */
   nir_def *fan =
      nir_bcsel(b, nir_ieq_imm(b, i, 2), nir_imm_int(b, 0), nir_iadd(b, p, nir_iadd_imm(b, i, 1)));

   /* LOOP: p + i, wrapping to 0 after the last vertex. */
   nir_def *loop_v = nir_iadd(b, p, i);
   nir_def *loop = nir_bcsel(b, nir_uge(b, loop_v, st->vertex_count), nir_imm_int(b, 0), loop_v);

   /* TRIANGLE STRIP WITH ADJACENCY, in the shader's order (v0, a01, v1, a12, v2, a20):
    *    even p:  2p, first ? 1 : 2p-2,  2p+2, last ? 2p+5 : 2p+6,  2p+4, 2p+3
    *    odd p:   2p+2, 2p-2,  2p, 2p+3,  2p+4, last ? 2p+5 : 2p+6 */
   nir_def *tsa = list;
   if (vpp == 6) {
      nir_def *p2 = nir_imul_imm(b, p, 2);
      nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, p, 1), 0);
      nir_def *first = nir_ieq_imm(b, p, 0);
      nir_def *last = nir_ieq(b, p, nir_iadd_imm(b, st->prims_per_instance, -1));
      nir_def *far = nir_bcsel(b, last, nir_iadd_imm(b, p2, 5), nir_iadd_imm(b, p2, 6));
      nir_def *even_v[6] = {
         p2, nir_bcsel(b, first, nir_imm_int(b, 1), nir_iadd_imm(b, p2, -2)),
         nir_iadd_imm(b, p2, 2), far,
         nir_iadd_imm(b, p2, 4), nir_iadd_imm(b, p2, 3),
      };
      nir_def *odd_v[6] = {
         nir_iadd_imm(b, p2, 2), nir_iadd_imm(b, p2, -2),
         p2, nir_iadd_imm(b, p2, 3),
         nir_iadd_imm(b, p2, 4), far,
      };
      nir_def *pick = nir_bcsel(b, odd, odd_v[5], even_v[5]);
      for (int k = 4; k >= 0; k--)
         pick = nir_bcsel(b, nir_ieq_imm(b, i, k), nir_bcsel(b, odd, odd_v[k], even_v[k]), pick);
      tsa = pick;
   }

   nir_def *seq = !vpp ? list :
      nir_bcsel(b, nir_ieq_imm(b, st->topology, PAN_GS_TOPO_STRIP), strip,
      nir_bcsel(b, nir_ieq_imm(b, st->topology, PAN_GS_TOPO_FAN), fan,
      nir_bcsel(b, nir_ieq_imm(b, st->topology, PAN_GS_TOPO_LOOP), loop,
      nir_bcsel(b, nir_ieq_imm(b, st->topology, PAN_GS_TOPO_TRI_STRIP_ADJ), tsa, list))));

   /* Through the index buffer, when there is one: the aligned 32-bit word holding the index,
    * then the index out of it. */
   nir_def *vtx = seq;
   nir_push_if(b, nir_ine_imm(b, st->index_size, 0));
   {
      nir_def *addr = nir_iadd(b, st->index_buf, nir_u2u64(b, nir_imul(b, seq, st->index_size)));
      nir_def *word = nir_load_global(b, 1, 32, nir_iand_imm(b, addr, ~3ull), .align_mul = 4);
      nir_def *shift = nir_imul_imm(b, nir_u2u32(b, nir_iand_imm(b, addr, 3)), 8);
      nir_def *shifted = nir_ushr(b, word, shift);
      nir_def *idx = nir_bcsel(b, nir_ieq_imm(b, st->index_size, 1), nir_iand_imm(b, shifted, 0xff),
                               nir_bcsel(b, nir_ieq_imm(b, st->index_size, 2),
                                         nir_iand_imm(b, shifted, 0xffff), word));
      vtx = nir_isub(b, idx, st->index_bias);
   }
   nir_pop_if(b, NULL);
   vtx = nir_if_phi(b, vtx, seq);

   return nir_iadd(b, vtx, nir_imul(b, st->instance, st->instance_stride));
}

/* Converts a loaded or stored value between bit sizes the way its type says. */
static nir_def *
convert_bits(nir_builder *b, nir_def *v, nir_alu_type type, unsigned bit_size)
{
   if (v->bit_size == bit_size)
      return v;

   switch (nir_alu_type_get_base_type(type)) {
   case nir_type_float:
      return nir_f2fN(b, v, bit_size);
   case nir_type_int:
      return nir_i2iN(b, v, bit_size);
   default:
      return nir_u2uN(b, v, bit_size);
   }
}

static nir_def *
lower_per_vertex_input(nir_builder *b, nir_intrinsic_instr *intr, struct lower_gs_state *st)
{
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   const unsigned location =
      sem.location + (nir_src_is_const(intr->src[1]) ? nir_src_as_uint(intr->src[1]) : 0);
   const unsigned first = nir_intrinsic_component(intr);
   const unsigned n = intr->def.num_components;
   const nir_alu_type type = nir_intrinsic_dest_type(intr);

   nir_def *vtx = gs_input_record(b, st, intr->src[0].ssa);

   if (location == VARYING_SLOT_POS) {
      nir_def *pos = load_components(b, st->in_pos, nir_imul_imm(b, vtx, 16), 4, 32);
      nir_def *c[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < n; i++)
         c[i] = nir_channel(b, pos, MIN2(first + i, 3));
      return convert_bits(b, nir_vec(b, c, n), type, intr->def.bit_size);
   }

   if (location == VARYING_SLOT_PSIZ) {
      nir_push_if(b, nir_ine_imm(b, st->in_psiz, 0));
      nir_def *psiz = load_components(b, st->in_psiz, nir_imul_imm(b, vtx, 2), 1, 16);
      nir_pop_if(b, NULL);
      psiz = nir_if_phi(b, nir_f2f32(b, psiz), nir_imm_float(b, 1.0f));
      return convert_bits(b, nir_replicate(b, psiz, n), nir_type_float, intr->def.bit_size);
   }

   /* A slot the vertex shader does not write reads as zero and is not loaded at all. */
   nir_def *entry = location < 64 ? nir_load_global(b, 1, 32,
                                                    nir_iadd_imm(b, st->in_offsets, location * 4),
                                                    .align_mul = 4)
                                  : nir_imm_int(b, PAN_GS_SLOT_UNUSED);
   nir_def *is16 = nir_ine_imm(b, nir_iand_imm(b, entry, PAN_GS_IN_16BIT), 0);
   nir_def *offset = nir_iand_imm(b, entry, ~PAN_GS_IN_16BIT);
   nir_def *rec = nir_iadd(b, nir_imul(b, vtx, st->in_general_stride), offset);

   nir_push_if(b, nir_ine_imm(b, entry, PAN_GS_SLOT_UNUSED));
   nir_push_if(b, is16);
   nir_def *v16 =
      load_components(b, st->in_general, nir_iadd_imm(b, rec, first * 2), n, 16);
   v16 = convert_bits(b, v16, type, intr->def.bit_size);
   nir_push_else(b, NULL);
   nir_def *v32 =
      load_components(b, st->in_general, nir_iadd_imm(b, rec, first * 4), n, 32);
   v32 = convert_bits(b, v32, type, intr->def.bit_size);
   nir_pop_if(b, NULL);
   nir_def *val = nir_if_phi(b, v16, v32);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, val, nir_imm_zero(b, n, intr->def.bit_size));
}

/* store_output -> the slot's variable, as 32-bit components. */
static void
lower_store_output(nir_builder *b, nir_intrinsic_instr *intr, struct lower_gs_state *st)
{
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   const unsigned location =
      sem.location + (nir_src_is_const(intr->src[1]) ? nir_src_as_uint(intr->src[1]) : 0);
   if (location >= 64 || !st->outputs[location])
      return;

   nir_def *v = convert_bits(b, intr->src[0].ssa, nir_intrinsic_src_type(intr), 32);
   const unsigned first = nir_intrinsic_component(intr);
   const unsigned mask = nir_intrinsic_write_mask(intr);

   nir_def *comps[4];
   nir_def *undef = nir_undef(b, 1, 32);
   for (unsigned c = 0; c < 4; c++) {
      comps[c] = (c >= first && c - first < v->num_components) ? nir_channel(b, v, c - first)
                                                               : undef;
   }
   nir_store_var(b, st->outputs[location], nir_vec(b, comps, 4), (mask << first) & 0xf);
}

/* EmitVertex(): the whole current vertex into this invocation's slot. */
static void
lower_emit_vertex(nir_builder *b, nir_intrinsic_instr *intr, struct lower_gs_state *st)
{
   const struct pan_gs_info *info = st->info;
   nir_def *counter = intr->src[0].ssa;

   /* A shader that emits more than it declared must not write into the next slot. */
   nir_push_if(b, nir_ult_imm(b, counter, info->max_verts));
   {
      nir_def *slot = nir_iadd(b, st->out_vtx_base, counter);
      nir_def *clip = st->outputs[VARYING_SLOT_POS]
                         ? nir_load_var(b, st->outputs[VARYING_SLOT_POS])
                         : nir_imm_vec4(b, 0, 0, 0, 1);

      nir_def *vp_scale, *vp_offset;
      if (info->writes_viewport && st->outputs[VARYING_SLOT_VIEWPORT]) {
         /* The transform of the viewport this vertex names, clamped: an index past the last
          * viewport is undefined, a load past the table is not allowed. */
         nir_def *idx =
            nir_umin(b, nir_channel(b, nir_load_var(b, st->outputs[VARYING_SLOT_VIEWPORT]), 0),
                     nir_imm_int(b, PAN_GS_MAX_VIEWPORTS - 1));
         nir_def *xf = nir_iadd(b, PARAM64(b, st, viewports), nir_u2u64(b, nir_imul_imm(b, idx, 24)));
         vp_scale = load_components(b, xf, nir_imm_int(b, 0), 3, 32);
         vp_offset = load_components(b, xf, nir_imm_int(b, 12), 3, 32);
      } else {
         vp_scale = nir_load_viewport_scale(b);
         vp_offset = nir_load_viewport_offset(b);
      }
      store_components(b, st->out_pos, nir_imul_imm(b, slot, 16),
                       clip_to_screen(b, clip, vp_scale, vp_offset));

      if (info->writes_psiz && st->outputs[VARYING_SLOT_PSIZ]) {
         nir_def *psiz = nir_channel(b, nir_load_var(b, st->outputs[VARYING_SLOT_PSIZ]), 0);
         nir_push_if(b, nir_ine_imm(b, st->out_psiz, 0));
         nir_store_global(b, nir_f2f16(b, psiz),
                          nir_iadd(b, st->out_psiz, nir_u2u64(b, nir_imul_imm(b, slot, 2))),
                          .align_mul = 2, .write_mask = 0x1);
         nir_pop_if(b, NULL);
      }

      nir_def *w = nir_channel(b, clip, 3);
      u_foreach_bit64(loc, info->generic_mask) {
         nir_variable *var = loc == PAN_GS_VIEWPORT_SLOT ? st->outputs[VARYING_SLOT_VIEWPORT]
                             : loc == PAN_GS_LAYER_SLOT  ? st->outputs[VARYING_SLOT_LAYER]
                                                         : st->outputs[loc];
         if (!var)
            continue;

         nir_def *val = nir_load_var(b, var);

         /* Cull distances become markers, see panfrost_lower_cull_vs. */
         if ((loc == VARYING_SLOT_CLIP_DIST0 || loc == VARYING_SLOT_CLIP_DIST1) &&
             info->cull_count) {
            nir_def *c[4];
            for (unsigned k = 0; k < 4; k++) {
               const unsigned idx = (loc - VARYING_SLOT_CLIP_DIST0) * 4 + k;
               c[k] = nir_channel(b, val, k);
               if (idx >= info->clip_count && idx < info->clip_count + info->cull_count) {
                  c[k] = nir_bcsel(b, nir_flt_imm(b, c[k], 0.0),
                                   nir_imm_float(b, -1.0 / 1024.0), nir_imm_float(b, 1.0));
               }
            }
            val = nir_vec(b, c, 4);
         }

         /* The tiler interpolates with perspective; a noperspective varying is premultiplied
          * by w, as pan_nir_lower_noperspective_vs does for a vertex shader. */
         if (loc >= VARYING_SLOT_VAR0 &&
             (st->noperspective & BITFIELD_BIT(loc - VARYING_SLOT_VAR0)))
            val = nir_fmul(b, val, nir_replicate(b, w, 4));

         nir_def *byte = nir_iadd_imm(b, nir_imul_imm(b, slot, info->out_stride),
                                      pan_gs_out_offset(info, loc));
         store_components(b, st->out_general, byte, val);
      }

      /* Points: nir_lower_gs_intrinsics removes EndPrimitive, every vertex is a point. */
      if (info->output_verts == 1) {
         nir_def *addr = nir_iadd(
            b, st->out_index,
            nir_u2u64(b, nir_imul_imm(b, nir_iadd(b, st->out_idx_base, counter), 4)));
         nir_store_global(b, slot, addr, .align_mul = 4, .write_mask = 0x1);
      }
   }
   nir_pop_if(b, NULL);
}

/* EndPrimitive(): the index list turning the strip just emitted into independent primitives.
 * src[0] is the running vertex total of this invocation, src[1] the vertices of this strip. */
static void
lower_end_primitive(nir_builder *b, nir_intrinsic_instr *intr, struct lower_gs_state *st)
{
   const unsigned vpp = st->info->output_verts;
   nir_def *vtx_start = nir_isub(b, intr->src[0].ssa, intr->src[1].ssa);
   nir_def *nprims =
      nir_imax(b, nir_iadd_imm(b, intr->src[1].ssa, -(int)(vpp - 1)), nir_imm_int(b, 0));
   nir_def *prim_base = nir_load_var(b, st->prims_written);

   nir_variable *i = nir_local_variable_create(b->impl, glsl_uint_type(), "gs_strip_i");
   nir_store_var(b, i, nir_imm_int(b, 0), 1);

   nir_push_loop(b);
   {
      nir_def *iv = nir_load_var(b, i);
      nir_push_if(b, nir_uge(b, iv, nprims));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      /* Only whole primitives fit the slot: a strip longer than max_vertices was cut by the
       * counter check in EmitVertex, and its indices must stay inside this invocation too. */
      nir_def *prim = nir_iadd(b, prim_base, iv);
      nir_push_if(b, nir_ult_imm(b, prim, st->info->max_prims));
      {
         nir_def *idx_slot = nir_iadd(b, st->out_idx_base, nir_imul_imm(b, prim, vpp));
         nir_def *vbase = nir_iadd(b, st->out_vtx_base, nir_iadd(b, vtx_start, iv));
         nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, iv, 1), 0);

         for (unsigned k = 0; k < vpp; k++) {
            nir_def *v = nir_iadd_imm(b, vbase, k);

            /* Odd triangles of a strip are reversed, keeping the provoking vertex where the
             * convention wants it: (v, v+2, v+1) for the first vertex, (v+1, v, v+2) for the
             * last. */
            if (vpp == 3) {
               nir_def *first_order = nir_iadd_imm(b, vbase, k == 0 ? 0 : k == 1 ? 2 : 1);
               nir_def *last_order = nir_iadd_imm(b, vbase, k == 0 ? 1 : k == 1 ? 0 : 2);
               nir_def *odd_v = nir_bcsel(b, nir_ine_imm(b, st->flatshade_first, 0),
                                          first_order, last_order);
               v = nir_bcsel(b, odd, odd_v, v);
            }

            nir_def *addr = nir_iadd(
               b, st->out_index, nir_u2u64(b, nir_imul_imm(b, nir_iadd_imm(b, idx_slot, k), 4)));
            nir_store_global(b, v, addr, .align_mul = 4, .write_mask = 0x1);
         }
      }
      nir_pop_if(b, NULL);

      nir_store_var(b, i, nir_iadd_imm(b, iv, 1), 1);
   }
   nir_pop_loop(b, NULL);

   nir_store_var(b, st->prims_written, nir_iadd(b, prim_base, nprims), 1);
}

/* Every index slot this invocation owns starts at the dead vertex (pan_gs_params::dead_vertex),
 * so what the shader never reaches is discarded by the tiler. */
static void
emit_degenerate_fill(nir_builder *b, struct lower_gs_state *st)
{
   const unsigned total = st->info->max_prims * st->info->output_verts;

   nir_variable *i = nir_local_variable_create(b->impl, glsl_uint_type(), "gs_fill_i");
   nir_store_var(b, i, nir_imm_int(b, 0), 1);

   nir_push_loop(b);
   {
      nir_def *iv = nir_load_var(b, i);
      nir_push_if(b, nir_uge_imm(b, iv, total));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_def *addr = nir_iadd(
         b, st->out_index, nir_u2u64(b, nir_imul_imm(b, nir_iadd(b, st->out_idx_base, iv), 4)));
      nir_store_global(b, st->dead_vertex, addr, .align_mul = 4, .write_mask = 0x1);

      nir_store_var(b, i, nir_iadd_imm(b, iv, 1), 1);
   }
   nir_pop_loop(b, NULL);
}

static bool
lower_gs_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_gs_state *st = data;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_per_vertex_input:
      nir_def_replace(&intr->def, lower_per_vertex_input(b, intr, st));
      return true;
   case nir_intrinsic_store_output:
      lower_store_output(b, intr, st);
      nir_instr_remove(&intr->instr);
      return true;
   case nir_intrinsic_emit_vertex_with_counter:
      lower_emit_vertex(b, intr, st);
      nir_instr_remove(&intr->instr);
      return true;
   case nir_intrinsic_end_primitive_with_counter:
      lower_end_primitive(b, intr, st);
      nir_instr_remove(&intr->instr);
      return true;
   case nir_intrinsic_set_vertex_and_primitive_count:
      nir_instr_remove(&intr->instr);
      return true;
   case nir_intrinsic_load_primitive_id:
      nir_def_replace(&intr->def, st->prim_id);
      return true;
   case nir_intrinsic_load_invocation_id:
      nir_def_replace(&intr->def, st->invocation_id);
      return true;
   default:
      return false;
   }
}

bool
pan_nir_lower_gs(nir_shader *nir, const struct pan_gs_info *info, uint32_t noperspective)
{
   assert(nir->info.stage == MESA_SHADER_GEOMETRY);

   /* The implicit EndPrimitive at the end of the shader: without it a strip the shader never
    * closed writes no indices. One right after another ends an empty strip and writes nothing. */
   {
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_builder b = nir_builder_create(impl);
      nir_foreach_pred(pred, impl->end_block) {
         b.cursor = nir_after_block_before_jump(pred);
         nir_end_primitive(&b, .stream_id = 0);
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }

   NIR_PASS(_, nir, nir_lower_gs_intrinsics,
            nir_lower_gs_intrinsics_count_primitives |
               nir_lower_gs_intrinsics_count_vertices_per_primitive |
               nir_lower_gs_intrinsics_overwrite_incomplete);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   struct lower_gs_state st = {.info = info, .noperspective = noperspective};

   u_foreach_bit64(loc, info->generic_mask | VARYING_BIT_POS | VARYING_BIT_PSIZ |
                           VARYING_BIT_VIEWPORT | VARYING_BIT_LAYER) {
      if (loc < 64 && (nir->info.outputs_written & BITFIELD64_BIT(loc)))
         st.outputs[loc] = nir_local_variable_create(impl, glsl_uvec4_type(), "gs_out");
   }

   st.prims_written = nir_local_variable_create(impl, glsl_uint_type(), "gs_prims_written");
   nir_store_var(&b, st.prims_written, nir_imm_int(&b, 0), 1);

   load_input_params(&b, &st);
   st.out_pos = PARAM64(&b, &st, out_pos);
   st.out_psiz = PARAM64(&b, &st, out_psiz);
   st.out_general = PARAM64(&b, &st, out_general);
   st.out_index = PARAM64(&b, &st, out_index);
   st.flatshade_first = PARAM32(&b, &st, flatshade_first);
   st.dead_vertex = PARAM32(&b, &st, dead_vertex);
   nir_def *num_prims = PARAM32(&b, &st, num_input_prims);

   /* One invocation per (input primitive, gl_InvocationID) pair, each with its own slots. */
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *prim_global = gid;
   st.invocation_id = nir_imm_int(&b, 0);
   if (info->invocations > 1) {
      prim_global = nir_udiv_imm(&b, gid, info->invocations);
      st.invocation_id = nir_umod_imm(&b, gid, info->invocations);
   }

   st.out_vtx_base = nir_imul_imm(&b, gid, info->max_verts);
   st.out_idx_base = nir_imul_imm(&b, gid, info->max_prims * info->output_verts);

   /* Trailing invocations of the last workgroup own no slots. */
   nir_push_if(&b, nir_uge(&b, prim_global, num_prims));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);

   /* gl_PrimitiveIDIn restarts with every instance. */
   nir_def *ppi = nir_umax(&b, st.prims_per_instance, nir_imm_int(&b, 1));
   st.instance = nir_udiv(&b, prim_global, ppi);
   st.prim_id = nir_isub(&b, prim_global, nir_imul(&b, st.instance, ppi));

   emit_degenerate_fill(&b, &st);

   /* A count only the GPU knows (the tessellator's index count for the tess eval stage): the
    * slots past it stay dead. */
   nir_def *count_addr = PARAM64(&b, &st, num_prims_addr);
   nir_push_if(&b, nir_ine_imm(&b, count_addr, 0));
   nir_def *counted = nir_load_global(&b, 1, 32, count_addr, .align_mul = 4);
   nir_pop_if(&b, NULL);
   nir_def *actual = nir_if_phi(&b, counted, num_prims);
   nir_push_if(&b, nir_uge(&b, prim_global, actual));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);

   bool progress = nir_shader_intrinsics_pass(nir, lower_gs_instr, nir_metadata_none, &st);

   /* It is a compute shader from here on, with no outputs left. */
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.outputs_written = 0;
   nir->info.inputs_read = 0;
   nir->info.workgroup_size[0] = PAN_GS_WORKGROUP_SIZE;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;

   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_dce);

   return progress;
}

/* -- TESSELLATION, see pan_gs.h ------------------------------------------------------------- */

static bool
lower_tcs_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_gs_state *st = data;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_per_vertex_input:
      nir_def_replace(&intr->def, lower_per_vertex_input(b, intr, st));
      return true;
   case nir_intrinsic_barrier: {
      /* The outputs become global memory (poly_nir_lower_tcs), and a patch does not fit in a
       * Bifrost subgroup, so the barrier keeps its workgroup scope and covers global memory.
       * Done before src/poly sees it, which would narrow it to a subgroup. */
      nir_variable_mode modes = nir_intrinsic_memory_modes(intr);
      if (!(modes & nir_var_shader_out))
         return false;
      nir_intrinsic_set_memory_modes(intr, (modes & ~nir_var_shader_out) | nir_var_mem_global);
      if (nir_intrinsic_execution_scope(intr) < SCOPE_WORKGROUP)
         nir_intrinsic_set_execution_scope(intr, SCOPE_WORKGROUP);
      if (nir_intrinsic_memory_scope(intr) < SCOPE_WORKGROUP)
         nir_intrinsic_set_memory_scope(intr, SCOPE_WORKGROUP);
      return true;
   }
   default:
      return false;
   }
}

bool
pan_nir_lower_tcs_inputs(nir_shader *nir)
{
   static const struct pan_gs_info tcs_info = {.input_verts = 0};

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   struct lower_gs_state st = {.info = &tcs_info};
   load_input_params(&b, &st);
   st.topology = nir_imm_int(&b, PAN_GS_TOPO_LIST);

   /* One workgroup per patch: x the patch within its instance, y the instance. */
   nir_def *wg = nir_load_workgroup_id(&b);
   st.prim_id = nir_channel(&b, wg, 0);
   st.instance = nir_channel(&b, wg, 1);

   return nir_shader_intrinsics_pass(nir, lower_tcs_instr, nir_metadata_none, &st);
}

void
pan_nir_tes_to_gs(nir_shader *nir)
{
   nir->info.stage = MESA_SHADER_GEOMETRY;
   memset(&nir->info.gs, 0, sizeof(nir->info.gs));
   nir->info.gs.input_primitive = MESA_PRIM_POINTS;
   nir->info.gs.output_primitive = MESA_PRIM_POINTS;
   nir->info.gs.vertices_in = 1;
   nir->info.gs.vertices_out = 1;
   nir->info.gs.invocations = 1;
   nir->info.gs.active_stream_mask = 1;

   /* The stores are collected into the output variables wherever they are
    * (pan_nir_lower_gs), so a single EmitVertex after everything flushes the finished vertex. */
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_emit_vertex(&b, 0);
   nir_progress(true, impl, nir_metadata_none);
}
