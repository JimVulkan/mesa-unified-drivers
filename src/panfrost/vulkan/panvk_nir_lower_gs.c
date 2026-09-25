/*
 * Copyright 2026 XclipseTools
 * SPDX-License-Identifier: MIT
 */

/*
 * Geometry shaders on hardware that has no geometry stage.
 *
 * The Mali job manager runs vertex -> tiler -> fragment and nothing else. There is no geometry
 * stage to program, which is why gallium panfrost stops at GLES 3.1. The vendor driver exposes
 * geometryShader on this same chip by emulating it, and this is that emulation.
 *
 * WHAT MAKES IT TRACTABLE ON BIFROST. The vertex shader does not hand its results to the tiler
 * through some private channel -- it WRITES VARYINGS TO MEMORY, and the tiler reads them back
 * from buffers whose addresses and strides live in descriptors we already build (position is a
 * vec4 at stride 16, point size an fp16 at stride 2, and everything else is packed into one
 * generic buffer at pan_varying_layout::generic_size_B). So a geometry shader is a COMPUTE JOB
 * SPLICED BETWEEN THE TWO: it reads the vertex shader's varying buffers, runs the geometry
 * shader body once per input primitive, and writes a fresh set of varying buffers for the tiler.
 * No new hardware path is needed, only a shader and a job.
 *
 * WORST-CASE ALLOCATION, DELIBERATELY. How many vertices a geometry shader emits is not known
 * until it runs, and the honest way to give the tiler that count would be to patch its descriptor
 * from the GPU. That is exactly the mechanism that does not work here -- a compute job's patch of
 * a tiler descriptor never lands on this job manager, which is the open indirect-draw bug -- so
 * this pass does not depend on it. Every input primitive gets a fixed slot of max_vertices output
 * vertices and (max_vertices - 2) output primitives, the counts are therefore known while the
 * command buffer is recorded, and slots the shader did not fill are made DEGENERATE so the tiler
 * discards them. It costs memory and some rasterisation work; it does not cost correctness.
 *
 * STRIPS BECOME LISTS VIA AN INDEX BUFFER. A geometry shader emits strips, and a strip that ran
 * off the end of one primitive's slot would connect it to the next one's. Rather than keep whole
 * vertices live in registers to re-emit them, the pass writes each vertex once and appends
 * INDICES describing the expanded primitive list. Unused index slots are filled with a repeated
 * index, which is a zero-area primitive and is culled. Indexed draws are the natural consumer for
 * that and they now work on this driver.
 *
 * The pass leaves a plain compute shader that reads its buffer addresses out of push constants;
 * see panvk_gs_push in panvk_shader.h for that block.
 */

#include "panvk_gs.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_xfb_info.h"

/* Same slot accounting panvk uses for its other stages, so the offsets nir_lower_io produces
 * mean the same thing here as they do for a vertex or fragment shader. */
void *panvk_gs_dbg_pos_cpu;
uint64_t panvk_gs_dbg_pos_gpu;
uint32_t panvk_gs_dbg_pos_verts;
void *panvk_gs_dbg_in_pos_cpu;
void *panvk_gs_dbg_idx_cpu;
uint32_t panvk_gs_dbg_idx_count;

static unsigned
gs_glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/* A multi-component store_global does NOT reach memory here: the 32-bit index writes landed
 * and the vec4 position writes silently did not, verified by dumping the buffer. Write every
 * component as its own 32-bit store, which is the form that demonstrably works. */
static void
store_components(nir_builder *b, nir_def *base, nir_def *byte_off, nir_def *val)
{
   for (unsigned c = 0; c < val->num_components; c++) {
      nir_def *addr =
         nir_iadd(b, base, nir_u2u64(b, nir_iadd_imm(b, byte_off, c * 4)));
      nir_store_global(b, nir_channel(b, val, c), addr, .align_mul = 4,
                       .write_mask = 0x1);
   }
}

/* The mirror of store_components: a multi-component nir_load_global does NOT come back with
 * the data here either. Reading a vec4 gl_Position with one 128-bit load returned all zeros from
 * an address that was verified correct and that the CPU could read the right values out of.
 * Component-at-a-time 32-bit loads work. */
static nir_def *
load_components(nir_builder *b, nir_def *base, nir_def *byte_off,
                unsigned num_components, unsigned bit_size)
{
   const unsigned sz = bit_size / 8;
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];

   for (unsigned c = 0; c < num_components; c++) {
      nir_def *addr =
         nir_iadd(b, base, nir_u2u64(b, nir_iadd_imm(b, byte_off, c * sz)));
      comps[c] = nir_load_global(b, 1, bit_size, addr, .align_mul = sz);
   }

   return nir_vec(b, comps, num_components);
}

struct lower_gs_state {
   const struct panvk_gs_lower_options *opts;

   /* The params block, loaded once at the top of the shader. */
   nir_def *in_pos, *in_general;
   nir_def *out_pos, *out_general, *out_index;
   nir_def *in_offsets, *out_offsets;
   nir_def *in_general_stride, *out_general_stride;
   nir_def *vp_scale, *vp_offset;
   nir_def *in_screen_space;
   nir_def *out_clip_space;
   /* gl_ViewportIndex: the shader's variable, and the per-viewport transform table. */
   nir_variable *viewport_var;
   nir_def *viewports;
   /* gl_PrimitiveIDIn (per instance) and gl_InvocationID. */
   nir_def *prim_id;
   nir_def *invocation_id;
   /* Input assembly: which instance this primitive belongs to, and the draw's parameters. */
   nir_def *instance;
   /* Input primitives per instance: the last one of a strip with adjacency is special. */
   nir_def *prims_per_instance;
   nir_def *index_buf, *index_size, *topology, *instance_stride;
   /* Subtracted from an index to get the vertex record: the vertex job's first vertex. */
   nir_def *index_bias;
   /* Vertices per input primitive when opts->input_verts_per_prim is 0 (a tess control
    * shader's patch, whose size is draw state). */
   nir_def *vpp_dyn;
   /* Tessellation: the draw's struct poly_tess_params. */
   nir_def *tess_params;

   /* Base output vertex slot for this invocation: prim_id * max_output_verts. */
   nir_def *out_vtx_base;
   /* Base output index slot: prim_id * max_output_prims * out_verts_per_prim. */
   nir_def *out_idx_base;
   /* How many primitives this invocation has already written indices for, per vertex stream.
    * Maintained here because nir_lower_gs_intrinsics does NOT hand it to us -- see
    * lower_end_primitive. */
   nir_variable *prims_written[4];
   /* geometryStreams: see panvk_gs_push::xfb_stream_*. */
   nir_def *stream_slots, *stream_idx, *stream_index, *xfb_invocations;

   /* This compute invocation, and transform feedback (xfb_dwords == 0: the shader captures
    * nothing). The two addresses are 0 on a draw that is not capturing. */
   nir_def *gid;
   unsigned xfb_dwords;
   nir_def *xfb_counts, *xfb_staging;
};

int
panvk_gs_xfb_table(const nir_xfb_info *xfb, uint32_t *table)
{
   if (!xfb)
      return 0;

   int n = 0;
   for (unsigned i = 0; i < xfb->output_count; i++) {
      const nir_xfb_output_info *o = &xfb->outputs[i];

      /* 32-bit data only: nothing here produces 16-bit varyings. Any stream: a record holds every
       * captured component, and a stream's copy only writes its own buffers. */
      if (o->data_is_16bit)
         return -1;

      /* component_mask holds absolute components of the location, and the output's dwords are
       * consecutive from its offset in the order of the set bits. */
      unsigned dw = o->offset / 4;
      u_foreach_bit(c, o->component_mask) {
         if (n >= PANVK_XFB_MAX_RECORD_DWORDS)
            return -1;
         table[n++] = ((uint32_t)o->buffer << 16) | dw++;
      }
   }

   return n;
}

/* Component comp of output location loc as the shader holds it right now, or NULL when no 32-bit
 * output variable covers it. Mirrors how nir_gather_xfb_info numbered the locations: a compact
 * float array (clip/cull distances) counts components across locations, an array or matrix takes
 * one location per element or column. */
static nir_def *
xfb_output_component(nir_builder *b, unsigned loc, unsigned comp)
{
   nir_foreach_shader_out_variable(var, b->shader) {
      const struct glsl_type *type = var->type;
      nir_deref_instr *d = nir_build_deref_var(b, var);

      if (var->data.compact) {
         const unsigned first = var->data.location * 4 + var->data.location_frac;
         const unsigned want = loc * 4 + comp;
         if (want < first || want >= first + glsl_get_length(type))
            continue;
         return nir_load_deref(b, nir_build_deref_array_imm(b, d, want - first));
      }

      const unsigned slots = glsl_count_attribute_slots(type, false);
      if (loc < var->data.location || loc >= var->data.location + slots)
         continue;

      if (glsl_type_is_array_or_matrix(type)) {
         const struct glsl_type *elem =
            glsl_type_is_array(type) ? glsl_get_array_element(type) : glsl_get_column_type(type);
         if (glsl_count_attribute_slots(elem, false) != 1)
            continue;
         d = nir_build_deref_array_imm(b, d, loc - var->data.location);
         type = elem;
      }

      if (!glsl_type_is_vector_or_scalar(type) || glsl_get_bit_size(type) != 32)
         continue;

      const unsigned frac = var->data.location_frac;
      if (comp < frac || comp >= frac + glsl_get_vector_elements(type))
         continue;

      return nir_channel(b, nir_load_deref(b, d), comp - frac);
   }

   return NULL;
}

/* EmitVertex with transform feedback: the vertex's captured components, in table order, into its
 * staging record. gl_Position is still CLIP space here -- clip_to_screen only touches the copy the
 * tiler gets -- which is what capture must see. */
static void
stage_xfb_record(nir_builder *b, struct lower_gs_state *st, nir_def *slot)
{
   const nir_xfb_info *xfb = st->opts->xfb;

   nir_push_if(b, nir_ine_imm(b, st->xfb_staging, 0));
   {
      nir_def *rec = nir_iadd(b, st->xfb_staging,
                              nir_u2u64(b, nir_imul_imm(b, slot, st->xfb_dwords * 4)));
      unsigned j = 0;
      for (unsigned i = 0; i < xfb->output_count; i++) {
         const nir_xfb_output_info *o = &xfb->outputs[i];
         u_foreach_bit(c, o->component_mask) {
            nir_def *v = xfb_output_component(b, o->location, c);
            if (!v)
               v = nir_imm_int(b, 0);
            nir_store_global(b, v, nir_iadd_imm(b, rec, j * 4), .align_mul = 4,
                             .write_mask = 0x1);
            j++;
         }
      }
   }
   nir_pop_if(b, NULL);
}

static nir_def *
load_param_u64(nir_builder *b, const struct panvk_gs_lower_options *opts, unsigned offset)
{
   return nir_load_push_constant(b, 1, 64, nir_imm_int(b, 0),
                                 .base = opts->params_base + offset, .range = 8);
}

static nir_def *
load_param_u32(nir_builder *b, const struct panvk_gs_lower_options *opts, unsigned offset)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0),
                                 .base = opts->params_base + offset, .range = 4);
}

static nir_def *
load_param_vec3f(nir_builder *b, const struct panvk_gs_lower_options *opts,
                 unsigned offset)
{
   nir_def *c[3];

   for (unsigned i = 0; i < 3; i++)
      c[i] = nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0),
                                    .base = opts->params_base + offset + i * 4,
                                    .range = 4);

   return nir_vec(b, c, 3);
}

/* -- THE VIEWPORT TRANSFORM -------------------------------------------------------------------
 * The tiler takes POSITION in screen space with w replaced by 1/w, which is what
 * nir_lower_viewport_transform makes a vertex shader write. A vertex shader feeding this emulation
 * is compiled WITHOUT it (pan_compile_inputs::vs_keep_clip_space), so the input here is clip
 * space, exactly what the geometry shader and transform feedback expect, and the transform is
 * applied once, on the way out.
 *
 * It used to be undone on the way in instead. That round trip is lossy: with rasterizer discard
 * the viewport is ignored and its scale is 0, and a flat depth range makes the z scale 0, and a
 * vertex shader that multiplied by 0 leaves nothing to recover -- transform feedback captured
 * (0, 0, 0, 1) for every vertex of a discard draw.
 */
/* The inverse, for a vertex shader that applied the transform itself (in_screen_space). Exact
 * enough whenever the draw rasterises: the viewport then has non-zero scales. A zero component
 * selects zero rather than dividing. */
static nir_def *
screen_to_clip(nir_builder *b, struct lower_gs_state *st, nir_def *p)
{
   nir_def *w = nir_frcp(b, nir_channel(b, p, 3));
   nir_def *zero3 = nir_replicate(b, nir_imm_float(b, 0.0f), 3);
   nir_def *rcp_scale =
      nir_bcsel(b, nir_feq(b, st->vp_scale, zero3), zero3, nir_frcp(b, st->vp_scale));
   nir_def *ndc =
      nir_fmul(b, nir_fsub(b, nir_trim_vector(b, p, 3), st->vp_offset), rcp_scale);
   nir_def *clip = nir_fmul(b, ndc, nir_replicate(b, w, 3));

   return nir_vec4(b, nir_channel(b, clip, 0), nir_channel(b, clip, 1),
                   nir_channel(b, clip, 2), w);
}

static nir_def *
clip_to_screen_xf(nir_builder *b, nir_def *p, nir_def *vp_scale, nir_def *vp_offset);

static nir_def *
clip_to_screen(nir_builder *b, struct lower_gs_state *st, nir_def *p)
{
   return clip_to_screen_xf(b, p, st->vp_scale, st->vp_offset);
}

static nir_def *
clip_to_screen_xf(nir_builder *b, nir_def *p, nir_def *vp_scale, nir_def *vp_offset)
{
   /* Same arithmetic and the same clamp as nir_lower_viewport_transform, so what this writes is
    * indistinguishable from what the vertex shader would have written. */
   nir_def *w_recip = nir_frcp(b, nir_channel(b, p, 3));
   w_recip = nir_fclamp(b, w_recip, nir_imm_float(b, -32768.0f),
                        nir_imm_float(b, 32768.0f));

   nir_def *ndc = nir_fmul(b, nir_trim_vector(b, p, 3), nir_replicate(b, w_recip, 3));
   nir_def *screen = nir_fadd(b, nir_fmul(b, ndc, vp_scale), vp_offset);

   /* A vertex shader's position reaches memory through the SNAP_4 varying format, which rounds
    * x and y to 1/256 px (measured: -0.9 came back -0.8984375 in a unit viewport). These are
    * plain stores, so snap here too, or a triangle's shared edge rasterises a pixel or two
    * differently from the same draw without a geometry shader. */
   nir_def *snap_x = nir_fmul_imm(b, nir_fround_even(b, nir_fmul_imm(b, nir_channel(b, screen, 0), 256.0)),
                                  1.0 / 256.0);
   nir_def *snap_y = nir_fmul_imm(b, nir_fround_even(b, nir_fmul_imm(b, nir_channel(b, screen, 1), 256.0)),
                                  1.0 / 256.0);

   return nir_vec4(b, snap_x, snap_y, nir_channel(b, screen, 2), w_recip);
}

/* slot -> byte offset, out of the table the draw path filled in. */
static nir_def *
load_slot_offset(nir_builder *b, nir_def *table, unsigned location)
{
   nir_def *addr = nir_iadd(b, table, nir_imm_int64(b, location * 4));
   return nir_load_global(b, 1, 32, addr, .align_mul = 4);
}

/* The address in an ATTRIBUTE_BUFFER descriptor: bits 6-55 of its first 64 bits. Done on 32-bit
 * halves -- the Bifrost backend cannot take a 64-bit AND with an immediate. */
static nir_def *
desc_pointer(nir_builder *b, nir_def *desc)
{
   nir_def *lo = nir_load_global(b, 1, 32, desc, .align_mul = 8);
   nir_def *hi = nir_load_global(b, 1, 32, nir_iadd_imm(b, desc, 4), .align_mul = 4);
   return nir_pack_64_2x32_split(b, nir_iand_imm(b, lo, 0xffffffc0),
                                 nir_iand_imm(b, hi, 0x00ffffff));
}

/* An indirect draw's input assembly, read inside the caller's indirect_cmd != 0 branch: the
 * instance count and firstIndex from the command, the vertex job's first index to subtract from
 * an index, the per-instance record stride from the vertex job's DRAW section, and where the draw
 * helper pointed the varying buffers. See struct panvk_gs_push. */
struct indirect_inputs {
   nir_def *inst, *index_buf, *bias, *stride, *pos, *gen;
};

static struct indirect_inputs
load_indirect_inputs(nir_builder *b, const struct panvk_gs_lower_options *opts, nir_def *cmd,
                     nir_def *index_buf, nir_def *index_size)
{
   struct indirect_inputs in;
   nir_def *indexed = nir_ine_imm(b, index_size, 0);
   in.inst = nir_load_global(b, 1, 32, nir_iadd_imm(b, cmd, 4), .align_mul = 4);
   nir_def *first = nir_load_global(b, 1, 32, nir_iadd_imm(b, cmd, 8), .align_mul = 4);

   in.index_buf = nir_bcsel(b, indexed,
                            nir_iadd(b, index_buf, nir_u2u64(b, nir_imul(b, first, index_size))),
                            nir_imm_int64(b, 0));
   /* Only an indexed draw has a minimum to read: the address is 0 otherwise, and a load from
    * it is an MMU fault that takes the whole chain down. */
   nir_def *min_addr = load_param_u64(b, opts, offsetof(struct panvk_gs_push, index_min));
   nir_push_if(b, indexed);
   nir_def *min_val = nir_load_global(b, 1, 32, min_addr, .align_mul = 4);
   nir_pop_if(b, NULL);
   in.bias = nir_if_phi(b, min_val, nir_imm_int(b, 0));

   /* Instance Size, a "padded" count: (2 * odd + 1) << shift, shift in bits 0-4. */
   nir_def *dcd = load_param_u64(b, opts, offsetof(struct panvk_gs_push, vertex_dcd));
   nir_def *w0 = nir_load_global(b, 1, 32, dcd, .align_mul = 4);
   nir_def *packed = nir_iand_imm(b, nir_ushr_imm(b, w0, 16), 0xff);
   nir_def *padded = nir_ishl(b, nir_iadd_imm(b, nir_imul_imm(b, nir_ushr_imm(b, packed, 5), 2), 1),
                              nir_iand_imm(b, packed, 0x1f));
   in.stride = nir_bcsel(b, nir_ugt_imm(b, in.inst, 1), padded, nir_imm_int(b, 0));

   /* ATTRIBUTE_BUFFER: the pointer is bits 6-55 of the first word. */
   nir_def *pdesc = load_param_u64(b, opts, offsetof(struct panvk_gs_push, in_pos_desc));
   nir_def *gdesc = load_param_u64(b, opts, offsetof(struct panvk_gs_push, in_general_desc));
   in.pos = desc_pointer(b, pdesc);
   in.gen = desc_pointer(b, gdesc);
   return in;
}

/* A direct indexed draw whose indices the CPU read starts its vertex job at the smallest one, so
 * its indices are biased by that minimum too; index_min is 0 when there is none to subtract.
 * Emitted before the indirect branch, whose phi must be the last if popped. */
static nir_def *
load_direct_bias(nir_builder *b, const struct panvk_gs_lower_options *opts, nir_def *index_size)
{
   nir_def *min_addr = load_param_u64(b, opts, offsetof(struct panvk_gs_push, index_min));
   nir_push_if(b, nir_iand(b, nir_ine_imm(b, min_addr, 0), nir_ine_imm(b, index_size, 0)));
   nir_def *v = nir_load_global(b, 1, 32, min_addr, .align_mul = 4);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, v, nir_imm_int(b, 0));
}

/*
 * Which of the vertex shader's varying records is vertex i of this invocation's input
 * primitive. Primitive assembly first -- the position of the vertex in the draw's vertex (or
 * index) stream -- then the index buffer, then the instance.
 */
static nir_def *
gs_input_record(nir_builder *b, struct lower_gs_state *st, nir_def *i)
{
   const unsigned vpp = st->opts->input_verts_per_prim;
   nir_def *p = st->prim_id;

   /* LIST: p * vpp + i */
   nir_def *list = vpp ? nir_iadd(b, nir_imul_imm(b, p, vpp), i)
                       : nir_iadd(b, nir_imul(b, p, st->vpp_dyn), i);

   /* STRIP: p + i. A triangle strip keeps its winding by taking vertices p, p+2, p+1 on odd p
    * (Vulkan's primitive order for strips). */
   nir_def *strip_i = i;
   if (vpp == 3) {
      nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, p, 1), 0);
      nir_def *swapped = nir_bcsel(b, nir_ieq_imm(b, i, 1), nir_imm_int(b, 2),
                                   nir_bcsel(b, nir_ieq_imm(b, i, 2), nir_imm_int(b, 1), i));
      strip_i = nir_bcsel(b, odd, swapped, i);
   }
   nir_def *strip = nir_iadd(b, p, strip_i);

   /* FAN: p + 1, p + 2, 0 */
   nir_def *fan = nir_bcsel(b, nir_ieq_imm(b, i, 2), nir_imm_int(b, 0),
                            nir_iadd(b, p, nir_iadd_imm(b, i, 1)));

   /* TRIANGLE STRIP WITH ADJACENCY. The strip is the even vertices s_k = 2k; triangle p is
    * s_p, s_p+1, s_p+2 (the first two swapped on odd p), and each edge's adjacent vertex is the
    * third vertex of the neighbouring strip triangle, or the odd vertex supplied for a boundary
    * edge. In the shader's order (v0, a01, v1, a12, v2, a20), which is the GL table:
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

   /* A patch is only ever a list. */
   nir_def *seq = !vpp ? list :
      nir_bcsel(b, nir_ieq_imm(b, st->topology, PANVK_GS_TOPO_STRIP), strip,
                nir_bcsel(b, nir_ieq_imm(b, st->topology, PANVK_GS_TOPO_FAN), fan,
                          nir_bcsel(b, nir_ieq_imm(b, st->topology, PANVK_GS_TOPO_TRI_STRIP_ADJ),
                                    tsa, list)));

   /* Through the index buffer, when there is one. */
   nir_def *vtx = seq;
   nir_push_if(b, nir_ine_imm(b, st->index_buf, 0));
   {
      nir_def *addr = nir_iadd(b, st->index_buf,
                               nir_u2u64(b, nir_imul(b, seq, st->index_size)));
      /* The aligned 32-bit word holding the index, then the index out of it. */
      nir_def *word_addr = nir_iand_imm(b, addr, ~3ull);
      nir_def *word = nir_load_global(b, 1, 32, word_addr, .align_mul = 4);
      nir_def *shift = nir_imul_imm(b, nir_u2u32(b, nir_iand_imm(b, addr, 3)), 8);
      nir_def *shifted = nir_ushr(b, word, shift);
      nir_def *idx = nir_bcsel(b, nir_ieq_imm(b, st->index_size, 1),
                               nir_iand_imm(b, shifted, 0xff),
                               nir_bcsel(b, nir_ieq_imm(b, st->index_size, 2),
                                         nir_iand_imm(b, shifted, 0xffff), word));
      vtx = nir_isub(b, idx, st->index_bias);
   }
   nir_pop_if(b, NULL);
   vtx = nir_if_phi(b, vtx, seq);

   return nir_iadd(b, vtx, nir_imul(b, st->instance, st->instance_stride));
}

/*
 * A geometry shader reads gl_in[i].<slot>. After nir_lower_io those are load_per_vertex_input
 * with the vertex index as src[0] and the slot in the io semantics. The vertex it wants is the
 * i'th vertex of THIS invocation's input primitive, which the vertex shader already wrote to its
 * varying buffers, so the load becomes an address computation and a global load.
 */
static nir_def *
lower_per_vertex_input(nir_builder *b, nir_intrinsic_instr *intr,
                       struct lower_gs_state *st)
{
   const struct panvk_gs_lower_options *opts = st->opts;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);

   nir_def *vtx = gs_input_record(b, st, intr->src[0].ssa);

   if (sem.location == VARYING_SLOT_POS && !opts->echo) {
      /* The vertex shader was compiled to leave gl_Position in clip space for us
       * (pan_compile_inputs::vs_keep_clip_space), so this is the value it wrote. */
      nir_def *byte = nir_imul_imm(b, vtx, opts->pos_stride);
      nir_def *raw = load_components(b, st->in_pos, byte, 4, 32);
      nir_def *clip = nir_bcsel(b, nir_ine_imm(b, st->in_screen_space, 0),
                                screen_to_clip(b, st, raw), raw);

      unsigned first = nir_intrinsic_component(intr);
      nir_def *c[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < intr->def.num_components; i++)
         c[i] = nir_channel(b, clip, MIN2(first + i, 3));

      return nir_vec(b, c, intr->def.num_components);
   }

   /* An input array element: the io semantics name the variable's first location and src[1]
    * the slot within it (constant: indirect input derefs were turned into if-trees). */
   const unsigned location =
      sem.location + (nir_src_is_const(intr->src[1]) ? nir_src_as_uint(intr->src[1]) : 0);

   nir_def *base, *stride, *offset;
   if (location == VARYING_SLOT_POS) {
      base = st->in_pos;
      stride = nir_imm_int(b, opts->pos_stride);
      offset = nir_imm_int(b, 0);
   } else {
      base = st->in_general;
      stride = st->in_general_stride;
      offset = load_slot_offset(b, st->in_offsets, location);
   }

   nir_def *byte = nir_iadd(b, nir_imul(b, vtx, stride),
                            nir_iadd_imm(b, offset, nir_intrinsic_component(intr) * 4));
   nir_def *addr = nir_iadd(b, base, nir_u2u64(b, byte));

   if (opts->echo && intr->def.num_components == 4 && intr->def.bit_size == 32) {
      /* Raw bits: the address low/high halves, then the base pointer's. */
      return nir_vec4(b, nir_unpack_64_2x32_split_x(b, addr),
                      nir_unpack_64_2x32_split_y(b, addr),
                      nir_unpack_64_2x32_split_x(b, base),
                      nir_unpack_64_2x32_split_y(b, base));
   }

   if (location == VARYING_SLOT_POS)
      return load_components(b, base, byte, intr->def.num_components, intr->def.bit_size);

   /* A slot the vertex shader does not write reads as zero -- and must not be loaded at all: the
    * offset is PANVK_GS_SLOT_UNUSED and the buffer may not even exist. */
   nir_push_if(b, nir_ine_imm(b, offset, PANVK_GS_SLOT_UNUSED));
   nir_def *val = load_components(b, base, byte, intr->def.num_components, intr->def.bit_size);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, val, nir_imm_zero(b, intr->def.num_components, intr->def.bit_size));
}

/* Output vertex slot counter of stream s, the address of entry `entry` of stream s's index list
 * for this invocation, and the address of its primitive count. Stream 0 is the rasterised one and
 * keeps the layout it always had. */
static nir_def *
stream_vtx_slot(nir_builder *b, struct lower_gs_state *st, unsigned s, nir_def *counter)
{
   nir_def *slot = nir_iadd(b, st->out_vtx_base, counter);
   return s ? nir_iadd(b, slot, nir_imul_imm(b, st->stream_slots, s)) : slot;
}

static nir_def *
stream_index_entry(nir_builder *b, struct lower_gs_state *st, unsigned s, nir_def *entry)
{
   nir_def *e = nir_iadd(b, st->out_idx_base, entry);
   if (!s)
      return nir_iadd(b, st->out_index, nir_u2u64(b, nir_imul_imm(b, e, 4)));
   e = nir_iadd(b, e, nir_imul_imm(b, st->stream_idx, s - 1));
   return nir_iadd(b, st->stream_index, nir_u2u64(b, nir_imul_imm(b, e, 4)));
}

static nir_def *
stream_count_addr(nir_builder *b, struct lower_gs_state *st, unsigned s)
{
   nir_def *i = s ? nir_iadd(b, st->gid, nir_imul_imm(b, st->xfb_invocations, s)) : st->gid;
   return nir_iadd(b, st->xfb_counts, nir_u2u64(b, nir_imul_imm(b, i, 4)));
}

/*
 * EmitVertex(). nir_lower_gs_intrinsics has already given us the running vertex counter, and
 * nir_lower_io_vars_to_temporaries has already turned the shader's outputs into variables we can read
 * here, so this is where a whole output vertex is flushed to memory.
 */
static void
lower_emit_vertex(nir_builder *b, nir_intrinsic_instr *intr, struct lower_gs_state *st)
{
   const struct panvk_gs_lower_options *opts = st->opts;
   nir_def *counter = intr->src[0].ssa;
   const unsigned stream = nir_intrinsic_stream_id(intr);

   /* A vertex on another stream than 0 is never rasterised: it only matters to transform
    * feedback, which stages it in its stream's slots and records it at EndStreamPrimitive. */
   if (stream != 0) {
      if (!st->xfb_dwords)
         return;
      nir_push_if(b, nir_iand(b, nir_ult_imm(b, counter, opts->max_output_verts),
                              nir_ine_imm(b, st->xfb_staging, 0)));
      {
         nir_def *slot = stream_vtx_slot(b, st, stream, counter);
         stage_xfb_record(b, st, slot);
         if (opts->output_verts_per_prim == 1) {
            nir_def *n = nir_load_var(b, st->prims_written[stream]);
            nir_store_global(b, slot, stream_index_entry(b, st, stream, n), .align_mul = 4,
                             .write_mask = 0x1);
            nir_def *written = nir_iadd_imm(b, n, 1);
            nir_store_var(b, st->prims_written[stream], written, 1);
            nir_store_global(b, written, stream_count_addr(b, st, stream), .align_mul = 4,
                             .write_mask = 0x1);
         }
      }
      nir_pop_if(b, NULL);
      return;
   }

   /* Refuse to run off the end of the slot we were given. A shader that emits more than it
    * declared is invalid, but it must not be allowed to scribble on the next primitive. */
   nir_push_if(b, nir_ult_imm(b, counter, opts->max_output_verts));
   {
      nir_def *slot = nir_iadd(b, st->out_vtx_base, counter);

      nir_foreach_shader_out_variable(var, b->shader) {
         const unsigned loc = var->data.location;

         if (loc == VARYING_SLOT_POS) {
            nir_def *clip = nir_load_var(b, var);
            nir_def *val = clip;
            if (!opts->hardpos && st->viewport_var) {
               /* The transform of the viewport this vertex names (clamped: an index past the
                * last viewport is undefined behaviour, a load past the table is not allowed). */
               nir_def *idx = nir_umin(b, nir_load_var(b, st->viewport_var),
                                       nir_imm_int(b, PANVK_GS_MAX_VIEWPORTS - 1));
               nir_def *xf = nir_iadd(b, st->viewports, nir_u2u64(b, nir_imul_imm(b, idx, 24)));
               nir_def *scale = load_components(b, xf, nir_imm_int(b, 0), 3, 32);
               nir_def *offset = load_components(b, xf, nir_imm_int(b, 12), 3, 32);
               val = clip_to_screen_xf(b, val, scale, offset);
            } else if (!opts->hardpos) {
               val = clip_to_screen(b, st, val);
            }
            if (!opts->hardpos)
               val = nir_bcsel(b, nir_ine_imm(b, st->out_clip_space, 0), clip, val);
            if (opts->hardpos) {
               /* Screen space, 64x64: (4,60) (32,60) (60,4). */
               nir_def *fc = nir_u2f32(b, counter);
               val = nir_vec4(b,
                              nir_fadd_imm(b, nir_fmul_imm(b, fc, 28.0), 4.0),
                              nir_bcsel(b, nir_ieq_imm(b, counter, 2),
                                        nir_imm_float(b, 4.0),
                                        nir_imm_float(b, 60.0)),
                              nir_imm_float(b, 0.5), nir_imm_float(b, 1.0));
            }
            store_components(b, st->out_pos, nir_imul_imm(b, slot, opts->pos_stride), val);
            continue;
         }

         /* gl_ViewportIndex goes on to the fragment shader, which discards outside that
          * viewport's scissor (PANVK_GS_VIEWPORT_INDEX_SLOT). */
         if (loc == VARYING_SLOT_VIEWPORT) {
            nir_def *offset = load_slot_offset(b, st->out_offsets, PANVK_GS_VIEWPORT_INDEX_SLOT);
            nir_push_if(b, nir_ine_imm(b, offset, PANVK_GS_SLOT_UNUSED));
            nir_def *byte = nir_iadd(b, nir_imul(b, slot, st->out_general_stride), offset);
            store_components(b, st->out_general, byte, nir_load_var(b, var));
            nir_pop_if(b, NULL);
            continue;
         }

         /* Point size lives in its own fp16 buffer, and the other built-ins (layer, clip
          * distances) have no path to the tiler here yet. */
         if (loc < VARYING_SLOT_VAR0 || var->data.compact)
            continue;

         /* One location per element of an output array, each at the component the variable
          * starts at (location_frac packs several outputs into one location). */
         const struct glsl_type *type = var->type;
         const unsigned elems = glsl_type_is_array(type) ? glsl_get_length(type) : 1;
         const struct glsl_type *elem_type = glsl_type_is_array(type) ? glsl_get_array_element(type)
                                                                      : type;
         if (!glsl_type_is_vector_or_scalar(elem_type))
            continue;

         for (unsigned e = 0; e < elems; e++) {
            nir_deref_instr *d = nir_build_deref_var(b, var);
            if (glsl_type_is_array(type))
               d = nir_build_deref_array_imm(b, d, e);
            nir_def *val = nir_load_deref(b, d);
            if (val->bit_size != 32)
               continue;

            nir_def *offset = load_slot_offset(b, st->out_offsets, loc + e);

            /* A slot the fragment shader does not read has no home in the generic buffer. */
            nir_push_if(b, nir_ine_imm(b, offset, PANVK_GS_SLOT_UNUSED));
            nir_def *byte = nir_iadd(b, nir_imul(b, slot, st->out_general_stride),
                                     nir_iadd_imm(b, offset, var->data.location_frac * 4));
            store_components(b, st->out_general, byte, val);
            nir_pop_if(b, NULL);
         }
      }

      if (st->xfb_dwords)
         stage_xfb_record(b, st, slot);

      /* Points: nir_lower_gs_intrinsics removes EndPrimitive outright, since every vertex is a
       * complete point, so the point is recorded here -- its index is its own slot. */
      if (opts->output_verts_per_prim == 1) {
         nir_def *addr = nir_iadd(b, st->out_index,
                                  nir_u2u64(b, nir_imul_imm(b, nir_iadd(b, st->out_idx_base,
                                                                        counter), 4)));
         nir_store_global(b, slot, addr, .align_mul = 4, .write_mask = 0x1);
         nir_def *written = nir_iadd_imm(b, counter, 1);
         nir_store_var(b, st->prims_written[0], written, 1);
         if (st->xfb_dwords) {
            nir_push_if(b, nir_ine_imm(b, st->xfb_counts, 0));
            nir_store_global(b, written,
                             nir_iadd(b, st->xfb_counts, nir_u2u64(b, nir_imul_imm(b, st->gid, 4))),
                             .align_mul = 4, .write_mask = 0x1);
            nir_pop_if(b, NULL);
         }
      }
   }
   nir_pop_if(b, NULL);
}

/*
 * EndPrimitive(), and the implicit one at the end of the shader. The vertices are already in
 * memory; what is written here is the index list that turns the strip the shader emitted into
 * the independent primitives the tiler will draw.
 */
static void
lower_end_primitive(nir_builder *b, nir_intrinsic_instr *intr, struct lower_gs_state *st)
{
   const struct panvk_gs_lower_options *opts = st->opts;
   const unsigned vpp = opts->output_verts_per_prim;
   const unsigned stream = nir_intrinsic_stream_id(intr);

   /* Another stream than 0 only feeds transform feedback. */
   if (stream != 0 && !st->xfb_dwords)
      return;

   /* nir_lower_gs_intrinsics passes end_primitive_with_counter(count, count_per_primitive):
    * src[0] is the RUNNING TOTAL of vertices emitted by this invocation so far, and src[1] is
    * the number of vertices in THIS primitive -- it is NOT a primitive index. Reading src[1]
    * as one put the first triangle's indices at slot (3+0)*3 = 9 of a 3-entry buffer: the
    * degenerate zeros the fill wrote were never overwritten, so every primitive was
    * zero-area, and the write landed 24 bytes past the allocation. */
   nir_def *vtx_total = intr->src[0].ssa;
   nir_def *vtx_in_prim = intr->src[1].ssa;

   /* Where this strip's vertices start, in this invocation's own vertex numbering. */
   nir_def *vtx_start = nir_isub(b, vtx_total, vtx_in_prim);

   /* A strip of N vertices expands to N - (vpp - 1) primitives. */
   nir_def *nprims =
      nir_imax(b, nir_iadd_imm(b, vtx_in_prim, -(int)(vpp - 1)), nir_imm_int(b, 0));

   nir_def *prim_base = nir_load_var(b, st->prims_written[stream]);

   nir_variable *i =
      nir_local_variable_create(b->impl, glsl_uint_type(), "gs_strip_i");
   nir_store_var(b, i, nir_imm_int(b, 0), 1);

   nir_push_loop(b);
   {
      nir_def *iv = nir_load_var(b, i);
      nir_push_if(b, nir_uge(b, iv, nprims));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_def *idx_entry = nir_imul_imm(b, nir_iadd(b, prim_base, iv), vpp);

      for (unsigned k = 0; k < vpp; k++) {
         /* Strip winding: every other triangle in a strip is reversed, and the tiler is drawing
          * an independent list, so the swap has to be baked into the indices here. */
         unsigned src = k;
         if (vpp == 3 && k != 0)
            src = k; /* order fixed up by the parity term below */

         nir_def *vbase = stream_vtx_slot(b, st, stream, nir_iadd(b, vtx_start, iv));
         nir_def *v = nir_iadd_imm(b, vbase, src);
         if (vpp == 3 && k != 0) {
            nir_def *odd = nir_iand_imm(b, iv, 1);
            nir_def *swapped = nir_iadd_imm(b, vbase, k == 1 ? 2 : 1);
            v = nir_bcsel(b, nir_ine_imm(b, odd, 0), swapped, v);
         }

         nir_def *addr = stream_index_entry(b, st, stream, nir_iadd_imm(b, idx_entry, k));
         nir_store_global(b, v, addr, .align_mul = 4, .write_mask = 0x1);
      }

      nir_store_var(b, i, nir_iadd_imm(b, iv, 1), 1);
   }
   nir_pop_loop(b, NULL);

   nir_store_var(b, st->prims_written[stream], nir_iadd(b, prim_base, nprims), 1);

   /* Transform feedback reads the count after the job, so keep it current at every EndPrimitive:
    * a shader that returns early still leaves the right number behind. */
   if (st->xfb_dwords) {
      nir_push_if(b, nir_ine_imm(b, st->xfb_counts, 0));
      nir_store_global(b, nir_iadd(b, prim_base, nprims), stream_count_addr(b, st, stream),
                       .align_mul = 4, .write_mask = 0x1);
      nir_pop_if(b, NULL);
   }
}

/*
 * Every index slot this invocation owns is filled with a degenerate primitive first, so the ones
 * the shader never reaches are discarded by the tiler rather than drawing whatever was in memory.
 * This is what makes the fixed worst-case allocation safe.
 */
static void
emit_degenerate_fill(nir_builder *b, struct lower_gs_state *st)
{
   const struct panvk_gs_lower_options *opts = st->opts;
   const unsigned total = opts->max_output_prims * opts->output_verts_per_prim;

   nir_variable *i = nir_local_variable_create(b->impl, glsl_uint_type(), "gs_fill_i");
   nir_store_var(b, i, nir_imm_int(b, 0), 1);

   nir_push_loop(b);
   {
      nir_def *iv = nir_load_var(b, i);
      nir_push_if(b, nir_uge_imm(b, iv, total));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      /* All three indices equal => zero area => culled. */
      nir_def *addr =
         nir_iadd(b, st->out_index,
                  nir_u2u64(b, nir_imul_imm(b, nir_iadd(b, st->out_idx_base, iv), 4)));
      nir_store_global(b, st->out_vtx_base, addr, .align_mul = 4, .write_mask = 0x1);

      nir_store_var(b, i, nir_iadd_imm(b, iv, 1), 1);
   }
   nir_pop_loop(b, NULL);

   nir_barrier(b, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
}

static bool
lower_gs_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_gs_state *st = data;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_per_vertex_input: {
      nir_def *val = lower_per_vertex_input(b, intr, st);
      nir_def_replace(&intr->def, val);
      return true;
   }
   case nir_intrinsic_emit_vertex_with_counter:
      lower_emit_vertex(b, intr, st);
      nir_instr_remove(&intr->instr);
      return true;
   case nir_intrinsic_end_primitive_with_counter:
      lower_end_primitive(b, intr, st);
      nir_instr_remove(&intr->instr);
      return true;
   case nir_intrinsic_set_vertex_and_primitive_count:
      /* The counts are fixed by the worst-case allocation, so the tiler does not need them. */
      nir_instr_remove(&intr->instr);
      return true;
   case nir_intrinsic_load_primitive_id:
      nir_def_replace(&intr->def, st->prim_id);
      return true;
   case nir_intrinsic_load_invocation_id:
      nir_def_replace(&intr->def, st->invocation_id);
      return true;
   case nir_intrinsic_load_tess_param_buffer_poly:
      nir_def_replace(&intr->def, st->tess_params);
      return true;
   default:
      return false;
   }
}

bool
panvk_nir_lower_gs(struct nir_shader *nir, const struct panvk_gs_lower_options *opts)
{
   assert(nir->info.stage == MESA_SHADER_GEOMETRY);

   /* A shader that returns ends its last primitive implicitly, and nir_lower_gs_intrinsics only
    * records the final counts there (set_vertex_and_primitive_count), so without this a strip the
    * shader never closed wrote no indices: a GS without a trailing EndPrimitive() drew nothing.
    * An EndPrimitive() right after another one ends an empty strip and writes nothing. */
   {
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_builder b = nir_builder_create(impl);
      nir_foreach_pred(pred, impl->end_block) {
         b.cursor = nir_after_block_before_jump(pred);
         u_foreach_bit(s, opts->stream_mask | 1)
            nir_end_primitive(&b, .stream_id = s);
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }

   /* Give every EmitVertex/EndPrimitive a running counter, and make an incomplete trailing
    * primitive discard itself rather than emit garbage. */
   /* nir_lower_gs_intrinsics sizes its per-stream counters from this, and nothing has gathered
    * it yet here. */
   nir->info.gs.active_stream_mask = opts->stream_mask | 1;
   NIR_PASS(_, nir, nir_lower_gs_intrinsics,
            nir_lower_gs_intrinsics_count_primitives |
               nir_lower_gs_intrinsics_count_vertices_per_primitive |
               nir_lower_gs_intrinsics_overwrite_incomplete |
               ((opts->stream_mask & ~1u) ? nir_lower_gs_intrinsics_per_stream : 0));

   /* NOT nir_lower_io_vars_to_temporaries here. That pass has the body write temporaries and
    * copies them into the real outputs once, at the end of the shader -- which is exactly wrong
    * for a geometry shader, where EmitVertex has to flush whatever the outputs hold RIGHT NOW,
    * possibly many times. The output variables are left alone so the body's stores land in them
    * and nir_load_var below reads back the current vertex.
    */

   /* SPIR-V hands us deref-based IO, and gl_in[i] is an array deref. Turn the inputs into
    * load_per_vertex_input intrinsics -- which is what the rewriting below matches on -- and
    * flatten the indirect derefs first so the vertex index is a constant. Outputs are left as
    * variables on purpose: EmitVertex has to read them back. */
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_in, UINT32_MAX);
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in, gs_glsl_type_size, 0);
   NIR_PASS(_, nir, nir_opt_constant_folding);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   struct lower_gs_state st = { .opts = opts };

   for (unsigned s = 0; s < 4; s++) {
      st.prims_written[s] =
         nir_local_variable_create(impl, glsl_uint_type(), "gs_prims_written");
      nir_store_var(&b, st.prims_written[s], nir_imm_int(&b, 0), 1);
   }

   nir_def *in_pos_direct = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, in_pos));
   nir_def *in_general_direct =
      load_param_u64(&b, opts, offsetof(struct panvk_gs_push, in_general));
   st.out_pos = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, out_pos));
   st.out_general = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, out_general));
   st.out_index = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, out_index));
   st.in_offsets = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, in_offsets));
   st.out_offsets = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, out_offsets));
   st.vp_scale =
      load_param_vec3f(&b, opts, offsetof(struct panvk_gs_push, viewport_scale));
   st.vp_offset =
      load_param_vec3f(&b, opts, offsetof(struct panvk_gs_push, viewport_offset));
   st.in_screen_space =
      load_param_u32(&b, opts, offsetof(struct panvk_gs_push, in_screen_space));
   st.out_clip_space =
      load_param_u32(&b, opts, offsetof(struct panvk_gs_push, out_clip_space));
   st.viewport_var =
      nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_VIEWPORT);
   if (st.viewport_var)
      st.viewports = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, viewports));
   st.in_general_stride =
      load_param_u32(&b, opts, offsetof(struct panvk_gs_push, in_general_stride));
   st.out_general_stride =
      load_param_u32(&b, opts, offsetof(struct panvk_gs_push, out_general_stride));
   st.tess_params = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, tess_params));

   nir_def *index_buf_direct = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, index_buf));
   st.index_size = load_param_u32(&b, opts, offsetof(struct panvk_gs_push, index_size));
   st.topology = load_param_u32(&b, opts, offsetof(struct panvk_gs_push, topology));
   nir_def *instance_stride_direct =
      load_param_u32(&b, opts, offsetof(struct panvk_gs_push, instance_stride));
   nir_def *ppi_direct =
      load_param_u32(&b, opts, offsetof(struct panvk_gs_push, prims_per_instance));
   nir_def *indirect_cmd =
      load_param_u64(&b, opts, offsetof(struct panvk_gs_push, indirect_cmd));

   /* One invocation per (input primitive, gl_InvocationID) pair, laid out one-dimensionally.
    * num_input_prims counts primitives over every instance; each pair owns its own output
    * slots, so the bases below are per compute invocation. */
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   const unsigned invocations = MAX2(opts->invocations, 1);
   nir_def *prim_global = gid;
   st.invocation_id = nir_imm_int(&b, 0);
   if (invocations > 1) {
      prim_global = nir_udiv_imm(&b, gid, invocations);
      st.invocation_id = nir_umod_imm(&b, gid, invocations);
   }

   nir_def *num_prims = load_param_u32(&b, opts, offsetof(struct panvk_gs_push, num_input_prims));

   st.out_vtx_base = nir_imul_imm(&b, gid, opts->max_output_verts);
   st.out_idx_base = nir_imul_imm(&b, gid,
                                  opts->max_output_prims * opts->output_verts_per_prim);

   /* Trailing invocations of the last workgroup own no output slots. */
   nir_push_if(&b, nir_uge(&b, prim_global, num_prims));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);

   /* Transform feedback: every dispatched invocation owns a count, and one that turns out to be
    * past an indirect draw's real primitive count must leave 0 there, so zero it first. */
   st.gid = gid;
   uint32_t xfb_table[PANVK_XFB_MAX_RECORD_DWORDS];
   const int xfb_dwords = panvk_gs_xfb_table(opts->xfb, xfb_table);
   if (xfb_dwords < 0)
      return false;
   st.xfb_dwords = xfb_dwords;
   if (st.xfb_dwords) {
      st.xfb_counts = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, xfb_counts));
      st.xfb_staging = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, xfb_staging));
      st.stream_slots = load_param_u32(&b, opts, offsetof(struct panvk_gs_push, xfb_stream_slots));
      st.stream_idx = load_param_u32(&b, opts, offsetof(struct panvk_gs_push, xfb_stream_idx));
      st.stream_index = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, xfb_stream_index));
      st.xfb_invocations =
         load_param_u32(&b, opts, offsetof(struct panvk_gs_push, xfb_invocations));
      nir_push_if(&b, nir_ine_imm(&b, st.xfb_counts, 0));
      u_foreach_bit(s, opts->stream_mask | 1)
         nir_store_global(&b, nir_imm_int(&b, 0), stream_count_addr(&b, &st, s), .align_mul = 4,
                          .write_mask = 0x1);
      nir_pop_if(&b, NULL);
   }

   /* An indirect draw: the real parameters are in the argument buffer, and where the vertex
    * shader wrote is in the descriptors the draw helper patched. */
   nir_def *prims_per_instance, *actual_prims;
   nir_def *bias_direct = load_direct_bias(&b, opts, st.index_size);
   nir_push_if(&b, nir_ine_imm(&b, indirect_cmd, 0));
   nir_def *ppi_ind, *total_ind, *index_buf_ind, *stride_ind, *bias_ind, *pos_ind, *gen_ind;
   {
      const unsigned vpp = opts->input_verts_per_prim;
      nir_def *n = nir_load_global(&b, 1, 32, indirect_cmd, .align_mul = 4);
      const struct indirect_inputs in =
         load_indirect_inputs(&b, opts, indirect_cmd, index_buf_direct, st.index_size);

      nir_def *list = nir_udiv_imm(&b, n, vpp);
      nir_def *strip = nir_bcsel(&b, nir_uge_imm(&b, n, vpp), nir_iadd_imm(&b, n, -(int)(vpp - 1)),
                                 nir_imm_int(&b, 0));
      nir_def *fan = nir_bcsel(&b, nir_uge_imm(&b, n, 3), nir_iadd_imm(&b, n, -2), nir_imm_int(&b, 0));
      nir_def *tsa = nir_bcsel(&b, nir_uge_imm(&b, n, 6), nir_udiv_imm(&b, nir_iadd_imm(&b, n, -4), 2),
                               nir_imm_int(&b, 0));
      ppi_ind = nir_bcsel(&b, nir_ieq_imm(&b, st.topology, PANVK_GS_TOPO_STRIP), strip,
                          nir_bcsel(&b, nir_ieq_imm(&b, st.topology, PANVK_GS_TOPO_FAN), fan,
                                    nir_bcsel(&b, nir_ieq_imm(&b, st.topology,
                                                              PANVK_GS_TOPO_TRI_STRIP_ADJ),
                                              tsa, list)));
      total_ind = nir_umin(&b, nir_imul(&b, ppi_ind, in.inst), num_prims);
      index_buf_ind = in.index_buf;
      bias_ind = in.bias;
      stride_ind = in.stride;
      pos_ind = in.pos;
      gen_ind = in.gen;
   }
   nir_pop_if(&b, NULL);
   prims_per_instance = nir_if_phi(&b, ppi_ind, ppi_direct);
   actual_prims = nir_if_phi(&b, total_ind, num_prims);
   st.index_buf = nir_if_phi(&b, index_buf_ind, index_buf_direct);
   st.instance_stride = nir_if_phi(&b, stride_ind, instance_stride_direct);
   st.index_bias = nir_if_phi(&b, bias_ind, bias_direct);
   st.in_pos = nir_if_phi(&b, pos_ind, in_pos_direct);
   st.in_general = nir_if_phi(&b, gen_ind, in_general_direct);

   /* A count only the GPU knows (the tessellator's index count). The address is 0 otherwise,
    * and a load from it is an MMU fault. After the phis above: nir_if_phi merges the if just
    * closed, and one in between made every indirect draw read its direct placeholders. */
   nir_def *count_addr = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, num_prims_addr));
   nir_push_if(&b, nir_ine_imm(&b, count_addr, 0));
   nir_def *counted = nir_umin(&b, nir_load_global(&b, 1, 32, count_addr, .align_mul = 4),
                               num_prims);
   nir_pop_if(&b, NULL);
   actual_prims = nir_if_phi(&b, counted, actual_prims);

   /* gl_PrimitiveIDIn restarts with every instance. */
   nir_def *ppi = nir_umax(&b, prims_per_instance, nir_imm_int(&b, 1));
   st.prims_per_instance = ppi;
   st.instance = nir_udiv(&b, prim_global, ppi);
   st.prim_id = nir_isub(&b, prim_global, nir_imul(&b, st.instance, ppi));

   emit_degenerate_fill(&b, &st);

   /* Invocation 0 tells the tiler how many indices there really are (stored minus one). No
    * primitive at all still leaves one degenerate one to draw. */
   nir_def *tiler_count_addr =
      load_param_u64(&b, opts, offsetof(struct panvk_gs_push, tiler_index_count));
   nir_push_if(&b, nir_iand(&b, nir_ieq_imm(&b, gid, 0), nir_ine_imm(&b, tiler_count_addr, 0)));
   {
      const unsigned per_invocation = opts->max_output_prims * opts->output_verts_per_prim;
      nir_def *count = nir_imul_imm(&b, actual_prims, invocations * per_invocation);
      count = nir_umax(&b, count, nir_imm_int(&b, MAX2(opts->output_verts_per_prim, 1)));
      nir_store_global(&b, nir_iadd_imm(&b, count, -1), tiler_count_addr, .align_mul = 4,
                       .write_mask = 0x1);
   }
   nir_pop_if(&b, NULL);

   /* Past the indirect draw's real primitive count: the slots stay degenerate. */
   nir_push_if(&b, nir_uge(&b, prim_global, actual_prims));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);


   bool progress = nir_shader_intrinsics_pass(
      nir, lower_gs_instr, nir_metadata_none, &st);

   /* THE OUTPUT VARIABLES MUST STOP BEING OUTPUTS HERE.
    * They were deliberately left as nir_var_shader_out so that the body's stores landed in
    * them and lower_emit_vertex() could read the CURRENT vertex back with nir_load_var(). That
    * is done now -- every EmitVertex has already been turned into explicit global stores. Left
    * as shader_out they reach the Bifrost backend as store_deref/load_deref, which it cannot
    * compile: it prints "Unhandled intrinsic store_deref" to stderr, drops the instruction and
    * carries on, so the shader runs, reports DONE, and writes nothing. That is the whole of the
    * "the position stores never land" symptom -- there were no stores.
    *
    * Retyping them to shader_temp and lowering to SSA is what makes the body's writes visible
    * to the nir_load_var() reads. nir_fixup_deref_modes is required because the deref
    * instructions cache the mode they were built with. */
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_shader_out)
      var->data.mode = nir_var_shader_temp;
   nir->info.outputs_written = 0;
   nir_fixup_deref_modes(nir);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_dce);

   /* It is a compute shader from here on. */
   nir->info.stage = MESA_SHADER_COMPUTE;
   nir->info.workgroup_size[0] = PANVK_GS_WORKGROUP_SIZE;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;

   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_opt_dce);

   return progress;
}

/* -- TESSELLATION ------------------------------------------------------------------------------
 * See panvk_gs.h. */

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

static bool
lower_tess_params_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_tess_param_buffer_poly)
      return false;
   nir_def_replace(&intr->def, (nir_def *)data);
   return true;
}

bool
panvk_nir_lower_tess_params(nir_shader *nir, const struct panvk_gs_lower_options *opts)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *p = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, tess_params));
   return nir_shader_intrinsics_pass(nir, lower_tess_params_instr, nir_metadata_control_flow, p);
}

bool
panvk_nir_lower_tcs_inputs(nir_shader *nir, const struct panvk_gs_lower_options *opts)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   struct lower_gs_state st = {.opts = opts};

   nir_def *in_pos = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, in_pos));
   nir_def *in_general = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, in_general));
   st.in_offsets = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, in_offsets));
   st.in_general_stride =
      load_param_u32(&b, opts, offsetof(struct panvk_gs_push, in_general_stride));
   st.vp_scale = load_param_vec3f(&b, opts, offsetof(struct panvk_gs_push, viewport_scale));
   st.vp_offset = load_param_vec3f(&b, opts, offsetof(struct panvk_gs_push, viewport_offset));
   st.in_screen_space =
      load_param_u32(&b, opts, offsetof(struct panvk_gs_push, in_screen_space));
   nir_def *index_buf = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, index_buf));
   st.index_size = load_param_u32(&b, opts, offsetof(struct panvk_gs_push, index_size));
   st.topology = nir_imm_int(&b, PANVK_GS_TOPO_LIST);
   nir_def *instance_stride =
      load_param_u32(&b, opts, offsetof(struct panvk_gs_push, instance_stride));
   st.prims_per_instance =
      load_param_u32(&b, opts, offsetof(struct panvk_gs_push, prims_per_instance));
   st.vpp_dyn = load_param_u32(&b, opts, offsetof(struct panvk_gs_push, patch_size));

   /* An indirect draw, read as panvk_nir_lower_gs reads one. The grid is right already:
    * panlib_tess_setup_indirect patched it. */
   nir_def *indirect_cmd = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, indirect_cmd));
   nir_def *bias_direct = load_direct_bias(&b, opts, st.index_size);
   nir_push_if(&b, nir_ine_imm(&b, indirect_cmd, 0));
   const struct indirect_inputs in =
      load_indirect_inputs(&b, opts, indirect_cmd, index_buf, st.index_size);
   nir_pop_if(&b, NULL);
   st.index_buf = nir_if_phi(&b, in.index_buf, index_buf);
   st.index_bias = nir_if_phi(&b, in.bias, bias_direct);
   st.instance_stride = nir_if_phi(&b, in.stride, instance_stride);
   st.in_pos = nir_if_phi(&b, in.pos, in_pos);
   st.in_general = nir_if_phi(&b, in.gen, in_general);
   st.tess_params = load_param_u64(&b, opts, offsetof(struct panvk_gs_push, tess_params));

   /* One workgroup per patch: x the patch within its instance, y the instance. */
   nir_def *wg = nir_load_workgroup_id(&b);
   st.prim_id = nir_channel(&b, wg, 0);
   st.instance = nir_channel(&b, wg, 1);

   return nir_shader_intrinsics_pass(nir, lower_tcs_instr, nir_metadata_none, &st);
}

void
panvk_nir_tes_to_gs(nir_shader *nir)
{
   nir->info.stage = MESA_SHADER_GEOMETRY;
   memset(&nir->info.gs, 0, sizeof(nir->info.gs));
   nir->info.gs.input_primitive = MESA_PRIM_POINTS;
   nir->info.gs.output_primitive = MESA_PRIM_POINTS;
   nir->info.gs.vertices_in = 1;
   nir->info.gs.vertices_out = 1;
   nir->info.gs.invocations = 1;
   nir->info.gs.active_stream_mask = 1;

   /* The outputs are copied out of their temporaries at the very end of the shader
    * (nir_lower_io_vars_to_temporaries in panvk_preprocess_nir), so a single EmitVertex after
    * everything flushes the finished vertex. */
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_emit_vertex(&b, 0);
   nir_progress(true, impl, nir_metadata_none);
}
