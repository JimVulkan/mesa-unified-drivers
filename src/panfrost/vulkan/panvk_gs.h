/*
 * Copyright 2026 XclipseTools
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_GS_H
#define PANVK_GS_H

/* Geometry shader emulation. Arch-independent on purpose: the lowering is pure NIR, and this
 * header carries only the contract between it and the draw path, so neither has to be built
 * per-arch to see it. */

#include <stdbool.h>
#include <stdint.h>

#include "compiler/shader_enums.h"

struct nir_shader;
struct nir_xfb_info;

/* --- geometry shader emulation -------------------------------------------------------------
 *
 * Mali has no geometry stage, so a geometry shader is compiled into a COMPUTE shader that runs
 * between the vertex job and the tiler job: it reads the varying buffers the vertex shader wrote
 * and writes a fresh set for the tiler, plus an index buffer expanding the strips it emitted into
 * independent primitives. See panvk_nir_lower_gs.c for why the allocation is worst-case.
 */
#define PANVK_GS_WORKGROUP_SIZE 64
#define PANVK_GS_SLOT_UNUSED    UINT32_MAX

/* How the draw's vertices assemble into the geometry shader's input primitives. Lists, and the
 * list-with-adjacency topologies, are LIST with the shader's own vertices-per-primitive. */
/* An indirect draw's parameters exist only on the GPU, so the emulation is dispatched for this
 * many input primitives (times gl_InvocationID) with outputs sized to match, and each invocation
 * reads the real count out of the draw's argument buffer. Primitives past the cap are dropped. */
#define PANVK_GS_INDIRECT_MAX_PRIMS 4096

enum panvk_gs_topology {
   PANVK_GS_TOPO_LIST = 0,
   /* primitive p is vertices p .. p + n - 1; triangle strips swap the last two on odd p */
   PANVK_GS_TOPO_STRIP = 1,
   /* triangle p is vertices p + 1, p + 2, 0 */
   PANVK_GS_TOPO_FAN = 2,
   /* triangle strip with adjacency: the strip is the even vertices, see gs_input_record() */
   PANVK_GS_TOPO_TRI_STRIP_ADJ = 3,
};

/* Push-constant block the lowered shader reads its addresses out of. Field order is load-bearing:
 * panvk_nir_lower_gs.c addresses it with offsetof(). */
struct panvk_gs_push {
   uint64_t in_pos;
   uint64_t in_general;
   uint64_t out_pos;
   uint64_t out_general;
   uint64_t out_index;
   /* slot -> byte offset inside the generic buffer, indexed by gl_varying_slot, 0xffffffff for a
    * slot nothing reads. Tables rather than baked constants because a geometry shader is compiled
    * before it is linked: the input offsets belong to the vertex shader's layout and the output
    * offsets to whatever the fragment shader ends up reading, and neither is known here. */
   uint64_t in_offsets;
   uint64_t out_offsets;
   uint32_t in_general_stride;
   uint32_t out_general_stride;
   uint32_t num_input_prims;
   uint32_t pad;
   /* The viewport transform, so the emulation can UNDO what the vertex shader applied and
    * re-apply it on the way out -- see panvk_nir_lower_gs.c. Padded to vec4 because these are
    * read as float3 out of the push-constant block.
    *
    * They cannot come from nir_load_viewport_scale/offset: panvk lowers those to the
    * GRAPHICS sysval block, and this shader runs as COMPUTE with a compute sysval block. */
   float viewport_scale[3];
   float pad2;
   float viewport_offset[3];
   float pad3;
   /* Input assembly. The vertex shader wrote one varying record per vertex it ran: for an
    * indexed draw that is one per index VALUE (min_index is 0), for a plain draw one per vertex
    * in draw order, and each instance's records follow the previous instance's at
    * instance_stride. index_buf is the draw's first index, or 0 for a plain draw. */
   uint64_t index_buf;
   uint32_t index_size;
   uint32_t topology;
   uint32_t prims_per_instance;
   uint32_t instance_stride;
   /* Indirect draws (indirect_cmd != 0): the Vk*IndirectCommand, the vertex job's varying buffer
    * descriptors as the draw helper patched them (where the vertex shader really wrote), its DRAW
    * section (Instance Size is the per-instance record stride), and for an indexed draw the
    * smallest index the search helper found (the vertex job starts at it). index_buf is then the
    * index buffer's base: firstIndex comes from the command. num_input_prims is the cap. */
   uint64_t indirect_cmd;
   uint64_t in_pos_desc;
   uint64_t in_general_desc;
   uint64_t vertex_dcd;
   uint64_t index_min;
   /* The tiler job's PRIMITIVE Index count word: the emulation writes the real count there, so
    * the tiler walks the primitives that exist rather than the whole capped buffer. */
   uint64_t tiler_index_count;
   /* Transform feedback, 0 when the draw is not capturing. xfb_counts: one uint32_t per compute
    * invocation, the complete primitives it emitted. xfb_staging: one record per output vertex
    * slot, the captured components packed in panvk_gs_xfb_table() order. The copy into the
    * application's buffers happens after this job, in draw order -- see panlib_xfb_scan. */
   uint64_t xfb_counts;
   uint64_t xfb_staging;
   /* The polygon-mode geometry shader culls faces itself (PANVK_GS_CULL_*), since the tiler
    * only sees the lines or points it emits. */
   uint32_t polygon_cull;
   /* The vertex shader was compiled for the tiler, not for this emulation (the polygon-mode
    * shader stands in behind any vertex shader): POSITION holds screen-space xyz and 1/w, and the
    * emulation undoes the viewport transform on the way in. */
   uint32_t in_screen_space;
   /* A shader writing gl_ViewportIndex: PANVK_GS_MAX_VIEWPORTS x {scale xyz, offset xyz}, the
    * transform each output vertex gets by its index. 0 otherwise. */
   uint64_t viewports;
   /* Tessellation: the draw's struct poly_tess_params, read by the tess control and tess eval
    * stages (nir_load_tess_param_buffer_poly). */
   uint64_t tess_params;
   /* When non-zero, where the real number of input primitives is: min(*num_prims_addr,
    * num_input_prims) run. The tess eval stage reads the tessellator's index count here. */
   uint64_t num_prims_addr;
   /* Vertices per input primitive when the shader was compiled for a variable count (the tess
    * control stage: patchControlPoints). */
   uint32_t patch_size;
   /* Leave output positions in clip space: the tess eval stage when a geometry shader reads
    * them, which a screen-space round trip would move by a rounding step. */
   uint32_t out_clip_space;
   /* geometryStreams. Every stream the shader emits on owns a copy of the output vertex slots
    * (xfb_stream_slots per stream, stream s at s * xfb_stream_slots) and, for capture, its own
    * primitive index list and counts; only stream 0 is rasterised. xfb_stream_index holds the
    * index lists of streams 1-3, xfb_stream_idx entries each (stream 0's is out_index), and the
    * counts of stream s start at s * xfb_invocations in xfb_counts. */
   uint64_t xfb_stream_index;
   uint32_t xfb_stream_slots;
   uint32_t xfb_stream_idx;
   uint32_t xfb_invocations;
   uint32_t pad4;
};

/* gl_ViewportIndex, carried from the emulation to the fragment shader. VARYING_SLOT_VIEWPORT is a
 * hardware attribute varying in pan's layout, which the emulation's buffer does not feed, so the
 * index travels in a slot Vulkan never uses and pan lays out as generic. */
#define PANVK_GS_VIEWPORT_INDEX_SLOT VARYING_SLOT_CLIP_VERTEX
#define PANVK_GS_MAX_VIEWPORTS       16

#define PANVK_GS_CULL_FRONT     (1u << 0)
#define PANVK_GS_CULL_BACK      (1u << 1)
#define PANVK_GS_FRONT_CCW      (1u << 2)

/* --- transform feedback ---------------------------------------------------------------------
 *
 * Captured from the geometry shader emulation, which already assembles primitives. Every captured
 * 32-bit component is one dword of a per-vertex staging record; the table entry for record dword
 * j says where it goes: xfb buffer (bits 16+) and the dword inside that buffer's vertex (bits
 * 0-15). 128 dwords is every component of 32 output locations. */
#define PANVK_XFB_MAX_RECORD_DWORDS 128

/* Fills table (PANVK_XFB_MAX_RECORD_DWORDS entries) and returns the record size in dwords, 0 for a
 * shader that captures nothing, or -1 for one this emulation cannot capture. */
int panvk_gs_xfb_table(const struct nir_xfb_info *xfb, uint32_t *table);

struct panvk_gs_lower_options {
   unsigned input_verts_per_prim;
   unsigned output_verts_per_prim;
   unsigned max_output_verts;
   unsigned max_output_prims;
   /* gl_InvocationID count: every (input primitive, invocation) pair is its own compute
    * invocation with its own output slots. */
   unsigned invocations;

   unsigned pos_stride;
   /* PANVK_GS_HARDPOS: emit a fixed screen-space triangle instead of the shader's own
    * gl_Position, to tell a bad INPUT READ apart from a bad STORE. A bisect, not a feature. */
   bool hardpos;
   /* PANVK_GS_ECHO: a per-vertex input load returns the ADDRESS it was about to read instead of
    * the data, so the dump shows whether in_pos arrived and whether the arithmetic is right. */
   bool echo;
   bool has_psiz;

   /* Where panvk_gs_push lives in the push-constant/FAU space. Added to every offsetof() so the
    * pass does not need to know panvk's per-arch sysval layout. */
   unsigned params_base;

   /* Transform feedback: stage a capture record at every EmitVertex (panvk_gs_xfb_table()). */
   const struct nir_xfb_info *xfb;
   /* The vertex streams the shader emits on (nir->info.gs.active_stream_mask). */
   unsigned stream_mask;
};

bool panvk_nir_lower_gs(struct nir_shader *nir, const struct panvk_gs_lower_options *opts);

/* --- tessellation ----------------------------------------------------------------------------
 *
 * The tess control shader runs as a compute shader after the vertex job: one workgroup per patch
 * (workgroup_id.x the patch, .y the instance), one invocation per output vertex. Its per-vertex
 * inputs are the vertex shader's varying records, read exactly as the geometry shader emulation
 * reads them (a LIST of patch_size vertices); everything else is src/poly's lowering. The tess
 * eval shader becomes a point-in, point-out geometry shader run once per tessellator index. */

/* The device heap the tessellator allocates from, and the most indices one draw's tess eval job
 * runs: its output buffers are sized for this before the GPU knows the real count. */
#define PANVK_TESS_HEAP_SIZE    (64u << 20)
#define PANVK_TESS_MAX_INDICES  (1u << 18)

/* The tessellator's grid for an indirect draw, whose patch count only the GPU knows: each thread
 * walks the patches at this stride. */
#define PANVK_TESS_INDIRECT_GRID 1024

/* Workgroup size of the tess eval compute job, which is the geometry shader emulation's. */
#define PANVK_TES_WORKGROUP_SIZE PANVK_GS_WORKGROUP_SIZE

/* opts: input_verts_per_prim 0 (variable, panvk_gs_push::patch_size), pos_stride, params_base. */
bool panvk_nir_lower_tcs_inputs(struct nir_shader *nir, const struct panvk_gs_lower_options *opts);

/* nir_load_tess_param_buffer_poly -> panvk_gs_push::tess_params, for the loads src/poly adds. */
bool panvk_nir_lower_tess_params(struct nir_shader *nir, const struct panvk_gs_lower_options *opts);

/* A tess eval shader whose inputs src/poly already lowered, into the geometry shader emulation's
 * input form: stage GEOMETRY, one point in, one point out, EmitVertex at the end. */
void panvk_nir_tes_to_gs(struct nir_shader *nir);

/* PANVK_GS_DUMP: the draw path stashes the emulation's output position buffer here so the queue
 * can print it once the GPU has reported completion (pan_kmod_kbase_submit blocks). The only way
 * to see what the job actually wrote -- judging it from a rendered image cannot tell a bad value
 * apart from a store that never happened. Bisect scaffolding, not a feature. */
extern void *panvk_gs_dbg_pos_cpu;
extern uint64_t panvk_gs_dbg_pos_gpu;
extern uint32_t panvk_gs_dbg_pos_verts;
extern void *panvk_gs_dbg_in_pos_cpu;
extern void *panvk_gs_dbg_idx_cpu;
extern uint32_t panvk_gs_dbg_idx_count;

#endif /* PANVK_GS_H */
