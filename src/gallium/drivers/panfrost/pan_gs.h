/*
 * Copyright 2026 XclipseTools
 * SPDX-License-Identifier: MIT
 */

#ifndef PAN_GS_H
#define PAN_GS_H

/*
 * Geometry shaders on the Bifrost job manager (v6/v7), which has no geometry stage. The same
 * emulation panvk uses (src/panfrost/vulkan/panvk_nir_lower_gs.c), adapted to gallium's lowered
 * IO: the geometry shader becomes a COMPUTE job between the vertex job and the tiler job. It reads
 * the varying records the vertex shader wrote (positions left in clip space), runs once per input
 * primitive and gl_InvocationID, and writes a fresh set of varying buffers for the tiler plus an
 * index buffer that expands the strips it emitted into independent primitives. Every invocation
 * owns a worst-case slice of the output (max_vertices), and slots it does not fill stay
 * degenerate, so the tiler job's counts are known when the draw is recorded.
 */

#include <stdbool.h>
#include <stdint.h>

#include "compiler/shader_enums.h"

struct nir_shader;

#define PAN_GS_WORKGROUP_SIZE 64
#define PAN_GS_SLOT_UNUSED    0xffffffffu
/* In pan_gs_params::in_offsets: the vertex shader stored this slot as 16-bit. */
#define PAN_GS_IN_16BIT       (1u << 31)

/* How the draw's vertices assemble into the geometry shader's input primitives. */
enum pan_gs_topology {
   PAN_GS_TOPO_LIST = 0,
   /* primitive p is vertices p .. p + n - 1 */
   PAN_GS_TOPO_STRIP = 1,
   /* triangle p is vertices p + 1, p + 2, 0 */
   PAN_GS_TOPO_FAN = 2,
   /* the strip is the even vertices, see gs_input_record() */
   PAN_GS_TOPO_TRI_STRIP_ADJ = 3,
   /* a line strip closed by a last line from vertex n - 1 back to 0 */
   PAN_GS_TOPO_LOOP = 4,
};

/* Per-draw parameters, in memory at the address of the GS_PARAMS sysval. Field order is
 * load-bearing: pan_nir_lower_gs.c reads them with offsetof(). */
struct pan_gs_params {
   /* The vertex shader's varying buffers: clip-space vec4 positions at stride 16, fp16 point
    * sizes at stride 2 (0 when it writes none), and the generic buffer. */
   uint64_t in_pos;
   uint64_t in_psiz;
   uint64_t in_general;
   /* uint32_t[64], by varying slot: the byte offset of the slot in the vertex shader's generic
    * record (| PAN_GS_IN_16BIT for a 16-bit slot), PAN_GS_SLOT_UNUSED when it writes none. */
   uint64_t in_offsets;
   /* The tiler's buffers: screen-space positions, point sizes, the generic buffer
    * (pan_gs_info::out_stride) and the index buffer. */
   uint64_t out_pos;
   uint64_t out_psiz;
   uint64_t out_general;
   uint64_t out_index;
   /* The draw's first index, 0 for a draw without indices. */
   uint64_t index_buf;
   uint32_t in_general_stride;
   uint32_t index_size;
   uint32_t topology;
   uint32_t prims_per_instance;
   /* Records between instances (the vertex job's padded count), 0 without instancing. */
   uint32_t instance_stride;
   /* Input primitives over every instance. */
   uint32_t num_input_prims;
   /* Subtracted from an index to get its vertex record: the vertex job's first index. */
   uint32_t index_bias;
   /* Vertices per instance, for PAN_GS_TOPO_LOOP. */
   uint32_t vertex_count;
   /* The first vertex provokes (GL_FIRST_VERTEX_CONVENTION): odd strip triangles keep it
    * first, otherwise last. */
   uint32_t flatshade_first;
   /* A vertex the draw path placed far outside the viewport. Every index slot the shader does
    * not reach points at it: equal indices alone would still draw a point. */
   uint32_t dead_vertex;
   /* Vertices per input primitive of the tess control stage: the patch size. */
   uint32_t patch_size;
   /* When non-zero, the real number of input primitives is min(*num_prims_addr,
    * num_input_prims): the tess eval stage runs on the tessellator's index count. */
   uint64_t num_prims_addr;
   /* A shader writing gl_ViewportIndex: PAN_GS_MAX_VIEWPORTS x {scale xyz, offset xyz}, the
    * transform each output vertex gets by its index. */
   uint64_t viewports;
};

#define PAN_GS_MAX_VIEWPORTS 16

/* gl_ViewportIndex and gl_Layer on their way to the fragment shader. Both are hardware varyings in
 * pan's layout, which the emulation's buffers do not feed, so they travel in slots GL never uses
 * as fragment inputs (clip and cull distances are merged into CLIP_DIST0/1) and pan lays out as
 * generic. The fragment shader's reads are moved to them (panfrost_shader_compile). */
#define PAN_GS_VIEWPORT_SLOT VARYING_SLOT_CULL_DIST0
#define PAN_GS_LAYER_SLOT    VARYING_SLOT_CULL_DIST1

/* What the draw path needs to know about a geometry shader, gathered when it is created. */
struct pan_gs_info {
   enum mesa_prim input_prim;
   enum mesa_prim output_prim;
   uint8_t input_verts;
   uint8_t output_verts;
   uint16_t max_verts;
   uint16_t max_prims;
   uint16_t invocations;
   bool writes_psiz;
   /* gl_ViewportIndex: each vertex gets its viewport's transform, and the fragment shader
    * discards outside that viewport (the tiler has one scissor for all of them). */
   bool writes_viewport;
   /* Output slots laid out in the generic buffer, 16 bytes each in slot order. */
   uint64_t generic_mask;
   uint32_t out_stride;
   /* The shader's own clip/cull distance counts (cull distances become per-fragment markers, as
    * the vertex shader's do). */
   uint8_t clip_count;
   uint8_t cull_count;
};

void pan_gs_gather_info(const struct nir_shader *nir, struct pan_gs_info *info);

/* Byte offset of an output slot in the generic record, PAN_GS_SLOT_UNUSED if none. */
static inline uint32_t
pan_gs_out_offset(const struct pan_gs_info *info, unsigned slot)
{
   if (slot >= 64 || !(info->generic_mask & (1ull << slot)))
      return PAN_GS_SLOT_UNUSED;
   return 16 * __builtin_popcountll(info->generic_mask & ((1ull << slot) - 1));
}

/*
 * Tessellation, on the same machinery. The tess control shader runs as a compute job after the
 * vertex job, one workgroup per patch (workgroup_id.x the patch, .y the instance) and one
 * invocation per output vertex; its per-vertex inputs are the vertex shader's varying records,
 * read as a LIST of pan_gs_params::patch_size vertices, and everything else is src/poly's
 * lowering. The tessellator runs as libpan kernels, and the tess eval shader becomes a point-in,
 * point-out geometry shader run once per tessellator index, whose identity index buffer the
 * tiler draws with the tessellated primitive type.
 */
bool pan_nir_lower_tcs_inputs(struct nir_shader *nir);
void pan_nir_tes_to_gs(struct nir_shader *nir);

/* Lowers a geometry shader (gallium lowered IO) into the compute shader described above.
 * noperspective: output slots VAR0 + i the fragment shader interpolates without perspective. */
bool pan_nir_lower_gs(struct nir_shader *nir, const struct pan_gs_info *info,
                      uint32_t noperspective);

#endif /* PAN_GS_H */
