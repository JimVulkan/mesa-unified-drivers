/*
 * Copyright 2025 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <stdatomic.h>
#include <stdint.h>
#include "compiler/libcl/libcl.h"
#include "genxml/gen_macros.h"

#pragma once

#if (PAN_ARCH == 6 || PAN_ARCH == 7)
/* Keep in sync with panvk_varying_buf_id */
enum panlib_varying_buf_id {
   PANLIB_VARY_BUF_GENERAL,
   PANLIB_VARY_BUF_POSITION,
   PANLIB_VARY_BUF_PSIZ,

   /* Keep last */
   PANLIB_VARY_BUF_MAX,
};

struct libpan_draw_helper_index_min_max_result {
    uint32_t min;
    uint32_t max;
};

struct libpan_draw_helper_varying_buf_info {
    uint64_t address;
    uint32_t size;
    atomic_uint offset;
};

struct libpan_draw_helper_attrib_buf_info {
    uint32_t divisor;
    uint32_t stride;
    bool per_instance;
};

struct libpan_draw_helper_attrib_info {
    uint32_t base_offset;

    /* When the attribute is per instance, the stride otherwise 0 */
    uint32_t stride;
};

/* One indirect draw (or indexed draw rerouted through the indirect path) of a batch. The CPU
 * writes one per draw and layer; panlib_draw_batch_minmax and panlib_draw_batch_patch run once
 * for all of them. Pointers are GPU addresses, 0 when absent. */
struct libpan_draw_batch_entry {
    uint64_t cmd;                  /* VkDraw(Indexed)IndirectCommand */
    uint64_t index_buffer_ptr;
    uint64_t index_min_out;        /* uint32_t: the index minimum, for the GS emulation */
    uint64_t varying_bufs_descs;
    uint64_t varying_bufs_info;
    uint64_t attrib_bufs_descs;
    uint64_t attrib_bufs_infos;
    uint64_t attribs_descs;
    uint64_t attribs_infos;
    uint64_t first_vertex_sysval;
    uint64_t first_instance_sysval;
    uint64_t raw_vertex_offset_sysval;
    uint64_t idvs_job;
    uint64_t vertex_job;
    uint64_t tiler_job;
    uint32_t index_size;           /* 0: not indexed */
    uint32_t primitive_restart;
    uint32_t primitive_vertex_count;
    uint32_t attrib_bufs_valid;
    uint32_t attribs_valid;
    uint32_t index_min;            /* scan result: UINT32_MAX and 0 before the scan and after */
    uint32_t index_max;            /* the patch, so the command buffer can run again */
    uint32_t pad;
};

struct libpan_draw_batch {
    uint64_t entries;              /* struct libpan_draw_batch_entry[count] */
    uint32_t count;
    uint32_t pad;
};

/* Transform feedback captured from the geometry shader emulation, one per capturing draw. The draw
 * path fills everything above `start`; panlib_xfb_scan fills the rest, panlib_xfb_copy reads it. */
struct libpan_xfb_draw {
    uint64_t buffer[4];      /* bound address */
    uint32_t size[4];        /* bound size in bytes */
    uint32_t stride[4];      /* the shader's xfb stride, 0 = not captured to */
    uint64_t offsets_in;     /* uint32_t[4]: write offsets before this draw */
    uint64_t offsets_out;    /* uint32_t[4]: after it */
    uint64_t counts;         /* uint32_t per emulation invocation: primitives it emitted */
    uint64_t first;          /* uint32_t per emulation invocation: its first global primitive */
    uint64_t indices;        /* the emulation's index buffer */
    uint64_t staging;        /* record_dwords per output vertex slot */
    uint64_t table;          /* uint32_t per record dword: buffer << 16 | dword in the vertex */
    uint32_t invocations;    /* emulation invocations dispatched */
    uint32_t idx_per_invocation;
    uint32_t vpp;            /* vertices per output primitive */
    uint32_t record_dwords;
    uint32_t start[4];       /* scan: byte offset this draw writes each buffer from */
    uint32_t written;        /* scan: primitives captured */
    uint32_t needed;         /* scan: primitives emitted */
};
#endif
