/*
 * Copyright 2026 JimVulkan
 * SPDX-License-Identifier: MIT
 */

/* GPU side of the Xclipse field profiler for RADV (util/u_xclipse_prof.c does the CPU side and
 * writes the report). Off unless MESA_XCLIPSE_PROF / debug.mesa_xclipse_prof is set. */
#ifndef RADV_XCLIPSE_PROF_H
#define RADV_XCLIPSE_PROF_H

#include <stdbool.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"

struct radv_cmd_buffer;
struct radv_device;

void radv_xprof_device_init(struct radv_device *device);
void radv_xprof_device_finish(struct radv_device *device);

/* Primary command buffers on the general queue only; everything else is ignored. */
void radv_xprof_begin_cmdbuf(struct radv_cmd_buffer *cmd_buffer);
void radv_xprof_end_cmdbuf(struct radv_cmd_buffer *cmd_buffer);

/* Pass boundaries: a render pass begins, a dispatch, and (from radv_xprof_draw) a new pixel shader. */
void radv_xprof_begin_rendering(struct radv_cmd_buffer *cmd_buffer, const VkRenderingInfo *info);
void radv_xprof_dispatch(struct radv_cmd_buffer *cmd_buffer, const uint32_t blocks[3]);
void radv_xprof_draw_slow(struct radv_cmd_buffer *cmd_buffer, uint32_t draw_count);

/* Clear census: every color/depth clear that reaches the fast-clear check, and why it was not a
 * fast clear. Returns reason == RADV_XCLR_FAST so a check can end in "return radv_xprof_clear(...)". */
enum radv_xprof_clear_reason {
   RADV_XCLR_FAST,
   RADV_XCLR_NO_SUPPORT, /* no view, or the view does not support fast clears (no DCC/HTILE, ...) */
   RADV_XCLR_LAYOUT,     /* the layout is not fast-clearable / not compressed */
   RADV_XCLR_RECT,       /* the clear does not cover the whole view */
   RADV_XCLR_VALUE,      /* the value cannot be packed, or the image has no clear value register */
   RADV_XCLR_DCC_PARAMS, /* the DCC fast clear codes cannot express the value */
   RADV_XCLR_MIPS,       /* a level without metadata */
   RADV_XCLR_RANGE,      /* depth outside [0, 1] */
   RADV_XCLR_TC_VALUE,   /* TC-compatible HTILE only fast-clears depth 0/1, stencil 0 */
   RADV_XCLR_FILL,       /* not fast, cleared by a memory fill instead of a draw (Xclipse) */
   RADV_XCLR_COUNT,
};
struct radv_image_view;
void radv_xprof_clear_slow(struct radv_cmd_buffer *cmd_buffer, const struct radv_image_view *iview,
                           bool depth, enum radv_xprof_clear_reason reason, const void *value,
                           unsigned value_size);
bool radv_xprof_sampling(void);
static inline bool
radv_xprof_clear(struct radv_cmd_buffer *cmd_buffer, const struct radv_image_view *iview, bool depth,
                 enum radv_xprof_clear_reason reason, const void *value, unsigned value_size)
{
   if (__builtin_expect(radv_xprof_sampling(), 0))
      radv_xprof_clear_slow(cmd_buffer, iview, depth, reason, value, value_size);
   return reason == RADV_XCLR_FAST;
}

#endif
