/*
 * Copyright 2026 JimVulkan
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_BC_EMU_H
#define PANVK_BC_EMU_H

#include <stdbool.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"

#include "util/list.h"
#include "util/simple_mtx.h"

/* BC formats on Bifrost texture units that lack them. An image of such a format gets a hidden
 * second plane in a format the hardware decodes, and a compute shader
 * (src/compiler/glsl/bc_decoder.glsl, shared with RADV) fills it after every copy into the image.
 * Plane 0 keeps the blocks the app wrote, so copies out of the image return them unchanged.
 *
 * Two kinds of part, measured (probe/mali/malibcn.c, malicarrier.c, astcmode.c):
 *
 *  BC1-BC3 in hardware (Mali-G52, MediaTek G76): BC4-BC7 only
 *    BC4_UNORM -> BC3 (the BC4 block is BC3's alpha half; the view swizzles R <- A)
 *    BC5_UNORM -> EAC_R11G11 (re-encoded per channel)
 *    BC7       -> BC3 (decoded and re-encoded, lossy)
 *
 *  no BC decoder at all (Exynos 9820's Mali-G76): every BC format
 *    BC1, BC2, BC3 -> ASTC 4x4 (repacked: BC1 exact to 2 LSB, BC2/BC3 alpha refit to 4 levels)
 *    BC7           -> ASTC 4x4 (decoded and re-encoded, lossy)
 *    BC4_UNORM     -> EAC_R11, BC5_UNORM -> EAC_R11G11
 *
 *  both: BC4/BC5_SNORM -> R8/R8G8_SNORM, BC6H -> R16G16B16A16_SFLOAT (decoded, exact)
 */

struct panvk_device;
struct panvk_physical_device;

enum panvk_bc_emu_profile {
   PANVK_BC_EMU_NONE,
   PANVK_BC_EMU_BC47, /* BC1-BC3 native */
   PANVK_BC_EMU_ALL,  /* no BC at all */
};

enum panvk_bc_emu_variant {
   PANVK_BC_EMU_U32X4, /* BC4 -> BC3, byte shuffle */
   PANVK_BC_EMU_EAC,   /* BC5 -> EAC_R11G11 */
   PANVK_BC_EMU_BC3,   /* BC7 -> BC3 */
   PANVK_BC_EMU_SNORM8,
   PANVK_BC_EMU_F16,
   PANVK_BC_EMU_ASTC,  /* BC1/BC2/BC3/BC7 -> ASTC 4x4 */
   PANVK_BC_EMU_EAC_R, /* BC4 -> EAC_R11 */
   PANVK_BC_EMU_VARIANT_COUNT,
};

struct panvk_bc_emu_state {
   /* Linear images of emulated formats that shaders sample. The host can write plane 0 through
    * a mapping at any time, so before every submission with command buffers the kbase path runs
    * the transcoder over all of them first (panvk_per_arch(bc_emu_refresh_begin)). A ring of
    * command buffers, so a submission does not wait for the one before it. */
   simple_mtx_t linear_lock;
   struct list_head linear_images;
   VkCommandPool refresh_pool;
   VkCommandBuffer refresh[4];
   unsigned refresh_next;

   simple_mtx_t lock;
   /* 4 KB of statistics the shader may write, the BC5 -> EAC tables (eac_lut.h), then the
    * shader's lookup tables (panvk_bc_tables.h). */
   VkBuffer buffer;
   VkDeviceMemory memory;
};

#define PANVK_BC_EMU_STATS_SIZE  4096u
#define PANVK_BC_EMU_LUT_SIZE    ((65536u + 131072u) * 4u)
/* The shader's lookup tables (panvk_bc_tables.h), after the EAC tables. */
#define PANVK_BC_EMU_TABLES_OFFSET (PANVK_BC_EMU_STATS_SIZE + PANVK_BC_EMU_LUT_SIZE)
#define PANVK_BC_EMU_TABLES_SIZE 4096u

/* Which formats this device emulates. PANVK_NO_BC_EMU=1 turns the emulation off. */
enum panvk_bc_emu_profile panvk_bc_emu_profile(const struct panvk_physical_device *pdev);

static inline bool
panvk_bc_emu_supported(const struct panvk_physical_device *pdev)
{
   return panvk_bc_emu_profile(pdev) != PANVK_BC_EMU_NONE;
}

/* The hardware format the texture unit samples for `format` on this device, or
 * VK_FORMAT_UNDEFINED when `format` is not emulated here. */
VkFormat panvk_bc_emu_carrier(const struct panvk_physical_device *pdev, VkFormat format);

static inline bool
panvk_bc_emu_format(const struct panvk_physical_device *pdev, VkFormat format)
{
   return panvk_bc_emu_carrier(pdev, format) != VK_FORMAT_UNDEFINED;
}

/* The view format the transcoder writes plane 1 through (one texel per block for compressed
 * carriers), and the one it reads plane 0 through. */
VkFormat panvk_bc_emu_store_format(const struct panvk_physical_device *pdev, VkFormat format);
VkFormat panvk_bc_emu_load_format(VkFormat format);

/* The shader's format code (BC_FMT_* in bc_decoder.glsl). */
int panvk_bc_emu_shader_format(VkFormat format);

/* The transcoder for `format`, over 2D arrays or (is_3d) 3D images. */
VkResult panvk_bc_emu_get_pipeline(struct panvk_device *dev, VkFormat format, bool is_3d,
                                   VkPipeline *pipeline_out, VkPipelineLayout *layout_out);
VkResult panvk_bc_emu_get_buffer(struct panvk_device *dev, VkBuffer *buffer_out);

void panvk_bc_emu_device_init(struct panvk_device *dev);
void panvk_bc_emu_device_finish(struct panvk_device *dev);

struct panvk_image;
struct panvk_device_memory;
struct panvk_cmd_buffer;

/* Linear sampled images of emulated formats: listed when bound, delisted (after the GPU is done
 * with their last refresh) when destroyed or when their memory is freed. */
void panvk_bc_emu_track_linear(struct panvk_device *dev, struct panvk_image *img);
void panvk_bc_emu_untrack_image(struct panvk_device *dev, struct panvk_image *img);
void panvk_bc_emu_untrack_memory(struct panvk_device *dev, const struct panvk_device_memory *mem);

#ifdef PAN_ARCH
#include "panvk_macros.h"

/* For the kbase submit path: with linear images listed, record the refresh into the next command
 * buffer of the ring and return it with linear_lock held; the caller submits it ahead of the
 * app's command buffers and calls bc_emu_refresh_end with the submission's sequence number
 * (0 if it failed). Without any, NULL and nothing held. */
struct panvk_cmd_buffer *panvk_per_arch(bc_emu_refresh_begin)(struct panvk_device *dev);
void panvk_per_arch(bc_emu_refresh_end)(struct panvk_device *dev,
                                        struct panvk_cmd_buffer *cmdbuf, uint64_t seq);
#endif

#endif
