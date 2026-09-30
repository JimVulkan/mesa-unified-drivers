/*
 * Copyright 2026 JimVulkan
 * SPDX-License-Identifier: MIT
 *
 * BC emulation on Bifrost: formats, pipelines and the shared lookup buffer. The per-arch
 * recording is panvk_per_arch(bc_emu_transcode) in panvk_vX_cmd_meta.c; see panvk_bc_emu.h.
 */

#include "panvk_bc_emu.h"

#include <string.h>

#include "util/u_debug.h"
#include "vk_format.h"
#include "vk_meta.h"

#include "kmod/kbase_kmod.h"
#include "panvk_device.h"
#include "panvk_image.h"
#include "panvk_physical_device.h"

#include "pan_props.h"

/* bc_decoder.glsl's BcTables block, generated from its constant arrays. */
#include "panvk_bc_tables.h"
static_assert(sizeof(panvk_bc_tables) <= PANVK_BC_EMU_TABLES_SIZE, "BC tables outgrew their slot");

static const uint32_t bc_emu_u32x4_spv[] = {
#include "bc_decoder_u32x4_spv.h"
};
static const uint32_t bc_emu_u32x4_3d_spv[] = {
#include "bc_decoder_u32x4_3d_spv.h"
};
static const uint32_t bc_emu_eac_spv[] = {
#include "bc_decoder_eac_spv.h"
};
static const uint32_t bc_emu_eac_3d_spv[] = {
#include "bc_decoder_eac_3d_spv.h"
};
static const uint32_t bc_emu_bc3_spv[] = {
#include "bc_decoder_bc3_spv.h"
};
static const uint32_t bc_emu_bc3_3d_spv[] = {
#include "bc_decoder_bc3_3d_spv.h"
};
static const uint32_t bc_emu_snorm8_spv[] = {
#include "bc_decoder_snorm8_spv.h"
};
static const uint32_t bc_emu_snorm8_3d_spv[] = {
#include "bc_decoder_snorm8_3d_spv.h"
};
static const uint32_t bc_emu_f16_spv[] = {
#include "bc_decoder_f16_spv.h"
};
static const uint32_t bc_emu_f16_3d_spv[] = {
#include "bc_decoder_f16_3d_spv.h"
};
static const uint32_t bc_emu_astc_spv[] = {
#include "bc_decoder_astc_spv.h"
};
static const uint32_t bc_emu_astc_3d_spv[] = {
#include "bc_decoder_astc_3d_spv.h"
};
static const uint32_t bc_emu_eac_r_spv[] = {
#include "bc_decoder_eac_r_spv.h"
};
static const uint32_t bc_emu_eac_r_3d_spv[] = {
#include "bc_decoder_eac_r_3d_spv.h"
};

/* The BC5 -> EAC_R11G11 tables, generated offline for RADV (eac_lut, eac_remap). */
#include "eac_lut.h"

/* Keeps the vk_meta cache keys apart from vk_meta's and panvk's own. */
#define PANVK_BC_EMU_META_KEY (VK_META_OBJECT_KEY_DRIVER_OFFSET + 0x4243u)

DEBUG_GET_ONCE_BOOL_OPTION(no_bc_emu, "PANVK_NO_BC_EMU", false)

/* TEXTURE_FEATURES_0 bits, the compressed format ids of the v6/v7 genxml: EAC R11 UNORM 2,
 * EAC RG11 UNORM 4, BC1..BC3 7..9, BC4_UNORM..BC7 10..16, ASTC 2D LDR 22. */
#define TF_BC13 (0x7u << 7)
#define TF_BC47 (0x7fu << 10)
#define TF_EAC  ((1u << 2) | (1u << 4))
#define TF_ASTC (1u << 22)

/* Does the texture unit decode any of BC1-BC3? Then the BC -> ASTC path is off, whatever else
 * the chip has: those formats stay native, and BC4/BC7 ride on the native BC3. */
static bool
bc_emu_native_bc13(const struct panvk_physical_device *pdev)
{
   return (pan_query_compressed_formats(&pdev->kmod.dev->props) & TF_BC13) != 0;
}

enum panvk_bc_emu_profile
panvk_bc_emu_profile(const struct panvk_physical_device *pdev)
{
   const unsigned arch = pan_arch(pdev->kmod.dev->props.gpu_id);
   if (arch < 6 || arch > 7 || debug_get_option_no_bc_emu())
      return PANVK_BC_EMU_NONE;

   const uint32_t tf = pan_query_compressed_formats(&pdev->kmod.dev->props);
   if ((tf & TF_EAC) != TF_EAC || (tf & TF_BC47) == TF_BC47)
      return PANVK_BC_EMU_NONE;

   /* The gate: with BC1-BC3 on the chip, never the ASTC path. All three native is the G52's
    * case; only some of them would leave BC1-BC3 half native, so no emulation at all then. */
   if (bc_emu_native_bc13(pdev))
      return (tf & TF_BC13) == TF_BC13 ? PANVK_BC_EMU_BC47 : PANVK_BC_EMU_NONE;

   /* No BC decoder at all (the Exynos 9820's G76): everything goes through ASTC and EAC. */
   return (tf & TF_ASTC) ? PANVK_BC_EMU_ALL : PANVK_BC_EMU_NONE;
}

VkFormat
panvk_bc_emu_carrier(const struct panvk_physical_device *pdev, VkFormat format)
{
   const enum panvk_bc_emu_profile profile = panvk_bc_emu_profile(pdev);
   if (profile == PANVK_BC_EMU_NONE)
      return VK_FORMAT_UNDEFINED;
   const bool all = profile == PANVK_BC_EMU_ALL;

   switch (format) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
   case VK_FORMAT_BC2_UNORM_BLOCK:
   case VK_FORMAT_BC3_UNORM_BLOCK:
      return all ? VK_FORMAT_ASTC_4x4_UNORM_BLOCK : VK_FORMAT_UNDEFINED;
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
   case VK_FORMAT_BC2_SRGB_BLOCK:
   case VK_FORMAT_BC3_SRGB_BLOCK:
      return all ? VK_FORMAT_ASTC_4x4_SRGB_BLOCK : VK_FORMAT_UNDEFINED;
   case VK_FORMAT_BC4_UNORM_BLOCK:
      return all ? VK_FORMAT_EAC_R11_UNORM_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
   case VK_FORMAT_BC4_SNORM_BLOCK:
      return VK_FORMAT_R8_SNORM;
   case VK_FORMAT_BC5_UNORM_BLOCK:
      return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
   case VK_FORMAT_BC5_SNORM_BLOCK:
      return VK_FORMAT_R8G8_SNORM;
   case VK_FORMAT_BC6H_UFLOAT_BLOCK:
   case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
   case VK_FORMAT_BC7_UNORM_BLOCK:
      return all ? VK_FORMAT_ASTC_4x4_UNORM_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
   case VK_FORMAT_BC7_SRGB_BLOCK:
      /* The decoded BC7 values are the sRGB-encoded ones, re-encoded as they are; the sRGB
       * carrier decodes them the same way. */
      return all ? VK_FORMAT_ASTC_4x4_SRGB_BLOCK : VK_FORMAT_BC3_SRGB_BLOCK;
   default:
      return VK_FORMAT_UNDEFINED;
   }
}

VkFormat
panvk_bc_emu_store_format(const struct panvk_physical_device *pdev, VkFormat format)
{
   const VkFormat carrier = panvk_bc_emu_carrier(pdev, format);
   if (!vk_format_is_compressed(carrier))
      return carrier;
   return vk_format_get_blocksize(carrier) == 8 ? VK_FORMAT_R32G32_UINT
                                                : VK_FORMAT_R32G32B32A32_UINT;
}

VkFormat
panvk_bc_emu_load_format(VkFormat format)
{
   return vk_format_get_blocksize(format) == 8 ? VK_FORMAT_R32G32_UINT
                                               : VK_FORMAT_R32G32B32A32_UINT;
}

int
panvk_bc_emu_shader_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC4_UNORM_BLOCK:
      return 0;
   case VK_FORMAT_BC4_SNORM_BLOCK:
      return 1;
   case VK_FORMAT_BC5_UNORM_BLOCK:
      return 2;
   case VK_FORMAT_BC5_SNORM_BLOCK:
      return 3;
   case VK_FORMAT_BC6H_UFLOAT_BLOCK:
      return 4;
   case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      return 5;
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
      return 7;
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      return 8;
   case VK_FORMAT_BC2_UNORM_BLOCK:
   case VK_FORMAT_BC2_SRGB_BLOCK:
      return 9;
   case VK_FORMAT_BC3_UNORM_BLOCK:
   case VK_FORMAT_BC3_SRGB_BLOCK:
      return 10;
   default:
      return 6;
   }
}

static enum panvk_bc_emu_variant
bc_emu_variant(const struct panvk_physical_device *pdev, VkFormat format)
{
   switch (panvk_bc_emu_carrier(pdev, format)) {
   case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
   case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
      return PANVK_BC_EMU_ASTC;
   case VK_FORMAT_EAC_R11_UNORM_BLOCK:
      return PANVK_BC_EMU_EAC_R;
   case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
      return PANVK_BC_EMU_EAC;
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8G8_SNORM:
      return PANVK_BC_EMU_SNORM8;
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return PANVK_BC_EMU_F16;
   default:
      /* BC3 planes come from BC4 (byte shuffle) or BC7 (decode and re-encode). */
      return format == VK_FORMAT_BC4_UNORM_BLOCK ? PANVK_BC_EMU_U32X4 : PANVK_BC_EMU_BC3;
   }
}

struct bc_emu_key {
   uint32_t type;
   uint32_t variant;
   uint32_t is_3d;
};

static VkResult
bc_emu_get_layout(struct panvk_device *dev, VkPipelineLayout *layout_out)
{
   const struct bc_emu_key key = {.type = PANVK_BC_EMU_META_KEY, .variant = ~0u, .is_3d = 0};

   /* Bindings as bc_decoder.glsl declares them. */
   const VkDescriptorSetLayoutBinding bindings[] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = ARRAY_SIZE(bindings),
      .pBindings = bindings,
   };
   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 32,
   };
   return vk_meta_get_pipeline_layout(&dev->vk, &dev->meta, &desc_info, &pc_range, &key,
                                      sizeof(key), layout_out);
}

VkResult
panvk_bc_emu_get_pipeline(struct panvk_device *dev, VkFormat format, bool is_3d,
                          VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   VkResult result = bc_emu_get_layout(dev, layout_out);
   if (result != VK_SUCCESS)
      return result;

   const struct panvk_physical_device *pdev = to_panvk_physical_device(dev->vk.physical);
   const enum panvk_bc_emu_variant variant = bc_emu_variant(pdev, format);

   /* The ASTC transcoder is for chips without BC1-BC3 only; never built where they are native. */
   if (variant == PANVK_BC_EMU_ASTC && bc_emu_native_bc13(pdev))
      return VK_ERROR_FEATURE_NOT_PRESENT;

   const struct bc_emu_key key = {
      .type = PANVK_BC_EMU_META_KEY, .variant = variant, .is_3d = is_3d};
   VkPipeline cached = vk_meta_lookup_pipeline(&dev->meta, &key, sizeof(key));
   if (cached != VK_NULL_HANDLE) {
      *pipeline_out = cached;
      return VK_SUCCESS;
   }

#define SPV(v, name)                                                                                  [v] = {{bc_emu_##name##_spv, sizeof(bc_emu_##name##_spv)},                                                {bc_emu_##name##_3d_spv, sizeof(bc_emu_##name##_3d_spv)}}
   static const struct {
      const uint32_t *code;
      size_t size;
   } spv[PANVK_BC_EMU_VARIANT_COUNT][2] = {
      SPV(PANVK_BC_EMU_U32X4, u32x4), SPV(PANVK_BC_EMU_EAC, eac),     SPV(PANVK_BC_EMU_BC3, bc3),
      SPV(PANVK_BC_EMU_SNORM8, snorm8), SPV(PANVK_BC_EMU_F16, f16),   SPV(PANVK_BC_EMU_ASTC, astc),
      SPV(PANVK_BC_EMU_EAC_R, eac_r),
   };
#undef SPV
   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv[variant][is_3d].size,
      .pCode = spv[variant][is_3d].code,
   };
   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &module_info,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
         },
      .layout = *layout_out,
   };
   return vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &pipeline_info, &key, sizeof(key),
                                          pipeline_out);
}

VkResult
panvk_bc_emu_get_buffer(struct panvk_device *dev, VkBuffer *buffer_out)
{
   struct panvk_bc_emu_state *st = &dev->bc_emu;
   VkResult result = VK_SUCCESS;

   simple_mtx_lock(&st->lock);
   if (st->buffer != VK_NULL_HANDLE)
      goto out;

   const struct vk_device_dispatch_table *t = &dev->vk.dispatch_table;
   const struct panvk_physical_device *pdev = to_panvk_physical_device(dev->vk.physical);
   VkDevice h = panvk_device_to_handle(dev);
   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = PANVK_BC_EMU_TABLES_OFFSET + PANVK_BC_EMU_TABLES_SIZE,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer buffer;
   result = t->CreateBuffer(h, &bci, NULL, &buffer);
   if (result != VK_SUCCESS)
      goto out;

   VkMemoryRequirements mr;
   t->GetBufferMemoryRequirements(h, buffer, &mr);
   uint32_t type = 0;
   const VkMemoryPropertyFlags host =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < pdev->memory.type_count; i++) {
      if ((mr.memoryTypeBits & BITFIELD_BIT(i)) &&
          (pdev->memory.types[i].propertyFlags & host) == host) {
         type = i;
         break;
      }
   }
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = type,
   };
   VkDeviceMemory memory;
   result = t->AllocateMemory(h, &mai, NULL, &memory);
   if (result != VK_SUCCESS) {
      t->DestroyBuffer(h, buffer, NULL);
      goto out;
   }
   t->BindBufferMemory(h, buffer, memory, 0);

   uint8_t *map;
   result = t->MapMemory(h, memory, 0, VK_WHOLE_SIZE, 0, (void **)&map);
   if (result != VK_SUCCESS) {
      t->DestroyBuffer(h, buffer, NULL);
      t->FreeMemory(h, memory, NULL);
      goto out;
   }
   memset(map, 0, PANVK_BC_EMU_STATS_SIZE);
   memcpy(map + PANVK_BC_EMU_STATS_SIZE, eac_lut, sizeof(eac_lut));
   memcpy(map + PANVK_BC_EMU_STATS_SIZE + sizeof(eac_lut), eac_remap, sizeof(eac_remap));
   memcpy(map + PANVK_BC_EMU_TABLES_OFFSET, panvk_bc_tables, sizeof(panvk_bc_tables));
   t->UnmapMemory(h, memory);

   st->memory = memory;
   st->buffer = buffer;

out:
   *buffer_out = st->buffer;
   simple_mtx_unlock(&st->lock);
   return result;
}

void
panvk_bc_emu_device_init(struct panvk_device *dev)
{
   simple_mtx_init(&dev->bc_emu.linear_lock, mtx_plain);
   list_inithead(&dev->bc_emu.linear_images);
   dev->bc_emu.refresh_pool = VK_NULL_HANDLE;
   memset(dev->bc_emu.refresh, 0, sizeof(dev->bc_emu.refresh));
   dev->bc_emu.refresh_next = 0;
   simple_mtx_init(&dev->bc_emu.lock, mtx_plain);
   dev->bc_emu.buffer = VK_NULL_HANDLE;
   dev->bc_emu.memory = VK_NULL_HANDLE;
}

void
panvk_bc_emu_device_finish(struct panvk_device *dev)
{
   const struct vk_device_dispatch_table *t = &dev->vk.dispatch_table;
   VkDevice h = panvk_device_to_handle(dev);

   if (dev->bc_emu.buffer != VK_NULL_HANDLE)
      t->DestroyBuffer(h, dev->bc_emu.buffer, NULL);
   if (dev->bc_emu.memory != VK_NULL_HANDLE)
      t->FreeMemory(h, dev->bc_emu.memory, NULL);
   dev->bc_emu.buffer = VK_NULL_HANDLE;
   dev->bc_emu.memory = VK_NULL_HANDLE;
   simple_mtx_destroy(&dev->bc_emu.lock);

   /* The pool frees its command buffers. The queue is idle by now. */
   if (dev->bc_emu.refresh_pool != VK_NULL_HANDLE)
      t->DestroyCommandPool(h, dev->bc_emu.refresh_pool, NULL);
   dev->bc_emu.refresh_pool = VK_NULL_HANDLE;
   simple_mtx_destroy(&dev->bc_emu.linear_lock);
}

void
panvk_bc_emu_track_linear(struct panvk_device *dev, struct panvk_image *img)
{
   simple_mtx_lock(&dev->bc_emu.linear_lock);
   if (!img->bc_emu.listed) {
      list_addtail(&img->bc_emu.link, &dev->bc_emu.linear_images);
      img->bc_emu.listed = true;
   }
   simple_mtx_unlock(&dev->bc_emu.linear_lock);
}

/* Under linear_lock. The refresh that last touched the image may still be on the GPU. */
static void
bc_emu_delist(struct panvk_device *dev, struct panvk_image *img)
{
   list_del(&img->bc_emu.link);
   img->bc_emu.listed = false;
   if (img->bc_emu.seq && pan_kmod_dev_is_kbase(dev->kmod.dev))
      pan_kmod_kbase_wait_seq(dev->kmod.dev, img->bc_emu.seq);
   img->bc_emu.seq = 0;
}

void
panvk_bc_emu_untrack_image(struct panvk_device *dev, struct panvk_image *img)
{
   if (!img->bc_emu.linear_sampled)
      return;
   simple_mtx_lock(&dev->bc_emu.linear_lock);
   if (img->bc_emu.listed)
      bc_emu_delist(dev, img);
   simple_mtx_unlock(&dev->bc_emu.linear_lock);
}

void
panvk_bc_emu_untrack_memory(struct panvk_device *dev, const struct panvk_device_memory *mem)
{
   simple_mtx_lock(&dev->bc_emu.linear_lock);
   list_for_each_entry_safe(struct panvk_image, img, &dev->bc_emu.linear_images, bc_emu.link) {
      if (img->planes[0].mem == mem)
         bc_emu_delist(dev, img);
   }
   simple_mtx_unlock(&dev->bc_emu.linear_lock);
}
