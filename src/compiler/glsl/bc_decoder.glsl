/*
 * Copyright 2026 JimVulkan
 * SPDX-License-Identifier: MIT
 *
 * GPU decoder/transcoder for BC4, BC5, BC6H and BC7, run once at upload.
 * One invocation handles one 4x4 block; the dispatch in radv_meta_bc_decode.c
 * is sized in blocks to match.
 *
 * The BC6H/BC7 and RGTC decode follows Mesa's CPU decoders, and its tables are generated from
 * them; do not hand-edit:
 *   src/util/format/texcompress_bptc_tmp.h  Copyright (C) 2014 Intel Corporation (MIT)
 *   src/util/format/u_format_rgtc.c         Copyright (C) 2011 Red Hat Inc. (MIT)
 */
#version 460
#extension GL_EXT_samplerless_texture_functions : require

/* BC_OUT_EACBUF: the BC5 -> EAC_R11G11 transcode reading the blocks straight from the copy's source
 * buffer (pc.src_va) instead of plane 0, so an upload is one pass. Used when the EAC plane shares
 * plane 0's memory (radv_image.xclipse_bc_alias). */
#if defined(BC_OUT_EACBUF)
#define BC_OUT_EAC
#define BC_SRC_BUFFER
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer BcSrcBlocks { uvec4 b[]; };
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

/* Compressed plane, one uint texel per 4x4 block: RG32_UINT for 8-byte blocks, RGBA32_UINT for
 * 16-byte blocks. */
/* BC_3D: the same for 3D images, whose depth slices take the place of the array layers. */
#if defined(BC_3D)
#define BC_TEXTURE utexture3D
#define BC_UIMAGE  uimage3D
#define BC_IMAGE   image3D
#else
#define BC_TEXTURE utexture2DArray
#define BC_UIMAGE  uimage2DArray
#define BC_IMAGE   image2DArray
#endif

layout(set = 0, binding = 0) uniform BC_TEXTURE s_blocks;

#if defined(BC_OUT_U32X4) || defined(BC_OUT_EAC) || defined(BC_OUT_BC3) || defined(BC_OUT_ASTC)
/* Transcode: the destination is a compressed image, one rgba32ui texel per 16-byte block.
 *   BC_OUT_U32X4 -- BC4 -> BC3 byte shuffle
 *   BC_OUT_EAC   -- BC5 -> EAC_R11G11 re-encode
 *   BC_OUT_BC3   -- BC7 -> BC3 decode and re-encode
 *   BC_OUT_ASTC  -- BC1/BC2/BC3 -> ASTC 4x4 repack, BC7 -> ASTC 4x4 decode and re-encode */
layout(set = 0, binding = 1, rgba32ui) uniform writeonly BC_UIMAGE o_img;
#elif defined(BC_OUT_EAC_R)
/* BC4 -> EAC_R11: an 8-byte destination block, one rg32ui texel. */
layout(set = 0, binding = 1, rg32ui) uniform writeonly BC_UIMAGE o_img;
#elif defined(BC_OUT_SNORM8)
layout(set = 0, binding = 1, rgba8_snorm) uniform writeonly BC_IMAGE o_img;
#elif defined(BC_OUT_F16)
layout(set = 0, binding = 1, rgba16f) uniform writeonly BC_IMAGE o_img;
#else
layout(set = 0, binding = 1, rgba8) uniform writeonly BC_IMAGE o_img;
#endif

/* Endpoint-spread histogram: hardware BC4 is exact when spread <= 32. Samples 1 in 64 blocks. */
layout(set = 0, binding = 2, std430) buffer Hist { uint bucket[]; } hist;
/* The histogram is the first 1024 dwords; the [BCCONTENT] ring follows in the same buffer. */
#define BC_HASH_BASE 1024u

#if defined(BC_OUT_EAC) || defined(BC_OUT_EAC_R)
/* Per BC4 endpoint pair, the best EAC_R11 config: base<<8 | mult<<4 | table. Precomputed offline
 * (eac_lut.h), exact for spread <= 32. Two 16-bit entries per uint. */
layout(set = 0, binding = 3, std430) readonly buffer EacLut { uint pair[]; } eaclut;
#endif

#if defined(BC_TABLES_BUFFER)
/* PanVK: the lookup tables below come from this buffer instead of constant arrays. Bifrost
 * has no constant-data path, so every constant array indexed with a variable was copied to
 * the stack of every thread, 10 KB per thread for BC7 -> ASTC. panvk_bc_tables_gen.py
 * builds the buffer from the constant definitions, in this member order. */
layout(set = 0, binding = 4, std430) readonly buffer BcTables {
   uint bc_part2[64];
   uint bc_part3[64];
   int bc_anchor2[64];
   int bc_anchor3b[64];
   int bc_anchor3c[64];
   int bc6_flags[18];
   int bc6_npb[18];
   int bc6_neb[18];
   int bc6_nib[18];
   int bc6_d0[18];
   int bc6_d1[18];
   int bc6_d2[18];
   int bc6_foff[19];
   uint bc6_fields[236];
   int bc_weights[28];
   int bc7_nsub[8];
   int bc7_npb[8];
   int bc7_rot[8];
   int bc7_isel[8];
   int bc7_cbits[8];
   int bc7_abits[8];
   int bc7_epb[8];
   int bc7_spb[8];
   int bc7_ib[8];
   int bc7_ib2[8];
   uint astc_trit_enc[61];
   uint astc_q48_unq[12];
   uint astc_q48_near[64];
   uint astc_q128_near[64];
};
#endif

layout(push_constant) uniform push_constants {
   ivec3 offset;
   int format;
   int image_type;
   /* Two scalars, not ivec2: ivec2 alignment would pad to offset 24 and break the packed layout. */
   int extent_x;
   int extent_y;
   /* Ring slot for the [BCCONTENT] census; -1 disables. */
   int hash_slot;
   /* BC_SRC_BUFFER only: the source blocks, row and slice pitch in blocks. */
   uvec2 src_va;
   uint src_row_blocks;
   uint src_slice_blocks;
} pc;

/* Keep in sync with enum vk_texcompress_bc_format. */
#define BC_FMT_BC4_UNORM  0
#define BC_FMT_BC4_SNORM  1
#define BC_FMT_BC5_UNORM  2
#define BC_FMT_BC5_SNORM  3
#define BC_FMT_BC6H_UF16  4
#define BC_FMT_BC6H_SF16  5
#define BC_FMT_BC7        6
/* Only the ASTC variant sees these: BC1-BC3 are transcoded only where nothing decodes them. */
#define BC_FMT_BC1_RGB    7
#define BC_FMT_BC1_RGBA   8
#define BC_FMT_BC2        9
#define BC_FMT_BC3        10

#if !defined(BC_TABLES_BUFFER)
const uint bc_part2[64] = uint[](0x50505050u, 0x40404040u, 0x54545454u, 0x54505040u, 0x50404000u, 0x55545450u, 0x55545040u, 0x54504000u, 0x50400000u, 0x55555450u, 0x55544000u, 0x54400000u, 0x55555440u, 0x55550000u, 0x55555500u, 0x55000000u, 0x55150100u, 0x00004054u, 0x15010000u, 0x00405054u, 0x00004050u, 0x15050100u, 0x05010000u, 0x40505054u, 0x00404050u, 0x05010100u, 0x14141414u, 0x05141450u, 0x01155440u, 0x00555500u, 0x15014054u, 0x05414150u, 0x44444444u, 0x55005500u, 0x11441144u, 0x05055050u, 0x05500550u, 0x11114444u, 0x41144114u, 0x44111144u, 0x15055054u, 0x01055040u, 0x05041050u, 0x05455150u, 0x14414114u, 0x50050550u, 0x41411414u, 0x00141400u, 0x00041504u, 0x00105410u, 0x10541000u, 0x04150400u, 0x50410514u, 0x41051450u, 0x05415014u, 0x14054150u, 0x41050514u, 0x41505014u, 0x40011554u, 0x54150140u, 0x50505500u, 0x00555050u, 0x15151010u, 0x54540404u);
const uint bc_part3[64] = uint[](0xaa685050u, 0x6a5a5040u, 0x5a5a4200u, 0x5450a0a8u, 0xa5a50000u, 0xa0a05050u, 0x5555a0a0u, 0x5a5a5050u, 0xaa550000u, 0xaa555500u, 0xaaaa5500u, 0x90909090u, 0x94949494u, 0xa4a4a4a4u, 0xa9a59450u, 0x2a0a4250u, 0xa5945040u, 0x0a425054u, 0xa5a5a500u, 0x55a0a0a0u, 0xa8a85454u, 0x6a6a4040u, 0xa4a45000u, 0x1a1a0500u, 0x0050a4a4u, 0xaaa59090u, 0x14696914u, 0x69691400u, 0xa08585a0u, 0xaa821414u, 0x50a4a450u, 0x6a5a0200u, 0xa9a58000u, 0x5090a0a8u, 0xa8a09050u, 0x24242424u, 0x00aa5500u, 0x24924924u, 0x24499224u, 0x50a50a50u, 0x500aa550u, 0xaaaa4444u, 0x66660000u, 0xa5a0a5a0u, 0x50a050a0u, 0x69286928u, 0x44aaaa44u, 0x66666600u, 0xaa444444u, 0x54a854a8u, 0x95809580u, 0x96969600u, 0xa85454a8u, 0x80959580u, 0xaa141414u, 0x96960000u, 0xaaaa1414u, 0xa05050a0u, 0xa0a5a5a0u, 0x96000000u, 0x40804080u, 0xa9a8a9a8u, 0xaaaaaa44u, 0x2a4a5254u);
const int bc_anchor2[64] = int[](15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15);
const int bc_anchor3b[64] = int[](3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3);
const int bc_anchor3c[64] = int[](15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8);
#endif

/* BC6H mode tables, generated from Mesa texcompress_bptc_tmp.h. Do not hand-edit. */
#if !defined(BC_TABLES_BUFFER)
const int bc6_flags[18] = int[](2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 0, 1);
const int bc6_npb[18] = int[](5, 5, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0);
const int bc6_neb[18] = int[](10, 7, 11, 10, 11, 11, 11, 12, 9, 16, 8, 0, 8, 0, 8, 0, 6, 0);
const int bc6_nib[18] = int[](3, 3, 3, 4, 3, 4, 3, 4, 3, 4, 3, 0, 3, 0, 3, 0, 3, 0);
const int bc6_d0[18] = int[](5, 6, 5, 10, 4, 9, 4, 8, 5, 4, 6, 0, 5, 0, 5, 0, 6, 0);
const int bc6_d1[18] = int[](5, 6, 4, 10, 5, 9, 4, 8, 5, 4, 5, 0, 6, 0, 5, 0, 6, 0);
const int bc6_d2[18] = int[](5, 6, 4, 10, 4, 9, 5, 8, 5, 4, 5, 0, 5, 0, 6, 0, 6, 0);
const int bc6_foff[19] = int[](0, 19, 42, 60, 66, 86, 95, 115, 124, 143, 152, 171, 171, 192, 192, 213, 213, 236, 236);
const uint bc6_fields[236] = uint[](326u, 330u, 331u, 2560u, 2564u, 2568u, 1281u, 327u, 1030u, 1285u, 267u, 1031u, 1289u, 283u, 1034u, 1282u, 299u, 1283u, 315u, 342u, 327u, 343u, 1792u, 267u, 283u, 330u, 1796u, 346u, 299u, 326u, 1800u, 315u, 347u, 331u, 1537u, 1030u, 1541u, 1031u, 1545u, 1034u, 1538u, 1539u, 2560u, 2564u, 2568u, 1281u, 416u, 1030u, 1029u, 420u, 267u, 1031u, 1033u, 424u, 283u, 1034u, 1282u, 299u, 1283u, 315u, 2560u, 2564u, 2568u, 2561u, 2565u, 2569u, 2560u, 2564u, 2568u, 1025u, 416u, 327u, 1030u, 1285u, 420u, 1031u, 1033u, 424u, 283u, 1034u, 1026u, 267u, 299u, 1027u, 326u, 315u, 2560u, 2564u, 2568u, 2305u, 416u, 2309u, 420u, 2313u, 424u, 2560u, 2564u, 2568u, 1025u, 416u, 330u, 1030u, 1029u, 420u, 267u, 1031u, 1289u, 424u, 1034u, 1026u, 283u, 299u, 1027u, 331u, 315u, 2560u, 2564u, 2568u, 2049u, 8864u, 2053u, 8868u, 2057u, 8872u, 2304u, 330u, 2308u, 326u, 2312u, 331u, 1281u, 327u, 1030u, 1285u, 267u, 1031u, 1289u, 283u, 1034u, 1282u, 299u, 1283u, 315u, 2560u, 2564u, 2568u, 1025u, 9888u, 1029u, 9892u, 1033u, 9896u, 2048u, 327u, 330u, 2052u, 299u, 326u, 2056u, 315u, 331u, 1537u, 1030u, 1285u, 267u, 1031u, 1289u, 283u, 1034u, 1538u, 1539u, 2048u, 267u, 330u, 2052u, 342u, 326u, 2056u, 343u, 331u, 1281u, 327u, 1030u, 1541u, 1031u, 1289u, 283u, 1034u, 1282u, 299u, 1283u, 315u, 2048u, 283u, 330u, 2052u, 346u, 326u, 2056u, 347u, 331u, 1281u, 327u, 1030u, 1285u, 267u, 1031u, 1545u, 1034u, 1282u, 299u, 1283u, 315u, 1536u, 327u, 267u, 283u, 330u, 1540u, 342u, 346u, 299u, 326u, 1544u, 343u, 315u, 347u, 331u, 1537u, 1030u, 1541u, 1031u, 1545u, 1034u, 1538u, 1539u);
#endif

/* ---------------------------------------------------------------- bit access */

/* Extract n bits at offset `off` from a 128-bit block in constant time (a field spans <= 2 dwords).
 * Guards: n <= 0 (bc7_npb[] is 0 for modes 4-6, 1u << 32 is undefined), s != 0 (shift by 32),
 * w < 3 (blk[4] out of bounds). */
uint bc_bits(uvec4 blk, int off, int n)
{
   if (n <= 0)
      return 0u;

   int w = off >> 5;
   int s = off & 31;

   uint v = blk[w] >> uint(s);
   if (s != 0 && w < 3)
      v |= blk[w + 1] << uint(32 - s);

   return n >= 32 ? v : (v & ((1u << uint(n)) - 1u));
}

/* Buckets 0..5 = spread <=8, 9-16, 17-32, 33-64, 65-128, >128; 6 = total sampled;
 * 7 = r0 > r1 (8-value) mode; 8 = 8-value mode and spread <= 32. */
void bc_record_spread(uvec4 blk, int base_bit, int r0, int r1)
{
   int s = r0 > r1 ? r0 - r1 : r1 - r0;
   int b = s <= 8 ? 0 : s <= 16 ? 1 : s <= 32 ? 2 : s <= 64 ? 3 : s <= 128 ? 4 : 5;
   atomicAdd(hist.bucket[b], 1u);
   atomicAdd(hist.bucket[6], 1u);
   if (r0 > r1)
      atomicAdd(hist.bucket[7], 1u);

   /* Representable = every texel used lies on the endpoint line. The 6-value mode is only off the
    * line when indices 6 or 7 (0.0 / 1.0) are used. */
   bool offline = false;
   if (r0 <= r1) {
      for (int t = 0; t < 16; t++) {
         uint idx = bc_bits(blk, base_bit + 16 + 3 * t, 3);
         if (idx >= 6u) { offline = true; break; }
      }
   }
   if (!offline && s <= 32)
      atomicAdd(hist.bucket[8], 1u);
   if (offline)
      atomicAdd(hist.bucket[9], 1u);
}

/* BC5 census: do R and G span the same range? ASTC 4x4 dual-plane with CEM 0 could carry BC5, but
 * both planes share one [L0, L1]. Reports the distribution, not a mean. */
void bc_record_rg(int r0, int r1, int g0, int g1)
{
   /* Compare spans, not raw endpoints: BC4 stores them in either order. */
   int rlo = min(r0, r1), rhi = max(r0, r1);
   int glo = min(g0, g1), ghi = max(g0, g1);
   int dlo = abs(rlo - glo), dhi = abs(rhi - ghi);
   int d = max(dlo, dhi);

   int b = d == 0 ? 0 : d <= 2 ? 1 : d <= 4 ? 2 : d <= 8 ? 3 : d <= 16 ? 4 : d <= 32 ? 5 : 6;
   atomicAdd(hist.bucket[10 + b], 1u);
   atomicAdd(hist.bucket[17], 1u);
}


#if defined(BC_OUT_EAC) || defined(BC_OUT_EAC_R)
/* ---- BC5 -> EAC_R11G11 encoder ----
 * Both are 16 bytes, two channels, 3-bit per-texel indices, so only the palette is re-encoded.
 * Tables, value formula, field positions and texel order verified on hardware. */
const int bc_eac_mod[128] = int[128](
   -3, -6,  -9, -15, 2, 5, 8, 14,   -3, -7, -10, -13, 2, 6, 9, 12,
   -2, -5,  -8, -13, 1, 4, 7, 12,   -2, -4,  -6, -13, 1, 3, 5, 12,
   -3, -6,  -8, -12, 2, 5, 7, 11,   -3, -7,  -9, -11, 2, 6, 8, 10,
   -4, -7,  -8, -11, 3, 6, 7, 10,   -3, -5,  -8, -11, 2, 4, 7, 10,
   -2, -6,  -8, -10, 1, 5, 7,  9,   -2, -5,  -8, -10, 1, 4, 7,  9,
   -2, -4,  -8, -10, 1, 3, 7,  9,   -2, -5,  -7, -10, 1, 4, 6,  9,
   -3, -4,  -7, -10, 2, 3, 6,  9,   -1, -2,  -3, -10, 0, 1, 2,  9,
   -4, -6,  -8,  -9, 3, 5, 7,  8,   -3, -5,  -7,  -9, 2, 4, 6,  8);

/* One BC4 palette entry, per the format definition. */
int bc_eac_bc4val(int i, int e0, int e1)
{
   if (i == 0) return e0;
   if (i == 1) return e1;
   if (e0 > e1) return ((8 - i) * e0 + (i - 1) * e1) / 7;
   if (i == 6) return 0;
   if (i == 7) return 255;
   return ((6 - i) * e0 + (i - 1) * e1) / 5;
}

/* Place a 3-bit index at bit p of the 64-bit EAC word (hiW = bits 63..32, loW = 31..0).
 * p = 30 and 31 straddle both halves. */
void bc_eac_put3(inout uint hiW, inout uint loW, int p, uint v)
{
   if (p >= 32)      hiW |= v << uint(p - 32);
   else if (p <= 29) loW |= v << uint(p);
   else            { loW |= v << uint(p); hiW |= v >> uint(32 - p); }
}

/* Encode one 8-byte BC4 channel block into one 8-byte EAC_R11 block.
 * `half01` is the channel's two source uints; returns the EAC block in MEMORY order. */
uvec2 bc_eac_encode(uvec2 half01)
{
   const int e0 = int(half01.x & 0xffu);
   const int e1 = int((half01.x >> 8) & 0xffu);

   /* Two configs per endpoint pair. In the 6-value mode indices 6/7 (0.0, 1.0) are off the line;
    * if the block uses them, take the config fitted to all eight entries. */
   bool uses_offline = false;
   if (e0 <= e1) {
      for (int t = 0; t < 16; t++)
         if (bc_bits(uvec4(half01, 0u, 0u), 16 + 3 * t, 3) >= 6u) { uses_offline = true; break; }
   }
   const uint li = uint(e0 * 256 + e1);
   const uint word = eaclut.pair[li];
   const uint cfg = uses_offline ? (word >> 16u) : (word & 0xffffu);
   const int base_cw = int(cfg >> 8u);
   const int mult    = int((cfg >> 4u) & 0xfu);
   const int tab     = int(cfg & 0xfu);

   /* Nearest EAC index for each of the 8 BC4 entries, looked up by (e0, e1, uses_offline).
    * Packed 3 bits per entry to avoid a dynamically indexed local array. */
   const uint rw = eaclut.pair[65536u + li * 2u + (uses_offline ? 1u : 0u)];

   /* Texel order: EAC slot j is texel (j/4, j%4), column-major; BC4 texel t is (t%4, t/4).
    * So texel t goes to slot (t%4)*4 + t/4. */
   uint hiW = (uint(base_cw) << 24) | (uint(mult) << 20) | (uint(tab) << 16);
   uint loW = 0u;
   for (int t = 0; t < 16; t++) {
      const uint bi = bc_bits(uvec4(half01, 0u, 0u), 16 + 3 * t, 3);
      const int j = (t % 4) * 4 + (t / 4);
      bc_eac_put3(hiW, loW, 45 - 3 * j, (rw >> (3u * bi)) & 7u);
   }

   /* The 64-bit word is big-endian and the image is read as little-endian uints: byte-reverse. */
   return uvec2(((hiW & 0xffu) << 24) | ((hiW & 0xff00u) << 8) | ((hiW >> 8) & 0xff00u) | (hiW >> 24),
                ((loW & 0xffu) << 24) | ((loW & 0xff00u) << 8) | ((loW >> 8) & 0xff00u) | (loW >> 24));
}
#endif

int bc_sign_extend(int v, int bits)
{
   int m = 1 << (bits - 1);
   return (v ^ m) - m;
}

/* The three weight tables in one array (2-bit at 0, 3-bit at 4, 4-bit at 12): one dynamic index
 * instead of three. */
#if !defined(BC_TABLES_BUFFER)
const int bc_weights[28] = int[](
   0, 21, 43, 64,                                                          /* 2-bit */
   0, 9, 18, 27, 37, 46, 55, 64,                                           /* 3-bit */
   0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64);           /* 4-bit */
#endif

int bc_interpolate(int a, int b, int index, int index_bits)
{
   int w = bc_weights[(index_bits == 2 ? 0 : (index_bits == 3 ? 4 : 12)) + index];
   return (a * (64 - w) + b * w + 32) >> 6;
}

#if defined(BC_OUT_UNORM8) || defined(BC_OUT_BC3) || defined(BC_OUT_ASTC)
/* The decoded BC7 block, RGBA8 per texel, four texels per vector. Mali keeps any array of more
 * than 8 entries on the stack of every thread (768 threads per core on the G76), so no 16-texel
 * array: bc7_decode_block() fills this, the encoders read it back. */
uvec4 bc_px[4];

ivec4 bc_px_get(int t)
{
   const uint p = bc_px[t >> 2][t & 3];
   return ivec4(int(p & 0xffu), int((p >> 8) & 0xffu), int((p >> 16) & 0xffu), int(p >> 24));
}
#endif

#if defined(BC_OUT_BC3)
/* ---- BC7 -> BC3 (DXT5) encoder ----
 * No table search: both halves are a line between the block's extremes and indices are a
 * projection, O(1) per texel. Costs 2-bit colour indices and RGB565 endpoints (gradient banding).
 * BC1 mode depends on endpoint order: c0 > c1 is 4-colour opaque, c0 <= c1 makes index 3
 * transparent. Endpoint order is handled explicitly below. */

/* Level along the line (0 = c1, 3 = c0) -> BC1 index. BC1 palette order is c0, c1, interpolants. */
const int bc_dxt_cidx[4] = int[](1, 3, 2, 0);

void bc_dxt_put(inout uint hiW, inout uint loW, int off, int n, uint v)
{
   for (int i = 0; i < n; i++)
      if ((v & (1u << uint(i))) != 0u) {
         int b = off + i;
         if (b >= 32) hiW |= 1u << uint(b - 32);
         else         loW |= 1u << uint(b);
      }
}

uvec4 bc_bc3_encode()
{
   /* ---- alpha half: a BC4 block. Endpoints are the extremes; 8 levels, 3-bit indices. ---- */
   int amin = 255, amax = 0;
   for (int t = 0; t < 16; t++) {
      amin = min(amin, bc_px_get(t).a);
      amax = max(amax, bc_px_get(t).a);
   }
   uint ahi = 0u, alo = 0u;
   bc_dxt_put(ahi, alo, 0, 8, uint(amax));   /* alpha0 > alpha1 selects the 8-value mode */
   bc_dxt_put(ahi, alo, 8, 8, uint(amin));
   if (amax > amin) {
      const int rng = amax - amin;
      for (int t = 0; t < 16; t++) {
         /* Index 0 = a0, 1 = a1, 2..7 interpolate down from a0, so level q maps to 8-q. */
         const int q = clamp(((bc_px_get(t).a - amin) * 7 + rng / 2) / rng, 0, 7);
         const int idx = (q == 7) ? 0 : ((q == 0) ? 1 : (8 - q));
         bc_dxt_put(ahi, alo, 16 + 3 * t, 3, uint(idx));
      }
   }
   /* amax == amin leaves every index 0, which decodes to alpha0 in either mode. */

   /* ---- colour half: a BC1 block. ---- */
   ivec3 lo = ivec3(255), hi = ivec3(0);
   for (int t = 0; t < 16; t++) {
      lo = min(lo, bc_px_get(t).rgb);
      hi = max(hi, bc_px_get(t).rgb);
   }
   /* Inset the bounding box by 1/16 of its extent (standard fast-DXT refinement). */
   const ivec3 ins = (hi - lo) / 16;
   vec3 f0 = vec3(hi - ins);
   vec3 f1 = vec3(lo + ins);

   /* Refine the box diagonal, which misses two-cluster blocks (BC7 mode 7): assign indices, solve
    * least-squares endpoints per channel with weights u in {0, 1/3, 2/3, 1}, then re-assign:
    *      [ S(1-u)^2   S(1-u)u ] [E1]   [ S(1-u)c ]
    *      [ S(1-u)u    S u^2   ] [E0] = [ S u c   ]
    * Two iterations (85.9% -> 90.2% -> 90.5% of texels within 10 LSB). */
   for (int pass = 0; pass < 2; pass++) {
      const ivec3 pq0 = ivec3(int(f0.r) >> 3, int(f0.g) >> 2, int(f0.b) >> 3);
      const ivec3 pq1 = ivec3(int(f1.r) >> 3, int(f1.g) >> 2, int(f1.b) >> 3);
      const ivec3 pd0 = ivec3((pq0.r << 3) | (pq0.r >> 2), (pq0.g << 2) | (pq0.g >> 4), (pq0.b << 3) | (pq0.b >> 2));
      const ivec3 pd1 = ivec3((pq1.r << 3) | (pq1.r >> 2), (pq1.g << 2) | (pq1.g >> 4), (pq1.b << 3) | (pq1.b >> 2));
      const ivec3 pdir = pd0 - pd1;
      const int pden = pdir.r * pdir.r + pdir.g * pdir.g + pdir.b * pdir.b;
      if (pden <= 0)
         break;

      float A = 0.0, B = 0.0, C = 0.0;
      vec3 P = vec3(0.0), Q = vec3(0.0);
      for (int t = 0; t < 16; t++) {
         const ivec3 wt = bc_px_get(t).rgb;
         const ivec3 rel = wt - pd1;
         const int num = clamp(rel.r * pdir.r + rel.g * pdir.g + rel.b * pdir.b, 0, pden);
         const float u = float(clamp((num * 3 + pden / 2) / pden, 0, 3)) / 3.0;
         const float v = 1.0 - u;
         A += v * v; B += v * u; C += u * u;
         P += v * vec3(wt); Q += u * vec3(wt);
      }
      const float det = A * C - B * B;
      if (abs(det) < 1e-4)
         break;                         /* every texel took the same index: nothing to solve */
      f1 = clamp((C * P - B * Q) / det, vec3(0.0), vec3(255.0));
      f0 = clamp((A * Q - B * P) / det, vec3(0.0), vec3(255.0));
   }

   ivec3 q0 = ivec3(int(f0.r) >> 3, int(f0.g) >> 2, int(f0.b) >> 3);
   ivec3 q1 = ivec3(int(f1.r) >> 3, int(f1.g) >> 2, int(f1.b) >> 3);
   uint p0 = uint((q0.r << 11) | (q0.g << 5) | q0.b);
   uint p1 = uint((q1.r << 11) | (q1.g << 5) | q1.b);
   if (p0 < p1) {                       /* keep c0 > c1: see the punch-through trap above */
      const uint pt = p0; p0 = p1; p1 = pt;
      const ivec3 qt = q0; q0 = q1; q1 = qt;
   }

   /* Dequantise exactly as the texture unit will, replicating high bits into the low ones. */
   const ivec3 d0 = ivec3((q0.r << 3) | (q0.r >> 2), (q0.g << 2) | (q0.g >> 4), (q0.b << 3) | (q0.b >> 2));
   const ivec3 d1 = ivec3((q1.r << 3) | (q1.r >> 2), (q1.g << 2) | (q1.g >> 4), (q1.b << 3) | (q1.b >> 2));

   const ivec3 dir = d0 - d1;
   const int den = dir.r * dir.r + dir.g * dir.g + dir.b * dir.b;
   uint cidx = 0u;
   if (den > 0) {
      for (int t = 0; t < 16; t++) {
         const ivec3 rel = bc_px_get(t).rgb - d1;
         const int num = clamp(rel.r * dir.r + rel.g * dir.g + rel.b * dir.b, 0, den);
         const int q = clamp((num * 3 + den / 2) / den, 0, 3);
         cidx |= uint(bc_dxt_cidx[q]) << uint(2 * t);
      }
   }
   /* den == 0: endpoints quantised equal; index 0 is still that colour in 3-colour mode. */

   /* BC3 is little-endian throughout -- no byte reversal, unlike the EAC carrier. */
   return uvec4(alo, ahi, p0 | (p1 << 16), cidx);
}
#endif

bool bc_is_anchor(int n_subsets, int part, int texel)
{
   if (texel == 0)
      return true;
   if (n_subsets == 2)
      return bc_anchor2[part] == texel;
   if (n_subsets == 3)
      return bc_anchor3b[part] == texel || bc_anchor3c[part] == texel;
   return false;
}

int bc_anchors_before(int n_subsets, int part, int texel)
{
   if (texel == 0)
      return 0;
   int count = 1;
   if (n_subsets == 2) {
      if (texel > bc_anchor2[part])
         count++;
   } else if (n_subsets == 3) {
      if (texel > bc_anchor3b[part])
         count++;
      if (texel > bc_anchor3c[part])
         count++;
   }
   return count;
}

/* ------------------------------------------------------------ BC4 / BC5 (RGTC) */

/* One RGTC channel of a whole 4x4 block. The palette is built once; only the index is per texel.
 * Palette expressions keep the original order, so float results are bit-exact. */
void bc_rgtc_block(uvec4 blk, int base_bit, bool is_signed, out float out_vals[16])
{
   uint r0 = bc_bits(blk, base_bit, 8);
   uint r1 = bc_bits(blk, base_bit + 8, 8);

   float e0, e1;
   bool eight;
   if (is_signed) {
      int s0 = bc_sign_extend(int(r0), 8);
      int s1 = bc_sign_extend(int(r1), 8);
      if (s0 == -128) s0 = -127;
      if (s1 == -128) s1 = -127;
      e0 = float(s0) / 127.0;
      e1 = float(s1) / 127.0;
      eight = s0 > s1;
   } else {
      e0 = float(r0) / 255.0;
      e1 = float(r1) / 255.0;
      eight = r0 > r1;
   }

   float pal[8];
   pal[0] = e0;
   pal[1] = e1;
   if (eight) {
      for (int i = 2; i < 8; i++)
         pal[i] = mix(e0, e1, float(uint(i) - 1u) / 7.0);
   } else {
      for (int i = 2; i < 6; i++)
         pal[i] = mix(e0, e1, float(uint(i) - 1u) / 5.0);
      pal[6] = is_signed ? -1.0 : 0.0;
      pal[7] = 1.0;
   }

   /* Per texel: one bit extract and one palette read. base_bit is not mutated. */
   for (int texel = 0; texel < 16; texel++)
      out_vals[texel] = pal[bc_bits(blk, base_bit + 16 + 3 * texel, 3)];
}

/* ------------------------------------------------------------------- BC7 */

/* Per-mode parameters (Mesa's bptc_unorm_modes). */
#if !defined(BC_TABLES_BUFFER)
const int bc7_nsub[8]  = int[](3, 2, 3, 2, 1, 1, 1, 2);
const int bc7_npb[8]   = int[](4, 6, 6, 6, 0, 0, 0, 6);
const int bc7_rot[8]   = int[](0, 0, 0, 0, 1, 1, 0, 0);
const int bc7_isel[8]  = int[](0, 0, 0, 0, 1, 0, 0, 0);
const int bc7_cbits[8] = int[](4, 6, 5, 7, 5, 7, 7, 5);
const int bc7_abits[8] = int[](0, 0, 0, 0, 6, 8, 7, 5);
const int bc7_epb[8]   = int[](1, 0, 0, 1, 0, 0, 1, 1);
const int bc7_spb[8]   = int[](0, 1, 0, 0, 0, 0, 0, 0);
const int bc7_ib[8]    = int[](3, 3, 2, 2, 2, 2, 4, 2);
const int bc7_ib2[8]   = int[](0, 0, 0, 0, 3, 2, 0, 0);
#endif

/* BC7 census: which modes and partitions does content use? BC7 weights equal ASTC's, but only
 * 30/64 two-subset and 11/64 three-subset partitions are reproducible in ASTC.
 * Modes 4/5/6 are single-subset. */
const uint bc7_ok2_lo = 0x27feffffu, bc7_ok2_hi = 0x00100003u;   /* the 30 reproducible 2-subset */
const uint bc7_ok3_lo = 0x00103f10u, bc7_ok3_hi = 0x02000018u;   /* the 11 reproducible 3-subset */

void bc_record_bc7(uvec4 blk)
{
   /* Mode is a unary prefix: the position of the lowest set bit. */
   int mode = -1;
   for (int m = 0; m < 8; m++)
      if ((blk.x & (1u << uint(m))) != 0u) { mode = m; break; }
   if (mode < 0)
      return;                                   /* reserved encoding */

   atomicAdd(hist.bucket[18 + mode], 1u);
   atomicAdd(hist.bucket[30], 1u);

   const int nsub = bc7_nsub[mode];
   if (nsub == 1)
      return;                                   /* modes 4/5/6 -- no partition to match */

   const int part = int(bc_bits(blk, mode + 1, bc7_npb[mode]));
   const uint lo = nsub == 2 ? bc7_ok2_lo : bc7_ok3_lo;
   const uint hi = nsub == 2 ? bc7_ok2_hi : bc7_ok3_hi;
   const bool ok = part < 32 ? (lo & (1u << uint(part))) != 0u
                             : (hi & (1u << uint(part - 32))) != 0u;

   atomicAdd(hist.bucket[(nsub == 2 ? 26 : 28) + (ok ? 0 : 1)], 1u);
}


int bc_expand(int v, int bits)
{
   return (v << (8 - bits)) | (v >> (2 * bits - 8));
}

/* Decode a whole 4x4 BC7 block: header parsed once, per-texel tail unchanged (bit-exact).
 * Guarded out of variants that cannot reach BC7, to cut pipeline compile time. */
#if defined(BC_OUT_UNORM8) || defined(BC_OUT_BC3) || defined(BC_OUT_ASTC)
void bc7_decode_block(uvec4 blk)
{
   int mode = -1;
   for (int i = 0; i < 8; i++) {
      if (bc_bits(blk, i, 1) != 0u) { mode = i; break; }
   }
   if (mode < 0)
      { bc_px = uvec4[4](uvec4(0u), uvec4(0u), uvec4(0u), uvec4(0u)); return; }  /* reserved */

   int bit = mode + 1;
   int nsub = bc7_nsub[mode];
   int part = int(bc_bits(blk, bit, bc7_npb[mode]));
   bit += bc7_npb[mode];

   uint subsets = nsub == 2 ? bc_part2[part] : (nsub == 3 ? bc_part3[part] : 0u);

   int rotation = 0;
   if (bc7_rot[mode] != 0) { rotation = int(bc_bits(blk, bit, 2)); bit += 2; }
   int isel = 0;
   if (bc7_isel[mode] != 0) { isel = int(bc_bits(blk, bit, 1)); bit += 1; }

   ivec4 ep[6];
   int cbits = bc7_cbits[mode];
   int abits = bc7_abits[mode];

   for (int c = 0; c < 3; c++)
      for (int s = 0; s < nsub; s++)
         for (int e = 0; e < 2; e++) {
            ep[s * 2 + e][c] = int(bc_bits(blk, bit, cbits));
            bit += cbits;
         }

   int ncomp = 3;
   if (abits > 0) {
      for (int s = 0; s < nsub; s++)
         for (int e = 0; e < 2; e++) {
            ep[s * 2 + e][3] = int(bc_bits(blk, bit, abits));
            bit += abits;
         }
      ncomp = 4;
   } else {
      for (int s = 0; s < nsub; s++)
         for (int e = 0; e < 2; e++)
            ep[s * 2 + e][3] = 255;
   }

   if (bc7_epb[mode] != 0) {
      for (int s = 0; s < nsub; s++)
         for (int e = 0; e < 2; e++) {
            int p = int(bc_bits(blk, bit, 1));
            bit += 1;
            for (int c = 0; c < ncomp; c++)
               ep[s * 2 + e][c] = (ep[s * 2 + e][c] << 1) | p;
         }
   } else if (bc7_spb[mode] != 0) {
      for (int s = 0; s < nsub; s++) {
         int p = int(bc_bits(blk, bit, 1));
         bit += 1;
         for (int e = 0; e < 2; e++)
            for (int c = 0; c < ncomp; c++)
               ep[s * 2 + e][c] = (ep[s * 2 + e][c] << 1) | p;
      }
   }

   int extra = bc7_epb[mode] + bc7_spb[mode];
   for (int s = 0; s < nsub; s++)
      for (int e = 0; e < 2; e++) {
         for (int c = 0; c < 3; c++)
            ep[s * 2 + e][c] = bc_expand(ep[s * 2 + e][c], cbits + extra);
         if (abits > 0)
            ep[s * 2 + e][3] = bc_expand(ep[s * 2 + e][3], abits + extra);
      }

   /* Per texel. `bit` is the block-constant base of the index data: do not mutate it here. */
   const int ib = bc7_ib[mode];
   const int ib2 = bc7_ib2[mode];
   const int bit_base = bit;
   for (int texel = 0; texel < 16; texel++) {
      int anchors = bc_anchors_before(nsub, part, texel);

      int sec_bit = bit_base + 16 * ib - nsub + ib2 * texel - anchors;
      int tbit = bit_base + ib * texel - anchors;

      int sub = int((subsets >> (texel * 2)) & 3u);
      bool anchor = bc_is_anchor(nsub, part, texel);

      int idx0 = int(bc_bits(blk, tbit, ib - (anchor ? 1 : 0)));
      int idx1 = 0;
      if (ib2 != 0)
         idx1 = int(bc_bits(blk, sec_bit, ib2 - (anchor ? 1 : 0)));

      int cidx = isel != 0 ? idx1 : idx0;
      int cib = isel != 0 ? ib2 : ib;

      /* Select the subset's endpoints once, so the component loop indexes a plain ivec4. */
      ivec4 epa = ep[sub * 2];
      ivec4 epb = ep[sub * 2 + 1];

      ivec4 res;
      for (int c = 0; c < 3; c++)
         res[c] = bc_interpolate(epa[c], epb[c], cidx, cib);

      int aidx, aib;
      if (ib2 != 0 && isel == 0) {
         aidx = idx1;
         aib = ib2;
      } else {
         aidx = idx0;
         aib = ib;
      }
      res[3] = bc_interpolate(epa[3], epb[3], aidx, aib);

      if (rotation != 0) {
         int t = res[rotation - 1];
         res[rotation - 1] = res[3];
         res[3] = t;
      }

      bc_px[texel >> 2][texel & 3] =
         uint(res.r) | (uint(res.g) << 8) | (uint(res.b) << 16) | (uint(res.a) << 24);
   }
}
#endif

/* ------------------------------------------------------------------ BC6H */

int bc6_unq_unsigned(int v, int bits)
{
   if (bits >= 15) return v;
   if (v == 0) return 0;
   if (v == (1 << bits) - 1) return 0xffff;
   return ((v << 15) + 0x4000) >> (bits - 1);
}

int bc6_unq_signed(int v, int bits)
{
   if (bits >= 16) return v;
   if (v == 0) return 0;
   bool neg = v < 0;
   if (neg) v = -v;
   if (v >= (1 << (bits - 1)) - 1)
      v = 0x7fff;
   else
      v = ((v << 15) + 0x4000) >> (bits - 1);
   return neg ? -v : v;
}

/* Decode a whole 4x4 BC6H block: mode, field walk, delta expansion and unquantisation run once.
 * Only reachable as BC_OUT_F16; guarded out of other variants. */
#if defined(BC_OUT_F16)
/* Texel t of the block at c0, unless it is outside the image (the last block of a row or column
 * of an image whose size is not a multiple of 4). */
void bc_store_texel(ivec3 c0, int bx, int by, int t, vec4 v)
{
   const int lx = t & 3, ly = t >> 2;
   if (bx * 4 + lx < pc.extent_x && by * 4 + ly < pc.extent_y)
      imageStore(o_img, ivec3(c0.x + lx, c0.y + ly, c0.z), v);
}

/* Stored texel by texel: a 16-texel array would be on the stack on Mali. */
void bc6h_decode_store(uvec4 blk, bool is_signed, ivec3 c0, int bx, int by)
{
   int mode, bit;
   if (bc_bits(blk, 1, 1) != 0u) {
      /* Mesa: mode = (((block[0] >> 1) & 0xe) | (block[0] & 1)) + 2 */
      mode = int(((bc_bits(blk, 0, 8) >> 1) & 0xeu) | bc_bits(blk, 0, 1)) + 2;
      bit = 5;
   } else {
      mode = int(bc_bits(blk, 0, 2));
      bit = 2;
   }

   if ((bc6_flags[mode] & 1) != 0)   /* reserved mode */
      { for (int t = 0; t < 16; t++) bc_store_texel(c0, bx, by, t, vec4(0.0, 0.0, 0.0, 1.0)); return; }

   int neb = bc6_neb[mode];
   int nib = bc6_nib[mode];
   int npb = bc6_npb[mode];
   int nend = npb != 0 ? 4 : 2;

   ivec4 ep[4];
   for (int e = 0; e < 4; e++)
      ep[e] = ivec4(0);

   for (int f = bc6_foff[mode]; f < bc6_foff[mode + 1]; f++) {
      uint fd = bc6_fields[f];
      int fep = int(fd & 3u);
      int fc = int((fd >> 2) & 3u);
      int foff = int((fd >> 4) & 15u);
      int fnb = int((fd >> 8) & 31u);
      bool rev = ((fd >> 13) & 1u) != 0u;

      int v = int(bc_bits(blk, bit, fnb));
      bit += fnb;

      if (rev) {
         /* Reverse the low fnb bits of v and deposit them at foff. fnb > 0 guards a shift by 32. */
         if (fnb > 0)
            ep[fep][fc] |= int((bitfieldReverse(uint(v)) >> uint(32 - fnb)) << uint(foff));
      } else {
         ep[fep][fc] |= v << foff;
      }
   }

   if ((bc6_flags[mode] & 2) != 0) {   /* endpoints are deltas from endpoint 0 */
      int d[3] = int[](bc6_d0[mode], bc6_d1[mode], bc6_d2[mode]);
      for (int e = 1; e < nend; e++)
         for (int c = 0; c < 3; c++) {
            int v = bc_sign_extend(ep[e][c], d[c]);
            ep[e][c] = (ep[0][c] + v) & ((1 << neb) - 1);
         }
   }

   for (int e = 0; e < nend; e++)
      for (int c = 0; c < 3; c++) {
         if (is_signed)
            ep[e][c] = bc6_unq_signed(bc_sign_extend(ep[e][c], neb), neb);
         else
            ep[e][c] = bc6_unq_unsigned(ep[e][c], neb);
      }

   int part = 0;
   int nsub = 1;
   uint subsets = 0u;
   if (npb != 0) {
      part = int(bc_bits(blk, bit, npb));
      bit += npb;
      subsets = bc_part2[part];
      nsub = 2;
   }

   /* Per texel. `bit` is the block-constant base of the index data: do not mutate it here. */
   const int bit_base = bit;
   for (int texel = 0; texel < 16; texel++) {
      int anchors = bc_anchors_before(nsub, part, texel);
      int tbit = bit_base + nib * texel - anchors;

      int sub = int((subsets >> (texel * 2)) & 3u);
      int index = int(bc_bits(blk, tbit, nib - (bc_is_anchor(nsub, part, texel) ? 1 : 0)));

      ivec4 epa = ep[sub * 2];
      ivec4 epb = ep[sub * 2 + 1];

      vec4 res = vec4(0.0, 0.0, 0.0, 1.0);
      for (int c = 0; c < 3; c++) {
         int v = bc_interpolate(epa[c], epb[c], index, nib);
         if (is_signed)
            v = v < 0 ? int(uint(-v * 31 / 32) | 0x8000u) : (v * 31 / 32);
         else
            v = v * 31 / 64;
         res[c] = unpackHalf2x16(uint(v) & 0xffffu).x;
      }
      bc_store_texel(c0, bx, by, texel, res);
   }
}
#endif

#if defined(BC_OUT_ASTC)
/* ---- BC1/BC2/BC3/BC7 -> ASTC 4x4, for texture units without any BC decoder ----------------------
 *
 * The Exynos 9820's Mali-G76 decodes no BC format at all; ASTC and EAC it does. The C reference of
 * everything below, and the check that the texture unit decodes each block mode exactly as
 * modelled, are probe/mali/astc_bc.h and astcmode.c. Measured there on real RE7 blocks: BC1 within
 * 2 LSB everywhere, BC2/BC3 colour within 4 LSB and alpha 94% within 10, BC7 colour 85.8% within 10.
 *
 * Block modes (4x4 weight grid):  0x042 single plane, 4 weight levels   0x051 single, 3 levels
 *                                 0x053 single, 8 levels                0x442 dual, 4 levels
 *                                 0x451 dual, 3 levels
 * Layout: mode bits 0-10, one partition, CEM at 13-16, the endpoint ISE from bit 17 upward, the
 * weight ISE from bit 127 downward (bit-reversed stream), CCS just below it for dual plane, and
 * dual-plane weights interleaved plane 0 / plane 1 per texel. Endpoint order r0 r1 g0 g1 b0 b1 a0 a1.
 * Endpoint ranges used: 256 (8 plain bits), 128 (7 bits), 48 (a trit and 4 bits). Direct modes
 * swap (with blue contraction) when e1 sums lower than e0, so the pairs are ordered to prevent it. */
/* Generated by probe/mali/astc_tables.py from the ASTC spec procedures. Bytes, four per uint. */
#if !defined(BC_TABLES_BUFFER)
const uint astc_trit_enc[61] = uint[](0x04020100u, 0x09080605u, 0x1211100au, 0x18161514u, 0x07031a19u, 0x1b17130bu, 0x200e0d0cu, 0x25242221u, 0x2a292826u, 0x34323130u, 0x39383635u, 0x2b27233au, 0x2c3b3733u, 0x41402e2du, 0x46454442u, 0x504a4948u, 0x55545251u, 0x5a595856u, 0x534b4743u, 0x4d4c5b57u, 0x8281804eu, 0x88868584u, 0x91908a89u, 0x96959492u, 0x839a9998u, 0x97938b87u, 0x8e8d8c9bu, 0xa4a2a1a0u, 0xa9a8a6a5u, 0xb2b1b0aau, 0xb8b6b5b4u, 0xa7a3bab9u, 0xbbb7b3abu, 0xc0aeadacu, 0xc5c4c2c1u, 0xcac9c8c6u, 0xd4d2d1d0u, 0xd9d8d6d5u, 0xcbc7c3dau, 0xccdbd7d3u, 0x6160cecdu, 0x66656462u, 0x706a6968u, 0x75747271u, 0x7a797876u, 0x736b6763u, 0x6d6c7b77u, 0xe2e1e06eu, 0xe8e6e5e4u, 0xf1f0eae9u, 0xf6f5f4f2u, 0xe3faf9f8u, 0xf7f3ebe7u, 0xeeedecfbu, 0x3c1e1d1cu, 0x5d5c3e3du, 0x9e9d9c5eu, 0xdcbebdbcu, 0x3f1fdeddu, 0xdfbf9f5fu, 0x007e7d7cu);
const uint astc_q48_unq[12] = uint[](0xef10ff00u, 0xcf30df20u, 0xae51be41u, 0x8e719e61u, 0xea15fa05u, 0xc936d926u, 0xa956b946u, 0x88779867u, 0xe41bf40bu, 0xc43bd42bu, 0xa35cb34cu, 0x837c936cu);
const uint astc_q48_near[64] = uint[](0x10000000u, 0x10101010u, 0x20202010u, 0x02022020u, 0x12020202u, 0x12121212u, 0x22222212u, 0x04042222u, 0x04040404u, 0x14141414u, 0x24242414u, 0x06062424u, 0x06060606u, 0x16161616u, 0x26262616u, 0x08082626u, 0x08080808u, 0x18181818u, 0x28281818u, 0x0a282828u, 0x0a0a0a0au, 0x1a1a1a1au, 0x2a2a1a1au, 0x0c2a2a2au, 0x0c0c0c0cu, 0x1c1c1c0cu, 0x2c2c1c1cu, 0x0e2c2c2cu, 0x0e0e0e0eu, 0x1e1e1e0eu, 0x2e2e1e1eu, 0x2e2e2e2eu, 0x2f2f2f2fu, 0x1f1f2f2fu, 0x0f1f1f1fu, 0x0f0f0f0fu, 0x2d2d2d0fu, 0x1d1d2d2du, 0x0d1d1d1du, 0x0d0d0d0du, 0x2b2b2b0du, 0x1b1b2b2bu, 0x1b1b1b1bu, 0x0b0b0b0bu, 0x2929290bu, 0x19192929u, 0x19191919u, 0x09090909u, 0x27270909u, 0x17272727u, 0x17171717u, 0x07070707u, 0x25250707u, 0x15252525u, 0x15151515u, 0x05050505u, 0x23230505u, 0x13232323u, 0x13131313u, 0x03030313u, 0x21210303u, 0x11212121u, 0x11111111u, 0x01010111u);
const uint astc_q128_near[64] = uint[](0x01010000u, 0x03030202u, 0x05050404u, 0x07070606u, 0x09090808u, 0x0b0b0a0au, 0x0d0d0c0cu, 0x0f0f0e0eu, 0x11111010u, 0x13131212u, 0x15151414u, 0x17171616u, 0x19191818u, 0x1b1b1a1au, 0x1d1d1c1cu, 0x1f1f1e1eu, 0x21212020u, 0x23232222u, 0x25252424u, 0x27272626u, 0x29292828u, 0x2b2b2a2au, 0x2d2d2c2cu, 0x2f2f2e2eu, 0x31313030u, 0x33333232u, 0x35353434u, 0x37373636u, 0x39393838u, 0x3b3b3a3au, 0x3d3d3c3cu, 0x3f3f3e3eu, 0x41404040u, 0x43424241u, 0x45444443u, 0x47464645u, 0x49484847u, 0x4b4a4a49u, 0x4d4c4c4bu, 0x4f4e4e4du, 0x5150504fu, 0x53525251u, 0x55545453u, 0x57565655u, 0x59585857u, 0x5b5a5a59u, 0x5d5c5c5bu, 0x5f5e5e5du, 0x6160605fu, 0x63626261u, 0x65646463u, 0x67666665u, 0x69686867u, 0x6b6a6a69u, 0x6d6c6c6bu, 0x6f6e6e6du, 0x7170706fu, 0x73727271u, 0x75747473u, 0x77767675u, 0x79787877u, 0x7b7a7a79u, 0x7d7c7c7bu, 0x7f7e7e7du);
#endif
int astc_byte(uint w, int i) { return int((w >> uint((i & 3) * 8)) & 0xffu); }
int astc_trit_enc_at(int i) { return astc_byte(astc_trit_enc[i >> 2], i); }
int astc_q48_unq_at(int i) { return astc_byte(astc_q48_unq[i >> 2], i); }
int astc_q48_near_at(int i) { return astc_byte(astc_q48_near[i >> 2], i); }
int astc_q128_near_at(int i) { return astc_byte(astc_q128_near[i >> 2], i); }

/* The block being built. e0 and e1 are the two endpoints per channel, which the ISE writes in the
 * order r0 r1 g0 g1 b0 b1 a0 a1. The weights are packed in the order the ISE writes them, 3 bits
 * each for 8 levels and 2 bits otherwise, plane 0 / plane 1 interleaved per texel for dual plane.
 * No arrays, here or below: Mali keeps any array of more than 8 entries on the stack. */
struct AstcBlk {
   int mode;
   int dual;
   int ccs;
   int cem;
   int ep_range;
   int w_levels;
   ivec4 e0;
   ivec4 e1;
   uvec2 w;
};

/* n <= 8 bits at bit `off` of a 64-bit value, read and replaced. */
uint astc_get64(uvec2 v, int off, int n)
{
   const uint x = off >= 32 ? v.y >> uint(off - 32)
                            : (v.x >> uint(off)) | (off == 0 ? 0u : v.y << uint(32 - off));
   return x & ((1u << uint(n)) - 1u);
}

void astc_set64(inout uvec2 v, int off, int n, uint val)
{
   const uint m = (1u << uint(n)) - 1u;
   val &= m;
   if (off >= 32) {
      v.y = (v.y & ~(m << uint(off - 32))) | (val << uint(off - 32));
   } else {
      v.x = (v.x & ~(m << uint(off))) | (val << uint(off));
      if (off + n > 32)
         v.y = (v.y & ~(m >> uint(32 - off))) | (val >> uint(32 - off));
   }
}

int astc_wbits(int levels)
{
   return levels == 8 ? 3 : 2;
}

int astc_w_get(AstcBlk b, int i)
{
   const int nb = astc_wbits(b.w_levels);
   return int(astc_get64(b.w, i * nb, nb));
}

void astc_w_set(inout AstcBlk b, int i, int v)
{
   const int nb = astc_wbits(b.w_levels);
   astc_set64(b.w, i * nb, nb, uint(v));
}

/* Endpoint k in ISE order. */
int astc_ep(AstcBlk b, int k)
{
   return (k & 1) != 0 ? b.e1[k >> 1] : b.e0[k >> 1];
}

/* n <= 32 bits of v at bit `off` of the block, whose bits there are still 0. */
void astc_put(inout uvec4 o, int off, int n, uint v)
{
   if (n < 32)
      v &= (1u << uint(n)) - 1u;
   const int w = off >> 5;
   const uint s = uint(off & 31);
   const uint lo = v << s;
   const uint hi = s != 0u && int(s) + n > 32 ? v >> (32u - s) : 0u;
   o.x |= w == 0 ? lo : 0u;
   o.y |= w == 1 ? lo : (w == 0 ? hi : 0u);
   o.z |= w == 2 ? lo : (w == 1 ? hi : 0u);
   o.w |= w == 3 ? lo : (w == 2 ? hi : 0u);
}

int astc_ep_unq(int range, int v)
{
   return range == 256 ? v : range == 128 ? ((v << 1) | (v >> 6)) : astc_q48_unq_at(v);
}

int astc_ep_near(int range, int x)
{
   return range == 256 ? x : range == 128 ? astc_q128_near_at(x) : astc_q48_near_at(x);
}

int astc_w_unq(int levels, int v)
{
   if (levels == 3)
      return v * 32;
   int r = levels == 8 ? ((v << 3) | v) : ((v << 4) | (v << 2) | v);
   return r > 32 ? r + 1 : r;
}

/* The endpoint ISE from bit 17: 8 or 7 plain bits per value, or for range 48 a trit and 4 bits,
 * five values per group laid out m0 T[1:0] m1 T[3:2] m2 T[4] m3 T[6:5] m4 T[7] and truncated
 * after the last value. */
void astc_put_ep(inout uvec4 o, AstcBlk b)
{
   const int nep = b.cem == 12 ? 8 : 6;
   if (b.ep_range != 48) {
      const int nb = b.ep_range == 128 ? 7 : 8;
      for (int k = 0; k < nep; k++)
         astc_put(o, 17 + k * nb, nb, uint(astc_ep(b, k)));
      return;
   }
   int pos = 17;
   for (int g = 0; g < nep; g += 5) {
      const int n = min(nep - g, 5);
      int tv = 0;
      for (int i = n - 1; i >= 0; i--)
         tv = tv * 3 + (astc_ep(b, g + i) >> 4);
      const uint T = uint(astc_trit_enc_at(tv));
      for (int i = 0; i < n; i++) {
         /* T bits after value i: from bit 0 2 4 5 7, 2 2 1 2 1 of them. */
         const uint tlo = (0x75420u >> uint(4 * i)) & 15u;
         const uint tn = (0x12122u >> uint(4 * i)) & 15u;
         astc_put(o, pos, 4, uint(astc_ep(b, g + i) & 15));
         astc_put(o, pos + 4, int(tn), (T >> tlo) & ((1u << tn) - 1u));
         pos += 4 + int(tn);
      }
   }
}

/* The weight ISE, bit 0 first, and its length in bits. With 3 levels a weight is a trit alone,
 * so a group of five is the 8 bits of T, truncated to 2 4 5 7 8 bits after 1 2 3 4 5 values. */
int astc_w_stream(AstcBlk b, out uvec2 ws)
{
   const int nw = b.dual != 0 ? 32 : 16;
   if (b.w_levels != 3) {
      const int nbits = nw * astc_wbits(b.w_levels);
      ws = uvec2(b.w.x, nbits >= 64 ? b.w.y : b.w.y & ((1u << uint(nbits - 32)) - 1u));
      return nbits;
   }
   ws = uvec2(0u);
   int pos = 0;
   for (int g = 0; g < nw; g += 5) {
      const int n = min(nw - g, 5);
      int tv = 0;
      for (int i = n - 1; i >= 0; i--)
         tv = tv * 3 + astc_w_get(b, g + i);
      const int nt = int((0x875420u >> uint(4 * n)) & 15u);
      astc_set64(ws, pos, nt, uint(astc_trit_enc_at(tv)));
      pos += nt;
   }
   return pos;
}

uvec4 astc_pack(AstcBlk b)
{
   uvec4 o = uvec4(0u);
   astc_put(o, 0, 11, uint(b.mode));
   astc_put(o, 13, 4, uint(b.cem));
   astc_put_ep(o, b);

   /* The weight stream runs down from bit 127, bit-reversed. */
   uvec2 ws;
   const int wbits = astc_w_stream(b, ws);
   o.w |= bitfieldReverse(ws.x);
   o.z |= bitfieldReverse(ws.y);
   if (b.dual != 0)
      astc_put(o, 128 - wbits - 2, 2, uint(b.ccs));
   return o;
}

int astc_sum3(AstcBlk b, bool hi)
{
   const ivec4 e = hi ? b.e1 : b.e0;
   return astc_ep_unq(b.ep_range, e.r) + astc_ep_unq(b.ep_range, e.g) +
          astc_ep_unq(b.ep_range, e.b);
}

void astc_order(inout AstcBlk b)
{
   if (astc_sum3(b, true) >= astc_sum3(b, false))
      return;
   const ivec4 t = b.e0;
   b.e0 = b.e1;
   b.e1 = t;
   /* Every weight w becomes levels - 1 - w: the complement for 4 and 8 levels, 0 <-> 2 for 3. */
   if (b.w_levels == 3)
      b.w ^= (~b.w & uvec2(0x55555555u)) << 1u;
   else
      b.w = ~b.w;
}

ivec3 astc_565(uint c)
{
   int r = int((c >> 11) & 31u), g = int((c >> 5) & 63u), bl = int(c & 31u);
   return ivec3((r << 3) | (r >> 2), (g << 2) | (g >> 4), (bl << 3) | (bl >> 2));
}

AstcBlk astc_blk_init()
{
   AstcBlk b;
   b.mode = 0;
   b.dual = 0;
   b.ccs = 3;
   b.cem = 8;
   b.ep_range = 256;
   b.w_levels = 4;
   b.e0 = ivec4(0);
   b.e1 = ivec4(0);
   b.w = uvec2(0u);
   return b;
}

/* BC1: a 4-colour block keeps its endpoints and indices; a 3-colour block needs the midpoint, so
 * trits; one with transparent texels also puts alpha on its own plane. `off` is the bit offset of
 * the colour block in blk. The index maps are nibble tables: entry i at bits 4i. */
uvec4 astc_from_bc1(uvec4 blk, int off, bool rgba)
{
   const uint c0 = bc_bits(blk, off, 16), c1 = bc_bits(blk, off + 16, 16);
   const uint idx = bc_bits(blk, off + 32, 32);
   const ivec3 e0 = astc_565(c0), e1 = astc_565(c1);
   AstcBlk b = astc_blk_init();
   const bool uses3 = (idx & (idx >> 1) & 0x55555555u) != 0u;

   if (c0 > c1) {
      b.mode = 0x042;
      for (int t = 0; t < 16; t++)
         astc_w_set(b, t, int((0x2130u >> (4u * ((idx >> uint(2 * t)) & 3u))) & 15u));
   } else if (!(uses3 && rgba)) {
      /* Index 3 of an opaque BC1 is black, off the line: take the palette entry nearest black. */
      const uint black = (e0.r + e0.g + e0.b) <= (e1.r + e1.g + e1.b) ? 0u : 2u;
      const uint map3 = 0x0120u | (black << 12);
      b.mode = 0x051;
      b.w_levels = 3;
      for (int t = 0; t < 16; t++)
         astc_w_set(b, t, int((map3 >> (4u * ((idx >> uint(2 * t)) & 3u))) & 15u));
   } else {
      b.mode = 0x451;
      b.dual = 1;
      b.cem = 12;
      b.ep_range = 128;
      b.w_levels = 3;
      for (int t = 0; t < 16; t++) {
         const uint i = (idx >> uint(2 * t)) & 3u;
         astc_w_set(b, t * 2, int((0x0120u >> (4u * i)) & 15u));
         astc_w_set(b, t * 2 + 1, i == 3u ? 0 : 2);
      }
      b.e0.a = 0;   /* alpha 0 */
      b.e1.a = 127; /* alpha 255 */
   }
   for (int c = 0; c < 3; c++) {
      b.e0[c] = astc_ep_near(b.ep_range, e0[c]);
      b.e1[c] = astc_ep_near(b.ep_range, e1[c]);
   }
   astc_order(b);
   return astc_pack(b);
}

/* BC2/BC3: the colour half (always four colours) on plane 0, the 16 alpha values on plane 1. */
uvec4 astc_from_bc23(uvec4 blk, bool bc3)
{
   int pal[8] = int[](0, 0, 0, 0, 0, 0, 0, 0);
   if (bc3) {
      const int a0 = int(bc_bits(blk, 0, 8)), a1 = int(bc_bits(blk, 8, 8));
      pal[0] = a0;
      pal[1] = a1;
      if (a0 > a1) {
         for (int i = 1; i < 7; i++)
            pal[i + 1] = ((7 - i) * a0 + i * a1 + 3) / 7;
      } else {
         for (int i = 1; i < 5; i++)
            pal[i + 1] = ((5 - i) * a0 + i * a1 + 2) / 5;
         pal[6] = 0;
         pal[7] = 255;
      }
   }

   const uint c0 = bc_bits(blk, 64, 16), c1 = bc_bits(blk, 80, 16), idx = bc_bits(blk, 96, 32);
   const ivec3 e0 = astc_565(c0), e1 = astc_565(c1);
   AstcBlk b = astc_blk_init();
   b.mode = 0x442;
   b.dual = 1;
   b.cem = 12;
   b.ep_range = 48;
   int amin = 255, amax = 0;
   for (int t = 0; t < 16; t++) {
      astc_w_set(b, t * 2, int((0x2130u >> (4u * ((idx >> uint(2 * t)) & 3u))) & 15u));
      const int a = bc3 ? pal[bc_bits(blk, 16 + 3 * t, 3)] : int(bc_bits(blk, 4 * t, 4)) * 17;
      amin = min(amin, a);
      amax = max(amax, a);
   }
   for (int c = 0; c < 3; c++) {
      b.e0[c] = astc_ep_near(48, e0[c]);
      b.e1[c] = astc_ep_near(48, e1[c]);
   }
   b.e0.a = astc_ep_near(48, amin);
   b.e1.a = astc_ep_near(48, amax);
   const int A0 = astc_ep_unq(48, b.e0.a) * 257, A1 = astc_ep_unq(48, b.e1.a) * 257;
   for (int t = 0; t < 16; t++) {
      const int a = bc3 ? pal[bc_bits(blk, 16 + 3 * t, 3)] : int(bc_bits(blk, 4 * t, 4)) * 17;
      int best = 0, bd = 1 << 30;
      for (int q = 0; q < 4; q++) {
         const int w = astc_w_unq(4, q);
         const int d = abs(((A0 * (64 - w) + A1 * w + 32) >> 6) - a * 257);
         if (d < bd) {
            bd = d;
            best = q;
         }
      }
      astc_w_set(b, t * 2 + 1, best);
   }
   astc_order(b);
   return astc_pack(b);
}

/* Decoded texels (BC7, in bc_px): CEM 8 with 8 weight levels when opaque; with alpha, dual-plane
 * CEM 12 (alpha on its own weights) or single-plane CEM 12 with 8-bit endpoints, whichever fits
 * better. */
void astc_fit_weights(inout AstcBlk b)
{
   ivec4 E0, E1;
   for (int c = 0; c < 3; c++) {
      E0[c] = astc_ep_unq(b.ep_range, b.e0[c]);
      E1[c] = astc_ep_unq(b.ep_range, b.e1[c]);
   }
   E0.a = b.cem == 12 ? astc_ep_unq(b.ep_range, b.e0.a) : 255;
   E1.a = b.cem == 12 ? astc_ep_unq(b.ep_range, b.e1.a) : 255;
   for (int t = 0; t < 16; t++) {
      const ivec4 tx = bc_px_get(t);
      for (int pl = 0; pl < (b.dual != 0 ? 2 : 1); pl++) {
         int best = 0, bd = 1 << 30;
         for (int q = 0; q < b.w_levels; q++) {
            const int w = astc_w_unq(b.w_levels, q);
            int d2 = 0;
            for (int c = 0; c < 4; c++) {
               if (b.dual != 0 && ((c == b.ccs) != (pl == 1)))
                  continue;
               const int d = ((E0[c] * (64 - w) + E1[c] * w + 32) >> 6) - tx[c];
               d2 += d * d;
            }
            if (d2 < bd) {
               bd = d2;
               best = q;
            }
         }
         astc_w_set(b, b.dual != 0 ? t * 2 + pl : t, best);
      }
   }
}

int astc_fit_err(AstcBlk b)
{
   ivec4 E0, E1;
   for (int c = 0; c < 3; c++) {
      E0[c] = astc_ep_unq(b.ep_range, b.e0[c]);
      E1[c] = astc_ep_unq(b.ep_range, b.e1[c]);
   }
   E0.a = b.cem == 12 ? astc_ep_unq(b.ep_range, b.e0.a) : 255;
   E1.a = b.cem == 12 ? astc_ep_unq(b.ep_range, b.e1.a) : 255;
   int e = 0;
   for (int t = 0; t < 16; t++) {
      const ivec4 tx = bc_px_get(t);
      for (int c = 0; c < 4; c++) {
         const int wi = b.dual != 0 ? astc_w_get(b, t * 2 + (c == b.ccs ? 1 : 0)) : astc_w_get(b, t);
         const int w = astc_w_unq(b.w_levels, wi);
         const int v = (E0[c] * 257 * (64 - w) + E1[c] * 257 * w + 32) >> 6;
         const int d = (v >> 8) - tx[c];
         e += d * d;
      }
   }
   return e;
}

uvec4 astc_from_texels()
{
   ivec4 lo = ivec4(255), hi = ivec4(0);
   for (int t = 0; t < 16; t++) {
      lo = min(lo, bc_px_get(t));
      hi = max(hi, bc_px_get(t));
   }
   if (lo.a == 255) {
      AstcBlk b = astc_blk_init();
      b.mode = 0x053;
      b.w_levels = 8;
      b.e0 = ivec4(lo.rgb, 0);
      b.e1 = ivec4(hi.rgb, 0);
      astc_fit_weights(b);
      return astc_pack(b);
   }
   AstcBlk best = astc_blk_init();
   int be = 1 << 30;
   for (int dual = 0; dual < 2; dual++) {
      AstcBlk b = astc_blk_init();
      b.mode = dual != 0 ? 0x442 : 0x042;
      b.dual = dual;
      b.cem = 12;
      b.ep_range = dual != 0 ? 48 : 256;
      for (int c = 0; c < 4; c++) {
         b.e0[c] = astc_ep_near(b.ep_range, lo[c]);
         b.e1[c] = astc_ep_near(b.ep_range, hi[c]);
      }
      astc_order(b);
      astc_fit_weights(b);
      const int e = astc_fit_err(b);
      if (e < be) {
         be = e;
         best = b;
      }
   }
   return astc_pack(best);
}
#endif

/* ------------------------------------------------------------------- main */

void main()
{
   /* One invocation per 4x4 block; radv_meta_bc_decode.c sizes the dispatch in blocks.
    * Skip out-of-range invocations: a zero block has no BC7 mode bit and would read the mode
    * tables out of bounds, corrupting other lanes. */
   ivec2 nblocks = ivec2((pc.extent_x + 3) >> 2, (pc.extent_y + 3) >> 2);
   if (gl_GlobalInvocationID.x >= uint(nblocks.x) || gl_GlobalInvocationID.y >= uint(nblocks.y))
      return;

   int bx = int(gl_GlobalInvocationID.x);
   int by = int(gl_GlobalInvocationID.y);
   int layer = int(gl_GlobalInvocationID.z);

   /* Top-left texel of this block. pc.offset is block-aligned, so >>2 gives the block index. */
   ivec3 c0 = ivec3(pc.offset.x + bx * 4, pc.offset.y + by * 4, pc.offset.z + layer);
#if defined(BC_SRC_BUFFER)
   uvec4 blk = BcSrcBlocks(pc.src_va).b[uint(layer) * pc.src_slice_blocks + uint(by) * pc.src_row_blocks + uint(bx)];
#else
   uvec4 blk = texelFetch(s_blocks, ivec3(c0.x >> 2, c0.y >> 2, c0.z), 0);
#endif

   /* [BCCONTENT]: sample 64 source blocks per dispatch for the content-repeat census.
    * Unsynchronised reads can show false distinct keys, never false matches. */
   if (pc.hash_slot >= 0) {
      const int total = nblocks.x * nblocks.y;
      const int idx = by * nblocks.x + bx;
      const int stride = max(1, total / 64);
      if (idx % stride == 0) {
         const int k = idx / stride;
         if (k < 64) {
            uint h = blk.x * 0x9e3779b9u;
            h ^= blk.y * 0x85ebca6bu;
            h ^= blk.z * 0xc2b2ae35u;
            h ^= blk.w * 0x27d4eb2fu;
            h ^= uint(idx) * 0x165667b1u;
            h ^= h >> 15;
            hist.bucket[BC_HASH_BASE + uint(pc.hash_slot) * 64u + uint(k)] = h;
         }
      }
   }

   /* 1 in 64 blocks feeds the histogram (endpoints are bytes 0-1, already read).
    * Skipped on the transcode path, which must stay ALU-free. */
#if !defined(BC_OUT_U32X4) && !defined(BC_OUT_EAC) && !defined(BC_OUT_BC3) && \
    !defined(BC_OUT_ASTC) && !defined(BC_OUT_EAC_R)
   if (((bx | by) & 7) == 0) {
      if (pc.format == BC_FMT_BC4_UNORM || pc.format == BC_FMT_BC4_SNORM) {
         bc_record_spread(blk, 0, int(bc_bits(blk, 0, 8)), int(bc_bits(blk, 8, 8)));
      } else if (pc.format == BC_FMT_BC5_UNORM || pc.format == BC_FMT_BC5_SNORM) {
         int r0 = int(bc_bits(blk, 0, 8)), r1 = int(bc_bits(blk, 8, 8));
         int g0 = int(bc_bits(blk, 64, 8)), g1 = int(bc_bits(blk, 72, 8));
         bc_record_spread(blk, 0, r0, r1);
         bc_record_spread(blk, 64, g0, g1);
         bc_record_rg(r0, r1, g0, g1);
      } else if (pc.format == BC_FMT_BC7) {
         bc_record_bc7(blk);
      }
   }
#endif

#if defined(BC_OUT_U32X4)
   /* BC4 -> BC3, no decode: BC3's alpha half is a BC4 block, so the hardware decodes it.
    * The colour half is zeroed; the view swizzles R <- A (radv_image_view.c).
    * The BC3 alpha unit is off by up to 2/255 on ~23% of texels.
    * BC4_SNORM is not routed here: the alpha block decodes as unorm. */
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), uvec4(blk.x, blk.y, 0u, 0u));
#elif defined(BC_OUT_EAC)
   /* BC5 -> EAC_R11G11, no decode: two BC4-channel re-encodes via lookup (see bc_eac_encode()).
    * The plane stays compressed at 16 bytes per block. BC5_SNORM is not routed here. */
   uvec2 r = bc_eac_encode(blk.xy);
   uvec2 g = bc_eac_encode(blk.zw);
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), uvec4(r.x, r.y, g.x, g.y));
#elif defined(BC_OUT_EAC_R)
   /* BC4 -> EAC_R11, the same re-encode for one channel. */
   uvec2 r = bc_eac_encode(blk.xy);
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), uvec4(r.x, r.y, 0u, 0u));
#elif defined(BC_OUT_ASTC)
   uvec4 o;
   if (pc.format == BC_FMT_BC7) {
      bc7_decode_block(blk);
      o = astc_from_texels();
   } else if (pc.format == BC_FMT_BC2 || pc.format == BC_FMT_BC3) {
      o = astc_from_bc23(blk, pc.format == BC_FMT_BC3);
   } else {
      o = astc_from_bc1(blk, 0, pc.format == BC_FMT_BC1_RGBA);
   }
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), o);
#elif defined(BC_OUT_BC3)
   /* BC7 -> BC3: same shape, no endpoint search. See bc_bc3_encode(). */
   bc7_decode_block(blk);
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), bc_bc3_encode());
#elif defined(BC_OUT_F16)
   if (pc.format == BC_FMT_BC6H_UF16 || pc.format == BC_FMT_BC6H_SF16) {
      bc6h_decode_store(blk, pc.format == BC_FMT_BC6H_SF16, c0, bx, by);
   } else {
      for (int t = 0; t < 16; t++)
         bc_store_texel(c0, bx, by, t, vec4(0.0, 0.0, 0.0, 1.0));
   }
#else
   vec4 texels[16];
   float red[16], green[16];
   switch (pc.format) {
/* RGTC reaches this switch only as BC_OUT_SNORM8 (BC4/BC5_UNORM go to U32X4/EAC). Guarded out of
 * the f16/unorm8 variants to cut compile time. */
#if defined(BC_OUT_SNORM8)
   case BC_FMT_BC4_UNORM:
      bc_rgtc_block(blk, 0, false, red);
      for (int t = 0; t < 16; t++) texels[t] = vec4(red[t], 0.0, 0.0, 1.0);
      break;
   case BC_FMT_BC4_SNORM:
      bc_rgtc_block(blk, 0, true, red);
      for (int t = 0; t < 16; t++) texels[t] = vec4(red[t], 0.0, 0.0, 1.0);
      break;
   case BC_FMT_BC5_UNORM:
      bc_rgtc_block(blk, 0, false, red);
      bc_rgtc_block(blk, 64, false, green);
      for (int t = 0; t < 16; t++) texels[t] = vec4(red[t], green[t], 0.0, 1.0);
      break;
   case BC_FMT_BC5_SNORM:
      bc_rgtc_block(blk, 0, true, red);
      bc_rgtc_block(blk, 64, true, green);
      for (int t = 0; t < 16; t++) texels[t] = vec4(red[t], green[t], 0.0, 1.0);
      break;
#endif
   default: /* BC_FMT_BC7 */
/* Every variant that consumes decoded BC7 texels must be listed here: the #else writes opaque
 * black instead of failing. See also bc_store_format() in radv_meta_bc_decode.c. */
#if defined(BC_OUT_UNORM8)
      bc7_decode_block(blk);
      for (int t = 0; t < 16; t++) texels[t] = vec4(bc_px_get(t)) / 255.0;
#else
      for (int t = 0; t < 16; t++) texels[t] = vec4(0.0, 0.0, 0.0, 1.0);
#endif
      break;
   }

   /* Bounds-check every texel: the image extent need not be a multiple of 4, so the last block in
    * a row or column is partially outside it. */
   for (int t = 0; t < 16; t++) {
      int lx = t & 3, ly = t >> 2;
      if (bx * 4 + lx >= pc.extent_x || by * 4 + ly >= pc.extent_y)
         continue;
      imageStore(o_img, ivec3(c0.x + lx, c0.y + ly, c0.z), texels[t]);
   }
#endif
}
