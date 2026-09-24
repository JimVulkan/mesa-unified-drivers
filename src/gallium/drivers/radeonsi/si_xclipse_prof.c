/*
 * Copyright 2026 JimVulkan
 * SPDX-License-Identifier: MIT
 */

/* GPU side of the Xclipse field profiler (util/u_xclipse_prof.c), for one gfx context while it
 * samples. No register reads: only timestamps the CP writes, as for queries.
 *  - Busy time: every IB writes a top-of-pipe timestamp when the CP starts it and a bottom-of-pipe
 *    one when its work retires; the union of the spans is the time the GPU spent on this app.
 *  - Passes: a bottom-of-pipe timestamp at every framebuffer change, pixel-shader bind and compute
 *    dispatch marks the end of everything before it, so the gap to the next one is that pass's GPU time. Passes are
 *    grouped by kind, shader (first word of its source BLAKE3), VGPRs and target. */

#include "si_pipe.h"
#include "sid.h"
#include "ac_cmdbuf_cp.h"
#include "util/u_xclipse_prof.h"

#if defined(__ANDROID__) && defined(__aarch64__)

#include <stdlib.h>

#define XPROF_SLOTS 16384
#define XPASS_SLOTS (1 << 18)

struct si_xprof_pass {
   uint32_t frame;
   uint32_t shader;
   uint32_t draws;
   uint32_t groups;
   uint16_t w, h;
   uint16_t cb0, zs;
   uint16_t vgprs;
   uint8_t ncb, kind; /* kind: 0 draws into a framebuffer, 1 compute dispatch */
};

static struct si_context *xprof_ctx;

static int
xprof_cmp(const void *a, const void *b)
{
   const uint64_t *x = a, *y = b;
   return x[0] < y[0] ? -1 : x[0] > y[0];
}

struct xpass_group {
   struct si_xprof_pass key;
   double ms;
   unsigned n;
   uint64_t draws, groups;
};

static int
xgroup_cmp(const void *a, const void *b)
{
   const struct xpass_group *x = a, *y = b;
   return x->ms < y->ms ? 1 : x->ms > y->ms ? -1 : 0;
}

static void
si_xprof_report(void *data, FILE *f)
{
   struct si_context *ctx = data;
   const double khz = ctx->screen->info.clock_crystal_freq;

   /* IB busy time */
   const unsigned n = MIN2(ctx->xprof_ts_next, XPROF_SLOTS);
   uint64_t (*pairs)[2] = malloc(sizeof(*pairs) * (n ? n : 1));
   unsigned m = 0;
   for (unsigned i = 0; i < n; i++) {
      const uint64_t s = ctx->xprof_ts_map[2 * i], e = ctx->xprof_ts_map[2 * i + 1];
      if (s && e && e >= s) {
         pairs[m][0] = s;
         pairs[m][1] = e;
         m++;
      }
   }
   qsort(pairs, m, sizeof(*pairs), xprof_cmp);
   uint64_t busy = 0, cur_s = 0, cur_e = 0;
   for (unsigned i = 0; i < m; i++) {
      if (!cur_e || pairs[i][0] > cur_e) {
         busy += cur_e - cur_s;
         cur_s = pairs[i][0];
         cur_e = pairs[i][1];
      } else if (pairs[i][1] > cur_e) {
         cur_e = pairs[i][1];
      }
   }
   busy += cur_e - cur_s;
   const double span = m ? (pairs[m - 1][1] - pairs[0][0]) / khz : 0;
   fprintf(f, "# gpu busy_ms %.1f span_ms %.1f ibs %u busy_pct %.1f\n", busy / khz, span, m,
           span > 0 ? 100.0 * busy / khz / span : 0.0);
   free(pairs);

   /* Passes */
   const unsigned np = MIN2(ctx->xprof_pass_n, XPASS_SLOTS);
   const struct si_xprof_pass *meta = ctx->xprof_pass_meta;
   struct xpass_group *g = calloc(np ? np : 1, sizeof(*g));
   unsigned ng = 0;
   uint32_t f0 = np ? meta[0].frame : 0, f1 = f0;
   double total = 0;
   for (unsigned i = 0; i + 1 < np; i++) {
      const uint64_t t0 = ctx->xprof_pass_map[i], t1 = ctx->xprof_pass_map[i + 1];
      if (!t0 || !t1 || t1 < t0)
         continue;
      const double ms = (t1 - t0) / khz;
      if (ms > 1000)
         continue;
      f1 = MAX2(f1, meta[i].frame);
      struct si_xprof_pass k = meta[i];
      k.frame = 0;
      k.draws = 0;
      k.groups = 0;
      unsigned j;
      for (j = 0; j < ng; j++)
         if (!memcmp(&g[j].key, &k, sizeof(k)))
            break;
      if (j == ng) {
         g[ng].key = k;
         ng++;
      }
      g[j].ms += ms;
      g[j].n++;
      g[j].draws += meta[i].draws;
      g[j].groups += meta[i].groups;
      total += ms;
   }
   qsort(g, ng, sizeof(*g), xgroup_cmp);
   const unsigned frames = MAX2(f1 - f0, 1);
   fprintf(f, "# passes %u groups %u frames %u gpu_ms_per_frame %.2f\n", np, ng, frames,
           total / frames);
   for (unsigned j = 0; j < ng && j < 40; j++) {
      const struct si_xprof_pass *k = &g[j].key;
      fprintf(f,
              "# pass %s shader %08x vgprs %u %ux%u cb %u:%s zs %s  ms/frame %.2f (%.1f%%)  "
              "per-pass %.3f ms  x%u  draws/pass %.1f  groups/pass %.0f\n",
              k->kind ? "CS" : "FB", k->shader, k->vgprs, k->w, k->h, k->ncb,
              k->ncb ? util_format_short_name(k->cb0) : "-",
              k->zs ? util_format_short_name(k->zs) : "-", g[j].ms / frames,
              total > 0 ? 100.0 * g[j].ms / total : 0, g[j].ms / g[j].n, g[j].n,
              (double)g[j].draws / g[j].n, (double)g[j].groups / g[j].n);
   }
   free(g);
}

static bool
si_xprof_setup(struct si_context *ctx)
{
   struct pipe_resource *buf = pipe_buffer_create(&ctx->screen->b, 0, PIPE_USAGE_STAGING,
                                                  XPROF_SLOTS * 16);
   struct pipe_resource *pbuf = pipe_buffer_create(&ctx->screen->b, 0, PIPE_USAGE_STAGING,
                                                   XPASS_SLOTS * 8);
   ctx->xprof_pass_meta = calloc(XPASS_SLOTS, sizeof(struct si_xprof_pass));
   if (!buf || !pbuf || !ctx->xprof_pass_meta)
      goto fail;
   ctx->xprof_ts = si_resource(buf);
   ctx->xprof_pass_ts = si_resource(pbuf);
   buf = pbuf = NULL;
   const unsigned map = PIPE_MAP_READ | PIPE_MAP_WRITE | PIPE_MAP_UNSYNCHRONIZED;
   ctx->xprof_ts_map = ctx->ws->buffer_map(ctx->ws, ctx->xprof_ts->buf, NULL, map);
   ctx->xprof_pass_map = ctx->ws->buffer_map(ctx->ws, ctx->xprof_pass_ts->buf, NULL, map);
   if (!ctx->xprof_ts_map || !ctx->xprof_pass_map)
      goto fail;
   memset(ctx->xprof_ts_map, 0, XPROF_SLOTS * 16);
   memset(ctx->xprof_pass_map, 0, XPASS_SLOTS * 8);
   return true;
fail:
   pipe_resource_reference(&buf, NULL);
   pipe_resource_reference(&pbuf, NULL);
   si_resource_reference(&ctx->xprof_ts, NULL);
   si_resource_reference(&ctx->xprof_pass_ts, NULL);
   free(ctx->xprof_pass_meta);
   ctx->xprof_pass_meta = NULL;
   return false;
}

void
si_xprof_begin_cs(struct si_context *ctx)
{
   if (likely(!u_xclipse_prof_active()) || !ctx->is_gfx_queue ||
       (ctx->context_flags & SI_CONTEXT_FLAG_AUX))
      return;

   if (!xprof_ctx) {
      /* The first gfx context seen while sampling is the one measured. */
      if (!si_xprof_setup(ctx))
         return;
      xprof_ctx = ctx;
      u_xclipse_prof_set_gpu_reporter(si_xprof_report, ctx);
   }
   if (xprof_ctx != ctx)
      return;

   radeon_add_to_buffer_list(ctx, &ctx->gfx_cs, ctx->xprof_pass_ts,
                             RADEON_USAGE_WRITE | RADEON_PRIO_QUERY);
   if (ctx->xprof_ts_next >= XPROF_SLOTS)
      return;
   ctx->xprof_slot = ctx->xprof_ts_next++;
   const uint64_t va = ctx->xprof_ts->gpu_address + ctx->xprof_slot * 16;
   radeon_add_to_buffer_list(ctx, &ctx->gfx_cs, ctx->xprof_ts, RADEON_USAGE_WRITE | RADEON_PRIO_QUERY);
   ac_emit_cp_copy_data(&ctx->gfx_cs.current, COPY_DATA_TIMESTAMP, COPY_DATA_DST_MEM, 0, va,
                        AC_CP_COPY_DATA_WR_CONFIRM | AC_CP_COPY_DATA_COUNT_SEL, false);
   ctx->xprof_slot_open = true;
}

void
si_xprof_end_cs(struct si_context *ctx)
{
   if (!ctx->xprof_slot_open)
      return;
   ctx->xprof_slot_open = false;
   const uint64_t va = ctx->xprof_ts->gpu_address + ctx->xprof_slot * 16 + 8;
   si_cp_release_mem(ctx, &ctx->gfx_cs, V_028A90_BOTTOM_OF_PIPE_TS, 0, EOP_DST_SEL_MEM,
                     EOP_INT_SEL_NONE, EOP_DATA_SEL_TIMESTAMP, NULL, va, 0,
                     PIPE_QUERY_TIMESTAMP);
}

/* Called before a framebuffer change (compute = false) or a dispatch (compute = true). */
void
si_xprof_pass_boundary(struct si_context *ctx, bool compute, unsigned groups)
{
   if (xprof_ctx != ctx || !u_xclipse_prof_active())
      return;

   struct si_xprof_pass *meta = ctx->xprof_pass_meta;
   const unsigned n = ctx->xprof_pass_n;

   /* Close the previous pass: its draws, and for draw passes the last pixel shader it used. */
   if (n && n <= XPASS_SLOTS) {
      struct si_xprof_pass *prev = &meta[n - 1];
      if (!prev->kind) {
         prev->draws = ctx->num_draw_calls - ctx->xprof_pass_draws0;
         struct si_shader_selector *ps = ctx->shader.ps.cso;
         if (ps) {
            memcpy(&prev->shader, ps->info.base.source_blake3, 4);
            if (ctx->shader.ps.current)
               prev->vgprs = ctx->shader.ps.current->config.num_vgprs;
         }
      }
   }
   if (n >= XPASS_SLOTS)
      return;

   struct si_xprof_pass *p = &meta[n];
   memset(p, 0, sizeof(*p));
   p->frame = ctx->xprof_frame;
   p->kind = compute;
   if (compute) {
      struct si_compute *prog = ctx->cs_shader_state.program;
      if (prog) {
         memcpy(&p->shader, prog->sel.info.base.source_blake3, 4);
         p->vgprs = prog->shader.config.num_vgprs;
      }
      p->groups = groups;
   } else {
      const struct pipe_framebuffer_state *fb = &ctx->framebuffer.state;
      p->w = fb->width;
      p->h = fb->height;
      p->ncb = fb->nr_cbufs;
      p->cb0 = fb->nr_cbufs && fb->cbufs[0].texture ? fb->cbufs[0].format : 0;
      p->zs = fb->zsbuf.texture ? fb->zsbuf.format : 0;
   }
   ctx->xprof_pass_draws0 = ctx->num_draw_calls;
   ctx->xprof_pass_n = n + 1;

   si_cp_release_mem(ctx, &ctx->gfx_cs, V_028A90_BOTTOM_OF_PIPE_TS, 0, EOP_DST_SEL_MEM,
                     EOP_INT_SEL_NONE, EOP_DATA_SEL_TIMESTAMP, NULL,
                     ctx->xprof_pass_ts->gpu_address + n * 8, 0, PIPE_QUERY_TIMESTAMP);
}

void
si_xprof_destroy(struct si_context *ctx)
{
   if (xprof_ctx != ctx)
      return;
   u_xclipse_prof_set_gpu_reporter(NULL, NULL);
   xprof_ctx = NULL;
   si_resource_reference(&ctx->xprof_ts, NULL);
   si_resource_reference(&ctx->xprof_pass_ts, NULL);
   free(ctx->xprof_pass_meta);
   ctx->xprof_pass_meta = NULL;
}

#else

void si_xprof_begin_cs(struct si_context *ctx) {}
void si_xprof_end_cs(struct si_context *ctx) {}
void si_xprof_pass_boundary(struct si_context *ctx, bool compute, unsigned groups) {}
void si_xprof_destroy(struct si_context *ctx) {}

#endif
