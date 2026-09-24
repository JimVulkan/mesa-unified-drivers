/*
 * Copyright 2026 XclipseTools
 * SPDX-License-Identifier: MIT
 *
 * kbase_kmod.c -- a pan_kmod backend for ARM's PROPRIETARY kbase kernel driver.
 *
 * WHY THIS EXISTS. pan_kmod ships two backends, panfrost_kmod (DRM panfrost) and panthor_kmod
 * (DRM panthor). Both talk to a DRM render node. A stock Samsung Exynos device has no /dev/dri
 * at all: its GPU is reached through ARM's kbase at /dev/mali0, which is world-openable
 * (crw-rw-rw-, SELinux gpu_device) and needs no root. So PanVK loads on such a device and then
 * enumerates zero physical devices -- correctly, because there is no DRM node to find.
 *
 * WHERE THE UAPI CAME FROM. Not from headers -- ARM's are not in this tree. Every ioctl number,
 * struct layout and semantic below was recovered by recording the VENDOR driver's own traffic
 * (an LD_PRELOAD spy around a real Vulkan workload) and diffing each payload before and after
 * the call, so input fields, output fields and padding separate themselves without anyone
 * deciding what a struct looks like. Three names in the public tables were WRONG for this DDK
 * and the traffic overruled them; they are marked below. See probe/mali/UAPI_NOTES.md.
 *
 * WHY THE MODEL FITS. kbase assigns the GPU VA at allocation time and offers no userspace VA
 * management -- and neither does DRM panfrost, whose backend refuses anything but
 * PAN_KMOD_VM_FLAG_AUTO_VA, one VM per device and whole-BO mappings. The two kernel drivers
 * agree on the shape that matters, so this backend mirrors panfrost_kmod's structure closely.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <pthread.h>
#include <time.h>

#include "drm-uapi/drm.h"
#include "util/macros.h"
#include "util/u_debug.h"
#include "util/u_sync_provider.h"

#include "pan_kmod_backend.h"
#include "kbase_kmod.h"

extern const struct pan_kmod_ops kbase_kmod_ops;

/* ---- kbase UAPI, recovered by measurement ------------------------------------------------- */

#define KBASE_IOCTL_TYPE 0x80

struct kbase_ioctl_version_check {
   uint16_t major;
   uint16_t minor;
};

struct kbase_ioctl_set_flags {
   uint32_t create_flags;
};

struct kbase_ioctl_get_gpuprops {
   uint64_t buffer;
   uint32_t size;
   uint32_t flags;
};

union kbase_ioctl_mem_alloc {
   struct {
      uint64_t va_pages;
      uint64_t commit_pages;
      uint64_t extension;
      uint64_t flags;
   } in;
   struct {
      uint64_t flags;
      uint64_t gpu_va;
   } out;
};

struct kbase_ioctl_mem_free {
   uint64_t gpu_addr;
};

/* ioctl 38 appears in no table we had, and the vendor issues it exactly once with va_pages =
 * 0x100000. It reserves the EXECUTABLE VA zone, and that is not a reading of the number -- it is
 * an A/B: without the call, an allocation with the GPU_EX bit comes back as a SAME_VA cookie and
 * the kernel ORs 0x2000 into the returned flags; with it, the SAME request returns
 * 0x7f00001000, the exact address the vendor got, with the flags unchanged. */
struct kbase_ioctl_mem_exec_init {
   uint64_t va_pages;
};

/* nr 0/1/3/5 are as the public tables have them. nr 7 (MEM_FREE) never appeared in the captured
 * traffic -- the workload never freed anything -- so it was measured directly instead: it
 * SUCCEEDS on an exec-zone allocation and is refused, EINVAL, for every SAME_VA one, mapped or
 * not. That is not a bug in the number; it is the model. See probe/mali/maliva.c. */
#define KBASE_IOCTL_VERSION_CHECK _IOWR(KBASE_IOCTL_TYPE, 0, struct kbase_ioctl_version_check)
#define KBASE_IOCTL_SET_FLAGS     _IOW(KBASE_IOCTL_TYPE, 1, struct kbase_ioctl_set_flags)
#define KBASE_IOCTL_GET_GPUPROPS  _IOW(KBASE_IOCTL_TYPE, 3, struct kbase_ioctl_get_gpuprops)
#define KBASE_IOCTL_MEM_ALLOC     _IOWR(KBASE_IOCTL_TYPE, 5, union kbase_ioctl_mem_alloc)
#define KBASE_IOCTL_MEM_FREE      _IOW(KBASE_IOCTL_TYPE, 7, struct kbase_ioctl_mem_free)
#define KBASE_IOCTL_MEM_EXEC_INIT _IOW(KBASE_IOCTL_TYPE, 38, struct kbase_ioctl_mem_exec_init)

/* nr 22, a dma-buf (UMM) import: phandle points at the fd. The shape is kbase's public one; the
 * vendor could not be spied importing (its ICD loads in the sphal namespace, out of LD_PRELOAD's
 * reach), so what settles it is probe/mali/vkanb.c reading the rendered pixels back through the
 * dma-buf. Like MEM_ALLOC, a SAME_VA import returns a cookie, and the anchor mapping of that
 * cookie is the GPU VA. */
union kbase_ioctl_mem_import {
   struct {
      uint64_t flags;
      uint64_t phandle;
      uint32_t type;
      uint32_t padding;
   } in;
   struct {
      uint64_t flags;
      uint64_t gpu_va;
      uint64_t va_pages;
   } out;
};
#define KBASE_IOCTL_MEM_IMPORT _IOWR(KBASE_IOCTL_TYPE, 22, union kbase_ioctl_mem_import)
#define BASE_MEM_IMPORT_TYPE_UMM 2
#define BASE_MEM_SAME_VA         (1ull << 13)
/* What a SAME_VA import reports back instead of SAME_VA: gpu_va is a cookie to mmap. Measured,
 * the returned flags are 0x500f for a request of 0x200f. */
#define BASE_MEM_NEED_MMAP       (1ull << 14)

/* kbase reserves low mmap page offsets for special handles. */
#define BASE_MEM_MAP_TRACKING_HANDLE (3ull << 12)

/* Allocation flags, taken VERBATIM from what the vendor driver was observed to pass. The
 * individual bit meanings are not fully decoded, and inventing a combination would be a guess
 * where a measurement is available: 0x204f is what it used for ordinary read/write buffers and
 * 0x17 for the executable ones. */
#define KBASE_MEM_FLAGS_RW   (0x204full | BASE_MEM_COHERENT_LOCAL)
#define KBASE_MEM_FLAGS_EXEC 0x17ull

/* Bit 11, which the vendor also passes (its other captured shape is 0x380f). Without it the
 * GPU's own units are not coherent with one another on this memory: the tiler did not see what
 * an earlier tiler job left in the shared tiler context, so of several draws in one render pass
 * only the LAST rasterised -- measured, two direct draws, one survived; with this bit both do.
 * The same incoherence is what the CACHE_FLUSH jobs in the draw path were working around.
 * (CACHED_CPU, bit 12, the vendor's other difference, made every draw vanish: nothing here
 * syncs CPU caches.) */
#define BASE_MEM_COHERENT_LOCAL (1ull << 11)

/* PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT, the tiler heap, the indirect varying buffer and the tess
 * heap: 336 MB between them, all of it committed at vkCreateDevice when this flag was ignored
 * (measured, MemAvailable fell 331 MB; the vendor's device costs 3). GROW_ON_GPF commits
 * KBASE_GROW_INITIAL_PAGES and lets a GPU fault commit the rest KBASE_GROW_STEP_PAGES at a time,
 * which is what the DRM drivers' heap BOs do. */
#define BASE_MEM_GROW_ON_GPF     (1ull << 9)
#define KBASE_GROW_INITIAL_PAGES 256ull
#define KBASE_GROW_STEP_PAGES    512ull

/* What the vendor reserves. Also the only shape the exec zone accepts: adding GPU_WR to
 * KBASE_MEM_FLAGS_EXEC is refused with ENOMEM, and so is dropping GPU_RD -- the zone is for
 * shader code and nothing else. */
#define KBASE_EXEC_ZONE_PAGES 0x100000ull

/* ---- GPU property stream ------------------------------------------------------------------
 *
 * GET_GPUPROPS returns a packed run of (u32 header, value) pairs where header = (id << 2) | w,
 * w selecting a 1/2/4/8-byte value. It returns the BYTE COUNT, not zero, so only a negative
 * return is a failure -- treating non-zero as an error once made a successful 723-byte read
 * report a stale ENOENT.
 *
 * The ids below are anchored to values known independently for the reference part, rather than
 * copied from a table that proved wrong: PRODUCT_ID reads 0x7211 which is exactly the deviceID
 * the vendor's own Vulkan reports (0x72110000); SHADER_PRESENT reads 0x770077, whose popcount is
 * 12, matching a Mali-G76 MP12; and MMU_FEATURES reads 0x2830, i.e. 48 VA bits and 40 PA bits.
 * Three independent confirmations, so the surrounding ids can be trusted by position. */
enum kbase_gpuprop {
   KBASE_GPUPROP_PRODUCT_ID = 1,
   KBASE_GPUPROP_VERSION_STATUS = 2,
   KBASE_GPUPROP_MINOR_REVISION = 3,
   KBASE_GPUPROP_MAJOR_REVISION = 4,
   KBASE_GPUPROP_TEXTURE_FEATURES_0 = 9,
   KBASE_GPUPROP_TEXTURE_FEATURES_1 = 10,
   KBASE_GPUPROP_TEXTURE_FEATURES_2 = 11,
   KBASE_GPUPROP_L2_LOG2_LINE_SIZE = 13,
   KBASE_GPUPROP_L2_LOG2_CACHE_SIZE = 14,
   KBASE_GPUPROP_L2_NUM_L2_SLICES = 15,
   KBASE_GPUPROP_TILER_BIN_SIZE_BYTES = 16,
   KBASE_GPUPROP_TILER_MAX_ACTIVE_LEVELS = 17,
   KBASE_GPUPROP_MAX_THREADS = 18,
   KBASE_GPUPROP_MAX_WORKGROUP_SIZE = 19,
   KBASE_GPUPROP_MAX_BARRIER_SIZE = 20,
   KBASE_GPUPROP_MAX_REGISTERS = 21,
   KBASE_GPUPROP_MAX_TASK_QUEUE = 22,
   KBASE_GPUPROP_IMPL_TECH = 24,
   KBASE_GPUPROP_RAW_SHADER_PRESENT = 25,
   KBASE_GPUPROP_RAW_TILER_PRESENT = 26,
   KBASE_GPUPROP_RAW_L2_PRESENT = 27,
   KBASE_GPUPROP_RAW_STACK_PRESENT = 28,
   KBASE_GPUPROP_RAW_L2_FEATURES = 29,
   KBASE_GPUPROP_RAW_CORE_FEATURES = 30,
   KBASE_GPUPROP_RAW_MEM_FEATURES = 31,
   KBASE_GPUPROP_RAW_MMU_FEATURES = 32,
   KBASE_GPUPROP_RAW_AS_PRESENT = 33,
   KBASE_GPUPROP_RAW_JS_PRESENT = 34,
   /* 51, NOT 84. 84 is TLS_ALLOC, which is also present in the stream, so reading it did not
    * fall back to the default -- it fed Mesa 0x300 where the real value is 0x809. Decoded, that
    * is a tiler with a 1-BYTE bin and 3 hierarchy levels instead of 512 bytes and 8, i.e. a
    * tiler with nowhere to put a primitive. Every job still completed without a fault; the
    * polygon list was simply empty. Confirmed against the bv_r32p1 UAPI header, which is the
    * DDK this device runs. */
   KBASE_GPUPROP_RAW_TILER_FEATURES = 51,
   KBASE_GPUPROP_TLS_ALLOC = 84,
};


struct kbase_sub;

/* Submission state of a device, see "asynchronous submission". */
struct kbase_queue {
   pthread_mutex_t lock;
   pthread_cond_t cond;              /* atom numbers freed, submissions completed */
   struct kbase_sub *owner[256];     /* by atom number, NULL when free */
   bool is_fence[256];
   unsigned free_atoms;              /* of the numbers 1..255 */
   uint8_t last_atom;                /* the newest atom in flight, 0 if none */
   uint64_t next_seq;
   /* Submissions whose callback has not run yet. A fence-only submission (a present) completes on
    * a soft atom, which can report after a later submission's GPU atoms: completion is tracked per
    * submission, not as a high-water mark. */
   uint64_t open_seq[256];
   unsigned nr_open_seq;
   unsigned in_flight;               /* submissions */
   pthread_t thread;
   bool thread_started, stop, fence_warned;
};

struct kbase_kmod_dev {
   struct pan_kmod_dev base;
   struct kbase_kmod_vm *vm;
   uint32_t next_handle;
   struct kbase_queue queue;
};

static void kbase_queue_init(struct kbase_queue *q);
static void kbase_queue_finish(struct kbase_kmod_dev *kdev);

struct kbase_kmod_bo {
   struct pan_kmod_bo base;
   uint64_t gpu_va;
   /* The mapping that FIXES gpu_va for an ordinary (SAME_VA) allocation, and whose lifetime is
    * the allocation's. NULL for an executable one, whose address comes from the kernel. */
   void *anchor;
   bool exec;
};

struct kbase_kmod_vm {
   struct pan_kmod_vm base;
};

static uint64_t
kbase_gpuprop_get(const uint8_t *buf, unsigned size, unsigned want, uint64_t dflt)
{
   for (unsigned off = 0; off + 4 <= size;) {
      uint32_t hdr;
      memcpy(&hdr, buf + off, 4);
      off += 4;

      const unsigned id = hdr >> 2, type = hdr & 3;
      const unsigned w = type == 0 ? 1 : type == 1 ? 2 : type == 2 ? 4 : 8;
      if (off + w > size)
         break;

      if (id == want) {
         uint64_t v = 0;
         memcpy(&v, buf + off, w);
         return v;
      }
      off += w;
   }
   return dflt;
}

static bool
kbase_kmod_query_props(int fd, struct pan_kmod_dev_props *props)
{
   struct kbase_ioctl_get_gpuprops req = {0};

   /* size=0 asks how big the stream is; the return value IS the byte count. */
   const int need = pan_kmod_ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, &req);
   if (need <= 0) {
      mesa_loge("kbase: GET_GPUPROPS size query failed (ret=%d, err=%d) -- not a kbase node?",
                need, errno);
      return false;
   }

   uint8_t *buf = calloc(1, need);
   if (!buf)
      return false;

   req.buffer = (uint64_t)(uintptr_t)buf;
   req.size = need;
   if (pan_kmod_ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, &req) < 0) {
      mesa_loge("kbase: GET_GPUPROPS read failed (err=%d)", errno);
      free(buf);
      return false;
   }

#define P(name, dflt) kbase_gpuprop_get(buf, need, KBASE_GPUPROP_##name, (dflt))

   /* pan_arch() reads the product id out of bits 16..31, so gpu_id has to be the whole GPU_ID
    * register, not just the product. Rebuilt from its parts: 0x7211 becomes 0x72110000, whose
    * PAN_ARCH_MAJOR is 7 -- Bifrost v7, which is exactly what PanVK builds a jm/bifrost target
    * for. Getting this wrong is silent: a bare 0x7211 would decode as some Midgard part. */
   props->gpu_id = (P(PRODUCT_ID, 0) << 16) | ((P(MAJOR_REVISION, 0) & 0xf) << 12) |
                   ((P(MINOR_REVISION, 0) & 0xff) << 4) | (P(VERSION_STATUS, 0) & 0xf);
   props->gpu_variant = 0;

   props->shader_present = P(RAW_SHADER_PRESENT, 1);
   props->tiler_features = P(RAW_TILER_FEATURES, 0x809);
   props->mem_features = P(RAW_MEM_FEATURES, 0);
   props->mmu_features = P(RAW_MMU_FEATURES, 0x2830);
   props->l2_features = P(RAW_L2_FEATURES, 0);

   props->texture_features[0] = P(TEXTURE_FEATURES_0, 0);
   props->texture_features[1] = P(TEXTURE_FEATURES_1, 0);
   props->texture_features[2] = P(TEXTURE_FEATURES_2, 0);
   props->texture_features[3] = 0;

   props->max_threads_per_core = P(MAX_THREADS, 768);
   props->max_threads_per_wg = P(MAX_WORKGROUP_SIZE, props->max_threads_per_core);
   props->max_tasks_per_core = MAX2(P(MAX_TASK_QUEUE, 0), 1);
   props->num_registers_per_core = P(MAX_REGISTERS, 0x6000);

   /* THREAD_TLS_ALLOC: how many thread-storage slots the hardware indexes per core, which is
    * not the thread count. The Mali-G52 in the Galaxy A31 reports 1024 slots for 768 threads;
    * sized for 768, the stack of a thread in a slot past it ran off the end of the allocation,
    * and any spilling kernel (the batch patch job spills 456 bytes) faulted the whole atom at
    * random, depending on which slots the threads landed in. kbase exposes it as TLS_ALLOC (84);
    * pan_kmod's contract falls back to max_threads_per_core when it is 0. */
   props->max_tls_instance_per_core = P(TLS_ALLOC, 0) ?: props->max_threads_per_core;

   /* Matching panfrost_kmod: Mali MMUs take 4K and 2M pages. Coherency is left off
    * because nothing observed says this SoC is IO-coherent, and claiming it would
    * silently drop cache maintenance. */
   props->pgsize_bitmap = PAN_PGSIZE_4K | PAN_PGSIZE_2M;
   props->is_io_coherent = false;

   /* Without this the mask is zero, every queue priority is refused, and vkCreateDevice
    * fails with VK_ERROR_NOT_PERMITTED -- which reads as a permissions problem rather
    * than a field nobody filled in. MEDIUM only: the JM backend does not hook up the
    * other priorities anyway (panvk filters them itself for arch < 10), and claiming a
    * priority kbase has not been asked for would be inventing a capability. */
   props->allowed_group_priorities_mask = PAN_KMOD_GROUP_ALLOW_PRIORITY_MEDIUM;

#if defined(__aarch64__)
   /* The GPU's SYSTEM_TIMESTAMP (what a WRITE_VALUE job stores) is the SoC's system counter, the
    * one the CPU reads as CNTVCT_EL0, so its frequency is CNTFRQ_EL0 and the current value can
    * be read without the kernel. */
   uint64_t freq;
   __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
   if (freq) {
      props->gpu_can_query_timestamp = true;
      props->timestamp_device_coherent = true;
      props->timestamp_frequency = freq;
      props->timestamp_cycles_to_ns_factor = 1000000000.0 / freq;
   }
#endif

#undef P

   free(buf);
   return true;
}

static struct pan_kmod_dev *
kbase_kmod_dev_create(int fd, uint32_t flags, const struct pan_kmod_driver *drv_info,
                      const struct pan_kmod_allocator *allocator)
{
   struct kbase_kmod_dev *kbase_dev = pan_kmod_alloc(allocator, sizeof(*kbase_dev));
   if (!kbase_dev) {
      mesa_loge("kbase: out of memory");
      return NULL;
   }
   memset(kbase_dev, 0, sizeof(*kbase_dev));

   /* The handshake, in the order the vendor driver performs it. VERSION_CHECK VALIDATES rather
    * than reports: the payload comes back unchanged, so a mismatched pair is refused by the
    * kernel rather than negotiated down. 11.31 is what the r32p1 DDK asks for on the reference
    * device.
    *
    * IT IS PER-OPEN, NOT PER-FD. PanVK builds its logical device by dup()ing the physical
    * device's fd, so this function runs a second time on a context that already exists, and
    * kbase refuses the handshake there with EPERM. Failing on that would mean no logical device
    * could ever be created. So the handshake is attempted and, if refused, we simply do not
    * repeat SET_FLAGS: the real gate is whether the node answers GET_GPUPROPS below, which a
    * working kbase context does however it was set up and a non-kbase node never does. */
   struct kbase_ioctl_version_check vc = {.major = 11, .minor = 31};
   const bool fresh_context = pan_kmod_ioctl(fd, KBASE_IOCTL_VERSION_CHECK, &vc) == 0;

   if (fresh_context) {
      struct kbase_ioctl_set_flags sf = {.create_flags = 0};
      if (pan_kmod_ioctl(fd, KBASE_IOCTL_SET_FLAGS, &sf)) {
         mesa_loge("kbase: SET_FLAGS failed (err=%d)", errno);
         goto err_free_dev;
      }

      /* THE ORDER IS SET_FLAGS FIRST, THEN THE TRACKING PAGE, and it is not cosmetic. Mapping
       * it first is refused with EPERM and leaves the context half initialised, after which
       * EVERY MEM_ALLOC fails with EINVAL -- including a single page with the exact flags the
       * vendor uses successfully. That is what the failure looks like, so it invites blaming the
       * allocation flags rather than the setup. This is the vendor's own order, read off the
       * captured trace. */
      if (mmap(NULL, 4096, PROT_NONE, MAP_SHARED, fd, BASE_MEM_MAP_TRACKING_HANDLE) ==
          MAP_FAILED) {
         mesa_loge("kbase: failed to map the tracking page (err=%d)", errno);
         goto err_free_dev;
      }

      /* Reserve the executable VA zone. Without this, an allocation asking for GPU_EX is
       * silently downgraded to SAME_VA and returns a cookie instead of an address -- silently,
       * because nothing fails: the allocation succeeds and only the address is wrong. Not fatal
       * if the kernel refuses it; shader BOs then just live in the ordinary zone. */
      struct kbase_ioctl_mem_exec_init ei = {.va_pages = KBASE_EXEC_ZONE_PAGES};
      if (pan_kmod_ioctl(fd, KBASE_IOCTL_MEM_EXEC_INIT, &ei))
         mesa_logw("kbase: MEM_EXEC_INIT failed (err=%d); executable BOs will be SAME_VA", errno);
   }

   pan_kmod_dev_init(&kbase_dev->base, fd, flags, drv_info, &kbase_kmod_ops, allocator);

   if (!kbase_kmod_query_props(fd, &kbase_dev->base.props)) {
      pan_kmod_dev_cleanup(&kbase_dev->base);
      goto err_free_dev;
   }

   kbase_dev->next_handle = 1;
   kbase_queue_init(&kbase_dev->queue);
   return &kbase_dev->base;

err_free_dev:
   pan_kmod_free(allocator, kbase_dev);
   return NULL;
}

static void
kbase_kmod_dev_destroy(struct pan_kmod_dev *dev)
{
   struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);

   kbase_queue_finish(kbase_dev);
   pan_kmod_dev_cleanup(dev);
   pan_kmod_free(dev->allocator, kbase_dev);
}

static struct pan_kmod_va_range
kbase_kmod_dev_query_user_va_range(const struct pan_kmod_dev *dev)
{
   /* kbase hands out the VA itself, so this only has to describe the space those addresses live
    * in. MMU_FEATURES carries the VA width in its low byte (0x30 = 48 bits on the reference
    * part). The first 4 GB are left alone: kbase uses low addresses for its own special
    * mappings, and the tracking page sits at 3 << 12. */
   const unsigned va_bits = MMU_FEATURES_VA_BITS(dev->props.mmu_features);

   return (struct pan_kmod_va_range){
      .start = 4ull * 1024 * 1024 * 1024,
      .size = (1ull << va_bits) - (4ull * 1024 * 1024 * 1024),
   };
}

/* ---- buffer objects ------------------------------------------------------------------------
 *
 * THE MEMORY MODEL IS NOT WHAT A FIRST READING OF THE TRAFFIC SUGGESTS, and the first version
 * of this backend got it wrong in a way nothing could catch until something was submitted.
 *
 * MEM_ALLOC's out.gpu_va is NOT always an address. Three vendor records:
 *
 *     flags 0x204f -> 0x41000     flags 0x380f -> 0x41000     (185 pages, and 1 page)
 *     flags 0x17   -> 0x0000007f00001000
 *
 * A value that repeats across allocations of different sizes is not an address. 0x41000 is a
 * COOKIE. The measurement that settles it is MEM_QUERY (probe/mali/maliva.c): the kernel does
 * not recognise 0x41000 as a region, and DOES recognise the address the subsequent mmap
 * returned. So for an ordinary allocation:
 *
 *     the GPU VA is the address mmap GIVES BACK, not the value MEM_ALLOC returned.
 *
 * kbase forces BASE_MEM_SAME_VA on for 64-bit clients -- it ORs 0x2000 into the flags it returns
 * even when we did not ask for it -- and a SAME_VA region's GPU address is its CPU address. The
 * one exception is the odd record above, and it is the only one with the GPU_EX bit:
 * executable allocations come from a separate zone with real addresses. That zone does not exist
 * until ioctl 38 reserves it, which is why the vendor calls it exactly once at startup and why,
 * without it, even flags 0x17 comes back as a cookie. Measured as an A/B.
 *
 * Consequences, each of them measured rather than reasoned:
 *
 *   - An ANCHOR mapping is taken at allocation time. It is what fixes the GPU VA, and it must
 *     outlive every mapping PanVK makes, because for a SAME_VA region munmap IS the free --
 *     MEM_FREE refuses it (EINVAL) in every form. Without the anchor, vkUnmapMemory would
 *     destroy the allocation.
 *   - A SAME_VA region can be mapped AGAIN at offset = its own address; the second mapping lands
 *     at a different CPU address and reads the same memory. That is what lets
 *     bo_get_mmap_offset keep working for PanVK on top of the anchor.
 *   - An EXEC region is the mirror image: MEM_FREE works on it, a second mmap does not (ENOMEM),
 *     and GPU_WR alongside GPU_EX is refused outright, so the zone is for shader code only.
 *
 * pan_kmod still wants a u32 handle to key its BO table on, so one is synthesised per device.
 */

static struct pan_kmod_bo *
kbase_kmod_bo_alloc(struct pan_kmod_dev *dev, struct pan_kmod_vm *exclusive_vm, uint64_t size,
                    uint32_t flags)
{
   struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);

   /* Nothing in the captured traffic shows how to ask for a GPU-uncached mapping, and guessing a
    * flag bit here would be silently wrong rather than loudly. Refuse, as panfrost_kmod does. */
   if (flags & PAN_KMOD_BO_FLAG_GPU_UNCACHED)
      return NULL;

   const bool exec = flags & PAN_KMOD_BO_FLAG_EXECUTABLE;
   const bool grow = !exec && (flags & PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT);

   struct kbase_kmod_bo *bo = pan_kmod_dev_alloc(dev, sizeof(*bo));
   if (!bo)
      return NULL;

   const uint64_t pages = DIV_ROUND_UP(size, 4096);
   union kbase_ioctl_mem_alloc req = {
      .in = {
         .va_pages = pages,
         .commit_pages = grow ? MIN2(pages, KBASE_GROW_INITIAL_PAGES) : pages,
         .extension = grow ? KBASE_GROW_STEP_PAGES : 0,
         .flags = exec ? KBASE_MEM_FLAGS_EXEC
                       : KBASE_MEM_FLAGS_RW | (grow ? BASE_MEM_GROW_ON_GPF : 0),
      },
   };

   if (pan_kmod_ioctl(dev->fd, KBASE_IOCTL_MEM_ALLOC, &req)) {
      mesa_loge("kbase: MEM_ALLOC of %" PRIu64 " pages failed (err=%d)", pages, errno);
      pan_kmod_dev_free(dev, bo);
      return NULL;
   }

   bo->exec = exec;
   bo->anchor = NULL;

   if (exec) {
      /* The exec zone hands back the address itself. Leave the mapping to whoever wants one:
       * this zone allows only a single mmap, so taking an anchor here would spend it. */
      bo->gpu_va = req.out.gpu_va;
   } else {
      /* req.out.gpu_va is a cookie. Spend it on the anchor, and the address that comes back is
       * the GPU VA. The anchor is never unmapped before bo_free, because unmapping it frees the
       * allocation. */
      void *anchor = mmap(NULL, pages * 4096, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
                          (off_t)req.out.gpu_va);
      if (anchor == MAP_FAILED) {
         mesa_loge("kbase: could not map the anchor for cookie 0x%" PRIx64 " (err=%d)",
                   (uint64_t)req.out.gpu_va, errno);
         pan_kmod_dev_free(dev, bo);
         return NULL;
      }
      bo->anchor = anchor;
      bo->gpu_va = (uint64_t)(uintptr_t)anchor;
   }

   pan_kmod_bo_init(&bo->base, dev, exclusive_vm, pages * 4096, flags,
                    kbase_dev->next_handle++);
   return &bo->base;
}

static void
kbase_kmod_bo_free(struct pan_kmod_bo *bo)
{
   struct kbase_kmod_bo *kbase_bo = container_of(bo, struct kbase_kmod_bo, base);

   if (kbase_bo->exec) {
      struct kbase_ioctl_mem_free req = {.gpu_addr = kbase_bo->gpu_va};
      if (pan_kmod_ioctl(bo->dev->fd, KBASE_IOCTL_MEM_FREE, &req))
         mesa_loge("kbase: MEM_FREE(0x%" PRIx64 ") failed (err=%d)", kbase_bo->gpu_va, errno);
   } else if (kbase_bo->anchor) {
      /* Dropping the anchor IS the free for a SAME_VA region. MEM_FREE refuses these. */
      if (munmap(kbase_bo->anchor, bo->size))
         mesa_loge("kbase: releasing the anchor at %p failed (err=%d)", kbase_bo->anchor, errno);

   }

   pan_kmod_dev_free(bo->dev, kbase_bo);
}

static off_t
kbase_kmod_bo_get_mmap_offset(struct pan_kmod_bo *bo)
{
   /* Both zones map at offset = the BO's GPU VA. For an exec BO that is the address the kernel
    * returned; for an ordinary one it is the anchor's address, and mapping it again yields a
    * second CPU view of the same memory. */
   return container_of(bo, struct kbase_kmod_bo, base)->gpu_va;
}

static bool
kbase_kmod_bo_wait(struct pan_kmod_bo *bo, int64_t timeout_ns, bool for_read_only_access)
{
   /* NOT IMPLEMENTED, and deliberately reported rather than faked as success. kbase expresses
    * completion through job atoms and its own fence fds, not through a per-BO wait, so this
    * needs the submission path before it can mean anything. Returning true here would turn a
    * missing sync into silent corruption. */
   mesa_loge("kbase: bo_wait is not implemented yet");
   return false;
}

static struct pan_kmod_bo *
kbase_kmod_bo_import_fd(struct pan_kmod_dev *dev, int fd, uint64_t size)
{
   struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);

   struct kbase_kmod_bo *bo = pan_kmod_dev_alloc(dev, sizeof(*bo));
   if (!bo)
      return NULL;

   /* CPU and GPU read/write, SAME_VA. Not COHERENT_LOCAL: that is a property of memory kbase
    * allocates, and an import is the exporter's. */
   int dmabuf = fd;
   union kbase_ioctl_mem_import req = {
      .in = {
         .flags = 0xfull | BASE_MEM_SAME_VA,
         .phandle = (uint64_t)(uintptr_t)&dmabuf,
         .type = BASE_MEM_IMPORT_TYPE_UMM,
      },
   };
   if (pan_kmod_ioctl(dev->fd, KBASE_IOCTL_MEM_IMPORT, &req)) {
      mesa_loge("kbase: MEM_IMPORT of dma-buf fd %d failed (err=%d)", fd, errno);
      pan_kmod_dev_free(dev, bo);
      return NULL;
   }

   const uint64_t pages = req.out.va_pages;
   if (!(req.out.flags & (BASE_MEM_SAME_VA | BASE_MEM_NEED_MMAP)) || pages * 4096 < size) {
      mesa_loge("kbase: MEM_IMPORT returned flags 0x%" PRIx64 ", %" PRIu64 " pages for %" PRIu64
                " bytes; only a SAME_VA import is handled",
                (uint64_t)req.out.flags, pages, size);
      pan_kmod_dev_free(dev, bo);
      return NULL;
   }

   void *anchor = mmap(NULL, pages * 4096, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
                       (off_t)req.out.gpu_va);
   if (anchor == MAP_FAILED) {
      mesa_loge("kbase: could not map the imported cookie 0x%" PRIx64 " (err=%d)",
                (uint64_t)req.out.gpu_va, errno);
      pan_kmod_dev_free(dev, bo);
      return NULL;
   }

   bo->exec = false;
   bo->anchor = anchor;
   bo->gpu_va = (uint64_t)(uintptr_t)anchor;
   pan_kmod_bo_init(&bo->base, dev, NULL, pages * 4096, PAN_KMOD_BO_FLAG_IMPORTED,
                    kbase_dev->next_handle++);
   return &bo->base;
}

static struct pan_kmod_bo *
kbase_kmod_bo_import(struct pan_kmod_dev *dev, uint32_t handle, uint64_t size)
{
   /* kbase imports a dma-buf FD through MEM_IMPORT; it has no GEM handle namespace for
    * pan_kmod's u32 handle to refer to. Wiring this needs the dma-buf path, not a translation. */
   mesa_loge("kbase: bo_import is not implemented yet");
   return NULL;
}

/* ---- address space -------------------------------------------------------------------------
 *
 * kbase gives a context exactly one GPU address space and assigns every VA itself at allocation
 * time, so there is nothing to create and nothing to bind. That is not a compromise forced by
 * this backend: DRM panfrost behaves the same way, and panfrost_kmod likewise refuses anything
 * but PAN_KMOD_VM_FLAG_AUTO_VA, one VM per device, and whole-BO mappings. */

static struct pan_kmod_vm *
kbase_kmod_vm_create(struct pan_kmod_dev *dev, uint32_t flags, uint64_t va_start,
                     uint64_t va_range)
{
   struct kbase_kmod_dev *kbase_dev = container_of(dev, struct kbase_kmod_dev, base);

   if (kbase_dev->vm) {
      mesa_loge("kbase: only one VM per device is supported");
      return NULL;
   }

   if (!(flags & PAN_KMOD_VM_FLAG_AUTO_VA)) {
      mesa_loge("kbase: only PAN_KMOD_VM_FLAG_AUTO_VA is supported -- the kernel owns the VA");
      return NULL;
   }

   struct kbase_kmod_vm *vm = pan_kmod_dev_alloc(dev, sizeof(*vm));
   if (!vm)
      return NULL;

   pan_kmod_vm_init(&vm->base, dev, 0, flags);
   kbase_dev->vm = vm;
   return &vm->base;
}

static void
kbase_kmod_vm_destroy(struct pan_kmod_vm *vm)
{
   struct kbase_kmod_dev *kbase_dev = container_of(vm->dev, struct kbase_kmod_dev, base);
   struct kbase_kmod_vm *kbase_vm = container_of(vm, struct kbase_kmod_vm, base);

   kbase_dev->vm = NULL;
   pan_kmod_vm_cleanup(vm);
   pan_kmod_dev_free(vm->dev, kbase_vm);
}

static int
kbase_kmod_vm_bind(struct pan_kmod_vm *vm, enum pan_kmod_vm_op_mode mode,
                   struct pan_kmod_vm_op *ops, uint32_t op_count)
{
   if (mode != PAN_KMOD_VM_OP_MODE_IMMEDIATE &&
       mode != PAN_KMOD_VM_OP_MODE_DEFER_TO_NEXT_IDLE_POINT) {
      mesa_loge("kbase: unsupported vm_bind mode %d", mode);
      return -1;
   }

   for (uint32_t i = 0; i < op_count; i++) {
      if (ops[i].type == PAN_KMOD_VM_OP_TYPE_MAP) {
         if (ops[i].va.start != PAN_KMOD_VM_MAP_AUTO_VA) {
            mesa_loge("kbase: only auto-VA mapping is possible -- the kernel picks the address");
            return -1;
         }
         /* The mapping already exists: MEM_ALLOC created it and returned its VA. Report that
          * address back so the caller learns where the kernel put it. */
         struct kbase_kmod_bo *bo = container_of(ops[i].map.bo, struct kbase_kmod_bo, base);
         ops[i].va.start = bo->gpu_va;
      } else if (ops[i].type != PAN_KMOD_VM_OP_TYPE_UNMAP) {
         mesa_loge("kbase: unsupported vm_bind op %d", ops[i].type);
         return -1;
      }
      /* UNMAP is a no-op: the mapping goes away with the allocation, in bo_free. */
   }

   return 0;
}

/* ---- sync objects ----------------------------------------------------------------------------
 *
 * kbase has no syncobjs, and the DRM syncobj ioctls panvk used to issue on the kbase fd only
 * worked on the Exynos kernel because it returns success for DRM ioctls it does not implement
 * (see "THE TRAP" below): every wait "succeeded" at once and nothing was ever signaled. A
 * MediaTek kbase (Galaxy A31, Mali-G52) rejects them, so panvk found no device there at all.
 *
 * So the objects live in this process: binary, in one process-wide table indexed by handle, with
 * waits on a condition variable (CLOCK_MONOTONIC, DRM's absolute timeouts and -ETIME). A GPU
 * signal is done by the kmod's event thread when the submission that carries it completes
 * (pan_kmod_kbase_submit_async); until then the object is SUBMITTED, which is what a DRM
 * syncobj holding an unsignaled fence is, and satisfies WAIT_AVAILABLE. A submitted object can
 * hold a sync file from a kbase fence-trigger atom that signals with the GPU work: that is what
 * an export hands to the compositor, so presenting never waits for the GPU. */

enum ks_state {
   KS_FREE = 0,
   KS_UNSIGNALED,
   KS_SIGNALED,
   KS_SUBMITTED,
};

struct ks_obj {
   uint8_t state;
   int fd; /* KS_SUBMITTED: a sync file signaling with it, or -1 */
};

static pthread_mutex_t ks_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ks_cond;
static pthread_once_t ks_once = PTHREAD_ONCE_INIT;
static struct ks_obj *ks_objs;
static uint32_t ks_cap, ks_hint;

/* The runtime (vk_drm_syncobj.c) tells a timeout from an error by errno == ETIME, the way
 * libdrm's wrappers leave it: every failure here sets errno too. Without it, a fence that was
 * merely not signaled yet read as a lost device -- which never happened while every submission
 * blocked, and killed Minecraft once they stopped blocking. */
static int
ks_ret(int ret)
{
   if (ret < 0)
      errno = -ret;
   return ret;
}

static void
ks_init(void)
{
   pthread_condattr_t a;
   pthread_condattr_init(&a);
   pthread_condattr_setclock(&a, CLOCK_MONOTONIC);
   pthread_cond_init(&ks_cond, &a);
   pthread_condattr_destroy(&a);
}

static struct ks_obj *
ks_get(uint32_t h)
{
   return h && h <= ks_cap && ks_objs[h - 1].state != KS_FREE ? &ks_objs[h - 1] : NULL;
}

static void
ks_set_state(struct ks_obj *o, uint8_t state, int fd)
{
   if (o->fd >= 0)
      close(o->fd);
   o->state = state;
   o->fd = fd;
}

static int
ks_create(struct util_sync_provider *p, uint32_t flags, uint32_t *handle)
{
   pthread_mutex_lock(&ks_lock);
   uint32_t i = ks_hint;
   while (i < ks_cap && ks_objs[i].state != KS_FREE)
      i++;
   if (i >= ks_cap) {
      for (i = 0; i < ks_cap && ks_objs[i].state != KS_FREE; i++)
         ;
   }
   if (i >= ks_cap) {
      const uint32_t cap = ks_cap ? ks_cap * 2 : 256;
      struct ks_obj *o = realloc(ks_objs, cap * sizeof(*o));
      if (!o) {
         pthread_mutex_unlock(&ks_lock);
         return ks_ret(-ENOMEM);
      }
      for (uint32_t k = ks_cap; k < cap; k++)
         o[k] = (struct ks_obj){KS_FREE, -1};
      i = ks_cap;
      ks_objs = o;
      ks_cap = cap;
   }
   ks_objs[i] = (struct ks_obj){
      (flags & DRM_SYNCOBJ_CREATE_SIGNALED) ? KS_SIGNALED : KS_UNSIGNALED, -1};
   ks_hint = i + 1;
   *handle = i + 1;
   pthread_mutex_unlock(&ks_lock);
   return 0;
}

static int
ks_destroy(struct util_sync_provider *p, uint32_t handle)
{
   pthread_mutex_lock(&ks_lock);
   struct ks_obj *o = ks_get(handle);
   if (o) {
      ks_set_state(o, KS_FREE, -1);
      ks_hint = MIN2(ks_hint, handle - 1);
   }
   pthread_mutex_unlock(&ks_lock);
   return ks_ret(o ? 0 : -EINVAL);
}

static int
ks_set(const uint32_t *handles, uint32_t count, uint8_t state)
{
   int ret = 0;
   pthread_mutex_lock(&ks_lock);
   for (uint32_t i = 0; i < count; i++) {
      struct ks_obj *o = ks_get(handles[i]);
      if (o)
         ks_set_state(o, state, -1);
      else
         ret = -EINVAL;
   }
   pthread_cond_broadcast(&ks_cond);
   pthread_mutex_unlock(&ks_lock);
   return ret;
}

static int
ks_signal(struct util_sync_provider *p, const uint32_t *handles, uint32_t count)
{
   return ks_ret(ks_set(handles, count, KS_SIGNALED));
}

static int
ks_reset(struct util_sync_provider *p, const uint32_t *handles, uint32_t count)
{
   return ks_ret(ks_set(handles, count, KS_UNSIGNALED));
}

void
pan_kmod_kbase_sync_submitted(const uint32_t *handles, unsigned count)
{
   pthread_once(&ks_once, ks_init);
   ks_set(handles, count, KS_SUBMITTED);
}

void
pan_kmod_kbase_sync_attach_fd(const uint32_t *handles, unsigned count, int fd)
{
   if (fd < 0)
      return;
   pthread_mutex_lock(&ks_lock);
   for (uint32_t i = 0; i < count; i++) {
      struct ks_obj *o = ks_get(handles[i]);
      /* Only if still submitted: the event thread may have signaled it already. */
      if (o && o->state == KS_SUBMITTED && o->fd < 0)
         o->fd = dup(fd);
   }
   pthread_mutex_unlock(&ks_lock);
}

static uint64_t
ks_now(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* Called with ks_lock held. WAIT_AVAILABLE is satisfied by a submitted object, everything else
 * waits for it to be signaled; WAIT_FOR_SUBMIT needs nothing extra, an unsubmitted object just
 * keeps the wait going. */
static int
ks_wait_locked(uint32_t *handles, unsigned n, int64_t timeout_nsec, unsigned flags,
               uint32_t *first_signaled)
{
   const bool all = flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;
   const bool available = flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE;

   for (;;) {
      unsigned done = 0, first = UINT32_MAX;
      for (unsigned i = 0; i < n; i++) {
         const struct ks_obj *o = ks_get(handles[i]);
         if (!o)
            return -EINVAL;
         if (o->state == KS_SIGNALED || (available && o->state == KS_SUBMITTED)) {
            done++;
            first = MIN2(first, i);
         }
      }
      if (all ? done == n : done > 0) {
         if (first_signaled)
            *first_signaled = first == UINT32_MAX ? 0 : first;
         return 0;
      }
      if (timeout_nsec != INT64_MAX && (uint64_t)timeout_nsec <= ks_now())
         return -ETIME;
      if (timeout_nsec == INT64_MAX) {
         pthread_cond_wait(&ks_cond, &ks_lock);
      } else {
         const struct timespec ts = {.tv_sec = timeout_nsec / 1000000000ll,
                                     .tv_nsec = timeout_nsec % 1000000000ll};
         pthread_cond_timedwait(&ks_cond, &ks_lock, &ts);
      }
   }
}

static int
ks_wait(struct util_sync_provider *p, uint32_t *handles, unsigned n, int64_t timeout_nsec,
        unsigned flags, uint32_t *first_signaled)
{
   pthread_mutex_lock(&ks_lock);
   const int ret = ks_wait_locked(handles, n, timeout_nsec, flags, first_signaled);
   pthread_mutex_unlock(&ks_lock);
   return ks_ret(ret);
}

/* Binary objects only (panvk drops the timeline feature before v10), but the runtime takes this
 * path for every VK_SYNC_WAIT_PENDING wait because only it carries WAIT_AVAILABLE. */
static int
ks_timeline_wait(struct util_sync_provider *p, uint32_t *handles, uint64_t *points,
                 unsigned n, int64_t timeout_nsec, unsigned flags, uint32_t *first_signaled)
{
   return ks_wait(p, handles, n, timeout_nsec, flags, first_signaled);
}

static int
ks_query(struct util_sync_provider *p, uint32_t *handles, uint64_t *points, uint32_t count,
         uint32_t flags)
{
   return ks_ret(-ENOSYS);
}

static int
ks_transfer(struct util_sync_provider *p, uint32_t dst, uint64_t dst_point, uint32_t src,
            uint64_t src_point, uint32_t flags)
{
   pthread_mutex_lock(&ks_lock);
   struct ks_obj *d = ks_get(dst), *s = ks_get(src);
   if (d && s && d != s) {
      ks_set_state(d, s->state, s->fd >= 0 ? dup(s->fd) : -1);
      pthread_cond_broadcast(&ks_cond);
   }
   pthread_mutex_unlock(&ks_lock);
   return ks_ret(d && s ? 0 : -EINVAL);
}

static int
ks_export_sync_file(struct util_sync_provider *p, uint32_t handle, int *out_fd)
{
   const uint64_t deadline = ks_now() + 10ull * 1000000000ull;
   pthread_mutex_lock(&ks_lock);
   /* Wait for the signal operation to be submitted (normally it is already). */
   int ret = ks_wait_locked(&handle, 1, deadline, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE, NULL);
   struct ks_obj *o = ret ? NULL : ks_get(handle);
   if (o && o->state == KS_SUBMITTED) {
      if (o->fd >= 0) {
         /* The GPU is still at it: hand out the fence-trigger sync file. */
         *out_fd = dup(o->fd);
         pthread_mutex_unlock(&ks_lock);
         return *out_fd >= 0 ? 0 : ks_ret(-errno);
      }
      /* Submitted without a sync file: all that is left is to wait for the GPU. */
      ret = ks_wait_locked(&handle, 1, deadline, 0, NULL);
   }
   pthread_mutex_unlock(&ks_lock);
   if (ret) {
      mesa_loge("kbase: sync file export of an object never signaled (%d)", ret);
      return ks_ret(ret);
   }
   *out_fd = -1;
   return 0;
}

static int
ks_import_sync_file(struct util_sync_provider *p, uint32_t handle, int fd)
{
   if (fd >= 0) {
      struct pollfd pfd = {.fd = fd, .events = POLLIN};
      if (poll(&pfd, 1, 10000) <= 0)
         mesa_loge("kbase: imported sync file did not signal within 10 s");
   }
   return ks_signal(p, &handle, 1);
}

static int
ks_no_fd(struct util_sync_provider *p, uint32_t handle, int *out_fd)
{
   return ks_ret(-ENOSYS);
}

static int
ks_no_handle(struct util_sync_provider *p, int fd, uint32_t *handle)
{
   return ks_ret(-ENOSYS);
}

static void
ks_finalize(struct util_sync_provider *p)
{
   free(p);
}

static struct util_sync_provider *
ks_clone(struct util_sync_provider *p)
{
   return pan_kmod_kbase_sync_provider();
}

struct util_sync_provider *
pan_kmod_kbase_sync_provider(void)
{
   pthread_once(&ks_once, ks_init);

   struct util_sync_provider *p = calloc(1, sizeof(*p));
   if (!p)
      return NULL;

   *p = (struct util_sync_provider){
      .create = ks_create,
      .destroy = ks_destroy,
      .handle_to_fd = ks_no_fd,
      .fd_to_handle = ks_no_handle,
      .import_sync_file = ks_import_sync_file,
      .export_sync_file = ks_export_sync_file,
      .wait = ks_wait,
      .timeline_wait = ks_timeline_wait,
      .query = ks_query,
      .reset = ks_reset,
      .signal = ks_signal,
      .transfer = ks_transfer,
      .finalize = ks_finalize,
      .clone = ks_clone,
   };
   return p;
}

/* ---- job submission -------------------------------------------------------------------------
 *
 * THE TRAP THAT MAKES THIS SECTION NECESSARY. This kernel returns SUCCESS for DRM ioctls it
 * does not implement. Measured, on our own driver, with the spy attached:
 *
 *    [IOCTL] fd=7 NON-KBASE type=0x64 nr=64  size=48 -> 0     DRM_IOCTL_PANFROST_SUBMIT
 *    [IOCTL] fd=7 NON-KBASE type=0x64 nr=191 size=8  -> 0     DRM_IOCTL_SYNCOBJ_CREATE
 *    [IOCTL] fd=7 NON-KBASE type=0x64 nr=195 size=40 -> 0     DRM_IOCTL_SYNCOBJ_WAIT
 *
 * So the unported JM path did not fail -- it reported success, every syncobj wait returned
 * immediately, and vkQueueSubmit and vkQueueWaitIdle both said rc=0 over a GPU that had been
 * asked for nothing at all. The only reason it was not mistaken for a working driver is that
 * the readback was verified. Nothing here may rely on an ioctl return alone.
 *
 * WHAT SUBMISSION LOOKS LIKE, read off the vendor rendering a triangle (probe/mali/glprobe2.c):
 * one JOB_SUBMIT carrying an array of 64-byte atoms, and completion reported by READING the
 * device fd -- 24-byte records, not a fence. The atom layout below is pinned by an unusually
 * strong check: the two udata words we put in an atom came back BYTE-IDENTICAL in that atom's
 * completion event, which fixes both structures at once against the same evidence.
 */

struct kbase_ioctl_job_submit {
   uint64_t addr;
   uint32_t nr_atoms;
   uint32_t stride;
};

struct kbase_jd_dependency {
   uint8_t atom_id;
   uint8_t dependency_type;
};

/* 64 bytes, which is exactly the stride the vendor passes. Every offset below was read out of a
 * captured atom, not out of a header. */
struct kbase_jd_atom {
   uint64_t seq_nr;
   uint64_t jc;
   uint64_t udata[2];
   uint64_t extres_list;
   uint16_t nr_extres;
   uint8_t jit_id[2];
   struct kbase_jd_dependency pre_dep[2];
   uint8_t atom_number;
   int8_t prio;
   uint8_t device_nr;
   uint8_t jobslot;
   uint32_t core_req;
   uint8_t renderpass_id;
   uint8_t padding[7];
};
static_assert(sizeof(struct kbase_jd_atom) == 64, "the captured stride is 64");

/* 24 bytes, and the read that returned two of them for a two-atom frame returned exactly 48. */
struct kbase_jd_event {
   uint32_t event_code;
   uint8_t atom_number;
   uint8_t padding[3];
   uint64_t udata[2];
};
static_assert(sizeof(struct kbase_jd_event) == 24, "two events came back as 48 bytes");

#define KBASE_IOCTL_JOB_SUBMIT _IOW(KBASE_IOCTL_TYPE, 2, struct kbase_ioctl_job_submit)

/* The only event code seen, on every atom that completed correctly. Anything else is reported
 * rather than interpreted: a table of fault codes we have not observed would be a guess, and the
 * number itself is what a reader needs. */
#define KBASE_JD_EVENT_DONE 1

bool
pan_kmod_dev_is_kbase(const struct pan_kmod_dev *dev)
{
   return dev->ops == &kbase_kmod_ops;
}

/* ---- asynchronous submission ------------------------------------------------------------
 *
 * Submitting used to mean waiting: the caller read the device fd until every atom it had sent
 * reported. So the app's thread sat idle while the GPU drew each frame, and the next frame's
 * recording could only start after it -- measured in Minecraft on the S10e, about 13 ms of CPU
 * and 30 ms of GPU per frame, one after the other.
 *
 * Now one thread per device reads the events, and a submission returns once the kernel has its
 * atoms. Atom numbers (1..255, unique among the atoms in flight) come from one allocator. Each
 * submission runs after everything submitted before it: its first atom waits (ORDER, so a fault
 * does not spread) on the newest atom still in flight. When a completion callback is given, it
 * runs on the event thread once every atom of the submission has reported, before
 * pan_kmod_kbase_wait_seq sees it complete.
 *
 * A fence-trigger soft atom (BASE_JD_REQ_SOFT_FENCE_TRIGGER, mali_kbase_softjobs.c) waiting on
 * the submission's last atom gives a sync file at submit time that signals with the work. The
 * kernel runs soft atoms from a worker, measurably later than the GPU atom they wait on, so the
 * fence atom is kept out of everything else: the submission completes, and the next one starts,
 * on its last GPU atom. It is tracked on its own, only to free its number. */

#define KBASE_JD_DEP_ORDER              2u
#define KBASE_JD_REQ_SOFT_FENCE_TRIGGER 0x202u /* BASE_JD_REQ_SOFT_JOB | 0x2 */
#define KBASE_ATOM_TAG                  0x50414e564bull /* "PANVK", comes back in the event */

struct kbase_sub {
   uint64_t seq;
   unsigned pending;
   bool ok;
   pan_kmod_kbase_done_cb done;
   void *data;
};

static struct kbase_queue *
kbase_queue(struct pan_kmod_dev *dev)
{
   return &container_of(dev, struct kbase_kmod_dev, base)->queue;
}

static void *
kbase_event_thread(void *arg)
{
   struct kbase_kmod_dev *kdev = arg;
   struct kbase_queue *q = &kdev->queue;

   for (;;) {
      pthread_mutex_lock(&q->lock);
      const bool stop = q->stop && !q->in_flight;
      pthread_mutex_unlock(&q->lock);
      if (stop)
         break;

      struct pollfd pfd = {.fd = kdev->base.fd, .events = POLLIN};
      if (poll(&pfd, 1, 100) <= 0)
         continue;

      struct kbase_jd_event ev[16];
      const ssize_t n = read(kdev->base.fd, ev, sizeof(ev));
      if (n <= 0 || n % sizeof(struct kbase_jd_event)) {
         if (n < 0 && (errno == EAGAIN || errno == EINTR))
            continue;
         mesa_loge("kbase: event read returned %zd, which is not a whole number of records", n);
         continue;
      }

      struct kbase_sub *finished[16];
      unsigned nr_finished = 0;

      pthread_mutex_lock(&q->lock);
      for (unsigned i = 0; i < n / sizeof(struct kbase_jd_event); i++) {
         if (ev[i].udata[0] != KBASE_ATOM_TAG)
            continue; /* not one of ours */
         const uint8_t an = ev[i].atom_number;
         struct kbase_sub *sub = q->owner[an];
         if (!sub) {
            mesa_loge("kbase: event for atom %u, which is not in flight", an);
            continue;
         }
         q->owner[an] = NULL;
         q->free_atoms++;
         if (q->last_atom == an)
            q->last_atom = 0;
         if (ev[i].event_code != KBASE_JD_EVENT_DONE) {
            if (q->is_fence[an] && !sub->done) {
               if (!q->fence_warned) {
                  mesa_loge("kbase: fence-trigger atom finished with event code %u; presents "
                            "will wait for the GPU", ev[i].event_code);
                  q->fence_warned = true;
               }
            } else {
               mesa_loge("kbase: atom %u finished with event code %u, not DONE", an,
                         ev[i].event_code);
               sub->ok = false;
            }
         }
         q->is_fence[an] = false;
         if (--sub->pending == 0)
            finished[nr_finished++] = sub;
      }
      pthread_cond_broadcast(&q->cond);
      pthread_mutex_unlock(&q->lock);

      for (unsigned i = 0; i < nr_finished; i++) {
         struct kbase_sub *sub = finished[i];
         if (sub->done)
            sub->done(sub->data, sub->ok);
         pthread_mutex_lock(&q->lock);
         /* seq 0: a fence atom of its own, not a submission. */
         for (unsigned k = 0; sub->seq && k < q->nr_open_seq; k++) {
            if (q->open_seq[k] == sub->seq) {
               q->open_seq[k] = q->open_seq[--q->nr_open_seq];
               break;
            }
         }
         q->in_flight--;
         pthread_cond_broadcast(&q->cond);
         pthread_mutex_unlock(&q->lock);
         free(sub);
      }
   }
   return NULL;
}

static void
kbase_queue_init(struct kbase_queue *q)
{
   pthread_mutex_init(&q->lock, NULL);
   /* CLOCK_MONOTONIC, for pan_kmod_kbase_wait_seq_timeout. */
   pthread_condattr_t attr;
   pthread_condattr_init(&attr);
   pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
   pthread_cond_init(&q->cond, &attr);
   pthread_condattr_destroy(&attr);
   q->free_atoms = 255;
   q->next_seq = 1;
}

static void
kbase_queue_finish(struct kbase_kmod_dev *kdev)
{
   struct kbase_queue *q = &kdev->queue;

   pthread_mutex_lock(&q->lock);
   q->stop = true;
   const bool started = q->thread_started;
   pthread_mutex_unlock(&q->lock);
   if (started)
      pthread_join(q->thread, NULL);
   pthread_cond_destroy(&q->cond);
   pthread_mutex_destroy(&q->lock);
}

uint64_t
pan_kmod_kbase_submit_async(struct pan_kmod_dev *dev, const struct pan_kmod_kbase_atom *atoms,
                            unsigned nr_atoms, int *fence_fd, pan_kmod_kbase_done_cb done,
                            void *data)
{
   struct kbase_kmod_dev *kdev = container_of(dev, struct kbase_kmod_dev, base);
   struct kbase_queue *q = &kdev->queue;

   if (fence_fd)
      *fence_fd = -1;

   if (nr_atoms > PAN_KMOD_KBASE_MAX_ATOMS) {
      mesa_loge("kbase: %u atoms in one submit is more than this path handles", nr_atoms);
      return 0;
   }

   pthread_mutex_lock(&q->lock);

   if (!q->thread_started) {
      if (pthread_create(&q->thread, NULL, kbase_event_thread, kdev)) {
         pthread_mutex_unlock(&q->lock);
         mesa_loge("kbase: could not start the event thread");
         return 0;
      }
      q->thread_started = true;
   }

   const uint64_t seq = q->next_seq++;
   const bool fence = fence_fd && (nr_atoms || q->last_atom);
   const unsigned need = nr_atoms + fence;

   if (!need) {
      /* Nothing to run and nothing on the GPU: complete once earlier submissions have run their
       * callbacks, so completion stays in order. */
      while (q->nr_open_seq)
         pthread_cond_wait(&q->cond, &q->lock);
      pthread_mutex_unlock(&q->lock);
      if (done)
         done(data, true);
      return seq;
   }

   while (q->free_atoms < need)
      pthread_cond_wait(&q->cond, &q->lock);

   /* With GPU atoms, the fence atom gets a record of its own (seq 0, no callback); a fence
    * alone is the submission and completes with it. */
   const bool split = fence && nr_atoms;
   struct kbase_sub *sub = calloc(1, sizeof(*sub));
   struct kbase_sub *fsub = split ? calloc(1, sizeof(*fsub)) : NULL;
   if (!sub || (split && !fsub)) {
      free(sub);
      free(fsub);
      pthread_mutex_unlock(&q->lock);
      return 0;
   }
   *sub = (struct kbase_sub){.seq = seq, .pending = split ? nr_atoms : need, .ok = true,
                             .done = done, .data = data};
   if (split)
      *fsub = (struct kbase_sub){.pending = 1, .ok = true};

   uint8_t real[PAN_KMOD_KBASE_MAX_ATOMS + 1];
   uint8_t rel_to_real[256] = {0};
   unsigned next = 1;
   for (unsigned i = 0; i < need; i++) {
      while (q->owner[next])
         next++;
      assert(next <= 255);
      real[i] = next;
      q->owner[next] = split && i == nr_atoms ? fsub : sub;
      q->is_fence[next] = fence && i == nr_atoms;
      next++;
   }
   for (unsigned i = 0; i < nr_atoms; i++)
      rel_to_real[atoms[i].atom_number] = real[i];

   struct kbase_jd_atom jd[PAN_KMOD_KBASE_MAX_ATOMS + 1];
   memset(jd, 0, need * sizeof(jd[0]));
   for (unsigned i = 0; i < nr_atoms; i++) {
      assert(atoms[i].atom_number != 0);
      jd[i].jc = atoms[i].jc;
      jd[i].core_req = atoms[i].core_req;
      jd[i].atom_number = real[i];
      /* udata is opaque to the kernel and comes back verbatim in the completion event: the tag
       * tells our events from anyone else's on the context. */
      jd[i].udata[0] = KBASE_ATOM_TAG;
      jd[i].udata[1] = seq;
      if (atoms[i].dep_atom) {
         jd[i].pre_dep[0].atom_id = rel_to_real[atoms[i].dep_atom];
         jd[i].pre_dep[0].dependency_type = KBASE_JD_DEP_DATA;
      } else if (q->last_atom) {
         jd[i].pre_dep[0].atom_id = q->last_atom;
         jd[i].pre_dep[0].dependency_type = KBASE_JD_DEP_ORDER;
      }
   }

   /* The kernel reads the struct at submit time and writes the sync file's fd back into it. */
   struct {
      int fd;
      int stream_fd;
   } base_fence = {-1, -1};
   if (fence) {
      struct kbase_jd_atom *f = &jd[nr_atoms];
      f->jc = (uintptr_t)&base_fence;
      f->core_req = KBASE_JD_REQ_SOFT_FENCE_TRIGGER;
      f->atom_number = real[nr_atoms];
      f->udata[0] = KBASE_ATOM_TAG;
      f->udata[1] = seq;
      f->pre_dep[0].atom_id = nr_atoms ? real[nr_atoms - 1] : q->last_atom;
      f->pre_dep[0].dependency_type = KBASE_JD_DEP_ORDER;
   }

   struct kbase_ioctl_job_submit req = {
      .addr = (uintptr_t)jd,
      .nr_atoms = need,
      .stride = sizeof(struct kbase_jd_atom),
   };

   if (pan_kmod_ioctl(dev->fd, KBASE_IOCTL_JOB_SUBMIT, &req)) {
      mesa_loge("kbase: JOB_SUBMIT of %u atoms failed (err=%d)", need, errno);
      for (unsigned i = 0; i < need; i++) {
         q->owner[real[i]] = NULL;
         q->is_fence[real[i]] = false;
      }
      pthread_mutex_unlock(&q->lock);
      free(sub);
      free(fsub);
      return 0;
   }

   q->free_atoms -= need;
   /* What the next submission waits on: the last GPU atom, not a fence atom. */
   /* A fence alone leaves it on the GPU atom the fence waits for. */
   if (nr_atoms)
      q->last_atom = real[nr_atoms - 1];
   q->in_flight += split ? 2 : 1;
   q->open_seq[q->nr_open_seq++] = seq;
   pthread_mutex_unlock(&q->lock);

   if (fence_fd)
      *fence_fd = base_fence.fd;
   return seq;
}

void
pan_kmod_kbase_wait_seq(struct pan_kmod_dev *dev, uint64_t seq)
{
   struct kbase_queue *q = kbase_queue(dev);

   pthread_mutex_lock(&q->lock);
   if (!seq)
      seq = q->next_seq - 1;
   for (;;) {
      bool open = false;
      for (unsigned k = 0; k < q->nr_open_seq && !open; k++)
         open = q->open_seq[k] <= seq;
      if (!open)
         break;
      pthread_cond_wait(&q->cond, &q->lock);
   }
   pthread_mutex_unlock(&q->lock);
}

bool
pan_kmod_kbase_wait_seq_timeout(struct pan_kmod_dev *dev, uint64_t seq, int64_t abs_timeout_ns)
{
   struct kbase_queue *q = kbase_queue(dev);

   if (!seq)
      return true;

   bool done = false;
   pthread_mutex_lock(&q->lock);
   for (;;) {
      bool open = false;
      for (unsigned k = 0; k < q->nr_open_seq && !open; k++)
         open = q->open_seq[k] <= seq;
      if (!open) {
         done = true;
         break;
      }
      if (abs_timeout_ns == INT64_MAX) {
         pthread_cond_wait(&q->cond, &q->lock);
         continue;
      }
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      const int64_t now_ns = (int64_t)now.tv_sec * 1000000000ll + now.tv_nsec;
      if (now_ns >= abs_timeout_ns)
         break;
      const struct timespec until = {.tv_sec = abs_timeout_ns / 1000000000ll,
                                     .tv_nsec = abs_timeout_ns % 1000000000ll};
      pthread_cond_timedwait(&q->cond, &q->lock, &until);
   }
   pthread_mutex_unlock(&q->lock);
   return done;
}

static void
kbase_blocking_done(void *data, bool ok)
{
   *(bool *)data = ok;
}

bool
pan_kmod_kbase_submit(struct pan_kmod_dev *dev, const struct pan_kmod_kbase_atom *atoms,
                      unsigned nr_atoms)
{
   bool ok = true;
   const uint64_t seq =
      pan_kmod_kbase_submit_async(dev, atoms, nr_atoms, NULL, kbase_blocking_done, &ok);
   if (!seq)
      return false;
   pan_kmod_kbase_wait_seq(dev, seq);
   return ok;
}

static uint64_t
kbase_kmod_query_timestamp(const struct pan_kmod_dev *dev)
{
#if defined(__aarch64__)
   uint64_t v;
   __asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v) : : "memory");
   return v;
#else
   return 0;
#endif
}

const struct pan_kmod_ops kbase_kmod_ops = {
   .dev_create = kbase_kmod_dev_create,
   .dev_destroy = kbase_kmod_dev_destroy,
   .dev_query_user_va_range = kbase_kmod_dev_query_user_va_range,
   .bo_alloc = kbase_kmod_bo_alloc,
   .bo_free = kbase_kmod_bo_free,
   .bo_import = kbase_kmod_bo_import,
   .bo_import_fd = kbase_kmod_bo_import_fd,
   .bo_get_mmap_offset = kbase_kmod_bo_get_mmap_offset,
   .bo_wait = kbase_kmod_bo_wait,
   .vm_create = kbase_kmod_vm_create,
   .vm_destroy = kbase_kmod_vm_destroy,
   .vm_bind = kbase_kmod_vm_bind,
   .query_timestamp = kbase_kmod_query_timestamp,
};
