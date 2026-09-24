# Mesa unified drivers

Mesa drivers for Android phones that the stock GPU drivers leave behind, in one source tree based
on Mesa 26.3.0-devel:

| Driver | API | GPUs | Package |
|---|---|---|---|
| RADV | Vulkan 1.4 | Samsung Xclipse 920 (Exynos 2200), Xclipse 530 (Exynos 1480) | `radv-xclipse-*.zip` |
| RadeonSI | OpenGL 4.6, OpenGL ES 3.2 | Samsung Xclipse 920, Xclipse 530 | `radeonsi-xclipse-*.zip` |
| PanVK | Vulkan 1.3 | Arm Mali Bifrost on the stock kbase kernel driver | `mali-panvk-*.zip` |
| Panfrost | OpenGL 4.6 (4.5 on Mali-G71/G72), OpenGL ES 3.2 | Arm Mali Bifrost on the stock kbase kernel driver | `mali-panfrost-*.zip` |

The Mali drivers run on the kernel driver the phone already has (Arm's kbase, `/dev/mali0`), so
they need no root and no custom kernel. PanVK is tested on the Mali-G76 (Galaxy S10e, Exynos
9820), the Mali-G72 (Galaxy Tab S6 Lite, Exynos 9611) and the Mali-G52 (Galaxy A31, MediaTek Helio
P65), Panfrost on the Mali-G76 and the Mali-G72. Other Xclipse models are not compatible for now.

## What works on Mali

- Vulkan 1.3 as PanVK exposes it on Bifrost, and desktop OpenGL 4.6 core through Panfrost (4.5 on
  the Mali-G71 and G72, which cannot run subgroup votes), both through the kbase job interface.
- Geometry shaders, tessellation and transform feedback, which Bifrost does not have in hardware,
  emulated with compute jobs in both drivers. In PanVK, indirect, indexed, instanced and
  multi-draw calls work with all of them.
- OpenGL: compute shaders with up to 1024 invocations per workgroup (Bifrost runs 256, so each
  runs the work of several), 64-bit floats in software, viewport arrays, cull distances, query
  buffer objects and indirect draw parameters.
- Presentation through Android's gralloc (Arm's handle layout, including MediaTek's).
- Minecraft Java through its Vulkan renderer (MojoLauncher), which the stock Mali driver cannot run
  because it only exposes Vulkan 1.1.

Not done yet: BCn texture formats on Mali, and in Panfrost a geometry shader together with
tessellation, and transform feedback from either.

On phones with 4 GB of RAM or less, give Minecraft about 600 MB of Java heap: GPU memory comes out
of the same RAM, and with a 1 GB heap the system kills the game while it loads a world.

## Requirements

Tools:

| Tool | Version | Needed for |
|---|---|---|
| Android NDK | r29 tested | all: the API 34 (Xclipse) or API 31 (Mali) aarch64 compiler and `llvm-strip` |
| Meson | 1.4.0 or newer | all |
| Ninja | any recent | all |
| Python | 3.10 or newer | all: with `mako` (0.8.0 or newer) and `packaging`, and `pyyaml` for Mali |
| glslangValidator | 12.2 or newer | RADV, PanVK |
| flex and bison | any recent | RadeonSI, Panfrost (on Windows, `win_flex` and `win_bison` work too) |
| Linux or WSL | | PanVK, Panfrost: their host tools need LLVM |
| pkg-config | any recent | PanVK, Panfrost |
| LLVM, clang, libclc, SPIR-V LLVM translator | one major version, 21 tested | PanVK, Panfrost |

On Ubuntu:

```sh
sudo apt install meson ninja-build glslang-tools pkg-config flex bison python3-mako \
    python3-packaging python3-yaml llvm-21-dev libclang-21-dev clang-21 libclc-21-dev \
    libllvmspirvlib-21-dev
```

Dependencies are fetched by Meson on the first configure, so that step needs network access:

| Library | Version | Use |
|---|---|---|
| libdrm | 2.4.133 | linked statically, with the patches in `subprojects/packagefiles/` |
| zlib | 1.3.1 | linked statically |
| Expat | 2.5.0 | required by the configure step, not linked into the drivers |

No Android platform libraries are needed: the build uses Mesa's Android stubs.

## Building

```sh
./build.sh <radv|radeonsi|panvk|panfrost> /path/to/android-ndk   # Linux, macOS, WSL
build.cmd <radv|radeonsi> C:\path\to\android-ndk                 # Windows
```

The NDK path can also come from `ANDROID_NDK_HOME` or `ANDROID_NDK_ROOT`. Each driver builds into
`build-<driver>/` (set `BUILD_DIR` to change it); the script strips the libraries and writes a zip
to `dist/`:

- RADV and PanVK: `meta.json`, the driver (`vulkan.radeon.so` or `libvulkan_panfrost.so`) and
  `NOTICE.txt`, for emulators and launchers that load custom Vulkan drivers from a zip.
- RadeonSI and Panfrost: `libEGL_mesa.so`, `libGLESv2_mesa.so`, `libgallium_dri.so` and
  `NOTICE.txt`. RadeonSI also builds Zink and softpipe (set `GALLIUM_DRIVERS` to change that).

PanVK and Panfrost build only on Linux or WSL: the script first builds three host tools into
`build-host/` (`mesa_clc`, `vtn_bindgen2`, `panfrost_compile`). The drivers' OpenCL helper kernels
are compiled with them, so they come from this tree.

`tools/glprobe/` has a small program that checks the OpenGL libraries from `adb shell`.

## Runtime switches

RADV. Each switch is an environment variable or, for apps that cannot set one, an Android property.

| Property | Environment | Effect |
|---|---|---|
| `debug.radv_xclipse_log` | `RADV_XCLIPSE_LOG` | `1` logs diagnostic markers to logcat, `2` adds the verbose trace |
| `debug.radv_xclipse_perf` | `RADV_XCLIPSE_PERF` | `1` logs whether the GPU or the app's CPU limits frame time |
| `debug.radv_xclipse_no_bc_emu` | `RADV_XCLIPSE_NO_BC_EMU` | `1` hides BC4-BC7 (no GPU decode at upload) |
| `debug.radv_xclipse_uf` | `RADV_XCLIPSE_UF` | `1` restores the user fence on graphics submits |
| `debug.radv_xclipse_mtype` | `RADV_XCLIPSE_MTYPE` | VA map MTYPE: `0` default, `3` upstream |
| `debug.radv_xclipse_pal_heaps` | `RADV_XCLIPSE_PAL_HEAPS` | `0` restores the single memory heap |
| `debug.radv_xclipse_dcc` | `RADV_XCLIPSE_DCC` | `0` turns off render target compression (DCC) |
| `debug.mesa_xclipse_prof` | `MESA_XCLIPSE_PROF` | `<delay>,<seconds>` profiles CPU and GPU time per frame and per render pass, then writes `mesa_prof_<pid>.txt` to `MESA_XCLIPSE_PROF_DIR` or the app's `Android/data/<package>/files` |

RadeonSI:

| Property | Environment | Effect |
|---|---|---|
| | `MESA_LOADER_DRIVER_OVERRIDE` | `radeonsi` selects this driver |
| `debug.mesa_xclipse_present_probe` | `MESA_XCLIPSE_PRESENT_PROBE` | `n` logs what every n-th presented frame contains |
| `debug.mesa_xclipse_hnd_dump` | | `1` logs each window buffer's gralloc handle |
| `debug.mesa_xclipse_async_present` | `MESA_XCLIPSE_ASYNC_PRESENT` | `0` presents from the app thread instead of a helper thread |
| `debug.mesa_xclipse_prof` | `MESA_XCLIPSE_PROF` | `<delay>,<seconds>` writes a CPU/GPU profile `mesa_prof_<pid>.txt` to `MESA_XCLIPSE_PROF_DIR` (else `$TMPDIR`) |
| `debug.mesa_xclipse_dcc` | `MESA_XCLIPSE_DCC` | render target compression (DCC): `1` on, `0` off; on by default on the Xclipse 530 only |

PanVK and Panfrost:

| Environment | Effect |
|---|---|
| `PANVK_DEBUG=startup` | logs why the PanVK device could not be created |
| `PANVK_FORCE_IDVS=1` | turns index-driven vertex shading back on in PanVK (off by default on Bifrost) |
| `MESA_KBASE_NODE` | the kbase device Panfrost opens (default `/dev/mali0`); set it for a surfaceless EGL display, which otherwise only looks for DRM devices |

## Disclaimer

This project was developed with heavy use of AI tools. Every change is built and tested before it
is published: the Xclipse drivers on a Galaxy S22 Ultra (SM-S908B, Xclipse 920) and a Galaxy A55
(SM-A556B, Xclipse 530), the Mali drivers on the devices above.

## License

The changes are MIT licensed (`LICENSE`). Files from Mesa keep the license in their headers,
mostly MIT; `THIRD_PARTY_NOTICES.md` lists the files under other licenses and `LICENSES/` has the
full texts. The notices for everything compiled into the drivers ship in every package as
`NOTICE.txt`.

This project is not affiliated with or endorsed by Samsung, AMD, Arm, MediaTek or the Mesa
project. Samsung, Galaxy, Exynos and Xclipse are trademarks of Samsung Electronics. AMD and Radeon
are trademarks of Advanced Micro Devices. Arm and Mali are trademarks of Arm Limited. MediaTek and
Helio are trademarks of MediaTek Inc.
