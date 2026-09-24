# glprobe

A small test program for the OpenGL libraries (RadeonSI or Panfrost), run from `adb shell`. It
loads `libEGL_mesa.so` by path (not the system EGL), creates a surfaceless desktop OpenGL context,
renders each test into an offscreen framebuffer and checks every pixel against values computed on
the CPU. Nothing is shown on screen.

| Test | Checks |
|---|---|
| 0 | context creation and the GL strings (no rendering) |
| 1 | clear |
| 2 | one triangle covering half the target |
| 3 | interpolated vertex attributes |
| 4 | `texelFetch`, `texture` and `textureGrad` on a 64x64 texture |
| 5 | alpha blending |
| 6 | depth test with two overlapping quads |
| 7 | uniform buffer |
| 8 | compute shader with 2D workgroups (`gl_GlobalInvocationID`) |
| 9 | occlusion query |
| 10 | 4x MSAA with a resolve blit |
| 11 | Minecraft's terrain sampling (`sampleNearest` through `textureGrad`) |
| 12 | ASTC 4x4 textures, UNORM and sRGB |
| 13 | ETC2 RGB8 textures |
| 14 | 3D ASTC texture, second slice sampled |
| 15 | discarded pixels don't write depth (cutout grass in front of water) |

## Building

With the Android NDK, from the repository root:

```sh
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang -O2 \
   -Iinclude tools/glprobe/glprobe.c -o glprobe -ldl -lm
```

## Running

Push the three libraries from the package and `glprobe` to the phone, then:

```sh
adb shell mkdir -p /data/local/tmp/gl
adb push libEGL_mesa.so libGLESv2_mesa.so libgallium_dri.so glprobe /data/local/tmp/gl/
adb shell "cd /data/local/tmp/gl && chmod 755 glprobe && \
   MESA_LOADER_DRIVER_OVERRIDE=radeonsi LD_LIBRARY_PATH=/data/local/tmp/gl \
   ./glprobe /data/local/tmp/gl/libEGL_mesa.so 1-15"
```

On a Mali phone, replace `MESA_LOADER_DRIVER_OVERRIDE=radeonsi` with `MESA_KBASE_NODE=/dev/mali0`
(the kbase GPU has no DRM node for a surfaceless context to find). The second argument selects tests as a
comma-separated list or ranges; the default is `0`. The last line is `=== n/m tests passed ===`,
and the exit code is non-zero if any test failed.
