#!/bin/sh
# Build one of the drivers for Android (arm64) and package it into dist/.
#
# usage: ./build.sh <driver> [path-to-android-ndk]
#   radv       RADV Vulkan for Samsung Xclipse (API 34)
#   radeonsi   RadeonSI OpenGL/EGL for Samsung Xclipse, with Zink and softpipe (API 34)
#   panvk      PanVK Vulkan for Arm Mali Bifrost on kbase (API 31)
#   panfrost   Panfrost OpenGL/EGL for Arm Mali Bifrost on kbase (API 31)
#
#   The NDK can also come from ANDROID_NDK_HOME or ANDROID_NDK_ROOT.
#   BUILD_DIR overrides the Android build directory (default: build-<driver>).
#   GALLIUM_DRIVERS overrides the radeonsi drivers (default: radeonsi,zink,softpipe).
#
# panvk and panfrost build on Linux (or WSL) only: they first build three host tools into
# build-host (mesa_clc, vtn_bindgen2, panfrost_compile) that compile the drivers' OpenCL helper
# kernels (libpan, poly), so those have to come from this tree and need LLVM.
#
# Requires: meson, ninja, python3 (mako, packaging, and pyyaml for the Mali drivers), and:
#   radv, panvk: glslangValidator
#   radeonsi, panfrost: flex and bison
#   panvk, panfrost: pkg-config, and LLVM with clang, libclc and the SPIR-V LLVM translator of one
#   major version (Ubuntu: llvm-dev libclang-dev libclc-XX-dev libllvmspirvlib-XX-dev clang)
set -eu

cd "$(dirname "$0")"

usage() {
   echo "usage: $0 <radv|radeonsi|panvk|panfrost> [path-to-android-ndk]" >&2
   exit 1
}

[ $# -ge 1 ] || usage
DRIVER="$1"
case "$DRIVER" in
   # The Xclipse phones all have ARMv8.2 CPUs; some Mali Bifrost phones (Exynos 9611) do not.
   radv|radeonsi) API=34; MALI=; MARCH=-march=armv8.2-a ;;
   panvk|panfrost) API=31; MALI=1; MARCH= ;;
   *) usage ;;
esac

NDK="${2:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK" ] || [ ! -d "$NDK" ]; then
   echo "error: pass the Android NDK path or set ANDROID_NDK_HOME" >&2
   exit 1
fi

case "$(uname -s)" in
   Linux*) HOST=linux-x86_64; EXE=; WRAP= ;;
   Darwin*) HOST=darwin-x86_64; EXE=; WRAP= ;;
   MINGW*|MSYS*|CYGWIN*) HOST=windows-x86_64; EXE=.exe; WRAP=.cmd ;;
   *) echo "error: unsupported host $(uname -s)" >&2; exit 1 ;;
esac
if [ -n "$MALI" ] && [ "$HOST" != linux-x86_64 ]; then
   echo "error: build $DRIVER on Linux (or WSL); its host tools need LLVM" >&2
   exit 1
fi

BIN="$NDK/toolchains/llvm/prebuilt/$HOST/bin"
if [ ! -x "$BIN/aarch64-linux-android$API-clang$WRAP" ] && [ ! -f "$BIN/aarch64-linux-android$API-clang$WRAP" ]; then
   echo "error: no API $API arm64 compiler in $BIN" >&2
   exit 1
fi

if [ ! -f "$BIN/llvm-strip$EXE" ]; then
   echo "error: llvm-strip not found in $BIN" >&2
   exit 1
fi

PYTHON=python3
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
TOOLS="meson ninja $PYTHON"
case "$DRIVER" in
   radv) TOOLS="$TOOLS glslangValidator" ;;
   radeonsi) TOOLS="$TOOLS flex bison" ;;
   panvk) TOOLS="$TOOLS pkg-config glslangValidator" ;;
   panfrost) TOOLS="$TOOLS pkg-config flex bison" ;;
esac
for tool in $TOOLS; do
   if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: $tool not found in PATH" >&2
      exit 1
   fi
done

# --- host tools (Mali drivers) ----------------------------------------------------------------
if [ -n "$MALI" ]; then
   HOSTB=build-host
   if [ ! -f "$HOSTB/build.ninja" ]; then
      meson setup "$HOSTB" \
         -Dbuildtype=debugoptimized \
         -Dmesa-clc=enabled -Dprecomp-compiler=enabled \
         -Dvulkan-drivers=panfrost -Dgallium-drivers= -Dplatforms= -Dllvm=enabled \
         -Dgles1=disabled -Dgles2=disabled -Degl=disabled -Dglx=disabled -Dgbm=disabled \
         -Dtools= -Dvideo-codecs= \
         -Dallow-fallback-for=libdrm --force-fallback-for=libdrm,expat,zlib \
         -Dlibdrm:default_library=static
   fi
   ninja -C "$HOSTB" src/compiler/clc/mesa_clc src/compiler/spirv/vtn_bindgen2 \
      src/panfrost/clc/panfrost_compile
   mkdir -p "$HOSTB/bin"
   cp "$HOSTB/src/compiler/clc/mesa_clc" "$HOSTB/src/compiler/spirv/vtn_bindgen2" \
      "$HOSTB/src/panfrost/clc/panfrost_compile" "$HOSTB/bin/"
   PATH="$PWD/$HOSTB/bin:$PATH"
   export PATH
fi

# --- the driver -------------------------------------------------------------------------------
BUILD="${BUILD_DIR:-build-$DRIVER}"
mkdir -p "$BUILD" "$BUILD/pkgconfig-empty"

# The Mali builds run pkg-config (for the host tools); keep the host's .pc files out of the cross
# build.
PKGBIN=
PKGPROP=
if [ -n "$MALI" ]; then
   PKGBIN="pkg-config = 'pkg-config'"
   PKGPROP="pkg_config_libdir = ['$PWD/$BUILD/pkgconfig-empty']"
fi

cat > "$BUILD/cross.ini" <<EOF
[binaries]
c = '$BIN/aarch64-linux-android$API-clang$WRAP'
cpp = '$BIN/aarch64-linux-android$API-clang++$WRAP'
ar = '$BIN/llvm-ar$EXE'
strip = '$BIN/llvm-strip$EXE'
$PKGBIN

[properties]
$PKGPROP
cpp_link_args = ['-static-libstdc++']

[host_machine]
system = 'android'
cpu_family = 'aarch64'
cpu = 'aarch64'
endian = 'little'
EOF

GL="-Degl=enabled -Dopengl=true -Dgles1=disabled -Dgles2=enabled -Dglx=disabled -Dgbm=disabled
    -Degl-lib-suffix=_mesa -Dgles-lib-suffix=_mesa"
NOGL="-Dglx=disabled -Dgbm=disabled -Degl=disabled -Dgles1=disabled -Dgles2=disabled"
case "$DRIVER" in
   radv) OPTS="-Dvulkan-drivers=amd -Dgallium-drivers= $NOGL" ;;
   radeonsi) OPTS="-Dgallium-drivers=${GALLIUM_DRIVERS:-radeonsi,zink,softpipe} -Dvulkan-drivers= $GL" ;;
   panvk) OPTS="-Dvulkan-drivers=panfrost -Dgallium-drivers= $NOGL" ;;
   panfrost) OPTS="-Dgallium-drivers=panfrost -Dvulkan-drivers= $GL" ;;
esac
if [ "$DRIVER" != radv ]; then
   OPTS="$OPTS -Dvideo-codecs="
fi
if [ -n "$MARCH" ]; then
   OPTS="$OPTS -Dc_args=$MARCH -Dcpp_args=$MARCH"
fi
if [ -n "$MALI" ]; then
   OPTS="$OPTS -Dmesa-clc=system -Dprecomp-compiler=system"
fi

if [ ! -f "$BUILD/build.ninja" ]; then
   # shellcheck disable=SC2086
   meson setup "$BUILD" --cross-file "$BUILD/cross.ini" \
      -Dbuildtype=debugoptimized -Db_ndebug=true \
      -Dplatforms=android -Dplatform-sdk-version=$API -Dandroid-stub=true -Dandroid-strict=false \
      -Dandroid-libbacktrace=disabled \
      $OPTS -Dvulkan-layers= -Dtools= \
      -Dllvm=disabled -Dzstd=disabled -Dlmsensors=disabled -Dperfetto=false \
      -Dallow-fallback-for=libdrm,perfetto --force-fallback-for=expat,libdrm,zlib \
      -Dlibdrm:default_library=static -Dexpat:default_library=static -Dzlib:default_library=static
fi

case "$DRIVER" in
   radv) LIBS="src/amd/vulkan/libvulkan_radeon.so" ;;
   panvk) LIBS="src/panfrost/vulkan/libvulkan_panfrost.so" ;;
   *) LIBS="src/egl/libEGL_mesa.so src/gallium/targets/dri/libgallium_dri.so
            src/mesa/glapi/es2api/libGLESv2_mesa.so" ;;
esac
# shellcheck disable=SC2086
ninja -C "$BUILD" $LIBS

mkdir -p "$BUILD/stripped" dist
for lib in $LIBS; do
   "$BIN/llvm-strip$EXE" -o "$BUILD/stripped/$(basename "$lib")" "$BUILD/$lib"
done
"$PYTHON" android/package.py "$DRIVER" "$BUILD/stripped" dist
