@echo off
rem Build one of the Xclipse drivers for Android (arm64, API 34) and package it into dist\.
rem
rem usage: build.cmd <radv|radeonsi> [path-to-android-ndk]
rem   radv       RADV Vulkan
rem   radeonsi   RadeonSI OpenGL/EGL, with Zink and softpipe
rem   The Mali drivers (panvk, panfrost) build on Linux or WSL only, with build.sh.
rem
rem   The NDK can also come from ANDROID_NDK_HOME or ANDROID_NDK_ROOT.
rem   BUILD_DIR overrides the build directory (default: build-<driver>).
rem   GALLIUM_DRIVERS overrides the radeonsi drivers (default: radeonsi,zink,softpipe).
rem Requires: meson, ninja, python (mako, packaging), the NDK's llvm-strip, and:
rem   radv: glslangValidator
rem   radeonsi: win_flex and win_bison (winflexbison) or flex and bison
setlocal

cd /d "%~dp0"

set "DRIVER=%~1"
if /i "%DRIVER%"=="radv" goto :driver_ok
if /i "%DRIVER%"=="radeonsi" goto :driver_ok
echo usage: build.cmd ^<radv^|radeonsi^> [path-to-android-ndk] 1>&2
echo The Mali drivers (panvk, panfrost) build on Linux or WSL only, with build.sh. 1>&2
exit /b 1
:driver_ok

set "NDK=%~2"
if "%NDK%"=="" set "NDK=%ANDROID_NDK_HOME%"
if "%NDK%"=="" set "NDK=%ANDROID_NDK_ROOT%"
if "%NDK%"=="" goto :no_ndk
if not exist "%NDK%\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android34-clang.cmd" goto :no_ndk

if not exist "%NDK%\toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-strip.exe" (
   echo error: llvm-strip.exe not found in the NDK 1>&2
   exit /b 1
)

set "BIN=%NDK:\=/%/toolchains/llvm/prebuilt/windows-x86_64/bin"

for %%T in (meson ninja python) do (
   where %%T >nul 2>nul || (
      echo error: %%T not found in PATH 1>&2
      exit /b 1
   )
)
if /i "%DRIVER%"=="radv" (
   where glslangValidator >nul 2>nul || (
      echo error: glslangValidator not found in PATH 1>&2
      exit /b 1
   )
) else (
   where win_flex >nul 2>nul || where flex >nul 2>nul || (
      echo error: neither win_flex nor flex found in PATH 1>&2
      exit /b 1
   )
   where win_bison >nul 2>nul || where bison >nul 2>nul || (
      echo error: neither win_bison nor bison found in PATH 1>&2
      exit /b 1
   )
)

if "%BUILD_DIR%"=="" (set "BUILD=build-%DRIVER%") else (set "BUILD=%BUILD_DIR%")
if "%GALLIUM_DRIVERS%"=="" (set "DRIVERS=radeonsi,zink,softpipe") else (set "DRIVERS=%GALLIUM_DRIVERS%")
if not exist "%BUILD%" mkdir "%BUILD%"

> "%BUILD%\cross.ini" (
   echo [binaries]
   echo c = '%BIN%/aarch64-linux-android34-clang.cmd'
   echo cpp = '%BIN%/aarch64-linux-android34-clang++.cmd'
   echo ar = '%BIN%/llvm-ar.exe'
   echo strip = '%BIN%/llvm-strip.exe'
   echo.
   echo [properties]
   echo cpp_link_args = ['-static-libstdc++']
   echo.
   echo [host_machine]
   echo system = 'android'
   echo cpu_family = 'aarch64'
   echo cpu = 'aarch64'
   echo endian = 'little'
)

if /i "%DRIVER%"=="radv" (
   set "OPTS=-Dvulkan-drivers=amd -Dgallium-drivers= -Dglx=disabled -Dgbm=disabled -Degl=disabled -Dgles1=disabled -Dgles2=disabled"
   set "LIBS=src\amd\vulkan\libvulkan_radeon.so"
) else (
   set "OPTS=-Dgallium-drivers=%DRIVERS% -Dvulkan-drivers= -Degl=enabled -Dopengl=true -Dgles1=disabled -Dgles2=enabled -Dglx=disabled -Dgbm=disabled -Degl-lib-suffix=_mesa -Dgles-lib-suffix=_mesa -Dvideo-codecs="
   set "LIBS=src\egl\libEGL_mesa.so src\gallium\targets\dri\libgallium_dri.so src\mesa\glapi\es2api\libGLESv2_mesa.so"
)

if exist "%BUILD%\build.ninja" goto :build
call meson setup "%BUILD%" --cross-file "%BUILD%\cross.ini" ^
   -Dbuildtype=debugoptimized -Db_ndebug=true ^
   -Dplatforms=android -Dplatform-sdk-version=34 -Dandroid-stub=true -Dandroid-strict=false ^
   -Dandroid-libbacktrace=disabled ^
   %OPTS% -Dvulkan-layers= -Dtools= ^
   -Dllvm=disabled -Dzstd=disabled -Dlmsensors=disabled -Dperfetto=false ^
   -Dallow-fallback-for=libdrm,perfetto --force-fallback-for=expat,libdrm,zlib ^
   -Dlibdrm:default_library=static -Dexpat:default_library=static -Dzlib:default_library=static ^
   -Dc_args=-march=armv8.2-a -Dcpp_args=-march=armv8.2-a
if errorlevel 1 exit /b 1

:build
set "TARGETS=%LIBS:\=/%"
ninja -C "%BUILD%" %TARGETS%
if errorlevel 1 exit /b 1

if not exist "%BUILD%\stripped" mkdir "%BUILD%\stripped"
if not exist dist mkdir dist
for %%L in (%LIBS%) do (
   "%NDK%\toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-strip.exe" -o "%BUILD%\stripped\%%~nxL" "%BUILD%\%%L"
   if errorlevel 1 exit /b 1
)
python android\package.py %DRIVER% "%BUILD%\stripped" dist
exit /b %errorlevel%

:no_ndk
echo error: pass the Android NDK path or set ANDROID_NDK_HOME 1>&2
exit /b 1
