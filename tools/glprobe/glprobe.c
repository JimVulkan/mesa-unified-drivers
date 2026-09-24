/*
 * Copyright 2026 JimVulkan
 * SPDX-License-Identifier: MIT
 */

/*
 * glprobe -- bring-up probe for an OpenGL driver on Android, run from adb shell.
 *
 * Loads a Mesa libEGL_mesa.so by path (dlopen, not the system EGL), makes a surfaceless desktop GL
 * context, and runs numbered tests into an offscreen FBO, each checked against values computed
 * here on the CPU. Nothing is displayed. One test failing does not stop the others.
 *
 *   glprobe <libEGL_mesa.so> [tests]      tests: comma list or ranges, default 0 (strings only)
 *
 *   0  strings          context creation, GL_VENDOR/RENDERER/VERSION; no rendering
 *   1  clear            glClear of an RGBA8 FBO, read back
 *   2  triangle         one constant-colour triangle covering half the target
 *   3  varyings         interpolated vertex colours, sampled at known points
 *   4  texture          coordinate-encoded 64x64 texture: texelFetch, texture(), textureGrad()
 *   5  blend            src-alpha blending over a cleared target
 *   6  depth            two overlapping quads with a depth test
 *   7  ubo              colour from a uniform block
 *   8  compute          2D workgroups writing gl_GlobalInvocationID to an SSBO
 *   9  occlusion        samples-passed query around a quad of known size
 *  10  msaa             4x renderbuffer, one triangle, blit-resolve
 *  11  terrain          Minecraft terrain.fsh sampling (sampleNearest -> textureGrad) on an atlas
 *  12  astc             2x2 blocks of ASTC 4x4 (void-extent colours), UNORM, then sRGB
 *  13  etc2             2x2 blocks of ETC2 RGB8 (individual mode, known colours)
 *  14  astc3d           sliced 3D ASTC 4x4: two slices of 2x2 blocks, the second one sampled
 *  15  discard          discarded fragments must not write depth (cutout grass in front of water)
 *
 * Driver selection comes from the environment (MESA_LOADER_DRIVER_OVERRIDE=radeonsi).
 */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glcorearb.h>

/* ---- EGL ---------------------------------------------------------------------------------- */
typedef void *(*PFN_getproc)(const char *);
static PFN_getproc egl_getproc;

#define EGLF(ret, name, args) static ret(*name) args;
EGLF(EGLint, p_eglGetError, (void))
EGLF(EGLBoolean, p_eglInitialize, (EGLDisplay, EGLint *, EGLint *))
EGLF(const char *, p_eglQueryString, (EGLDisplay, EGLint))
EGLF(EGLBoolean, p_eglBindAPI, (EGLenum))
EGLF(EGLContext, p_eglCreateContext, (EGLDisplay, EGLConfig, EGLContext, const EGLint *))
EGLF(EGLBoolean, p_eglMakeCurrent, (EGLDisplay, EGLSurface, EGLSurface, EGLContext))
EGLF(EGLBoolean, p_eglChooseConfig, (EGLDisplay, const EGLint *, EGLConfig *, EGLint, EGLint *))
EGLF(EGLBoolean, p_eglTerminate, (EGLDisplay))
EGLF(EGLBoolean, p_eglDestroyContext, (EGLDisplay, EGLContext))
static PFNEGLGETPLATFORMDISPLAYEXTPROC p_eglGetPlatformDisplayEXT;

/* ---- GL ----------------------------------------------------------------------------------- */
#define GL_FUNCS(X)                                                                                \
   X(PFNGLGETSTRINGPROC, glGetString)                                                              \
   X(PFNGLGETSTRINGIPROC, glGetStringi)                                                            \
   X(PFNGLGETINTEGERVPROC, glGetIntegerv)                                                          \
   X(PFNGLGETERRORPROC, glGetError)                                                                \
   X(PFNGLGENFRAMEBUFFERSPROC, glGenFramebuffers)                                                  \
   X(PFNGLBINDFRAMEBUFFERPROC, glBindFramebuffer)                                                  \
   X(PFNGLGENRENDERBUFFERSPROC, glGenRenderbuffers)                                                \
   X(PFNGLBINDRENDERBUFFERPROC, glBindRenderbuffer)                                                \
   X(PFNGLRENDERBUFFERSTORAGEPROC, glRenderbufferStorage)                                          \
   X(PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC, glRenderbufferStorageMultisample)                    \
   X(PFNGLFRAMEBUFFERRENDERBUFFERPROC, glFramebufferRenderbuffer)                                  \
   X(PFNGLCHECKFRAMEBUFFERSTATUSPROC, glCheckFramebufferStatus)                                    \
   X(PFNGLDELETEFRAMEBUFFERSPROC, glDeleteFramebuffers)                                            \
   X(PFNGLDELETERENDERBUFFERSPROC, glDeleteRenderbuffers)                                          \
   X(PFNGLBLITFRAMEBUFFERPROC, glBlitFramebuffer)                                                  \
   X(PFNGLVIEWPORTPROC, glViewport)                                                                \
   X(PFNGLCLEARCOLORPROC, glClearColor)                                                            \
   X(PFNGLCLEARDEPTHPROC, glClearDepth)                                                            \
   X(PFNGLCLEARPROC, glClear)                                                                      \
   X(PFNGLREADPIXELSPROC, glReadPixels)                                                            \
   X(PFNGLFINISHPROC, glFinish)                                                                    \
   X(PFNGLCREATESHADERPROC, glCreateShader)                                                        \
   X(PFNGLSHADERSOURCEPROC, glShaderSource)                                                        \
   X(PFNGLCOMPILESHADERPROC, glCompileShader)                                                      \
   X(PFNGLGETSHADERIVPROC, glGetShaderiv)                                                          \
   X(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog)                                                \
   X(PFNGLCREATEPROGRAMPROC, glCreateProgram)                                                      \
   X(PFNGLATTACHSHADERPROC, glAttachShader)                                                        \
   X(PFNGLLINKPROGRAMPROC, glLinkProgram)                                                          \
   X(PFNGLGETPROGRAMIVPROC, glGetProgramiv)                                                        \
   X(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog)                                              \
   X(PFNGLUSEPROGRAMPROC, glUseProgram)                                                            \
   X(PFNGLDELETEPROGRAMPROC, glDeleteProgram)                                                      \
   X(PFNGLDELETESHADERPROC, glDeleteShader)                                                        \
   X(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays)                                                  \
   X(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray)                                                  \
   X(PFNGLGENBUFFERSPROC, glGenBuffers)                                                            \
   X(PFNGLBINDBUFFERPROC, glBindBuffer)                                                            \
   X(PFNGLBUFFERDATAPROC, glBufferData)                                                            \
   X(PFNGLBINDBUFFERBASEPROC, glBindBufferBase)                                                    \
   X(PFNGLMAPBUFFERRANGEPROC, glMapBufferRange)                                                    \
   X(PFNGLUNMAPBUFFERPROC, glUnmapBuffer)                                                          \
   X(PFNGLDELETEBUFFERSPROC, glDeleteBuffers)                                                      \
   X(PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer)                                          \
   X(PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray)                                  \
   X(PFNGLDRAWARRAYSPROC, glDrawArrays)                                                            \
   X(PFNGLGENTEXTURESPROC, glGenTextures)                                                          \
   X(PFNGLBINDTEXTUREPROC, glBindTexture)                                                          \
   X(PFNGLTEXIMAGE2DPROC, glTexImage2D)                                                            \
   X(PFNGLCOMPRESSEDTEXIMAGE2DPROC, glCompressedTexImage2D)                                        \
   X(PFNGLCOMPRESSEDTEXIMAGE3DPROC, glCompressedTexImage3D)                                        \
   X(PFNGLTEXPARAMETERIPROC, glTexParameteri)                                                      \
   X(PFNGLACTIVETEXTUREPROC, glActiveTexture)                                                      \
   X(PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation)                                            \
   X(PFNGLUNIFORM1IPROC, glUniform1i)                                                              \
   X(PFNGLUNIFORM1FPROC, glUniform1f)                                                              \
   X(PFNGLGETUNIFORMBLOCKINDEXPROC, glGetUniformBlockIndex)                                        \
   X(PFNGLUNIFORMBLOCKBINDINGPROC, glUniformBlockBinding)                                          \
   X(PFNGLENABLEPROC, glEnable)                                                                    \
   X(PFNGLDISABLEPROC, glDisable)                                                                  \
   X(PFNGLBLENDFUNCPROC, glBlendFunc)                                                              \
   X(PFNGLDEPTHFUNCPROC, glDepthFunc)                                                              \
   X(PFNGLDEPTHMASKPROC, glDepthMask)                                                              \
   X(PFNGLDISPATCHCOMPUTEPROC, glDispatchCompute)                                                  \
   X(PFNGLMEMORYBARRIERPROC, glMemoryBarrier)                                                      \
   X(PFNGLGENQUERIESPROC, glGenQueries)                                                            \
   X(PFNGLBEGINQUERYPROC, glBeginQuery)                                                            \
   X(PFNGLENDQUERYPROC, glEndQuery)                                                                \
   X(PFNGLGETQUERYOBJECTUIVPROC, glGetQueryObjectuiv)                                              \
   X(PFNGLDELETEQUERIESPROC, glDeleteQueries)                                                      \
   X(PFNGLDELETETEXTURESPROC, glDeleteTextures)                                                    \
   X(PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays)

#define DECL(type, name) static type name;
GL_FUNCS(DECL)
#undef DECL

#define W 64
#define H 64

static int load_gl(void)
{
   int missing = 0;
#define LOAD(type, name)                                                                           \
   name = (type)egl_getproc(#name);                                                                \
   if (!name) {                                                                                    \
      printf("missing %s\n", #name);                                                               \
      missing++;                                                                                   \
   }
   GL_FUNCS(LOAD)
#undef LOAD
   return missing;
}

/* ---- helpers ------------------------------------------------------------------------------ */
static int gl_err(const char *where)
{
   GLenum e, first = GL_NO_ERROR;
   while ((e = glGetError()) != GL_NO_ERROR)
      if (first == GL_NO_ERROR)
         first = e;
   if (first != GL_NO_ERROR)
      printf("   GL error 0x%04x at %s\n", first, where);
   return first != GL_NO_ERROR;
}

static GLuint compile(GLenum type, const char *src)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok = 0;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[2048] = {0};
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      printf("   shader compile failed:\n%s\n", log);
      glDeleteShader(s);
      return 0;
   }
   return s;
}

static GLuint program(const char *vs, const char *fs, const char *cs)
{
   GLuint p = glCreateProgram();
   GLuint a = 0, b = 0;
   if (cs) {
      if (!(a = compile(GL_COMPUTE_SHADER, cs)))
         return 0;
      glAttachShader(p, a);
   } else {
      if (!(a = compile(GL_VERTEX_SHADER, vs)) || !(b = compile(GL_FRAGMENT_SHADER, fs)))
         return 0;
      glAttachShader(p, a);
      glAttachShader(p, b);
   }
   glLinkProgram(p);
   GLint ok = 0;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (a)
      glDeleteShader(a);
   if (b)
      glDeleteShader(b);
   if (!ok) {
      char log[2048] = {0};
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      printf("   link failed:\n%s\n", log);
      glDeleteProgram(p);
      return 0;
   }
   return p;
}

struct fbo {
   GLuint fb, color, depth;
};

static int fbo_create(struct fbo *f, GLenum color_fmt, int with_depth, int samples)
{
   memset(f, 0, sizeof(*f));
   glGenFramebuffers(1, &f->fb);
   glBindFramebuffer(GL_FRAMEBUFFER, f->fb);
   glGenRenderbuffers(1, &f->color);
   glBindRenderbuffer(GL_RENDERBUFFER, f->color);
   if (samples > 1)
      glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, color_fmt, W, H);
   else
      glRenderbufferStorage(GL_RENDERBUFFER, color_fmt, W, H);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, f->color);
   if (with_depth) {
      glGenRenderbuffers(1, &f->depth);
      glBindRenderbuffer(GL_RENDERBUFFER, f->depth);
      if (samples > 1)
         glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH24_STENCIL8, W, H);
      else
         glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, W, H);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
                                f->depth);
   }
   GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   if (st != GL_FRAMEBUFFER_COMPLETE) {
      printf("   FBO incomplete: 0x%04x\n", st);
      return 0;
   }
   glViewport(0, 0, W, H);
   return 1;
}

static void fbo_destroy(struct fbo *f)
{
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &f->fb);
   glDeleteRenderbuffers(1, &f->color);
   if (f->depth)
      glDeleteRenderbuffers(1, &f->depth);
}

static uint8_t px[W * H * 4];

static void readback(void)
{
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   glFinish();
}

static const uint8_t *at(int x, int y) { return px + (y * W + x) * 4; }

/* Count pixels differing from the expectation by more than tol in any channel. */
typedef void (*expect_fn)(int x, int y, uint8_t out[4]);

static int verify(const char *what, expect_fn fn, int tol)
{
   int bad = 0, fx = -1, fy = -1;
   uint8_t fe[4] = {0};
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
         uint8_t e[4];
         fn(x, y, e);
         const uint8_t *g = at(x, y);
         for (int c = 0; c < 4; c++)
            if (abs((int)g[c] - (int)e[c]) > tol) {
               if (!bad) {
                  fx = x;
                  fy = y;
                  memcpy(fe, e, 4);
               }
               bad++;
               break;
            }
      }
   if (bad) {
      const uint8_t *g = at(fx, fy);
      printf("   %s: %d/%d pixels wrong; first (%d,%d) got %u,%u,%u,%u want %u,%u,%u,%u\n", what,
             bad, W * H, fx, fy, g[0], g[1], g[2], g[3], fe[0], fe[1], fe[2], fe[3]);
   } else {
      printf("   %s: all %d pixels right\n", what, W * H);
   }
   return bad;
}

static const char *VS_POS2 =
   "#version 330 core\n"
   "layout(location=0) in vec2 pos;\n"
   "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";

static GLuint vao, vbo;

static void set_verts(const float *v, int n_floats)
{
   if (!vao) {
      glGenVertexArrays(1, &vao);
      glGenBuffers(1, &vbo);
   }
   glBindVertexArray(vao);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, n_floats * sizeof(float), v, GL_STATIC_DRAW);
}

static const float FULLSCREEN[] = {-1, -1, 3, -1, -1, 3};

/* ---- tests -------------------------------------------------------------------------------- */
static void e_clear(int x, int y, uint8_t o[4]) { (void)x; (void)y; o[0] = 64; o[1] = 128; o[2] = 191; o[3] = 255; }

static int t_clear(void)
{
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 0, 1))
      return 1;
   glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   readback();
   int r = gl_err("clear") + verify("clear", e_clear, 1);
   fbo_destroy(&f);
   return r;
}

/* Lower-left half: the triangle (-1,-1) (1,-1) (-1,1); pixel centres with x + y < W - 1 inside. */
static void e_tri(int x, int y, uint8_t o[4])
{
   const int inside = (x + 0.5f) + (y + 0.5f) < W;
   o[0] = inside ? 255 : 0;
   o[1] = inside ? 128 : 0;
   o[2] = 0;
   o[3] = 255;
}

static int t_triangle(void)
{
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 0, 1))
      return 1;
   GLuint p = program(VS_POS2,
                      "#version 330 core\nout vec4 c;\nvoid main() { c = vec4(1.0, 0.5, 0.0, 1.0); }\n",
                      NULL);
   if (!p)
      return 1;
   const float v[] = {-1, -1, 1, -1, -1, 1};
   set_verts(v, 6);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glClearColor(0, 0, 0, 1);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(p);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   readback();
   /* Pixels on the diagonal depend on the fill rule; skip them by tolerating the edge. */
   int bad = 0;
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
         if (x + y == W - 1)
            continue;
         uint8_t e[4];
         e_tri(x, y, e);
         const uint8_t *g = at(x, y);
         if (abs(g[0] - e[0]) > 1 || abs(g[1] - e[1]) > 1 || g[2] > 1 || g[3] < 254)
            bad++;
      }
   printf("   triangle: %d pixels wrong (diagonal excluded); (5,5)=%u,%u,%u,%u (60,60)=%u,%u,%u,%u\n",
          bad, at(5, 5)[0], at(5, 5)[1], at(5, 5)[2], at(5, 5)[3], at(60, 60)[0], at(60, 60)[1],
          at(60, 60)[2], at(60, 60)[3]);
   glDeleteProgram(p);
   fbo_destroy(&f);
   return bad + gl_err("triangle");
}

/* Colour = (u, v, 0.5, 1) with u,v the pixel-centre position in [0,1]. */
static void e_var(int x, int y, uint8_t o[4])
{
   o[0] = (uint8_t)lrintf((x + 0.5f) / W * 255.0f);
   o[1] = (uint8_t)lrintf((y + 0.5f) / H * 255.0f);
   o[2] = 128;
   o[3] = 255;
}

static int t_varyings(void)
{
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 0, 1))
      return 1;
   GLuint p = program("#version 330 core\n"
                      "layout(location=0) in vec2 pos;\n"
                      "out vec2 uv;\n"
                      "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n",
                      "#version 330 core\n"
                      "in vec2 uv;\nout vec4 c;\n"
                      "void main() { c = vec4(uv, 0.5, 1.0); }\n",
                      NULL);
   if (!p)
      return 1;
   set_verts(FULLSCREEN, 6);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glUseProgram(p);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   readback();
   int r = gl_err("varyings") + verify("varyings", e_var, 1);
   glDeleteProgram(p);
   fbo_destroy(&f);
   return r;
}

/* Texture texel (i,j) = (i*4, j*4, 255 - i*2, 255). */
static uint8_t tex[W * H * 4];

static void fill_tex(void)
{
   for (int j = 0; j < H; j++)
      for (int i = 0; i < W; i++) {
         uint8_t *t = tex + (j * W + i) * 4;
         t[0] = i * 4;
         t[1] = j * 4;
         t[2] = 255 - i * 2;
         t[3] = 255;
      }
}

static void e_tex(int x, int y, uint8_t o[4]) { memcpy(o, tex + (y * W + x) * 4, 4); }

static int t_texture(void)
{
   fill_tex();
   GLuint t;
   glGenTextures(1, &t);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, t);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   gl_err("teximage");

   static const char *modes[3] = {
      "c = texelFetch(s, ivec2(gl_FragCoord.xy), 0);",
      "c = texture(s, gl_FragCoord.xy / 64.0);",
      "c = textureGrad(s, gl_FragCoord.xy / 64.0, dFdx(gl_FragCoord.xy / 64.0), dFdy(gl_FragCoord.xy / 64.0));",
   };
   static const char *names[3] = {"texelFetch", "texture", "textureGrad"};
   int bad = 0;
   for (int m = 0; m < 3; m++) {
      char fs[1024];
      snprintf(fs, sizeof(fs),
               "#version 330 core\nuniform sampler2D s;\nout vec4 c;\nvoid main() { %s }\n",
               modes[m]);
      struct fbo f;
      if (!fbo_create(&f, GL_RGBA8, 0, 1))
         return 1;
      GLuint p = program(VS_POS2, fs, NULL);
      if (!p)
         return 1;
      set_verts(FULLSCREEN, 6);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glEnableVertexAttribArray(0);
      glUseProgram(p);
      glUniform1i(glGetUniformLocation(p, "s"), 0);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      readback();
      bad += gl_err(names[m]) + verify(names[m], e_tex, 0);
      glDeleteProgram(p);
      fbo_destroy(&f);
   }
   glDeleteTextures(1, &t);
   return bad;
}

/* Blend (0,1,0,0.25) over (1,0,0,1): rgb = src*0.25 + dst*0.75 = (0.75, 0.25, 0). */
static void e_blend(int x, int y, uint8_t o[4]) { (void)x; (void)y; o[0] = 191; o[1] = 64; o[2] = 0; o[3] = 255; }

static int t_blend(void)
{
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 0, 1))
      return 1;
   GLuint p = program(VS_POS2,
                      "#version 330 core\nout vec4 c;\nvoid main() { c = vec4(0.0, 1.0, 0.0, 0.25); }\n",
                      NULL);
   if (!p)
      return 1;
   set_verts(FULLSCREEN, 6);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glClearColor(1, 0, 0, 1);
   glClear(GL_COLOR_BUFFER_BIT);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glUseProgram(p);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glDisable(GL_BLEND);
   readback();
   /* Destination alpha: 0.25*0.25 + 1*0.75 = 0.8125 -> 207. */
   int bad = 0;
   for (int i = 0; i < W * H; i++) {
      const uint8_t *g = px + i * 4;
      if (abs(g[0] - 191) > 1 || abs(g[1] - 64) > 1 || g[2] > 1 || abs(g[3] - 207) > 1)
         bad++;
   }
   printf("   blend: %d pixels wrong; (0,0)=%u,%u,%u,%u want 191,64,0,207\n", bad, px[0], px[1],
          px[2], px[3]);
   (void)e_blend;
   glDeleteProgram(p);
   fbo_destroy(&f);
   return bad + gl_err("blend");
}

/* Near quad (z=-0.5, green) over the whole target, drawn first with depth writes, but its fragment
 * shader discards the left half; then a far quad (z=0.5, red) without depth writes, like
 * Minecraft's water behind cutout grass. Discarded fragments must not write depth. */
static void e_discard(int x, int y, uint8_t o[4])
{
   (void)y;
   const int left = x < W / 2;
   o[0] = left ? 255 : 0;
   o[1] = left ? 0 : 255;
   o[2] = 0;
   o[3] = 255;
}

static int t_discard(void)
{
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 1, 1))
      return 1;
   char fs[256];
   snprintf(fs, sizeof(fs),
            "#version 330 core\nin vec4 col;\nout vec4 c;\n"
            "void main() { if (col.g > 0.5 && gl_FragCoord.x < %d.0) discard; c = col; }\n", W / 2);
   GLuint p = program("#version 330 core\n"
                      "layout(location=0) in vec3 pos;\n"
                      "out vec4 col;\n"
                      "void main() { col = pos.z < 0.0 ? vec4(0,1,0,1) : vec4(1,0,0,1);"
                      " gl_Position = vec4(pos, 1.0); }\n",
                      fs, NULL);
   if (!p)
      return 1;
   const float v[] = {
      -1, -1, -0.5f, 1, -1, -0.5f, -1, 1, -0.5f, 1, -1, -0.5f, 1, 1, -0.5f, -1, 1, -0.5f,
      -1, -1, 0.5f, 1, -1, 0.5f, -1, 1, 0.5f, 1, -1, 0.5f, 1, 1, 0.5f, -1, 1, 0.5f,
   };
   set_verts(v, sizeof(v) / sizeof(v[0]));
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glClearColor(0, 0, 0, 1);
   glClearDepth(1.0);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_LESS);
   glUseProgram(p);
   glDrawArrays(GL_TRIANGLES, 0, 6);
   glDepthMask(GL_FALSE);
   glDrawArrays(GL_TRIANGLES, 6, 6);
   glDepthMask(GL_TRUE);
   glDisable(GL_DEPTH_TEST);
   readback();
   int r = gl_err("discard") + verify("discard", e_discard, 1);
   glDeleteProgram(p);
   fbo_destroy(&f);
   return r;
}

/* Far quad (z=0.5, red) everywhere, near quad (z=-0.5, green) on the left half, drawn first. */
static void e_depth(int x, int y, uint8_t o[4])
{
   (void)y;
   const int left = x < W / 2;
   o[0] = left ? 0 : 255;
   o[1] = left ? 255 : 0;
   o[2] = 0;
   o[3] = 255;
}

static int t_depth(void)
{
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 1, 1))
      return 1;
   GLuint p = program("#version 330 core\n"
                      "layout(location=0) in vec3 pos;\n"
                      "out vec4 col;\n"
                      "void main() { col = pos.z < 0.0 ? vec4(0,1,0,1) : vec4(1,0,0,1);"
                      " gl_Position = vec4(pos, 1.0); }\n",
                      "#version 330 core\nin vec4 col;\nout vec4 c;\nvoid main() { c = col; }\n",
                      NULL);
   if (!p)
      return 1;
   const float v[] = {
      /* near, left half */
      -1, -1, -0.5f, 0, -1, -0.5f, -1, 1, -0.5f, 0, -1, -0.5f, 0, 1, -0.5f, -1, 1, -0.5f,
      /* far, whole target */
      -1, -1, 0.5f, 1, -1, 0.5f, -1, 1, 0.5f, 1, -1, 0.5f, 1, 1, 0.5f, -1, 1, 0.5f,
   };
   set_verts(v, sizeof(v) / sizeof(v[0]));
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glClearColor(0, 0, 0, 1);
   glClearDepth(1.0);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_LESS);
   glUseProgram(p);
   glDrawArrays(GL_TRIANGLES, 0, 12);
   glDisable(GL_DEPTH_TEST);
   readback();
   int r = gl_err("depth") + verify("depth", e_depth, 1);
   glDeleteProgram(p);
   fbo_destroy(&f);
   return r;
}

static void e_ubo(int x, int y, uint8_t o[4]) { (void)x; (void)y; o[0] = 32; o[1] = 96; o[2] = 160; o[3] = 224; }

static int t_ubo(void)
{
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 0, 1))
      return 1;
   GLuint p = program(VS_POS2,
                      "#version 330 core\n"
                      "layout(std140) uniform U { vec4 pad[3]; vec4 col; };\n"
                      "out vec4 c;\nvoid main() { c = col; }\n",
                      NULL);
   if (!p)
      return 1;
   float ub[16] = {0};
   ub[12] = 32 / 255.0f;
   ub[13] = 96 / 255.0f;
   ub[14] = 160 / 255.0f;
   ub[15] = 224 / 255.0f;
   GLuint b;
   glGenBuffers(1, &b);
   glBindBuffer(GL_UNIFORM_BUFFER, b);
   glBufferData(GL_UNIFORM_BUFFER, sizeof(ub), ub, GL_STATIC_DRAW);
   glUniformBlockBinding(p, glGetUniformBlockIndex(p, "U"), 2);
   glBindBufferBase(GL_UNIFORM_BUFFER, 2, b);
   set_verts(FULLSCREEN, 6);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glUseProgram(p);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   readback();
   int r = gl_err("ubo") + verify("ubo", e_ubo, 1);
   glDeleteBuffers(1, &b);
   glDeleteProgram(p);
   fbo_destroy(&f);
   return r;
}

static int t_compute(void)
{
   GLuint p = program(NULL, NULL,
                      "#version 430 core\n"
                      "layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;\n"
                      "layout(std430, binding = 0) buffer B { uvec4 v[]; };\n"
                      "void main() {\n"
                      "   uvec3 g = gl_GlobalInvocationID;\n"
                      "   uint i = g.x + g.y * 32u + g.z * 32u * 16u;\n"
                      "   v[i] = uvec4(g, gl_LocalInvocationIndex);\n"
                      "}\n");
   if (!p)
      return 1;
   const int nx = 32, ny = 16, nz = 4, n = nx * ny * nz;
   GLuint b;
   glGenBuffers(1, &b);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
   glBufferData(GL_SHADER_STORAGE_BUFFER, n * 16, NULL, GL_DYNAMIC_READ);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b);
   glUseProgram(p);
   glDispatchCompute(nx / 8, ny / 4, nz / 2);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
   glFinish();
   const uint32_t *m = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, n * 16, GL_MAP_READ_BIT);
   int bad = 0, first = -1;
   if (!m) {
      printf("   compute: map failed\n");
      return 1;
   }
   for (int z = 0; z < nz; z++)
      for (int y = 0; y < ny; y++)
         for (int x = 0; x < nx; x++) {
            const int i = x + y * nx + z * nx * ny;
            const uint32_t li = (x % 8) + (y % 4) * 8 + (z % 2) * 32;
            if (m[i * 4] != (uint32_t)x || m[i * 4 + 1] != (uint32_t)y ||
                m[i * 4 + 2] != (uint32_t)z || m[i * 4 + 3] != li) {
               if (first < 0)
                  first = i;
               bad++;
            }
         }
   if (bad)
      printf("   compute: %d/%d invocations wrong; first #%d got (%u,%u,%u) li=%u\n", bad, n, first,
             m[first * 4], m[first * 4 + 1], m[first * 4 + 2], m[first * 4 + 3]);
   else
      printf("   compute: all %d invocations right\n", n);
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteBuffers(1, &b);
   glDeleteProgram(p);
   return bad + gl_err("compute");
}

static int t_occlusion(void)
{
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 1, 1))
      return 1;
   GLuint p = program(VS_POS2,
                      "#version 330 core\nout vec4 c;\nvoid main() { c = vec4(1); }\n", NULL);
   if (!p)
      return 1;
   /* A quad covering exactly the left 16 columns: 16 * 64 = 1024 samples. */
   const float x1 = -1.0f + 2.0f * 16 / W;
   const float v[] = {-1, -1, x1, -1, -1, 1, x1, -1, x1, 1, -1, 1};
   set_verts(v, 12);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_LESS);
   GLuint q;
   glGenQueries(1, &q);
   glUseProgram(p);
   glBeginQuery(GL_SAMPLES_PASSED, q);
   glDrawArrays(GL_TRIANGLES, 0, 6);
   glEndQuery(GL_SAMPLES_PASSED);
   GLuint res = 0;
   glGetQueryObjectuiv(q, GL_QUERY_RESULT, &res);
   glDisable(GL_DEPTH_TEST);
   printf("   occlusion: samples passed = %u (want 1024)\n", res);
   glDeleteQueries(1, &q);
   glDeleteProgram(p);
   fbo_destroy(&f);
   return (res != 1024) + gl_err("occlusion");
}

static int t_msaa(void)
{
   struct fbo ms, ss;
   if (!fbo_create(&ms, GL_RGBA8, 0, 4))
      return 1;
   GLuint p = program(VS_POS2,
                      "#version 330 core\nout vec4 c;\nvoid main() { c = vec4(1.0, 0.5, 0.0, 1.0); }\n",
                      NULL);
   if (!p)
      return 1;
   const float v[] = {-1, -1, 1, -1, -1, 1};
   set_verts(v, 6);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glClearColor(0, 0, 0, 1);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(p);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   GLuint msfb = ms.fb;
   if (!fbo_create(&ss, GL_RGBA8, 0, 1))
      return 1;
   glBindFramebuffer(GL_READ_FRAMEBUFFER, msfb);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ss.fb);
   glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST);
   glBindFramebuffer(GL_FRAMEBUFFER, ss.fb);
   readback();
   /* Interior pixels must be fully covered or empty; the diagonal gets partial coverage. */
   int bad = 0, partial = 0;
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
         const uint8_t *g = at(x, y);
         const int d = x + y - (W - 1);
         if (d < -1) {
            if (abs(g[0] - 255) > 1 || abs(g[1] - 128) > 1)
               bad++;
         } else if (d > 1) {
            if (g[0] > 1 || g[1] > 1)
               bad++;
         } else if (g[0] > 1 && g[0] < 254) {
            partial++;
         }
      }
   printf("   msaa: %d interior pixels wrong, %d partially covered edge pixels (want > 0)\n", bad,
          partial);
   glDeleteProgram(p);
   fbo_destroy(&ss);
   fbo_destroy(&ms);
   return bad + (partial == 0) + gl_err("msaa");
}

/* Minecraft 26.x terrain.fsh sampleNearest(): texel-snapped UVs through textureGrad on a 16x16
 * atlas of 4x4-texel sprites. Each texel encodes its own atlas position, so a wrong V is visible. */
static uint8_t atlas[W * H * 4];

static void e_terrain(int x, int y, uint8_t o[4])
{
   /* The quad maps the whole atlas 1:1, so pixel (x,y) must read texel (x,y). */
   memcpy(o, atlas + (y * W + x) * 4, 4);
}

static int t_terrain(void)
{
   for (int j = 0; j < H; j++)
      for (int i = 0; i < W; i++) {
         uint8_t *t = atlas + (j * W + i) * 4;
         t[0] = i * 4;
         t[1] = j * 4;
         t[2] = ((i / 4) + (j / 4) * 16) & 0xff;
         t[3] = 255;
      }
   GLuint t;
   glGenTextures(1, &t);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, t);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, atlas);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 0, 1))
      return 1;
   GLuint p = program(
      "#version 330 core\n"
      "layout(location=0) in vec2 pos;\n"
      "out vec2 uv;\n"
      "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n",
      "#version 330 core\n"
      "uniform sampler2D Sampler0;\n"
      "in vec2 uv;\nout vec4 c;\n"
      "vec4 sampleNearest(sampler2D source, vec2 uv, vec2 pixelSize, vec2 du, vec2 dv, vec2 texelScreenSize) {\n"
      "   vec2 uvTexelCoords = uv / pixelSize;\n"
      "   vec2 texelCenter = round(uvTexelCoords) - 0.5;\n"
      "   vec2 texelOffset = uvTexelCoords - texelCenter;\n"
      "   texelOffset = (texelOffset - 0.5) * pixelSize / texelScreenSize;\n"
      "   texelOffset = clamp(texelOffset + 0.5, 0.0, 1.0);\n"
      "   uv = (texelCenter + texelOffset) * pixelSize;\n"
      "   return textureGrad(source, uv, du, dv);\n"
      "}\n"
      "void main() {\n"
      "   vec2 pixelSize = 1.0 / vec2(textureSize(Sampler0, 0));\n"
      "   vec2 du = dFdx(uv), dv = dFdy(uv);\n"
      "   vec2 texelScreenSize = sqrt(du * du + dv * dv);\n"
      "   c = sampleNearest(Sampler0, uv, pixelSize, du, dv, texelScreenSize);\n"
      "}\n",
      NULL);
   if (!p)
      return 1;
   set_verts(FULLSCREEN, 6);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glUseProgram(p);
   glUniform1i(glGetUniformLocation(p, "Sampler0"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   readback();
   int r = gl_err("terrain") + verify("terrain", e_terrain, 0);
   glDeleteProgram(p);
   fbo_destroy(&f);
   glDeleteTextures(1, &t);
   return r;
}

/* Compressed formats: a 2x2-block texture, each block one known colour, drawn nearest over the
 * whole target, so each quarter of it must be its block's colour. */
static uint8_t cmp_want[4][4];

static void e_cmp(int x, int y, uint8_t o[4])
{
   memcpy(o, cmp_want[(y >= H / 2) * 2 + (x >= W / 2)], 4);
}

static int draw_compressed(const char *what, GLenum fmt, const uint8_t *data, int size, int tol)
{
   GLuint t;
   glGenTextures(1, &t);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, t);
   glCompressedTexImage2D(GL_TEXTURE_2D, 0, fmt, 8, 8, 0, size, data);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   int r = gl_err(what);
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 0, 1))
      return 1;
   GLuint p = program("#version 330 core\nlayout(location=0) in vec2 pos;\nout vec2 uv;\n"
                      "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n",
                      "#version 330 core\nuniform sampler2D s;\nin vec2 uv;\nout vec4 c;\n"
                      "void main() { c = texture(s, uv); }\n",
                      NULL);
   if (!p)
      return 1;
   static const float quad[] = {-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1};
   set_verts(quad, 12);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glUseProgram(p);
   glUniform1i(glGetUniformLocation(p, "s"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 6);
   readback();
   r += gl_err(what) + verify(what, e_cmp, tol);
   glDeleteProgram(p);
   fbo_destroy(&f);
   glDeleteTextures(1, &t);
   return r;
}

/* ASTC void-extent block: a constant RGBA16 colour. */
static void astc_const(uint8_t *b, uint16_t r, uint16_t g, uint16_t bl, uint16_t a)
{
   static const uint8_t head[8] = {0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
   memcpy(b, head, 8);
   const uint16_t c[4] = {r, g, bl, a};
   for (int i = 0; i < 4; i++) {
      b[8 + 2 * i] = c[i] & 0xff;
      b[9 + 2 * i] = c[i] >> 8;
   }
}

static int t_astc(void)
{
   uint8_t d[4 * 16];
   astc_const(d + 0, 0xffff, 0, 0, 0xffff);
   astc_const(d + 16, 0, 0xffff, 0, 0xffff);
   astc_const(d + 32, 0, 0, 0xffff, 0xffff);
   astc_const(d + 48, 0x8080, 0x8080, 0x8080, 0x8080);
   static const uint8_t unorm[4][4] = {{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255},
                                       {128, 128, 128, 128}};
   memcpy(cmp_want, unorm, sizeof(unorm));
   int r = draw_compressed("astc 4x4 unorm", 0x93B0 /* GL_COMPRESSED_RGBA_ASTC_4x4_KHR */, d,
                           sizeof(d), 1);
   /* sRGB: 0x80 decodes to linear 0.2158 -> 55; alpha stays linear. */
   static const uint8_t srgb[4][4] = {{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255},
                                      {55, 55, 55, 128}};
   memcpy(cmp_want, srgb, sizeof(srgb));
   r += draw_compressed("astc 4x4 srgb", 0x93D0 /* GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR */, d,
                        sizeof(d), 2);
   return r;
}

/* ETC2 RGB8 individual mode, both halves one 4-bit colour, table 0, all indices 0: every texel is
 * the colour expanded to 8 bits plus 2. */
static int t_etc2(void)
{
   static const uint8_t d[4 * 8] = {
      0xff, 0x00, 0x00, 0, 0, 0, 0, 0, /* red   -> 255, 2, 2 */
      0x00, 0xff, 0x00, 0, 0, 0, 0, 0, /* green -> 2, 255, 2 */
      0x00, 0x00, 0xff, 0, 0, 0, 0, 0, /* blue  -> 2, 2, 255 */
      0x88, 0x88, 0x88, 0, 0, 0, 0, 0, /* grey  -> 138 */
   };
   static const uint8_t want[4][4] = {{255, 2, 2, 255}, {2, 255, 2, 255}, {2, 2, 255, 255},
                                      {138, 138, 138, 255}};
   memcpy(cmp_want, want, sizeof(want));
   return draw_compressed("etc2 rgb8", 0x9274 /* GL_COMPRESSED_RGB8_ETC2 */, d, sizeof(d), 1);
}

/* Sliced 3D ASTC: slice 0 all black, slice 1 the four colours; sample at r = 0.75 (slice 1). */
static int t_astc3d(void)
{
   uint8_t d[8 * 16];
   for (int i = 0; i < 4; i++)
      astc_const(d + 16 * i, 0, 0, 0, 0xffff);
   astc_const(d + 64, 0xffff, 0, 0, 0xffff);
   astc_const(d + 80, 0, 0xffff, 0, 0xffff);
   astc_const(d + 96, 0, 0, 0xffff, 0xffff);
   astc_const(d + 112, 0x8080, 0x8080, 0x8080, 0x8080);
   static const uint8_t want[4][4] = {{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255},
                                      {128, 128, 128, 128}};
   memcpy(cmp_want, want, sizeof(want));
   GLuint t;
   glGenTextures(1, &t);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_3D, t);
   glCompressedTexImage3D(GL_TEXTURE_3D, 0, 0x93B0, 8, 8, 2, 0, sizeof(d), d);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   int r = gl_err("astc3d upload");
   struct fbo f;
   if (!fbo_create(&f, GL_RGBA8, 0, 1))
      return 1;
   GLuint p = program("#version 330 core\nlayout(location=0) in vec2 pos;\nout vec2 uv;\n"
                      "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n",
                      "#version 330 core\nuniform sampler3D s;\nin vec2 uv;\nout vec4 c;\n"
                      "void main() { c = texture(s, vec3(uv, 0.75)); }\n",
                      NULL);
   if (!p)
      return 1;
   static const float quad[] = {-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1};
   set_verts(quad, 12);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glUseProgram(p);
   glUniform1i(glGetUniformLocation(p, "s"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 6);
   readback();
   r += gl_err("astc3d") + verify("astc 3d slice 1", e_cmp, 1);
   glDeleteProgram(p);
   fbo_destroy(&f);
   glDeleteTextures(1, &t);
   return r;
}

/* ---- main --------------------------------------------------------------------------------- */
static int want[32];

static void parse_tests(const char *s)
{
   while (*s) {
      char *e;
      long a = strtol(s, &e, 10), b = a;
      if (e == s)
         break;
      if (*e == '-') {
         s = e + 1;
         b = strtol(s, &e, 10);
      }
      for (long i = a; i <= b && i < 32; i++)
         if (i >= 0)
            want[i] = 1;
      s = *e == ',' ? e + 1 : e;
   }
}

int main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   if (argc < 2) {
      fprintf(stderr, "usage: %s <libEGL_mesa.so> [tests]\n", argv[0]);
      return 2;
   }
   parse_tests(argc > 2 ? argv[2] : "0");

   void *egl = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
   if (!egl) {
      printf("dlopen %s: %s\n", argv[1], dlerror());
      return 1;
   }
   egl_getproc = (PFN_getproc)dlsym(egl, "eglGetProcAddress");
#define ESYM(n) p_##n = (void *)dlsym(egl, #n)
   ESYM(eglGetError);
   ESYM(eglInitialize);
   ESYM(eglQueryString);
   ESYM(eglBindAPI);
   ESYM(eglCreateContext);
   ESYM(eglMakeCurrent);
   ESYM(eglChooseConfig);
   ESYM(eglTerminate);
   ESYM(eglDestroyContext);
   p_eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)egl_getproc("eglGetPlatformDisplayEXT");
   if (!egl_getproc || !p_eglInitialize || !p_eglGetPlatformDisplayEXT) {
      printf("EGL entry points missing\n");
      return 1;
   }

   EGLDisplay dpy = p_eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   EGLint maj = 0, min = 0;
   if (dpy == EGL_NO_DISPLAY || !p_eglInitialize(dpy, &maj, &min)) {
      printf("eglInitialize failed: 0x%x\n", p_eglGetError());
      return 1;
   }
   printf("EGL %d.%d  vendor: %s\n", maj, min, p_eglQueryString(dpy, EGL_VENDOR));
   if (!p_eglBindAPI(EGL_OPENGL_API)) {
      printf("eglBindAPI(OpenGL) failed: 0x%x\n", p_eglGetError());
      return 1;
   }

   static const EGLint ver[][2] = {{4, 6}, {4, 3}, {3, 3}};
   EGLContext ctx = EGL_NO_CONTEXT;
   for (unsigned i = 0; i < 3 && ctx == EGL_NO_CONTEXT; i++) {
      const EGLint attr[] = {EGL_CONTEXT_MAJOR_VERSION, ver[i][0], EGL_CONTEXT_MINOR_VERSION,
                             ver[i][1], EGL_CONTEXT_OPENGL_PROFILE_MASK,
                             EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
      ctx = p_eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attr);
      if (ctx == EGL_NO_CONTEXT)
         printf("core %d.%d context: failed 0x%x\n", ver[i][0], ver[i][1], p_eglGetError());
   }
   if (ctx == EGL_NO_CONTEXT || !p_eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("no current context: 0x%x\n", p_eglGetError());
      return 1;
   }
   if (load_gl()) {
      printf("GL entry points missing\n");
      return 1;
   }
   GLint next = 0;
   glGetIntegerv(GL_NUM_EXTENSIONS, &next);
   printf("GL_VENDOR   %s\nGL_RENDERER %s\nGL_VERSION  %s\nGLSL        %s\nextensions  %d\n",
          (const char *)glGetString(GL_VENDOR), (const char *)glGetString(GL_RENDERER),
          (const char *)glGetString(GL_VERSION),
          (const char *)glGetString(GL_SHADING_LANGUAGE_VERSION), next);

   static const struct {
      const char *name;
      int (*fn)(void);
   } tests[] = {
      {"strings", NULL},     {"clear", t_clear},     {"triangle", t_triangle},
      {"varyings", t_varyings}, {"texture", t_texture}, {"blend", t_blend},
      {"depth", t_depth},    {"ubo", t_ubo},         {"compute", t_compute},
      {"occlusion", t_occlusion}, {"msaa", t_msaa},  {"terrain", t_terrain},
      {"astc", t_astc},      {"etc2", t_etc2},      {"astc3d", t_astc3d},
      {"discard", t_discard},
   };
   int fails = 0, ran = 0;
   for (unsigned i = 1; i < sizeof(tests) / sizeof(tests[0]); i++) {
      if (!want[i])
         continue;
      printf("[%u] %s\n", i, tests[i].name);
      const int r = tests[i].fn();
      printf("[%u] %s: %s\n", i, tests[i].name, r ? "FAIL" : "PASS");
      fails += r != 0;
      ran++;
   }
   printf("=== %d/%d tests passed ===\n", ran - fails, ran);

   p_eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   p_eglDestroyContext(dpy, ctx);
   p_eglTerminate(dpy);
   return fails ? 1 : 0;
}
