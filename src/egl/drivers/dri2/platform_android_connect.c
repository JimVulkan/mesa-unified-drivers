/*
 * Copyright 2026 JimVulkan
 * SPDX-License-Identifier: MIT
 */

/* Producer connect/disconnect for an ANativeWindow. Kept apart from platform_android.c because
 * it needs the full struct ANativeWindow from <system/window.h>, which clashes with the
 * <vndk/window.h> that file uses. */

#include <system/window.h>

int droid_native_window_api_connect(struct ANativeWindow *window);
int droid_native_window_api_disconnect(struct ANativeWindow *window);

int
droid_native_window_api_connect(struct ANativeWindow *window)
{
   return native_window_api_connect(window, NATIVE_WINDOW_API_EGL);
}

int
droid_native_window_api_disconnect(struct ANativeWindow *window)
{
   return native_window_api_disconnect(window, NATIVE_WINDOW_API_EGL);
}
