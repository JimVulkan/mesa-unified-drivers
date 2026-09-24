#!/usr/bin/env python3
"""Package one driver's stripped libraries as a zip with the third-party notices (NOTICE.txt).

usage: package.py <radv|radeonsi|panvk|panfrost> <dir with the stripped libraries> <output dir>

The Vulkan drivers (radv, panvk) are adrenotools-style zips with a meta.json, for emulators and
launchers that load custom Vulkan drivers from a zip. The OpenGL drivers (radeonsi, panfrost) are
plain zips of libEGL_mesa.so, libGLESv2_mesa.so and libgallium_dri.so.
"""
import json
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GL_LIBS = ('libEGL_mesa.so', 'libGLESv2_mesa.so', 'libgallium_dri.so')

DRIVERS = {
    'radv': {
        'zip': 'radv-xclipse',
        'libs': {'libvulkan_radeon.so': 'vulkan.radeon.so'},
        'name': 'RADV Xclipse',
        'description': 'RADV for Samsung Xclipse GPUs, based on Mesa %s.',
        'minApi': 34,
    },
    'radeonsi': {
        'zip': 'radeonsi-xclipse',
        'libs': {name: name for name in GL_LIBS},
    },
    'panvk': {
        'zip': 'mali-panvk',
        'libs': {'libvulkan_panfrost.so': 'libvulkan_panfrost.so'},
        'name': 'PanVK Mali Bifrost (kbase)',
        'description': 'PanVK for Mali Bifrost GPUs on the stock Arm kbase kernel driver, based on '
                       'Mesa %s.',
        'minApi': 31,
        # The headers say 1.4; PanVK reports 1.3 on Bifrost (v6/v7), which is what it is for.
        'vulkan_minor': 3,
    },
    'panfrost': {
        'zip': 'mali-panfrost',
        'libs': {name: name for name in GL_LIBS},
    },
}


def git(*args):
    try:
        run = lambda *a: subprocess.run(['git', '-C', ROOT, *a], capture_output=True, text=True,
                                        check=True).stdout.strip()
        # Only this repository's history, never an enclosing one.
        if os.path.normcase(os.path.abspath(run('rev-parse', '--show-toplevel'))) != os.path.normcase(ROOT):
            return ''
        return run(*args)
    except (OSError, subprocess.CalledProcessError):
        return ''


def unstripped_sections(path):
    """Names of symbol-table and debug sections left in an ELF64 little-endian file."""
    data = open(path, 'rb').read()
    if data[:4] != b'\x7fELF' or data[4] != 2 or data[5] != 1:
        sys.exit('error: %s is not a 64-bit little-endian ELF file' % path)
    shoff = int.from_bytes(data[0x28:0x30], 'little')
    shentsize = int.from_bytes(data[0x3a:0x3c], 'little')
    shnum = int.from_bytes(data[0x3c:0x3e], 'little')
    shstrndx = int.from_bytes(data[0x3e:0x40], 'little')
    sections = [data[shoff + i * shentsize:shoff + (i + 1) * shentsize] for i in range(shnum)]
    strtab = sections[shstrndx]
    str_off = int.from_bytes(strtab[0x18:0x20], 'little')
    names = []
    for s in sections:
        name_off = str_off + int.from_bytes(s[0:4], 'little')
        names.append(data[name_off:data.index(b'\0', name_off)].decode('ascii', 'replace'))
    return [n for n in names if n == '.symtab' or n.startswith('.debug')]


def vulkan_version(minor=None):
    header = open(os.path.join(ROOT, 'include', 'vulkan', 'vulkan_core.h')).read()
    api = re.search(r'VK_HEADER_VERSION_COMPLETE VK_MAKE_API_VERSION\(0, (\d+), (\d+), VK_HEADER_VERSION\)', header)
    patch = re.search(r'#define VK_HEADER_VERSION (\d+)', header)
    return 'Vulkan %s.%s.%s' % (api.group(1), api.group(2) if minor is None else minor, patch.group(1))


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in DRIVERS:
        sys.exit(__doc__)
    driver, lib_dir, out_dir = sys.argv[1:]
    d = DRIVERS[driver]

    for name in d['libs']:
        path = os.path.join(lib_dir, name)
        if not os.path.isfile(path):
            sys.exit('error: %s is missing' % path)
        leftovers = unstripped_sections(path)
        if leftovers:
            sys.exit('error: %s is not stripped (%s); run llvm-strip first' % (path, ', '.join(leftovers[:4])))

    mesa_version = open(os.path.join(ROOT, 'VERSION')).read().strip()
    commit = git('rev-parse', '--short', 'HEAD')
    count = git('rev-list', '--count', 'HEAD') or '1'
    suffix = '-' + commit if commit else ''

    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, '%s-%s%s.zip' % (d['zip'], mesa_version, suffix))
    info = ''
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        if 'name' in d:
            vulkan = vulkan_version(d.get('vulkan_minor'))
            (library,) = d['libs'].values()
            meta = {
                'schemaVersion': 1,
                'name': '%s (Mesa %s%s)' % (d['name'], mesa_version, suffix),
                'description': d['description'] % mesa_version,
                'author': 'JimVulkan',
                'packageVersion': count,
                'vendor': 'Mesa',
                'driverVersion': vulkan,
                'minApi': d['minApi'],
                'libraryName': library,
            }
            z.writestr('meta.json', json.dumps(meta, indent=2) + '\n')
            info = vulkan + ', '
        for name, packaged in d['libs'].items():
            z.write(os.path.join(lib_dir, name), packaged)
        z.write(os.path.join(ROOT, 'android', 'NOTICE.txt'), 'NOTICE.txt')
    print('packaged %s (%s%.1f MB)' % (zip_path, info, os.path.getsize(zip_path) / 1e6))


if __name__ == '__main__':
    main()
