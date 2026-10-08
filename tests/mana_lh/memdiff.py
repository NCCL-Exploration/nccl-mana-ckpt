#!/usr/bin/env python3
"""Diff a live process's file-backed mappings of a library against the file.

  memdiff.py <pid> <library-path-substring>

Prints each differing byte run (offset in the file, file bytes, memory bytes),
and the maps lines of the library.  Used on MANA's upper half (2026-10-08) to
find what overwrote libmana.so's .dynsym.
"""

import sys

pid, lib = int(sys.argv[1]), sys.argv[2]
maps = [line.split() for line in open(f"/proc/{pid}/maps")]
regs = [m for m in maps if len(m) >= 6 and lib in m[5]]
for m in regs:
    print("  ", " ".join(m[:3]), m[5])
mem = open(f"/proc/{pid}/mem", "rb", buffering=0)
path = regs[0][5]
data = open(path, "rb").read()
for m in regs:
    lo, hi = (int(x, 16) for x in m[0].split("-"))
    off = int(m[2], 16)
    if "w" in m[1]:
        continue  # relocated data differs by design
    mem.seek(lo)
    buf = mem.read(hi - lo)
    ref = data[off : off + len(buf)]
    i, n = 0, min(len(buf), len(ref))
    while i < n:
        if buf[i] != ref[i]:
            j = i
            while j < n and (buf[j] != ref[j] or (j + 8 < n and buf[j : j + 8] != ref[j : j + 8])):
                j += 1
            print(f"   diff file+0x{off + i:x} (addr 0x{lo + i:x}) len {j - i}:")
            print(f"      file: {ref[i:min(j, i + 48)].hex(' ')}")
            print(f"      mem : {buf[i:min(j, i + 48)].hex(' ')}")
            i = j
        else:
            i += 1
