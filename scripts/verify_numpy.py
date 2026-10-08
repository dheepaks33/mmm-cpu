#!/usr/bin/env python3
"""Cross-check a kernel's output against numpy (the article checked against PyTorch).

    ./build/mmm parallel --tile 16 --dump /tmp/mmm_dump
    python3 scripts/verify_numpy.py /tmp/mmm_dump
"""
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1 else "dump")
rows, cols, inners = map(int, (d / "shape.txt").read_text().split())
A = np.fromfile(d / "A.bin", dtype=np.float32).reshape(rows, inners)
B = np.fromfile(d / "B.bin", dtype=np.float32).reshape(inners, cols)
C = np.fromfile(d / "C.bin", dtype=np.float32).reshape(rows, cols)

expected = A @ B
ok = np.allclose(C, expected, rtol=1e-4, atol=1e-3)
print("max abs diff vs numpy: %.3e  ->  %s" % (np.max(np.abs(C - expected)), "MATCH" if ok else "MISMATCH"))
sys.exit(0 if ok else 1)
