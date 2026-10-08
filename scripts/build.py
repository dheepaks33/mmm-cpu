#!/usr/bin/env python3
"""Build the benchmark binaries on Windows, Linux or macOS.

    python scripts/build.py                 # build/mmm_O0(.exe) and build/mmm(.exe)
    python scripts/build.py --asm           # also write build/bench.s
    python scripts/build.py --vecreport     # show which loops were vectorised
    python scripts/build.py --n 2048        # different matrix size (multiple of 256)
    python scripts/build.py --compiler g++  # GCC instead of clang
    python scripts/build.py --extra=-mprefer-vector-width=512   # 512-bit vectors (AVX-512); note the =

Windows: uses clang++ from PATH, or from C:\\msys64\\ucrt64\\bin if it is not on PATH.
The OpenMP runtime DLL (libomp.dll) and friends are copied next to the .exe,
so the binaries run from any terminal without PATH changes.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmmlib import BUILD, IS_MAC, IS_WIN, ROOT, exe, find_compiler  # noqa: E402

RUNTIME_DLLS = ["libomp.dll", "libgomp-1.dll", "libwinpthread-1.dll", "libgcc_s_seh-1.dll",
                "libstdc++-6.dll", "libc++.dll", "libunwind.dll"]


def flags_for(cxx):
    """(optimisation flags, OpenMP compile flags, OpenMP link flags, extra link flags)."""
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    is_gcc = "g++" in Path(cxx).name and "clang" not in Path(cxx).name
    if IS_MAC:
        prefix = ""
        try:
            prefix = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True,
                                    text=True).stdout.strip()
        except OSError:
            pass
        opt = ["-O3", "-mcpu=apple-m1" if arm else "-march=native", "-ffast-math"]
        if is_gcc:
            return opt, ["-fopenmp"], ["-fopenmp"], []
        omp_c = ["-Xpreprocessor", "-fopenmp"] + (["-I%s/include" % prefix] if prefix else [])
        omp_l = (["-L%s/lib" % prefix] if prefix else []) + ["-lomp"]
        return opt, omp_c, omp_l, []
    opt = ["-O3", "-mcpu=native" if arm else "-march=native", "-ffast-math"]
    extra = ["-static-libgcc", "-static-libstdc++"] if IS_WIN else []
    return opt, ["-fopenmp"], ["-fopenmp"], extra


def run(cmd, env):
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env)


def hint_missing_openmp():
    if IS_WIN:
        print("\nHint: install clang and the OpenMP runtime in MSYS2:\n"
              '  C:\\msys64\\usr\\bin\\bash.exe -lc "pacman -S --needed --noconfirm '
              'mingw-w64-ucrt-x86_64-clang mingw-w64-ucrt-x86_64-llvm-openmp"')
    elif IS_MAC:
        print("\nHint: xcode-select --install && brew install libomp")
    else:
        print("\nHint: sudo apt install clang libomp-dev")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compiler", help="clang++ (default) or g++, or a full path")
    ap.add_argument("--n", type=int, default=1024, help="matrix size, multiple of 256")
    ap.add_argument("--asm", action="store_true", help="also write build/bench.s")
    ap.add_argument("--vecreport", action="store_true", help="print the vectorisation report")
    ap.add_argument("--extra", default="", help="extra compiler flags; write it as --extra=-flag")
    a = ap.parse_args()

    cxx = find_compiler(a.compiler)
    if not cxx:
        print("No C++ compiler found (looked for %s)." % (a.compiler or "clang++, g++"))
        hint_missing_openmp()
        return 1
    env = os.environ.copy()
    cxx_dir = str(Path(cxx).parent)
    env["PATH"] = cxx_dir + os.pathsep + env.get("PATH", "")  # so clang finds ld/as next to it

    BUILD.mkdir(exist_ok=True)
    src = str(ROOT / "src" / "bench.cpp")
    opt, omp_c, omp_l, extra = flags_for(cxx)
    opt = opt + a.extra.split()
    base = [cxx, "-std=c++17", "-Wall", "-DMMM_N=%d" % a.n]
    print("compiler: %s" % cxx)
    ver = subprocess.run([cxx, "--version"], capture_output=True, text=True, env=env).stdout
    print(ver.splitlines()[0] if ver else "")

    # Step 1 binary: no optimisation at all (the article's "default flags" row).
    r = run(base + ["-O0", "-Wno-unknown-pragmas", '-DBUILD_FLAGS="-O0"', src,
                    "-o", str(exe("mmm_O0"))] + extra, env)
    if r.returncode != 0:
        hint_missing_openmp()
        return 1

    # Steps 2-6 binary: -O3 -march=native -ffast-math + OpenMP.
    flag_text = " ".join(opt + ["-fopenmp"])
    r = run(base + opt + omp_c + ['-DBUILD_FLAGS="%s"' % flag_text, src,
                                  "-o", str(exe("mmm"))] + omp_l + extra, env)
    if r.returncode != 0:
        hint_missing_openmp()
        return 1

    if IS_WIN:  # make build\*.exe self-contained: copy runtime DLLs next to them
        copied = []
        for dll in RUNTIME_DLLS:
            p = Path(cxx_dir) / dll
            if p.exists():
                shutil.copy2(p, BUILD / dll)
                copied.append(dll)
        if copied:
            print("copied next to the .exe: " + ", ".join(copied))

    # Smoke test: the binary starts and finds its OpenMP runtime.
    smoke = subprocess.run([str(exe("mmm")), "topology"], capture_output=True, text=True)
    if smoke.returncode != 0:
        print("build/mmm failed to start (exit code %d)." % smoke.returncode)
        if IS_WIN:
            print("Exit code 3221225781 / -1073741515 means a DLL is missing (usually libomp.dll).\n"
                  "Quick fix for this terminal:  $env:Path = \"%s;$env:Path\"" % cxx_dir)
        hint_missing_openmp()
        return 1
    print("built: %s, %s" % (exe("mmm_O0"), exe("mmm")))

    if a.asm:
        out = BUILD / "bench.s"
        r = run(base + opt + omp_c + ["-S", "-fno-asynchronous-unwind-tables", src,
                                      "-o", str(out)], env)
        if r.returncode == 0:
            text = out.read_text(encoding="utf-8", errors="replace")
            packed = len(re.findall(r"vfmadd\d{3}ps", text))
            scalar = len(re.findall(r"vfmadd\d{3}ss", text))
            print("wrote %s: %d packed FMAs (vfmadd...ps, 8-16 floats each), %d scalar FMAs "
                  "(vfmadd...ss), %d ARM fmla" % (out, packed, scalar, text.count("fmla")))

    if a.vecreport:
        if "clang" in Path(cxx).name:
            rep = ["-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize"]
        else:
            rep = ["-fopt-info-vec-optimized"]
        r = subprocess.run(base + opt + omp_c + rep + ["-c", src, "-o", os.devnull],
                           capture_output=True, text=True, env=env)
        lines = sorted({ln.strip() for ln in (r.stderr + r.stdout).splitlines()
                        if "kernels.h" in ln and ("remark" in ln or "optimized" in ln)})
        print("\n".join(lines) if lines else "no vectorisation remarks for kernels.h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
