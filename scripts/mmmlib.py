"""Shared helpers for the mmm-cpu scripts (Windows, Linux, macOS).

Only the standard library is used here, so build.py works before numpy is installed.
"""
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
EXE = ".exe" if IS_WIN else ""

# Where MSYS2 puts clang on Windows (searched when clang++ is not on PATH).
MSYS2_BIN_DIRS = [
    Path(r"C:\msys64\ucrt64\bin"),
    Path(r"C:\msys64\clang64\bin"),
    Path(r"C:\msys64\mingw64\bin"),
]


def exe(name):
    """Path of a built binary, e.g. exe('mmm') -> build/mmm(.exe)."""
    return BUILD / (name + EXE)


def logical_cpus():
    return os.cpu_count() or 1


def find_compiler(preferred=None):
    """Return the full path of a C++ compiler, or None."""
    names = [preferred] if preferred else ["clang++", "g++"]
    for name in names:
        hit = shutil.which(name)
        if hit:
            return hit
        if IS_WIN:
            for d in MSYS2_BIN_DIRS:
                cand = d / (name + ".exe")
                if cand.exists():
                    return str(cand)
    return None


def cpu_model():
    if IS_WIN:
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            return platform.processor() or "unknown"
    if IS_MAC:
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip() or "unknown"
        except OSError:
            return "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _kernel32():
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    return k32


def get_affinity_mask():
    """Hex mask of the logical CPUs this process may use (first 64), or 'n/a'."""
    try:
        if IS_WIN:
            import ctypes
            from ctypes import wintypes
            k32 = _kernel32()
            proc, system = ctypes.c_size_t(), ctypes.c_size_t()
            k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                                   ctypes.POINTER(ctypes.c_size_t)]
            if k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(system)):
                return "0x%X" % proc.value
            return "unknown"
        if hasattr(os, "sched_getaffinity"):
            mask = 0
            for c in os.sched_getaffinity(0):
                if c < 64:
                    mask |= 1 << c
            return "0x%X" % mask
    except Exception:
        return "unknown"
    return "n/a"


def set_affinity(mask):
    """Restrict this process (and every child it starts) to the CPUs in `mask`."""
    if IS_WIN:
        import ctypes
        from ctypes import wintypes
        k32 = _kernel32()
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.SetProcessAffinityMask.restype = wintypes.BOOL
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
            raise OSError("SetProcessAffinityMask failed (error %d)" % ctypes.get_last_error())
        return
    if hasattr(os, "sched_setaffinity"):
        cpus = {i for i in range(64) if mask >> i & 1}
        os.sched_setaffinity(0, cpus)
        return
    raise OSError("setting CPU affinity is not supported on this OS (macOS)")


def cpus_in_mask(mask):
    return sum(1 for i in range(64) if mask >> i & 1)


def disable_power_throttling():
    """Windows 11: opt this process out of EcoQoS so it is not moved to E-cores
    or slowed down when its window is not in the foreground. No-op elsewhere."""
    if not IS_WIN:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        k32 = _kernel32()
        state = PowerThrottlingState(1, 1, 0)  # current version, EXECUTION_SPEED, off
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        process_power_throttling = 4  # PROCESS_INFORMATION_CLASS value
        return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), process_power_throttling,
                                              ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:
        return False
