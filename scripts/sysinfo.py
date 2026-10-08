#!/usr/bin/env python3
"""Record everything that affects the numbers, so a run can be reproduced or
compared later. run_all.py saves this as results/<stamp>/system.txt.

    python scripts/sysinfo.py
"""
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmmlib import IS_MAC, IS_WIN, ROOT, cpu_model, exe, find_compiler, get_affinity_mask  # noqa: E402

# Windows 11 "Power mode" overlay GUIDs (Settings > System > Power & battery)
POWER_MODES = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "Better performance",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}


def section(title):
    print("\n===== %s =====" % title)


def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return "(%s: %s)" % (cmd[0], e.__class__.__name__)


def windows_power():
    import ctypes
    from ctypes import wintypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]

    s = SystemPowerStatus()
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        ac = {0: "on battery", 1: "plugged in (AC)"}.get(s.ACLineStatus, "unknown")
        print("power source: %s, battery %s%%" % (ac, s.BatteryLifePercent if s.BatteryLifePercent != 255 else "?"))
        print("battery saver: %s" % ("ON" if s.SystemStatusFlag == 1 else "off"))
    print(sh(["powercfg", "/getactivescheme"]))
    try:
        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                        ("Data4", ctypes.c_ubyte * 8)]

        g = GUID()
        fn = ctypes.windll.powrprof.PowerGetEffectiveOverlayScheme
        if fn(ctypes.byref(g)) == 0:
            text = "%08x-%04x-%04x-%s-%s" % (g.Data1, g.Data2, g.Data3, bytes(g.Data4[:2]).hex(),
                                              bytes(g.Data4[2:]).hex())
            print("power mode: %s (%s)" % (POWER_MODES.get(text, "unknown"), text))
    except (AttributeError, OSError):
        print("power mode: not readable here - note it by hand")


def main():
    section("date / host")
    print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    print(platform.platform())
    print("python %s on %s" % (platform.python_version(), platform.machine()))

    section("cpu")
    print("model: %s" % cpu_model())
    print("logical processors: %s   this process may use: %s" % (os.cpu_count(), get_affinity_mask()))
    if IS_WIN:
        print(sh(["powershell", "-NoProfile", "-Command",
                  "Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,"
                  "NumberOfLogicalProcessors,L2CacheSize,L3CacheSize,MaxClockSpeed,CurrentClockSpeed; "
                  "Get-CimInstance Win32_ComputerSystem | Format-List Manufacturer,Model,TotalPhysicalMemory"],
                 timeout=60))
    elif IS_MAC:
        print(sh(["sysctl", "hw.physicalcpu", "hw.logicalcpu", "hw.perflevel0.physicalcpu",
                  "hw.perflevel1.physicalcpu", "hw.l1dcachesize", "hw.perflevel0.l2cachesize", "hw.memsize"]))
    else:
        print(sh(["lscpu"]))
        print(sh(["free", "-h"]))

    section("core types / simd (mmm topology)")
    print(sh([str(exe("mmm")), "topology"]) if exe("mmm").exists() else "build/mmm not built yet")

    section("power / frequency policy")
    if IS_WIN:
        windows_power()
    elif IS_MAC:
        print(sh(["pmset", "-g", "batt"]))
    else:
        cpufreq = Path("/sys/devices/system/cpu/cpu0/cpufreq")
        if cpufreq.exists():
            for f in ("scaling_governor", "scaling_driver"):
                p = cpufreq / f
                print("%s: %s" % (f, p.read_text().strip() if p.exists() else "?"))
        else:
            print("cpufreq not exposed (VM / WSL2): note the host's power mode by hand")
        for p in (Path("/sys/devices/system/cpu/intel_pstate/no_turbo"),
                  Path("/sys/devices/system/cpu/cpufreq/boost"), Path("/sys/firmware/acpi/platform_profile")):
            if p.exists():
                print("%s: %s" % (p, p.read_text().strip()))
        for p in sorted(Path("/sys/class/power_supply").glob("A*/online")):
            print("%s = %s (1 = plugged in)" % (p, p.read_text().strip()))
        if hasattr(os, "getloadavg"):
            print("load average: %.2f %.2f %.2f" % os.getloadavg())

    section("compilers")
    for name in ("clang++", "g++"):
        c = find_compiler(name)
        if c:
            v = sh([c, "--version"]).splitlines()
            print("%s: %s" % (c, v[0] if v else "?"))

    section("OpenMP environment")
    omp = {k: v for k, v in os.environ.items() if k.startswith(("OMP_", "KMP_", "GOMP_"))}
    print("\n".join("%s=%s" % kv for kv in sorted(omp.items())) or "(none set)")

    section("python / numpy / BLAS")
    try:
        import numpy as np
        print("numpy", np.__version__)
        try:
            from threadpoolctl import threadpool_info
            for lib in threadpool_info():
                print({k: lib.get(k) for k in ("user_api", "internal_api", "version", "num_threads",
                                               "architecture")})
        except ImportError:
            print("threadpoolctl not installed")
    except ImportError:
        print("numpy not installed")

    section("git")
    print(sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]))
    print(sh(["git", "-C", str(ROOT), "status", "--short"]) or "(clean)")


if __name__ == "__main__":
    main()
