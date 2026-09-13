#!/usr/bin/env python3
"""Record the software and hardware a measurement ran on, without anything
that identifies the machine, the account or the person (no hostname, no user,
no paths). Prints a table and, if given a path, writes it as JSON."""
import json, platform, subprocess, sys

info = {
    "os": platform.platform(),
    "python": sys.version.split()[0],
}

try:
    import numpy
    info["numpy"] = numpy.__version__
except ImportError:
    info["numpy"] = None

try:
    import cupy as cp
    info["cupy"] = cp.__version__
    info["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
    info["cuda_driver_api"] = cp.cuda.runtime.driverGetVersion()
    p = cp.cuda.runtime.getDeviceProperties(0)
    resident = p["multiProcessorCount"] * p["maxThreadsPerMultiProcessor"]
    info["gpu"] = {
        "name": p["name"].decode(),
        "compute_capability": ".".join(cp.cuda.Device(0).compute_capability),
        "sms": p["multiProcessorCount"],
        "l2_bytes": p["l2CacheSize"],
        "l2_bytes_per_resident_thread": round(p["l2CacheSize"] / resident, 1),
        "memory_bytes": p["totalGlobalMem"],
        "peak_bandwidth_gbs": round(
            2 * p["memoryClockRate"] * 1e3 * (p["memoryBusWidth"] / 8) / 1e9, 1),
    }
except ImportError:
    info["cupy"] = None

try:
    o = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True,
                       timeout=5)
    info["nvidia_driver"] = o.stdout.strip().splitlines()[0]
except Exception:
    info["nvidia_driver"] = None

for k, v in info.items():
    if isinstance(v, dict):
        for k2, v2 in v.items():
            print(f"{'gpu.' + k2:32} {v2}")
    else:
        print(f"{k:32} {v}")

if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as f:
        json.dump(info, f, indent=2)
