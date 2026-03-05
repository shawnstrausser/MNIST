"""
system_info.py — Collects hardware, software, and environment context.

When you're training models, it matters WHERE you trained them. Was it CPU
or GPU? What PyTorch version? How much RAM? This module captures all of that
so every training run is fully reproducible and debuggable.

Saved into metadata JSON so you can always answer: "what machine produced
these results?"
"""

import os
import platform
import sys

import torch


def get_system_info():
    """
    Collect hardware/software/environment details.

    Returns:
        dict with system context — ready to dump into metadata JSON
    """
    info = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "cpu": _get_cpu_name(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": _get_ram_gb(),
        "gpu": _get_gpu_info(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)

    return info


def _get_cpu_name():
    """Get CPU model name (best effort, platform-dependent)."""
    try:
        if platform.system() == "Windows":
            return platform.processor() or "Unknown"
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


def _get_ram_gb():
    """Get total system RAM in GB (best effort)."""
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024**3), 2)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 2)
    except Exception:
        pass
    return None


def print_system_info(info):
    """Print system info to terminal in a clean format."""
    print(f"\n{'=' * 55}")
    print(f"  System Info")
    print(f"{'=' * 55}")
    print(f"    Python:    {info['python_version']}")
    print(f"    PyTorch:   {info['pytorch_version']}")
    print(f"    OS:        {info['os']}")
    print(f"    CPU:       {info['cpu']}")
    print(f"    Cores:     {info['cpu_cores']}")
    if info['ram_gb']:
        print(f"    RAM:       {info['ram_gb']} GB")
    print(f"    GPU:       {info['gpu']}")
    if info['cuda_available']:
        print(f"    CUDA:      {info.get('cuda_version', 'N/A')}")
        print(f"    GPU Mem:   {info.get('gpu_memory_gb', 'N/A')} GB")
    print(f"{'=' * 55}\n")


def _get_gpu_info():
    """Get GPU name if CUDA is available."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "None (CPU only)"
