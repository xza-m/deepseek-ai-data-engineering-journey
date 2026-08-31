"""生成不包含本地敏感路径的环境能力报告。"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _command_report(command: str) -> dict[str, Any]:
    path = shutil.which(command)
    if path is None:
        return {"available": False, "version": None}

    try:
        result = subprocess.run(
            [command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (result.stdout or result.stderr).strip().splitlines()
        version = output[0][:200] if output else None
    except (OSError, subprocess.TimeoutExpired):
        version = None

    return {"available": True, "version": version}


def build_environment_report() -> dict[str, Any]:
    """返回用于 Lab 00 的确定性环境摘要。"""

    system = platform.system()
    python_supported = (3, 10) <= sys.version_info[:2] < (3, 13)
    rdma_devices = Path("/sys/class/infiniband")
    rdma_detected = system == "Linux" and rdma_devices.is_dir() and any(rdma_devices.iterdir())
    cmake_available = shutil.which("cmake") is not None

    threefs_reasons = []
    if system != "Linux":
        threefs_reasons.append("3FS 原生服务端实验需要 Linux")
    if not rdma_detected:
        threefs_reasons.append("未检测到 InfiniBand/RDMA 设备")
    if not cmake_available:
        threefs_reasons.append("未检测到 CMake")

    return {
        "schema_version": "0.1",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "supported": python_supported,
        },
        "platform": {
            "system": system,
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "tools": {name: _command_report(name) for name in ("git", "uv", "docker", "gh")},
        "capabilities": {
            "local_labs_ready": python_supported,
            "rdma_device_detected": rdma_detected,
            "basic_threefs_prerequisites_detected": not threefs_reasons,
            "threefs_notes": threefs_reasons
            or ["仅检测到基础前置条件；仍需按官方部署指南验证 FoundationDB、网络和存储"],
        },
    }
