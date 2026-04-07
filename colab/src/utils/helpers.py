import os
import json
import random
import shutil
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone

try:
    import psutil
except ImportError:
    psutil = None


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, data: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def normalize_split_value(x: str) -> str:
    x = str(x).strip().lower()
    split_map = {
        "training": "train", "train": "train", "tr": "train",
        "validation": "val", "valid": "val", "val": "val", "dev": "val",
        "testing": "test", "test": "test", "ts": "test",
    }
    return split_map.get(x, x)


def get_disk_free_gb(path="."):
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def get_system_metrics():
    metrics = {
        "cpu_percent": None, "ram_used_mb": None, "ram_total_mb": None,
        "ram_percent": None, "gpu_name": None, "gpu_memory_used_mb": None,
        "gpu_memory_reserved_mb": None, "gpu_memory_total_mb": None,
        "gpu_utilization_percent": None, "disk_free_gb": None,
    }

    if psutil is not None:
        try:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.2)
            vm = psutil.virtual_memory()
            metrics["ram_used_mb"] = vm.used / (1024 ** 2)
            metrics["ram_total_mb"] = vm.total / (1024 ** 2)
            metrics["ram_percent"] = vm.percent
        except Exception:
            pass

    try:
        metrics["disk_free_gb"] = get_disk_free_gb(".")
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            metrics["gpu_name"] = torch.cuda.get_device_name(0)
            metrics["gpu_memory_used_mb"] = torch.cuda.memory_allocated(0) / (1024 ** 2)
            metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(0) / (1024 ** 2)
            props = torch.cuda.get_device_properties(0)
            metrics["gpu_memory_total_mb"] = props.total_memory / (1024 ** 2)
        except Exception:
            pass
        metrics["gpu_utilization_percent"] = None

    return metrics
