"""Small helpers shared by every script: seeding, saving results, timing, GPU info."""
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_builtin(value):
    """Make numpy / torch numbers JSON-serialisable."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(data), f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[saved] {path}")


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[saved] {path}")


class Timer:
    """with Timer() as t: ...  then t.seconds"""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def run_plots(step: str) -> None:
    """Draw a step's figures in a separate Python process without PyTorch (see src/plots.py)."""
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot.py"
    subprocess.run([sys.executable, str(script), step], check=True)


def device_info() -> dict:
    import torch
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
    return info


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_peak_vram() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb():
    """Peak PyTorch-allocated GPU memory since the last reset (None on CPU)."""
    import torch
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    return None
