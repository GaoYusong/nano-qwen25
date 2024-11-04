
import os
from pathlib import Path

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct")


def newest_snapshot(model_path: str) -> str:
    with open(Path(model_path) / "refs" / "main", "r") as f:
        snapshot_code = f.read().strip()
    return Path(model_path) / "snapshots" / snapshot_code
