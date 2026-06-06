import json
from pathlib import Path
from datetime import datetime

import numpy as np


def create_run_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = Path("results") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir




def json_converter(obj):
    # NumPy integer tipovi
    if isinstance(obj, np.integer):
        return int(obj)

    # NumPy float tipovi
    if isinstance(obj, np.floating):
        return float(obj)

    # NumPy nizovi
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
            default=json_converter
        )