import json
import numpy as np
from pathlib import Path


def save_json(features, json_out: str | Path) -> None:
    with open(str(json_out), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=4, cls=NpEncoder)
    print(f"Saved features to {json_out}")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
