import json
from pathlib import Path
import torch
import torch.nn.functional as F

def load_temperature(path: str = "artifacts/temperature.json") -> float:
    p = Path(path)
    if not p.exists():
        # fallback: no calibration
        return 1.0
    return float(json.loads(p.read_text())["temperature"])

def calibrated_softmax(logits: torch.Tensor, T: float) -> torch.Tensor:
    T = max(float(T), 1e-6)
    return F.softmax(logits / T, dim=-1)