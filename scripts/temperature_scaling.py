import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ARTIFACTS_DIR = Path("artifacts")
OUT_PATH = ARTIFACTS_DIR / "temperature.json"


class TemperatureScaler(nn.Module):
    """
    Learns a single scalar temperature T > 0 to calibrate logits:
        calibrated_logits = logits / T
    """
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        # optimize log(T) so T stays positive
        self.log_temp = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp)
        return logits / temp

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temp).detach().cpu().item())


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logits_path = ARTIFACTS_DIR / "val_logits.npy"
    labels_path = ARTIFACTS_DIR / "val_labels.npy"
    if not logits_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Missing val_logits.npy / val_labels.npy in artifacts/. "
            "Run your finetune script first."
        )

    logits = np.load(logits_path)   # shape [N, C]
    labels = np.load(labels_path)   # shape [N]

    # Convert to torch
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    scaler = TemperatureScaler(init_temp=1.0)
    criterion = nn.CrossEntropyLoss()

    # LBFGS works great for this 1-parameter optimization
    optimizer = optim.LBFGS([scaler.log_temp], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    # Evaluate before
    with torch.no_grad():
        nll_before = float(criterion(logits_t, labels_t).item())

    optimizer.step(closure)

    # Evaluate after
    with torch.no_grad():
        nll_after = float(criterion(scaler(logits_t), labels_t).item())

    T = scaler.temperature

    # Save artifact
    payload = {
        "temperature": T,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "num_samples": int(labels_t.shape[0]),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"✅ Learned temperature T = {T:.4f}")
    print(f"NLL before: {nll_before:.4f}  -> after: {nll_after:.4f}")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()