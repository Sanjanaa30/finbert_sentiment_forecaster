from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "final_model.pkl"
CONFIG_PATH = ROOT / "artifacts" / "final_selected_config.json"


@dataclass
class ModelBundle:
    model: Any
    features: list[str]
    config: dict
    model_version: str


_bundle: ModelBundle | None = None


def _build_model_version(config: dict) -> str:
    # Option A: timestamp-based
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # include key config to make version readable
    h = config.get("horizon_days", "na")
    t = config.get("threshold_bps", "na")
    m = config.get("model", "na")
    return f"{m}-h{h}-t{t}-{ts}"


def load_bundle() -> ModelBundle:
    global _bundle
    if _bundle is not None:
        return _bundle

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config artifact: {CONFIG_PATH}")

    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    _bundle = ModelBundle(
        model=payload["model"],
        features=payload["features"],
        config=config,
        model_version=_build_model_version(config),
    )
    return _bundle


def get_bundle() -> ModelBundle:
    if _bundle is None:
        raise RuntimeError("Model bundle not loaded. Call load_bundle() at startup.")
    return _bundle
