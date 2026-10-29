"""config.yaml loading. Relative paths resolve against the config file's folder,
so every script and the server behave the same from any working directory."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

SKU_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

DEFAULT_DEFECT_TYPES = ["missing_tuft", "stray_bristle", "uneven_trim", "contamination", "other"]

# Operator-facing names. Gujarati first; override or extend in config.yaml
# under `defect_labels:` without touching code.
DEFECT_LABELS = {
    "missing_tuft": {"gu": "ગુચ્છો ખૂટે છે", "en": "Missing tuft"},
    "sparse_tuft": {"gu": "પાતળો ગુચ્છો", "en": "Sparse tuft"},
    "splayed_tuft": {"gu": "ફેલાયેલો ગુચ્છો", "en": "Splayed tuft"},
    "stray_bristle": {"gu": "છૂટો બ્રિસલ", "en": "Stray bristle"},
    "uneven_trim": {"gu": "અસમાન ટ્રિમિંગ", "en": "Uneven trim"},
    "contamination": {"gu": "ડાઘ / ગંદકી", "en": "Contamination"},
    "other": {"gu": "અન્ય ખામી", "en": "Other defect"},
    "good": {"gu": "બરાબર (ખોટું એલાર્મ)", "en": "Good (false alarm)"},
}

DEFAULTS: dict = {
    "server": {"host": "0.0.0.0", "port": 8600, "threads": 8},
    "data_root": "data",
    "models_root": "models",
    "serve_data": "serve/data",
    "storage": {"keep_days": 60},
    "camera": {
        "source": None,
        "width": 1280,
        "height": 720,
        "backend": "auto",
        "settle_ms": 250,
        "jpeg_quality": 92,
        "fps": 15,
        "lock": {},
        "replay_advance_s": 2.0,
        "freeze_guard": True,
    },
    "skus": {},
    "defect_labels": {},
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = {}
    if p.exists():
        from .data import read_text_any

        raw = yaml.safe_load(read_text_any(p)) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{p} must contain a YAML mapping")
    elif path:
        raise ConfigError(f"config file not found: {p}")
    cfg = _merge(DEFAULTS, raw)
    cfg["_root"] = str(p.parent.resolve())
    cfg["_path"] = str(p.resolve())
    pin = (cfg.get("server") or {}).get("pin")
    if pin is not None and not isinstance(pin, str):
        raise ConfigError(f"server.pin must be quoted text, e.g. pin: '0123' (YAML read {pin!r})")
    for sku in cfg["skus"] or {}:
        check_sku(sku)
    return cfg


def check_sku(sku: str) -> str:
    if not isinstance(sku, str) or not SKU_RE.match(sku):
        raise ConfigError(
            f"invalid SKU id {sku!r}: use letters, digits, '_' or '-' (max 64 chars)"
        )
    return sku


def root(cfg: dict) -> Path:
    return Path(cfg.get("_root", REPO_ROOT))


def resolve(cfg: dict, rel) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root(cfg) / p


def sku_cfg(cfg: dict, sku: str) -> dict:
    check_sku(sku)
    base = {
        "display_name": sku,
        "model_dir": f"{cfg.get('models_root', 'models')}/{sku}",
        "defect_types": list(DEFAULT_DEFECT_TYPES),
    }
    return _merge(base, (cfg.get("skus") or {}).get(sku) or {})


def data_dir(cfg: dict, sku: str) -> Path:
    return resolve(cfg, cfg.get("data_root", "data")) / check_sku(sku)


def model_dir(cfg: dict, sku: str) -> Path:
    return resolve(cfg, sku_cfg(cfg, sku)["model_dir"])


def serve_data_dir(cfg: dict) -> Path:
    return resolve(cfg, cfg.get("serve_data", "serve/data"))


def defect_label(cfg: dict, key: str) -> dict:
    custom = (cfg.get("defect_labels") or {}).get(key)
    if custom:
        return {"gu": custom.get("gu", key), "en": custom.get("en", key)}
    if key in DEFECT_LABELS:
        return dict(DEFECT_LABELS[key])
    pretty = key.replace("_", " ").strip().capitalize()
    return {"gu": pretty, "en": pretty}


def run_cli(main) -> None:
    """Run a script's main(); setup mistakes print one line instead of a traceback."""
    from .data import ROIError

    try:
        main()
    except (ConfigError, ROIError) as exc:
        raise SystemExit(f"error: {exc}")
