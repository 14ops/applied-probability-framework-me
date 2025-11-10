from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class Secrets:
    alpaca_key: str
    alpaca_secret: str
    alpaca_base_url: str


def load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return yaml.safe_load(p.read_text())


def load_secrets(path: str | Path = "config/secrets.yaml") -> Secrets:
    data = load_yaml(path)
    return Secrets(
        alpaca_key=data["ALPACA_API_KEY_ID"],
        alpaca_secret=data["ALPACA_API_SECRET_KEY"],
        alpaca_base_url=data["ALPACA_PAPER_BASE_URL"],
    )


def load_settings(path: str | Path = "config/settings.yaml") -> Dict[str, Any]:
    return load_yaml(path)


