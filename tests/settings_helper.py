"""Test settings come from the real config/settings.yaml, never from copies of its values."""

from pathlib import Path

from quant.config.settings import load_settings

SETTINGS = load_settings(Path(__file__).resolve().parents[1] / "config" / "settings.yaml")
