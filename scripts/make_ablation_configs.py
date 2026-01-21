#!/usr/bin/env python3
"""Create baseline/controlled YAML configs from an existing config."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML is required to read YAML configs.")
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def write_yaml(path: Path, data: dict) -> None:
    if yaml is None:
        raise SystemExit("PyYAML is required to write YAML configs.")
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate baseline/controlled configs that differ only by controller_enabled."
    )
    parser.add_argument("--config", required=True, help="Input YAML config path")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: same directory as config)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else config_path.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    base_data = load_yaml(config_path)

    controlled = dict(base_data)
    controlled["controller_enabled"] = True

    baseline = dict(base_data)
    baseline["controller_enabled"] = False

    write_yaml(out_dir / "config_controlled.yaml", controlled)
    write_yaml(out_dir / "config_baseline.yaml", baseline)


if __name__ == "__main__":
    main()
