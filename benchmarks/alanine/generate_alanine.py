"""
Generate ACE–ALA–NME alanine dipeptide benchmarks using AmberTools tleap.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

IMPLICIT_TLEAP = (
    "source leaprc.protein.ff19SB\n"
    "source leaprc.gaff2\n"
    "set default PBRadii mbondi3\n"
    "mol = sequence { ACE ALA NME }\n"
    "saveamberparm mol complex.parm7 complex.rst7\n"
    "quit\n"
)

EXPLICIT_TLEAP = (
    "source leaprc.protein.ff19SB\n"
    "source leaprc.water.opc\n"
    "loadamberparams frcmod.ionslm_126_opc\n"
    "set default PBRadii mbondi3\n"
    "mol = sequence { ACE ALA NME }\n"
    "solvateoct mol OPCBOX 8.0\n"
    "saveamberparm mol complex.parm7 complex.rst7\n"
    "quit\n"
)


def _assert_tleap_available() -> None:
    amberhome = os.environ.get("AMBERHOME")
    if amberhome and Path(amberhome).exists():
        return
    if shutil.which("tleap"):
        return
    raise RuntimeError("AmberTools tleap not found: set AMBERHOME or ensure tleap is on PATH.")


def _write_tleap_input(folder: Path, content: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    tleap_path = folder / "tleap.in"
    tleap_path.write_text(content, encoding="utf-8")
    return tleap_path


def _run_tleap(folder: Path) -> None:
    log_path = folder / "tleap.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            ["tleap", "-f", "tleap.in"],
            cwd=folder,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"tleap failed in {folder}. See log: {log_path}")
    for filename in ("complex.parm7", "complex.rst7"):
        output_path = folder / filename
        if not output_path.exists():
            raise FileNotFoundError(f"Expected output not found: {output_path}")


def _write_config(folder: Path, sim_type: str) -> Path:
    config_text = "\n".join(
        [
            "parmFile: complex.parm7",
            "crdFile: complex.rst7",
            f"simType: {sim_type}",
            "platform: CUDA",
            "require_gpu: true",
            "cuda_device_index: 0",
            "cuda_precision: mixed",
            "precision: mixed",
            "",
        ]
    )
    config_path = folder / "config.yml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def generate_alanine(out_root: Path) -> dict[str, Path]:
    _assert_tleap_available()
    out_root.mkdir(parents=True, exist_ok=True)

    implicit_dir = out_root / "implicit"
    explicit_dir = out_root / "explicit"

    _write_tleap_input(implicit_dir, IMPLICIT_TLEAP)
    _write_tleap_input(explicit_dir, EXPLICIT_TLEAP)

    _run_tleap(implicit_dir)
    _write_config(implicit_dir, "protein.implicit")

    _run_tleap(explicit_dir)
    _write_config(explicit_dir, "explicit")

    return {"implicit": implicit_dir, "explicit": explicit_dir}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output root for benchmarks (default: benchmarks/alanine).",
    )
    args = parser.parse_args()
    locations = generate_alanine(args.out)
    for label, path in locations.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
