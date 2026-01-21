# Benchmarks

## Alanine dipeptide (ACE–ALA–NME)

### Requirements

- AmberTools with `tleap` available (either `AMBERHOME` is set or `tleap` is on `PATH`).

### Generate benchmark inputs

```bash
python cli.py bench_alanine --out benchmarks/alanine
```

This creates two benchmark folders:

- `benchmarks/alanine/implicit/` (ff19SB + mbondi3, no water)
- `benchmarks/alanine/explicit/` (ff19SB + OPC + ionslm_126_opc frcmod, 8.0 Å octahedral box)

Each folder includes `complex.parm7`, `complex.rst7`, `tleap.in`, `tleap.log`, and a
pipeline-ready `config.yml` that enforces CUDA usage.

### Run the pipeline

```bash
python cli.py pipeline --config benchmarks/alanine/implicit/config.yml --out benchmarks/alanine/implicit/out
python cli.py pipeline --config benchmarks/alanine/explicit/config.yml --out benchmarks/alanine/explicit/out
```
