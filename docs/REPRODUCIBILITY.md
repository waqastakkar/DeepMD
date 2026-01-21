# Reproducibility (Nature Methods supplementary software)

This document describes how to reproduce the benchmark outputs used in the
supplementary material for the DeepMD workflow. The benchmark uses a real
ACE–ALA–NME alanine dipeptide system built with AmberTools `tleap` and runs the
full pipeline on the resulting Amber `parm7`/`rst7` inputs.

## Environment

Use Python 3.10+ with the dependencies needed by the benchmark code:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducing the benchmark outputs

The benchmark lives in `benchmarks/alanine`. The `bench_alanine` generator
creates both implicit and explicit systems plus pipeline-ready `config.yml`
files that point at the generated `parm7`/`rst7` files.

1. Generate benchmark inputs with AmberTools `tleap`:

   ```bash
   python cli.py bench_alanine --out benchmarks/alanine
   ```

2. Run the pipeline on the implicit solvent system:

   ```bash
   python cli.py pipeline \
     --config benchmarks/alanine/implicit/config.yml \
     --out benchmarks/alanine/implicit/out
   ```

3. Repeat for explicit solvent if desired:

   ```bash
   python cli.py pipeline \
     --config benchmarks/alanine/explicit/config.yml \
     --out benchmarks/alanine/explicit/out
   ```

## Notes

- The benchmark requires an AmberTools installation with `tleap` available
  (`AMBERHOME` set or `tleap` on `PATH`).
- The configs default to CUDA and require a GPU; edit `config.yml` if you want
  to run on CPU.
- To run GPU smoke tests locally, use `RUN_GPU_TESTS=1 pytest -q -m gpu`.
- The opt-in pipeline regression test is gated behind `RUN_REAL_MD_TESTS=1`:
  `RUN_REAL_MD_TESTS=1 pytest -q -k real_md_pipeline`.
