# Reproducibility (Nature Methods supplementary software)

This document describes how to reproduce the benchmark outputs used in the
supplementary material for the DeepMD workflow. The included benchmark is a
minimal, synthetic alanine-dipeptide-style series that generates a potential of
mean force (PMF) and summary statistics.

## Environment

Use Python 3.10+ with the dependencies needed by the benchmark code:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

## Reproducing the benchmark outputs

The benchmark lives in `benchmarks/alanine`. The default configuration is
`benchmarks/alanine/config.json` and includes the seed, mixture parameters, and
number of steps used to generate the synthetic series.

1. Run the benchmark script:

   ```bash
   python benchmarks/alanine/run_benchmark.py \
     --config benchmarks/alanine/config.json \
     --outdir benchmarks/alanine/out
   ```

2. Verify that the outputs are created:

   - `benchmarks/alanine/out/pmf.json`
   - `benchmarks/alanine/out/metrics.json`
   - `benchmarks/alanine/out/runtime.json`

## Reproducing with custom parameters

You can override the number of steps, bin count, and RNG seed from the CLI. The
example below regenerates outputs with a new seed and smaller run size:

```bash
python benchmarks/alanine/run_benchmark.py \
  --config benchmarks/alanine/config.json \
  --outdir benchmarks/alanine/out_custom \
  --steps 2000 \
  --bins 50 \
  --seed 2026
```

## Notes

- The benchmark is deterministic for a fixed seed.
- The runtime report includes the elapsed time and steps/second to help compare
  runs across hardware.
- To run GPU smoke tests locally, use `RUN_GPU_TESTS=1 pytest -q -m gpu`.
