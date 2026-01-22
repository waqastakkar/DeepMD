# GaMD GNN pipeline

This module provides a residue-level graph neural network pipeline for GaMD-enhanced protein dynamics.

## Data expectations

The CLI expects an `.npz` bundle with arrays keyed as follows:

- `node_features`: `(T, N, F)` residue features per frame (e.g., torsions, RMSF, secondary structure, energies).
- `positions`: `(T, N, 3)` residue coordinates (e.g., Cα positions).
- Optional edge masks: `contacts`, `hbonds`, `salts`, `covariance` with shape `(T, N, N)`.
- Labels (optional): `delta_v`, `state`, `rmsd`, `rg`, `latent`.

## CLI usage

```bash
python scripts/gnn_pipeline.py --npz path/to/your.npz --out out_dir --epochs 25 --sequence 8 --batch 4
```

## Example data

Generate synthetic alanine dipeptide and 100-residue examples:

```bash
python scripts/gnn_example.py --out examples
```

Then run the pipeline:

```bash
python scripts/gnn_pipeline.py --npz examples/alanine_gamd.npz --out out_alanine
python scripts/gnn_pipeline.py --npz examples/protein_gamd.npz --out out_protein
```

## Outputs

- `gamd_gnn_model.keras` trained model
- `importance.csv` residue-level importance map
- Optional edge list exports for allosteric networks
- JSON UMAP-ready latent embedding payloads
