# GNN Pipeline and Saliency Mapping

DeepMD includes a residue-level graph neural network (GNN) pipeline for analyzing GaMD trajectories and producing saliency maps. The implementation is in `src/paddle/learn/gnn_pipeline.py` with CLI entry points in `scripts/gnn_pipeline.py` and `scripts/gnn_visualize.py`.

## Input data format (`.npz`)

The GNN pipeline expects a NumPy `.npz` file with the following keys:

- `node_features`: `(T, N, F)` per-residue features per frame
- `positions`: `(T, N, 3)` per-residue coordinates (e.g., Cα)
- Optional edge masks: `contacts`, `hbonds`, `salts`, `covariance` with shape `(T, N, N)`
- Optional global features: `global_features`: `(T, G)`
- Optional labels: `delta_v`, `state`, `rmsd`, `rg`, `latent`
- Optional `state_classes`: number of state classes for classification

Use `scripts/gnn_example.py` to generate a synthetic example dataset.

## Graph construction

Graphs are built per frame using a configurable contact cutoff. Edge features can include:

- Contact mask values
- Hydrogen-bond and salt-bridge indicators (if provided)
- Covariance-based interaction scores (optional)
- Distance and direction vectors between residues

## Model architecture

The model combines spatial and temporal learning:

1. **Graph encoder**
   - SE(3)-aware message passing (distance + direction)
   - Multi-head graph attention layers
2. **Temporal encoder**
   - 1D convolutional stack over windows
   - Multi-head self-attention for temporal context
3. **Multi-task heads**
   - ΔV regression
   - State classification
   - RMSD and radius of gyration (Rg)
   - Latent projection for downstream analysis

## Training procedure

Training uses sliding windows of length `sequence_len`. Each window yields a single set of labels (typically the last frame in the window). The CLI exposes the key hyperparameters:

```bash
python scripts/gnn_pipeline.py \
  --npz path/to/residue_features.npz \
  --out out_gnn \
  --epochs 25 \
  --sequence 8 \
  --batch 4
```

The trained model is saved as `out_gnn/gamd_gnn_model.keras`.

## Saliency mapping

DeepMD supports several saliency mechanisms:

- **Gradient attribution:** ∂output/∂node features
- **Integrated gradients:** path-integrated gradients
- **Attention rollout:** GAT attention aggregation
- **GraphCAM:** gradient-weighted node embeddings

The default CLI computes gradient-based residue scores and writes:

- `importance.csv` (residue ID + importance)
- `importance.csv.json` (metadata)

Example summary report:

```bash
python scripts/gnn_visualize.py \
  --importance out_gnn/importance.csv \
  --out out_gnn/saliency_report.txt
```

## Recommended data sources

Residue-level node features can include:

- Dihedral angles, secondary structure labels
- Per-residue energy terms or ΔV statistics
- Structural observables (RMSF, contact counts)

Ensure that the `positions` array aligns with the chosen residue ordering.
