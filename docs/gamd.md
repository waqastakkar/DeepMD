# GaMD Theory and Implementation Notes

Gaussian accelerated MD (GaMD) enhances conformational sampling by adding a harmonic boost potential whenever the system’s potential energy falls below a threshold. DeepMD implements dual-boost GaMD using custom OpenMM integrators with explicit tracking of dihedral and total potential energies.

## Boost potential

For a potential energy **V** and threshold **E**, the GaMD boost is defined as:

- **ΔV = ½ k (E − V)^2** when **V < E**
- **ΔV = 0** when **V ≥ E**

The corresponding force scaling factor is:

- **1 − k (E − V)/(Vmax − Vmin)** for **V < E**

This yields a smooth, continuous modification of forces that preserves the overall dynamics while reducing energy barriers.

## Dual boost (dihedral + total)

DeepMD applies two boosts simultaneously:

1. **Dihedral boost** on the dihedral (torsion) energy component
2. **Total boost** on the total potential energy

Both boosts are implemented in custom integrators and combined multiplicatively in the force scaling of the dihedral term. This is the recommended GaMD configuration for proteins and is the default in the pipeline.

## Threshold energy and force constant

For each boost component, DeepMD estimates running minimum/maximum energies (**Vmin**, **Vmax**) and uses a dimensionless parameter **k0** to define the effective force constant:

- **E = Vmin + (Vmax − Vmin)/k0**
- **k = k0 / (Vmax − Vmin)** (implemented via scaled force factors)

The configuration parameters `k0_initial`, `k0_min`, `k0_max`, `refED_factor`, and `refEP_factor` define bounds and safety clamps for stable GaMD operation.

## Reweighting and diagnostics

GaMD boost potentials are designed to be near-Gaussian, allowing unbiased ensemble averages via cumulant expansion. DeepMD evaluates:

- ΔV mean, standard deviation, skewness, excess kurtosis
- Tail risk metrics
- Effective sample size and entropy-based reweighting diagnostics

These diagnostics are written to `bias_plan_cycle_*.json` and `metrics.jsonl`, and can be visualized with the plotting scripts in `scripts/`.

## Practical guidance

- **Explicit solvent:** combine GaMD with barostatted NPT density equilibration before GaMD cycles.
- **Monitoring:** inspect `gamd-diagnostics-cycle*.csv` to ensure ΔV distributions remain approximately Gaussian.
- **Stability:** if ΔV skewness or excess kurtosis grows, consider lowering `k0_max` or adjusting `refE*_factor` bounds.

For configuration details, see `docs/config.md`.
