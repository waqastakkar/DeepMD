"""
Minimal GaMD smoke test (50k steps) to validate dihedral energy + boost.

Example:
  python scripts/gamd_smoke_test.py --parm topology/complex.parm7 --crd topology/complex.rst7 --outdir out_smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

from openmm import unit

from paddle.core.engine import EngineOptions, create_simulation
from paddle.core.integrators import make_conventional, make_dual_prod
from paddle.core.params import BoostParams
from paddle.stages.equil_prod import _attach_integrator, _estimate_bounds, _make_gamd_diagnostic_sampler
from paddle.io.report import CSVLogger, ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal GaMD smoke test (50k steps)")
    ap.add_argument("--parm", required=True, help="Amber parm7")
    ap.add_argument("--crd", required=True, help="Amber rst7/inpcrd")
    ap.add_argument("--outdir", default="out_smoke")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--report", type=int, default=1_000)
    ap.add_argument("--sim-type", default="protein.explicit")
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--k0", type=float, default=0.5)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    opts = EngineOptions(
        sim_type=args.sim_type,
        platform_name=args.platform,
        precision=args.precision,
        cuda_precision=args.precision,
        add_barostat="explicit" in args.sim_type,
        barostat_temperature_kelvin=300.0,
    )

    integ = make_conventional(dt_ps=0.002, temperature_K=300.0, collision_rate_ps=1.0)
    sim = create_simulation(args.parm, args.crd, integ, opts)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin)

    vmin_d, vmax_d, vmin_p, vmax_p, vavg_d, vavg_p, vstd_d, vstd_p, *_ = _estimate_bounds(
        sim, steps=5_000, interval=max(10, args.report)
    )
    params = BoostParams(
        VminD=vmin_d,
        VmaxD=vmax_d,
        VminP=vmin_p,
        VmaxP=vmax_p,
        k0D=args.k0,
        k0P=args.k0,
    )
    gamd = make_dual_prod(dt_ps=0.002, temperature_K=300.0, params=params)
    _attach_integrator(sim, gamd)

    diag_logger = CSVLogger(outdir / "gamd-diagnostics.csv", [
        "step",
        "E_potential_kJ",
        "E_bond_kJ",
        "E_angle_kJ",
        "E_dihedral_kJ",
        "E_nonbonded_kJ",
        "Temperature_K",
        "Density_g_per_ml",
        "DihedralRef_kJ",
        "TotalRef_kJ",
        "Dihedral_k",
        "Total_k",
        "DihedralBoost_kJ",
        "TotalBoost_kJ",
        "DeltaV_kJ",
        "BoostedDihedral_kJ",
        "BoostedTotal_kJ",
        "VminD_kJ",
        "VmaxD_kJ",
        "VminP_kJ",
        "VmaxP_kJ",
        "Dihedral_mean_kJ",
        "Dihedral_std_kJ",
        "Dihedral_min_kJ",
        "Dihedral_max_kJ",
        "Dihedral_skewness",
        "Dihedral_excess_kurtosis",
        "Dihedral_tail_risk",
        "Total_mean_kJ",
        "Total_std_kJ",
        "Total_min_kJ",
        "Total_max_kJ",
        "Total_skewness",
        "Total_excess_kurtosis",
        "Total_tail_risk",
        "DeltaV_mean_kJ",
        "DeltaV_std_kJ",
        "DeltaV_min_kJ",
        "DeltaV_max_kJ",
        "DeltaV_skewness",
        "DeltaV_excess_kurtosis",
        "DeltaV_tail_risk",
    ])
    ml_logger = CSVLogger(outdir / "gamd-ml.csv", [
        "step",
        "E_potential",
        "E_bond",
        "E_angle",
        "E_dihedral",
        "E_nonbonded",
        "Temperature",
        "Density",
        "GaMD_E_threshold",
        "GaMD_k",
        "GaMD_dV",
        "GaMD_V_star",
        "GaMD_V_avg",
        "GaMD_V_std",
        "GaMD_skew",
        "GaMD_kurtosis",
        "GaMD_tail_risk",
    ])
    sample_fn = _make_gamd_diagnostic_sampler(sim, gamd, diag_logger, ml_logger=ml_logger)

    steps = int(args.steps)
    interval = max(1, int(args.report))
    blocks = steps // interval
    for _ in range(blocks):
        sim.step(interval)
        sample_fn()
    remainder = steps % interval
    if remainder:
        sim.step(remainder)
        sample_fn()

    E_dihedral = sim.context.getState(getEnergy=True, groups={3}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    deltaV = float(gamd.getGlobalVariableByName("DihedralBoostPotential")) + float(
        gamd.getGlobalVariableByName("TotalBoostPotential")
    )
    print(f"[SMOKE] E_dihedral={E_dihedral:.3f} kJ/mol")
    print(f"[SMOKE] DeltaV={deltaV:.3f} kJ/mol")


if __name__ == "__main__":
    main()
