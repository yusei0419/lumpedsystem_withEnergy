"""
Driver: integrate the two-bar bone-remodeling model on a grid of (D1_0, D2_0)
initial conditions for several values of lambda, for both series and parallel
configurations. Saves results to outputs/simulation_results.json and renders
an interactive Plotly HTML at outputs/two_bar_remodeling_phase.html.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from two_bar_model import (
    Mode,
    Params,
    _strains,
    compute_fluxes,
    ode_rhs,
)
from plot_results import build_phase_html


def run_simulation_for_initial_condition(
    D0: np.ndarray,
    params: Params,
    mode: Mode,
    t_span=(0.0, 200.0),
    n_eval: int = 201,
) -> dict[str, np.ndarray]:
    """Integrate one (D1, D2) trajectory with solve_ivp."""
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)

    sol = solve_ivp(
        fun=lambda t, y: ode_rhs(t, y, params, mode),
        t_span=t_span,
        y0=np.asarray(D0, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
        max_step=1.0,
    )

    # Clip for plotting only; the solver uses the unclipped state.
    D_traj = np.clip(sol.y, params.D_min, params.D_max)
    t = sol.t

    # Recompute strains/fluxes along the trajectory for diagnostics.
    strains = np.zeros_like(D_traj)
    Jd = np.zeros_like(D_traj)
    Jr = np.zeros_like(D_traj)
    for k in range(D_traj.shape[1]):
        eps = _strains(D_traj[:, k], params, mode)
        f = compute_fluxes(D_traj[:, k], eps, params)
        strains[:, k] = eps
        Jd[:, k] = f["Jd"]
        Jr[:, k] = f["Jr"]

    return {
        "t": t,
        "D": D_traj,
        "eps": strains,
        "Jd": Jd,
        "Jr": Jr,
        "success": bool(sol.success),
    }


def run_grid_simulations(
    params: Params,
    mode: Mode,
    t_span=(0.0, 200.0),
    n_eval: int = 201,
) -> list[dict[str, Any]]:
    """Sweep all (D1_0, D2_0) pairs on the grid; return one record per IC."""
    results = []
    for D1_0 in params.D_grid:
        for D2_0 in params.D_grid:
            D0 = np.array([D1_0, D2_0], dtype=float)
            traj = run_simulation_for_initial_condition(D0, params, mode, t_span, n_eval)
            results.append({
                "D0": D0.tolist(),
                "t": traj["t"].tolist(),
                "D1": traj["D"][0].tolist(),
                "D2": traj["D"][1].tolist(),
                "eps1": traj["eps"][0].tolist(),
                "eps2": traj["eps"][1].tolist(),
                "Jd1": traj["Jd"][0].tolist(),
                "Jd2": traj["Jd"][1].tolist(),
                "Jr1": traj["Jr"][0].tolist(),
                "Jr2": traj["Jr"][1].tolist(),
            })
    return results


def main():
    """Run the grid sweeps for several lambda values and save outputs."""
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Base parameters shared across lambda values.
    base_params = Params(
        E0=1.0, A=1.0, L=1.0,
        b=1.0, alpha=1.0,
        control="force", F0=0.4,
        D_max=0.99, D_min=0.0,
        D_grid=np.round(np.arange(0.0, 0.95, 0.1), 3),
    )

    # Sweep repair stiffness values.
    lambda_values = [0.1, 1.0, 10.0]
    t_span = (0.0, 200.0)
    n_eval = 201

    # Aggregate results keyed by mode and lambda.
    all_results: dict[str, dict[str, list]] = {"series": {}, "parallel": {}}
    for lam in lambda_values:
        params = Params(**{**asdict(base_params), "lam": lam})
        # asdict converts D_grid to list; restore as ndarray
        params.D_grid = np.asarray(params.D_grid)

        for mode in ("series", "parallel"):
            print(f"[run] mode={mode:8s}  lambda={lam}")
            recs = run_grid_simulations(params, mode, t_span, n_eval)
            all_results[mode][f"{lam}"] = recs

    # Store metadata + trajectories for post-processing.
    meta = {
        "params": {
            "E0": base_params.E0, "A": base_params.A, "L": base_params.L,
            "b": base_params.b, "alpha": base_params.alpha,
            "control": base_params.control, "F0": base_params.F0,
            "D_max": base_params.D_max, "D_min": base_params.D_min,
            "D_grid": base_params.D_grid.tolist(),
            "t_span": list(t_span),
        },
        "lambda_values": lambda_values,
        "results": all_results,
    }

    # Save JSON and render the interactive phase diagram.
    json_path = os.path.join(out_dir, "simulation_results.json")
    with open(json_path, "w") as f:
        json.dump(meta, f)
    print(f"[save] {json_path}")

    html_path = os.path.join(out_dir, "two_bar_remodeling_phase.html")
    build_phase_html(meta, html_path)
    print(f"[save] {html_path}")


if __name__ == "__main__":
    main()
