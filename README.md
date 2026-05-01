# Two-bar Bone Remodeling Simulation

Phase-portrait simulation of a two-bar lumped system with continuum-damage
mechanics and target-remodeling repair, comparing **series** and **parallel**
configurations.

## Files

```
.
├── two_bar_model.py             # Params, energies, fluxes, ODE RHS
├── simulate_two_bar_remodeling.py  # solve_ivp driver, grid sweep, JSON+HTML output
├── plot_results.py              # Plotly phase-diagram renderer
├── outputs/
│   ├── two_bar_remodeling_phase.html   # interactive phase portrait
│   └── simulation_results.json
└── README.md
```

## Run

```bash
pip install numpy scipy plotly
python3 simulate_two_bar_remodeling.py
```

This regenerates `outputs/two_bar_remodeling_phase.html`. Open it in a browser.
The slider / button row at the top of the figure switches between λ values; the
two side-by-side panels show series (left) and parallel (right) phase portraits
in the (D1, D2) plane. White circles = initial points, black lines = trajectories,
green squares = end states.

## Equations (mirrored in code comments)

Free energy of bar i:
```
psi_i = 1/2 (1 - D_i) E0 eps_i^2  +  1/2 lambda b^2 D_i^2
```
The repair part comes from
`rho_i = a exp(-D_i)`, `c_i = -b log(rho_i)`, `c0 = -b log(a)`,
so `c_i - c0 = b D_i` and `psi_repair_i = 1/2 lambda (c_i - c0)^2 = 1/2 lambda b^2 D_i^2`.
The variable `c_i` is named `targeted_remodeling_activity` in the discussion;
in code it is implicit because the energy is written directly in `D_i`.

Driving force and fluxes:
```
Yd_i  = -d psi_i / d D_i = 1/2 E0 eps_i^2 - lambda b^2 D_i
Jd_i  = alpha * 1/2 E0 eps_i^2          (damage growth)
Jr_i  = alpha * lambda b^2 D_i           (repair / suppression)
dD_i/dt = Jd_i - Jr_i = alpha Yd_i
```

Mechanical conditions (stress control is the default):

The bars are circular cylinders of radius `r`, so `A = π r²`. Loading is
specified by a per-bar reference stress `σ0` that each bar would experience
if undamaged:

* **Series**, F1 = F2 = σ0 · A:
  `eps_i = σ0 / ((1 - D_i) E0)`
* **Parallel**, F_total = 2 · σ0 · A:
  `eps1 = eps2 = 2 σ0 / ((2 - D1 - D2) E0)`

The factor 2 in the parallel total force is *not* arbitrary — it is the
direct consequence of requiring per-bar stress = σ0 at the undamaged state
in both configurations. This makes the two phase portraits directly comparable.

## Parameters (in `simulate_two_bar_remodeling.py::main`)

| Symbol | Code             | Default     | Units | Meaning                                   |
|--------|------------------|-------------|-------|-------------------------------------------|
| E0     | `Params.E0`      | 2.0e5       | MPa   | undamaged Young modulus                   |
| r      | `Params.r`       | 1.0         | mm    | cylinder radius                           |
| A      | `Params.A`       | π·r² = π    | mm²   | cross-section (derived)                   |
| L      | `Params.L`       | 1.0         | mm    | bar length                                |
| σ0     | `Params.sigma0`  | 20.0        | MPa   | reference per-bar stress at D=0           |
| F_series | derived        | σ0·A ≈ 62.83| N     | per-bar force in series                   |
| F_parallel_total | derived| 2·σ0·A ≈ 125.66 | N | total force in parallel                  |
| b      | `Params.b`       | 1.0         | -     | sensitivity in `c_i = b D_i + const`      |
| α      | `Params.alpha`   | 100.0       | 1/(t·MPa) | kinetic rate for `dD/dt`              |
| λ      | `Params.lam`     | swept       | MPa   | repair stiffness; sweep `[1e-3, 1e-2, 1e-1]` |
| D_max  | `Params.D_max`   | 0.99        | -     | hard cap to keep `(1-D) E0` from vanishing|
| D_min  | `Params.D_min`   | 0.0         | -     | lower bound on D                          |
| grid   | `Params.D_grid`  | 0.0…0.9 (step 0.1) | - | initial-condition grid for D1, D2     |
| T      | `t_span`         | (0, 200)    | t     | integration window                        |

`solve_ivp` uses RK45, `rtol=1e-6`, `atol=1e-9`, `max_step=1.0`.
Damage is held at the boundaries by zeroing `dD_i/dt` whenever it would push
`D_i` past `[D_min, D_max]`.

The energy scale at the undamaged state is
`ψ_mech_0 = ½ E0 ε0² = σ0² / (2 E0) = 1e-3 MPa`. The choice
`α = 100` makes the natural damage growth rate `α · ψ_mech_0 ≈ 0.1` per time
unit, so D evolves on O(10) time units and T = 200 is comfortable.

## Expected behavior (sanity-check)

Single-bar equilibrium under stress control:
`Jd = α σ0² / (2 (1−D)² E0)`, `Jr = α λ b² D`, so

```
D (1 − D)² = σ0² / (2 E0 λ b²)  =  ψ_mech_0 / λ
```

(Note the `(1−D)²` in stress control; under force control with fixed F it
would be `D(1−D)` — the difference is that ε itself depends on D.) The
function `D(1−D)²` peaks at `D = 1/3` with value `4/27 ≈ 0.148`, so a real
fixed point exists only when `ψ_mech_0 / λ ≤ 4/27`, i.e. `λ ≥ 6.75e−3`.
With `σ0 = 20 MPa`, `E0 = 2e5 MPa`, `b = 1`:

| λ [MPa] | `ψ_mech_0/λ` | predicted fixed point(s) | observed end-state |
|---------|--------------|---------------------------|--------------------|
| 1e-3    | 1.0          | none → runaway            | all reach D_max = 0.99 |
| 1e-2    | 0.1          | D ≈ 0.133 (stable), D ≈ 0.781 (unstable) | bistable, see below |
| 1e-1    | 0.01         | D ≈ 0.0102 (stable), D ≈ 0.9395 (unstable) | bistable, mostly low |

Observed in the produced HTML at λ = 1e-2 (bistable regime):

| IC (D1₀, D2₀) | series end | parallel end |
|---------------|------------|--------------|
| (0.0, 0.0)    | (0.133, 0.133) | (0.133, 0.133) |
| (0.4, 0.4)    | (0.133, 0.133) | (0.133, 0.133) |
| (0.2, 0.7)    | **(0.133, 0.99)** — localizes | (0.133, 0.133) — homogenizes |
| (0.3, 0.6)    | **(0.133, 0.99)** — localizes | (0.133, 0.133) — homogenizes |

The bistable regime is where series and parallel differ qualitatively:
in series under stress control the more-damaged bar still carries the
same force `F = σ0 A` but at lower stiffness `(1−D) E0`, so its strain
`σ0 / ((1−D) E0)` is larger than the other bar's — driving runaway damage
on that bar while the partner can heal. In parallel both bars share a
common strain, so `Jd` is the same for both and the asymmetry decays.

## Extending

* **Different loading** — pass `control="displacement"` and set `eps_total` in
  `Params`, or change `sigma0` (per-bar reference stress) / `r` (radius). The
  strain helpers in `two_bar_model.py` already branch on `params.control`.
* **More λ values** — extend `lambda_values` in `main`; the slider/buttons
  pick them up automatically.
* **Different IC grid / time window** — change `Params.D_grid`, `t_span`,
  `n_eval` in `main`.
* **Different repair signal** — replace `Jr_i = alpha lambda b^2 D_i` in
  `compute_fluxes`. To use a non-linear `c_i(D_i)`, redefine the energy and
  retake `Yd_i = -d psi/d D` analytically before editing the function.
* **More than two bars** — generalize `Params.D_grid`, `compute_strains_*`,
  and the ODE state vector. The series/parallel formulae extend to N bars
  (compliance sum or stiffness sum respectively).
* **Time-series plots** — `simulation_results.json` already contains
  `t, D1, D2, eps1, eps2, Jd1, Jd2, Jr1, Jr2` per IC; extend `plot_results.py`
  to add additional subplots driven by an IC selector.
