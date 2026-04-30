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

Mechanical conditions (force control is the default):

* **Series**, F1 = F2 = F0:
  `eps_i = F0 / ((1 - D_i) E0 A)`
* **Parallel**, F_total = 2 F0:
  `eps1 = eps2 = 2 F0 / ((2 - D1 - D2) E0 A)`

The factor 2 in the parallel total force makes the per-bar load match the
series case at D1 = D2 = 0, so the two phase portraits are directly comparable.

## Parameters (in `simulate_two_bar_remodeling.py::main`)

| Symbol | Code           | Default | Meaning                                   |
|--------|----------------|---------|-------------------------------------------|
| E0     | `Params.E0`    | 1.0     | base Young modulus                        |
| A      | `Params.A`     | 1.0     | cross-section                             |
| L      | `Params.L`     | 1.0     | bar length                                |
| b      | `Params.b`     | 1.0     | sensitivity in `c_i = b D_i + const`      |
| α      | `Params.alpha` | 1.0     | kinetic rate for `dD/dt`                  |
| F0     | `Params.F0`    | 0.4     | reference force; series uses F0, parallel uses 2 F0 |
| λ      | `Params.lam`   | swept   | repair stiffness; sweep `[0.1, 1.0, 10.0]`|
| D_max  | `Params.D_max` | 0.99    | hard cap to keep `(1-D) E0` from vanishing|
| D_min  | `Params.D_min` | 0.0     | lower bound on D                          |
| grid   | `Params.D_grid`| 0.0…0.9 (step 0.1) | initial-condition grid for D1, D2 |
| T      | `t_span`       | (0, 200)| integration window                        |

`solve_ivp` uses RK45, `rtol=1e-6`, `atol=1e-9`, `max_step=1.0`.
Damage is held at the boundaries by zeroing `dD_i/dt` whenever it would push
`D_i` past `[D_min, D_max]`.

## Expected behavior (sanity-check)

Equilibrium of a single bar in force control balances
`F0^2 / (2 lambda b^2 (1 - D))  =  D`, i.e. `D(1 - D) = F0^2 / (2 lambda)`.
With `F0 = 0.4`:

* λ = 0.1: `F0^2 / 2λ = 0.8 > 0.25` → **no real fixed point**, damage runs to D_max
* λ = 1.0: two roots, `D ≈ 0.087` (stable) and `D ≈ 0.913` (unstable) → **bistable**
* λ = 10.0: `D ≈ 0.008` (stable) → damage decays everywhere

Observed in the produced HTML:

| λ    | series end states                                | parallel end states |
|------|--------------------------------------------------|---------------------|
| 0.1  | all (0.99, 0.99)                                 | all (0.99, 0.99)    |
| 1.0  | localizes: e.g. D0=(0.2, 0.7) → (0.098, 0.99)    | homogenizes: (0.098, 0.098) |
| 10.0 | all (0.008, 0.008)                               | all (0.008, 0.008)  |

The bistable regime λ = 1 is where series and parallel differ qualitatively:
series amplifies asymmetry (load is shared equally → the damaged bar carries
the same force at lower stiffness → higher strain → more damage), while
parallel forces equal strain so D1 and D2 evolve toward the same basin.

## Extending

* **Different loading** — pass `control="displacement"` and set `eps_total` in
  `Params`, or change `F0`. The strain helpers in `two_bar_model.py` already
  branch on `params.control`.
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
