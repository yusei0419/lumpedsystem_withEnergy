"""
Two-bar bone remodeling model.

Free energy of bar i (i = 1, 2):
    psi_i = 1/2 * (1 - D_i) * E0 * eps_i^2 + 1/2 * lambda * b^2 * D_i^2

Generalized driving force for damage:
    Yd_i = -d psi_i / d D_i
         = 1/2 * E0 * eps_i^2 - lambda * b^2 * D_i

Damage flux / repair flux:
    Jd_i = alpha * 1/2 * E0 * eps_i^2          (strain-energy driven damage growth)
    Jr_i = alpha * lambda * b^2 * D_i           (target-remodeling driven repair)

Damage evolution:
    dD_i/dt = Jd_i - Jr_i = alpha * Yd_i

Background on the repair signal:
    rho_i = a * exp(-D_i)                       (osteocyte density)
    c_i   = -b * log(rho_i) = b*D_i - b*log(a)  (targeted_remodeling_activity)
    c0    = -b * log(a)
    => c_i - c0 = b*D_i
    => psi_repair_i = 1/2 * lambda * (c_i - c0)^2 = 1/2 * lambda * b^2 * D_i^2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Mode = Literal["series", "parallel"]
Control = Literal["stress", "displacement"]


@dataclass
class Params:
    """
    Stress-based parameterization.

    Loading is specified by a reference stress sigma0 that each bar experiences
    when undamaged (D = 0). The bars are circular cylinders of radius r, so
    A = pi * r^2. From this:

        F_series        = sigma0 * A              (each bar carries the same F)
        F_parallel_total = 2 * sigma0 * A          (two bars in parallel each see sigma0 at D=0)

    Units: SI-ish with stresses/moduli in MPa and lengths in mm.
        E0     in MPa
        sigma0 in MPa
        r, L   in mm
        A      in mm^2
        F      in N (= MPa * mm^2)
    """

    # Material / geometry
    E0: float = 2.0e5        # base Young modulus [MPa]
    r: float = 1.0           # cylinder radius [mm]
    L: float = 1.0           # length of each bar [mm]

    # Loading
    control: Control = "stress"
    sigma0: float = 20.0     # reference stress on each bar at D=0 [MPa]
    eps_total: float = 1.0e-4  # used only in displacement control

    # Repair / remodeling
    lam: float = 1.0e-2      # lambda: repair stiffness [MPa]
    b: float = 1.0           # signal sensitivity (c_i = b*D_i + const)
    alpha: float = 100.0     # kinetic coefficient for dD/dt [1/(time*MPa)]

    # Numerical safety
    D_max: float = 0.99
    D_min: float = 0.0
    E_min_ratio: float = 1e-4   # E_eff >= E_min_ratio * E0

    # Initial-condition grid
    D_grid: np.ndarray = field(default_factory=lambda: np.round(np.arange(0.0, 0.95, 0.1), 3))

    # ------- derived -------
    @property
    def A(self) -> float:
        """Cross-sectional area of a cylindrical bar: A = pi r^2 [mm^2]."""
        return math.pi * self.r ** 2

    @property
    def F_series(self) -> float:
        """Per-bar force in the series configuration: F = sigma0 * A [N]."""
        return self.sigma0 * self.A

    @property
    def F_parallel_total(self) -> float:
        """Total force in the parallel configuration: F = 2 * sigma0 * A [N]."""
        return 2.0 * self.sigma0 * self.A


def compute_effective_modulus(D: np.ndarray | float, E0: float, E_min_ratio: float = 1e-4) -> np.ndarray:
    """Return E_i = (1 - D_i) * E0 with a floor for stability."""
    E_eff = (1.0 - np.asarray(D)) * E0
    return np.maximum(E_eff, E_min_ratio * E0)


def compute_strains_series(D: np.ndarray, params: Params) -> np.ndarray:
    """
    Series: F1 = F2 = F_series, total displacement delta_total = delta_1 + delta_2.

    Stress control (per-bar reference sigma0 at D=0):
        F_series = sigma0 * A
        eps_i    = F_series / (E_i * A) = sigma0 / E_i = sigma0 / ((1-D_i) E0)

    Displacement control (total strain eps_total fixed, delta_total = 2 L eps_total):
        F_series = 2 eps_total * A / (1/E1 + 1/E2)
        eps_i    = F_series / (E_i * A)
    """
    # Effective stiffness of each bar after damage.
    E1 = compute_effective_modulus(D[0], params.E0, params.E_min_ratio)
    E2 = compute_effective_modulus(D[1], params.E0, params.E_min_ratio)
    A, L = params.A, params.L

    if params.control == "stress":
        F = params.F_series
    else:  # displacement
        delta_total = 2.0 * L * params.eps_total
        compliance = L / (E1 * A) + L / (E2 * A)
        F = delta_total / compliance

    # Convert force to bar-wise strains.
    eps1 = F / (E1 * A)
    eps2 = F / (E2 * A)
    return np.array([eps1, eps2])


def compute_strains_parallel(D: np.ndarray, params: Params) -> np.ndarray:
    """
    Parallel: eps1 = eps2 = eps_parallel, F_total = F1 + F2.

    Stress control: each bar should see stress sigma0 at D = 0, so the total
    force is the sum of the two reference forces:
        F_total = 2 * sigma0 * A
        eps     = F_total / ((E1 + E2) A)
                = 2 sigma0 / ((2 - D1 - D2) E0)

    Note this is *not* an arbitrary doubling — it is the natural consequence
    of specifying a per-bar stress sigma0 at the undamaged state.

    Displacement control:
        eps = eps_total
    """
    # Effective stiffness of each bar after damage.
    E1 = compute_effective_modulus(D[0], params.E0, params.E_min_ratio)
    E2 = compute_effective_modulus(D[1], params.E0, params.E_min_ratio)
    A = params.A

    if params.control == "stress":
        F_total = params.F_parallel_total
        eps = F_total / ((E1 + E2) * A)
    else:
        eps = params.eps_total

    return np.array([eps, eps])


def compute_fluxes(D: np.ndarray, strains: np.ndarray, params: Params) -> dict:
    """
    Returns Jd, Jr, Yd, dDdt for both bars.

        Jd_i   = alpha * 1/2 * E0 * eps_i^2
        Jr_i   = alpha * lambda * b^2 * D_i
        Yd_i   = 1/2 * E0 * eps_i^2 - lambda * b^2 * D_i
        dDdt_i = alpha * Yd_i = Jd_i - Jr_i
    """
    E0 = params.E0
    lam = params.lam
    b = params.b
    alpha = params.alpha

    # Damage/repair kinetics derived from the free-energy gradients.
    Jd = alpha * 0.5 * E0 * strains ** 2
    Jr = alpha * lam * b ** 2 * np.asarray(D)
    Yd = 0.5 * E0 * strains ** 2 - lam * b ** 2 * np.asarray(D)
    dDdt = Jd - Jr
    return {"Jd": Jd, "Jr": Jr, "Yd": Yd, "dDdt": dDdt}


def _strains(D: np.ndarray, params: Params, mode: Mode) -> np.ndarray:
    """Dispatch strain computation based on the bar configuration."""
    if mode == "series":
        return compute_strains_series(D, params)
    elif mode == "parallel":
        return compute_strains_parallel(D, params)
    raise ValueError(f"unknown mode: {mode}")


def ode_rhs(t: float, D: np.ndarray, params: Params, mode: Mode) -> np.ndarray:
    """RHS for solve_ivp. Damage evolution with one-sided clipping at boundaries."""
    D = np.asarray(D)
    strains = _strains(D, params, mode)
    fluxes = compute_fluxes(D, strains, params)
    dDdt = fluxes["dDdt"].copy()

    # One-sided saturation: do not push past [D_min, D_max].
    for i in range(2):
        if D[i] >= params.D_max and dDdt[i] > 0.0:
            dDdt[i] = 0.0
        if D[i] <= params.D_min and dDdt[i] < 0.0:
            dDdt[i] = 0.0
    return dDdt
