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

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Mode = Literal["series", "parallel"]
Control = Literal["force", "displacement"]


@dataclass
class Params:
    # Material / geometry
    E0: float = 1.0          # base Young modulus
    A: float = 1.0           # cross-section
    L: float = 1.0           # length of each bar

    # Repair / remodeling
    lam: float = 1.0         # lambda: repair stiffness
    b: float = 1.0           # signal sensitivity (c_i = b*D_i + const)
    alpha: float = 1.0       # kinetic coefficient for dD/dt

    # Loading
    control: Control = "force"
    F0: float = 0.4          # reference force; series uses F0, parallel uses 2*F0
    eps_total: float = 0.4   # used only in displacement control (series: 2L*eps_total, parallel: eps)

    # Numerical safety
    D_max: float = 0.99
    D_min: float = 0.0
    E_min_ratio: float = 1e-4   # E_eff >= E_min_ratio * E0

    # Initial-condition grid
    D_grid: np.ndarray = field(default_factory=lambda: np.round(np.arange(0.0, 0.95, 0.1), 3))


def compute_effective_modulus(D: np.ndarray | float, E0: float, E_min_ratio: float = 1e-4) -> np.ndarray:
    """E_i = (1 - D_i) * E0, floored at E_min_ratio * E0 for numerical stability."""
    E_eff = (1.0 - np.asarray(D)) * E0
    return np.maximum(E_eff, E_min_ratio * E0)


def compute_strains_series(D: np.ndarray, params: Params) -> np.ndarray:
    """
    Series: F1 = F2 = F_series, total displacement delta_total = delta_1 + delta_2.

    Force control:
        F_series = F0
        eps_i    = F0 / (E_i * A)

    Displacement control (total strain eps_total fixed, so delta_total = 2L*eps_total):
        F_series = (2 L eps_total) / (L/(E1 A) + L/(E2 A))
                 = 2 eps_total * A / (1/E1 + 1/E2)
        eps_i    = F_series / (E_i * A)
    """
    E1 = compute_effective_modulus(D[0], params.E0, params.E_min_ratio)
    E2 = compute_effective_modulus(D[1], params.E0, params.E_min_ratio)
    A, L = params.A, params.L

    if params.control == "force":
        F = params.F0
    else:  # displacement
        delta_total = 2.0 * L * params.eps_total
        compliance = L / (E1 * A) + L / (E2 * A)
        F = delta_total / compliance

    eps1 = F / (E1 * A)
    eps2 = F / (E2 * A)
    return np.array([eps1, eps2])


def compute_strains_parallel(D: np.ndarray, params: Params) -> np.ndarray:
    """
    Parallel: eps1 = eps2 = eps_parallel, F_total = F1 + F2.

    To make the parallel case directly comparable to the series case
    (same per-bar load when D1 = D2 = 0), the total force is set to 2*F0.

    Force control:
        F_total = 2 * F0
        eps     = F_total / ((E1 + E2) * A)

    Displacement control:
        eps = eps_total
    """
    E1 = compute_effective_modulus(D[0], params.E0, params.E_min_ratio)
    E2 = compute_effective_modulus(D[1], params.E0, params.E_min_ratio)
    A = params.A

    if params.control == "force":
        F_total = 2.0 * params.F0
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

    Jd = alpha * 0.5 * E0 * strains ** 2
    Jr = alpha * lam * b ** 2 * np.asarray(D)
    Yd = 0.5 * E0 * strains ** 2 - lam * b ** 2 * np.asarray(D)
    dDdt = Jd - Jr
    return {"Jd": Jd, "Jr": Jr, "Yd": Yd, "dDdt": dDdt}


def _strains(D: np.ndarray, params: Params, mode: Mode) -> np.ndarray:
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

    # one-sided saturation: do not push past [D_min, D_max]
    for i in range(2):
        if D[i] >= params.D_max and dDdt[i] > 0.0:
            dDdt[i] = 0.0
        if D[i] <= params.D_min and dDdt[i] < 0.0:
            dDdt[i] = 0.0
    return dDdt
