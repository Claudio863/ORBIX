"""Modelo kepleriano de velocidad radial en numpy (misma formulación que los .stan)."""
from __future__ import annotations

import numpy as np


def kepler(M, e, tol: float = 1.234e-10, max_iter: int = 200) -> np.ndarray:
    """Resuelve E - e sin E = M por Newton, vectorizado (criterio de kepler_eq.hpp)."""
    M = np.asarray(M, dtype=float)
    E = M.copy()
    for _ in range(max_iter):
        En = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        with np.errstate(divide="ignore", invalid="ignore"):
            done = np.all(np.abs((En - E) / np.where(En == 0, 1.0, En)) <= tol)
        E = En
        if done:
            break
    return E


def velocidad_radial(t, T, P, e, w_rad, V0, K, signo: float = 1.0) -> np.ndarray:
    """V(t) = V0 + signo·K·[cos(w + nu) + e cos w], con T = época del periastro."""
    t = np.asarray(t, dtype=float)
    E = kepler(2 * np.pi * (t - T) / P, e)
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    return V0 + signo * K * (np.cos(w_rad + nu) + e * np.cos(w_rad))


def curvas(params: dict, t, tipo: str) -> dict:
    """Curvas del modelo para un conjunto de parámetros (claves P, T, e, w_deg, V0, K | K1, K2)."""
    w = np.deg2rad(params["w_deg"])
    if tipo == "SB1":
        return {"v1": velocidad_radial(t, params["T"], params["P"], params["e"], w, params["V0"], params["K"], 1.0)}
    return {"v1": velocidad_radial(t, params["T"], params["P"], params["e"], w, params["V0"], params["K1"], 1.0),
            "v2": velocidad_radial(t, params["T"], params["P"], params["e"], w, params["V0"], params["K2"], -1.0)}


def fase(t, P, t_ref) -> np.ndarray:
    """Fase orbital en [0, 1) respecto de t_ref."""
    return np.mod((np.asarray(t, dtype=float) - t_ref) / P, 1.0)
