"""Búsqueda global del período para sembrar L-BFGS (Fase 1).

Reemplaza la siembra en ±5 % alrededor del único pico de Lomb-Scargle, que con e alta cae en alias.

Periodograma kepleriano
-----------------------
Para cada período de prueba P (grilla uniforme en frecuencia dentro de la ventana del sistema) y cada
par (e, fase del periastro φ) de una grilla gruesa, la curva de cada componente es LINEAL en el resto:

    v_c(t) = γ_c + A_c cos ν(t) + B_c sin ν(t)     (A = K cos ω, B = −K sin ω para v1; signo opuesto en v2)

así que γ, A, B salen por mínimos cuadrados ponderados (1/err²) en forma cerrada. La potencia es

    potencia(P) = 1 − χ²_min(P) / χ²_0          (χ²_0: solo constantes γ_c)

con χ²_min el mínimo sobre la grilla (e, φ). Con e = 0 se reduce al Lomb-Scargle flotante; con e > 0 captura
los picos de periastro que el modelo sinusoidal no ve. Para SB2 se calcula sobre v1, v2 y la combinación
(χ² sumado, mismo (P, e, φ) y (γ, A, B) propios por componente).

Siembra
-------
Los arranques de L-BFGS se reparten entre (a) los K picos más altos del periodograma (dispersión local
del orden de una fracción del ancho del pico) y (b) una grilla log-uniforme estratificada sobre toda la
ventana [Pmin, Pmax]. Nada depende de la literatura: ni el período, ni e, ni la fase.
"""
from __future__ import annotations

def refinar_pico(comps: dict, t_dias: dict, P_d: float, t0_d: float,
                 base_d: float, Pmin_d: float, Pmax_d: float,
                 n_frecuencias: int = 41) -> dict:
    """Refina una vecindad de frecuencia con RV, sin utilizar literatura.

    La grilla inicial puede ubicar un pico sin resolver los submodos estrechos
    que producen datos de alta precision. Se reevalua fase/e en 41 frecuencias
    de un ancho de Rayleigh y se ajustan los parametros fisicos conjuntamente.
    Solo produce una semilla: la inferencia final sigue siendo Stan.
    """
    from scipy.optimize import least_squares
    t, series = _series(comps, t_dias)
    datos = list(comps.values())
    y = np.concatenate([d.RV.to_numpy(float) for d in datos])
    err = np.concatenate([d.err.to_numpy(float) for d in datos])
    tref = float(np.average(t, weights=1/err**2))
    rango = float(np.ptp(y))
    vmin, vmax = float(y.min()-rango), float(y.max()+rango)
    limites_k = [max(2*float(np.ptp(d.RV)), 1e-6) for d in datos]
    cortes = np.cumsum([0]+[len(d) for d in datos])
    f0 = 1.0/P_d
    flo = max(1.0/Pmax_d, f0-.5/base_d)
    fhi = min(1.0/Pmin_d, f0+.5/base_d)
    mejor = None
    intentos = []
    for frecuencia in np.linspace(flo, fhi, n_frecuencias):
        periodo_d = 1.0/frecuencia
        semilla = solucion_en_periodo(comps,t_dias,periodo_d,t0_d)
        mref = float(2*np.pi*((tref-t0_d)/periodo_d-semilla["T0"])%(2*np.pi))
        x0 = [np.log(periodo_d),mref,semilla["e"],semilla["w"],semilla["V0"]]
        x0 += [np.log(max(semilla[f"K_{c}"],1e-8)) for c in comps]
        inferior = [np.log(1/fhi),mref-2*np.pi,0,-np.inf,vmin]+[np.log(1e-9)]*len(datos)
        superior = [np.log(1/flo),mref+2*np.pi,.99,np.inf,vmax]+list(np.log(limites_k))
        x0 = np.maximum(np.minimum(x0,np.array(superior)-1e-10),np.array(inferior)+1e-10)
        def residuos(x):
            M=np.remainder(2*np.pi*(t-tref)/np.exp(x[0])+x[1]+np.pi,2*np.pi)-np.pi
            E=_kepler(M,x[2],n_iter=40)
            nu=2*np.arctan2(np.sqrt(1+x[2])*np.sin(E/2),np.sqrt(1-x[2])*np.cos(E/2))
            forma=np.cos(x[3]+nu)+x[2]*np.cos(x[3])
            modelo=np.full_like(y,x[4])
            for j in range(len(datos)):
                sl=slice(cortes[j],cortes[j+1])
                modelo[sl]+=(1 if j==0 else -1)*np.exp(x[5+j])*forma[sl]
            return (modelo-y)/err
        try:
            fit=least_squares(residuos,x0,bounds=(inferior,superior),x_scale="jac",
                              max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            chi2=float(np.sum(fit.fun**2))
            finito=bool(np.isfinite(chi2) and np.all(np.isfinite(fit.x)))
            intentos.append(dict(P_semilla_d=periodo_d,chi2=chi2,nfev=fit.nfev,
                                 exito=bool(fit.success),finito=finito))
            if finito and (mejor is None or chi2<mejor["chi2"]):
                x=fit.x
                mejor=dict(P_d=float(np.exp(x[0])),e=float(x[2]),
                           T0=float(((tref-t0_d)/np.exp(x[0])-x[1]/(2*np.pi))%1),
                           w=float(x[3]%(2*np.pi)),V0=float(x[4]),chi2=chi2,
                           **{f"K_{c}":float(np.exp(x[5+j])) for j,c in enumerate(comps)})
        except (ValueError,FloatingPointError) as exc:
            intentos.append(dict(P_semilla_d=periodo_d,exito=False,error=str(exc)[:200]))
    if mejor is None:
        raise RuntimeError("Ningun refinamiento finito para el pico de periodo")
    mejor["intentos"]=intentos
    return mejor


import numpy as np
import pandas as pd

E_GRID = (0.0, 0.3, 0.5, 0.65, 0.75, 0.85, 0.92)
N_FASE = 40


def _kepler(M, e, n_iter: int = 14):
    """E − e sin E = M por Newton con arranque de Danby (vectorizado; suficiente para la grilla)."""
    E = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(n_iter):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def _chi2_lineal(nu, y, w):
    """χ² mínimo y coeficientes (γ, A, B) de y ≈ γ + A cos ν + B sin ν, por WLS. nu: (..., N)."""
    X = np.stack([np.ones_like(nu), np.cos(nu), np.sin(nu)], -1)          # (..., N, 3)
    XtW = X * w[:, None]
    M = np.einsum("...ni,...nj->...ij", XtW, X)
    b = np.einsum("...ni,n->...i", XtW, y)
    coef = np.linalg.solve(M + 1e-12 * np.eye(3), b[..., None])[..., 0]
    return np.sum(w * y * y) - np.sum(coef * b, -1), coef


def _ajuste_grilla(t, comps, f, t0, e_grid, n_fase):
    """Para frecuencias f (nf,), devuelve χ² (nf, ne, nφ) por componente y sus coeficientes."""
    e = np.asarray(e_grid)[None, :, None, None]
    fase = (np.arange(n_fase) / n_fase)[None, None, :, None]
    M = np.mod(2 * np.pi * (f[:, None, None, None] * (t - t0)[None, None, None, :] - fase), 2 * np.pi)
    E = _kepler(M, e)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    out = {}
    for c, (sl, y, w) in comps.items():
        out[c] = _chi2_lineal(nu[..., sl], y, w)
    return out


def _series(comps: dict, t_dias: dict):
    """Concatena las componentes; devuelve t total y {c: (slice, y, w)}."""
    t, d, i = [], {}, 0
    for c, df in comps.items():
        n = len(df)
        t.append(np.asarray(t_dias[c], float))
        d[c] = (slice(i, i + n), df.RV.to_numpy(float), 1.0 / df.err.to_numpy(float) ** 2)
        i += n
    return np.concatenate(t), d


def periodograma_kepleriano(comps: dict, t_dias: dict, Pmin_d: float, Pmax_d: float,
                            muestras_por_pico: int = 20, e_grid=E_GRID, n_fase: int = N_FASE,
                            bloque: int = 64) -> dict:
    """Periodograma kepleriano en días para cada componente y, si hay varias, su combinación.

    Devuelve {"tabla": DataFrame(P_d, frecuencia, potencia_<c>, e_<c>, ..., potencia_comb, e_comb),
              "t0_d": época de referencia de las fases (mínimo de todas las épocas), "base_d": línea de base}.
    """
    t, series = _series(comps, t_dias)
    t0, base = float(t.min()), float(t.max() - t.min())
    f = np.arange(1.0 / Pmax_d, 1.0 / Pmin_d, 1.0 / base / muestras_por_pico)
    chi0 = {c: float(np.sum(w * (y - np.sum(w * y) / np.sum(w)) ** 2)) for c, (_, y, w) in series.items()}
    nombres = list(series) + (["comb"] if len(series) > 1 else [])
    res = {c: (np.empty(len(f)), np.empty(len(f))) for c in nombres}
    eg = np.asarray(e_grid)
    for i in range(0, len(f), bloque):
        fb = f[i:i + bloque]
        out = _ajuste_grilla(t, series, fb, t0, e_grid, n_fase)
        chis = {c: v[0] for c, v in out.items()}
        if len(series) > 1:
            chis["comb"] = sum(chis[c] for c in series)
        for c in nombres:
            x = chis[c].reshape(len(fb), -1)
            j = np.argmin(x, 1)
            res[c][0][i:i + bloque] = x[np.arange(len(fb)), j]
            res[c][1][i:i + bloque] = eg[j // n_fase]
    chi0["comb"] = sum(chi0[c] for c in series)
    tabla = pd.DataFrame({"P_d": 1.0 / f, "frecuencia": f})
    for c in nombres:
        tabla[f"potencia_{c}"] = 1.0 - res[c][0] / chi0[c]
        tabla[f"e_{c}"] = res[c][1]
    return dict(tabla=tabla.sort_values("P_d").reset_index(drop=True), t0_d=t0, base_d=base)


def picos(tabla: pd.DataFrame, base_d: float, columna: str, k: int = 8) -> pd.DataFrame:
    """K picos más altos (máximos locales) separados al menos por 1/base en frecuencia."""
    tb = tabla.sort_values("frecuencia").reset_index(drop=True)
    p = tb[columna].to_numpy()
    loc = np.where((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1
    loc = loc[np.argsort(p[loc])[::-1]]
    elegidos = []
    for j in loc:
        if all(abs(tb.frecuencia[j] - tb.frecuencia[i]) > 1.0 / base_d for i in elegidos):
            elegidos.append(j)
        if len(elegidos) == k:
            break
    return tb.loc[elegidos].assign(rango=np.arange(1, len(elegidos) + 1), criterio=columna).reset_index(drop=True)


def picos_combinados(per: dict, k: int = 8) -> pd.DataFrame:
    """Picos para sembrar: de la combinación (SB2) o de v1 (SB1); se agregan los mejores de cada componente
    que no estén ya, para no perder un período que solo se vea en una serie (p. ej. v2 con pocos puntos)."""
    tb, base = per["tabla"], per["base_d"]
    cols = [c for c in tb.columns if c.startswith("potencia_")]
    principal = "potencia_comb" if "potencia_comb" in cols else "potencia_v1"
    sel = picos(tb, base, principal, k)
    for c in cols:
        if c == principal:
            continue
        extra = picos(tb, base, c, 2)
        for _, r in extra.iterrows():
            if np.all(np.abs(sel.frecuencia - r.frecuencia) > 1.0 / base):
                sel = pd.concat([sel, r.to_frame().T], ignore_index=True)
    sel["P_d"] = sel["P_d"].astype(float)
    sel["frecuencia"] = sel["frecuencia"].astype(float)
    return sel.reset_index(drop=True)


def solucion_en_periodo(comps: dict, t_dias: dict, P_d: float, t0_d: float,
                        e_grid=E_GRID, n_fase: int = N_FASE) -> dict:
    """Mejor (e, φ, ω, K_c) de la grilla para un período dado (sirve para sembrar fase y ω de los picos).
    φ es la fase del periastro respecto de t0_d, la misma convención que T0 en los .stan."""
    t, series = _series(comps, t_dias)
    out = _ajuste_grilla(t, series, np.array([1.0 / P_d]), t0_d, e_grid, n_fase)
    chi = sum(v[0] for v in out.values())[0]
    ie, iphi = np.unravel_index(np.argmin(chi), chi.shape)
    d = dict(e=float(e_grid[ie]), T0=float(iphi / n_fase), chi2=float(chi[ie, iphi]))
    for c, (_, coef) in out.items():
        g, A, B = coef[0, ie, iphi]
        d[f"K_{c}"] = float(np.hypot(A, B))
        if c == "v1":            # v1 = V0 + K[cos(ν+ω) + e cos ω]  →  A = K cos ω, B = −K sin ω, γ = V0 + e A
            d["w"] = float(np.mod(np.arctan2(-B, A), 2 * np.pi))
            d["V0"] = float(g - d["e"] * A)
    return d


def plan_semillas(n: int, picos_df: pd.DataFrame, Pmin_d: float, Pmax_d: float, base_d: float,
                  frac_picos: float = 0.6) -> pd.DataFrame:
    """Reparte n arranques: frac_picos entre los picos (por igual) y el resto en una grilla log-uniforme
    estratificada sobre [Pmin, Pmax] (un arranque por estrato, posición aleatoria dentro del estrato)."""
    n_picos = int(round(n * frac_picos)) if len(picos_df) else 0
    filas = []
    for i in range(n_picos):
        j = i % len(picos_df)
        r = picos_df.iloc[j]
        filas.append(dict(origen=f"pico_{j + 1}", criterio=r.criterio.replace("potencia_", ""),
                          P_centro_d=float(r.P_d), sigma_lnP=0.25 * float(r.P_d) / base_d))
    n_grilla = n - n_picos
    bordes = np.linspace(np.log(Pmin_d), np.log(Pmax_d), n_grilla + 1)
    for i in range(n_grilla):
        filas.append(dict(origen="grilla_log", P_centro_d=np.nan, sigma_lnP=np.nan,
                          lnP_lo=bordes[i], lnP_hi=bordes[i + 1]))
    return pd.DataFrame(filas)
