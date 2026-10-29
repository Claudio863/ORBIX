"""Resumen de posteriores, diagnósticos y χ²/residuos (sin compilar ni correr Stan)."""
from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from . import fechas, rv

PARAMS_SALIDA = {"SB1": ["P", "T", "e", "w_deg", "V0", "K"],
                 "SB2": ["P", "T", "e", "w_deg", "V0", "K1", "K2", "q"]}
POSITIVOS = {"P", "K", "K1", "K2", "q"}        # soporte (0, ∞)
UNIDADES_FIJAS = {"e": "-", "w_deg": "deg", "V0": "km/s", "K": "km/s", "K1": "km/s", "K2": "km/s", "q": "-"}


def normalizar_booleanos(serie: pd.Series) -> pd.Series:
    """Interpreta flags CSV por valor, nunca por la veracidad de una cadena."""
    def convertir(valor):
        if pd.isna(valor):
            return pd.NA
        texto = str(valor).strip().lower()
        if texto in ("true", "1", "1.0"):
            return True
        if texto in ("false", "0", "0.0"):
            return False
        raise ValueError(f"Flag booleano no reconocido: {valor!r}")
    return serie.map(convertir).astype("boolean")


def unidad(parametro: str, formato_trabajo: str) -> str:
    if parametro == "P":
        return fechas.unidad_periodo(formato_trabajo)
    if parametro == "T":
        return formato_trabajo
    return UNIDADES_FIJAS.get(parametro, "")


def unidad_out(parametro: str, formato_salida: str) -> str:
    if parametro == "P":
        return fechas.unidad_periodo(formato_salida)
    if parametro == "T":
        return formato_salida
    return UNIDADES_FIJAS.get(parametro, "")


# --------------------------------------------------------------------------- lectura de cadenas
def leer_draws(dir_draws: Path) -> pd.DataFrame:
    """Lee los CSV de CmdStan de una corrida (una por cadena) → DataFrame con columna 'chain'."""
    # solo los CSV de CmdStan (empiezan con comentarios '#'); se excluyen datos_*.csv
    archivos = [a for a in sorted(Path(dir_draws).glob("*.csv"))
                if not a.name.startswith("datos_") and a.read_text(encoding="utf-8", errors="ignore")[:1] == "#"]
    if not archivos:
        raise FileNotFoundError(f"No hay cadenas en {dir_draws}")
    partes = []
    for i, a in enumerate(archivos, start=1):
        d = pd.read_csv(a, comment="#")
        d.insert(0, "chain", i)
        d.insert(1, "draw", np.arange(len(d)))
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def _por_cadena(draws: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([g[col].to_numpy() for _, g in draws.groupby("chain")])


# --------------------------------------------------------------------------- variables circulares
def recentrar_circulares(draws: pd.DataFrame, epoca_referencia: float | None = None) -> pd.DataFrame:
    """Desenrolla ω (período 360°), T (período P) y lam_ref (2π) alrededor de su media circular.

    Una posterior que cruza 0°/360° (o T que cruza un período) daría HPDI y R̂ falsos si se
    trata como lineal. Tras recentrar, ω puede salir de [0, 360) (p. ej. [−5°, 12°]): es el
    mismo intervalo en el círculo.
    """
    d = draws.copy()

    def _desenrollar(x, periodo):
        ang = 2 * np.pi * x / periodo
        c = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) * periodo / (2 * np.pi)
        c = c % periodo                       # centro en [0, periodo): ω se informa en [0°, 360°) si no cruza 0
        return x - periodo * np.round((x - c) / periodo)

    if "w_deg" in d:
        d["w_deg"] = _desenrollar(d["w_deg"].to_numpy(), 360.0)
    if "lam_ref" in d:
        d["lam_ref"] = _desenrollar(d["lam_ref"].to_numpy(), 2 * np.pi)
    if "T" in d and "P" in d and "T0" in d:
        # T/P depende del origen arbitrario de las fechas cuando P varia.
        # La fase generada por Stan usa el origen fijo min_t: T = min_t + P*T0.
        fase = d["T0"].to_numpy()
        centrada = _desenrollar(fase, 1.0)
        if epoca_referencia is None:
            d["T"] = d["T"].to_numpy() + d["P"].to_numpy() * (centrada - fase)
        else:
            # Recuperar T desde la fase evita la cuantizacion de las epocas
            # absolutas guardadas con pocas cifras en cadenas historicas.
            d["T"] = float(epoca_referencia) + d["P"].to_numpy() * centrada
        d["T0"] = centrada
    # Sin T0 no se conoce un origen fijo: conservar T antes que inventarlo.
    return d


# --------------------------------------------------------------------------- resumen
def resumir(draws: pd.DataFrame, tipo: str, formato_trabajo: str,
            formato_salida: str | None = None, hdi_prob: float = 0.95,
            epoca_referencia: float | None = None) -> pd.DataFrame:
    if formato_salida is None:
        formato_salida = formato_trabajo
    draws = recentrar_circulares(draws, epoca_referencia)
    i_map = int(draws["lp__"].to_numpy().argmax())
    filas = []
    for p in PARAMS_SALIDA[tipo]:
        x = draws[p].to_numpy()
        fijo = bool(np.nanstd(x) < 1e-12 * max(1.0, abs(np.nanmean(x))))
        lo, hi = (x[0], x[0]) if fijo else az.hdi(x, hdi_prob=hdi_prob)
        fila = dict(parametro=p, unidad=unidad(p, formato_trabajo), fijo=fijo,
                    MAP=float(x[i_map]), mediana=float(np.median(x)), media=float(np.mean(x)),
                    sd=float(np.std(x, ddof=1)), hpdi_inf=float(lo), hpdi_sup=float(hi), ancho_hpdi=float(hi - lo),
                    R_hat=np.nan, ESS_bulk=np.nan, ESS_tail=np.nan)
        valores = np.array([fila["MAP"], fila["mediana"], fila["media"], fila["hpdi_inf"], fila["hpdi_sup"]])
        valores_out = fechas.convertir_parametro(valores, p, formato_trabajo, formato_salida)
        factor = fechas.escala_anchura(p, formato_trabajo, formato_salida)
        fila.update(unidad_out=unidad_out(p, formato_salida),
                    MAP_out=float(valores_out[0]), mediana_out=float(valores_out[1]),
                    media_out=float(valores_out[2]), hpdi_inf_out=float(valores_out[3]),
                    hpdi_sup_out=float(valores_out[4]), sd_out=float(fila["sd"] * factor),
                    ancho_hpdi_out=float(fila["ancho_hpdi"] * factor))
        if not fijo:
            arr = _por_cadena(draws, p)
            fila.update(R_hat=float(az.rhat(arr)), ESS_bulk=float(az.ess(arr, method="bulk")),
                        ESS_tail=float(az.ess(arr, method="tail")))
        # Soporte físico: cantidades positivas no pueden tener límite inferior < 0
        fila["soporte_ok"] = not (p in POSITIVOS and lo < 0) and not (p == "e" and (lo < 0 or hi >= 1))
        filas.append(fila)
    return pd.DataFrame(filas)


def diagnosticar(draws: pd.DataFrame, resumen: pd.DataFrame, max_treedepth: int, umbral_rhat: float = 1.01) -> dict:
    libres = resumen[~resumen.fijo]
    # incluir también los parámetros internos muestreados (T0, log_P) si existen
    draws = recentrar_circulares(draws)
    internos = [c for c in ("log_P", "lam_ref", "e") if c in draws.columns and draws[c].std() > 0]
    rh_int = [float(az.rhat(_por_cadena(draws, c))) for c in internos]
    energia = [g["energy__"].to_numpy() for _, g in draws.groupby("chain")]
    ebfmi = [np.sum(np.diff(E) ** 2) / np.sum((E - E.mean()) ** 2) for E in energia]
    rhat_max = float(np.nanmax(list(libres.R_hat) + rh_int)) if len(libres) or rh_int else np.nan
    d = dict(n_cadenas=int(draws.chain.nunique()), n_muestras=int(len(draws)),
             divergencias=int(draws["divergent__"].sum()),
             treedepth_max=int((draws["treedepth__"] >= max_treedepth).sum()),
             ebfmi_min=float(np.min(ebfmi)), R_hat_max=rhat_max,
             ESS_bulk_min=float(libres.ESS_bulk.min()) if len(libres) else np.nan,
             ESS_tail_min=float(libres.ESS_tail.min()) if len(libres) else np.nan,
             soporte_ok=bool(resumen.soporte_ok.all()))
    d["flag_rhat"] = bool(rhat_max > umbral_rhat)
    d["flag_divergencias"] = d["divergencias"] > 0
    d["flag_ebfmi"] = d["ebfmi_min"] < 0.3
    d["convergencia_ok"] = not (d["flag_rhat"] or d["flag_divergencias"])
    return d


# --------------------------------------------------------------------------- χ² y residuos
def parametros_punto(resumen: pd.DataFrame, estadistico: str = "MAP") -> dict:
    return dict(zip(resumen.parametro, resumen[estadistico]))


def chi2_residuos(resumen: pd.DataFrame, comps: dict, tipo: str, esquema: str, estadistico: str = "MAP"):
    """χ² y residuos (obs − modelo) evaluados en el MAP de las muestras.

    k (parámetros libres que afectan a cada componente):
      SB1: T0, P*, e*, w, V0, K           (* si no están fijos)
      SB2: por componente, los compartidos (T0, P*, e*, w, V0) + su propia K;
           fila 'total' con N = N1 + N2 y k = todos los libres del modelo.
    χ²_red = χ² / (N − k).
    """
    par = parametros_punto(resumen, estadistico)
    fix_P, fix_e = esquema in ("fix_P", "fix_eP"), esquema in ("fix_e", "fix_eP")
    k_comp = 4 + (not fix_P) + (not fix_e)          # T0, w, V0, K + P? + e?
    filas, residuos = [], []
    for comp, df in comps.items():
        modelo = rv.curvas(par, df.epoch.values, tipo)[comp]
        r = df.RV.values - modelo
        z = r / df.err.values
        chi2 = float(np.sum(z ** 2))
        n = len(df)
        filas.append(dict(componente=comp, N=n, k=k_comp, gl=n - k_comp, chi2=chi2,
                          chi2_red=chi2 / (n - k_comp) if n > k_comp else np.nan,
                          rms_residuo=float(np.sqrt(np.mean(r ** 2))), media_residuo=float(np.mean(r)),
                          max_abs_z=float(np.max(np.abs(z)))))
        residuos.append(pd.DataFrame(dict(componente=comp, epoch=df.epoch.values, RV=df.RV.values, err=df.err.values,
                                          modelo=modelo, residuo=r, residuo_norm=z)))
    if tipo == "SB2":
        n = sum(f["N"] for f in filas)
        k = k_comp + 1
        chi2 = sum(f["chi2"] for f in filas)
        filas.append(dict(componente="total", N=n, k=k, gl=n - k, chi2=chi2, chi2_red=chi2 / (n - k),
                          rms_residuo=float(np.sqrt(np.mean(np.concatenate([d.residuo for d in residuos]) ** 2))),
                          media_residuo=float(np.mean(np.concatenate([d.residuo for d in residuos]))),
                          max_abs_z=max(f["max_abs_z"] for f in filas)))
    return pd.DataFrame(filas), pd.concat(residuos, ignore_index=True)
