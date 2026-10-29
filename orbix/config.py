"""Carga de configuracion y resolucion de priors para corrida_masiva.

Politica metodologica: los valores de literatura en sistemas.csv se conservan
solo para comparacion y para los esquemas explicitos fix_*. Los priors de los
parametros orbitales son uniformes sobre rangos fisicos o ventanas amplias
derivadas de los datos/config, en linea con Videla et al. 2022.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

from . import fechas

PARAMETROS_PRIOR = {"SB1": ["P", "e", "T0", "V0", "K"],
                    "SB2": ["P", "e", "T0", "V0", "K1", "K2"]}
FAMILIAS = {"P": ("lognormal", "flat"), "e": ("beta", "flat"), "T0": ("beta", "flat"),
            "V0": ("normal", "flat"), "K": ("normal", "lognormal", "flat"),
            "K1": ("normal", "lognormal", "flat"), "K2": ("normal", "lognormal", "flat")}
REGLAS_AUTO = {
    ("P", "p1"): "P_init en la unidad de trabajo (manual > Lomb-Scargle > literatura fallback)",
    ("V0", "p1"): "media ponderada (1/err^2) de todas las RV del sistema",
    ("V0", "p2"): "max(10 km/s, rango total de las RV)",
    ("K", "p1"): "semiamplitud observada (max-min)/2 de la serie; lognormal: su logaritmo",
    ("K", "p2"): "normal: igual a p1 (sigma = 100 %); lognormal: 1.0",
}


def _es_vacio(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() == "")


def cargar_sistemas(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", dtype={"sistema": str})
    df["activo"] = df.get("activo", True)
    df["activo"] = df["activo"].astype(str).str.lower().isin(["true", "1", "si", "sí", "yes"])
    oblig = ["sistema", "tipo", "datos_v1", "epoch_format_in", "epoch_format_work", "esquemas"]
    falt = [c for c in oblig if c not in df.columns]
    if falt:
        raise ValueError(f"sistemas.csv: faltan columnas {falt}")
    if "epoch_format_out" not in df.columns:
        df["epoch_format_out"] = ""
    for idx, r in df.iterrows():
        fechas.formato(r.epoch_format_in)
        fechas.formato(r.epoch_format_work)
        if _es_vacio(r.get("epoch_format_out")):
            df.loc[idx, "epoch_format_out"] = r.epoch_format_work
        fechas.formato(df.loc[idx, "epoch_format_out"])
        if r.tipo == "SB2" and _es_vacio(r.get("datos_v2")):
            raise ValueError(f"{r.sistema}: tipo SB2 requiere datos_v2")
    return df


def cargar_priors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", dtype=str).fillna("")
    for c in ("sistema", "tipo", "parametro", "familia", "p1", "p2"):
        if c not in df.columns:
            raise ValueError(f"priors.csv: falta la columna {c}")
    for _, r in df.iterrows():
        base = "K" if r.parametro in ("K", "K1", "K2") else r.parametro
        if base not in FAMILIAS:
            raise ValueError(f"priors.csv: parametro desconocido {r.parametro!r}")
        if r.familia not in FAMILIAS[base]:
            raise ValueError(f"priors.csv: familia {r.familia!r} no valida para {r.parametro} (opciones {FAMILIAS[base]})")
        if (r.p1.strip() == "lit" or r.p2.strip() == "lit") and r.familia != "flat":
            raise ValueError("priors.csv no puede usar 'lit' para centrar priors; literatura = solo comparacion")
    return df


def fila_prior(priors: pd.DataFrame, sistema: str, tipo: str, parametro: str) -> pd.Series:
    for s, t in ((sistema, tipo), (sistema, "*"), ("*", tipo), ("*", "*")):
        m = priors[(priors.sistema == s) & (priors.tipo == t) & (priors.parametro == parametro)]
        if len(m):
            return m.iloc[-1]
    raise ValueError(f"No hay prior (ni default) para {sistema}/{tipo}/{parametro}")


def cargar_componente(path: Path, formato_in: str, formato_work: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns[:3]) != ["epoch", "RV", "err"]:
        raise ValueError(f"{path}: columnas esperadas epoch,RV,err; hay {list(df.columns)}")
    df = df[["epoch", "RV", "err"]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if (df.err <= 0).any():
        raise ValueError(f"{path}: hay errores <= 0")
    df["epoch_in"] = df["epoch"]
    df["epoch"] = fechas.convertir_epocas(df["epoch_in"], formato_in, formato_work)
    return df


def periodo_lomb_scargle(df: pd.DataFrame, formato_work: str, Pmin_d=None, Pmax_d=None) -> dict:
    """Lomb-Scargle en dias, con ventana configurable por sistema."""
    t_d = fechas.a_jd(df.epoch.values, formato_work)
    base = float(t_d.max() - t_d.min())
    Pmin = 0.5 if _es_vacio(Pmin_d) else float(Pmin_d)
    Pmax = 2.0 * base if _es_vacio(Pmax_d) else float(Pmax_d)
    if len(df) < 5 or base <= 0 or Pmax <= Pmin:
        return dict(P_ls_d=np.nan, P_ls_work=np.nan, ls_Pmin_d=Pmin, ls_Pmax_d=Pmax, ls_base_d=base)
    freq, power = LombScargle(t_d, df.RV.values, df.err.values).autopower(
        minimum_frequency=1 / Pmax, maximum_frequency=1 / Pmin, samples_per_peak=20)
    P_d = float(1 / freq[np.argmax(power)])
    return dict(P_ls_d=P_d, P_ls_work=float(fechas.convertir_duracion(P_d, "d", fechas.unidad_periodo(formato_work))),
                ls_Pmin_d=Pmin, ls_Pmax_d=Pmax, ls_base_d=base)


def periodo_literatura_work(cfg: pd.Series) -> float:
    if _es_vacio(cfg.get("P_lit")):
        return np.nan
    return float(fechas.convertir_duracion(float(cfg.P_lit), cfg.P_lit_unit, fechas.unidad_periodo(cfg.epoch_format_work)))


def periodo_inicial(cfg: pd.Series, df_v1: pd.DataFrame) -> dict:
    """P_init en unidad de trabajo: manual > Lomb-Scargle > literatura fallback."""
    ls = periodo_lomb_scargle(df_v1, cfg.epoch_format_work, cfg.get("ls_Pmin_d"), cfg.get("ls_Pmax_d"))
    u_work = fechas.unidad_periodo(cfg.epoch_format_work)
    if not _es_vacio(cfg.get("P_init")):
        P, fuente = float(fechas.convertir_duracion(float(cfg.P_init), cfg.P_init_unit, u_work)), "manual"
    elif not np.isnan(ls["P_ls_work"]):
        P, fuente = ls["P_ls_work"], "lomb-scargle"
    elif not np.isnan(periodo_literatura_work(cfg)):
        P, fuente = periodo_literatura_work(cfg), "literatura_fallback"
    else:
        raise ValueError(f"{cfg.sistema}: no hay P_init manual, no se pudo calcular Lomb-Scargle ni hay P_lit fallback")
    return dict(P_init_work=P, P_init_fuente=fuente, P_unidad_work=u_work, **ls)


def _num(x, nombre):
    try:
        return float(x)
    except (TypeError, ValueError):
        raise ValueError(f"valor no numerico para {nombre}: {x!r}") from None


def _rango_v0(rv_all: pd.DataFrame) -> tuple[float, float]:
    rango = float(rv_all.RV.max() - rv_all.RV.min())
    margen = max(rango, 10.0)
    return float(rv_all.RV.min() - margen), float(rv_all.RV.max() + margen)


def _kmax(serie: pd.DataFrame) -> float:
    rango = float(serie.RV.max() - serie.RV.min())
    return max(2.0 * rango, 1.0)


def _ventana_periodo_work(P_info: dict, formato_work: str) -> tuple[float, float]:
    u_work = fechas.unidad_periodo(formato_work)
    pmin = float(fechas.convertir_duracion(P_info["ls_Pmin_d"], "d", u_work))
    pmax = float(fechas.convertir_duracion(P_info["ls_Pmax_d"], "d", u_work))
    if not (np.isfinite(pmin) and np.isfinite(pmax) and 0 < pmin < pmax):
        raise ValueError(f"ventana LS invalida para P: [{pmin}, {pmax}] {u_work}")
    return pmin, pmax


def resolver_priors(priors: pd.DataFrame, cfg: pd.Series, esquema: str, comps: dict,
                    P_info: dict) -> tuple[dict, list[dict]]:
    """Devuelve (datos de Stan para priors/cotas, registro legible)."""
    tipo, sistema = cfg.tipo, cfg.sistema
    fix_P, fix_e = esquema in ("fix_P", "fix_eP"), esquema in ("fix_e", "fix_eP")
    rv_all = pd.concat(list(comps.values()))
    stan, reg = {}, []

    v0_min, v0_max = _rango_v0(rv_all)
    stan.update(V0_min=v0_min, V0_max=v0_max)
    if tipo == "SB1":
        stan["K_max"] = _kmax(comps["v1"])
    else:
        stan["K1_max"] = _kmax(comps["v1"])
        stan["K2_max"] = _kmax(comps["v2"])
    if not fix_P:
        P_min, P_max = _ventana_periodo_work(P_info, cfg.epoch_format_work)
        stan.update(log_P_min=float(np.log(P_min)), log_P_max=float(np.log(P_max)))

    for par in PARAMETROS_PRIOR[tipo]:
        if (par == "P" and fix_P) or (par == "e" and fix_e):
            continue
        r = fila_prior(priors, sistema, tipo, par)
        fam, p1, p2 = r.familia, r.p1.strip(), r.p2.strip()
        nota = []
        if par == "P":
            u_work = fechas.unidad_periodo(cfg.epoch_format_work)
            if fam == "lognormal":
                if p1 == "auto":
                    centro, nota = P_info["P_init_work"], ["p1=auto: P_init"]
                elif p1 == "lit":
                    raise ValueError(f"{sistema}: literatura no puede centrar prior de P")
                else:
                    u = r.get("unidad", "") or u_work
                    centro = float(fechas.convertir_duracion(_num(p1, "P.p1"), u, u_work))
                stan.update(prior_logP_fam=1, mu_logP=float(np.log(centro)), sigma_logP=_num(p2, "P.p2"))
                desc = f"log P ~ Normal(ln {centro:.6g} {u_work}, {stan['sigma_logP']:g})"
            else:
                P_min, P_max = np.exp(stan["log_P_min"]), np.exp(stan["log_P_max"])
                stan.update(prior_logP_fam=0, mu_logP=0.0, sigma_logP=1.0)
                # El parametro muestreado es log_P y la rama flat de Stan
                # no suma log_P: la densidad efectiva es uniforme en log P.
                desc = f"P ~ LogUniform({P_min:.6g}, {P_max:.6g}) {u_work}"
                nota = [f"ventana LS {P_info['ls_Pmin_d']:.6g}-{P_info['ls_Pmax_d']:.6g} d; flat en log_P, sin jacobiano +log_P"]
        elif par in ("e", "T0"):
            key = "e" if par == "e" else "T0"
            if fam == "beta":
                a, b = _num(p1, f"{par}.p1"), _num(p2, f"{par}.p2")
                stan.update({f"prior_{key}_fam": 1, f"a_{key}": a, f"b_{key}": b})
                desc = f"{par} ~ Beta({a:g}, {b:g})"
            else:
                stan.update({f"prior_{key}_fam": 0, f"a_{key}": 1.0, f"b_{key}": 1.0})
                desc = f"{par} uniforme en su soporte"
        elif par == "V0":
            if fam == "normal":
                if p1 == "auto":
                    wgt = 1 / rv_all.err ** 2
                    mu, nota = float(np.sum(wgt * rv_all.RV) / np.sum(wgt)), ["p1=auto: media ponderada RV"]
                elif p1 == "lit":
                    raise ValueError(f"{sistema}: literatura no puede centrar prior de V0")
                else:
                    mu = _num(p1, "V0.p1")
                if p2 == "auto":
                    sd = float(max(10.0, rv_all.RV.max() - rv_all.RV.min()))
                    nota.append("p2=auto: max(10, rango RV)")
                else:
                    sd = _num(p2, "V0.p2")
                stan.update(prior_V0_fam=1, mu_V0=mu, sigma_V0=sd)
                desc = f"V0 ~ Normal({mu:.4g}, {sd:.4g}) km/s"
            else:
                stan.update(prior_V0_fam=0, mu_V0=0.0, sigma_V0=1.0)
                desc = f"V0 ~ Uniform({v0_min:.4g}, {v0_max:.4g}) km/s"
                nota = ["cotas amplias desde min/max RV"]
        else:
            comp = "v2" if par == "K2" else "v1"
            serie = comps[comp]
            semi = float((serie.RV.max() - serie.RV.min()) / 2)
            if fam in ("normal", "lognormal"):
                if p1 == "auto":
                    centro, nota = semi, [f"p1=auto: semiamplitud observada de {comp}"]
                elif p1 == "lit":
                    raise ValueError(f"{sistema}: literatura no puede centrar prior de {par}")
                else:
                    centro = _num(p1, f"{par}.p1")
                if fam == "normal":
                    sd = centro if p2 == "auto" else _num(p2, f"{par}.p2")
                    stan.update({f"prior_{par}_fam": 1, f"mu_{par}": centro, f"sigma_{par}": sd})
                    desc = f"{par} ~ Normal({centro:.4g}, {sd:.4g}) T[0,] km/s"
                else:
                    sd = 1.0 if p2 == "auto" else _num(p2, f"{par}.p2")
                    stan.update({f"prior_{par}_fam": 2, f"mu_{par}": float(np.log(centro)), f"sigma_{par}": sd})
                    desc = f"{par} ~ LogNormal(ln {centro:.4g}, {sd:.4g}) km/s"
            else:
                stan.update({f"prior_{par}_fam": 0, f"mu_{par}": 0.0, f"sigma_{par}": 1.0})
                desc = f"{par} ~ Uniform(0, {stan[f'{par}_max']:.4g}) km/s"
                nota = [f"Kmax=2*rango_RV de {comp}"]
        reg.append(dict(sistema=sistema, tipo=tipo, esquema=esquema, parametro=par, familia=fam,
                        p1_config=p1, p2_config=p2, prior_resuelto=desc, regla="; ".join(nota),
                        fila_origen=f"{r.sistema}/{r.tipo}", fuente=r.get("fuente", ""), nota=r.get("nota", "")))

    if fix_P:
        P_lit = periodo_literatura_work(cfg)
        if np.isnan(P_lit):
            raise ValueError(f"{sistema}: esquema {esquema} requiere P_lit (dato faltante)")
        stan["P_fixed"] = P_lit
        reg.append(dict(sistema=sistema, tipo=tipo, esquema=esquema, parametro="P", familia="fijo",
                        prior_resuelto=f"P fijo = {P_lit:.6g} {fechas.unidad_periodo(cfg.epoch_format_work)}",
                        regla="P_lit convertido a la unidad de trabajo; esquema fix_*",
                        fuente=cfg.get("fuente_lit", "")))
    if fix_e:
        if _es_vacio(cfg.get("e_lit")):
            raise ValueError(f"{sistema}: esquema {esquema} requiere e_lit (dato faltante)")
        stan["e_fixed"] = float(cfg.e_lit)
        reg.append(dict(sistema=sistema, tipo=tipo, esquema=esquema, parametro="e", familia="fijo",
                        prior_resuelto=f"e fijo = {float(cfg.e_lit):g}", regla="e_lit; esquema fix_*",
                        fuente=cfg.get("fuente_lit", "")))
    return stan, reg
