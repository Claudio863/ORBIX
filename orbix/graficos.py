"""Figuras y tablas de presentación (notebook 02). Solo LEE resultados: no compila ni corre Stan."""
from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import fechas, resumen, rv

ETIQUETAS = {"P": "P", "T": "T", "e": "e", "w_deg": "ω", "V0": "V0", "K": "K", "K1": "K1", "K2": "K2", "q": "q"}
ORDEN = ["P", "T", "e", "w_deg", "V0", "K", "K1", "K2", "q"]


# --------------------------------------------------------------------------- carga
class Resultados:
    """Acceso a lo que dejó el notebook 01 en resultados/."""

    def __init__(self, raiz: Path):
        self.raiz = Path(raiz)
        self.tablas = self.raiz / "resultados" / "tablas"
        self.draws = self.raiz / "resultados" / "draws"
        req = ["corridas.csv", "diagnosticos.csv", "chi2_residuos.csv", "resumen_parametros.csv"]
        faltan = [r for r in req if not (self.tablas / r).exists()]
        if faltan:
            raise FileNotFoundError(f"Faltan resultados del notebook 01 en {self.tablas}: {faltan}. "
                                    "Corre primero 01_corrida_masiva.ipynb.")
        self.corridas = pd.read_csv(self.tablas / "corridas.csv")
        self.diagnosticos = pd.read_csv(self.tablas / "diagnosticos.csv")
        for tabla in (self.corridas, self.diagnosticos):
            for col in tabla.columns:
                if col.startswith("flag_") or col in ("convergencia_ok", "soporte_ok", "fechas_plausibles"):
                    tabla[col] = resumen.normalizar_booleanos(tabla[col])
        self.chi2 = pd.read_csv(self.tablas / "chi2_residuos.csv")
        self.parametros = pd.read_csv(self.tablas / "resumen_parametros.csv")

    def faltantes(self, esperadas: list[str]) -> list[str]:
        """Corridas esperadas que no tienen resultados (o fallaron)."""
        ok = set(self.corridas.loc[self.corridas.estado == "ok", "run_id"])
        return [r for r in esperadas if r not in ok or not (self.draws / r).exists()]

    def meta(self, run_id: str) -> pd.Series:
        return self.corridas.set_index("run_id").loc[run_id]

    def draws_de(self, run_id: str) -> pd.DataFrame:
        datos = self.datos_de(run_id)
        referencia = min(d.epoch.min() for d in datos.values())
        return resumen.recentrar_circulares(resumen.leer_draws(self.draws / run_id), referencia)

    def datos_de(self, run_id: str) -> dict:
        d = self.draws / run_id
        return {c: pd.read_csv(d / f"datos_{c}.csv") for c in ("v1", "v2") if (d / f"datos_{c}.csv").exists()}


# --------------------------------------------------------------------------- unidades de salida
def formato_salida(meta: pd.Series, salida: dict | None) -> str:
    """Formato de época para mostrar: salida[sistema] > salida['*'] > formato de trabajo."""
    if not salida:
        return meta.epoch_format_work
    return salida.get(meta.sistema, salida.get("*", meta.epoch_format_work))


def convertir_resumen(tabla: pd.DataFrame, corridas: pd.DataFrame, salida: dict | None) -> pd.DataFrame:
    """Convierte P y T (MAP, mediana, media, HPDI, anchura, sd) a la unidad de salida de cada corrida."""
    t = tabla.copy()
    fmt_w = t.run_id.map(corridas.set_index("run_id").epoch_format_work)
    fmt_o = [formato_salida(corridas.set_index("run_id").loc[r], salida) for r in t.run_id]
    for i, (p, fw, fo) in enumerate(zip(t.parametro, fmt_w, fmt_o)):
        if p not in ("P", "T") or fw == fo:
            continue
        for c in ("MAP", "mediana", "media", "hpdi_inf", "hpdi_sup"):
            t.iat[i, t.columns.get_loc(c)] = float(fechas.convertir_parametro(t.iat[i, t.columns.get_loc(c)], p, fw, fo))
        f = fechas.escala_anchura(p, fw, fo)
        for c in ("ancho_hpdi", "sd"):
            t.iat[i, t.columns.get_loc(c)] *= f
        t.iat[i, t.columns.get_loc("unidad")] = fechas.unidad_periodo(fo) if p == "P" else fo
    return t


# --------------------------------------------------------------------------- tablas
def _fmt(x, nd=4):
    return "" if pd.isna(x) else f"{x:.{nd}g}"


def tabla_por_sistema(par: pd.DataFrame) -> pd.DataFrame:
    """Mediana + HPDI 95 % por parámetro (una fila por corrida, una columna por parámetro)."""
    par = par.copy()
    par["texto"] = [("= " + _fmt(m, 6) + " (fijo)") if f else f"{_fmt(m, 6)} [{_fmt(lo, 6)}, {_fmt(hi, 6)}]"
                    for m, lo, hi, f in zip(par.mediana, par.hpdi_inf, par.hpdi_sup, par.fijo)]
    par["columna"] = [f"{ETIQUETAS.get(p, p)} [{u}]" for p, u in zip(par.parametro, par.unidad)]
    orden_cols = sorted(par.columna.unique(), key=lambda c: ORDEN.index(next(k for k, v in ETIQUETAS.items()
                                                                              if c.startswith(v + " "))))
    t = par.pivot_table(index=["sistema", "tipo", "esquema"], columns="columna", values="texto", aggfunc="first")
    return t.reindex(columns=orden_cols)


def tabla_incertidumbre(par: pd.DataFrame) -> pd.DataFrame:
    """Comparación de incertidumbre PARÁMETRO POR PARÁMETRO (Tablas 4.6 / 4.10 reestructuradas).

    Filas: sistema-tipo-esquema. Columnas: para cada parámetro, mediana, anchura del HPDI 95 %
    y anchura relativa (%). No se promedian magnitudes con unidades distintas.
    """
    p = par[~par.fijo].copy()
    p["ancho_rel_%"] = 100 * p.ancho_hpdi / p.mediana.abs()
    p.loc[p.parametro.isin(["T", "w_deg"]), "ancho_rel_%"] = np.nan   # épocas/ángulos: % no tiene sentido
    p["col"] = [f"{ETIQUETAS.get(x, x)} [{u}]" for x, u in zip(p.parametro, p.unidad)]
    largo = p.melt(id_vars=["sistema", "tipo", "esquema", "parametro", "col"],
                   value_vars=["mediana", "ancho_hpdi", "ancho_rel_%"], var_name="estadistico", value_name="valor")
    # la unidad va en el encabezado: si dos corridas usan unidades distintas, quedan en columnas distintas
    t = largo.pivot_table(index=["sistema", "tipo", "esquema"], columns=["col", "estadistico"],
                          values="valor", aggfunc="first")
    orden = {c: (ORDEN.index(x), c) for x, c in zip(p.parametro, p.col)}
    cols = sorted(t.columns.get_level_values(0).unique(), key=lambda c: orden[c])
    return t.reindex(columns=pd.MultiIndex.from_product([cols, ["mediana", "ancho_hpdi", "ancho_rel_%"]])) \
            .dropna(axis=1, how="all")


def unidades_de(par: pd.DataFrame) -> pd.DataFrame:
    """Unidad declarada por (corrida, parámetro), para acompañar cada tabla exportada."""
    return par.pivot_table(index=["sistema", "tipo", "esquema"], columns="parametro", values="unidad", aggfunc="first") \
              .reindex(columns=[x for x in ORDEN if x in set(par.parametro)])


def exportar(df: pd.DataFrame, carpeta: Path, nombre: str) -> list[Path]:
    """Export a numerical table in CSV without writing manuscript sources."""
    carpeta.mkdir(parents=True, exist_ok=True)
    path = carpeta / f"{nombre}.csv"
    df.to_csv(path)
    return [path]


# --------------------------------------------------------------------------- figuras
def figura_ajuste(res: Resultados, run_id: str, salida: dict | None = None, n_muestras: int = 100,
                  semilla: int = 0) -> plt.Figure:
    """RV vs fase y vs época: curva MAP, banda de muestras posteriores, datos ± error, residuos."""
    meta = res.meta(run_id)
    tipo, fw = meta.tipo, meta.epoch_format_work
    fo = formato_salida(meta, salida)
    draws = res.draws_de(run_id)
    datos = res.datos_de(run_id)
    par = res.parametros[res.parametros.run_id == run_id]
    pmap = dict(zip(par.parametro, par.MAP))
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(draws), size=min(n_muestras, len(draws)), replace=False)
    t_all = np.concatenate([d.epoch.values for d in datos.values()])
    t_fino = np.linspace(t_all.min(), t_all.max(), 2000)
    t_ref = pmap["T"]
    fase_fina = np.linspace(0, 1, 500)
    colores = {"v1": "tab:blue", "v2": "tab:red"}

    fig, ax = plt.subplots(2, 2, figsize=(13, 7), gridspec_kw=dict(height_ratios=[3, 1]), sharex="col")
    for i in idx:
        pi_ = {k: draws[k].iat[i] for k in pmap if k in draws}
        for c, y in rv.curvas(pi_, t_ref + fase_fina * pi_["P"], tipo).items():
            ax[0, 0].plot(fase_fina, y, color=colores[c], alpha=0.03, lw=0.8)
    for c, y in rv.curvas(pmap, t_ref + fase_fina * pmap["P"], tipo).items():
        ax[0, 0].plot(fase_fina, y, color=colores[c], lw=1.8, label=f"MAP {c}")
    for c, y in rv.curvas(pmap, t_fino, tipo).items():
        ax[0, 1].plot(fechas.convertir_epocas(t_fino, fw, fo), y, color=colores[c], lw=1.2)
    for c, d in datos.items():
        modelo = rv.curvas(pmap, d.epoch.values, tipo)[c]
        f = rv.fase(d.epoch.values, pmap["P"], t_ref)
        te = fechas.convertir_epocas(d.epoch.values, fw, fo)
        ax[0, 0].errorbar(f, d.RV, d.err, fmt=".", color="k" if c == "v1" else "0.45", ms=4, capsize=2, label=f"datos {c}")
        ax[0, 1].errorbar(te, d.RV, d.err, fmt=".", color="k" if c == "v1" else "0.45", ms=4, capsize=2)
        ax[1, 0].errorbar(f, d.RV - modelo, d.err, fmt=".", color=colores[c], ms=4, capsize=2)
        ax[1, 1].errorbar(te, d.RV - modelo, d.err, fmt=".", color=colores[c], ms=4, capsize=2)
    for a in ax[1]:
        a.axhline(0, color="0.5", lw=0.8)
    ax[0, 0].set_ylabel("RV [km/s]")
    ax[1, 0].set_ylabel("O − C [km/s]")
    ax[1, 0].set_xlabel("Fase (desde el periastro)")
    ax[1, 1].set_xlabel(f"Época [{fechas.etiqueta_unidad(fo)}]")
    ax[0, 0].legend(fontsize=8, loc="best")
    P_o = fechas.convertir_parametro(pmap["P"], "P", fw, fo)
    fig.suptitle(f"{meta.sistema} · {tipo} · {meta.esquema}   (P = {P_o:.6g} {fechas.etiqueta_unidad(fechas.unidad_periodo(fo))})")
    fig.tight_layout()
    return fig


def figura_traza(res: Resultados, run_id: str) -> plt.Figure:
    """Trace plots de los parámetros libres (para corridas con R̂ alto)."""
    meta = res.meta(run_id)
    draws = res.draws_de(run_id)
    par = res.parametros[(res.parametros.run_id == run_id) & (~res.parametros.fijo)]
    nombres = [p for p in ORDEN if p in set(par.parametro)]
    datos = {p: np.stack([g[p].to_numpy() for _, g in draws.groupby("chain")]) for p in nombres}
    axes = az.plot_trace(datos, compact=False, figsize=(12, 2.0 * len(nombres)))
    fig = axes.ravel()[0].figure
    fig.suptitle(f"Trazas · {run_id}  (R̂ máx = {res.diagnosticos.set_index('run_id').loc[run_id, 'R_hat_max']:.3f})")
    fig.tight_layout()
    return fig


def figura_pares(res: Resultados, run_id: str, max_puntos: int = 2000) -> plt.Figure:
    draws = res.draws_de(run_id)
    par = res.parametros[(res.parametros.run_id == run_id) & (~res.parametros.fijo)]
    nombres = [p for p in ORDEN if p in set(par.parametro)]
    datos = {p: np.stack([g[p].to_numpy() for _, g in draws.groupby("chain")]) for p in nombres}
    axes = az.plot_pair(datos, kind="scatter", marginals=True, scatter_kwargs=dict(alpha=0.2, s=2),
                        figsize=(2.2 * len(nombres), 2.2 * len(nombres)))
    fig = np.ravel(axes)[0].figure
    fig.suptitle(f"Pares · {run_id}")
    return fig
