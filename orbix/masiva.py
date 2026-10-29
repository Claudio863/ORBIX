"""Corrida masiva: todos los (sistema, tipo, esquema) con persistencia incremental en resultados/tablas/."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import ajuste, config, resumen

COLS_DIAG = ["run_id", "sistema", "tipo", "esquema", "estado", "motivo", "R_hat_max", "ESS_bulk_min", "ESS_tail_min",
             "divergencias", "treedepth_max", "ebfmi_min", "flag_rhat", "flag_divergencias", "flag_ebfmi",
             "soporte_ok", "convergencia_ok", "chi2_red", "max_abs_z", "flag_chi2_alto", "siembra_P",
             "origen_mejor_inicio", "n_inicios", "n_inicios_ok", "segundos"]


def lista_corridas(sistemas: pd.DataFrame, esquemas=None, solo=None) -> list[tuple[pd.Series, str]]:
    """(cfg, esquema) a correr: sistemas activos, filtrados por `solo` (nombres) y `esquemas`."""
    tareas = []
    for _, cfg in sistemas[sistemas.activo].iterrows():
        if solo and cfg.sistema not in solo:
            continue
        for esq in [e.strip() for e in cfg.esquemas.split(",")]:
            if esquemas and esq not in esquemas:
                continue
            tareas.append((cfg, esq))
    return tareas


def _leer(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        tabla = pd.read_csv(path)
        for col in tabla.columns:
            if col.startswith("flag_") or col in ("convergencia_ok", "soporte_ok", "fechas_plausibles", "fijo"):
                tabla[col] = resumen.normalizar_booleanos(tabla[col])
        return tabla
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _upsert(viejo: pd.DataFrame, nuevo: pd.DataFrame, clave="run_id") -> pd.DataFrame:
    if viejo.empty:
        return nuevo
    if nuevo.empty:
        return viejo
    return pd.concat([viejo[~viejo[clave].isin(nuevo[clave].unique())], nuevo], ignore_index=True)


def correr_todo(sistemas: pd.DataFrame, priors: pd.DataFrame, rutas: ajuste.Rutas, aj: ajuste.Ajustes,
                esquemas=None, solo=None, reanudar: bool = True, log=print) -> dict:
    """Corre todo y guarda: corridas.csv, diagnosticos.csv, resumen_parametros.csv,
    chi2_residuos.csv, priors_usados.csv y fechas_corridas.csv
    (se actualizan tras cada corrida)."""
    T = rutas.tablas
    T.mkdir(parents=True, exist_ok=True)
    (T / "por_corrida").mkdir(exist_ok=True)
    arch = {k: T / f"{k}.csv" for k in ("corridas", "diagnosticos", "resumen_parametros", "chi2_residuos",
                                        "priors_usados", "fechas_aplicadas")}
    # fechas_aplicadas.csv es la auditoria por componente generada por el notebook.
    arch["fechas_aplicadas"] = T / "fechas_corridas.csv"
    tablas = {k: _leer(v) for k, v in arch.items()}
    tareas = lista_corridas(sistemas, esquemas, solo)
    hechas = set()
    if reanudar and not tablas["corridas"].empty:
        ok = tablas["corridas"][tablas["corridas"].estado == "ok"].run_id
        hechas = {r for r in ok if (rutas.draws / r).exists()}
    log(f"{len(tareas)} corridas programadas; {sum(ajuste.run_id(c.sistema, c.tipo, e) in hechas for c, e in tareas)} "
        f"ya hechas (reanudar={reanudar})")
    t0 = time.time()
    for i, (cfg, esq) in enumerate(tareas, 1):
        rid = ajuste.run_id(cfg.sistema, cfg.tipo, esq)
        if rid in hechas:
            log(f"[salta] {rid} (ya existe)")
            continue
        log(f"({i}/{len(tareas)}) {rid} ...")
        out = ajuste.correr(cfg, esq, priors, rutas, aj, log=log)
        reg = pd.DataFrame([{k: v for k, v in out["registro"].items() if k != "traceback"}])
        tablas["corridas"] = _upsert(tablas["corridas"], reg)
        tablas["diagnosticos"] = _upsert(tablas["diagnosticos"], reg.reindex(columns=COLS_DIAG))
        fechas = reg.reindex(columns=["run_id", "sistema", "tipo", "esquema", "epoch_format_in",
                                      "epoch_format_work", "epoch_format_out", "escala_tiempo",
                                      "anio_min", "anio_max", "fechas_plausibles"])
        tablas["fechas_aplicadas"] = _upsert(tablas["fechas_aplicadas"], fechas)
        if out["resumen"] is not None:
            res = out["resumen"].assign(sistema=cfg.sistema, tipo=cfg.tipo, esquema=esq)
            tablas["resumen_parametros"] = _upsert(tablas["resumen_parametros"], res)
            tablas["chi2_residuos"] = _upsert(tablas["chi2_residuos"], out["chi2"])
        if out["priors"]:
            pr = pd.DataFrame(out["priors"]).assign(run_id=rid)
            tablas["priors_usados"] = _upsert(tablas["priors_usados"], pr)
        for k, v in tablas.items():
            v.to_csv(arch[k], index=False)
    log(f"Corrida masiva terminada en {(time.time() - t0) / 60:.1f} min")
    return tablas
