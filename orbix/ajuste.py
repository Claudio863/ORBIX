"""Corrida de un (sistema, tipo, esquema): datos → fechas → priors → L-BFGS → NUTS → CSV."""
from __future__ import annotations

import json
import shutil
import time
import traceback
import zlib
from tempfile import TemporaryDirectory
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd

from . import config, fechas, modelos, periodo, resumen, rv


@dataclass
class Ajustes:
    """Ajustes numéricos de la corrida (editables desde el notebook)."""
    n_inicios_SB1: int = 200          # arranques aleatorios de L-BFGS
    n_inicios_SB2: int = 300
    algoritmo: str = "lbfgs"
    iter_opt: int = 4000
    tol_obj: float = 1e-6
    tol_grad: float = 1e-6
    tol_param: float = 1e-6
    history_size: int = 20
    cadenas: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 1000
    adapt_delta: float = 0.95
    max_treedepth: int = 15
    semilla: int = 20261005
    umbral_rhat: float = 1.01
    hdi_prob: float = 0.95
    # Siembra del período para L-BFGS (Fase 1):
    #   "global": periodograma kepleriano → K picos + grilla log-uniforme en toda la ventana (orbix/periodo.py);
    #             ningún arranque usa literatura (tampoco e).
    #   "local" : comportamiento anterior: log P ~ N(ln P_init, 0.05) alrededor del pico de Lomb-Scargle y
    #             e inicial en e_lit ± 0.1 (se conserva solo para reproducir corridas viejas).
    siembra_P: str = "global"
    k_picos: int = 8                  # picos del periodograma a sembrar
    frac_picos: float = 0.6           # fracción de arranques en los picos; el resto, grilla log-uniforme
    refinar_picos: bool = True        # resolver submodos antes de perturbar las semillas
    n_frecuencias_refino: int = 41
    sig_figs: int = 18                # conservar precision de epocas absolutas en CSV
    umbral_chi2: float = 5.0          # flag_chi2_alto si χ²_red (MAP de las muestras) supera este valor
    extra: dict = field(default_factory=dict)


@dataclass
class Rutas:
    raiz: Path                         # corrida_masiva/ (las rutas de datos de sistemas.csv son relativas a esta)
    stan: Path                         # app/src/Variaciones_stan/
    resultados: Path | None = None     # por defecto raiz/resultados

    @property
    def _res(self) -> Path: return Path(self.resultados) if self.resultados else self.raiz / "resultados"
    @property
    def tablas(self) -> Path: return self._res / "tablas"
    @property
    def draws(self) -> Path: return self._res / "draws"
    @property
    def build(self) -> Path: return self.raiz / ".build"


def run_id(sistema: str, tipo: str, esquema: str) -> str:
    return f"{sistema}_{tipo}_{esquema}"


# --------------------------------------------------------------------------- inicializaciones
E_MAX = 0.999   # mismo factor que en los .stan: e = E_MAX r²/(1 + r²)


class Geometria:
    """Épocas de referencia que usan los .stan: min_t (origen de T0) y t_ref (media)."""
    def __init__(self, comps: dict):
        t = np.concatenate([d.epoch.values for d in comps.values()])
        err = np.concatenate([d.err.values for d in comps.values()])
        self.min_t = float(t.min())
        self.t_ref = float(np.average(t, weights=1/err**2))

    def parametros_muestreo(self, T0, w, e, P, fix_e) -> dict:
        """(T0, w, e) físicos → parámetros que muestrea el modelo (lam_ref y hk o w_raw)."""
        Mref = 2 * np.pi * ((self.t_ref - self.min_t) / P - T0)
        d = {"lam_ref": float(np.mod(Mref + w, 2*np.pi))}
        if fix_e:
            d["w_raw"] = float(w)
        else:
            e = float(np.clip(e, 1e-6, E_MAX - 1e-4))
            r = np.sqrt(e / (E_MAX - e))
            d["hk"] = [float(r * np.cos(w)), float(r * np.sin(w))]
        return d


def _interior(valor, minimo, maximo):
    margen = (maximo - minimo) * 1e-6
    return float(np.clip(valor, minimo + margen, maximo - margen))


def _acotar_inicio(ini, data):
    if "lam_ref" in ini:
        ini["lam_ref"] = float(np.clip(np.mod(ini["lam_ref"], 2*np.pi), 1e-10, 2*np.pi-1e-10))
    ini["V0"] = _interior(ini["V0"], data["V0_min"], data["V0_max"])
    if "log_P" in ini:
        ini["log_P"] = _interior(ini["log_P"], data["log_P_min"], data["log_P_max"])
    for k in ("K", "K1", "K2"):
        if k in ini:
            ini[k] = _interior(ini[k], 0.0, data[f"{k}_max"])
    return ini


def _generador_inicios(rng, tipo, esquema, comps, geo, P_ini, e_lit, data, plan=None, soluciones=None,
                       dias_a_work=None):
    """gen(i) → (inits, info). Sin `plan`: siembra "local" (anterior). Con `plan` (siembra "global"): el
    arranque i toma su período del plan (pico del periodograma o estrato de la grilla log-uniforme) y la mitad
    de los arranques de cada pico parte además de (e, T0, ω, V0, K) de la solución de la grilla en ese pico.
    `info` registra de dónde salió cada semilla (queda en inicios_lbfgs.csv)."""
    fix_P, fix_e = modelos.fijos(esquema)
    rv_all = pd.concat(list(comps.values()))
    semi = {c: float((d.RV.max() - d.RV.min()) / 2) for c, d in comps.items()}
    if plan is not None:
        e_lo, e_hi = 0.0, 0.95                     # global: e inicial sin literatura
    else:
        e_lo, e_hi = ((max(0.0, e_lit - 0.1), min(0.95, e_lit + 0.1)) if e_lit is not None and np.isfinite(e_lit)
                      else (0.02, 0.7))

    def gen(i=0):
        ini = {"V0": float(rng.uniform(rv_all.RV.min(), rv_all.RV.max()))}
        T0, w, e = rng.uniform(0, 1), rng.uniform(0, 2 * np.pi), rng.uniform(max(e_lo, 0.001), e_hi)
        k_fac = {c: rng.uniform(0.5, 1.5) for c in comps}
        K = {c: semi[c] * k_fac[c] + 1e-3 for c in comps}
        info = dict(inicio=i, origen="fijo" if fix_P else "local_LS", criterio="", usa_solucion_pico=False)
        P = P_ini
        if not fix_P:
            if plan is None:
                ini["log_P"] = float(np.log(P_ini) + rng.normal(0, 0.05))
            else:
                r = plan.iloc[i % len(plan)]
                info.update(origen=r.origen, criterio=r.get("criterio", ""))
                if r.origen == "grilla_log":
                    lnP_d = rng.uniform(r.lnP_lo, r.lnP_hi)
                else:
                    sol = soluciones[r.origen]
                    ancla = i < len(soluciones) and "P_d" in sol
                    lnP_d = np.log(sol.get("P_d", r.P_centro_d))
                    if not ancla:
                        lnP_d += rng.normal(0, r.sigma_lnP)
                    info["ancla_refinada"] = ancla
                    if ancla:
                        info["usa_solucion_pico"] = True
                        T0, w, e = sol["T0"], sol["w"], sol["e"]
                        ini["V0"] = sol["V0"]
                        K = {c: max(sol[f"K_{c}"], 1e-2) for c in comps}
                    elif rng.uniform() < 0.5:
                        info["usa_solucion_pico"] = True
                        T0 = np.mod(sol["T0"] + rng.normal(0, 0.02), 1.0)
                        w = np.mod(sol["w"] + rng.normal(0, 0.2), 2 * np.pi)
                        e = float(np.clip(sol["e"] + rng.uniform(-0.05, 0.05), 0.001, 0.95))
                        ini["V0"] = sol["V0"]
                        K = {c: max(sol[f"K_{c}"], 1e-2) * k_fac[c] ** 0.25 for c in comps}
                ini["log_P"] = float(np.log(dias_a_work(np.exp(lnP_d))))
            ini["log_P"] = _interior(ini["log_P"], data["log_P_min"], data["log_P_max"])
            P = float(np.exp(ini["log_P"]))
        ini.update(geo.parametros_muestreo(T0, w, e, P, fix_e))
        if tipo == "SB1":
            ini["K"] = K["v1"]
        else:
            ini["K1"], ini["K2"] = K["v1"], K["v2"]
        info.update(P_semilla=P, e_semilla=float(e) if not fix_e else np.nan, T0_semilla=float(T0),
                    w_semilla_deg=float(np.rad2deg(w)))
        return _acotar_inicio(ini, data), info
    return gen


_CACHE_PERIODOGRAMA: dict = {}


def busqueda_global(cfg, comps, pinfo, aj: "Ajustes") -> dict:
    """Periodograma kepleriano (en días) + picos + solución de grilla en cada pico. Se cachea por sistema/tipo
    y ventana, porque es el mismo para todos los esquemas con P libre."""
    t_d = {c: fechas.a_jd(d.epoch.values, cfg.epoch_format_work) for c, d in comps.items()}
    Pmin_d, Pmax_d = float(pinfo["ls_Pmin_d"]), float(pinfo["ls_Pmax_d"])
    clave = (cfg.sistema, cfg.tipo, Pmin_d, Pmax_d, aj.k_picos, aj.refinar_picos, aj.n_frecuencias_refino,
             tuple(zlib.crc32(np.ascontiguousarray(d[["epoch", "RV", "err"]].to_numpy()).tobytes())
                   for d in comps.values()))
    if clave not in _CACHE_PERIODOGRAMA:
        per = periodo.periodograma_kepleriano(comps, t_d, Pmin_d, Pmax_d)
        pk = periodo.picos_combinados(per, aj.k_picos)
        sols, refinamientos = {}, []
        for j, r in pk.iterrows():
            nombre = f"pico_{j+1}"
            if aj.refinar_picos:
                sol = periodo.refinar_pico(comps, t_d, float(r.P_d), per["t0_d"],
                                           per["base_d"], Pmin_d, Pmax_d, aj.n_frecuencias_refino)
                refinamientos.extend(dict(origen=nombre, **x) for x in sol.pop("intentos"))
            else:
                sol = periodo.solucion_en_periodo(comps, t_d, float(r.P_d), per["t0_d"])
            sols[nombre] = sol
        _CACHE_PERIODOGRAMA[clave] = dict(per, picos=pk, soluciones=sols, Pmin_d=Pmin_d, Pmax_d=Pmax_d,
                                        refinamiento=pd.DataFrame(refinamientos))
    return _CACHE_PERIODOGRAMA[clave]


def _chi2_red(params: dict, comps: dict, tipo: str, k: int) -> float:
    """χ²_red de un punto (P, T, e, w_deg, V0, K…) sobre todas las componentes, con k parámetros libres."""
    chi2, n = 0.0, 0
    for c, d in comps.items():
        modelo = rv.curvas(params, d.epoch.values, tipo)[c]
        chi2 += float(np.sum(((d.RV.values - modelo) / d.err.values) ** 2))
        n += len(d)
    return chi2 / (n - k) if n > k else np.nan


def _jitter(rng, best, tipo, esquema, geo, data):
    """Inicialización de una cadena: el mejor L-BFGS (en variables físicas) con una perturbación pequeña."""
    fix_P, fix_e = modelos.fijos(esquema)
    ini = {"V0": float(best["V0"] + rng.normal(0, 1e-2))}
    if not fix_P:
        ini["log_P"] = float(best["log_P"] + rng.normal(0, 1e-5))
        ini["log_P"] = _interior(ini["log_P"], data["log_P_min"], data["log_P_max"])
    P = float(np.exp(ini["log_P"])) if not fix_P else float(best["P"])
    ini.update(geo.parametros_muestreo(float(np.mod(best["T0"] + rng.normal(0, 1e-4), 1.0)),
                                       float(np.mod(best["w"] + rng.normal(0, 1e-3), 2 * np.pi)),
                                       float(best["e"]) * (1 + rng.normal(0, 1e-3)), P, fix_e))
    for k in (["K"] if tipo == "SB1" else ["K1", "K2"]):
        ini[k] = float(abs(best[k] * (1 + rng.normal(0, 1e-3))) + 1e-6)
    if "lam_ref" in best:
        ini["lam_ref"] = float(np.mod(best["lam_ref"] + rng.normal(0, 1e-5), 2*np.pi))
    return _acotar_inicio(ini, data)


# --------------------------------------------------------------------------- corrida
def _optimizar(model, **kwargs):
    kwargs.setdefault("sig_figs", 18)
    # CmdStanPy 1.2.x no reconoce el comentario final de optimize en CmdStan 2.40.
    with TemporaryDirectory(prefix="orbix_opt_") as output_dir:
        try:
            return model.optimize(output_dir=output_dir, **kwargs).optimized_params_dict
        except ValueError as exc:
            if "could not convert string to float: '# Optimization terminated normally:" not in str(exc):
                raise
            archivos = list(Path(output_dir).glob("*.csv"))
            if len(archivos) != 1:
                raise
            valores = pd.read_csv(archivos[0], comment="#")
            if valores.empty or "lp__" not in valores:
                raise
            return valores.iloc[-1].to_dict()


def correr(cfg: pd.Series, esquema: str, priors: pd.DataFrame, rutas: Rutas, aj: Ajustes,
           log=print) -> dict:
    """Corre un (sistema, tipo, esquema) completo y persiste resultados. Nunca lanza excepción:
    devuelve un registro con 'estado' = 'ok' | 'fallo' y 'motivo'."""
    rid = run_id(cfg.sistema, cfg.tipo, esquema)
    t_inicio = time.time()
    reg = dict(run_id=rid, sistema=cfg.sistema, tipo=cfg.tipo, esquema=esquema,
               modelo=modelos.nombre_modelo(cfg.tipo, esquema),
               epoch_format_in=cfg.epoch_format_in, epoch_format_work=cfg.epoch_format_work,
               epoch_format_out=cfg.get("epoch_format_out", cfg.epoch_format_work),
               escala_tiempo=fechas.formato(cfg.epoch_format_in).escala,
                 estado="fallo", motivo="", semilla=aj.semilla, cmdstan=".".join(map(str, cmdstanpy.cmdstan_version() or ())),
                 modelo_version="fase_acotada_tref_ponderado_20261027")
    reg.update(timeout_opt=aj.extra.get('timeout_opt'),timeout_sample=aj.extra.get('timeout_sample'))
    try:
        # 1) datos + fechas
        comps = {"v1": config.cargar_componente(rutas.raiz / cfg.datos_v1, cfg.epoch_format_in, cfg.epoch_format_work)}
        if cfg.tipo == "SB2":
            comps["v2"] = config.cargar_componente(rutas.raiz / cfg.datos_v2, cfg.epoch_format_in, cfg.epoch_format_work)
        val = fechas.validar_epocas(pd.concat(list(comps.values())).epoch, cfg.epoch_format_work)
        reg.update({f"N_{c}": len(d) for c, d in comps.items()}, anio_min=val["anio_min"], anio_max=val["anio_max"],
                   fechas_plausibles=val["plausible"])
        if not val["plausible"]:
            raise ValueError(f"épocas fuera de rango plausible tras la conversión: {val}")

        # 2) período inicial (registrado con su fuente)
        pinfo = config.periodo_inicial(cfg, comps["v1"])
        reg.update(pinfo)

        # 3) priors y datos Stan
        hiper, reg_priors = config.resolver_priors(priors, cfg, esquema, comps, pinfo)
        data = dict(N_v1=len(comps["v1"]), t_v1=comps["v1"].epoch.values, v1_obs=comps["v1"].RV.values,
                    v1_err=comps["v1"].err.values, **hiper)
        if cfg.tipo == "SB2":
            data.update(N_v2=len(comps["v2"]), t_v2=comps["v2"].epoch.values, v2_obs=comps["v2"].RV.values,
                        v2_err=comps["v2"].err.values)
        reg["priors"] = " | ".join(p["prior_resuelto"] for p in reg_priors)

        # 4) modelo
        model = modelos.compilar(cfg.tipo, esquema, rutas.stan, rutas.build)

        # 5) L-BFGS multi-arranque (mejor solución por lp__)
        rng = np.random.default_rng([aj.semilla, zlib.crc32(rid.encode())])   # reproducible por corrida
        e_lit = float(cfg.e_lit) if not config._es_vacio(cfg.get("e_lit")) else None
        P_ini = hiper.get("P_fixed", pinfo["P_init_work"])
        geo = Geometria(comps)
        n_ini = aj.n_inicios_SB1 if cfg.tipo == "SB1" else aj.n_inicios_SB2
        fix_P, fix_e = modelos.fijos(esquema)
        out = rutas.draws / rid
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        # Persistir entradas antes de ejecutar: también quedan disponibles si falla NUTS.
        for c, d in comps.items():
            d.to_csv(out / f"datos_{c}.csv", index=False)
        (out / 'datos_stan.json').write_text(json.dumps({k:(v.tolist() if hasattr(v,'tolist') else v)
            for k,v in data.items()},indent=1),encoding='utf-8')
        plan = sols = None
        global_ = aj.siembra_P == "global" and not fix_P
        reg["siembra_P"] = aj.siembra_P if not fix_P else "P_fijo"
        if global_:
            bg = busqueda_global(cfg, comps, pinfo, aj)
            plan = periodo.plan_semillas(n_ini, bg["picos"], bg["Pmin_d"], bg["Pmax_d"], bg["base_d"], aj.frac_picos)
            sols = bg["soluciones"]
            bg["tabla"].to_csv(out / "periodograma.csv", index=False)
            pk = bg["picos"].assign(**{f"sol_{k}": [sols[f"pico_{j + 1}"][k] for j in range(len(bg["picos"]))]
                                       for k in sols["pico_1"]})
            pk.to_csv(out / "picos_periodo.csv", index=False)
            if not bg["refinamiento"].empty:
                bg["refinamiento"].to_csv(out / "refinamiento_picos.csv", index=False)
            reg.update(P_pico1_d=float(bg["picos"].P_d.iloc[0]),
                       picos_P_d=" ".join(f"{x:.6g}" for x in bg["picos"].P_d))
        u_work = fechas.unidad_periodo(cfg.epoch_format_work)
        gen = _generador_inicios(rng, cfg.tipo, esquema, comps, geo, P_ini, e_lit, data, plan=plan, soluciones=sols,
                                 dias_a_work=lambda x: float(fechas.convertir_duracion(x, "d", u_work)))
        best, ok = None, 0
        ultimo_error = ""
        bitacora = []
        for i in range(n_ini):
            ini, info = gen(i)
            info["semilla_cmdstan"] = int(rng.integers(1, 2**31 - 1))
            try:
                d = _optimizar(model, data=data, inits=ini, algorithm=aj.algoritmo, iter=aj.iter_opt,
                                   tol_obj=aj.tol_obj, tol_grad=aj.tol_grad, tol_param=aj.tol_param,
                                   history_size=aj.history_size, require_converged=False, show_console=False,
                                   seed=info["semilla_cmdstan"], sig_figs=aj.sig_figs,
                                   timeout=aj.extra.get('timeout_opt'))
            except Exception as exc:
                ultimo_error = f"{type(exc).__name__}: {exc}"
                bitacora.append(dict(info, ok=False, error=ultimo_error[:200]))
                continue
            finito = bool(np.all(np.isfinite(list(d.values()))))
            bitacora.append(dict(info, ok=finito, lp=d.get("lp__"), P_final=d.get("P"), e_final=d.get("e")))
            if not finito:
                continue
            ok += 1
            if best is None or d["lp__"] > best["lp__"]:
                best = d
        bita = pd.DataFrame(bitacora)
        bita["P_unidad"] = u_work
        bita.to_csv(out / "inicios_lbfgs.csv", index=False)
        reg.update(n_inicios=n_ini, n_inicios_ok=ok)
        if best is None:
            raise RuntimeError("ninguna optimización L-BFGS terminó con valores finitos"
                               + (f"; ultimo error: {ultimo_error}" if ultimo_error else ""))
        reg.update(lp_lbfgs=float(best["lp__"]),
                   **{f"lbfgs_{k}": float(best[k]) for k in resumen.PARAMS_SALIDA[cfg.tipo] if k in best})
        k_libres = 4 + (not fix_P) + (not fix_e) + (cfg.tipo == "SB2")
        reg["chi2_red_lbfgs"] = _chi2_red(best, comps, cfg.tipo, k_libres)
        (out/'mejor_lbfgs.json').write_text(json.dumps(best,indent=2),encoding='utf-8')
        if bitacora and "lp" in bita:
            i_best = int(bita.lp.astype(float).idxmax())
            reg["origen_mejor_inicio"] = bita.origen[i_best]

        # 6) NUTS, guardando las cadenas
        fit = model.sample(data=data, chains=aj.cadenas, parallel_chains=aj.cadenas,
                           iter_warmup=aj.iter_warmup, iter_sampling=aj.iter_sampling,
                           adapt_delta=aj.adapt_delta, max_treedepth=aj.max_treedepth,
                           inits=[_jitter(rng, best, cfg.tipo, esquema, geo, data) for _ in range(aj.cadenas)],
                           seed=aj.semilla, output_dir=str(out), show_progress=False, show_console=False,
                           sig_figs=aj.sig_figs,timeout=aj.extra.get('timeout_sample'))
        # datos usados (para que el notebook 2 grafique sin recalcular nada)
        for c, d in comps.items():
            d.to_csv(out / f"datos_{c}.csv", index=False)
        (out / "datos_stan.json").write_text(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v)
                                                         for k, v in data.items()}, indent=1), encoding="utf-8")

        # 7) resumen, diagnósticos, χ²
        draws = resumen.leer_draws(out)
        formato_out = cfg.get("epoch_format_out", cfg.epoch_format_work)
        res = resumen.resumir(draws, cfg.tipo, cfg.epoch_format_work, formato_out, aj.hdi_prob,
                              epoca_referencia=geo.min_t)
        reg.update(sig_figs=aj.sig_figs, refinar_picos=aj.refinar_picos,
                   n_frecuencias_refino=aj.n_frecuencias_refino)
        diag = resumen.diagnosticar(draws, res, aj.max_treedepth, aj.umbral_rhat)
        chi, resid = resumen.chi2_residuos(res, comps, cfg.tipo, esquema)
        meta = {k: reg[k] for k in ("run_id", "sistema", "tipo", "esquema", "modelo", "epoch_format_in",
                                     "epoch_format_work", "epoch_format_out", "P_init_fuente")}
        (rutas.tablas / "por_corrida").mkdir(parents=True, exist_ok=True)
        res.insert(0, "run_id", rid)
        res.assign(**{k: v for k, v in meta.items() if k != "run_id"}).to_csv(
            rutas.tablas / "por_corrida" / f"{rid}_summary.csv", index=False)
        resid.assign(run_id=rid).to_csv(rutas.tablas / "por_corrida" / f"{rid}_residuos.csv", index=False)
        reg.update(diag)
        fila_chi = chi[chi.componente == ("total" if cfg.tipo == "SB2" else "v1")].iloc[0]
        reg.update(chi2_red=float(fila_chi.chi2_red), max_abs_z=float(fila_chi.max_abs_z),
                   flag_chi2_alto=bool(fila_chi.chi2_red > aj.umbral_chi2))
        problemas = [k for k in ("flag_rhat", "flag_divergencias", "flag_chi2_alto") if reg[k]]
        reg.update(estado="ok", motivo="" if not problemas else
                   ("no converge: " if not diag["convergencia_ok"] else "") + ", ".join(problemas))
        return dict(registro=reg, resumen=res, chi2=chi.assign(**meta), priors=reg_priors)
    except Exception as exc:  # noqa: BLE001 — se registra y se sigue con la siguiente corrida
        reg["motivo"] = f"{type(exc).__name__}: {exc}"
        reg["traceback"] = traceback.format_exc(limit=3)
        return dict(registro=reg, resumen=None, chi2=None, priors=[])
    finally:
        reg["segundos"] = round(time.time() - t_inicio, 1)
        log(f"[{reg['estado']:5s}] {rid:40s} {reg['segundos']:7.1f}s  {reg['motivo']}")
