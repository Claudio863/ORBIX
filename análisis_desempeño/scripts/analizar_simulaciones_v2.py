"""Auditoría reproducible de los resultados de las 100 simulaciones ORBIX v2.

Solo lee resultados existentes. No vuelve a ejecutar optimizaciones ni NUTS.
Escribe tablas numéricas y un resumen de auditoría en JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE.parent / "simulaciones"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def main() -> None:
    reg = pd.read_csv(RESULTADOS / "registro.csv")
    opt = pd.read_csv(RESULTADOS / "comparacion_busquedas.csv")
    recovery = pd.read_csv(RESULTADOS / "recuperacion_parametros.csv")

    # A case succeeds at the empirical-reference criterion if its better of two
    # starts is within 0.001 log-density units of the best observed reference.
    best = (opt.sort_values("lp_mejor", ascending=False)
            .groupby(["caso", "estrategia"], as_index=False).first())
    gaps: dict[str, pd.Series] = {}
    rows = []
    for strategy, group in best.groupby("estrategia"):
        per_case = group.set_index("caso")["gap_referencia"]
        gaps[strategy] = per_case
        all_rows = opt[opt.estrategia == strategy]
        rows.append(dict(
            estrategia=strategy,
            casos=len(per_case),
            referencias=int((per_case <= 0.001).sum()),
            gap_mediana=float(per_case.median()),
            gap_p95=float(per_case.quantile(0.95)),
            gap_max=float(per_case.max()),
            inicios=int(all_rows.n_inicios.sum()),
            finitos=int(all_rows.n_finitos.sum()),
            terminacion_normal=int(all_rows.n_terminacion_normal.sum()),
            segundos_opt=float(all_rows.segundos_opt.sum()),
        ))
    resumen = pd.DataFrame(rows).set_index("estrategia")

    strict_fail = reg[~reg.diagnosticos_estrictos_ok.astype(bool)]
    out_support = reg[~reg.verdad_en_soporte.astype(bool)]
    # Determine the exact violated bound(s) for every generated truth.
    support_violations = []
    for row in out_support.itertuples():
        folder = RESULTADOS / "casos" / row.caso
        truth = json.loads((folder / "verdad.json").read_text(encoding="utf-8"))
        data = json.loads((folder / "datos_stan.json").read_text(encoding="utf-8"))
        if not data["log_P_min"] < np.log(truth["P"]) < data["log_P_max"]:
            support_violations.append((row.caso, "P"))
        if not data["V0_min"] < truth["V0"] < data["V0_max"]:
            support_violations.append((row.caso, "V0"))
        for par in (["K"] if truth["tipo"] == "SB1" else ["K1", "K2"]):
            if not 0 < truth[par] < data[f"{par}_max"]:
                support_violations.append((row.caso, par))
    violated = pd.Series([p for _, p in support_violations]).value_counts().to_dict()

    recovery = recovery.merge(
        reg[["caso", "verdad_en_soporte", "diagnosticos_estrictos_ok"]],
        on="caso", suffixes=("", "_registro"), validate="many_to_one")
    eligible = recovery[
        recovery.diagnosticos_ok.astype(bool)
        & recovery.verdad_en_soporte.astype(bool)
    ]
    coverage_rows = []
    for par, group in eligible.groupby("parametro"):
        n = len(group)
        k = int(group.cubre.sum())
        lo, hi = wilson(k, n)
        coverage_rows.append(dict(parametro=par, n=n, cubiertos=k,
                                  cobertura=k / n, wilson_lo=lo, wilson_hi=hi))
    coverage = pd.DataFrame(coverage_rows).sort_values("parametro")

    summary_path = RESULTADOS / "resumen_optimizacion.csv"
    resumen.reset_index().to_csv(summary_path, index=False)
    coverage.to_csv(RESULTADOS / "cobertura_filtrada_soporte_diagnosticos.csv", index=False)

    total_pre = float(reg.segundos_preproceso_global.sum())
    failed_by_condition = (strict_fail.groupby("condicion").caso
                           .agg(lambda x: ", ".join(x)).to_dict())
    out_by_condition = (out_support.groupby("condicion").caso
                        .agg(lambda x: ", ".join(x)).to_dict())
    global_row, local_row = resumen.loc["global"], resumen.loc["local"]

    report = dict(casos=len(reg), diagnosticos_estrictos=int(reg.diagnosticos_estrictos_ok.sum()),
        fuera_soporte=len(out_support), violaciones=violated, segundos_preproceso=total_pre,
        referencia_empirica_global=int(global_row.referencias), referencia_empirica_local=int(local_row.referencias),
        inferencia_ejecutada=False)
    (RESULTADOS/'auditoria_resultados.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
