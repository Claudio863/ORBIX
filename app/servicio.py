"""Adapt uploaded observations to the shared ORBIX inference pipeline."""
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
import hashlib
import json
import re
import shutil
import uuid
import zipfile

import numpy as np
import pandas as pd

from orbix import ajuste, config, fechas, modelos

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'app' / 'resultados'


def agrupar_archivos(archivos, tipo):
    """Validate an entire upload batch before allowing any inference."""
    groups = {}
    for filename, content in archivos:
        stem = Path(filename).stem
        match = re.fullmatch(r'(.+)_v([12])', stem)
        if tipo == 'SB2' and not match:
            raise ValueError('En SB2 use pares nombre_v1.csv y nombre_v2.csv.')
        name, component = (match[1], 'v'+match[2]) if tipo == 'SB2' else (stem, 'v1')
        name = re.sub(r'[^\w-]+', '_', name)[:80]
        if not name or component in groups.setdefault(name, {}):
            raise ValueError(f'Nombre de sistema vacío o componente duplicada: {filename}')
        groups[name][component] = leer_csv(content)
    for name, components in groups.items():
        if tipo == 'SB2' and set(components) != {'v1', 'v2'}:
            raise ValueError(f'{name}: falta una componente del par SB2.')
    return groups


def leer_csv(content):
    data = pd.read_csv(BytesIO(content))
    required = ['epoch', 'RV', 'err']
    if not set(required).issubset(data.columns):
        raise ValueError('El CSV debe contener las columnas epoch, RV y err.')
    data = data[required].apply(pd.to_numeric, errors='raise')
    if len(data) < 8:
        raise ValueError('Se requieren al menos ocho observaciones por componente.')
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError('Hay valores vacíos o no finitos en las observaciones.')
    if (data.err <= 0).any() or data.epoch.nunique() < 5:
        raise ValueError('Los errores deben ser positivos y se requieren cinco épocas distintas.')
    return data


def preparar(componentes, formato, sistema, tipo, esquema, pmin, pmax,
             p_fijo, e_fija, p_inicial, priors):
    if not 0 < pmin < pmax:
        raise ValueError('El período mínimo debe ser positivo y menor que el máximo.')
    fixed_p, fixed_e = modelos.fijos(esquema)
    if fixed_p and (not np.isfinite(p_fijo) or p_fijo <= 0):
        raise ValueError('El período fijado debe ser positivo.')
    if fixed_e and not 0 <= e_fija < .999:
        raise ValueError('La excentricidad fijada debe estar entre 0 y 0.999, excluido el extremo superior.')
    if tipo == 'SB2' and 'v2' not in componentes:
        raise ValueError('SB2 requiere los archivos de ambas componentes.')
    cfg = pd.Series(dict(sistema=sistema, tipo=tipo, datos_v1='entrada/v1.csv',
                         datos_v2='entrada/v2.csv', epoch_format_in=formato,
                         epoch_format_work=formato, epoch_format_out=formato,
                         P_lit=p_fijo if fixed_p else np.nan, P_lit_unit='d',
                         e_lit=e_fija if fixed_e else np.nan,
                         P_init=p_inicial, P_init_unit='d',
                         ls_Pmin_d=pmin, ls_Pmax_d=pmax,
                         fuente_lit='Valor fijado explícitamente por el usuario'))
    epocas = pd.concat(list(componentes.values())).epoch
    if not fechas.validar_epocas(epocas, formato)['plausible']:
        raise ValueError('Las épocas no son plausibles para el formato seleccionado.')
    info = config.periodo_inicial(cfg, componentes['v1'])
    for row in priors.itertuples():
        if row.familia == 'flat':
            continue
        first, second = float(row.p1), float(row.p2)
        if not np.isfinite([first, second]).all() or second <= 0:
            raise ValueError(f'Prior de {row.parametro}: parámetros finitos y anchura positiva requeridos.')
        if row.familia in ('beta', 'lognormal') and first <= 0:
            raise ValueError(f'Prior de {row.parametro}: el primer parámetro debe ser positivo.')
    hyper, registro = config.resolver_priors(priors, cfg, esquema, componentes, info)
    return cfg, info, hyper, pd.DataFrame(registro)


def ejecutar(componentes, cfg, esquema, priors, ajustes, log=print, *, etapa="preliminar", progreso=None):
    """Each request owns a unique output directory, including failed attempts."""
    RUNS.mkdir(parents=True, exist_ok=True)
    folder = RUNS / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    folder.mkdir()
    (folder / 'entrada').mkdir()
    for name, data in componentes.items():
        data.to_csv(folder / 'entrada' / f'{name}.csv', index=False)
    cfg.to_frame().T.to_csv(folder / 'configuracion.csv', index=False)
    priors.to_csv(folder / 'priors.csv', index=False)
    (folder / 'ajustes.json').write_text(json.dumps(asdict(ajustes), indent=2), encoding='utf-8')
    # Private compilation prevents simultaneous browser sessions from sharing
    # a generated executable/header in the source directory.
    model_dir = folder / '.build' / 'modelos'
    model_dir.mkdir(parents=True)
    for name in [modelos.nombre_modelo(cfg.tipo, esquema)+'.stan', 'kepler_eq.hpp']:
        shutil.copy2(ROOT/'modelos'/name, model_dir/name)
    rutas = ajuste.Rutas(raiz=folder, stan=model_dir)
    guardar_resultado(folder, ajuste.correr(cfg, esquema, priors, rutas, ajustes, log=log, etapa=etapa, progreso=progreso))
    provenance = {}
    for path in [ROOT/'orbix/ajuste.py', ROOT/'orbix/config.py',
                 ROOT/'modelos'/f'{modelos.nombre_modelo(cfg.tipo, esquema)}.stan',
                 ROOT/'modelos/kepler_eq.hpp', *sorted((folder/'entrada').glob('*.csv'))]:
        name = str(path.relative_to(ROOT))
        provenance[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder/'procedencia.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return folder


def guardar_resultado(folder, result):
    (folder / 'registro.json').write_text(json.dumps(result['registro'], indent=2, default=str), encoding='utf-8')
    if result['resumen'] is not None:
        result['resumen'].to_csv(folder / 'parametros.csv', index=False)
        result['chi2'].to_csv(folder / 'residuos_resumen.csv', index=False)
        pd.DataFrame(result['priors']).to_csv(folder / 'priors_efectivos.csv', index=False)


def muestrear(folder, log=print):
    folder = Path(folder)
    registro, *_ = cargar_resultado(folder)
    cfg = pd.read_csv(folder/'configuracion.csv', float_precision='round_trip').iloc[0]
    priors = pd.read_csv(folder/'priors.csv', dtype=str, keep_default_na=False)
    ajustes = ajuste.Ajustes(**json.loads((folder/'ajustes.json').read_text(encoding='utf-8')))
    rutas = ajuste.Rutas(raiz=folder, stan=folder/'.build/modelos')
    result = ajuste.correr(cfg, registro['esquema'], priors, rutas, ajustes, log=log, etapa='muestreo')
    guardar_resultado(folder, result)
    return result['registro']


def preliminar(folder, registro):
    out = Path(folder)/'resultados/draws'/registro['run_id']
    path = out/'preliminar.json'
    if not path.exists():
        return None, {}
    candidatos = json.loads(path.read_text(encoding='utf-8'))['candidatos']
    components = {name: pd.read_csv(out/f'datos_{name}.csv')
                  for name in ('v1', 'v2') if (out/f'datos_{name}.csv').exists()}
    return candidatos, components


def cargar_resultado(folder):
    folder = Path(folder)
    registro = json.loads((folder/'registro.json').read_text(encoding='utf-8'))
    if registro['estado'] != 'ok':
        return registro, None, None, None
    table = pd.read_csv(folder/'parametros.csv')
    chi = pd.read_csv(folder/'residuos_resumen.csv')
    residual = pd.read_csv(folder/'resultados/tablas/por_corrida'/f"{registro['run_id']}_residuos.csv")
    return registro, table, chi, residual


def diagnosticos_estrictos(registro):
    return (registro['R_hat_max'] <= 1.01 and registro['ESS_bulk_min'] >= 400
            and registro['ESS_tail_min'] >= 400 and registro['divergencias'] == 0
            and registro['treedepth_max'] == 0 and registro['ebfmi_min'] >= .3
            and registro['soporte_ok'])


def descargar(folder):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(folder).rglob('*')):
            if path.is_file() and '.build' not in path.parts:
                archive.write(path, path.relative_to(folder))
    return buffer.getvalue()
