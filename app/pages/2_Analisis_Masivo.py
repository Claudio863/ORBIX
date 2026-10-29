# -*- coding: utf-8 -*-
import os
import io
import sys
import json
import time
import zipfile
import tempfile
import shutil
from typing import List, Dict, Any

import streamlit as st

# Asegurar acceso a módulos en src/
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/'app'))
sys.path.insert(0,str(REPO/'app/src'))

from data_utils import load_user_data, estimate_period_lomb_scargle

# Intentar usar analisis_stan.py (raíz). Si no, caer a src/stan_analysis.py
try:
    from analisis_stan import analisis_stan_ as run_sb1
    try:
        from analisis_stan import analisis_stan_sb2 as run_sb2  # puede no existir
    except Exception:
        from stan_analysis import analisis_stan_sb2 as run_sb2
except Exception:
    from stan_analysis import analisis_stan_ as run_sb1
    from stan_analysis import analisis_stan_sb2 as run_sb2

# Nota: set_page_config suele llamarse en la app principal
# st.set_page_config(page_title='Análisis masivo', layout='wide')
st.title('Análisis masivo')

st.markdown(
    "Carga múltiples sistemas (SB1/SB2), define priors y parámetros clave, y ejecuta todos en lote.\n"
    "Al finalizar, descarga un ZIP con una carpeta por sistema.")

# -----------------------------
# Configuración global sencilla
# -----------------------------
with st.expander('Parámetros globales de optimización (opcional, minimalista)', expanded=False):
    iter_opt = st.number_input('Iteraciones (optimización)', min_value=200, max_value=10000, value=2000, step=100)
    algorithm = st.selectbox('Algoritmo', ['lbfgs', 'bfgs'], index=0)
    history_size = st.slider('History size', 5, 30, 15)
    tol_obj = st.number_input('Objective tolerance', min_value=1e-12, max_value=1e-3, value=1e-6, step=1e-7, format="%e")
    tol_grad = st.number_input('Gradient tolerance', min_value=1e-12, max_value=1e-3, value=1e-6, step=1e-7, format="%e")
    tol_param = st.number_input('Parameter tolerance', min_value=1e-12, max_value=1e-3, value=1e-6, step=1e-7, format="%e")

optim_config = dict(
    iter=iter_opt, algorithm=algorithm, history_size=history_size,
    tol_obj=tol_obj, tol_grad=tol_grad, tol_param=tol_param
)

# ---------------------------------
# Definición de sistemas a procesar
# ---------------------------------
st.subheader('Sistemas a cargar')

# -------------------------------------------------------------
# CARGA DE ARCHIVOS (requerido): múltiples CSV
# -------------------------------------------------------------
multi_files = st.file_uploader(
    'Sube uno o varios CSV de distintos sistemas', type=['csv'], accept_multiple_files=True, key='multi_csv'
)

# Opciones globales básicas aplicadas a cada sistema por defecto
auto_convert_epochs = st.checkbox('Convertir epoch a JD/HJD (por defecto)', value=True, key='auto_conv')
auto_epoch_format = st.selectbox('Formato de epoch (por defecto)',
                                 ('Heliocentric Julian Day (HJD)', 'Julian Day (JD)', 'Years (yr)'),
                                 key='auto_fmt')

# Agrupar archivos por nombre base (detecta V1/V2) y cargar a DataFrame
autogen_sistemas = []
if multi_files:
    def split_base_role(filename: str):
        name = filename.lower()
        base = name.rsplit('.', 1)[0]
        role = 'v1'
        if 'v2' in base:
            role = 'v2'
        for suf in ['_v1', '-v1', ' v1', '_v2', '-v2', ' v2']:
            base = base.replace(suf, '')
        return base, role

    grouped = {}
    for f in multi_files:
        base, role = split_base_role(f.name)
        if base not in grouped:
            grouped[base] = {'name': base, 'csv_v1': None, 'csv_v2': None}
        if role == 'v2':
            grouped[base]['csv_v2'] = f
        else:
            grouped[base]['csv_v1'] = f

    # Cargar cada par V1/V2 a DataFrame y construir autogen_sistemas
    for base, info in grouped.items():
        df_v1 = None
        df_v2 = None

        if info['csv_v1'] is not None:
            try:
                df_v1 = load_user_data(info['csv_v1'], auto_convert_epochs, auto_epoch_format)
            except Exception as e:
                st.error(f"{base}: error leyendo CSV V1 -> {e}")

        if info['csv_v2'] is not None:
            try:
                df_v2 = load_user_data(info['csv_v2'], auto_convert_epochs, auto_epoch_format)
            except Exception as e:
                st.error(f"{base}: error leyendo CSV V2 -> {e}")

        is_sb2 = df_v2 is not None

        autogen_sistemas.append({
            'name': base,
            'type': 'SB2' if is_sb2 else 'SB1',
            'convert_epochs': auto_convert_epochs,
            'epoch_format': auto_epoch_format,

            # Guardamos también los archivos originales por compatibilidad
            'csv_v1': info['csv_v1'],
            'csv_v2': info['csv_v2'],

            # ⬇️ DataFrames ya cargados
            'df_v1': df_v1,
            'df_v2': df_v2,

            # Priors por defecto (como antes)
            'method_P': 'Automático (Lomb–Scargle)',
            'P_init': None,
            'fix_P': False,
            'e_mode': 'Valor único',
            'e_init': 0.3,
            'e_min': None,
            'e_max': None,
            'fix_e': False,
            'q_init': 0.5,
            'asin_i_init': 0.1,
            'plx_init': 0.01,
        })

    st.success(f"Detectados {len(autogen_sistemas)} sistema(s) y CSVs cargados a DataFrame.")
else:
    st.info('Sube al menos un CSV para comenzar.')

# Al haber archivos, el modo autogenerado está activo
autogen_active = bool(multi_files)
#st.write(autogen_sistemas)
# -------------------------------------------------------------
# Controles por sistema (se crean al cargar CSVs)
# -------------------------------------------------------------
if autogen_active:
    for s in autogen_sistemas:
        sys_id = s['name']
        default_type = 'SB2' if s['csv_v2'] is not None else 'SB1'
        with st.expander(f"Parámetros: {sys_id}", expanded=False):
            disp_name = st.text_input('Nombre del sistema', value=sys_id, key=f'{sys_id}_name')
            s['name'] = disp_name or sys_id
            s['type'] = st.radio('Tipo de sistema', ['SB1', 'SB2'], index=(0 if default_type=='SB1' else 1), key=f'{sys_id}_type')
            
            s['convert_epochs'] = st.checkbox('Convertir epoch a JD/HJD', value=s['convert_epochs'], key=f'{sys_id}_conv')
            s['epoch_format'] = st.selectbox('Formato de epoch',
                                             ('Heliocentric Julian Day (HJD)', 'Julian Day (JD)', 'Years (yr)'),
                                             index=(0 if s['epoch_format'].startswith('Heliocentric') else (1 if s['epoch_format'].startswith('Julian') else 2)),
                                             key=f'{sys_id}_fmt')
            #st.write(s)
            st.dataframe(s["df_v1"])
            st.write(s["convert_epochs"])
            st.markdown('— Priors —')
            s['method_P'] = st.radio('Periodo (P)', ['Manual', 'Automático (Lomb–Scargle)'], index=(1 if s['method_P'].startswith('Auto') else 0), key=f'{sys_id}_mp')
            if s['method_P'] == 'Manual':
                s['P_init'] = st.number_input('P estimado [días]', min_value=0.0, max_value=1e7, value=(s['P_init'] or 100.0), step=0.000001, format='%.6f', key=f'{sys_id}_P')
            else:
                s['P_init'] = None
                st.caption('Se estimará con Lomb–Scargle al ejecutar.')
            s['fix_P'] = st.checkbox('Fijar P (sin muestrear)', value=s['fix_P'], key=f'{sys_id}_fixP')

            s['e_mode'] = st.radio('Excentricidad (e)', ['Valor único', 'Rango'], index=(0 if s['e_mode']=='Valor único' else 1), key=f'{sys_id}_em')
            if s['e_mode'] == 'Valor único':
                s['e_init'] = st.number_input('e estimado', min_value=0.0, max_value=1.0, value=s['e_init'], step=0.000001, format='%.6f', key=f'{sys_id}_e')
                s['e_min'], s['e_max'] = None, None
            else:
                e_pair = st.slider('Intervalo de e', min_value=0.0, max_value=1.0, value=(s['e_min'] or 0.2, s['e_max'] or 0.4), step=0.01, key=f'{sys_id}_es')
                s['e_min'], s['e_max'] = e_pair
                s['e_init'] = 0.5 * (s['e_min'] + s['e_max'])
            s['fix_e'] = st.checkbox('Fijar e (sin muestrear)', value=s['fix_e'], key=f'{sys_id}_fixE')

            if s['type'] == 'SB2':
                s['q_init'] = st.number_input('q = M2/M1', min_value=0.01, max_value=2.0, value=s['q_init'], step=0.01, format='%.3f', key=f'{sys_id}_q')
                s['asin_i_init'] = st.number_input('a*sin(i) (arcsec)', min_value=0.001, max_value=10.0, value=s['asin_i_init'], step=0.001, format='%.3f', key=f'{sys_id}_asi')
                s['plx_init'] = st.number_input('Parallax (arcsec)', min_value=0.001, max_value=1.0, value=s['plx_init'], step=0.001, format='%.3f', key=f'{sys_id}_plx')
num_sistemas = st.number_input('Número de sistemas', min_value=1, max_value=20, value=1, step=1, disabled=autogen_active)

sistemas: List[Dict[str, Any]] = []

for i in range(0 if autogen_active else int(num_sistemas)):
    with st.expander(f'Sistema #{i+1}', expanded=True if i == 0 else False):
        sistema: Dict[str, Any] = {}
        sistema['name'] = st.text_input('Nombre del sistema', value=f'System_{i+1}', key=f'name_{i}')
        sistema['type'] = st.radio('Tipo de sistema', ['SB1', 'SB2'], key=f'type_{i}')

        # Epoch handling
        sistema['convert_epochs'] = st.checkbox('Convertir epoch a JD/HJD', value=True, key=f'conv_{i}')
        sistema['epoch_format'] = st.selectbox('Formato de epoch',
                                               ('Heliocentric Julian Day (HJD)', 'Julian Day (JD)', 'Years (yr)'),
                                               key=f'fmt_{i}')
        
        # Uploaders
        sistema['csv_v1'] = st.file_uploader('CSV V1', type=['csv'], key=f'v1_{i}')
        if sistema['type'] == 'SB2':
            sistema['csv_v2'] = st.file_uploader('CSV V2', type=['csv'], key=f'v2_{i}')
        else:
            sistema['csv_v2'] = None

        st.markdown('---')
        st.markdown('Priors iniciales')
        # Period
        sistema['method_P'] = st.radio('Periodo (P):', ['Manual', 'Automático (Lomb–Scargle)'], key=f'mp_{i}')
        if sistema['method_P'] == 'Manual':
            sistema['P_init'] = st.number_input('P estimado [días]', min_value=0.0, max_value=1e7, value=100.0,
                                                step=0.000001, format='%.6f', key=f'P_{i}')
        else:
            sistema['P_init'] = None  # se estimará más adelante
            st.caption('Se estimará con Lomb–Scargle al ejecutar.')
        sistema['fix_P'] = st.checkbox('Fijar P (sin muestrear)', value=False, key=f'fixP_{i}')

        # Eccentricity
        sistema['e_mode'] = st.radio('Excentricidad (e):', ['Valor único', 'Rango'], key=f'ev_{i}')
        if sistema['e_mode'] == 'Valor único':
            sistema['e_init'] = st.number_input('e estimado', min_value=0.0, max_value=1.0, value=0.3,
                                                step=0.000001, format='%.6f', key=f'e_{i}')
            sistema['e_min'] = None
            sistema['e_max'] = None
        else:
            e_min, e_max = st.slider('Intervalo de e', min_value=0.0, max_value=1.0, value=(0.2, 0.4), step=0.01,
                                     key=f'es_{i}')
            sistema['e_init'] = 0.5 * (e_min + e_max)
            sistema['e_min'] = e_min
            sistema['e_max'] = e_max
        sistema['fix_e'] = st.checkbox('Fijar e (sin muestrear)', value=False, key=f'fixE_{i}')

        # SB2 extras
        if sistema['type'] == 'SB2':
            sistema['q_init'] = st.number_input('q = M2/M1', min_value=0.01, max_value=2.0, value=0.5,
                                                step=0.01, format='%.3f', key=f'q_{i}')
            sistema['asin_i_init'] = st.number_input('a*sin(i) (arcsec)', min_value=0.001, max_value=10.0, value=0.1,
                                                     step=0.001, format='%.3f', key=f'asi_{i}')
            sistema['plx_init'] = st.number_input('Parallax (arcsec)', min_value=0.001, max_value=1.0, value=0.01,
                                                   step=0.001, format='%.3f', key=f'plx_{i}')
        sistemas.append(sistema)

st.markdown('---')
run_btn = st.button('Ejecutar análisis masivo', type='primary')

if run_btn:
    # Preferir sistemas autogenerados si el usuario subió múltiples CSV y lo seleccionó
    sistemas_to_run = autogen_sistemas if (autogen_active and autogen_sistemas) else sistemas
    if not sistemas_to_run:
        st.warning('No hay sistemas definidos.')
        st.stop()

    # Directorio temporal maestro
    master_tmp = tempfile.mkdtemp(prefix='batch_')
    master_zip_path = os.path.join(master_tmp, 'batch_results.zip')

    overall_progress = st.progress(0.0)
    status_area = st.empty()

    exitos = 0
    for idx, cfg in enumerate(sistemas_to_run):
        sys_name = cfg['name'] or f'System_{idx+1}'
        status_area.info(f'Procesando {sys_name} ({cfg["type"]})...')

        # Preparar carpeta de resultados por sistema
        sys_dir = os.path.join(master_tmp, sys_name)
        os.makedirs(sys_dir, exist_ok=True)

        try:
            # Cargar datos
            if cfg.get('csv_v1') is None or (cfg['type'] == 'SB2' and cfg.get('csv_v2') is None):
                st.warning(f"{sys_name}: faltan CSVs requeridos.")
                continue

            df_v1 = load_user_data(cfg['csv_v1'], cfg['convert_epochs'], cfg['epoch_format'])
            df_v2 = None
            if cfg['type'] == 'SB2' and cfg['csv_v2'] is not None:
                df_v2 = load_user_data(cfg['csv_v2'], cfg['convert_epochs'], cfg['epoch_format'])

            # Periodo inicial
            P_init = cfg.get('P_init')
            if P_init is None:
                # Si no se convierte, estimate_period_lomb_scargle asume 'HJD' dummy como en App.py
                epoch_fmt_for_ls = cfg['epoch_format'] if cfg['convert_epochs'] else 'HJD'
                P_init = estimate_period_lomb_scargle(df_v1, epoch_fmt_for_ls)

            # Eccentricity desfase
            if cfg['e_mode'] == 'Valor único':
                e_init = cfg['e_init']
                desfase_min, desfase_max = -0.05 + e_init, 0.05 + e_init
            else:
                e_init = 0.0
                desfase_min, desfase_max = cfg.get('e_min'), cfg.get('e_max')

            # Limpieza session_state para aislar cada corrida
            for key in ['resultados_generados', 'pdf_data', 'rar_data', 'pdf_filename', 'rar_filename']:
                st.session_state.pop(key, None)

            # Ejecutar análisis
            V_delta = df_v1.RV.max() - df_v1.RV.min()
            if cfg['type'] == 'SB1':
                # Soportar ambas firmas (con/sin optim_config)
                try:
                    run_sb1(
                        df_v1=df_v1,
                        P_init=P_init,
                        e_init=e_init,
                        desfase_max=desfase_max,
                        desfase_min=desfase_min,
                        V_delta=V_delta,
                        system=sys_name,
                        nombre_componente='1',
                        number_component='1',
                        fix_e=cfg['fix_e'],
                        fix_P=cfg['fix_P'],
                        optim_config=optim_config
                    )
                except TypeError:
                    # Fallback para analisis_stan.py antiguo
                    run_sb1(
                        df_v1, sys_name, P_init, '1', V_delta, e_init,
                        desfase_min, desfase_max, '1', cfg['fix_P'], cfg['fix_e']
                    )
            else:
                if df_v2 is None or df_v2.empty:
                    st.warning(f"{sys_name}: SB2 requiere CSV V2.")
                    continue
                try:
                    run_sb2(
                        df_v1=df_v1,
                        df_v2=df_v2,
                        P_init=P_init,
                        e_init=e_init,
                        desfase_max=desfase_max,
                        desfase_min=desfase_min,
                        V_delta=V_delta,
                        system=sys_name,
                        q_init=cfg.get('q_init', 0.5),
                        asin_i_init=cfg.get('asin_i_init', 0.1),
                        plx_init=cfg.get('plx_init', 0.01),
                        fix_e=cfg['fix_e'],
                        fix_P=cfg['fix_P'],
                        optim_config=optim_config
                    )
                except TypeError:
                    # Si existiese una versión antigua con firma distinta, aquí se podría añadir otro fallback
                    run_sb2(
                        df_v1=df_v1,
                        df_v2=df_v2,
                        P_init=P_init,
                        e_init=e_init,
                        desfase_max=desfase_max,
                        desfase_min=desfase_min,
                        V_delta=V_delta,
                        system=sys_name,
                        q_init=cfg.get('q_init', 0.5),
                        asin_i_init=cfg.get('asin_i_init', 0.1),
                        plx_init=cfg.get('plx_init', 0.01),
                        fix_e=cfg['fix_e'],
                        fix_P=cfg['fix_P'],
                    )

            # Recuperar ZIP generado por el módulo de análisis
            if 'rar_data' in st.session_state and 'rar_filename' in st.session_state:
                # Guardar zip del sistema
                sys_zip_path = os.path.join(sys_dir, st.session_state.rar_filename)
                with open(sys_zip_path, 'wb') as f:
                    f.write(st.session_state.rar_data)
                # Extraer contenido dentro de la carpeta del sistema para el ZIP maestro
                try:
                    with zipfile.ZipFile(io.BytesIO(st.session_state.rar_data), 'r') as zf:
                        zf.extractall(sys_dir)
                except Exception:
                    pass

                # Guardar metadatos de configuración
                meta = {
                    'name': sys_name,
                    'type': cfg['type'],
                    'priors': {
                        'P_init': P_init,
                        'e_mode': cfg['e_mode'],
                        'e_init': cfg['e_init'],
                        'e_min': cfg['e_min'],
                        'e_max': cfg['e_max'],
                        'fix_P': cfg['fix_P'],
                        'fix_e': cfg['fix_e']
                    },
                    'epoch': {
                        'convert': cfg['convert_epochs'],
                        'format': cfg['epoch_format']
                    }
                }
                if cfg['type'] == 'SB2':
                    meta['sb2'] = {
                        'q_init': cfg.get('q_init', 0.5),
                        'asin_i_init': cfg.get('asin_i_init', 0.1),
                        'plx_init': cfg.get('plx_init', 0.01)
                    }
                with open(os.path.join(sys_dir, 'config.json'), 'w', encoding='utf-8') as jf:
                    json.dump(meta, jf, ensure_ascii=False, indent=2)

                exitos += 1
            else:
                st.warning(f"{sys_name}: no se encontró ZIP de salida en session_state.")

        except Exception as e:
            st.error(f"Error procesando {sys_name}: {str(e)}")
        finally:
            # Limpieza del estado para evitar colisiones entre sistemas
            for key in ['resultados_generados', 'pdf_data', 'rar_data', 'pdf_filename', 'rar_filename']:
                st.session_state.pop(key, None)

    total_to_run = len(sistemas_to_run) if sistemas_to_run else 1
    overall_progress.progress((idx + 1) / total_to_run)

    # Construir ZIP maestro con todas las carpetas
    final_zip_name = 'batch_results.zip'
    final_zip_bytes = io.BytesIO()
    with zipfile.ZipFile(final_zip_bytes, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(master_tmp):
            # saltar el propio zip si existe
            for f in files:
                fp = os.path.join(root, f)
                if fp == master_zip_path:
                    continue
                arcname = os.path.relpath(fp, master_tmp)
                zf.write(fp, arcname)
    final_zip_bytes.seek(0)

    st.success(f'Análisis masivo completado. Sistemas exitosos: {exitos}/{int(num_sistemas)}')
    st.download_button(
        label='Descargar resultados (ZIP)',
        data=final_zip_bytes.getvalue(),
        file_name=final_zip_name,
        mime='application/zip'
    )

    # Limpieza del directorio temporal al final de la sesión: opcional
    # shutil.rmtree(master_tmp, ignore_errors=True)
