"""Streamlit controls for the current shared SB1/SB2 inference pipeline."""
from pathlib import Path
import re
from time import monotonic

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from orbix import ajuste, fechas, modelos, rv
from app import servicio

SCHEMES = {'P y e libres': 'free', 'P fijo': 'fix_P',
           'e fija': 'fix_e', 'P y e fijados': 'fix_eP'}
FORMATS = {'Año besseliano': 'year_besselian', 'Año juliano': 'year_julian',
           'HJD': 'HJD', 'BJD': 'BJD', 'JD': 'JD',
           'RJD (JD − 2400000)': 'RJD', 'MJD (JD − 2400000.5)': 'MJD',
           'TJD (JD − 2450000)': 'TJD'}
LABELS = {'P': 'Período P', 'e': 'Excentricidad e', 'T0': 'Fase del periastro',
          'V0': 'Velocidad sistémica γ', 'K': 'Semiamplitud K',
          'K1': 'Semiamplitud K₁', 'K2': 'Semiamplitud K₂'}


def controles_priors(tipo, esquema, components, pmin, pmax):
    fixed_p, fixed_e = modelos.fijos(esquema)
    parameters = ['P', 'e', 'T0', 'V0'] + (['K'] if tipo == 'SB1' else ['K1', 'K2'])
    rows = []
    for parameter in parameters:
        fixed = (parameter == 'P' and fixed_p) or (parameter == 'e' and fixed_e)
        if fixed:
            continue
        title = LABELS[parameter]
        options = {'Uniforme en el soporte': 'flat'}
        if parameter == 'P':
            options = {'Loguniforme en la ventana': 'flat', 'Lognormal': 'lognormal'}
        elif parameter in ['e', 'T0']:
            options['Beta'] = 'beta'
        else:
            options['Normal'] = 'normal'
            if parameter.startswith('K'):
                options['Lognormal'] = 'lognormal'
        left, middle, right = st.columns([2, 1, 1])
        family = options[left.selectbox(title, list(options), key=f'prior_{parameter}')]
        first = second = ''
        if family != 'flat':
            data = components.get('v2' if parameter == 'K2' else 'v1')
            center = float(np.sqrt(pmin*pmax)) if parameter == 'P' else 1.0
            spread = .5 if family == 'lognormal' else 1.0
            if data is not None and parameter.startswith('K'):
                center = max(float(np.ptp(data.RV))/2, .1)
                spread = 1.0 if family == 'lognormal' else center
            if data is not None and parameter == 'V0':
                center = float(np.average(data.RV, weights=1/data.err**2))
                spread = max(float(np.ptp(data.RV)), 10.)
            first_label = 'α' if family == 'beta' else ('Mediana' if family == 'lognormal' else 'Media')
            second_label = 'β' if family == 'beta' else ('σ de log' if family == 'lognormal' else 'σ')
            first = str(middle.number_input(f'{first_label} · {title}', value=center,
                                            format='%.8g', key=f'center_{parameter}_{family}'))
            second = str(right.number_input(f'{second_label} · {title}', min_value=.000001,
                                            value=spread, format='%.8g', key=f'width_{parameter}_{family}'))
        rows.append(dict(sistema='*', tipo=tipo, parametro=parameter, familia=family,
                         p1=first, p2=second, unidad='d' if parameter == 'P' else '',
                         fuente='Configuración explícita de la interfaz'))
    st.caption('P se configura en días. γ y K se expresan en km/s. Los valores sugeridos '
               'para γ y K se calculan con las observaciones. La fase está en [0, 1) y ω es uniforme en [0, 2π).')
    return pd.DataFrame(rows)


def grafico_datos(components, formato):
    fig, ax = plt.subplots(figsize=(9, 3))
    for name, data in components.items():
        ax.errorbar(data.epoch, data.RV, yerr=data.err, fmt='o', ms=3,
                    label='Primaria' if name == 'v1' else 'Secundaria')
    ax.set_xlabel(f'Época ({formato})')
    ax.set_ylabel('RV (km/s)')
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def mostrar_resultado(folder):
    registro, table, chi, residual = servicio.cargar_resultado(folder)
    st.subheader(f"Resultados · {registro['sistema']} · {registro['tipo']}")
    st.caption(f"Modelo {registro['modelo']} · {registro['esquema']} · Semilla {registro['semilla']}")
    if table is None:
        if registro.get('fallo_entorno') or 'WinError 4551' in registro['motivo']:
            st.error('Windows bloqueó el ejecutable de Stan mediante Control de aplicaciones '
                     '(WinError 4551). El ajuste no pudo ejecutarse: este resultado no indica '
                     'una falla del método de optimización.')
        else:
            st.error('La ejecución terminó con un error: ' + registro['motivo'])
    else:
        passed = servicio.diagnosticos_estrictos(registro)
        if passed:
            st.success('La ejecución cumple los criterios estrictos de muestreo.')
        else:
            st.warning('La ejecución terminó con alertas de muestreo. Revise los diagnósticos antes de interpretar los intervalos.')
        columns = st.columns(4)
        columns[0].metric('χ² reducido', f"{registro['chi2_red']:.3f}")
        columns[1].metric('R̂ máximo', f"{registro['R_hat_max']:.4f}")
        columns[2].metric('ESS mínimo', f"{min(registro['ESS_bulk_min'], registro['ESS_tail_min']):.0f}")
        columns[3].metric('Divergencias', registro['divergencias'])
        tab_parameters, tab_curves, tab_diagnostics, tab_priors = st.tabs(
            ['Parámetros', 'Curvas y residuos', 'Diagnósticos', 'Configuración ejecutada'])
        with tab_parameters:
            shown = table[['parametro', 'unidad_out', 'mediana_out', 'hpdi_inf_out', 'hpdi_sup_out', 'fijo']].copy()
            shown.loc[shown.fijo, ['hpdi_inf_out', 'hpdi_sup_out']] = np.nan
            shown.columns = ['Parámetro', 'Unidad', 'Mediana / valor fijo', 'HPDI 95% inferior', 'HPDI 95% superior', 'Fijado']
            st.dataframe(shown, hide_index=True, width='stretch', column_config={
                name: st.column_config.NumberColumn(format='%.6g') for name in
                ['Mediana / valor fijo', 'HPDI 95% inferior', 'HPDI 95% superior']})
            st.caption('Los intervalos son HPDI del 95%. En parámetros fijados se omite el intervalo.'
                       + (' q = K₁/K₂ se calcula para cada muestra conjunta.' if registro['tipo'] == 'SB2' else ''))
        with tab_curves:
            point = table.set_index('parametro').MAP.to_dict()
            phase = np.linspace(0, 1, 1200)
            fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1]})
            for name, group in residual.groupby('componente'):
                phases = rv.fase(group.epoch.to_numpy(), point['P'], point['T'])
                label = 'Primaria' if name == 'v1' else 'Secundaria'
                color = '#0072B2' if name == 'v1' else '#D55E00'
                axes[0].errorbar(phases, group.RV, yerr=group.err, fmt='o', ms=3, color=color, label=label)
                axes[0].plot(phase, rv.curvas(point, point['T']+phase*point['P'], registro['tipo'])[name], color=color)
                axes[1].scatter(phases, group.residuo_norm, s=10, color=color)
            axes[0].set_ylabel('RV (km/s)')
            axes[0].legend()
            axes[1].axhspan(-3, 3, color='gray', alpha=.15)
            axes[1].axhline(0, color='gray', lw=.7)
            axes[1].set_ylabel('Residuo / error')
            axes[1].set_xlabel('Fase orbital')
            fig.tight_layout()
            st.pyplot(fig)
            fig.savefig(Path(folder)/'curvas_residuos.png', dpi=180)
            plt.close(fig)
            st.caption('Curvas evaluadas en la muestra conjunta de mayor densidad retenida. '
                       'El sombreado corresponde a residuos normalizados entre −3 y +3.')
            st.dataframe(chi[['componente', 'N', 'chi2_red', 'rms_residuo', 'media_residuo']], hide_index=True)
        with tab_diagnostics:
            st.dataframe(table[['parametro', 'R_hat', 'ESS_bulk', 'ESS_tail']], hide_index=True)
            st.write(f"E-BFMI mínimo: {registro['ebfmi_min']:.3f}. Saturaciones de profundidad: {registro['treedepth_max']}.")
            st.caption('Criterios: R̂ ≤ 1.01, ESS bulk y tail ≥ 400, cero divergencias, '
                       'cero saturaciones, E-BFMI ≥ 0.3 y soporte físico válido. χ² reducido evalúa la calidad residual por separado.')
        with tab_priors:
            st.dataframe(pd.read_csv(Path(folder)/'priors_efectivos.csv'), hide_index=True)
            st.write(f"Período inicial procedente de {registro['P_init_fuente']}. "
                     f"Ventana: {registro['ls_Pmin_d']:g}–{registro['ls_Pmax_d']:g} días.")
    st.download_button('Descargar resultados completos', servicio.descargar(folder),
                       file_name=f"ORBIX_{registro['run_id']}.zip", key=f'download_{Path(folder).name}')


def renderizar():
    st.title('ORBIX')
    st.write('Ajuste orbital de binarias espectroscópicas')
    st.caption('Carga de observaciones · búsqueda del período · L-BFGS · muestreo NUTS · resultados reproducibles')
    components = {}
    with st.expander('1. Datos observacionales', expanded=True):
        tipo = st.radio('Tipo de sistema', ['SB1', 'SB2'], horizontal=True)
        system_name = st.text_input('Nombre del sistema', value='HIP89000')
        formato = FORMATS[st.selectbox('Formato de las épocas', list(FORMATS))]
        st.caption('Archivos CSV con epoch, RV y err. Velocidades y errores en km/s. '
                   'La etiqueta temporal declara la escala de las fechas suministradas.')
        try:
            left, right = st.columns(2)
            first = left.file_uploader('Componente primaria', type='csv', key='upload_v1')
            second = right.file_uploader('Componente secundaria', type='csv', key='upload_v2') if tipo == 'SB2' else None
            if first:
                components['v1'] = servicio.leer_csv(first.getvalue())
            if second:
                components['v2'] = servicio.leer_csv(second.getvalue())
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(str(exc))
            components = {}
        if components:
            st.write(' · '.join(f"{'Primaria' if k == 'v1' else 'Secundaria'}: {len(v)} observaciones" for k, v in components.items()))
            grafico_datos(components, formato)
    with st.sidebar.expander('2. Esquema y búsqueda del período', expanded=True):
        esquema = SCHEMES[st.selectbox('Parámetros fijados', list(SCHEMES))]
        fixed_p, fixed_e = modelos.fijos(esquema)
        left, right = st.columns(2)
        pmin = left.number_input('Período mínimo de búsqueda (días)', min_value=.00001, value=30., format='%.8g')
        pmax = right.number_input('Período máximo de búsqueda (días)', min_value=.00001, value=1000., format='%.8g')
        p_fixed = left.number_input('Período fijado (días)', min_value=.00001, value=199.4, format='%.10g') if fixed_p else np.nan
        e_fixed = right.number_input('Excentricidad fijada', min_value=0., max_value=.998999, value=.3, format='%.8g') if fixed_e else np.nan
        manual = st.checkbox('Indicar un período inicial manual')
        p_initial = st.number_input('Período inicial (días)', min_value=.00001, value=199.4) if manual else np.nan
        st.caption('La ventana define el dominio de búsqueda. ' +
                   ('K se estima con el prior elegido.' if tipo == 'SB1' else 'K₁ y K₂ se estiman y q se deriva de su razón.'))
    with st.sidebar.expander('3. Priors', expanded=True):
        priors = controles_priors(tipo, esquema, components, pmin, pmax)
    with st.sidebar.expander('4. Configuración de ejecución', expanded=False):
        starts = st.number_input('Arranques de L-BFGS', min_value=8, max_value=1000, value=200 if tipo == 'SB1' else 300)
        warmup = st.number_input('Iteraciones de adaptación por cadena', min_value=100, value=1000)
        sampling = st.number_input('Muestras por cadena', min_value=100, value=1000)
        seed = st.number_input('Semilla', min_value=1, max_value=2147483647, value=20261005)
        st.caption('Cuatro cadenas. Búsqueda global con refinamiento de picos, adapt_delta = 0.95 y profundidad máxima = 15.')
    ready = 'v1' in components and (tipo == 'SB1' or 'v2' in components)
    cfg = None
    if ready:
        try:
            safe_name = re.sub(r'[^\w-]+', '_', system_name.strip())[:80] or 'Sistema'
            cfg, pinfo, hyper, effective = servicio.preparar(
                components, formato, safe_name, tipo, esquema, pmin, pmax,
                p_fixed, e_fixed, p_initial, priors)
            with st.expander('Priors y límites efectivos antes de ejecutar', expanded=True):
                st.dataframe(effective[['parametro', 'prior_resuelto']], hide_index=True, width='stretch')
                amplitude_limits = {k: v for k, v in hyper.items() if k.endswith('_max') and k.startswith('K')}
                st.write('Límites superiores de amplitud (km/s)', amplitude_limits)
                st.caption('Las cotas de amplitud se calculan con el rango observado. Una cobertura orbital incompleta puede limitar ese rango.')
                st.write(f"Período inicial: {pinfo['P_ls_d']:.6g} días por Lomb–Scargle" if not manual else f"Período inicial manual: {p_initial:g} días")
        except (ValueError, KeyError, FloatingPointError) as exc:
            st.error(str(exc))
            cfg = None
    st.subheader('Optimización del sistema')
    st.caption('Ejecuta L-BFGS con los ajustes de la barra lateral. Revisa las soluciones antes de iniciar NUTS.')
    if st.button('Ejecutar optimización del sistema', type='primary', disabled=cfg is None):
        settings = ajuste.Ajustes(n_inicios_SB1=int(starts), n_inicios_SB2=int(starts),
                                 iter_warmup=int(warmup), iter_sampling=int(sampling), semilla=int(seed),
                                 extra={'timeout_opt': 60, 'timeout_sample': 1800})
        barra = st.progress(0.0, text='Preparando la optimización…')
        inicio_arranques = None

        def actualizar_progreso(fase, hechos, total):
            nonlocal inicio_arranques
            if not total:
                barra.progress(0.0, text=f'{fase} · estimación disponible al comenzar los arranques')
                return
            if inicio_arranques is None:
                inicio_arranques = monotonic()
            detalle = f'{fase} · {hechos}/{total} arranques'
            if 0 < hechos < total:
                segundos = int(np.ceil((monotonic()-inicio_arranques)*(total-hechos)/hechos))
                minutos, segundos = divmod(segundos, 60)
                detalle += f' · Tiempo restante estimado: {minutos} min {segundos:02d} s'
            barra.progress(hechos/total, text=detalle)

        folder = servicio.ejecutar(components, cfg, esquema, priors, settings,
                                   progreso=actualizar_progreso)
        registro, *_ = servicio.cargar_resultado(folder)
        if registro['estado'] == 'preliminar':
            barra.progress(1.0, text=f"Optimización terminada en {registro['segundos']:g} s")
        else:
            barra.empty()
        st.session_state['corrida_consultada'] = str(folder)

    outputs = [str(p.parent) for p in sorted(servicio.RUNS.glob('*/registro.json'), reverse=True)]
    if not outputs:
        return
    folder = st.selectbox('Resultado a consultar', outputs, key='corrida_consultada',
        format_func=lambda f: servicio.cargar_resultado(f)[0]['run_id']+' · '+Path(f).name)
    registro, *_ = servicio.cargar_resultado(folder)
    candidatos, datos = servicio.preliminar(folder, registro)
    if candidatos:
        mostrar_preliminar(candidatos, datos, registro)
        st.caption('NUTS usa los datos y ajustes guardados con esta optimización. '
                   'Para aplicar cambios de la barra lateral, ejecuta otra optimización del sistema.')
        if registro['estado'] != 'ok':
            if st.button('Continuar con muestreo NUTS'):
                with st.spinner('Ejecutando las cadenas NUTS…'):
                    registro = servicio.muestrear(folder)
            if registro['estado'] == 'preliminar':
                st.download_button('Descargar optimización del sistema', servicio.descargar(folder),
                                   file_name='ORBIX_optimizacion_del_sistema.zip')
    if registro['estado'] != 'preliminar':
        mostrar_resultado(folder)

def mostrar_preliminar(candidatos, datos, registro):
    representantes = []
    for candidato in candidatos:
        p = candidato['parametros']
        for grupo in representantes:
            q = grupo['parametros']
            if np.isclose(p['P'], q['P'], rtol=1e-4, atol=0) and abs(p['e']-q['e']) <= .001:
                grupo['arranques'] += 1
                break
        else:
            representantes.append(dict(candidato, arranques=1))
    top = representantes[:5]
    mejor = top[0]['parametros']['lp__']
    unidad = fechas.unidad_periodo(registro['epoch_format_work'])
    st.write(f"{len(candidatos)} de {registro['n_inicios']} arranques devolvieron valores finitos.")
    st.dataframe(pd.DataFrame([
        {'Solución': i, 'Período (días)': float(fechas.convertir_duracion(c['parametros']['P'], unidad, 'd')),
         'Excentricidad': c['parametros']['e'], 'lp__': c['parametros']['lp__'],
         'Δlp respecto a la mejor': c['parametros']['lp__']-mejor, 'χ² reducido': c['chi2_red'],
         'Origen': c['origen'], 'Arranques agrupados': c['arranques']}
        for i, c in enumerate(top, 1)
    ]), hide_index=True)
    st.caption('Se muestran hasta cinco grupos por período y excentricidad '
               '(tolerancia relativa de período 0,01 % y absoluta de excentricidad 0,001). '
               'La solución 1 tiene el mayor lp__ e inicializa NUTS.')
    selected = st.selectbox('Solución para ver curvas y residuos', range(1, len(top)+1))
    point = top[selected-1]['parametros']
    phase = np.linspace(0, 1, 1000)
    grid = point['T'] + phase*point['P']
    curves = rv.curvas(point, grid, registro['tipo'])
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for (name, data), color in zip(datos.items(), ('#0072B2', '#D55E00')):
        x = rv.fase(data.epoch, point['P'], point['T'])
        predicted = rv.curvas(point, data.epoch, registro['tipo'])[name]
        axes[0].errorbar(x, data.RV, yerr=data.err, fmt='o', ms=3, color=color,
                         label='Primaria' if name == 'v1' else 'Secundaria')
        axes[0].plot(phase, curves[name], color=color)
        axes[1].scatter(x, (data.RV-predicted)/data.err, s=12, color=color)
    axes[0].set_ylabel('RV (km/s)')
    axes[0].legend()
    axes[1].axhline(0, color='gray', lw=1)
    axes[1].set_ylabel('Residuo / error')
    axes[1].set_xlabel('Fase orbital')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
