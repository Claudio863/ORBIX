import os
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from PyAstronomy.pyasl import foldAt
from utils import update_namespace, getRVSingle, getRV
from cmdstanpy import CmdStanModel
from orbix.modelos import preparar_header
from pdf_utils import create_pdf_and_zip
import streamlit as st
import tempfile
import shutil

# -------------------------------------------------------------
# Utilidades para reencuadre inteligente y detección de recorte
def _smart_xlim(x, *, method='hdi', hdi=None, q=(0.001, 0.999), pad=0.05):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 1.0)
    if method == 'hdi' and hdi is not None:
        lo, hi = float(hdi[0]), float(hdi[1])
    elif method == 'quantile':
        lo, hi = np.quantile(x, q)
    else:  # 'full'
        lo, hi = x.min(), x.max()
    w = (hi - lo) if np.isfinite(hi - lo) and (hi > lo) else 0.0
    return (lo - w*pad, hi + w*pad)

def _fraction_outside(x, lims):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0
    left = np.mean(x < lims[0])
    right = np.mean(x > lims[1])
    return float(left), float(right)

def validate_init_values(init_dict, fix_P=False, fix_e=False):
    """
    Valida que los valores iniciales estén en rangos apropiados para Stan
    """
    valid = True
    
    # Validar T0
    if 'T0' in init_dict:
        if not (0.001 <= init_dict['T0'] <= 0.999):
            valid = False
    
    # Validar e (si no está fijo)
    if 'e' in init_dict and not fix_e:
        if not (0.001 <= init_dict['e'] <= 0.99):
            valid = False
    
    # Validar log_P (si no está fijo)
    if 'log_P' in init_dict and not fix_P:
        if not (-5 <= init_dict['log_P'] <= 8):  # Roughly 0.007 to 3000 days
            valid = False
    
    # Validar w
    if 'w' in init_dict:
        if not (0 <= init_dict['w'] <= 2*np.pi):
            valid = False
    
    # Validar V0
    if 'V0' in init_dict:
        if not (-200 <= init_dict['V0'] <= 200):  # Reasonable RV range
            valid = False
    
    # Validar asin_i
    if 'asin_i' in init_dict:
        if not (0.001 <= init_dict['asin_i'] <= 10):
            valid = False
    
    # Validar plx
    if 'plx' in init_dict:
        if not (0.001 <= init_dict['plx'] <= 100):
            valid = False
    
    # Validar q
    if 'q' in init_dict:
        if not (0.1 <= init_dict['q'] <= 2.0):
            valid = False
    
    # Verificar que todos los valores son finitos
    if not all(np.isfinite(list(init_dict.values()))):
        valid = False
    
    return valid

def analisis_stan_(df_v1, system, P_init, nombre_componente,
                   V_delta, e_init, desfase_min, desfase_max,
                   number_component, fix_P, fix_e, optim_config,
                   epoch_format_graphs):
    """
    Realiza el análisis bayesiano y genera resultados y PDF.
    optim_config: dict con claves iter, tol_obj, tol_grad, tol_param, history_size, algorithm

    Parameters
    ----------
    epoch_format_graphs : str
        Texto a usar como unidad/formato temporal para P y T en todas las etiquetas y gráficos
        (reemplaza "D" y "HJD", por ejemplo "MJD", "BJD_TDB", "JD", etc.).
    """
    # Etiquetas dinámicas para P y T según epoch_format_graphs
    p_label = rf'$P$ $[{epoch_format_graphs}]$'
    t_label = rf'$T$ $[{epoch_format_graphs}]$'

    if 'resultados_generados' not in st.session_state:
        from pathlib import Path
        # Selección de variante Stan
        carpeta_base = Path(__file__).resolve().parents[2] / "modelos"
        if fix_P and fix_e:
            stan_file = carpeta_base / "sb1_mod_fix_eP.stan"
            def init_func():
                return dict(T0=np.random.uniform(0, 1), w=np.random.uniform(0, 2*np.pi), V0=np.random.uniform(df_v1.RV.min(), df_v1.RV.max()), K=np.random.uniform(0, V_delta))
        elif fix_P:
            stan_file = carpeta_base / "sb1_mod_fix_P.stan"
            def init_func():
                return dict(T0=np.random.uniform(0, 1), e=np.random.uniform(desfase_min, desfase_max), w=np.random.uniform(0, 2*np.pi), V0=np.random.uniform(df_v1.RV.min(), df_v1.RV.max()), K=np.random.uniform(0, V_delta))
        elif fix_e:
            stan_file = carpeta_base / "sb1_mod_fix_e.stan"
            def init_func():
                return dict(T0=np.random.uniform(0, 1), log_P=np.log(P_init), w=np.random.uniform(0, 2*np.pi), V0=np.random.uniform(df_v1.RV.min(), df_v1.RV.max()), K=np.random.uniform(0, V_delta))
        else:
            stan_file = carpeta_base / "sb1_mod.stan"
            def init_func():
                return dict(T0=np.random.uniform(0, 1), e=np.random.uniform(desfase_min, desfase_max), log_P=np.log(P_init), w=np.random.uniform(0, 2*np.pi), V0=np.random.uniform(df_v1.RV.min(), df_v1.RV.max()), K=np.random.uniform(0, V_delta))
        stan_file = str(stan_file)
        user_header = str(preparar_header(Path(stan_file).stem, Path(__file__).resolve().parents[2] / "modelos" / "kepler_eq.hpp", Path(tempfile.gettempdir()) / "orbix_app_build"))
        update_namespace(user_header, Path(stan_file).stem)
        model = CmdStanModel(stan_file=stan_file, user_header=user_header)
        log_P_lower = np.log(P_init * 0.9)
        log_P_upper = np.log(P_init * 1.1)
        e_lower     = max(1e-4,  e_init * 0.9)
        e_upper     = min(0.99,  e_init * 1.1)
        # Datos para Stan
        data = dict(N_v1=len(df_v1), 
                    t_v1=df_v1.epoch.values, 
                    v1_obs=df_v1.RV.values,
                    v1_err=df_v1.err.values)
        if fix_e:
            data["e_fixed"] = float(e_init)
        if fix_P:
            data["P_fixed"] = float(P_init)
        else:
            data["P_lower"] = log_P_lower
            data["P_upper"] = log_P_upper
            data["e_lower"] = e_lower
            data["e_upper"] = e_upper
        # Optimización y muestreo
        st.markdown("⏳ <b>Optimization and sampling</b>", unsafe_allow_html=True)
        st.markdown("#### Parameter initialization for preliminary optimization")
        best_mle = dict(lp__=-np.inf)
        progress_bar = st.progress(0)
        time_text = st.empty()
        import time
        start_time = time.time()
        total_iter = 400
        for i in range(total_iter):
            try:
                mle = model.optimize(
                    data=data,
                    inits=init_func(),
                    require_converged=False,
                    algorithm=optim_config["algorithm"],
                    history_size=optim_config["history_size"],
                    tol_param=optim_config["tol_param"],
                    iter=optim_config["iter"],
                    tol_grad=optim_config["tol_grad"],
                    tol_obj=optim_config["tol_obj"]
                ).optimized_params_dict
            except RuntimeError:
                continue
            if mle['lp__'] > best_mle['lp__']:
                best_mle = mle
                st.write(f"Current maximum log-probability: {best_mle['lp__']:.4f}")
            progress_bar.progress((i+1)/total_iter)
            elapsed = time.time() - start_time
            est_total = elapsed / (i+1) * total_iter
            remaining = est_total - elapsed
            minutes, seconds = divmod(int(remaining), 60)
            time_text.text(f"Estimated time of completion: {minutes:02d}:{seconds:02d}")
        fit = model.sample(
            data=data,
            chains=4,
            parallel_chains=4,
            iter_warmup=1000,
            iter_sampling=1000,
            adapt_delta=0.95,
            max_treedepth=15,
            inits=best_mle,
            sig_figs=10
        )
        st.write(fit.diagnose())
        summary_txt = fit.summary().to_string()
        st.text(summary_txt)
        # Procesamiento de resultados
        draws = fit.draws_pd()
        if 'e' not in draws.columns:
            draws['e'] = float(e_init)
        if 'P' not in draws.columns and fix_P:
            draws['P'] = float(P_init)
        cols = ['P', 'T', 'e', 'w_deg', 'V0', 'K', 'lp__']
        df = draws[cols]
        # Etiquetas de columnas con unidades dinámicas para P y T
        df.columns = [p_label, t_label, r'$e$', r'$\omega$ $[°]$', r'$V_{0}$ $[km/s]$', r'$K$ $[km/s]$', 'lp__']
        samples = df.drop(columns=['lp__']).to_numpy()
        map_sample = samples[df[['lp__']].to_numpy().argmax()]
        lb = az.hdi(samples, 0.95)[:,0]
        ub = az.hdi(samples, 0.95)[:,1]
        summary_table = [[df.columns[i], r'${:.3f}_{{{:.3f}}}^{{{:.3f}}}$'.format(map_sample[i], lb[i], ub[i])] for i in range(len(map_sample))]
        df_summary = pd.DataFrame(summary_table, columns=['Parameter', 'Estimate'])
        # --- Nuevo: Selectbox para tipo de gráfico ---
        plot_type = st.selectbox('Select plot type:', ['Phase curve', 'RV vs Time'])
        # --- Función para el gráfico de phase ---
        def plot_phase_curve():
            period = map_sample[0]
            t = np.linspace(df_v1.epoch.min(), df_v1.epoch.max(), 500)
            t0 = df_v1.epoch.min()
            phaseAn = foldAt(t, period, t0)
            sortIndi = np.argsort(phaseAn)
            phase = foldAt(df_v1.epoch, period, t0)
            plt.figure(figsize=(6,4))
            rnd_idx = np.random.choice(samples.shape[0], 1000, replace=False)
            for j in range(1000):
                P, T, e, w, V0, K = samples[rnd_idx[j]]
                V1 = getRVSingle(t, T, P, e, np.deg2rad(w), V0, K)
                plt.plot(phaseAn[sortIndi], V1[sortIndi], color='tab:cyan', alpha=0.005)
            P, T, e, w, V0, K = map_sample
            V1 = getRVSingle(t, T, P, e, np.deg2rad(w), V0, K)
            plt.plot(phaseAn[sortIndi], V1[sortIndi], linewidth=1, color='tab:blue')
            plt.errorbar(phase, df_v1.RV, yerr=df_v1.err, fmt=".k", capsize=3, ms=1)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
            plt.ylabel('Radial velocity [$km/s$]', fontsize=12)
            plt.xlabel('Phase', fontsize=12)
            plt.tight_layout()
            plt.savefig('RV_phase_plot.png', dpi=600)
            plt.close()
        # --- Función para el gráfico RV vs Tiempo ---
        def plot_rv_vs_time():
            plt.figure(figsize=(10,6))
            rnd_idx = np.random.choice(samples.shape[0], 1000, replace=False)
            t_continuous = np.linspace(df_v1.epoch.min(), df_v1.epoch.max(), 1000)
            for j in range(1000):
                P, T, e, w, V0, K = samples[rnd_idx[j]]
                V1_continuous = getRVSingle(t_continuous, T, P, e, np.deg2rad(w), V0, K)
                plt.plot(t_continuous, V1_continuous, color='tab:cyan', alpha=0.005)
            P, T, e, w, V0, K = map_sample
            V1_continuous = getRVSingle(t_continuous, T, P, e, np.deg2rad(w), V0, K)
            plt.plot(t_continuous, V1_continuous, linewidth=2, color='tab:blue', label='Best fit')
            plt.errorbar(df_v1.epoch, df_v1.RV, yerr=df_v1.err, fmt=".k", capsize=3, ms=4, label='Observed data')
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))
            plt.ylabel('Radial velocity [km/s]')
            # Eje X con formato dinámico
            plt.xlabel(f'Epoch [{epoch_format_graphs}]')
            plt.legend()
            plt.title('Radial Velocity Curve vs Time')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('RV_time_plot.png', dpi=600)
            plt.close()
        # --- Guardar ambos gráficos ---
        plot_phase_curve()
        plot_rv_vs_time()
        # Gráficas y PDF
        # Filtrar parámetros fijos para NO mostrarlos en histogramas/corner
        full_param_names = df.drop(columns=['lp__']).columns.tolist()
        plot_param_names = full_param_names.copy()
        if fix_P:
            # etiqueta bonita de P (dinámica)
            if p_label in plot_param_names:
                plot_param_names.remove(p_label)
        if fix_e:
            if r'$e$' in plot_param_names:
                plot_param_names.remove(r'$e$')

        # Indices a utilizar para los vectores numéricos
        idx_map = [full_param_names.index(n) for n in plot_param_names]

        plt.figure(figsize=(30,14))
        # Importante: usar arrays filtrados para histogramas
        for i_plot, j in enumerate(idx_map):
            plt.subplot(2,3,i_plot+1)
            sns.histplot(samples[:,j], stat='density', kde=True, color='tab:blue', alpha=0.4, **{"edgecolor": None})
            min_ylim, max_ylim = plt.ylim()
            plt.axvline(map_sample[j], color='red', linewidth=1)
            plt.text(map_sample[j], max_ylim * 0.9, ' {:.3f}'.format(map_sample[j]), color='red', fontsize=25)
            plt.hlines(y=max_ylim * 0.07, xmin=lb[j], xmax=ub[j], linewidth=5, color='black')
            plt.text(lb[j], max_ylim * 0.1, ' {:.3f}'.format(lb[j]), color='black', horizontalalignment='right', fontsize=25)
            plt.text(ub[j], max_ylim * 0.1, ' {:.3f}'.format(ub[j]), color='black', fontsize=23)
            ax = plt.gca()
            # Reencuadre centrado en el 95% HDI (+15% de margen)
            ax.set_xlim(_smart_xlim(samples[:, j], method='hdi', hdi=(lb[j], ub[j]), pad=0.15))
            # Detección de recorte
            fl, fr = _fraction_outside(samples[:, j], ax.get_xlim())
            if (fl + fr) > 0.001:
                st.warning(f"{plot_param_names[i_plot]}: {100*(fl+fr):.2f}% de muestras quedan fuera del rango mostrado (izq {fl:.2%}, der {fr:.2%}).")
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            # Desactivar notación científica en X si molesta
            ax.ticklabel_format(style='plain', useOffset=False, axis='x')
            plt.xlabel(plot_param_names[i_plot], fontsize=23)
            plt.ylabel(None)
            plt.xticks(fontsize=23)
            plt.yticks(fontsize=23)
        plt.tight_layout()
        plt.savefig('dis_params1.png', dpi=300)
        plt.close()
        # Pairplot con columnas filtradas
        df_plot = df.drop(columns='lp__')
        df_plot = df_plot[plot_param_names] if plot_param_names else df_plot.iloc[:, :0]
        if df_plot.shape[1] >= 1:
            pairplot_fig = sns.pairplot(df_plot, kind='hist', corner=True)
        else:
            # Crear figura vacía para mantener el pipeline (archivo requerido)
            plt.figure(figsize=(4,3))
            plt.text(0.5, 0.5, 'Sin parámetros a mostrar', ha='center', va='center')
            plt.axis('off')
            plt.savefig('dis_params2.png', dpi=300)
            plt.close()
            pairplot_fig = None
        # Reencuadre y chequeo en pairplot
        if pairplot_fig is not None:
            xvars = pairplot_fig.x_vars
            yvars = pairplot_fig.y_vars
            lims = {col: _smart_xlim(df_plot[col].values, method='quantile', q=(0.001, 0.999), pad=0.05) for col in xvars}
            for i_ax, ycol in enumerate(yvars):
                for j_ax, xcol in enumerate(xvars[:i_ax+1]):  # lower triangle
                    ax = pairplot_fig.axes[i_ax, j_ax]
                    if ax is None:
                        continue
                    ax.set_xlim(lims[xcol])
                    if i_ax != j_ax:
                        ax.set_ylim(lims[ycol])
                    flx, frx = _fraction_outside(df_plot[xcol].values, ax.get_xlim())
                    if (flx + frx) > 0.001:
                        st.warning(f"Corner ({xcol} vs {ycol}) eje X recorta {100*(flx+frx):.2f}% de muestras.")
                    if i_ax != j_ax:
                        fly, fry = _fraction_outside(df_plot[ycol].values, ax.get_ylim())
                        if (fly + fry) > 0.001:
                            st.warning(f"Corner ({xcol} vs {ycol}) eje Y recorta {100*(fly+fry):.2f}% de muestras.")
            for ax in pairplot_fig.axes.flatten():
                if ax is not None:
                    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                        label.set_fontsize(label.get_fontsize() + 0)
                    ax.title.set_fontsize(ax.title.get_fontsize() + 0)
            pairplot_fig.savefig('dis_params2.png', dpi=600)
            plt.close()

        plt.figure(figsize=(6,3))
        plt.title('Orbital Parameters and their Uncertainties', fontsize=14)
        table = plt.table(cellText=summary_table, loc='center', cellLoc='center', colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.axis('off')
        plt.savefig('latex_table.png', dpi=600)
        plt.close()

        # PDF y ZIP
        images = [
            ('dis_params1.png', 500, 200),
            ('dis_params2.png', 500, 380),
            ('RV_phase_plot.png', 500, 300),
            ('RV_time_plot.png', 500, 300),
            ('latex_table.png', 400, 200)
        ]

        # Construir CSV de resultados (estimación y diagnóstico)
        try:
            fit_summary_df = fit.summary()
        except Exception:
            fit_summary_df = pd.DataFrame()
        # Mapeo de nombres crudos a etiquetas bonitas (dinámicas para P y T)
        raw_cols = ['P', 'T', 'e', 'w_deg', 'V0', 'K']
        label_cols = [p_label, t_label, r'$e$', r'$\omega$ $[°]$', r'$V_{0}$ $[km/s]$', r'$K$ $[km/s]$']

        rows = []
        for idx, raw in enumerate(raw_cols):
            if raw in fit_summary_df.index or idx < len(map_sample):
                row = {
                    'parameter': raw,
                    'label': label_cols[idx],
                    'MAP': float(map_sample[idx]),
                    'HDI_2.5%': float(lb[idx]),
                    'HDI_97.5%': float(ub[idx])
                }
                if not fit_summary_df.empty and raw in fit_summary_df.index:
                    s = fit_summary_df.loc[raw]
                    # Copiar columnas típicas si existen
                    for col in ['Mean', 'StdDev', 'MCSE', 'R_hat', 'N_Eff', 'N_Eff/s']:
                        if col in s.index:
                            row[col] = s[col]
                rows.append(row)
        results_df = pd.DataFrame(rows)

        create_pdf_and_zip(
            pdf_filename=system + '_Summary.pdf',
            summary_table=summary_table,
            results_df=results_df,
            images=images,
            system=system,
            nombre_componente=nombre_componente,
            number_component=number_component
        )
        st.session_state.resultados_generados = True
        # No mostrar imágenes ni botón de descarga aquí

    def cleanup_files():
        archivos = ['dis_params1.png', 'dis_params2.png', 'RV_phase_plot.png', 'RV_time_plot.png', 'latex_table.png']
        for archivo in archivos:
            try:
                if os.path.exists(archivo):
                    os.remove(archivo)
            except Exception as e:
                st.error(f"Error deleting {archivo}: {e}")
        try:
            if os.path.exists(st.session_state.rar_filename):
                os.remove(st.session_state.rar_filename)
        except Exception as e:
            st.error(f"Error deleting the .rar file: {e}")

    if 'rar_data' in st.session_state:
        st.download_button(
            label="Download .rar with PDF and figures",
            data=st.session_state.rar_data,
            file_name=st.session_state.rar_filename,
            mime="application/x-rar-compressed",
            on_click=cleanup_files
        )


def analisis_stan_sb2(df_v1, df_v2, system, P_init, V_delta, e_init, desfase_min, desfase_max,
                      fix_P, fix_e, optim_config, epoch_format_graphs, **kwargs):
    """
    Realiza el análisis bayesiano para sistemas SB2 y genera resultados y PDF.

    Parameters
    ----------
    epoch_format_graphs : str
        Texto a usar como unidad/formato temporal para P y T en todas las etiquetas y gráficos
        (reemplaza "D" y "HJD", por ejemplo "MJD", "BJD_TDB", "JD", etc.).
    """
    # Etiquetas dinámicas para P y T según epoch_format_graphs
    p_label = rf'$P$ $[{epoch_format_graphs}]$'
    t_label = rf'$T$ $[{epoch_format_graphs}]$'

    if 'resultados_generados' not in st.session_state:
        from pathlib import Path
        
        # Validación de datos de entrada
        if df_v1.empty or df_v2.empty:
            st.error("Los datos de velocidad radial están vacíos.")
            return
        
        # Verificar que no hay NaN en los datos
        if df_v1[['epoch', 'RV', 'err']].isnull().any().any() or df_v2[['epoch', 'RV', 'err']].isnull().any().any():
            st.error("Se encontraron valores NaN en los datos. Por favor, limpie los datos antes de continuar.")
            return
        
        # Validar parámetros iniciales mínimos necesarios
        if not all(np.isfinite([P_init, e_init])):
            st.error("Los parámetros iniciales contienen valores no finitos (NaN o Inf).")
            return
            
        # Validar que los parámetros están en rangos razonables
        if P_init <= 0 or P_init > 10000:
            st.error(f"El período inicial ({P_init}) debe estar entre 0 y 10000.")
            return
        if e_init < 0 or e_init >= 1:
            st.error(f"La excentricidad inicial ({e_init}) debe estar entre 0 y 0.99.")
            return
        # Nota: Se omiten validaciones para q, asin_i y plx; no se ingresan ni se usan como entradas
        
        # Calcular rangos seguros basados en los datos
        rv_min = min(df_v1.RV.min(), df_v2.RV.min())
        rv_max = max(df_v1.RV.max(), df_v2.RV.max())
        rv_range = rv_max - rv_min
        
        # Evitar rangos muy pequeños
        if rv_range < 1.0:
            rv_range = max(10.0, abs(rv_min + rv_max) * 0.1)
        
    # Se omiten rangos para asin_i, plx y q (no se ingresan ni se usan como entradas)
        
        e_min = max(0.001, desfase_min)
        e_max = min(0.98, desfase_max)
        
        # Asegurar que el periodo esté en un rango razonable
        P_log_min = np.log(max(0.1, P_init * 0.5))
        P_log_max = np.log(min(1000.0, P_init * 2.0))
        
        # Selección de variante Stan para SB2
        carpeta_base = Path(__file__).resolve().parents[2] / "modelos"
        if fix_P and fix_e:
            stan_file = carpeta_base / "sb2_fix_eP.stan"
            if not stan_file.exists():
                st.error("sb2_fix_eP.stan not implemented yet. Please use only one fixed parameter at a time.")
                return
            def init_func():
                max_attempts = 50
                for attempt in range(max_attempts):
                    init_values = dict(
                        T0=np.random.uniform(0.1, 0.9), 
                        w=np.random.uniform(0, 2*np.pi), 
                        V0=np.random.uniform(rv_min - rv_range*0.1, rv_max + rv_range*0.1)
                    )
                    if validate_init_values(init_values, fix_P=True, fix_e=True):
                        return init_values
                return dict(T0=0.5, w=np.pi, V0=(rv_min + rv_max)/2)
        elif fix_P:
            stan_file = carpeta_base / "sb2_fix_P.stan"
            def init_func():
                max_attempts = 50
                for attempt in range(max_attempts):
                    init_values = dict(
                        T0=np.random.uniform(0.1, 0.9), 
                        e=np.random.uniform(e_min, e_max), 
                        w=np.random.uniform(0, 2*np.pi), 
                        V0=np.random.uniform(rv_min - rv_range*0.1, rv_max + rv_range*0.1)
                    )
                    if validate_init_values(init_values, fix_P=True, fix_e=False):
                        return init_values
                return dict(T0=0.5, e=max(0.01, min(0.8, e_init)), w=np.pi, V0=(rv_min + rv_max)/2)
        elif fix_e:
            stan_file = carpeta_base / "sb2_fix_e.stan"
            def init_func():
                max_attempts = 100
                for attempt in range(max_attempts):
                    init_values = dict(
                        T0=np.random.uniform(0.1, 0.9), 
                        log_P=np.random.uniform(P_log_min, P_log_max), 
                        w=np.random.uniform(0, 2*np.pi), 
                        V0=np.random.uniform(rv_min - rv_range*0.1, rv_max + rv_range*0.1)
                    )
                    if validate_init_values(init_values, fix_P=False, fix_e=True):
                        return init_values
                return dict(T0=0.5, log_P=np.log(max(1.0, min(500.0, P_init))), w=np.pi, V0=(rv_min + rv_max)/2)
        else:
            stan_file = carpeta_base / "sb2.stan"
            def init_func():
                max_attempts = 50
                for attempt in range(max_attempts):
                    init_values = dict(
                        T0=np.random.uniform(0.1, 0.9), 
                        e=np.random.uniform(e_min, e_max), 
                        log_P=np.random.uniform(P_log_min, P_log_max), 
                        w=np.random.uniform(0, 2*np.pi), 
                        V0=np.random.uniform(rv_min - rv_range*0.1, rv_max + rv_range*0.1)
                    )
                    if validate_init_values(init_values, fix_P=False, fix_e=False):
                        return init_values
                return dict(
                    T0=0.5, e=max(0.01, min(0.8, e_init)), 
                    log_P=np.log(max(1.0, min(500.0, P_init))), w=np.pi, 
                    V0=(rv_min + rv_max)/2
                )

        stan_file = str(stan_file)
        user_header = str(preparar_header(Path(stan_file).stem, Path(__file__).resolve().parents[2] / "modelos" / "kepler_eq.hpp", Path(tempfile.gettempdir()) / "orbix_app_build"))
        update_namespace(user_header, Path(stan_file).stem)
        
        st.info(f"**Modelo Stan seleccionado:** {Path(stan_file).name}")
        if fix_P:
            st.info(f"**Período fijo:** {P_init} {epoch_format_graphs}")
        if fix_e:
            st.info(f"**Excentricidad fija:** {e_init}")
        
        model = CmdStanModel(stan_file=stan_file, user_header=user_header)

        # Datos para Stan (SB2)
        data = dict(
            N_v1=len(df_v1), t_v1=df_v1.epoch.values, v1_obs=df_v1.RV.values, v1_err=df_v1.err.values,
            N_v2=len(df_v2), t_v2=df_v2.epoch.values, v2_obs=df_v2.RV.values, v2_err=df_v2.err.values
        )
        
        if fix_e:
            data["e_fixed"] = float(e_init)
        if fix_P:
            data["P_fixed"] = float(P_init)
            
        # Verificación de datos
        st.write("Validando datos para Stan...")
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    st.error(f"Valor no válido en {key}: {value}")
                    return
            elif hasattr(value, '__iter__'):
                if not all(np.isfinite(value)):
                    st.error(f"Valores no válidos encontrados en {key}")
                    return
        st.success("Todos los datos son válidos para Stan")

        # Optimización y muestreo
        st.markdown("⏳ <b>Optimization and sampling (SB2)</b>", unsafe_allow_html=True)
        st.markdown("#### Parameter initialization for preliminary optimization")
        st.info(f"""
        **Rangos de inicialización calculados:**
        - e: [{e_min:.3f}, {e_max:.3f}]
        - V0: [{rv_min - rv_range*0.1:.1f}, {rv_max + rv_range*0.1:.1f}] km/s
        """)
        best_mle = dict(lp__=-np.inf)
        progress_bar = st.progress(0)
        time_text = st.empty()
        import time
        start_time = time.time()
        total_iter =1500
        
        st.write("Validating data...")
        if df_v1.empty or df_v2.empty:
            st.error("Empty data detected")
            return
        if any(df_v1.err <= 0) or any(df_v2.err <= 0):
            st.error("Invalid error values detected (must be positive)")
            return
        
        successful_optimizations = 0
        
        for i in range(total_iter):
            try:
                init_values = init_func()
                if not validate_init_values(init_values, fix_P=fix_P, fix_e=fix_e):
                    st.warning(f"Skipping iteration {i+1}: invalid initial values generated")
                    continue
                
                mle = model.optimize(
                    data=data,
                    inits=init_values,
                    require_converged=False,
                    algorithm=optim_config["algorithm"],
                    history_size=optim_config["history_size"],
                    tol_param=optim_config["tol_param"],
                    iter=optim_config["iter"],
                    tol_grad=optim_config["tol_grad"],
                    tol_obj=optim_config["tol_obj"]
                ).optimized_params_dict
                
                if not all(np.isfinite(list(mle.values()))):
                    continue
                successful_optimizations += 1
                
            except Exception as e:
                #st.warning(f"Optimization attempt {i+1} failed: {str(e)[:100]}...")
                continue
                
            if mle['lp__'] > best_mle['lp__']:
                best_mle = mle
                st.write(f"Current maximum log-probability: {best_mle['lp__']:.4f}")
                
            progress_bar.progress((i+1)/total_iter)
            elapsed = time.time() - start_time
            est_total = elapsed / (i+1) * total_iter
            remaining = est_total - elapsed
            minutes, seconds = divmod(int(remaining), 60)
            time_text.text(f"Estimated time of completion: {minutes:02d}:{seconds:02d}")

        if successful_optimizations == 0:
            st.error("No se pudieron realizar optimizaciones exitosas. Verifique los parámetros iniciales y los datos.")
            return
            
        if best_mle['lp__'] == -np.inf:
            st.error("No se encontraron parámetros válidos durante la optimización.")
            return
            
        if not all(np.isfinite(v) for k, v in best_mle.items() if k != 'lp__'):
            st.error("Los mejores parámetros encontrados contienen valores no válidos.")
            return
            
        st.success(f"Optimización completada. {successful_optimizations}/{total_iter} intentos exitosos.")
        st.write(f"Mejor log-probabilidad encontrada: {best_mle['lp__']:.4f}")

        def best_init():
            init = best_mle.copy()
            for key in init:
                if key != 'lp__':
                    if 'log_P' in key:
                        init[key] += np.random.normal(0, 0.01)
                    elif key == 'e':
                        init[key] = np.clip(init[key] + np.random.normal(0, 0.01), 0.001, 0.99)
                    elif key == 'T0':
                        init[key] = np.clip(init[key] + np.random.normal(0, 0.01), 0.001, 0.999)
                    elif key in ['asin_i', 'plx', 'q']:
                        init[key] = max(0.001, init[key] + np.random.normal(0, abs(init[key]) * 0.01))
                    else:
                        init[key] += np.random.normal(0, abs(init[key]) * 0.01 if init[key] != 0 else 0.01)
            del init['lp__']
            return init

        init_values = [best_init() for _ in range(4)]
        for i, init_val in enumerate(init_values):
            if not validate_init_values(init_val, fix_P=fix_P, fix_e=fix_e):
                st.warning(f"Cadena {i+1}: valores iniciales no válidos, usando valores por defecto")
                safe_init = dict(T0=0.5, V0=(rv_min + rv_max)/2, w=np.pi)
                if not fix_e:
                    safe_init['e'] = max(0.01, min(0.8, e_init))
                if not fix_P:
                    safe_init['log_P'] = np.log(max(1.0, min(500.0, P_init)))
                init_values[i] = safe_init

        try:
            fit = model.sample(
                data=data,
                chains=4,
                parallel_chains=4,
                iter_warmup=1000,
                iter_sampling=1000,
                adapt_delta=0.95,
                max_treedepth=15,
                inits=init_values,
                sig_figs=10
            )
        except Exception as e:
            st.error(f"Error durante el muestreo: {str(e)}")
            st.error("Intentando con configuración más conservadora...")
            try:
                fit = model.sample(
                    data=data,
                    chains=2,
                    parallel_chains=2,
                    iter_warmup=500,
                    iter_sampling=500,
                    adapt_delta=0.8,
                    max_treedepth=10,
                    inits=init_values[:2],
                    sig_figs=6
                )
            except Exception as e2:
                st.error(f"Error en el segundo intento de muestreo: {str(e2)}")
                return
        
        st.write(fit.diagnose())
        summary_txt = fit.summary().to_string()
        st.text(summary_txt)

        # Procesamiento de resultados para SB2
        draws = fit.draws_pd()
        if 'e' not in draws.columns:
            draws['e'] = float(e_init)
        if 'P' not in draws.columns and fix_P:
            draws['P'] = float(P_init)
        
        # Columnas específicas para SB2 (crudos)
        cols = ['P', 'T', 'e', 'w_deg', 'V0', 'asin_i', 'plx_mas', 'q', 'f_plx', 'lp__']
        available_cols = [col for col in cols if col in draws.columns]
        df = draws[available_cols]
        
        # Renombrar columnas (bonitas) con P/T dinámicos
        col_names_map = {
            'P': p_label,
            'T': t_label,
            'e': r'$e$',
            'w_deg': r'$\omega$ $[°]$',
            'V0': r'$V_{0}$ $[km/s]$',
            'asin_i': r'$a \sin i$ $[arcsec]$',
            'plx_mas': r'$\pi$ $[mas]$',
            'q': r'$q$',
            'f_plx': r'$f/\pi$ $[pc]$',
            'lp__': 'lp__'
        }
        df.columns = [col_names_map.get(col, col) for col in df.columns]
        
        samples = df.drop(columns=['lp__']).to_numpy()
        map_sample = samples[df[['lp__']].to_numpy().argmax()]
        lb = az.hdi(samples, 0.95)[:,0]
        ub = az.hdi(samples, 0.95)[:,1]
        
        summary_table = [[df.columns[i], r'${:.3f}_{{{:.3f}}}^{{{:.3f}}}$'.format(map_sample[i], lb[i], ub[i])] for i in range(len(map_sample))]
        df_summary = pd.DataFrame(summary_table, columns=['Parameter', 'Estimate'])

        # Generar gráficos para SB2 (usar TODOS los parámetros para curvas)
        full_param_names = list(df.columns[:-1])
        generate_sb2_plots(df_v1, df_v2, draws, map_sample, samples, full_param_names, epoch_format_graphs)

        # Gráficas de distribución de parámetros: ocultar P/e si están fijados
        plot_param_names = full_param_names.copy()
        if fix_P and p_label in plot_param_names:
            plot_param_names.remove(p_label)
        if fix_e and r'$e$' in plot_param_names:
            plot_param_names.remove(r'$e$')

        # Mapear índices para arrays
        idx_map = [full_param_names.index(n) for n in plot_param_names]
        if idx_map:
            samples_plot = samples[:, idx_map]
            map_sample_plot = map_sample[idx_map]
            lb_plot = lb[idx_map]
            ub_plot = ub[idx_map]
        else:
            samples_plot = samples[:, :0]
            map_sample_plot = map_sample[:0]
            lb_plot = lb[:0]
            ub_plot = ub[:0]

        plot_parameter_distributions_sb2(samples_plot, map_sample_plot, lb_plot, ub_plot, plot_param_names)
        
        # Tabla de parámetros
        plt.figure(figsize=(8,6))
        plt.title('Orbital Parameters and their Uncertainties (SB2)', fontsize=14)
        table = plt.table(cellText=summary_table, loc='center', cellLoc='center', colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.axis('off')
        plt.savefig('latex_table.png', dpi=600)
        plt.close()

        # PDF y ZIP
        images = [
            ('dis_params1.png', 500, 200),
            ('dis_params2.png', 500, 380),
            ('RV_phase_plot_sb2.png', 500, 300),
            ('RV_time_plot_sb2.png', 500, 300),
            ('latex_table.png', 400, 200)
        ]
        
        # Construir CSV de resultados (estimación y diagnóstico) para SB2
        try:
            fit_summary_df = fit.summary()
        except Exception:
            fit_summary_df = pd.DataFrame()
        raw_to_label = {raw: label for raw, label in zip(available_cols, df.columns)}
        rows = []
        params_for_csv = [c for c in available_cols if c != 'lp__']
        for idx, raw in enumerate(params_for_csv):
            try:
                pos = list(df.drop(columns=['lp__']).columns).index(raw_to_label[raw])
            except Exception:
                pos = idx if idx < len(map_sample) else None
            if pos is None:
                continue
            row = {
                'parameter': raw,
                'label': raw_to_label.get(raw, raw),
                'MAP': float(map_sample[pos]),
                'HDI_2.5%': float(lb[pos]),
                'HDI_97.5%': float(ub[pos])
            }
            if not fit_summary_df.empty and raw in fit_summary_df.index:
                s = fit_summary_df.loc[raw]
                for col in ['Mean', 'StdDev', 'MCSE', 'R_hat', 'N_Eff', 'N_Eff/s']:
                    if col in s.index:
                        row[col] = s[col]
            rows.append(row)
        results_df = pd.DataFrame(rows)

        create_pdf_and_zip(
            pdf_filename=system + '_SB2_Summary.pdf',
            summary_table=summary_table,
            results_df=results_df,
            images=images,
            system=system,
            nombre_componente='SB2',
            number_component='SB2'
        )
        
        st.session_state.resultados_generados = True

    def cleanup_files():
        archivos = ['dis_params1.png', 'dis_params2.png', 'RV_phase_plot_sb2.png', 'RV_time_plot_sb2.png', 'latex_table.png']
        for archivo in archivos:
            try:
                if os.path.exists(archivo):
                    os.remove(archivo)
            except Exception as e:
                st.error(f"Error deleting {archivo}: {e}")
        try:
            if os.path.exists(st.session_state.rar_filename):
                os.remove(st.session_state.rar_filename)
        except Exception as e:
            st.error(f"Error deleting the .rar file: {e}")

    if 'rar_data' in st.session_state:
        st.download_button(
            label="Download .rar with PDF and figures (SB2)",
            data=st.session_state.rar_data,
            file_name=st.session_state.rar_filename,
            mime="application/x-rar-compressed",
            on_click=cleanup_files
        )


def generate_sb2_plots(df_v1, df_v2, draws, map_sample, samples, param_names, epoch_format_graphs):
    """Genera gráficos específicos para SB2 con etiquetas P/T dinámicas."""
    # Etiquetas dinámicas
    p_label = rf'$P$ $[{epoch_format_graphs}]$'
    t_label = rf'$T$ $[{epoch_format_graphs}]$'

    # Obtener parámetros MAP (según etiquetas bonitas presentes)
    param_dict = dict(zip(param_names, map_sample))
    
    P_map = param_dict.get(p_label, map_sample[0])
    T_map = param_dict.get(t_label, map_sample[1])
    e_map = param_dict.get(r'$e$', map_sample[2])
    w_map = param_dict.get(r'$\omega$ $[°]$', map_sample[3])
    V0_map = param_dict.get(r'$V_{0}$ $[km/s]$', map_sample[4])
    
    asin_i_map = param_dict.get(r'$a \sin i$ $[arcsec]$', 0.1)
    plx_mas_map = param_dict.get(r'$\pi$ $[mas]$', 10)
    plx_map = plx_mas_map / 1000 if plx_mas_map > 1 else plx_mas_map  # mas→arcsec si aplica
    q_map = param_dict.get(r'$q$', 0.5)
    
    # Gráfico de fase para SB2
    period = P_map
    t_continuous = np.linspace(min(df_v1.epoch.min(), df_v2.epoch.min()), 
                               max(df_v1.epoch.max(), df_v2.epoch.max()), 1000)
    t0 = min(df_v1.epoch.min(), df_v2.epoch.min())
    
    phaseAn = foldAt(t_continuous, period, t0)
    sortIndi = np.argsort(phaseAn)
    
    phase_v1 = foldAt(df_v1.epoch, period, t0)
    phase_v2 = foldAt(df_v2.epoch, period, t0)
    
    plt.figure(figsize=(10,6))
    
    # Dibujar muestras
    rnd_idx = np.random.choice(samples.shape[0], 300, replace=False)
    for j in range(300):
        sample_params = samples[rnd_idx[j]]
        sample_dict = dict(zip(param_names, sample_params))
        
        P_s = sample_dict.get(p_label, sample_params[0])
        T_s = sample_dict.get(t_label, sample_params[1])
        e_s = sample_dict.get(r'$e$', sample_params[2])
        w_s = sample_dict.get(r'$\omega$ $[°]$', sample_params[3])
        V0_s = sample_dict.get(r'$V_{0}$ $[km/s]$', sample_params[4])
        asin_i_s = sample_dict.get(r'$a \sin i$ $[arcsec]$', asin_i_map)
        plx_mas_s = sample_dict.get(r'$\pi$ $[mas]$', plx_mas_map)
        plx_s = plx_mas_s / 1000 if plx_mas_s > 1 else plx_mas_s
        q_s = sample_dict.get(r'$q$', q_map)
        
        try:
            t_phase = t_continuous[sortIndi]
            V1_sample, V2_sample = getRV(t_phase, T_s, P_s, e_s, asin_i_s, np.deg2rad(w_s), np.pi/2, V0_s, plx_s, q_s)
            plt.plot(phaseAn[sortIndi], V1_sample, alpha=0.02)
            plt.plot(phaseAn[sortIndi], V2_sample, alpha=0.02)
        except:
            V_amp = 20 * asin_i_s / plx_s
            V1_sample = V0_s + V_amp * (np.cos(np.deg2rad(w_s) + 2*np.pi*phaseAn[sortIndi]) + e_s * np.cos(np.deg2rad(w_s)))
            V2_sample = V0_s - V_amp * (np.cos(np.deg2rad(w_s) + 2*np.pi*phaseAn[sortIndi]) + e_s * np.cos(np.deg2rad(w_s)))
            plt.plot(phaseAn[sortIndi], V1_sample, alpha=0.02)
            plt.plot(phaseAn[sortIndi], V2_sample, alpha=0.02)
    
    # Curva MAP
    try:
        t_phase_map = t_continuous[sortIndi]
        V1_map, V2_map = getRV(t_phase_map, T_map, P_map, e_map, asin_i_map, np.deg2rad(w_map), np.pi/2, V0_map, plx_map, q_map)
    except:
        V_amp_map = 20 * asin_i_map / plx_map
        V1_map = V0_map + V_amp_map * (np.cos(np.deg2rad(w_map) + 2*np.pi*phaseAn[sortIndi]) + e_map * np.cos(np.deg2rad(w_map)))
        V2_map = V0_map - V_amp_map * (np.cos(np.deg2rad(w_map) + 2*np.pi*phaseAn[sortIndi]) + e_map * np.cos(np.deg2rad(w_map)))
    
    plt.plot(phaseAn[sortIndi], V1_map, linewidth=2, label='V1 (Primary)')
    plt.plot(phaseAn[sortIndi], V2_map, linewidth=2, label='V2 (Secondary)')
    
    # Datos observados
    plt.errorbar(phase_v1, df_v1.RV, yerr=df_v1.err, fmt=".b", capsize=3, ms=4, alpha=0.7, label='V1 data')
    plt.errorbar(phase_v2, df_v2.RV, yerr=df_v2.err, fmt=".r", capsize=3, ms=4, alpha=0.7, label='V2 data')
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.ylabel('Radial velocity [km/s]')
    plt.xlabel('Phase')
    plt.title('SB2 Phase Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('RV_phase_plot_sb2.png', dpi=600)
    plt.close()
    
    # Gráfico vs tiempo para SB2
    plt.figure(figsize=(12,8))
    for j in range(100):
        sample_params = samples[rnd_idx[j]]
        sample_dict = dict(zip(param_names, sample_params))
        
        P_s = sample_dict.get(p_label, sample_params[0])
        T_s = sample_dict.get(t_label, sample_params[1])
        e_s = sample_dict.get(r'$e$', sample_params[2])
        w_s = sample_dict.get(r'$\omega$ $[°]$', sample_params[3])
        V0_s = sample_dict.get(r'$V_{0}$ $[km/s]$', sample_params[4])
        asin_i_s = sample_dict.get(r'$a \sin i$ $[arcsec]$', asin_i_map)
        plx_mas_s = sample_dict.get(r'$\pi$ $[mas]$', plx_mas_map)
        plx_s = plx_mas_s / 1000 if plx_mas_s > 1 else plx_mas_s
        q_s = sample_dict.get(r'$q$', q_map)
        
        try:
            V1_continuous, V2_continuous = getRV(t_continuous, T_s, P_s, e_s, asin_i_s, np.deg2rad(w_s), np.pi/2, V0_s, plx_s, q_s)
            plt.plot(t_continuous, V1_continuous, alpha=0.02)
            plt.plot(t_continuous, V2_continuous, alpha=0.02)
        except:
            V_amp = 20 * asin_i_s / plx_s
            time_phase = 2*np.pi*(t_continuous - T_s)/P_s
            V1_continuous = V0_s + V_amp * (np.cos(np.deg2rad(w_s) + time_phase) + e_s * np.cos(np.deg2rad(w_s)))
            V2_continuous = V0_s - V_amp * (np.cos(np.deg2rad(w_s) + time_phase) + e_s * np.cos(np.deg2rad(w_s)))
            plt.plot(t_continuous, V1_continuous, alpha=0.02)
            plt.plot(t_continuous, V2_continuous, alpha=0.02)
    
    # Curva MAP
    try:
        V1_continuous_map, V2_continuous_map = getRV(t_continuous, T_map, P_map, e_map, asin_i_map, np.deg2rad(w_map), np.pi/2, V0_map, plx_map, q_map)
    except:
        V_amp_map = 20 * asin_i_map / plx_map
        time_phase_map = 2*np.pi*(t_continuous - T_map)/P_map
        V1_continuous_map = V0_map + V_amp_map * (np.cos(np.deg2rad(w_map) + time_phase_map) + e_map * np.cos(np.deg2rad(w_map)))
        V2_continuous_map = V0_map - V_amp_map * (np.cos(np.deg2rad(w_map) + time_phase_map) + e_map * np.cos(np.deg2rad(w_map)))
    
    plt.plot(t_continuous, V1_continuous_map, linewidth=2, label='V1 Best fit')
    plt.plot(t_continuous, V2_continuous_map, linewidth=2, label='V2 Best fit')
    
    # Datos observados
    plt.errorbar(df_v1.epoch, df_v1.RV, yerr=df_v1.err, fmt=".b", capsize=3, ms=4, label='V1 Observed data')
    plt.errorbar(df_v2.epoch, df_v2.RV, yerr=df_v2.err, fmt=".r", capsize=3, ms=4, label='V2 Observed data')
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.ylabel('Radial velocity [km/s]')
    plt.xlabel(f'Epoch [{epoch_format_graphs}]')  # ← etiqueta dinámica
    plt.legend()
    plt.title('SB2 Radial Velocity Curve vs Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('RV_time_plot_sb2.png', dpi=600)
    plt.close()

def plot_parameter_distributions_sb2(samples, map_sample, lb, ub, param_names):
    """Genera gráficos de distribución de parámetros para SB2"""
    n_params = samples.shape[1]
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    plt.figure(figsize=(30, 5*n_rows))
    
    for i in range(n_params):
        plt.subplot(n_rows, n_cols, i+1)
        sns.histplot(samples[:,i], stat='density', kde=True, color='tab:blue', alpha=0.4, **{"edgecolor": None})
        min_ylim, max_ylim = plt.ylim()
        plt.axvline(map_sample[i], color='red', linewidth=1)
        plt.text(map_sample[i], max_ylim * 0.9, ' {:.3f}'.format(map_sample[i]), color='red', fontsize=25)
        plt.hlines(y=max_ylim * 0.07, xmin=lb[i], xmax=ub[i], linewidth=5, color='black')
        plt.text(lb[i], max_ylim * 0.1, ' {:.3f}'.format(lb[i]), color='black', horizontalalignment='right', fontsize=25)
        plt.text(ub[i], max_ylim * 0.1, ' {:.3f}'.format(ub[i]), color='black', fontsize=23)
        ax = plt.gca()
        ax.set_xlim(_smart_xlim(samples[:, i], method='hdi', hdi=(lb[i], ub[i]), pad=0.15))
        fl, fr = _fraction_outside(samples[:, i], ax.get_xlim())
        if (fl + fr) > 0.001:
            st.warning(f"{param_names[i]}: {100*(fl+fr):.2f}% de muestras quedan fuera del rango mostrado (izq {fl:.2%}, der {fr:.2%}).")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.ticklabel_format(style='plain', useOffset=False, axis='x')
        plt.xlabel(param_names[i], fontsize=23)
        plt.ylabel(None)
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
    
    plt.tight_layout()
    plt.savefig('dis_params1.png', dpi=300)
    plt.close()
    
    # Corner plot
    df_plot = pd.DataFrame(samples, columns=param_names)
    if df_plot.shape[1] >= 1:
        pairplot_fig = sns.pairplot(df_plot, kind='hist', corner=True)
    else:
        plt.figure(figsize=(4,3))
        plt.text(0.5, 0.5, 'Sin parámetros a mostrar', ha='center', va='center')
        plt.axis('off')
        plt.savefig('dis_params2.png', dpi=300)
        plt.close()
        return
    # Reencuadre y chequeo en pairplot (SB2)
    xvars = pairplot_fig.x_vars
    yvars = pairplot_fig.y_vars
    lims = {col: _smart_xlim(df_plot[col].values, method='quantile', q=(0.001, 0.999), pad=0.05) for col in xvars}
    for i_ax, ycol in enumerate(yvars):
        for j_ax, xcol in enumerate(xvars[:i_ax+1]):  # lower triangle
            ax = pairplot_fig.axes[i_ax, j_ax]
            if ax is None:
                continue
            ax.set_xlim(lims[xcol])
            if i_ax != j_ax:
                ax.set_ylim(lims[ycol])
            flx, frx = _fraction_outside(df_plot[xcol].values, ax.get_xlim())
            if (flx + frx) > 0.001:
                st.warning(f"Corner ({xcol} vs {ycol}) eje X recorta {100*(flx+frx):.2f}% de muestras.")
            if i_ax != j_ax:
                fly, fry = _fraction_outside(df_plot[ycol].values, ax.get_ylim())
                if (fly + fry) > 0.001:
                    st.warning(f"Corner ({xcol} vs {ycol}) eje Y recorta {100*(fly+fry):.2f}% de muestras.")
    for ax in pairplot_fig.axes.flatten():
        if ax is not None:
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(label.get_fontsize() + 0)
            ax.title.set_fontsize(ax.title.get_fontsize() + 0)
    pairplot_fig.savefig('dis_params2.png', dpi=600)
    plt.close()
