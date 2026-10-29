import os
import sys
import logging
import numpy as np
import arviz as az
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm.notebook import tqdm
import cmdstanpy
from cmdstanpy import CmdStanModel
from orbix.modelos import preparar_header

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'app/src'))
from utils import update_namespace, read_in3, getOrbit, getRV, getRVSingle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st
from lectura_datos import lec_type_examples, lec_type_star02, lec_type_star01
from PIL import Image as PILImage
import patoolib
import tempfile
import shutil
from PyAstronomy.pyasl import foldAt


def analisis_stan_(df_v1, system, P_init, nombre_componente,
                   V_delta, e_init, desfase_min, desfase_max,
                   number_component, fix_P, fix_e):

    factor_crecimiento_texto = 0

    if 'resultados_generados' not in st.session_state:

        from pathlib import Path

        # ── 1) Selección de la variante Stan ──────────────────────────────
        carpeta_base = REPO / "modelos"

        if fix_P and fix_e:
            stan_file = carpeta_base / "sb1_mod_fix_eP.stan"
            def init_func():
                return dict(
                    T0     = np.random.uniform(0, 1),
                    w      = np.random.uniform(0, 2*np.pi),
                    V0     = np.random.uniform(df_v1.RV.min(), df_v1.RV.max()),
                    K      = np.random.uniform(0, V_delta)
                )
        elif fix_P:
            stan_file = carpeta_base / "sb1_mod_fix_P.stan"
            def init_func():
                return dict(
                    T0     = np.random.uniform(0, 1),
                    e      = np.random.uniform(desfase_min, desfase_max),
                    w      = np.random.uniform(0, 2*np.pi),
                    V0     = np.random.uniform(df_v1.RV.min(), df_v1.RV.max()),
                    K      = np.random.uniform(0, V_delta)
                )
        elif fix_e:
            stan_file = carpeta_base / "sb1_mod_fix_e.stan"
            def init_func():
                return dict(
                    T0     = np.random.uniform(0, 1),
                    log_P  = np.log(P_init),
                    w      = np.random.uniform(0, 2*np.pi),
                    V0     = np.random.uniform(df_v1.RV.min(), df_v1.RV.max()),
                    K      = np.random.uniform(0, V_delta)
                )
        else:
            stan_file = carpeta_base / "sb1_mod.stan"
            def init_func():
                return dict(
                    T0     = np.random.uniform(0, 1),
                    e      = np.random.uniform(desfase_min, desfase_max),
                    log_P  = np.log(P_init),
                    w      = np.random.uniform(0, 2*np.pi),
                    V0     = np.random.uniform(df_v1.RV.min(), df_v1.RV.max()),
                    K      = np.random.uniform(0, V_delta)
                )


        stan_file   = str(stan_file)
        user_header = str(preparar_header(Path(stan_file).stem, REPO / "modelos" / "kepler_eq.hpp", Path(tempfile.gettempdir()) / "orbix_app_build"))
        update_namespace(user_header, Path(stan_file).stem)
        model = CmdStanModel(stan_file=stan_file, user_header=user_header)

        # ── 2) Datos para Stan ────────────────────────────────────────────
        epoch_normalized = df_v1.epoch.values
        data = dict(
            N_v1   = len(df_v1),
            t_v1   = epoch_normalized,
            v1_obs = df_v1.RV.values,
            v1_err = df_v1.err.values,
        )
        if fix_e:
            data["e_fixed"] = float(e_init)
        if fix_P:
            data["P_fixed"] = float(P_init)

        # Función de inicializació
        

        # ── 3) Optimización y muestreo (todo en expander) ────────────────
        with st.expander("⏳ Optimización y muestreo – mostrar/ocultar"):

            hyperparameter_configs = [
                {"iter": 4000, "tol_obj": 1e-12, "tol_grad": 1e-8,  "tol_param": 1e-8, "history_size": 5,  "algorithm": "lbfgs"},
                {"iter": 4000, "tol_obj": 1e-6,  "tol_grad": 1e-6,  "tol_param": 1e-6, "history_size": 20, "algorithm": "lbfgs"},
                {"iter": 4000, "tol_obj": 1e-5,  "tol_grad": 1e-5,  "tol_param": 1e-5, "history_size": 15, "algorithm": "lbfgs"},
                {"iter": 4000, "tol_obj": 1e-7,  "tol_grad": 1e-7,  "tol_param": 1e-7, "history_size": 12, "algorithm": "lbfgs"},
                {"iter": 4000, "tol_obj": 1e-6,  "tol_grad": 1e-6,  "tol_param": 1e-6, "history_size": 18, "algorithm": "lbfgs"},
            ]
            best_mle = dict(lp__=-np.inf)
            logging.getLogger().setLevel(logging.FATAL)

            st.markdown("#### Parameter initialization for preliminary optimization")

            num_configs       = len(hyperparameter_configs)
            iter_per_config   = 200
            total_iter        = num_configs * iter_per_config

            progress_bar      = st.progress(0)
            time_text         = st.empty()

            import time
            start_time   = time.time()
            current_iter = 0

            for config in hyperparameter_configs:
                for _ in range(iter_per_config):
                    current_iter += 1
                    try:
                        mle = model.optimize(
                            data=data,
                            inits=init_func(),
                            require_converged=False,
                            algorithm     =config["algorithm"],
                            history_size  =config["history_size"],
                            tol_param     =config["tol_param"],
                            iter          =config["iter"],
                            tol_grad      =config["tol_grad"],
                            tol_obj       =config["tol_obj"]
                        ).optimized_params_dict
                    except RuntimeError:
                        continue

                    if mle['lp__'] > best_mle['lp__']:
                        best_mle = mle
                        st.write(f"Current maximum log-probability: {best_mle['lp__']:.4f}")

                    progress_bar.progress(current_iter / total_iter)

                    elapsed    = time.time() - start_time
                    est_total  = elapsed / current_iter * total_iter
                    remaining  = est_total - elapsed
                    minutes, seconds = divmod(int(remaining), 60)
                    time_text.text(f"Estimated time of completion: {minutes:02d}:{seconds:02d}")

            # —— MCMC Sampling ——
            fit = model.sample(
                data            = data,
                chains          = 4,
                parallel_chains = 4,
                iter_warmup     = 1000,
                iter_sampling   = 1000,
                adapt_delta     = 0.95,
                max_treedepth   = 15,
                inits           = best_mle,
                sig_figs        = 10
            )

            st.write(fit.diagnose())

            # Mostrar summary dentro del expander
            summary_txt = fit.summary().to_string()
            st.text(summary_txt)

        # ── 4) Continuación del flujo (fuera del expander) ───────────────
        draws = fit.draws_pd()
        # … resto de tu código (gráficas, ZIP, etc.) …

        # Añade e fija cuando no se muestreó
        if 'e' not in draws.columns:
            draws['e'] = float(e_init)          # mismo valor en todas las filas

        # Si P también está fijo haz lo mismo (opcional)
        if 'P' not in draws.columns and fix_P:
            draws['P'] = float(P_init)

        # Selección de columnas en el orden previsto
        cols = ['P', 'T', 'e', 'w_deg', 'V0', 'K', 'lp__']
        df = draws[cols]
        # Etiquetas human-readable que se usan en las figuras
        P_LABEL   = r'$P$ $[D]$'
        T_LABEL   = r'$T$ $[HJD]$'
        E_LABEL   = r'$e$'
        W_LABEL   = r'$\omega$ $[°]$'
        V0_LABEL  = r'$V_{0}$ $[km/s]$'
        K_LABEL   = r'$K$ $[km/s]$'
        df.columns = [P_LABEL, T_LABEL, E_LABEL, W_LABEL, V0_LABEL, K_LABEL, 'lp__']

        # Muestras completas (todas las variables libres o fijas) para cálculos posteriores
        samples_all = df.drop(columns=['lp__']).to_numpy()
        best_idx = df[['lp__']].to_numpy().argmax()
        map_sample_all = samples_all[best_idx]
        lb_all = az.hdi(samples_all, 0.95)[:, 0]
        ub_all = az.hdi(samples_all, 0.95)[:, 1]

        # Construir subconjunto para graficar, removiendo parámetros fijados
        to_drop_labels = []
        if fix_P:
            to_drop_labels.append(P_LABEL)
        if fix_e:
            to_drop_labels.append(E_LABEL)

        df_plot = df.drop(columns=(['lp__'] + to_drop_labels))
        params_names_plot = df_plot.columns.tolist()
        samples_plot = df_plot.to_numpy()
        map_sample_plot = df_plot.iloc[best_idx].to_numpy()
        # HPDI solo para columnas visibles
        lb_plot = az.hdi(samples_plot, 0.95)[:, 0] if samples_plot.size else np.array([])
        ub_plot = az.hdi(samples_plot, 0.95)[:, 1] if samples_plot.size else np.array([])
        
        summary_table = [[df.columns[i],
                          r'${:.3f}_{{{:.3f}}}^{{{:.3f}}}$'.format(map_sample_all[i], lb_all[i], ub_all[i])]
                         for i in range(len(map_sample_all))]
        df_summary = pd.DataFrame(summary_table, columns=['Parameter', 'Estimate'])
        
        # Gráfica de distribución para cada parámetro (PNG)
        plt.figure(figsize=(30,14))
        # Información de depuración opcional
        st.write("Columns (all):", df.columns)
        st.write("Columns (plotted):", params_names_plot)
        # Número de parámetros a graficar (excluye fijados)
        for i in range(samples_plot.shape[1]):
            plt.subplot(2,3,i+1)
            sns.histplot(samples_plot[:, i], stat='density', kde=True, color='tab:blue', alpha=0.4, **{"edgecolor": None})
            min_ylim, max_ylim = plt.ylim()
            plt.axvline(map_sample_plot[i], color='red', linewidth=1)
            plt.text(map_sample_plot[i], max_ylim * 0.9, ' {:.3f}'.format(map_sample_plot[i]), color='red', fontsize=25+factor_crecimiento_texto)
            plt.hlines(y=max_ylim * 0.07, xmin=lb_plot[i], xmax=ub_plot[i], linewidth=5, color='black')
            plt.text(lb_plot[i], max_ylim * 0.1, ' {:.3f}'.format(lb_plot[i]), color='black', horizontalalignment='right', fontsize=25+factor_crecimiento_texto)
            plt.text(ub_plot[i], max_ylim * 0.1, ' {:.3f}'.format(ub_plot[i]), color='black', fontsize=23+factor_crecimiento_texto)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
            plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
            plt.xlabel(params_names_plot[i], fontsize=19+factor_crecimiento_texto)
            # Si se desea, se puede ajustar ylabel; actualmente se deja sin etiqueta
            plt.ylabel(None)
            # Agrandamos tick labels de ambos ejes
            plt.xticks(fontsize=23+factor_crecimiento_texto)
            plt.yticks(fontsize=23+factor_crecimiento_texto)
            plt.xlabel(params_names_plot[i], fontsize=23+factor_crecimiento_texto)
        plt.tight_layout()
        plt.savefig('dis_params1.png', dpi=300)
        plt.close()
        
        # PairPlot con alta calidad (PNG) usando solo parámetros visibles
        df_pair = df_plot.copy()
        if df_pair.shape[1] > 0:
            pairplot_fig = sns.pairplot(df_pair, kind='hist', corner=True)
            # Ajuste de tamaños de textos en cada eje del pairplot
            for ax in pairplot_fig.axes.flatten():
                if ax is not None:
                    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                        label.set_fontsize(label.get_fontsize() + factor_crecimiento_texto)
                    ax.title.set_fontsize(ax.title.get_fontsize() + factor_crecimiento_texto)
            pairplot_fig.savefig('dis_params2.png', dpi=600)
            plt.close()
        else:
            # Si no hay columnas para graficar (todos fijados), elimina archivo previo si existe
            try:
                if os.path.exists('dis_params2.png'):
                    os.remove('dis_params2.png')
            except Exception:
                pass
        
        # Gráfico de velocidad radial (RV plot - PNG)
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = mcolors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        
        cmap = plt.get_cmap('Greys_r')
        cmap_p = plt.get_cmap('Blues_r')
        cmap_s = plt.get_cmap('Reds_r')
        period = map_sample_all[0]
        
        t = np.linspace(df_v1.epoch.min(), df_v1.epoch.max(), 500)
        t0 = df_v1.epoch.min()
        phaseAn = foldAt(t, period, t0)
        sortIndi = np.argsort(phaseAn)
        phase = foldAt(df_v1.epoch, period, t0)
    
        plt.figure(figsize=(6,4))
        rnd_idx = np.random.choice(samples_all.shape[0], 1000, replace=False)
        for j in tqdm(range(1000)):
            P, T, e, w, V0, K = samples_all[rnd_idx[j]]
            V1 = getRVSingle(t, T, P, e, np.deg2rad(w), V0, K)
            plt.plot(phaseAn[sortIndi], V1[sortIndi], color='tab:cyan', alpha=0.005)
        P, T, e, w, V0, K = map_sample_all
        V1 = getRVSingle(t, T, P, e, np.deg2rad(w), V0, K)
        plt.plot(phaseAn[sortIndi], V1[sortIndi], linewidth=1, color='tab:blue')
        plt.errorbar(phase, df_v1.RV, yerr=df_v1.err, fmt=".k", capsize=3, ms=1)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.ylabel('radial velocity [$km/s$]', fontsize=12+factor_crecimiento_texto)
        plt.xlabel('Phase', fontsize=12+factor_crecimiento_texto)
        plt.tight_layout()
        plt.savefig('RV plot.png', dpi=600)
        plt.close()
        
        # Gráfico de la tabla de parámetros (PNG)
        plt.figure(figsize=(6,3))
        plt.title('Orbital Parameters and their Uncertainties', fontsize=14+factor_crecimiento_texto)
        table = plt.table(cellText=summary_table, loc='center', cellLoc='center', colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10+factor_crecimiento_texto)
        table.scale(1, 1.5)
        plt.axis('off')
        plt.savefig('latex_table.png', dpi=600)
        plt.close()
        
        # Creación del PDF con un resumen textual (se han eliminado las imágenes)
        styles = getSampleStyleSheet()
        pdf_filename = system + '_Summary.pdf'
        if system in ['HIP89000', 'HIP109951', 'HIP111170']:
            pdf_filename = system + '_' + nombre_componente + '_Summary.pdf'
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        elements = []
        title_style = styles['Title']
        title_text = '<b>Analysis Summary</b>'
        title = Paragraph(title_text, title_style)
        elements.append(title)
        process_details = Paragraph(
            'Se realizó un análisis de la astrometría y espectroscopía de los datos. '
            'El modelo bayesiano aplicado permitió estimar los parámetros orbitales del sistema.',
            styles['Normal']
        )
        # 1. Después de añadir los párrafos
        elements.append(Spacer(1, 12))

        # 2. Añade cada figura (usa rutas absolutas si es un directorio temporal)
        for fname, w, h in [
                ('dis_params1.png', 500, 200),
                ('dis_params2.png', 500, 380),
                ('RV plot.png',     500, 300),
                ('latex_table.png', 400, 200)]:
            path = os.path.abspath(fname)
            if os.path.exists(path):
                elements.append(Image(path, width=w, height=h, hAlign='CENTER'))
                elements.append(Spacer(1, 12))
            else:
                logging.warning(f'No se encontró {path}')

        elements.append(process_details)
        doc.build(elements)
        
        # Creación del archivo .rar que contiene el PDF y las imágenes PNG
        import zipfile

        # Crear una lista con los nombres de los archivos a incluir
        images = ['dis_params1.png', 'dis_params2.png', 'RV plot.png', 'latex_table.png']
        zip_filename = f'{system}_component_{number_component}_graphs_and_summary.zip'

        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Copiar el PDF al directorio temporal
                pdf_destination = os.path.join(tmpdirname, pdf_filename)
                shutil.copy(pdf_filename, pdf_destination)
                
                # Crear una carpeta "figures" y copiar las imágenes allí
                figures_dir = os.path.join(tmpdirname, "figures")
                os.makedirs(figures_dir, exist_ok=True)
                for img_file in images:
                    shutil.copy(img_file, os.path.join(figures_dir, os.path.basename(img_file)))
                
                # Crear el archivo zip usando zipfile
                current_dir = os.getcwd()
                zip_path = os.path.join(current_dir, zip_filename)
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Incluir todos los archivos y carpetas del directorio temporal
                    for root, dirs, files in os.walk(tmpdirname):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, tmpdirname)
                            zipf.write(file_path, arcname)
            
            # Leer el archivo zip creado y almacenarlo en session_state
            with open(zip_filename, "rb") as f:
                st.session_state.rar_data = f.read()
            st.session_state.rar_filename = zip_filename

        except Exception as e:
            st.error(f"Could not create the archive file: {e}")
        
        st.session_state.resultados_generados = True

        # Mostrar las imágenes PNG en la página antes de la descarga
        st.image('dis_params1.png', caption='Distribution for Each Parameter', use_column_width=True)
        st.image('dis_params2.png', caption='Corner Plot', use_column_width=True)
        st.image('RV plot.png', caption='Radial Velocity Curve', use_column_width=True)
        st.image('latex_table.png', use_column_width=True)

    # Función para limpiar (eliminar) los archivos locales
    def cleanup_files():
        archivos = ['dis_params1.png', 'dis_params2.png', 'RV plot.png', 'latex_table.png']
        for archivo in archivos:
            try:
                if os.path.exists(archivo):
                    os.remove(archivo)
            except Exception as e:
                st.error(f"Error eliminando {archivo}: {e}")
        # Eliminar el archivo .rar
        try:
            if os.path.exists(st.session_state.rar_filename):
                os.remove(st.session_state.rar_filename)
        except Exception as e:
            st.error(f"Error eliminando el archivo .rar: {e}")

    # Botón de descarga del archivo .rar
    if 'rar_data' in st.session_state:
        st.download_button(
            label="Download .rar with PDF and figures",
            data=st.session_state.rar_data,
            file_name=st.session_state.rar_filename,
            mime="application/x-rar-compressed",
            on_click=cleanup_files
        )
