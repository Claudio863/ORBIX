# BinaryStar_Optimizer.py
import streamlit as st
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'app/src'))

#  --- utilidades propias ---
from data_utils import load_user_data, estimate_period_lomb_scargle
from plot_utils import plot_initial_data, plot_sb2_combined_data
from stan_analysis import analisis_stan_, analisis_stan_sb2

# -----------------------------------------------------------------
st.set_page_config(page_title='ORBIX', layout='wide')

# --------- CSS para cuerpos grandes -------------------------------------------------
st.markdown("""
    <style>
    .main > div { font-size: 1.5em !important; }
    .stMarkdown p, .stMarkdown ul, .stMarkdown li,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 { font-size: 1.5em !important; }
    .stButton button { font-size: 1.3em !important; }
    .stExpanderHeader { font-size: 1.3em !important; }
    .stRadio label, .stSelectbox label, .stSlider label,
    .stNumberInput label { font-size: 1.2em !important; }
    </style>
""", unsafe_allow_html=True)

st.title('ORBIX')

st.markdown("""
<details style='border: 0.5px solid #ccc; border-radius: 5px;
         padding: 10px; background-color: #d9f7e8;'>
<summary style='cursor: pointer;'>Description</summary>
<p>
Interactive application for the analysis of spectroscopic binary systems using Bayesian inference.
Load your data, configure parameters, and adjust optimization visually and intuitively.
</p>
</details>
""", unsafe_allow_html=True)

# =============================================================================
# 1️⃣ DATA UPLOAD
# =============================================================================
with st.expander('1️⃣ Data upload', expanded=True):
    #data_election = st.radio('Data source:', ['User data', 'Example data'])
    data_election= "User data"
    # Selector de tipo de sistema
    system_type = st.radio('System type:',
                           ['SB1 (Single-lined)', 'SB2 (Double-lined)'])

    df_v1 = df_v2 = None          # inicialización
    component_name = ''
    number_component = '1'
    system = ''

    # -------------------------------------------------------------------------
    # OPCIÓN: CARGA DE DATOS DEL USUARIO
    # -------------------------------------------------------------------------
    if data_election == 'User data':
        st.info('The file must have the columns: epoch, RV, err')

        # ★   checkbox: ¿convertir época?
        convert_epochs = st.checkbox(
            'Convert epoch column to JD (recommended)',
            value=True,
            help=('Uncheck if your "epoch" column está en unidades definitivas '
                  'y prefieres no hacer ninguna transformación.'))

        # selector de formato (si se va a convertir)
        epoch_format = st.selectbox(
            'Epoch format',
            ('Heliocentric Julian Day (HJD)', 'Julian Day (JD)', 'Years (yr)'),
            help='Select the format of the epoch column in your file.')
        dic_epoch_formats = {
            'Heliocentric Julian Day (HJD)': 'HJD',
            'Julian Day (JD)': 'D',
            'Years (yr)': 'yr'
        }
        
        epoch_format_graphs = dic_epoch_formats[epoch_format]

        if convert_epochs:
            epoch_format_graphs= "D"
        # --------------  SB1  ----------------
        if system_type == 'SB1 (Single-lined)':
            uploaded_file = st.file_uploader('Select a file', type=['csv'])

            if uploaded_file:
                df_v1 = load_user_data(
                    uploaded_file,
                    convert_epochs=convert_epochs,
                    epoch_format=epoch_format
                )
                number_component = st.selectbox('Component', ('1', '2'))
                system = st.text_input('System name', value='System 1')
                st.pyplot(plot_initial_data(df_v1, number_component, system,epoch_format))

        # --------------  SB2  ----------------
        else:   # SB2 (Double-lined)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Primary component (V1)**")
                uploaded_file_v1 = st.file_uploader(
                    'Select file for V1', type=['csv'], key='v1_file')
            with col2:
                st.markdown("**Secondary component (V2)**")
                uploaded_file_v2 = st.file_uploader(
                    'Select file for V2', type=['csv'], key='v2_file')

            system = st.text_input('System name', value='SB2 System 1')

            # Conversión idéntica en ambos ficheros, si procede
            if uploaded_file_v1 and uploaded_file_v2:
                df_v1 = load_user_data(
                    uploaded_file_v1, convert_epochs, epoch_format)
                df_v2 = load_user_data(
                    uploaded_file_v2, convert_epochs, epoch_format)
                st.markdown("**Combined V1 and V2 data preview:**")
                st.pyplot(plot_sb2_combined_data(df_v1, df_v2, system,epoch_format))

            elif uploaded_file_v1:
                df_v1 = load_user_data(
                    uploaded_file_v1, convert_epochs, epoch_format)
                st.markdown("**V1 data preview (V2 pending):**")
                st.pyplot(plot_initial_data(df_v1, 'V1', system,epoch_format))

            elif uploaded_file_v2:
                df_v2 = load_user_data(
                    uploaded_file_v2, convert_epochs, epoch_format)
                st.markdown("**V2 data preview (V1 pending):**")
                st.pyplot(plot_initial_data(df_v2, 'V2', system,epoch_format))

    # -------------------------------------------------------------------------
    # OPCIÓN: DATOS DE EJEMPLO (sin cambios salvo que df_v1 siempre ya es HJD)
    # -------------------------------------------------------------------------
    else:
        from lectura_datos import lec_type_examples, lec_type_star02, lec_type_star01

        system = st.selectbox('Select system',
                              ['HIP89000', 'HIP109951', 'HIP111170',
                               'star01', 'star02'])

        if system_type == 'SB1 (Single-lined)':
            if system in ['HIP89000', 'HIP109951', 'HIP111170']:
                component = st.radio('Component', ['V1', 'V2'])
                df_v1, df_v2_temp = lec_type_examples(system)
                component_name = 'V1'
                if component == 'V2' and not df_v2_temp.empty:
                    df_v1 = df_v2_temp
                    component_name = 'V2'
                elif component == 'V2':
                    st.warning('No data for V2, V1 will be used.')
            elif system == 'star01':
                df_v1 = lec_type_star01(str(REPO/'app/star01.dat'))
            else:  # star02
                df_v1 = lec_type_star02(str(REPO/'app/star02.dat'))

            st.pyplot(plot_initial_data(df_v1, component_name or '1', system,epoch_format))

        else:  # SB2 example
            if system in ['HIP89000', 'HIP109951', 'HIP111170']:
                df_v1, df_v2 = lec_type_examples(system)
                if not df_v2.empty:
                    st.markdown("**Combined V1 and V2 data preview:**")
                    st.pyplot(plot_sb2_combined_data(df_v1, df_v2, system,epoch_format))
                else:
                    st.warning('No V2 data available for this system. Using V1 data only.')
                    df_v1, _ = lec_type_examples(system)
                    st.pyplot(plot_initial_data(df_v1, 'V1', system))
            else:
                st.warning('SB2 mode is only available for HIP systems with both V1 and V2 data.')
                df_v1 = lec_type_star01(str(REPO/'app/star01.dat')) \
                    if system == 'star01' else lec_type_star02(str(REPO/'app/star02.dat'))
                st.pyplot(plot_initial_data(df_v1, '1', system))

# =============================================================================
#  SIDEBAR  –  PARAMETER CONFIGURATION
# =============================================================================
if df_v1 is not None:
    with st.sidebar:
        st.header('⚙️ Parameter configuration')
        st.markdown('**Initial parameters**')

        method_P = st.radio(
            'Period initialization',
            ['Manual', 'Automatic (Lomb–Scargle)'])

        if method_P == 'Manual':
            P_init = st.number_input(
                'Estimated period', min_value=0.0,
                max_value=1e7, value=100.0, step=0.000001, format="%.6f")
        else:
            st.warning('Automatic estimation using Lomb–Scargle may not be close '
                       'to the optimum for all systems. Please review and adjust manually if necessary.',
                       icon="⚠️")

            # ★ Si NO se convierte la época, asumimos que estimate_period_lomb_scargle
            #   ya trabaja con las unidades correctas -> pasamos 'HJD' (dummy).
            epoch_fmt_for_ls = epoch_format if convert_epochs else 'HJD'
            P_init = estimate_period_lomb_scargle(df_v1, epoch_fmt_for_ls)
            st.success(f'Estimated period: {P_init:.4f} days')

        fix_P = st.checkbox('Fix period during sampling', value=False)

        # ==================================================
        #   Eccentricity
        # ==================================================
        entry_type = st.radio('Eccentricity (e)',
                              ['Single value', 'Range'])

        if entry_type == 'Single value':
            e_init = st.number_input(
                'Estimated eccentricity (e)', min_value=0.0, max_value=1.0,
                value=0.3, step=0.000001, format="%.6f")
            e_init_min = e_init_max = None
        else:
            e_init_min, e_init_max = st.slider(
                'e interval', min_value=0.0, max_value=1.0,
                value=(0.2, 0.4), step=0.01)
            e_init = 0.5 * (e_init_min + e_init_max)
        fix_e = st.checkbox('Fix eccentricity (no sampling)', value=False)

        # Nota: Se eliminan inputs SB2 de q, a*sin(i) y Parallax; no se usan como entradas

    # =============================================================================
    # 3️⃣ ADVANCED OPTIMIZATION SETTINGS
    # =============================================================================
    with st.expander('3️⃣ Advanced optimization settings', expanded=False):
        st.markdown(
            'You can modify the optimization hyperparameters. '
            'Default values are usually suitable.')

        iter_opt = st.number_input(
            'Iterations (optimization)', min_value=100,
            max_value=10000, value=4000, step=100,
            help='Number of iterations for initial optimization.')

        tol_obj = st.number_input('Objective tolerance',
                                  min_value=1e-12, max_value=1e-3,
                                  value=1e-6, step=1e-7, format="%e")
        tol_grad = st.number_input('Gradient tolerance',
                                   min_value=1e-12, max_value=1e-3,
                                   value=1e-6, step=1e-7, format="%e")
        tol_param = st.number_input('Parameter tolerance',
                                    min_value=1e-12, max_value=1e-3,
                                    value=1e-6, step=1e-7, format="%e")
        history_size = st.slider('History size', min_value=5,
                                 max_value=30, value=20)
        algorithm = st.selectbox(
            'Algorithm', ['lbfgs', 'bfgs'],
            help='Optimization method')

        optim_config = dict(iter=iter_opt, tol_obj=tol_obj,
                            tol_grad=tol_grad, tol_param=tol_param,
                            history_size=history_size,
                            algorithm=algorithm)

    # =============================================================================
    # 4️⃣ OPTIMIZATION AND ANALYSIS
    # =============================================================================
    with st.expander('4️⃣ Optimization and analysis', expanded=True):
        ready = True
        for v in [P_init, e_init]:
            if v is None:
                ready = False

        if system_type == 'SB2 (Double-lined)':
            if df_v2 is None or df_v2.empty:
                ready = False
                st.warning('SB2 analysis requires data for both components (V1 and V2).')

        if ready and st.button('Run analysis', type='primary'):
            # Limpiar posibles variables de una ejecución previa
            for key in ['resultados_generados', 'pdf_data',
                        'rar_data', 'pdf_filename', 'rar_filename']:
                st.session_state.pop(key, None)

            # Garantizar índices consecutivos y sin NaN
            df_v1 = df_v1.dropna().reset_index(drop=True)
            if system_type == 'SB2 (Double-lined)' and df_v2 is not None:
                df_v2 = df_v2.dropna().reset_index(drop=True)

            V_delta = df_v1.RV.max() - df_v1.RV.min()

            if entry_type == 'Single value':
                desfase_min, desfase_max = -0.05 + e_init, 0.05 + e_init
            else:
                e_init = 0.0
                desfase_min, desfase_max = e_init_min, e_init_max

            # ----------  LANZA STAN  ----------
            if system_type == 'SB1 (Single-lined)':
                analisis_stan_(
                    df_v1=df_v1,
                    P_init=P_init,
                    e_init=e_init,
                    desfase_max=desfase_max,
                    desfase_min=desfase_min,
                    V_delta=V_delta,
                    system=system,
                    nombre_componente=component_name or number_component,
                    number_component=number_component,
                    fix_e=fix_e,
                    fix_P=fix_P,
                    optim_config=optim_config,
                    epoch_format_graphs=epoch_format_graphs
                )
            else:  # SB2
                analisis_stan_sb2(
                    df_v1=df_v1,
                    df_v2=df_v2,
                    P_init=P_init,
                    e_init=e_init,
                    desfase_max=desfase_max,
                    desfase_min=desfase_min,
                    V_delta=V_delta,
                    system=system,
                    fix_e=fix_e,
                    fix_P=fix_P,
                    optim_config=optim_config,
                    epoch_format_graphs=epoch_format_graphs
                )
        else:
            st.info('Configure the parameters and click "Run analysis" to see the results.')

    # =============================================================================
    # 5️⃣ FINAL RESULTS AND PLOTS
    # =============================================================================
    if st.session_state.get('resultados_generados', False):
        with st.expander('5️⃣ Final results and plots', expanded=True):
            import os

            # 1. Distribución de parámetros
            if os.path.exists('dis_params1.png'):
                st.image('dis_params1.png',
                         caption='Distribution for Each Parameter',
                         use_column_width=True)

            # 2. Selector de curvas
            if system_type == 'SB1 (Single-lined)':
                plot_options = {
                    'Radial Velocity Curve (Phase)': ('RV_phase_plot.png', 'Phase curve'),
                    'Radial Velocity Curve (Time)' : ('RV_time_plot.png',  'RV vs Time')
                }
            else:
                plot_options = {
                    'SB2 Radial Velocity Curve (Phase)': ('RV_phase_plot_sb2.png', 'SB2 Phase curve'),
                    'SB2 Radial Velocity Curve (Time)' : ('RV_time_plot_sb2.png',  'SB2 RV vs Time')
                }

            plot_choice = st.selectbox('Select plot to display:',
                                       list(plot_options.keys()))
            plot_file, plot_caption = plot_options[plot_choice]

            if os.path.exists(plot_file):
                st.image(plot_file, caption=plot_caption, use_column_width=True)

            # 3. Otros gráficos
            for fname, caption in [('dis_params2.png', 'Corner Plot'),
                                   ('latex_table.png', None)]:
                if os.path.exists(fname):
                    st.image(fname, caption=caption, use_column_width=True)

            # 4. Botón download .rar
            if 'rar_data' in st.session_state:
                def cleanup_files():
                    # eliminar imágenes y .rar temporales
                    if system_type == 'SB1 (Single-lined)':
                        files = ['dis_params1.png', 'dis_params2.png',
                                 'RV_phase_plot.png', 'RV_time_plot.png',
                                 'latex_table.png']
                    else:
                        files = ['dis_params1.png', 'dis_params2.png',
                                 'RV_phase_plot_sb2.png', 'RV_time_plot_sb2.png',
                                 'latex_table.png']
                    for f in files:
                        try:
                            if os.path.exists(f):
                                os.remove(f)
                        except Exception:
                            pass
                    try:
                        if os.path.exists(st.session_state.rar_filename):
                            os.remove(st.session_state.rar_filename)
                    except Exception:
                        pass

                st.download_button(
                    label=f"Download .rar with PDF and figures "
                          f"({'SB1' if system_type.startswith('SB1') else 'SB2'})",
                    data=st.session_state.rar_data,
                    file_name=st.session_state.rar_filename,
                    mime="application/x-rar-compressed",
                    on_click=cleanup_files,
                    key="download_rar_button_final_results"
                )
