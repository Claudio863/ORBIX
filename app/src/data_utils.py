import pandas as pd
import streamlit as st
import astropy.units as u
from astropy.timeseries import LombScargle

import pandas as pd

# -----------------------------------------------------------
#  CARGA DEL CSV  +  CONVERSIÓN OPCIONAL DE 'epoch' A  HJDmod
# -----------------------------------------------------------
def load_user_data(
        uploaded_file,
        convert_epochs=True,
        epoch_format='Heliocentric Julian Day (HJD)'
    ):
    """
    Parameters
    ----------
    uploaded_file : IO buffer o ruta al CSV.
    convert_epochs : bool
        Si True se crea la columna 'HJD' con la siguiente regla:
           - 'Heliocentric Julian Day (HJD)':  HJD = epoch − 2 400 000.0
           - 'Julian Day (JD)'              :  HJD = epoch − 2 400 000.0
           - 'Years (yr)'                   :  HJD = ((epoch−2000)*365.25 + 2 451 545.0) − 2 400 000.0
        y luego se reemplaza df['epoch'] por df['HJD'].
        Si False, simplemente se asegura de que los tipos sean numéricos y
        copia la columna original en `df['HJD']` para compatibilidad.
    epoch_format : str
        Etiqueta seleccionada por el usuario en la interfaz.
    """
    # ----------  lectura ----------
    df = pd.read_csv(uploaded_file)

    # Asegurar tipos numéricos
    for col in ('epoch', 'RV', 'err'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['epoch'])

    # ----------  conversión opcional ----------
    if convert_epochs:
        # Diccionario de funciones de conversión
        convert = {
            'Heliocentric Julian Day (HJD)': lambda s: s - 2_400_000.0,
            'Julian Day (JD)'              : lambda s: s - 2_400_000.0,
            'Years (yr)'                   : lambda s: ((s - 2000.0) * 365.25 + 2_451_545.0) - 2_400_000.0,
        }
        if epoch_format not in convert:
            raise ValueError(f"Formato de época no reconocido: {epoch_format}")

        df['HJD']  = convert[epoch_format](df['epoch'])
        df['epoch'] = df['HJD']          # se usa siempre la escala HJDmod
    else:
        # Sin transformación: copiamos para mantener compatibilidad
        df['HJD'] = df['epoch']

    return df

def estimate_period_lomb_scargle(df_v1, formato_epoca):
    if formato_epoca == 'Years (yr)':
        t_lomb = (df_v1['epoch'].values * u.year).to(u.day)
    else:
        t_lomb = df_v1['epoch'].values * u.day
    y_lomb  = df_v1['RV'].values  * (u.km/u.s)
    dy_lomb = df_v1['err'].values * (u.km/u.s)
    min_period = 10 * u.day
    max_period = 5000 * u.day
    ls = LombScargle(t_lomb, y_lomb, dy_lomb)
    freq, power = ls.autopower(
        minimum_frequency=1/max_period,
        maximum_frequency=1/min_period
    )
    best_frequency = freq[power.argmax()]
    P_init = (1/best_frequency).to(u.day).value
    return P_init
