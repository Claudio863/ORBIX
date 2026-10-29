import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import arviz as az
import matplotlib.colors as mcolors
from PyAstronomy.pyasl import foldAt

def plot_initial_data(df_v1, number_component, system,format_epoch):
    plt.figure(figsize=(10, 6))
    
    # Usar la columna de tiempo apropiada
    time_col = 'HJD' if 'HJD' in df_v1.columns else 'epoch'
    time_label = f'{format_epoch}'
    
    plt.errorbar(df_v1[time_col], df_v1['RV'], yerr=df_v1['err'],
                 fmt='.', elinewidth=1, ms=5, alpha=0.75,
                 label=f'Radial Velocity ({system})')
    plt.ylabel('Radial Velocity [km/s]')
    plt.xlabel(time_label)
    plt.title(f'Spectroscopy of component {number_component} of the {system} system')
    plt.legend(loc='upper left')
    plt.tight_layout()
    return plt

def plot_sb2_combined_data(df_v1, df_v2, system, format_epoch):
    """
    Plot both V1 and V2 components in a single figure with different colors.
    
    Parameters:
    df_v1: DataFrame with V1 data
    df_v2: DataFrame with V2 data  
    system: System name for the title
    """
    plt.figure(figsize=(12, 7))
    
    # Usar la columna de tiempo apropiada
    time_col_v1 = 'HJD' if 'HJD' in df_v1.columns else 'epoch'
    time_col_v2 = 'HJD' if 'HJD' in df_v2.columns else 'epoch'
    time_label = f'{format_epoch}'
    
    # Plot V1 component (primary) en azul
    plt.errorbar(df_v1[time_col_v1], df_v1['RV'], yerr=df_v1['err'],
                 fmt='.', elinewidth=1, ms=5, alpha=0.75,
                 label='V1 (Primary component)')
    
    # Plot V2 component (secondary) en rojo
    plt.errorbar(df_v2[time_col_v2], df_v2['RV'], yerr=df_v2['err'],
                 fmt='.', elinewidth=1, ms=5, alpha=0.75,
                 label='V2 (Secondary component)')
    
    plt.ylabel('Radial Velocity [km/s]')
    plt.xlabel(time_label)
    plt.title(f'Spectroscopy of the {system} binary system (SB2)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

# Puedes agregar aquí más funciones de gráficos reutilizables
