"""Independent simulation primitives from the archived experimental implementation."""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
RAIZ=ROOT/'análisis_desempeño/corridas_masivas'
STAN=ROOT/'modelos'
sys.path.insert(0,str(ROOT))

def rv_independiente(t, verdad, componente):
    """Solución por raíz acotada; no utiliza el Newton ni el C++ de ORBIX."""
    M = np.remainder(2*np.pi*(np.asarray(t)-verdad['T'])/verdad['P'], 2*np.pi)
    e = verdad['e']
    E = np.array([brentq(lambda x: x-e*np.sin(x)-m, 0., 2*np.pi, xtol=1e-13) for m in M])
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    w = np.deg2rad(verdad['w_deg'])
    k = verdad['K'] if verdad['tipo']=='SB1' else verdad['K1' if componente=='v1' else 'K2']
    return verdad['V0']+(1 if componente=='v1' else -1)*k*(np.cos(w+nu)+e*np.cos(w))

def wilson(k, n):
    if n==0:
        return np.nan, np.nan
    z = 1.959963984540054
    den = 1+z*z/n
    centro = (k/n+z*z/(2*n))/den
    radio = z*np.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/den
    return max(0., centro-radio), min(1., centro+radio)
