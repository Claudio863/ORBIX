"""Conversión de fechas de ORBIX.

Todas las conversiones pasan por la fecha juliana absoluta (JD):

    formato de entrada  ──a_jd()──▶  JD  ──desde_jd()──▶  formato de trabajo / de salida

Así, cualquier combinación entrada → trabajo → salida es posible y queda registrada.
El modelo Stan es indiferente a la unidad: recibe épocas en el formato de trabajo y
devuelve P y T en esa misma unidad (días para formatos tipo JD, años para formatos
tipo año).

Escala de tiempo (JD / HJD / BJD): el código NO aplica correcciones heliocéntricas
ni baricéntricas (requieren coordenadas del objeto). La etiqueta se conserva como
metadato; convertir HJD→BJD aquí solo cambia el nombre, no el valor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Constantes astronómicas
JD_J2000 = 2451545.0               # 2000-01-01 12:00 TT
ANIO_JULIANO = 365.25              # días
JD_B1900 = 2415020.31352           # época besseliana B1900.0
ANIO_BESSELIANO = 365.242198781    # año trópico (días)


@dataclass(frozen=True)
class Formato:
    nombre: str
    tipo: str            # 'dia' | 'anio'
    offset: float        # para 'dia': valor = JD - offset
    escala: str          # 'JD' | 'HJD' | 'BJD' | 'sin especificar'
    descripcion: str


FORMATOS: dict[str, Formato] = {f.nombre: f for f in [
    Formato("JD",  "dia", 0.0,       "JD",  "Fecha juliana completa (~2.45e6)"),
    Formato("HJD", "dia", 0.0,       "HJD", "Fecha juliana heliocéntrica completa"),
    Formato("BJD", "dia", 0.0,       "BJD", "Fecha juliana baricéntrica completa"),
    Formato("RJD", "dia", 2400000.0, "sin especificar", "JD reducida: JD − 2 400 000 (p. ej. 'HJD−2400000')"),
    Formato("MJD", "dia", 2400000.5, "sin especificar", "JD modificada: JD − 2 400 000.5"),
    Formato("TJD", "dia", 2450000.0, "sin especificar", "JD truncada: JD − 2 450 000 (p. ej. 'BJD−2450000')"),
    Formato("year_julian",    "anio", np.nan, "sin especificar", "Época juliana decimal (J2000, año = 365.25 d)"),
    Formato("year_besselian", "anio", np.nan, "sin especificar", "Época besseliana decimal (B1900, año = 365.242198781 d)"),
]}
# Formato genérico 'JD-<offset>' (p. ej. 'JD-2455000') se resuelve en formato()


def formato(nombre: str) -> Formato:
    """Devuelve el Formato por nombre. Acepta 'JD-<offset>' para offsets arbitrarios."""
    if nombre in FORMATOS:
        return FORMATOS[nombre]
    if isinstance(nombre, str) and nombre.upper().startswith("JD-"):
        off = float(nombre[3:])
        return Formato(nombre, "dia", off, "sin especificar", f"JD − {off:g}")
    raise ValueError(f"Formato de fecha desconocido: {nombre!r}. Opciones: {list(FORMATOS)} o 'JD-<offset>'")


# --------------------------------------------------------------------------- épocas
def a_jd(t, nombre_formato: str) -> np.ndarray:
    """Convierte épocas en `nombre_formato` a JD absoluto."""
    f = formato(nombre_formato)
    t = np.asarray(t, dtype=float)
    if f.tipo == "dia":
        return t + f.offset
    if f.nombre == "year_julian":
        return JD_J2000 + (t - 2000.0) * ANIO_JULIANO
    if f.nombre == "year_besselian":
        return JD_B1900 + (t - 1900.0) * ANIO_BESSELIANO
    raise ValueError(nombre_formato)


def desde_jd(jd, nombre_formato: str) -> np.ndarray:
    """Convierte JD absoluto al formato pedido."""
    f = formato(nombre_formato)
    jd = np.asarray(jd, dtype=float)
    if f.tipo == "dia":
        return jd - f.offset
    if f.nombre == "year_julian":
        return 2000.0 + (jd - JD_J2000) / ANIO_JULIANO
    if f.nombre == "year_besselian":
        return 1900.0 + (jd - JD_B1900) / ANIO_BESSELIANO
    raise ValueError(nombre_formato)


def convertir_epocas(t, de: str, a: str) -> np.ndarray:
    """Convierte épocas entre dos formatos cualesquiera (vía JD)."""
    if de == a:
        return np.asarray(t, dtype=float)
    return desde_jd(a_jd(t, de), a)


# --------------------------------------------------------------------------- duraciones (P)
def unidad_periodo(nombre_formato: str) -> str:
    """Unidad en que queda P cuando las épocas están en `nombre_formato`."""
    f = formato(nombre_formato)
    if f.tipo == "dia":
        return "d"
    return "yr_J" if f.nombre == "year_julian" else "yr_B"


_DIAS_POR = {"d": 1.0, "yr_J": ANIO_JULIANO, "yr_B": ANIO_BESSELIANO,
             "yr": ANIO_JULIANO}   # 'yr' genérico = juliano (se registra la ambigüedad)


def convertir_duracion(x, de: str, a: str) -> np.ndarray:
    """Convierte una duración (p. ej. P o su incertidumbre) entre 'd', 'yr_J', 'yr_B'."""
    try:
        return np.asarray(x, dtype=float) * _DIAS_POR[de] / _DIAS_POR[a]
    except KeyError as exc:
        raise ValueError(f"Unidad de duración desconocida: {exc}. Opciones: {list(_DIAS_POR)}") from None


def etiqueta_unidad(nombre_formato_o_unidad: str) -> str:
    """Texto corto para ejes y encabezados."""
    if nombre_formato_o_unidad in _DIAS_POR:
        return {"d": "d", "yr_J": "yr (juliano)", "yr_B": "yr (besseliano)", "yr": "yr"}[nombre_formato_o_unidad]
    f = formato(nombre_formato_o_unidad)
    if f.tipo == "anio":
        return "año juliano" if f.nombre == "year_julian" else "año besseliano"
    return etiqueta_origen(f.nombre)


def etiqueta_origen(nombre_formato: str, escala: str | None = None) -> str:
    """Unidad y ORIGEN de una época, para encabezados: p. ej. 'BJD − 2400000', 'HJD − 2450000',
    'año besseliano'. `escala` (JD/HJD/BJD, de sistemas.csv) reemplaza la escala genérica del formato."""
    f = formato(nombre_formato)
    if f.tipo == "anio":
        return "año juliano" if f.nombre == "year_julian" else "año besseliano"
    esc = escala if escala and escala != "sin especificar" else ("JD" if f.escala == "sin especificar" else f.escala)
    if f.nombre == "MJD":
        return f"MJD ({esc} − 2400000.5)"
    return esc if f.offset == 0 else f"{esc} − {f.offset:.10g}"


# --------------------------------------------------------------------------- detección y validación
def anio_calendario(jd) -> np.ndarray:
    """Año calendario aproximado (juliano) de un JD, para chequeos de plausibilidad."""
    return 2000.0 + (np.asarray(jd, dtype=float) - JD_J2000) / ANIO_JULIANO


def sugerir_formato(t) -> tuple[str, str]:
    """Sugiere el formato de una serie de épocas a partir de su rango.

    Es solo una ayuda: los rangos se solapan (p. ej. años besselianos vs julianos son
    indistinguibles; RJD vs MJD difieren en 0.5 d). La decisión final va en
    config/sistemas.csv y queda registrada.
    """
    t = np.asarray(t, dtype=float)
    lo, hi = np.nanmin(t), np.nanmax(t)
    if lo > 2.0e6:
        return "JD", "valores ~2.4e6: JD/HJD/BJD completo (la escala no se puede inferir)"
    if 1000 < lo and hi < 3000:
        return "year_besselian", "valores 1000–3000: año decimal (juliano o besseliano: no distinguible)"
    if 10000 <= lo < 100000:
        return "RJD", "valores 1e4–1e5: JD − 2 400 000 (o MJD, difieren en 0.5 d)"
    if 0 <= lo and hi < 10000:
        return "TJD", "valores 0–1e4: probablemente JD − 2 450 000 (verificar en la fuente)"
    return "desconocido", f"rango [{lo:g}, {hi:g}] no reconocido"


def validar_epocas(t, nombre_formato: str, rango_anios=(1850.0, 2100.0)) -> dict:
    """Comprueba que las épocas, interpretadas en `nombre_formato`, caigan en fechas plausibles."""
    jd = a_jd(t, nombre_formato)
    y = anio_calendario(jd)
    ok = bool(np.all((y >= rango_anios[0]) & (y <= rango_anios[1])))
    return dict(formato=nombre_formato, anio_min=float(np.min(y)), anio_max=float(np.max(y)), plausible=ok)


def tabla_formatos() -> pd.DataFrame:
    """Formatos soportados (para mostrar en el notebook)."""
    return pd.DataFrame([dict(formato=f.nombre, tipo=f.tipo, offset=f.offset, escala=f.escala,
                              unidad_P=unidad_periodo(f.nombre), descripcion=f.descripcion)
                         for f in FORMATOS.values()])


# --------------------------------------------------------------------------- conversión de resultados
def convertir_parametro(valores, parametro: str, formato_trabajo: str, formato_salida: str):
    """Convierte valores de P (duración) o T (época) del formato de trabajo al de salida.

    Ambas transformaciones son lineales y monótonas crecientes, por lo que mediana,
    MAP y límites del HPDI se transforman directamente.
    Parámetros distintos de P/T se devuelven sin cambios.
    """
    if parametro == "P":
        return convertir_duracion(valores, unidad_periodo(formato_trabajo), unidad_periodo(formato_salida))
    if parametro == "T":
        return convertir_epocas(valores, formato_trabajo, formato_salida)
    return np.asarray(valores, dtype=float)


def escala_anchura(parametro: str, formato_trabajo: str, formato_salida: str) -> float:
    """Factor para convertir anchuras/desviaciones (de P o de T) entre formatos."""
    if parametro in ("P", "T"):
        return float(convertir_duracion(1.0, unidad_periodo(formato_trabajo), unidad_periodo(formato_salida)))
    return 1.0
