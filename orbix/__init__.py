"""ORBIX — Optimización de paRámetros orBitales en bInarias espectroscópicas con inferencia bayesiana.

Módulos:
    fechas   conversión de épocas entre formatos (JD, HJD, BJD, RJD, MJD, TJD, años julianos/besselianos)
    config   carga de config/sistemas.csv y config/priors.csv, P_init y resolución de priors
    periodo  periodograma kepleriano y siembra global del período para L-BFGS (Fase 1)
    modelos  selección y compilación de los .stan (sb1_K*, sb2_K*)
    ajuste   corrida completa de un (sistema, tipo, esquema): L-BFGS + NUTS + persistencia
    resumen  resumen de posteriores, diagnósticos, χ² y residuos (lee las cadenas guardadas)
    rv       modelo kepleriano de velocidad radial en numpy
    graficos figuras y tablas de presentación (notebook 02)
"""
