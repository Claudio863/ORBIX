"""Selección y compilación de los modelos Stan de ORBIX (sb1_K*, sb2_K*)."""
from __future__ import annotations

import re
import os
import shutil
from pathlib import Path

from cmdstanpy import CmdStanModel
from cmdstanpy.utils.cmdstan import cxx_toolchain_path

ESQUEMAS = ("free", "fix_P", "fix_e", "fix_eP")


def preparar_header(nombre: str, plantilla: Path, dir_build: Path) -> Path:
    """Copia privada del solver C++; conserva inmutable la plantilla compartida."""
    header = Path(dir_build) / "headers" / nombre / "kepler_eq.hpp"
    header.parent.mkdir(parents=True, exist_ok=True)
    texto = re.sub(r"\S*_model_namespace", f"{nombre}_model_namespace",
                   Path(plantilla).read_text(encoding="utf-8"))
    if not header.exists() or header.read_text(encoding="utf-8") != texto:
        header.write_text(texto, encoding="utf-8")
    return header


def nombre_modelo(tipo: str, esquema: str) -> str:
    if tipo not in ("SB1", "SB2"):
        raise ValueError(f"tipo debe ser SB1 o SB2, no {tipo!r}")
    if esquema not in ESQUEMAS:
        raise ValueError(f"esquema debe ser uno de {ESQUEMAS}, no {esquema!r}")
    sufijo = "" if esquema == "free" else f"_{esquema}"
    return f"sb{tipo[-1]}_K{sufijo}"


def fijos(esquema: str) -> tuple[bool, bool]:
    """(fix_P, fix_e) para un esquema."""
    return esquema in ("fix_P", "fix_eP"), esquema in ("fix_e", "fix_eP")


def compilar(tipo: str, esquema: str, dir_stan: Path, dir_build: Path) -> CmdStanModel:
    """Compila (o reutiliza) el modelo.

    El header kepler_eq.hpp debe declarar el namespace `<modelo>_model_namespace`.
    En vez de reescribir el header compartido (como hace la app), se genera una copia
    por modelo en `dir_build`, así varias corridas o la app no se pisan.
    """
    nombre = nombre_modelo(tipo, esquema)
    if os.name == "nt" and not shutil.which(os.environ.get("MAKE", "mingw32-make")):
        cxx_toolchain_path()
    stan_file = Path(dir_stan) / f"{nombre}.stan"
    if not stan_file.exists():
        raise FileNotFoundError(stan_file)
    header = preparar_header(nombre, Path(dir_stan) / "kepler_eq.hpp", dir_build)
    return CmdStanModel(stan_file=str(stan_file), user_header=str(header))
