"""ORBIX interactive entry point using the shared scientific pipeline."""
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.interfaz import renderizar

st.set_page_config(page_title="ORBIX", layout="wide")
renderizar()
