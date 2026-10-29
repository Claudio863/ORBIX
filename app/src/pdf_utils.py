from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import os
import shutil
import tempfile
import zipfile
import streamlit as st
import pandas as pd
from typing import Optional

def create_pdf_and_zip(pdf_filename, summary_table, images, system, nombre_componente, number_component, results_df: Optional[pd.DataFrame] = None):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []
    title_style = styles['Title']
    title_text = '<b>Analysis Summary</b>'
    title = Paragraph(title_text, title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    for fname, w, h in images:
        path = os.path.abspath(fname)
        if os.path.exists(path):
            elements.append(Image(path, width=w, height=h, hAlign='CENTER'))
            elements.append(Spacer(1, 12))
    doc.build(elements)
    zip_filename = f'{system}_component_{number_component}_graphs_and_summary.zip'
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            pdf_destination = os.path.join(tmpdirname, pdf_filename)
            shutil.copy(pdf_filename, pdf_destination)
            # Si se proporciona un DataFrame de resultados, guardarlo como CSV
            if results_df is not None and not results_df.empty:
                csv_path = os.path.join(tmpdirname, 'results_summary.csv')
                try:
                    results_df.to_csv(csv_path, index=False)
                except Exception as e:
                    # Fallback: intentar construir un DataFrame desde summary_table
                    try:
                        pd.DataFrame(summary_table, columns=['Parameter', 'Estimate']).to_csv(csv_path, index=False)
                    except Exception:
                        pass
            figures_dir = os.path.join(tmpdirname, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            for img_file, _, _ in images:
                shutil.copy(img_file, os.path.join(figures_dir, os.path.basename(img_file)))
            current_dir = os.getcwd()
            zip_path = os.path.join(current_dir, zip_filename)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(tmpdirname):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, tmpdirname)
                        zipf.write(file_path, arcname)
        with open(zip_filename, "rb") as f:
            st.session_state.rar_data = f.read()
        st.session_state.rar_filename = zip_filename
    except Exception as e:
        st.error(f"Could not create the archive file: {e}")
