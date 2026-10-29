"""Verificación autocontenida del archivo científico; no ejecuta inferencia."""
from pathlib import Path
import hashlib,json,subprocess,sys
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parents[2]

def main():
    manifest=json.loads((ROOT/'reports/manifest_archivos_cientificos.json').read_text(encoding='utf-8'))
    for name,expected in manifest['archivos'].items():
        path=ROOT/name
        assert path.resolve().is_relative_to(ROOT) and path.is_file(),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    observed=ROOT/'análisis_desempeño/corridas_masivas/resultados/observados_final_20261029'
    table=pd.read_csv(observed/'tablas/evaluacion_estricta.csv')
    assert len(table)==36 and not table.run_id.duplicated().any()
    assert table.estricto.sum()==34
    for name in table.run_id:
        assert len(list((observed/'draws'/name).glob('sb*_K*.csv')))==4,name
    simulation=ROOT/'análisis_desempeño/simulaciones'
    reg=pd.read_csv(simulation/'registro.csv')
    assert len(reg)==100 and reg.estado.eq('ok').all()
    for case in reg.caso:
        assert len(list((simulation/'casos'/case/'draws').glob('sb*_K*.csv')))==4,case
    refs=pd.read_csv(ROOT/'análisis_desempeño/corridas_masivas/config/referencias_comparacion.csv')
    assert len(refs)==62 and not refs.duplicated(['sistema','tipo','parametro']).any()
    books=list((ROOT/'análisis_desempeño').rglob('*.ipynb'))
    assert len(books)==3
    code_cells=0
    for path in books:
        n=nbformat.read(path,as_version=4);nbformat.validate(n)
        for c in n.cells:
            if c.cell_type!='code':continue
            assert c.execution_count is not None,path
            assert not any(o.output_type=='error' for o in c.outputs),path
            code_cells+=1
    inference=nbformat.read(ROOT/'análisis_desempeño/corridas_masivas/01_corrida_masiva.ipynb',as_version=4)
    assert 'RUN_INFERENCE = False' in '\n'.join(c.source for c in inference.cells)
    for path in ROOT.rglob('*'):
        if '.git' in path.parts or '__pycache__' in path.parts: continue
        if path.is_file():
            assert path.suffix.lower() not in ['.tex','.bib','.exe','.dll','.pyc','.log'],str(path.relative_to(ROOT))
            assert path.stat().st_size<100*1024**2,str(path.relative_to(ROOT))
    for name in ['tesis','bibliografia','revision_bibliografica']:
        assert not (ROOT/name).exists(),name
    result=dict(archivos_sha256_verificados=len(manifest['archivos']),observados=36,cadenas_observadas=144,
        simulaciones=100,cadenas_simuladas=400,referencias_numericas=62,notebooks=3,celdas_ejecutadas=code_cells,
        inferencia_ejecutada=False,dependencia_de_datos_externos=False,CI_remoto_ejecutado=False)
    print(json.dumps(result,ensure_ascii=True,indent=2))
    return result

if __name__=='__main__':main()
