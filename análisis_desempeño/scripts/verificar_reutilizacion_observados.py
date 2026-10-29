"""Comprueba las entradas de las cuatro corridas reutilizadas, sin ajustarlas."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'análisis_desempeño/corridas_masivas'
OUT=BASE/'resultados/observados_final_20261029'
sys.path.insert(0,str(ROOT))
from orbix import config
systems=config.cargar_sistemas(OUT/'config/sistemas.csv')
priors=config.cargar_priors(OUT/'config/priors.csv')
old=pd.read_csv(BASE/'resultados/fase1_revision_20261027/tablas/corridas.csv')
records=[]
for _,m in old.iterrows():
    cfg=systems[systems.sistema.eq(m.sistema)&systems.tipo.eq(m.tipo)].iloc[0]
    comps={c:config.cargar_componente(BASE/cfg['datos_'+c],cfg.epoch_format_in,cfg.epoch_format_work)
        for c in ['v1','v2'] if pd.notna(cfg.get('datos_'+c)) and str(cfg.get('datos_'+c)).strip()}
    pi={k:m[k] for k in ['P_init_work','ls_Pmin_d','ls_Pmax_d']}
    hyper,_=config.resolver_priors(priors,cfg,m.esquema,comps,pi)
    data=dict(hyper)
    for c,d in comps.items():data.update({f'N_{c}':len(d),f't_{c}':d.epoch.to_numpy(),f'{c}_obs':d.RV.to_numpy(),f'{c}_err':d.err.to_numpy()})
    p=OUT/'draws'/m.run_id/'datos_stan.json';saved=json.loads(p.read_text(encoding='utf-8'))
    assert set(data)==set(saved),(m.run_id,'claves distintas')
    for k in data:np.testing.assert_allclose(data[k],saved[k],rtol=1e-12,atol=1e-10,err_msg=m.run_id+'/'+k)
    assert m.modelo_version=='fase_acotada_tref_ponderado_20261027'
    records.append(dict(run_id=m.run_id,entradas_coinciden=True,modelo_version=m.modelo_version,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(OUT/'verificacion_reutilizacion.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Cuatro corridas reutilizadas: datos, hiperparámetros, unidades y versión coinciden.')
