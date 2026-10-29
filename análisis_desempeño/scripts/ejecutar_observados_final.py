"""Completa 36 configuraciones con protocolo congelado y checkpoints separados."""
from pathlib import Path
import os,sys,json,hashlib,shutil,time,logging
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'análisis_desempeño/corridas_masivas'
OUT=BASE/'resultados/observados_final_20261029'
OLD=BASE/'resultados/fase1_revision_20261027'
sys.path.insert(0,str(ROOT))
from orbix import config,ajuste,modelos,masiva
AJ=ajuste.Ajustes(extra={'timeout_opt':60,'timeout_sample':1800})
def persist(out):
    reg=dict(out['registro']);rid=reg['run_id'];reg.pop('traceback',None)
    target=OUT/'checkpoints'/rid;target.mkdir(parents=True,exist_ok=True)
    (target/'registro.json').write_text(json.dumps(reg,indent=2,default=str),encoding='utf-8')
    for key in ['resumen','chi2']:
        if out[key] is not None:out[key].to_csv(target/f'{key}.csv',index=False)
    pd.DataFrame(out['priors']).to_csv(target/'priors.csv',index=False)
def worker(cfg):
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    rutas=ajuste.Rutas(BASE,ROOT/'modelos',OUT)
    priors=config.cargar_priors(OUT/'config/priors.csv')
    for esq in cfg.esquemas.split(','):
        rid=ajuste.run_id(cfg.sistema,cfg.tipo,esq)
        if (OUT/'checkpoints'/rid/'registro.json').exists():continue
        print('INICIO '+rid,flush=True)
        result=ajuste.correr(cfg,esq,priors,rutas,AJ)
        result['registro']['origen_ejecucion']='nueva_20261029'
        persist(result)
        print('CHECKPOINT '+rid,flush=True)
    return cfg.sistema+' '+cfg.tipo
def consolidate():
    rows=[];sums=[];chis=[];prior=[]
    for folder in sorted((OUT/'checkpoints').glob('*')):
        p=folder/'registro.json'
        if not p.exists():continue
        reg=json.loads(p.read_text(encoding='utf-8'));rows.append(reg)
        for name,store in [('resumen',sums),('chi2',chis),('priors',prior)]:
            p=folder/f'{name}.csv'
            if p.exists() and p.stat().st_size>2:
                df=pd.read_csv(p)
                for k in ['run_id','sistema','tipo','esquema']:
                    if k not in df:df[k]=reg[k]
                store.append(df)
    tables=OUT/'tablas';tables.mkdir(exist_ok=True)
    if rows:
        regs=pd.DataFrame(rows);regs.to_csv(tables/'corridas.csv',index=False)
        regs.to_csv(tables/'diagnosticos.csv',index=False)
    for name,store in [('resumen_parametros',sums),('chi2_residuos',chis),('priors_usados',prior)]:
        if store:pd.concat(store,ignore_index=True).to_csv(tables/f'{name}.csv',index=False)
    (OUT/'estado.json').write_text(json.dumps({'terminadas':len(rows),'objetivo':36,'ok':sum(r['estado']=='ok' for r in rows),'pid':os.getpid(),'actualizacion':time.time()},indent=2),encoding='utf-8')
def main():
    for d in ['config','checkpoints','draws','tablas']: (OUT/d).mkdir(parents=True,exist_ok=True)
    for name in ['sistemas.csv','priors.csv']:
        dst=OUT/'config'/name
        if not dst.exists():shutil.copy2(BASE/'config'/name,dst)
    systems=config.cargar_sistemas(OUT/'config/sistemas.csv')
    # Reusar los cuatro ajustes ya realizados con idéntica parametrización y datos.
    for _,r in pd.read_csv(OLD/'tablas/corridas.csv').iterrows():
        if (OUT/'checkpoints'/r.run_id/'registro.json').exists():continue
        src=OLD/'draws'/r.run_id;dst=OUT/'draws'/r.run_id
        shutil.copytree(src,dst,dirs_exist_ok=True)
        reg=r.to_dict();reg['origen_ejecucion']='reutilizada_fase1_20261027'
        def table(name):
            t=pd.read_csv(OLD/'tablas'/f'{name}.csv');return t[t.run_id.eq(r.run_id)].copy()
        persist(dict(registro=reg,resumen=table('resumen_parametros'),chi2=table('chi2_residuos'),priors=table('priors_usados').to_dict('records')))
    models=ROOT/'modelos'
    for kind in ['SB1','SB2']:
        for esq in modelos.ESQUEMAS:modelos.compilar(kind,esq,models,BASE/'.build')
    sources=list((models).glob('sb*_K*.stan'))+[models/'kepler_eq.hpp',ROOT/'orbix/ajuste.py',ROOT/'orbix/config.py',OUT/'config/sistemas.csv',OUT/'config/priors.csv']
    (OUT/'protocolo.json').write_text(json.dumps({'ajustes':asdict(AJ),'workers':2,'regla':'No cambiar modelos, datos, priors ni valores fijados para rescatar ajustes. Reusar cuatro corridas corregidas; completar 32. Preservar fallos y alertas.',
        'fuentes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2),encoding='utf-8')
    consolidate()
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,cfg) for _,cfg in systems[systems.activo].iterrows()]
        for f in as_completed(futures):
            print('GRUPO '+f.result(),flush=True);consolidate()
    consolidate();print('OBSERVADOS FINAL: terminado',flush=True)
if __name__=='__main__':main()
