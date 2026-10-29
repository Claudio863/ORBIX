"""Auditoría causal acotada: solo lee las cien simulaciones ya realizadas."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'análisis_desempeño/simulaciones'
OUT=ROOT/'análisis_desempeño/resultados_validacion';OUT.mkdir(exist_ok=True,parents=True)
FIG=ROOT/'análisis_desempeño/figuras/auditoria';FIG.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(ROOT))
from orbix import resumen
sources={Path(__file__)}
def read(p):sources.add(p);return pd.read_csv(p)
def js(p):sources.add(p);return json.loads(p.read_text(encoding='utf-8'))
def flag(v):return str(v).lower()=='true'
reg=read(BASE/'registro.csv');opt=read(BASE/'comparacion_busquedas.csv')
rows=[];alerts=[];details=[];optimizers=[]
for r in reg.itertuples():
    folder=BASE/'casos'/r.caso;t=js(folder/'verdad.json');data=js(folder/'datos_stan.json')
    for strategy in ['local','global']:
        candidates=[js(folder/f'busqueda_{strategy}_{i}.json') for i in [1,2]]
        candidate=max(candidates,key=lambda x:x['registro']['lp_mejor'] if x['registro'].get('lp_mejor') is not None else -np.inf)
        best=candidate.get('mejor') or {};record=candidate['registro']
        optimizers.append(dict(caso=r.caso,estrategia=strategy,codigo_terminacion=best.get('converged__'),
            gradiente_inf=record.get('gradiente_inf'),lp_mejor=record.get('lp_mejor'),
            terminacion_normal=best.get('converged__') in [10,20,21,30,31]))
    for comp,par in [('v1','K' if t['tipo']=='SB1' else 'K1')]+([('v2','K2')] if t['tipo']=='SB2' else []):
        d=read(folder/f'datos_{comp}.csv');latent=read(folder/f'verdad_{comp}.csv')
        rng=float(d.RV.max()-d.RV.min());latent_range=float(latent.RV_sin_ruido.max()-latent.RV_sin_ruido.min())
        M=np.remainder(2*np.pi*(d.epoch.to_numpy()-t['T'])/t['P'],2*np.pi)
        E=np.array([brentq(lambda x:x-t['e']*np.sin(x)-m,0,2*np.pi) for m in M])
        nu=2*np.arctan2(np.sqrt(1+t['e'])*np.sin(E/2),np.sqrt(1-t['e'])*np.cos(E/2))
        rows.append(dict(caso=r.caso,condicion=r.condicion,componente=comp,parametro=par,verdad=t[par],
            cota=data[par+'_max'],rango_RV=rng,rango_sin_ruido=latent_range,cota_sin_ruido=max(2*latent_range,1),
            razon_cota_verdad=data[par+'_max']/t[par],fuera=not 0<t[par]<data[par+'_max'],
            N=len(d),base_sobre_P=float(np.ptp(d.epoch)/t['P']),N_periastro=int((np.abs(nu)<=np.pi/4+1e-8).sum())))
    if not flag(r.diagnosticos_estrictos_ok):
        p=read(folder/'resumen.csv');bad=p[(p.R_hat>1.01)|(p.ESS_bulk<400)|(p.ESS_tail<400)]
        for q in bad.itertuples():details.append(dict(caso=r.caso,parametro=q.parametro,R_hat=q.R_hat,ESS_bulk=q.ESS_bulk,ESS_tail=q.ESS_tail))
        chain_files=sorted((folder/'draws').glob('*.csv'));sources.update(chain_files)
        draws=resumen.recentrar_circulares(resumen.leer_draws(folder/'draws'),min(data['t_v1']+data.get('t_v2',[])))
        kp='K' if t['tipo']=='SB1' else 'K1'
        corr=float(draws[['e',kp]].corr().iloc[0,1])
        best=opt[(opt.caso==r.caso)&(opt.estrategia=='global')].gap_referencia.min()
        flags=[]
        for name,value in [('Rhat',r.R_hat_max>1.01),('ESSbulk',r.ESS_bulk_min<400),('ESStail',r.ESS_tail_min<400),('divergencias',r.divergencias>0),('profundidad',r.treedepth_max>0),('EBFMI',r.ebfmi_min<.3)]:
            if value:flags.append(name)
        alerts.append(dict(caso=r.caso,condicion=r.condicion,alertas=';'.join(flags),Rhat=r.R_hat_max,
            ESSbulk=r.ESS_bulk_min,ESStail=r.ESS_tail_min,divergencias=r.divergencias,
            correlacion_e_K=corr,gap_global=best,verdad_en_soporte=flag(r.verdad_en_soporte),
            parametros_alerta=';'.join(bad.parametro)))
support=pd.DataFrame(rows);a=pd.DataFrame(alerts);d=pd.DataFrame(details)
optimizers=pd.DataFrame(optimizers);optimizers.to_csv(OUT/'terminacion_optimizadores.csv',index=False)
support.to_csv(OUT/'soportes_y_cobertura.csv',index=False);a.to_csv(OUT/'trece_alertas.csv',index=False)
d.to_csv(OUT/'diagnosticos_por_parametro.csv',index=False)
excluded=support[support.fuera]
assert len(excluded)==10 and set(excluded.parametro)=={'K2'} and len(a)==13
counts={name:int(a.alertas.str.contains(name,regex=False).sum()) for name in ['Rhat','ESSbulk','ESStail','divergencias','profundidad','EBFMI']}
print(json.dumps({'excluidos':len(excluded),'alertas':counts,'condiciones':a.condicion.value_counts().to_dict(),
    'K2max_rango':[excluded.cota.min(),excluded.cota.max()], 'sin_ruido_excluidos':int((excluded.cota_sin_ruido<excluded.verdad).sum())},indent=2))
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
for comp,color in [('K1','#0072B2'),('K2','#D55E00')]:
    g=support[support.condicion.eq('SB2_e85_ausente')&support.parametro.eq(comp)]
    axes[0].scatter(range(1,len(g)+1),g.razon_cota_verdad,label=comp,c=color)
axes[0].axhline(1,c='0.3',ls='--');axes[0].set(xlabel='Realización de ruido',ylabel='Cota superior / amplitud verdadera',title='SB2 excéntrico sin periastro');axes[0].legend()
for condition,color in [('SB1_e85_ausente','#0072B2'),('OBS_HIP109951','#D55E00'),('OBS_HIP20601','#009E73')]:
    g=a[a.condicion.eq(condition)];axes[1].scatter(g.ESSbulk,g.Rhat,label=condition,c=color)
axes[1].axvline(400,c='0.3',ls='--');axes[1].axhline(1.01,c='0.3',ls='--')
axes[1].set(xlabel='ESS bulk mínimo',ylabel='R-hat máximo',title='Trece alertas: diagnósticos distintos');axes[1].legend(fontsize=8)
for ext in ['pdf','png']:fig.savefig(FIG/f'auditoria_simulada.{ext}',bbox_inches='tight',dpi=180)
plt.close(fig)
(OUT/'procedencia.json').write_text(json.dumps({'inferencia_ejecutada':False,'archivos':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}},indent=2),encoding='utf-8')
