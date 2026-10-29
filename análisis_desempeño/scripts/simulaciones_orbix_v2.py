"""Diseño revisado: 80 controles factoriales + 20 inyecciones en cadencias reales."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import time
import traceback
import numpy as np
import pandas as pd
from simulation_primitives import BASE, RAIZ, STAN, rv_independiente, wilson
from orbix import ajuste, config, fechas, modelos, periodo, resumen

DESTINO=BASE.parent/'simulaciones_nuevas'
MASTER=2026102802
TEMPLATES=['2MASS_J05215658+4359220','HIP109951','HIP20601','HIP14157']

def guardar_json(path,dato):
    def convertir(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dato,ensure_ascii=False,indent=2,default=convertir),encoding='utf-8');temp.replace(path)

def verdad_de_fila(fila,comps):
    P=float(fechas.convertir_duracion(fila.P_lit,fila.P_lit_unit,'d'))
    T=float(fechas.convertir_epocas([fila.T_lit],fila.T_lit_format,'TJD')[0])
    v=dict(tipo=fila.tipo,P=P,T=T,e=float(fila.e_lit),w_deg=float(fila.w_lit_deg),V0=float(fila.V0_lit))
    for comp,par,col in [('v1','K' if fila.tipo=='SB1' else 'K1','K_v1_lit')]+([('v2','K2','K_v2_lit')] if fila.tipo=='SB2' else []):
        valor=fila.get(col)
        v[par]=float((comps[comp].RV.max()-comps[comp].RV.min())/2 if config._es_vacio(valor) else valor)
    if fila.tipo=='SB2':v['q']=v['K1']/v['K2']
    return v

def ventana_periastro(e,nu=np.pi/4):
    E=2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(nu/2))
    return float((E-e*np.sin(E))/(2*np.pi))

def cadencia_control(comps,v,cobertura):
    """Rescala una cadencia observada a 5 P y fija exactamente 2 o 0 puntos de periastro por serie."""
    todos=np.concatenate([d.epoch.to_numpy() for d in comps.values()])
    origen=float(todos.min());base=float(np.ptp(todos))
    verdad=dict(v,T=origen+.31*v['P'])
    salida={}
    for comp,d in comps.items():
        t=origen+(d.epoch.to_numpy()-origen)/base*(5*v['P'])
        fase=(t-verdad['T'])/v['P'];ciclo=np.floor(fase);phi=np.remainder(fase,1)
        peri=ventana_periastro(v['e'])
        dentro=np.minimum(phi,1-phi)<=peri
        # Reubicar, no eliminar: conserva N, errores y duración total de cinco períodos.
        desplazamiento=(np.flatnonzero(dentro)+1)/(len(phi)+1)
        phi[dentro]=np.where(ciclo[dentro]<0,.8+.1*desplazamiento,.45+.1*desplazamiento)
        t=verdad['T']+(ciclo+phi)*v['P']
        if cobertura=='cubierto':
            orden=np.argsort(t)
            for idx,sgn in zip([orden[len(t)//3],orden[2*len(t)//3]],[-1,1]):
                n=int(np.clip(round((t[idx]-verdad['T'])/v['P']),0,4))
                t[idx]=verdad['T']+n*v['P']+sgn*ventana_periastro(v['e'],np.pi/8)*v['P']
        salida[comp]=pd.DataFrame(dict(epoch=t,err=d.err.to_numpy()))
        f=np.remainder((t-verdad['T'])/v['P'],1)
        cuenta=int(np.sum(np.minimum(f,1-f)<=peri+1e-10))
        esperado=2 if cobertura=='cubierto' else 0
        if cuenta!=esperado:raise RuntimeError(f'Control de periastro inválido: {cuenta}, esperado {esperado}')
    return verdad,salida

def preparar():
    DESTINO.mkdir(parents=True,exist_ok=True)
    if (DESTINO/'plan.json').exists():return json.loads((DESTINO/'plan.json').read_text(encoding='utf-8'))
    filas=config.cargar_sistemas(RAIZ/'config'/'sistemas.csv')
    fuentes={};tablas={};verdades={}
    for nombre in TEMPLATES:
        fila=filas[filas.sistema==nombre].iloc[0];fuentes[nombre]=fila
        comps={}
        for comp,col in [('v1','datos_v1')]+([('v2','datos_v2')] if fila.tipo=='SB2' else []):
            comps[comp]=config.cargar_componente(RAIZ/fila[col],fila.epoch_format_in,'TJD')
        tablas[nombre]=comps;verdades[nombre]=verdad_de_fila(fila,comps)
    condiciones=[]
    for e in (.25,.85):
        for cobertura in ('cubierto','ausente'):
            for tipo,ancla in [('SB1',TEMPLATES[0]),('SB2',TEMPLATES[2])]:
                condiciones.append(dict(condicion=f'{tipo}_e{int(e*100):02d}_{cobertura}',tipo=tipo,
                    ancla=ancla,e=e,cobertura=cobertura,bloque='control_factorial'))
    casos=[]
    for rep in range(10):
        especificaciones=[dict(c,repeticion=rep+1) for c in condiciones]
        if rep<5:
            especificaciones += [dict(condicion=f'OBS_{n}',tipo=fuentes[n].tipo,ancla=n,
                repeticion=rep+1,bloque='cadencia_observada') for n in TEMPLATES]
        for spec in especificaciones:
            num=len(casos)+1;nombre=f'V2_{num:03d}';carpeta=DESTINO/'casos'/nombre;carpeta.mkdir(parents=True)
            fila=fuentes[spec['ancla']];comps=tablas[spec['ancla']];v=dict(verdades[spec['ancla']])
            if spec['bloque']=='control_factorial':
                v['e']=spec['e'];v,tiempos=cadencia_control(comps,v,spec['cobertura'])
            else:tiempos={c:d[['epoch','err']].copy() for c,d in comps.items()}
            rng=np.random.default_rng([MASTER,1,num])
            archivos={};auditoria=[]
            for comp,d in tiempos.items():
                t=d.epoch.to_numpy();err=d.err.to_numpy();limpia=rv_independiente(t,v,comp)
                y=limpia+rng.normal(0,err)
                obs=pd.DataFrame(dict(epoch=t,RV=y,err=err,id_obs=np.arange(len(t))))
                obs=obs.sort_values('epoch');p=carpeta/f'datos_{comp}.csv';obs.to_csv(p,index=False);archivos[comp]=str(p.resolve())
                pd.DataFrame(dict(epoch=t,RV_sin_ruido=limpia,id_obs=np.arange(len(t)))).sort_values('epoch').to_csv(carpeta/f'verdad_{comp}.csv',index=False)
                f=np.remainder((t-v['T'])/v['P'],1);peri=ventana_periastro(v['e'])
                amp=v['K'] if v['tipo']=='SB1' else v['K1' if comp=='v1' else 'K2']
                auditoria.append(dict(componente=comp,N=len(t),base_sobre_P=float(np.ptp(t)/v['P']),
                    mediana_sigma=float(np.median(err)),K_sobre_sigma=float(amp/np.median(err)),
                    puntos_periastro_45=int(np.sum(np.minimum(f,1-f)<=peri+1e-10))))
            cfg=dict(sistema=nombre,tipo=spec['tipo'],datos_v1=archivos['v1'],datos_v2=archivos.get('v2',''),
                epoch_format_in='TJD',epoch_format_work='TJD',epoch_format_out='TJD',esquemas='free',
                P_init='',P_init_unit='d',P_lit='',P_lit_unit='d',e_lit='',
                ls_Pmin_d=float(fila.ls_Pmin_d),ls_Pmax_d=float(fila.ls_Pmax_d))
            guardar_json(carpeta/'config.json',cfg);guardar_json(carpeta/'verdad.json',v)
            metadata=dict(spec,caso=nombre,numero=num,carpeta=str(carpeta.resolve()),archivos=archivos,
                semilla_ruido=[MASTER,1,num],semillas_busqueda=[MASTER+10*num+1,MASTER+10*num+2],
                semilla_nuts=MASTER+10000+num,auditoria=auditoria,
                fuente_verdad='Escalas P/e/omega/V0/K de config vigente. HIP109951 K=(maxRV-minRV)/2, escala ilustrativa; no medición publicada.',
                fuente_tiempos=str(fila.datos_v1),fuente_errores='Errores individuales originales; en controles se preservan N y errores y se modifican las fechas.',
                hashes_datos={c:hashlib.sha256(Path(p).read_bytes()).hexdigest() for c,p in archivos.items()})
            guardar_json(carpeta/'metadata.json',metadata);casos.append(metadata)
    plan=dict(version='v2_generacion_justificada_20261028',total=100,control_factorial=80,cadencias_observadas=20,
        diseno='2 tipos x 2 excentricidades x 2 coberturas de periastro x 10 ruidos + 4 cadencias observadas x 5 ruidos',
        objetivo_primario='Comparación apareada de inicialización local/global y estabilidad ante 2 semillas; inferencia posterior como evaluación separada.',
        casos=casos,ajustes=dict(inicios_SB1=200,inicios_SB2=300,replicas_busqueda=2,cadenas=4,warmup=1000,muestras_por_cadena=1000,
            delta_lp_referencia=1e-3,timeout_opt_segundos=60,timeout_nuts_segundos=1800),
        limites='Cobertura por condición con n=10 o 5 es exploratoria. No es SBC, simulación de población ni garantía de óptimo global. No se modifican priors según la verdad.')
    guardar_json(DESTINO/'plan.json',plan)
    pd.DataFrame([{k:c[k] for k in ('caso','condicion','bloque','ancla','tipo','repeticion')} for c in casos]).to_csv(DESTINO/'diseno.csv',index=False)
    pd.DataFrame([dict(caso=c['caso'],condicion=c['condicion'],**a) for c in casos for a in c['auditoria']]).to_csv(DESTINO/'auditoria_generacion.csv',index=False)
    return plan

def params_stan(p,geo,tipo):
    x=dict(V0=float(p['V0']),log_P=float(np.log(p['P'])),
        **geo.parametros_muestreo((float(p['T'])-geo.min_t)/p['P'],np.deg2rad(p['w_deg']),p['e'],p['P'],False))
    for k in (['K'] if tipo=='SB1' else ['K1','K2']):x[k]=float(p[k])
    return x

def evaluar_punto(model,p,data,geo,tipo):
    try:
        d=model.log_prob(params=params_stan(p,geo,tipo),data=data,jacobian=False,sig_figs=18)
        g=d.iloc[0].drop(labels=['lp__'],errors='ignore').to_numpy(float)
        return dict(lp_recalculado=float(d.lp__.iloc[0]),gradiente_inf=float(np.max(np.abs(g))))
    except Exception as exc:return dict(error_evaluacion=f'{type(exc).__name__}: {exc}')

def optimizar(model,data,ini,seed,logfile):
    r={};p=None;text=''
    with TemporaryDirectory(prefix='orbix_v2_opt_') as temp:
        try:
            fit=model.optimize(data=data,inits=ini,seed=int(seed),algorithm='lbfgs',iter=4000,
                tol_obj=1e-6,tol_grad=1e-6,tol_param=1e-6,history_size=20,require_converged=False,
                show_console=False,sig_figs=18,jacobian=False,timeout=60,output_dir=temp)
            p=fit.optimized_params_dict
        except ValueError as exc:
            if "could not convert string to float: '# Optimization terminated normally:" in str(exc):
                csvs=list(Path(temp).glob('*.csv'))
                if len(csvs)==1:p=pd.read_csv(csvs[0],comment='#').iloc[-1].to_dict()
            else:r['error']=str(exc)[:300]
        except Exception as exc:r['error']=f'{type(exc).__name__}: {exc}'[:300]
        finally:
            text='\n'.join(a.read_text(encoding='utf-8',errors='replace') for a in Path(temp).glob('*.txt'))
            with gzip.open(logfile,'at',encoding='utf-8') as log:log.write(f'\n===== SEMILLA {seed} =====\n{text}\n')
    r.update(finito=bool(p is not None and np.all(np.isfinite(list(p.values())))),
        terminacion_normal='Optimization terminated normally' in text,
        terminacion_error='Optimization terminated with error' in text)
    if r['finito']:r.update(lp=float(p['lp__']),P_final=float(p['P']),e_final=float(p['e']))
    return p if r['finito'] else None,r

def generador_local(rng,comps,geo,P,data,tipo):
    """Control LS local sin literatura, con rango inicial de e igual al global."""
    todo=pd.concat(list(comps.values()))
    def gen(i):
        p=float(np.exp(np.clip(np.log(P)+rng.normal(0,.05),data['log_P_min']+1e-8,data['log_P_max']-1e-8)))
        e=rng.uniform(.001,.95);w=rng.uniform(0,2*np.pi);fase=rng.uniform(0,1)
        x=dict(V0=float(rng.uniform(todo.RV.min(),todo.RV.max())),log_P=np.log(p),**geo.parametros_muestreo(fase,w,e,p,False))
        for c,d in comps.items():
            k='K' if tipo=='SB1' else ('K1' if c=='v1' else 'K2')
            x[k]=float(np.ptp(d.RV)*.5*rng.uniform(.5,1.5)+1e-3)
        return ajuste._acotar_inicio(x,data),dict(inicio=i,origen='LS_local',P_semilla=p,e_semilla=e)
    return gen

def ejecutar_caso(caso):
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    carpeta=Path(caso['carpeta']);fin=carpeta/'resultado.json'
    if fin.exists():return str(fin)
    inicio=time.perf_counter()
    for c,p in caso['archivos'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=caso['hashes_datos'][c]:raise RuntimeError('Los datos cambiaron respecto del plan.')
    cfg=pd.Series(json.loads((carpeta/'config.json').read_text(encoding='utf-8')))
    comps={c:config.cargar_componente(Path(p),'TJD','TJD') for c,p in caso['archivos'].items()}
    geo=ajuste.Geometria(comps);aj=ajuste.Ajustes();priors=config.cargar_priors(RAIZ/'config'/'priors.csv')
    t=time.perf_counter();pinfo=config.periodo_inicial(cfg,comps['v1']);t_ls=time.perf_counter()-t
    hiper,reg_priors=config.resolver_priors(priors,cfg,'free',comps,pinfo)
    data=dict(N_v1=len(comps['v1']),t_v1=comps['v1'].epoch.values,v1_obs=comps['v1'].RV.values,v1_err=comps['v1'].err.values,**hiper)
    if cfg.tipo=='SB2':data.update(N_v2=len(comps['v2']),t_v2=comps['v2'].epoch.values,v2_obs=comps['v2'].RV.values,v2_err=comps['v2'].err.values)
    guardar_json(carpeta/'datos_stan.json',data);pd.DataFrame(reg_priors).to_csv(carpeta/'priors.csv',index=False)
    model=modelos.compilar(cfg.tipo,'free',STAN,RAIZ/'.build')
    n=aj.n_inicios_SB1 if cfg.tipo=='SB1' else aj.n_inicios_SB2
    resultados=[];mejores={};bg=None;t_global=0.
    # Alternar orden de estrategias para no atribuir al método una ventaja de orden.
    orden=['local','global'] if caso['numero']%2 else ['global','local']
    for estrategia in orden:
        if estrategia=='global':
            t=time.perf_counter()
            try:
                bg=ajuste.busqueda_global(cfg,comps,pinfo,aj)
                bg['tabla'].to_csv(carpeta/'periodograma.csv',index=False)
            except Exception as exc:
                bg=None
                guardar_json(carpeta/'error_busqueda_global.json',dict(error=str(exc),traceback=traceback.format_exc()))
            t_global=time.perf_counter()-t
        for replica,seed in enumerate(caso['semillas_busqueda'],start=1):
            etiqueta=f'{estrategia}_{replica}';checkpoint=carpeta/f'busqueda_{etiqueta}.json'
            if checkpoint.exists():
                cp=json.loads(checkpoint.read_text(encoding='utf-8'));resultados.append(cp['registro']);mejores[etiqueta]=cp['mejor'];continue
            if estrategia=='global' and bg is None:
                r=dict(caso=caso['caso'],condicion=caso['condicion'],estrategia=estrategia,replica=replica,semilla=seed,
                    n_inicios=n,n_finitos=0,n_terminacion_normal=0,segundos_opt=0.,lp_mejor=None,error='Falló el preprocesamiento global; consultar registro.')
                guardar_json(checkpoint,dict(registro=r,mejor=None));resultados.append(r);mejores[etiqueta]=None;continue
            t=time.perf_counter();rng=np.random.default_rng(seed)
            if estrategia=='local':gen=generador_local(rng,comps,geo,pinfo['P_init_work'],data,cfg.tipo)
            else:
                plan=periodo.plan_semillas(n,bg['picos'],bg['Pmin_d'],bg['Pmax_d'],bg['base_d'],aj.frac_picos)
                gen=ajuste._generador_inicios(rng,cfg.tipo,'free',comps,geo,pinfo['P_init_work'],None,data,plan=plan,soluciones=bg['soluciones'],dias_a_work=lambda x:x)
            mejor=None;bitacora=[]
            for i in range(n):
                ini,info=gen(i);p,registro=optimizar(model,data,ini,rng.integers(1,2**31-1),carpeta/f'opt_{etiqueta}.log.gz')
                bitacora.append(dict(info,**registro))
                if p is not None and (mejor is None or p['lp__']>mejor['lp__']):mejor=p
            pd.DataFrame(bitacora).to_csv(carpeta/f'inicios_{etiqueta}.csv',index=False)
            r=dict(caso=caso['caso'],condicion=caso['condicion'],estrategia=estrategia,replica=replica,semilla=seed,
                n_inicios=n,n_finitos=sum(b['finito'] for b in bitacora),n_terminacion_normal=sum(b['terminacion_normal'] for b in bitacora),
                segundos_opt=time.perf_counter()-t,lp_mejor=mejor['lp__'] if mejor else None)
            if mejor:r.update(evaluar_punto(model,mejor,data,geo,cfg.tipo))
            guardar_json(checkpoint,dict(registro=r,mejor=mejor));resultados.append(r);mejores[etiqueta]=mejor
    # La verdad se abre DESPUÉS de terminar las búsquedas; solo sirve para evaluación.
    v=json.loads((carpeta/'verdad.json').read_text(encoding='utf-8'))
    truth_support=bool(data['log_P_min']<np.log(v['P'])<data['log_P_max'] and data['V0_min']<v['V0']<data['V0_max']
        and all(0<v[k]<data[k+'_max'] for k in (['K'] if cfg.tipo=='SB1' else ['K1','K2'])))
    evaluacion_verdad=evaluar_punto(model,v,data,geo,cfg.tipo) if truth_support else {}
    referencia=max([r['lp_mejor'] for r in resultados if r['lp_mejor'] is not None]+[evaluacion_verdad.get('lp_recalculado',-np.inf)])
    for r in resultados:
        r['gap_referencia']=referencia-r['lp_mejor'] if r['lp_mejor'] is not None else None
        r['alcanza_referencia']=bool(r['gap_referencia'] is not None and r['gap_referencia']<=1e-3)
    pd.DataFrame(resultados).to_csv(carpeta/'comparacion_busquedas.csv',index=False)
    reg=dict(caso=caso['caso'],condicion=caso['condicion'],bloque=caso['bloque'],tipo=cfg.tipo,
        estado='fallo',verdad_en_soporte=truth_support,segundos_ls=t_ls,segundos_preproceso_global=t_global,
        referencia_lp=referencia,referencia_es_optimo_global_certificado=False,evaluacion_verdad=evaluacion_verdad)
    globales=[p for k,p in mejores.items() if k.startswith('global') and p is not None]
    if globales:
        best=max(globales,key=lambda p:p['lp__']);t=time.perf_counter();out=carpeta/'draws'
        if out.exists() and list(out.glob('*.csv')):
            archivo=carpeta/'interrumpidos'/f'draws_{time.time_ns()}'
            if not out.resolve().is_relative_to(DESTINO.resolve()) or not archivo.resolve().is_relative_to(DESTINO.resolve()):raise RuntimeError('Ruta de archivo fuera de la evaluación.')
            archivo.parent.mkdir(exist_ok=True);out.rename(archivo)
        out.mkdir(exist_ok=True)
        rng=np.random.default_rng(caso['semilla_nuts'])
        try:
            fit=model.sample(data=data,chains=4,parallel_chains=4,iter_warmup=1000,iter_sampling=1000,
                adapt_delta=.95,max_treedepth=15,inits=[ajuste._jitter(rng,best,cfg.tipo,'free',geo,data) for _ in range(4)],
                seed=caso['semilla_nuts'],output_dir=str(out),sig_figs=18,show_progress=False,timeout=1800)
            draws=resumen.leer_draws(out);s=resumen.resumir(draws,cfg.tipo,'TJD','TJD',.95,epoca_referencia=geo.min_t)
            diag=resumen.diagnosticar(draws,s,15,1.01);chi,resid=resumen.chi2_residuos(s,comps,cfg.tipo,'free')
            s.to_csv(carpeta/'resumen.csv',index=False);chi.to_csv(carpeta/'chi2.csv',index=False);resid.to_csv(carpeta/'residuos.csv',index=False)
            reg.update(diag,estado='ok',chi2_red=float(chi.iloc[-1].chi2_red))
            reg['diagnosticos_estrictos_ok']=bool(np.isfinite(diag['R_hat_max']) and diag['R_hat_max']<=1.01 and diag['ESS_bulk_min']>=400 and diag['ESS_tail_min']>=400 and diag['divergencias']==0 and diag['treedepth_max']==0 and diag['ebfmi_min']>=.3 and diag['soporte_ok'])
            evaluaciones=[]
            for _,p in s.iterrows():
                verdad=float(v[p.parametro])
                if p.parametro=='T':verdad+=v['P']*round((p.mediana-verdad)/v['P'])
                if p.parametro=='w_deg':verdad+=360*round((p.mediana-verdad)/360)
                evaluaciones.append(dict(caso=caso['caso'],condicion=caso['condicion'],parametro=p.parametro,
                    verdad=verdad,mediana=p.mediana,error=p.mediana-verdad,ancho=p.ancho_hpdi,
                    cubre=bool(p.hpdi_inf<=verdad<=p.hpdi_sup),diagnosticos_ok=reg['diagnosticos_estrictos_ok']))
            pd.DataFrame(evaluaciones).to_csv(carpeta/'recuperacion.csv',index=False)
        except Exception as exc:reg.update(motivo=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        reg['segundos_nuts']=time.perf_counter()-t
    else:reg['motivo']='Ninguna búsqueda global produjo un punto finito.'
    reg['segundos_totales']=time.perf_counter()-inicio
    guardar_json(fin,reg);ajuste._CACHE_PERIODOGRAMA.clear();return str(fin)

def informe(plan):
    regs=[];comparaciones=[];pars=[]
    for c in plan['casos']:
        p=Path(c['carpeta'])
        if (p/'resultado.json').exists():regs.append(json.loads((p/'resultado.json').read_text(encoding='utf-8')))
        if (p/'comparacion_busquedas.csv').exists():comparaciones.append(pd.read_csv(p/'comparacion_busquedas.csv'))
        if (p/'recuperacion.csv').exists():pars.append(pd.read_csv(p/'recuperacion.csv'))
    estado=dict(total=100,terminadas=len(regs),posteriores_terminadas=sum(r['estado']=='ok' for r in regs),
        diagnosticos_estrictos_ok=sum(r.get('diagnosticos_estrictos_ok',False) for r in regs),actualizado_unix=time.time(),version=plan['version'])
    guardar_json(DESTINO/'estado.json',estado)
    if regs:pd.DataFrame(regs).to_csv(DESTINO/'registro.csv',index=False)
    if comparaciones:pd.concat(comparaciones,ignore_index=True).to_csv(DESTINO/'comparacion_busquedas.csv',index=False)
    if pars:
        datos=pd.concat(pars,ignore_index=True);datos.to_csv(DESTINO/'recuperacion_parametros.csv',index=False)
        cov=[]
        for (cond,par),g in datos.groupby(['condicion','parametro']):
            lo,hi=wilson(int(g.cubre.sum()),len(g))
            validas=g[g.diagnosticos_ok]
            vlo,vhi=wilson(int(validas.cubre.sum()),len(validas))
            cov.append(dict(condicion=cond,parametro=par,n=len(g),cobertura=g.cubre.mean(),wilson_inf=lo,wilson_sup=hi,
                n_diagnosticos_ok=len(validas),cobertura_diagnosticos_ok=validas.cubre.mean(),wilson_validos_inf=vlo,wilson_validos_sup=vhi,
                sesgo=g.error.mean(),rmse=float(np.sqrt(np.mean(g.error**2))),ancho_medio=g.ancho.mean()))
        pd.DataFrame(cov).to_csv(DESTINO/'cobertura.csv',index=False)
    resumen_bench=[]
    if comparaciones:
        bench=pd.concat(comparaciones,ignore_index=True)
        for estrategia,g in bench.groupby('estrategia'):
            por_caso=g.groupby('caso').gap_referencia.min()
            resumen_bench.append(dict(estrategia=estrategia,busquedas=len(g),con_candidato_finito=int(g.lp_mejor.notna().sum()),
                casos=len(por_caso),casos_alcanzan_referencia=int((por_caso<=.001).sum()),mediana_gap_mejor_semilla=float(por_caso.median()),
                tiempo_opt_total_segundos=float(g.segundos_opt.sum())))
        pd.DataFrame(resumen_bench).to_csv(DESTINO/'resumen_optimizacion.csv',index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--ejecutar',action='store_true');p.add_argument('--preparar',action='store_true');p.add_argument('--informe',action='store_true');p.add_argument('--limite',type=int,default=100);p.add_argument('--trabajadores',type=int,default=2)
    a=p.parse_args();plan=preparar()
    if a.preparar or (not a.ejecutar and not a.informe):informe(plan);print('100 conjuntos v2 preparados y auditados.');return
    if a.informe:informe(plan);return
    import psutil
    bloqueo=DESTINO/'ejecucion.pid'
    if bloqueo.exists():
        pid=int(bloqueo.read_text())
        if psutil.pid_exists(pid) and 'simulaciones_orbix_v2.py' in ' '.join(psutil.Process(pid).cmdline()):raise RuntimeError('La tanda v2 ya está activa.')
    bloqueo.write_text(str(os.getpid()))
    archivos=[Path(__file__),BASE/'simulation_primitives.py',*STAN.glob('sb*_K*.stan'),STAN/'kepler_eq.hpp',*(RAIZ/'orbix').glob('*.py')]
    guardar_json(DESTINO/f'procedencia_{os.getpid()}.json',dict(python=sys.executable,inicio_unix=time.time(),fuentes_sha256={str(f.relative_to(BASE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in archivos}))
    try:
        for tipo in ('SB1','SB2'):modelos.compilar(tipo,'free',STAN,RAIZ/'.build')
        pendientes=[c for c in plan['casos'][:a.limite] if not (Path(c['carpeta'])/'resultado.json').exists()]
        informe(plan)
        with ProcessPoolExecutor(max_workers=a.trabajadores) as pool:
            futures={pool.submit(ejecutar_caso,c):c for c in pendientes}
            for f in as_completed(futures):
                c=futures[f]
                try:r=json.loads(Path(f.result()).read_text(encoding='utf-8'))
                except Exception as exc:
                    guardar_json(Path(c['carpeta'])/'error_infraestructura.json',dict(error=str(exc),traceback=traceback.format_exc()))
                    for otro in futures:otro.cancel()
                    raise
                print(c['caso'],c['condicion'],r['estado'],round(r['segundos_totales'],1),'s',flush=True)
                informe(plan)
        informe(plan);print('Tanda v2 terminada.',flush=True)
    finally:bloqueo.unlink(missing_ok=True)

if __name__=='__main__':main()
