"""Standard four-panel reports from archived chains.

The marginal histogram/KDE, red point estimate, black HPDI and lower-triangle
histogram layout follows app/src/stan_analysis.py. The Keplerian curves use
orbix.rv and the current joint samples. No inference or manuscript export.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import rv,fechas,benchmark as b
OUT=b.RAIZ/'figuras/informe_estandar'
NAMES={'free':'Libre','fix_P':'P fijo','fix_e':'e fija','fix_eP':'P y e fijos'}
SYMBOL={'P':'P','T':'T','e':'e','w_deg':r'$\omega$','V0':r'$\gamma$','K':'K','K1':r'$K_1$','K2':r'$K_2$','q':'q'}

def guardar(fig,path):
    for ext in ['pdf','png']:fig.savefig(path.with_suffix('.'+ext),bbox_inches='tight',dpi=180)
    plt.close(fig)

def corrida(cat,rid):
    m=cat.meta(rid);table=cat.parametros[cat.parametros.run_id.eq(rid)].set_index('parametro')
    d=cat.draws_de(rid);data=cat.datos_de(rid)
    point=table.MAP.to_dict()
    names=[n for n in b.ORDEN if n in table.index and not bool(table.loc[n,'fijo'])]
    origin=float(np.floor(d['T'].median()))
    units={'yr_B':'año B.','year_besselian':'año B.','d':'d','deg':'°','km/s':'km/s','-':''}
    labels={n:SYMBOL[n]+(f' − {origin:g}' if n=='T' else '')+(' ['+units.get(str(table.loc[n,'unidad']),str(table.loc[n,'unidad']))+']' if n not in ['e','q'] else '') for n in names}
    shown=pd.DataFrame({labels[n]:d[n].values-(origin if n=='T' else 0) for n in names})
    prefix=OUT/m.run_id_original
    rows=int(np.ceil(len(names)/3))
    with plt.rc_context({'font.size':15,'axes.spines.top':True,'axes.spines.right':True,'axes.titlesize':15,'pdf.fonttype':42}):
        fig,axs=plt.subplots(rows,3,figsize=(13,3.1*rows),layout='constrained',squeeze=False)
        for ax,n in zip(axs.flat,names):
            x=shown[labels[n]].values;q=table.loc[n];shift=origin if n=='T' else 0
            sns.histplot(x=x,stat='density',kde=True,color='tab:blue',alpha=.4,edgecolor=None,ax=ax)
            ax.axvline(q.MAP-shift,c='red',lw=1)
            ymax=ax.get_ylim()[1];ax.hlines(.07*ymax,q.hpdi_inf-shift,q.hpdi_sup-shift,color='black',lw=3)
            ax.set_xlabel(labels[n]);ax.set_ylabel('');ax.set_yticks([])
            ax.ticklabel_format(style='plain',useOffset=False,axis='x');ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.tick_params(labelsize=12)
        for ax in list(axs.flat)[len(names):]:ax.set_visible(False)
        guardar(fig,Path(str(prefix)+'_marginales'))
        pares(d,table,names,prefix)
        phase=np.linspace(0,1,1500)
        alltime=np.concatenate([x.epoch.values for x in data.values()])
        tmin,tmax=alltime.min(),alltime.max()
        # At least 150 samples per orbit, capped for long observing spans.
        ngrid=min(100000,max(2000,int((tmax-tmin)/point['P']*150)))
        grid=np.unique(np.r_[np.linspace(tmin,tmax,ngrid),alltime])
        for mode in ['fase','tiempo']:
            fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
            for comp,color,marker in [('v1','tab:blue','o'),('v2','tab:red','^')]:
                if comp not in data:continue
                x=data[comp]
                xx=rv.fase(x.epoch.values,point['P'],point['T']) if mode=='fase' else x.epoch.values
                tt=point['T']+phase*point['P'] if mode=='fase' else grid
                gx=phase if mode=='fase' else grid
                ax.errorbar(xx,x.RV,yerr=x.err,fmt=marker,ms=4,c='black' if m.tipo=='SB1' else color,lw=.8,label='Observaciones' if m.tipo=='SB1' else comp)
                ax.plot(gx,rv.curvas(point,tt,m.tipo)[comp],c=color,lw=1.2)
            ax.set_xlabel('Fase orbital' if mode=='fase' else 'Época ['+str(m.epoch_format_work)+']')
            ax.set_ylabel('Velocidad radial (km/s)');ax.legend(fontsize=11);ax.tick_params(labelsize=13)
            guardar(fig,Path(str(prefix)+'_'+mode))
    sources={str(p.relative_to(b.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(cat.folders[rid].glob('*.csv'))}
    return dict(run_id=m.run_id_original,version=m.version,esquema=m.esquema,origen_T=origin,parametros=names,archivos_fuente=sources,estimador_curva='Muestra conjunta de mayor lp__ retenido; no máximo continuo certificado',intervalo='HPDI marginal 95%',inferencia_ejecutada=False)

def generar(cat):
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for m in cat.seleccion.itertuples():
        print('Formato estándar:',m.run_id_original,flush=True)
        records.append(corrida(cat,m.run_id))
    sources=[Path(__file__),b.ROOT/'app/src/stan_analysis.py',b.ROOT/'orbix/rv.py']
    result=dict(corridas=records,inferencia_ejecutada=False,formato='Marginales y correlaciones de la aplicación; RV en fase y tiempo',implementacion={str(p.relative_to(b.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'procedencia.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result


def generar_residuos_desde_tablas():
    """Nine current free-fit RV/residual panels, from archived summary points."""
    dest=b.RAIZ/'figuras/observados_libres';dest.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(b.SALIDA/'parametros_reprocesados_todas_versiones.csv')
    meta=pd.read_csv(b.RAIZ/'resultados/observados_final_20261029/tablas/corridas.csv')
    for r in meta[meta.esquema.eq('free')].itertuples():
        p=table[table.run_id.eq(r.run_id+'__vigente')].set_index('parametro').MAP.to_dict()
        folder=b.RAIZ/'resultados/observados_final_20261029/draws'/r.run_id
        data={c:pd.read_csv(folder/f'datos_{c}.csv') for c in ['v1','v2'] if (folder/f'datos_{c}.csv').exists()}
        phase=np.linspace(0,1,1500)
        with plt.rc_context({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
            fig,ax=plt.subplots(2,1,figsize=(8.6,5.8),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
            for c,color in [('v1','#0072B2'),('v2','#D55E00')]:
                if c not in data:continue
                d=data[c];f=rv.fase(d.epoch.values,p['P'],p['T'])
                pred=rv.curvas(p,d.epoch.values,r.tipo)[c]
                ax[0].errorbar(f,d.RV,yerr=d.err,fmt='o',ms=3,c=color,lw=.7,label=c)
                ax[0].plot(phase,rv.curvas(p,p['T']+phase*p['P'],r.tipo)[c],c=color,lw=1)
                ax[1].scatter(f,(d.RV-pred)/d.err,s=12,c=color)
            ax[0].legend();ax[1].axhline(0,c='0.4',lw=.8);ax[1].axhspan(-3,3,color='0.5',alpha=.1)
            ax[0].set_ylabel('RV (km/s)');ax[1].set_ylabel('Residuo / error');ax[1].set_xlabel('Fase orbital')
            guardar(fig,dest/r.run_id)
    return dest


def pares(d,table,names,prefix):
    """Affine rescaling preserves pair dependence and keeps short readable ticks."""
    from matplotlib.ticker import FuncFormatter
    cols=[n for n in names if n!='q']
    shown=pd.DataFrame({SYMBOL[n]:(d[n].values-float(table.loc[n,'mediana']))/float(table.loc[n,'hpdi_sup']-table.loc[n,'hpdi_inf']) for n in cols})
    pair=sns.pairplot(shown,kind='hist',corner=True,height=1.8,diag_kws={'color':'tab:blue'},plot_kws={'color':'tab:blue','bins':24})
    for ax in pair.axes.flat:
        if ax is None:continue
        ax.tick_params(labelsize=23);ax.xaxis.label.set_size(26);ax.yaxis.label.set_size(26)
        ax.xaxis.set_major_locator(plt.MaxNLocator(2));ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.1g}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.1g}'))
        for coll in ax.collections:coll.set_rasterized(True)
    guardar(pair.fig,Path(str(prefix)+'_pares'))

def actualizar_pares_archivados():
    from . import resumen
    table=pd.read_csv(b.SALIDA/'parametros_reprocesados_todas_versiones.csv')
    meta=pd.read_csv(b.RAIZ/'resultados/observados_final_20261029/tablas/corridas.csv')
    for r in meta.itertuples():
        folder=b.RAIZ/'resultados/observados_final_20261029/draws'/r.run_id
        data=[pd.read_csv(p) for p in folder.glob('datos_v*.csv')]
        origin=min(x.epoch.min() for x in data)
        d=resumen.recentrar_circulares(resumen.leer_draws(folder),origin)
        t=table[table.run_id.eq(r.run_id+'__vigente')].set_index('parametro')
        names=[n for n in b.ORDEN if n in t.index and not bool(t.loc[n,'fijo'])]
        pares(d,t,names,OUT/r.run_id)
    path=OUT/'procedencia.json';m=json.loads(path.read_text(encoding='utf-8'))
    m['ejes_pares']='(theta - mediana) / ancho HPDI 95%; transformación afín, q derivado omitido del diagrama de pares'
    m['implementacion']['orbix/informe_estandar.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')
