"""Scientific figures from archived chains; no manuscript generation or new fits."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import benchmark as b, fechas, rv

ROOT=b.ROOT
FIG=b.RAIZ/'figuras/observados_detallados'
SCHEMES=['free','fix_P','fix_e','fix_eP']
NAMES={'free':'Libre','fix_P':'P fijo','fix_e':'e fija','fix_eP':'P y e fijos'}
ORDER=[('2MASS_J05215658+4359220','SB1'),('HD170582','SB1'),('HD50526','SB1'),('HIP109951','SB1'),('HIP89000','SB1'),('HIP117186','SB2'),('HIP14157','SB2'),('HIP20601','SB2'),('HIP89000','SB2')]
LABEL={'w_deg':r'$\omega$','V0':r'$\gamma$','K1':r'$K_1$','K2':r'$K_2$','q':r'$q$','e':r'$e$','P':r'$P$','T':r'$T$','K':r'$K$'}

def figure_save(fig,name,manifest):
    for ext in ['pdf','png']:
        p=FIG/f'{name}.{ext}'
        fig.savefig(p,dpi=160,bbox_inches='tight')
        manifest['figuras'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    plt.close(fig)

def parameters(cat,rid):
    p=cat.parametros[cat.parametros.run_id.eq(rid)]
    return p,dict(zip(p.parametro,p.MAP))

def consolidated(cat,rid,manifest):
    m=cat.meta(rid);p,point=parameters(cat,rid);d=cat.draws_de(rid);data=cat.datos_de(rid)
    names=[x for x in b.ORDEN if x in set(p.loc[~p.fijo,'parametro'])]
    rows=2+int(np.ceil(len(names)/3))
    fig=plt.figure(figsize=(10,2.05*rows+0.4),layout='constrained')
    gs=fig.add_gridspec(rows,6,height_ratios=[1.3,.7]+[1]*(rows-2))
    axes=[fig.add_subplot(gs[i,j*3:(j+1)*3]) for i in range(2) for j in range(2)]
    ph=np.linspace(0,1,700)
    times=np.concatenate([x.epoch.values for x in data.values()])
    grid=np.linspace(times.min(),times.max(),4000)
    for c,color,marker in [('v1','#0072B2','o'),('v2','#D55E00','^')]:
        if c not in data:continue
        x=data[c];phase=rv.fase(x.epoch.values,point['P'],point['T'])
        outtime=fechas.convertir_epocas(x.epoch.values,m.epoch_format_work,m.epoch_format_out)
        y=rv.curvas(point,x.epoch.values,m.tipo)[c];z=(x.RV.values-y)/x.err.values
        axes[0].errorbar(phase,x.RV,yerr=x.err,fmt=marker,ms=3,c=color,lw=.6,label=c)
        axes[0].plot(ph,rv.curvas(point,point['T']+ph*point['P'],m.tipo)[c],c=color,lw=1)
        axes[1].errorbar(outtime,x.RV,yerr=x.err,fmt=marker,ms=3,c=color,lw=.6)
        axes[1].plot(fechas.convertir_epocas(grid,m.epoch_format_work,m.epoch_format_out),rv.curvas(point,grid,m.tipo)[c],c=color,lw=.8)
        axes[2].scatter(phase,z,c=color,marker=marker,s=12)
        axes[3].scatter(outtime,z,c=color,marker=marker,s=12)
    axes[0].set(ylabel='RV (km/s)',title='Órbita en fase');axes[0].legend(fontsize=8)
    axes[1].set(title='Órbita en el tiempo')
    axes[2].set(xlabel='Fase de periastro',ylabel='Residuo / σ')
    axes[3].set(xlabel='Época ('+str(m.epoch_format_out)+')')
    for ax in axes[2:]:ax.axhline(0,c='0.4',lw=.7);ax.axhspan(-3,3,color='0.8',alpha=.25)
    for i,k in enumerate(names):
        ax=fig.add_subplot(gs[2+i//3,(i%3)*2:(i%3+1)*2]);v=d[k].to_numpy()
        q=p[p.parametro.eq(k)].iloc[0]
        if k in ['P','T']:
            v=fechas.convertir_parametro(v,k,m.epoch_format_work,m.epoch_format_out)
        offset=np.floor(q.mediana_out) if k=='T' else 0
        v=v-offset
        ax.hist(v,bins=35,color='#0072B2',alpha=.7,density=True)
        ax.axvspan(q.hpdi_inf_out-offset,q.hpdi_sup_out-offset,color='#D55E00',alpha=.18)
        ax.axvline(q.mediana_out-offset,c='#D55E00',lw=1)
        label=LABEL[k]+(f' − {offset:g}' if offset else '')
        ax.set(xlabel=label+' ['+str(q.unidad_out)+']',ylabel='Densidad')
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.locator_params(axis='x',nbins=4);ax.tick_params(labelsize=8)
    fig.suptitle(f'{b.short(m.sistema)} · {m.tipo} · {NAMES[m.esquema]}',fontsize=13)
    figure_save(fig,f'{m.run_id_original}_consolidado',manifest)

def correlations(cat,rid,manifest):
    m=cat.meta(rid);d=cat.draws_de(rid)
    names=['P','e','K1','K2'] if m.tipo=='SB2' else ['P','e','w_deg','K']
    fig,ax=plt.subplots(4,4,figsize=(9,9),layout='constrained')
    sample=d.iloc[np.linspace(0,len(d)-1,min(1500,len(d)),dtype=int)]
    for i,y in enumerate(names):
        for j,x in enumerate(names):
            a=ax[i,j]
            if i<j:a.set_visible(False);continue
            if i==j:a.hist(sample[x],bins=32,color='#0072B2',density=True)
            else:a.scatter(sample[x],sample[y],s=2,alpha=.15,c='#0072B2',rasterized=True)
            if i==3:a.set_xlabel(LABEL[x])
            else:a.tick_params(labelbottom=False)
            if j==0 and i!=j:a.set_ylabel(LABEL[y])
            elif i!=j:a.tick_params(labelleft=False)
            a.tick_params(labelsize=8);a.locator_params(nbins=3)
    fig.suptitle(f'{b.short(m.sistema)} · {m.tipo} · libre: correlaciones seleccionadas',fontsize=12)
    figure_save(fig,f'{m.run_id_original}_pares',manifest)

def traces(cat,rid,manifest):
    m=cat.meta(rid);d=cat.draws_de(rid)
    names=['P','w_deg','V0','K']
    fig,ax=plt.subplots(4,1,figsize=(10,8),sharex=True,layout='constrained')
    for a,k in zip(ax,names):
        for chain,g in d.groupby('chain'):
            a.plot(np.arange(len(g))+1,g[k].values,lw=.5,alpha=.8,label=f'Cadena {int(chain)}')
            if 'divergent__' in g:
                mask=g.divergent__.eq(1).values
                a.scatter((np.arange(len(g))+1)[mask],g[k].values[mask],marker='x',s=25,c='black',zorder=5)
        a.set_ylabel(LABEL[k])
    ax[0].legend(ncol=4,fontsize=8);ax[-1].set_xlabel('Iteración retenida (adaptación excluida)')
    fig.suptitle(f'HIP109951 · {NAMES[m.esquema]}: trazas y divergencias (×)',fontsize=12)
    figure_save(fig,f'{m.run_id_original}_trazas',manifest)


def generar(cat):
    FIG.mkdir(exist_ok=True,parents=True)
    manifest={'inferencia_ejecutada':False,'corridas':{},'figuras':{}}
    for m in cat.seleccion.itertuples():
        folder=cat.folders[m.run_id]
        manifest['corridas'][m.run_id_original]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('sb*_K*.csv'))}
        consolidated(cat,m.run_id,manifest)
        if m.esquema=='free': correlations(cat,m.run_id,manifest)
        if m.sistema=='HIP109951' and m.esquema in ['free','fix_e']: traces(cat,m.run_id,manifest)
    (FIG/'procedencia.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return manifest
