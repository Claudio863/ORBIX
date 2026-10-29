"""Comparación bibliográfica y de incertidumbres desde cadenas existentes.

Entrada única para 02_graficos_tablas.ipynb. No importa el ejecutor de ajustes,
no compila modelos y no modifica configuraciones ni resultados de inferencia.
"""
from pathlib import Path
import hashlib, json, time
import numpy as np
import pandas as pd
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from . import resumen, fechas, rv

ROOT=Path(__file__).resolve().parents[1]
RAIZ=ROOT/'análisis_desempeño/corridas_masivas'
REFERENCIAS=RAIZ/'config/referencias_comparacion.csv'
SALIDA=RAIZ/'resultados/tablas/presentacion/benchmark'
FIG=RAIZ/'figuras/benchmark'
BLUE,ORANGE='#0072B2','#D55E00'
ORDEN=['P','T','e','w_deg','V0','K','K1','K2','q']
SIMBOLOS={'P':'P','T':'T','e':'e','w_deg':'ω','V0':'γ','K':'K','K1':'K₁','K2':'K₂','q':'q'}
FUENTE={'Videla2022':'V22','Halbwachs2016':'H16','Mendez2017':'M17','Tokovinin2018':'T18','Mennickent2015':'Me15','Rosales2021':'R21','Thompson2019':'Th19'}
ESQUEMAS=['free','fix_P','fix_e','fix_eP']


def fmt(x):return '--' if pd.isna(x) else f'{float(x):.7g}'
def celda_intervalo(center,lo,hi):
    f=lambda x:f'{float(x):.8g}'
    return r'\shortstack[l]{'+f(center)+r'\\{}['+f(lo)+', '+f(hi)+']}'
def short(s):return str(s).replace('2MASS_J05215658+4359220','2MASS J0521')

class Catalogo:
    def __init__(self):
        self.sources=set([REFERENCIAS,Path(__file__)])
        self.raw={};self.folders={};meta=[];parameters=[];diagnostics=[];chi=[]
        started=time.perf_counter()
        versions=[('historica',RAIZ/'resultados'),('fase1',RAIZ/'resultados/fase1_revision_20261027')]
        final=RAIZ/'resultados/observados_final_20261029'
        if (final/'tablas/corridas.csv').exists():versions.append(('vigente',final))
        for version,folder in versions:
            path=folder/'tablas/corridas.csv';self.sources.add(path)
            tab=pd.read_csv(path)
            if version=='vigente' and len(tab)!=36:
                raise ValueError('La revisión observada sigue incompleta; no publicar una mezcla parcial.')
            for _,m in tab.iterrows():
                original=m.run_id;rid=original+'__'+version;directory=folder/'draws'/original
                if m.estado!='ok':
                    md=m.to_dict();md.update(run_id=rid,run_id_original=original,version=version,firma_datos=None)
                    meta.append(md);diagnostics.append(md);continue
                self.folders[rid]=directory
                data={c:pd.read_csv(directory/f'datos_{c}.csv') for c in ['v1','v2'] if (directory/f'datos_{c}.csv').exists()}
                files=list(directory.glob('*.csv'))
                self.sources.update(files)
                reference=min(d.epoch.min() for d in data.values())
                d=resumen.recentrar_circulares(resumen.leer_draws(directory),reference)
                self.raw[rid]=d
                p=resumen.resumir(d,m.tipo,m.epoch_format_work,m.epoch_format_out,.95,reference)
                diag=resumen.diagnosticar(d,p,15,1.01)
                sig=hashlib.sha256(''.join(c+df[['epoch','RV','err']].to_csv(index=False) for c,df in sorted(data.items())).encode()).hexdigest()
                md=m.to_dict();md.update(run_id=rid,run_id_original=original,version=version,firma_datos=sig,**diag)
                meta.append(md)
                for key,val in [('run_id',rid),('sistema',m.sistema),('tipo',m.tipo),('esquema',m.esquema),('version',version),('firma_datos',sig)]:p[key]=val
                parameters.append(p)
                diagnostics.append(dict(run_id=rid,sistema=m.sistema,tipo=m.tipo,esquema=m.esquema,version=version,estado='ok',motivo='',**diag))
                ch,_=resumen.chi2_residuos(p,data,m.tipo,m.esquema)
                ch['run_id']=rid;ch['version']=version;chi.append(ch)
        self.corridas=pd.DataFrame(meta)
        self.parametros=pd.concat(parameters,ignore_index=True)
        self.diagnosticos=pd.DataFrame(diagnostics)
        self.chi2=pd.concat(chi,ignore_index=True)
        # Última versión disponible por sistema/tipo/esquema. La comparación
        # entre esquemas SIEMPRE conserva la versión: no mezcla fases.
        selected=self.corridas.assign(prioridad=lambda t:t.version.map({'historica':0,'fase1':1,'vigente':2})).sort_values('prioridad').drop_duplicates(['sistema','tipo','esquema'],keep='last')
        self.seleccion=selected
        self.parametros_seleccionados=self.parametros[self.parametros.run_id.isin(selected.run_id)].copy()
        self.segundos_reproceso=time.perf_counter()-started

    def meta(self,rid):return self.corridas.set_index('run_id').loc[rid]
    def draws_de(self,rid):return self.raw[rid]
    def datos_de(self,rid):
        directory=self.folders[rid]
        return {c:pd.read_csv(directory/f'datos_{c}.csv') for c in ['v1','v2'] if (directory/f'datos_{c}.csv').exists()}

def comparar(cat):
    refs=pd.read_csv(REFERENCIAS)
    assert not refs.duplicated(['sistema','tipo','parametro']).any()
    out=[]
    for _,p in cat.parametros_seleccionados.iterrows():
        g=refs[(refs.sistema==p.sistema)&(refs.tipo==p.tipo)&(refs.parametro==p.parametro)]
        if len(g)!=1:raise ValueError(f'Referencia faltante o ambigua: {p.sistema}/{p.tipo}/{p.parametro}')
        ref=g.iloc[0];unit=p.unidad_out
        v=np.array([ref.valor,ref.inferior,ref.superior],dtype=float)
        if p.parametro=='P':v=fechas.convertir_duracion(v,ref.unidad,unit)
        elif p.parametro=='T':v=fechas.convertir_epocas(v,ref.unidad,unit)
        elif p.parametro=='w_deg' and ref.unidad=='rad':v=np.rad2deg(v)
        med=float(p.mediana_out);mapv=float(p.MAP_out);lo=float(p.hpdi_inf_out);hi=float(p.hpdi_sup_out)
        shift=0
        if p.parametro=='T':
            d=cat.raw[p.run_id]
            fw=cat.meta(p.run_id).epoch_format_work
            period=float(fechas.convertir_duracion(d.P.median(),fechas.unidad_periodo(fw),fechas.unidad_periodo(unit)))
            shift=int(np.rint((v[0]-med)/period))
            vals=fechas.convertir_epocas(d['T'].to_numpy()+shift*d['P'].to_numpy(),fw,unit)
            med=float(np.median(vals));lo,hi=map(float,az.hdi(vals,hdi_prob=.95));mapv=float(vals[int(d.lp__.to_numpy().argmax())])
        if p.parametro=='w_deg':
            shift=int(np.rint((v[0]-med)/360))
            med+=shift*360;lo+=shift*360;hi+=shift*360;mapv+=shift*360
        out.append(dict(run_id=p.run_id,sistema=p.sistema,tipo=p.tipo,esquema=p.esquema,version=p.version,parametro=p.parametro,
            unidad=unit,mediana=med,MAP_muestra=mapv,hpdi_inf=lo,hpdi_sup=hi,fijo=bool(p.fijo),literatura=v[0],lit_inf=v[1],lit_sup=v[2],
            fuente=ref.fuente,tabla=ref.tabla,pagina_pdf=ref.pagina_pdf,estimador_lit=ref.estimador,intervalo_lit=ref.intervalo,
            solucion_lit=ref.solucion,estado_lit=ref.estado,nota_lit=ref.nota,desplazamiento_ciclos=shift,
            redondeo_intervalo=bool(v[1]==v[2]),punto_lit_en_HPDI=bool(lo<=v[0]<=hi) if not p.fijo and v[1]!=v[2] else None,
            nota='Comparación descriptiva con datos compartidos; sin prueba de significación independiente.'))
    return pd.DataFrame(out)

def incertidumbres(cat):
    data=cat.parametros_seleccionados.copy();out=[]
    for _,p in data.iterrows():
        free=data[(data.sistema==p.sistema)&(data.tipo==p.tipo)&(data.version==p.version)&(data.parametro==p.parametro)&data.esquema.eq('free')]
        baseline=free.iloc[0] if len(free) else None
        same=baseline is not None and baseline.firma_datos==p.firma_datos
        width=np.nan if p.fijo else float(p.ancho_hpdi_out)
        denom=float(baseline.ancho_hpdi_out) if baseline is not None else np.nan
        ratio=width/denom if same and denom>0 and not p.fijo else np.nan
        # No porcentajes respecto a ceros arbitrarios de tiempo, ángulo o RV.
        rel=100*width/abs(p.mediana_out) if p.parametro in ['P','K','K1','K2','q'] and abs(p.mediana_out)>0 else np.nan
        out.append(dict(run_id=p.run_id,sistema=p.sistema,tipo=p.tipo,esquema=p.esquema,version=p.version,parametro=p.parametro,
                        unidad=p.unidad_out,mediana=p.mediana_out,hpdi_inf=np.nan if p.fijo else p.hpdi_inf_out,
                        hpdi_sup=np.nan if p.fijo else p.hpdi_sup_out,ancho_hpdi=width,ancho_rel_porcentaje=rel,fijo=bool(p.fijo),
                        valor_condicionado=p.mediana_out if p.fijo else np.nan,ratio_a_free=ratio,datos_pareados=bool(same)))
    return pd.DataFrame(out)

def verificar_offset(cat):
    meta=cat.seleccion[cat.seleccion.sistema.eq('HD50526')&cat.seleccion.esquema.eq('free')].iloc[0]
    data=cat.datos_de(meta.run_id)['v1']
    t=fechas.convertir_epocas(data.epoch,meta.epoch_format_work,'RJD')
    rows=[]
    for tag,epoch in [('literal_tabla5',50566.392),('reconstruccion_tabla6',54566.392)]:
        p=dict(P=6.7048,T=epoch,e=.04974,w_deg=np.rad2deg(.76636),K=153.008,V0=-5.16994)
        residual=data.RV.to_numpy()-rv.curvas(p,t,'SB1')['v1']
        rows.append(dict(hipotesis=tag,T_RJD=epoch,RMS_kms=float(np.sqrt(np.mean(residual**2))),chi2=float(np.sum((residual/data.err.to_numpy())**2)),N=len(data)))
    result=pd.DataFrame(rows);result.to_csv(SALIDA/'verificacion_offset_HD50526.csv',index=False)
    return result

def save(fig,name):
    for folder in [FIG]:
        fig.savefig(folder/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(folder/f'{name}.png',bbox_inches='tight',dpi=180)
    plt.close(fig)

def plot_literatura(lit):
    outputs=[]
    for (s,kind),group in lit[lit.esquema.eq('free')].groupby(['sistema','tipo'],sort=False):
        pars=[p for p in ORDEN if p in set(group.parametro)]
        fig,axes=plt.subplots(int(np.ceil(len(pars)/2)),2,figsize=(8.2,1.85*int(np.ceil(len(pars)/2))),layout='constrained')
        for ax,par in zip(axes.flat,pars):
            p=group[group.parametro.eq(par)].iloc[0]
            ax.plot([p.hpdi_inf,p.hpdi_sup],[1,1],c=BLUE,lw=2)
            ax.scatter([p.mediana],[1],c=BLUE,s=25,zorder=3)
            ax.scatter([p.MAP_muestra],[1],c='0.4',marker='x',s=22,zorder=4)
            if not p.redondeo_intervalo:ax.plot([p.lit_inf,p.lit_sup],[0,0],c=ORANGE,lw=2)
            ax.scatter([p.literatura],[0],edgecolor=ORANGE,facecolor='none' if p.redondeo_intervalo else ORANGE,marker='D',s=25,zorder=3)
            unit={'year_besselian':'año B.','yr_B':'año B.','RJD':'día; JD−2400000','TJD':'día; JD−2450000','-':'adim.'}.get(p.unidad,p.unidad)
            ax.set(title=f'{SIMBOLOS[par]} [{unit}]',yticks=[0,1],yticklabels=[FUENTE[p.fuente],'ORBIX'],ylim=(-.65,1.65))
            ax.xaxis.set_major_locator(MaxNLocator(3));ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.tick_params(axis='x',labelsize=8)
            if p.redondeo_intervalo:
                # V22 imprime estos P/T con tres decimales. Esta banda es
                # resolución de impresión y nunca se usa como intervalo probabilístico.
                ax.axvspan(p.literatura-.0005,p.literatura+.0005,color=ORANGE,alpha=.10,zorder=0)
                ax.text(.02,.02,'Banda: resolución de impresión (±0.0005)',transform=ax.transAxes,fontsize=7)
        for ax in list(axes.flat)[len(pars):]:ax.set_visible(False)
        version=group.version.iloc[0]
        fig.suptitle(f'{short(s)} / {kind} / ajuste libre ({version})',fontsize=12)
        fig.legend(handles=[Line2D([],[],c=BLUE,marker='o',label='ORBIX: mediana y HPDI 95%'),Line2D([],[],c='0.4',ls='',marker='x',label='Muestra de mayor log posterior'),Line2D([],[],c=ORANGE,marker='D',label='Literatura: intervalo original')],loc='outside lower center',ncol=1,fontsize=9)
        name=f'literatura_{s}_{kind}';save(fig,name);outputs.append((s,kind,name))
    return outputs

def plot_incertidumbres(unc):
    outputs=[]
    for kind in ['SB1','SB2']:
        g=unc[unc.tipo.eq(kind)]
        pairs=list(g[['sistema','version']].drop_duplicates().itertuples(index=False,name=None))
        # Solo grupos con alguna comparación disponible; los libres aislados
        # permanecen en las tablas, sin dibujar cocientes sin contraparte.
        pairs=[(s,v) for s,v in pairs if len(set(g[(g.sistema==s)&(g.version==v)].esquema))>1]
        pars=[p for p in ORDEN if p in set(g.parametro)]
        fig,axes=plt.subplots(int(np.ceil(len(pars)/2)),2,figsize=(8.8,2.2*int(np.ceil(len(pars)/2))),layout='constrained')
        for ax,par in zip(axes.flat,pars):
            for y,(s,v) in enumerate(pairs):
                for scheme,color,marker,offset in [('fix_P',BLUE,'o',-.18),('fix_e',ORANGE,'s',0),('fix_eP','#009E73','^',.18)]:
                    q=g[(g.sistema==s)&(g.version==v)&(g.parametro==par)&(g.esquema==scheme)]
                    if len(q) and np.isfinite(q.ratio_a_free.iloc[0]):ax.scatter(q.ratio_a_free.iloc[0],y+offset,c=color,marker=marker,s=22)
            ax.axvline(1,c='0.4',ls='--',lw=.8)
            ax.set(title=SIMBOLOS[par],yticks=range(len(pairs)),yticklabels=[short(s)+' ['+{'fase1':'F1','historica':'H','vigente':'V'}[v]+']' for s,v in pairs],xlabel='Anchura HPDI / anchura libre')
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=8)
        for ax in list(axes.flat)[len(pars):]:ax.set_visible(False)
        fig.suptitle(f'{kind}: comparación por parámetro y dentro de cada versión',fontsize=12)
        fig.legend(handles=[Line2D([],[],c=c,marker=m,ls='',label=s) for s,c,m in [('P fijo',BLUE,'o'),('e fija',ORANGE,'s'),('P y e fijos','#009E73','^')]],loc='outside lower center',ncol=3,fontsize=9)
        name=f'incertidumbres_{kind}';save(fig,name);outputs.append(name)
    return outputs


def generar():
    for p in [SALIDA,FIG]:p.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    cat=Catalogo();lit=comparar(cat);unc=incertidumbres(cat);offset=verificar_offset(cat)
    cat.parametros.to_csv(SALIDA/'parametros_reprocesados_todas_versiones.csv',index=False)
    cat.corridas.to_csv(SALIDA/'corridas_versionadas.csv',index=False)
    cat.diagnosticos.to_csv(SALIDA/'diagnosticos_versionados.csv',index=False)
    cat.chi2.to_csv(SALIDA/'chi2_versionados.csv',index=False)
    lit.to_csv(SALIDA/'comparacion_literatura.csv',index=False)
    unc.to_csv(SALIDA/'incertidumbre_por_parametro.csv',index=False)
    outputs=plot_literatura(lit);uncfig=plot_incertidumbres(unc)
    manifest=dict(corridas=len(cat.corridas),seleccion=len(cat.seleccion),referencias_libres=len(lit[lit.esquema.eq('free')]),
        versiones=cat.corridas.version.value_counts().to_dict(),segundos_reproceso=cat.segundos_reproceso,
        fuentes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(cat.sources)},
        regla='Selección última versión por sistema/tipo/esquema; comparación de anchuras dentro de versión y datos.',
        inferencia_ejecutada=False)
    (SALIDA/'procedencia.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'{len(cat.corridas)} corridas leídas; {len(outputs)} figuras bibliográficas y {len(uncfig)} de incertidumbres. Sin ejecutar Stan.',flush=True)
    return cat,lit,unc
