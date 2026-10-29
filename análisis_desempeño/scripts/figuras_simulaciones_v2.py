"""Figuras de los resultados conservados v2; no ejecuta optimización ni Stan."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import brentq
from analizar_simulaciones_v2 import wilson

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'análisis_desempeño/simulaciones'
OUT = ROOT / 'análisis_desempeño/figuras/simulaciones'
BLUE, RED, ORANGE, PURPLE = '#0072B2', '#D55E00', '#E69F00', '#8B4C96'
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
                     'legend.fontsize': 9, 'pdf.fonttype': 42, 'savefig.dpi': 180,
                     'axes.spines.top': False, 'axes.spines.right': False})
SOURCES = set()

def csv(path):
    SOURCES.add(path)
    return pd.read_csv(path)

def js(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding='utf-8'))

def boolean(s):
    v = s.astype(str).str.lower()
    if not v.isin(['true', 'false']).all():
        raise ValueError('Flag booleano no reconocido')
    return v.eq('true')

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight')
    plt.close(fig)

def label(c):
    if c.startswith('OBS_'):
        return 'Obs. ' + c[4:].replace('2MASS_J05215658+4359220', '2MASS')
    kind, e, peri = c.split('_')
    return f'{kind} / e=0.{e[1:]} / ' + ('con peri.' if peri == 'cubierto' else 'sin peri.')

def rv(t, p, comp):
    m = np.remainder(2*np.pi*(np.asarray(t)-p['T'])/p['P'], 2*np.pi)
    e = p['e']
    E = np.array([brentq(lambda x: x-e*np.sin(x)-a, 0, 2*np.pi, xtol=1e-12) for a in m])
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    k = p.get('K', p.get('K1' if comp == 'v1' else 'K2'))
    return p['V0'] + (1 if comp == 'v1' else -1)*k*(np.cos(np.deg2rad(p['w_deg'])+nu)+e*np.cos(np.deg2rad(p['w_deg'])))

def best_params(case, strategy):
    items = [js(DATA / 'casos' / case / f'busqueda_{strategy}_{i}.json')['mejor'] for i in [1, 2]]
    return max((p for p in items if p is not None), key=lambda p: p['lp__'])

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    SOURCES.add(Path(__file__).resolve())
    reg = csv(DATA / 'registro.csv').sort_values('caso')
    opt = csv(DATA / 'comparacion_busquedas.csv')
    rec = csv(DATA / 'recuperacion_parametros.csv')
    cov = csv(DATA / 'cobertura_filtrada_soporte_diagnosticos.csv')
    for col in ['diagnosticos_estrictos_ok', 'verdad_en_soporte']:
        reg[col] = boolean(reg[col])
    conditions = list(reg.condicion.drop_duplicates())
    names = [label(c) for c in conditions]
    best = opt.sort_values('lp_mejor', ascending=False).groupby(['caso', 'estrategia'], as_index=False).first()
    gap = best.pivot(index='caso', columns='estrategia', values='gap_referencia').clip(lower=0)
    timings = opt.groupby(['caso', 'estrategia']).segundos_opt.sum().unstack().reindex(reg.caso)
    if len(reg) != 100 or len(best) != 200 or len(opt) != 400:
        raise ValueError('Cantidad de casos o búsquedas inesperada')

    # 1. Cadencia: primer caso de cada condición de control, todas sus fechas.
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6), sharex=True, sharey=True, layout='constrained')
    for ax, kind, eccentricity in zip(axes.flat, ['SB1', 'SB1', 'SB2', 'SB2'], [25, 85, 25, 85]):
        for row, peri in enumerate(['ausente', 'cubierto']):
            case = reg[reg.condicion.eq(f'{kind}_e{eccentricity}_{peri}')].iloc[0].caso
            truth = js(DATA / 'casos' / case / 'verdad.json')
            for comp, color, marker, offset in [('v1', BLUE, 'o', -.07), ('v2', RED, '^', .07)]:
                path = DATA / 'casos' / case / f'datos_{comp}.csv'
                if not path.exists(): continue
                d = csv(path)
                phase = np.remainder((d.epoch-truth['T'])/truth['P']+.5, 1)-.5
                ax.scatter(phase, np.full(len(d), row+offset), c=color, marker=marker, s=24, alpha=.8)
            e = truth['e']
            E = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(np.pi/8))
            width = (E-e*np.sin(E))/(2*np.pi)
        ax.axvspan(-width, width, color=ORANGE, alpha=.18)
        ax.axvline(0, color='0.5', lw=.6)
        ax.set(title=f'{kind}, e = {eccentricity/100:.2f}', xlim=(-.51, .51), ylim=(-.3, 1.3),
               yticks=[0, 1], yticklabels=['Sin periastro', 'Con periastro'])
        ax.grid(axis='x', alpha=.2)
    for ax in axes[-1]: ax.set_xlabel('Fase respecto al periastro verdadero')
    fig.legend(handles=[Line2D([], [], color=BLUE, marker='o', ls='', label='Primaria'),
                        Line2D([], [], color=RED, marker='^', ls='', label='Secundaria')],
               loc='outside upper center', ncol=2)
    save(fig, '01_cadencia')

    # 2. Éxito por condición y gaps individuales (mejor de dos semillas).
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), layout='constrained', gridspec_kw={'width_ratios':[1.35, 1]})
    summary = []
    for strategy, color, offset in [('local', RED, -.12), ('global', BLUE, .12)]:
        for y, cond in enumerate(conditions):
            g = best[(best.estrategia == strategy) & (best.condicion == cond)]
            n, k = len(g), int((g.gap_referencia <= .001).sum())
            lo, hi = wilson(k, n)
            axes[0].errorbar(k/n, y+offset, xerr=[[max(0,k/n-lo)], [max(0,hi-k/n)]], fmt='o', color=color, ms=4, capsize=2)
            axes[0].text(1.04 if strategy == 'global' else 1.22, y+offset, f'{k}/{n}', color=color, va='center', fontsize=7)
            summary.append(dict(condicion=cond, estrategia=strategy, n=n, k=k, wilson_lo=lo, wilson_hi=hi))
    axes[0].set(yticks=range(len(names)), yticklabels=names, xlim=(-.04, 1.4), xticks=[0,.25,.5,.75,1], xlabel='Fracción que alcanza la referencia', title='a) Éxito por condición; intervalos Wilson 95%')
    axes[0].invert_yaxis()
    for y, case in enumerate(reg.caso):
        a, b = gap.loc[case, ['local', 'global']]
        axes[1].plot([max(a,1e-10), max(b,1e-10)], [y,y], color='0.83', lw=.65)
    for strategy, color in [('local',RED),('global',BLUE)]:
        axes[1].scatter(np.maximum(gap.reindex(reg.caso)[strategy], 1e-10), np.arange(100), s=9, color=color, label=strategy.capitalize())
    axes[1].axvline(.001, color='0.35', ls='--', lw=.8)
    axes[1].set(xscale='log', xlabel='Déficit de log densidad respecto a la referencia', ylabel='Caso (orden del registro)', title='b) Mejor candidato de dos semillas')
    axes[1].legend(loc='upper left')
    axes[1].text(.02,.02,'Déficits < 10⁻¹⁰ se dibujan en 10⁻¹⁰', transform=axes[1].transAxes, fontsize=7)
    save(fig, '02_optimizacion')
    pd.DataFrame(summary).to_csv(OUT / 'exito_por_condicion.csv', index=False)

    # 3. Errores de medianas posteriores; no confundir con candidatos L-BFGS.
    rec = rec.merge(reg[['caso','diagnosticos_estrictos_ok','verdad_en_soporte']], on='caso', validate='many_to_one')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), layout='constrained')
    for ax, par in zip(axes.flat, ['P','e','K','K1','K2','q']):
        sub = rec[rec.parametro.eq(par)].copy()
        sub['plot_error'] = sub.error if par == 'e' else 100*sub.error/sub.verdad
        for i, cond in enumerate(conditions):
            group = sub[sub.condicion.eq(cond)].sort_values('caso')
            x = i + np.linspace(-.16, .16, len(group))
            status = np.where(~group.verdad_en_soporte, 2, np.where(~group.diagnosticos_estrictos_ok, 1, 0))
            for code, color, marker in [(0,BLUE,'o'),(1,ORANGE,'x'),(2,PURPLE,'^')]:
                keep = status == code
                ax.scatter(x[keep], group.plot_error.to_numpy()[keep], color=color, marker=marker, s=23, alpha=.85)
        ax.axhline(0, color='0.35', lw=.8)
        if par == 'e':
            ax.yaxis.set_major_locator(MaxNLocator(5))
        else:
            ax.set_yscale('symlog', linthresh=.1)
        ax.set(title=f'{par}: n = {len(sub)}', xticks=range(len(names)), xticklabels=[str(i+1) for i in range(len(names))],
               ylabel='Diferencia de e' if par == 'e' else 'Error relativo (%)')
        ax.grid(axis='y', alpha=.18)
        ax.set_xlabel('Condición (numeración de la tabla de diseño)')
    fig.legend(handles=[Line2D([],[],color=c,marker=m,ls='',label=l) for c,m,l in
                        [(BLUE,'o','Soporte y diagnósticos aprobados'),(ORANGE,'x','Alerta diagnóstica'),(PURPLE,'^','Verdad fuera del soporte')]],
               loc='outside upper center', ncol=3)
    save(fig, '03_recuperacion')

    # 4. Cobertura condicional y pérdidas de elegibilidad por condición.
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.3), layout='constrained', gridspec_kw={'width_ratios':[1,1.4]})
    y = np.arange(len(cov))
    axes[0].errorbar(cov.cobertura*100, y, xerr=np.vstack([(cov.cobertura-cov.wilson_lo)*100,(cov.wilson_hi-cov.cobertura)*100]), fmt='o', color=BLUE, capsize=3)
    axes[0].axvline(95, color='0.4', ls='--')
    axes[0].set(yticks=y, yticklabels=[f'{p} (n={n})' for p,n in zip(cov.parametro,cov.n)], xlim=(65,101), xlabel='Cobertura (%)',title='a) HPDI 95%: subconjunto elegible')
    good, diag, outside = [], [], []
    for cond in conditions:
        g=reg[reg.condicion.eq(cond)]
        outside.append(int((~g.verdad_en_soporte).sum()))
        diag.append(int((g.verdad_en_soporte & ~g.diagnosticos_estrictos_ok).sum()))
        good.append(int((g.verdad_en_soporte & g.diagnosticos_estrictos_ok).sum()))
    for vals,left,color,title in [(good,np.zeros(len(good)),BLUE,'Elegible'),(diag,np.array(good),ORANGE,'Alerta; verdad en soporte'),(outside,np.array(good)+diag,PURPLE,'Verdad fuera del soporte')]:
        axes[1].barh(range(len(names)), vals, left=left, color=color, label=title)
    axes[1].set(yticks=range(len(names)),yticklabels=names,xlabel='Número de casos',xticks=[0,2,4,6,8,10],title='b) Los 100 casos, incluidos los no elegibles')
    axes[1].invert_yaxis()
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5,-.10), frameon=False, fontsize=7)
    save(fig, '04_cobertura')

    # 5. Regla determinista de selección, definida sin inspección visual.
    rindex = reg.set_index('caso')
    controls = reg[reg.bloque.eq('control_factorial') & reg.verdad_en_soporte & reg.diagnosticos_estrictos_ok].caso
    agree = next(c for c in controls if gap.loc[c,'local'] <= .001)
    miss = gap.loc[list(controls),'local'].idxmax()
    excluded = reg.loc[~reg.verdad_en_soporte,'caso'].iloc[0]
    examples = [('Acuerdo entre estrategias',agree),('Mayor déficit local entre controles elegibles',miss),('Primera verdad fuera del soporte',excluded)]
    fig, axes = plt.subplots(6, 1, figsize=(10, 11), layout='constrained', gridspec_kw={'height_ratios':[2,1]*3})
    for j,(description,case) in enumerate(examples):
        folder=DATA/'casos'/case
        truth=js(folder/'verdad.json'); lp=best_params(case,'local'); gp=best_params(case,'global')
        comps=[(c,csv(folder/f'datos_{c}.csv')) for c in ['v1','v2'] if (folder/f'datos_{c}.csv').exists()]
        start=min(d.epoch.min() for _,d in comps); end=max(d.epoch.max() for _,d in comps)
        t=np.linspace(start,end,5000)
        ax,res=axes[2*j:2*j+2]
        for comp,d in comps:
            color=BLUE if comp=='v1' else RED
            ax.errorbar(d.epoch-start,d.RV,yerr=d.err,fmt='o' if comp=='v1' else '^',ms=3,color=color,elinewidth=.6)
            for p,style,c in [(truth,'-','0.55'),(lp,':',RED),(gp,'--',BLUE)]:
                ax.plot(t-start,rv(t,p,comp),style,color=c,lw=1.05,alpha=.9)
            for p,marker,color2 in [(lp,'x',RED),(gp,'o',BLUE)]:
                res.scatter(d.epoch-start,(d.RV-rv(d.epoch,p,comp))/d.err,marker=marker,c=color2,s=12)
        ax.set(title=f'{description}: {case} / {label(rindex.loc[case,"condicion"])}',ylabel='RV (km/s)')
        res.axhline(0,color='0.4',lw=.7);res.axhspan(-3,3,color='0.6',alpha=.13)
        res.set(yscale='symlog', ylabel='Residuo / σ',xlabel=f'Tiempo desde la primera observación (d; TJD inicial = {start:.2f})')
        res.set_xlim(ax.get_xlim())
    fig.legend(handles=[Line2D([],[],color='0.55',label='Órbita verdadera'),Line2D([],[],color=RED,ls=':',label='Mejor candidato local'),Line2D([],[],color=BLUE,ls='--',label='Mejor candidato global')],loc='outside upper center',ncol=3)
    save(fig, '05_ejemplos')

    # 6. Duraciones: sumas de búsquedas por proceso, no tiempo de reloj paralelo.
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    x=timings.local.to_numpy(); y=timings['global'].to_numpy()+reg.segundos_preproceso_global.to_numpy()
    for kind,marker,color in [('SB1','o',BLUE),('SB2','^',RED)]:
        keep=reg.tipo.eq(kind).to_numpy()
        axes[0].scatter(x[keep],y[keep],marker=marker,color=color,s=22,label=kind,alpha=.8)
    lim=[min(x.min(),y.min())*.8,max(x.max(),y.max())*1.3]
    axes[0].plot(lim,lim,color='0.5',ls='--',lw=.8)
    axes[0].set(xscale='log',yscale='log',xlabel='Local: dos búsquedas (s)',ylabel='Global: preparación + dos búsquedas (s)',title='a) Comparación por conjunto');axes[0].legend()
    total_local=x.sum()/3600;total_global=timings['global'].sum()/3600;pre=reg.segundos_preproceso_global.sum()/3600;nuts=reg.segundos_nuts.sum()/3600
    axes[1].barh([0,1],[total_local,total_global],color=BLUE,label='Optimización')
    axes[1].barh([1],[pre],left=[total_global],color=ORANGE,label='Preparación global')
    axes[1].barh([2],[nuts],color=PURPLE,label='NUTS posterior a global')
    axes[1].set(yticks=[0,1,2],yticklabels=['Local','Global','Posterior'],xlabel='Horas acumuladas por etapa',title='b) Suma de duraciones registradas')
    axes[1].legend(loc='lower right',frameon=False)
    save(fig,'06_costos')

    # 7. Diagnósticos y la cota que excluyó K2.
    fig,axes=plt.subplots(2,2,figsize=(10,6.8),layout='constrained')
    ids=np.arange(1,101)
    colors=np.where(reg.diagnosticos_estrictos_ok,BLUE,ORANGE)
    for ax,col,threshold,title in [(axes[0,0],'R_hat_max',1.01,'R-hat máximo'),(axes[0,1],'ESS_bulk_min',400,'ESS bulk mínimo')]:
        ax.scatter(ids,reg[col],c=colors,s=17);ax.axhline(threshold,color=RED,ls='--',lw=.8)
        ax.set(title=title,xlabel='Caso',ylabel=title)
    axes[0,0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0,1].set_yscale('log')
    axes[1,0].scatter(ids,reg.divergencias,c=colors,s=17)
    axes[1,0].set(title='Divergencias por conjunto',xlabel='Caso',ylabel='Número de divergencias')
    axes[1,0].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    support=[]
    for case in reg.loc[~reg.verdad_en_soporte,'caso']:
        d=js(DATA/'casos'/case/'datos_stan.json');truth=js(DATA/'casos'/case/'verdad.json')
        support.append((case,d['K2_max'],truth['K2']))
    axes[1,1].scatter(range(10),[a[1] for a in support],c=PURPLE,label='Límite superior de K₂')
    axes[1,1].axhline(support[0][2],color='0.3',ls='--',label='K₂ verdadero = 48.5')
    axes[1,1].set(title='Diez verdades excluidas del soporte',ylabel='K₂ (km/s)',xticks=range(10),xticklabels=[a[0][3:] for a in support],xlabel='Caso (sufijo V2)')
    axes[1,1].legend(fontsize=7)
    save(fig,'07_diagnosticos_soporte')

    captions = ['01_cadencia','02_optimizacion','03_recuperacion','04_cobertura','05_ejemplos','06_costos','07_diagnosticos_soporte']
    manifest=dict(casos=100,figuras=list(captions),ejemplos=[dict(regla=d,caso=c) for d,c in examples],
                  fuentes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)},
                  nota='Solo representación de resultados conservados; ninguna ejecución de Stan.')
    (OUT/'procedencia.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'figuras':len(captions),'casos':len(reg),'ejemplos':examples,'elegibles':sum(good)},ensure_ascii=False))

if __name__=='__main__':main()
