import unittest
import numpy as np
import pandas as pd
from orbix import ajuste,periodo,rv

class InicializacionTests(unittest.TestCase):
    def test_fase_acotada_conserva_anomalia_y_usa_referencia_ponderada(self):
        d=pd.DataFrame({'epoch':[0.,2.,100.],'RV':[0.,0.,0.],'err':[.1,.1,2.]})
        g=ajuste.Geometria({'v1':d})
        self.assertAlmostEqual(g.t_ref,np.average(d.epoch,weights=1/d.err**2))
        P,phi,w=13.3,.2,1.1
        ini=g.parametros_muestreo(phi,w,.4,P,False)
        self.assertGreaterEqual(ini['lam_ref'],0)
        self.assertLess(ini['lam_ref'],2*np.pi)
        m_model=2*np.pi*(d.epoch.to_numpy()-g.t_ref)/P+ini['lam_ref']-w
        m_fisico=2*np.pi*(d.epoch.to_numpy()-g.min_t-P*phi)/P
        np.testing.assert_allclose(np.exp(1j*m_model),np.exp(1j*m_fisico),atol=1e-12)

    def test_ancla_no_se_perturba_ni_usa_literatura(self):
        comps={'v1':pd.DataFrame({'epoch':[0.,1.,2.],'RV':[1.,3.,2.],'err':[1.,1.,1.]})}
        geo=ajuste.Geometria(comps)
        plan=pd.DataFrame([{'origen':'pico_1','criterio':'comb','P_centro_d':8.,'sigma_lnP':.02}])
        sol={'pico_1':{'P_d':8.123,'T0':.25,'w':.7,'e':.8,'V0':2.,'K_v1':1.5}}
        data={'V0_min':-10.,'V0_max':10.,'K_max':10.,'log_P_min':np.log(1.),'log_P_max':np.log(20.)}
        resultados=[]
        for semilla,elit in [(123,.1),(456,.9)]:
            gen=ajuste._generador_inicios(np.random.default_rng(semilla),'SB1','free',comps,geo,9.,elit,data,
                                         plan=plan,soluciones=sol,dias_a_work=lambda x:x)
            ini,info=gen(0)
            self.assertTrue(info['ancla_refinada'])
            self.assertAlmostEqual(np.exp(ini['log_P']),8.123)
            self.assertAlmostEqual(ini['K'],1.5)
            self.assertAlmostEqual(info['e_semilla'],.8)
            resultados.append(ini)
        self.assertEqual(resultados[0],resultados[1])

    def test_refinamiento_recupera_sb2_excentrica_con_ruido(self):
        rng=np.random.default_rng(99)
        t=np.sort(rng.uniform(0,200,60))
        verdad={'P':17.3,'T':3.5,'e':.82,'w_deg':125.,'V0':8.,'K1':25.,'K2':35.}
        curvas=rv.curvas(verdad,t,'SB2')
        comps={c:pd.DataFrame({'epoch':t,'RV':y+rng.normal(0,.3,len(t)),'err':np.full(len(t),.3)})
               for c,y in curvas.items()}
        sol=periodo.refinar_pico(comps,{c:t for c in comps},17.35,float(t.min()),float(np.ptp(t)),
                                10.,25.,n_frecuencias=11)
        self.assertLess(abs(sol['P_d']/verdad['P']-1),.002)
        self.assertLess(sol['chi2']/(2*len(t)-7),2.)
        self.assertLess(abs(sol['e']-verdad['e']),.02)

if __name__=='__main__':
    unittest.main()

