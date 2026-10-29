import unittest
import numpy as np
import pandas as pd
from orbix.resumen import normalizar_booleanos, recentrar_circulares


class ResumenTests(unittest.TestCase):
    def test_flags_csv_mixtos(self):
        valores = pd.Series(['True', 'False', '0.0', '1.0', True, False, 0, 1, None])
        r = normalizar_booleanos(valores)
        self.assertEqual(r.fillna(False).tolist(), [True, False, False, True, True, False, False, True, False])
        self.assertEqual(int(r.sum()), 4)

    def test_flag_desconocido_no_se_acepta(self):
        with self.assertRaises(ValueError):
            normalizar_booleanos(pd.Series(['desconocido']))

    def test_epoca_invariante_a_origen_y_equivalente_modulo_periodo(self):
        P = np.array([8.0, 9.0, 10.0, 11.0])
        fase = np.array([0.98, 0.99, 0.01, 0.02])
        d = pd.DataFrame({'P': P, 'T0': fase, 'T': 100.0 + P*fase})
        r = recentrar_circulares(d)
        trasladada = d.assign(T=d['T']+2450000)
        rt = recentrar_circulares(trasladada)
        np.testing.assert_allclose(rt['T']-2450000, r['T'], atol=1e-9)
        np.testing.assert_allclose((r['T']-d['T'])/P, np.round((r['T']-d['T'])/P))
        self.assertLess(np.ptp(r['T0']), 0.05)
        pd.testing.assert_frame_equal(recentrar_circulares(r), r)

    def test_sin_fase_no_se_inventa_origen(self):
        d = pd.DataFrame({'P': [8., 9.], 'T': [2450000., 2450001.]})
        pd.testing.assert_frame_equal(recentrar_circulares(d), d)

    def test_epoca_se_reconstruye_desde_origen_fijo_y_fase(self):
        d=pd.DataFrame({'P':[.42815,.42816], 'T0':[.10161,.10162], 'T':[1971.9827,1971.9827]})
        origen=1971.9391621989296
        r=recentrar_circulares(d,origen)
        np.testing.assert_allclose(r['T'],origen+d['P']*d['T0'],atol=1e-12)
        self.assertGreater(np.ptp(r['T']),0)


if __name__ == '__main__':
    unittest.main()
