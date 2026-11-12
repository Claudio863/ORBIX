"""Checks of the upload/configuration boundary, without running inference."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
from app import servicio
from orbix import ajuste, modelos


class InterfazTests(unittest.TestCase):
    def setUp(self):
        root = servicio.ROOT/'análisis_desempeño/corridas_masivas/data/HIP89000'
        self.first = (root/'component_1.csv').read_bytes()
        self.second = (root/'component_2.csv').read_bytes()
        self.components = {'v1': servicio.leer_csv(self.first), 'v2': servicio.leer_csv(self.second)}

    def test_invalid_observations_rejected(self):
        for payload in [b'epoch,RV\n1990,2\n', self.first.replace(b',0.5', b',0.0')]:
            if payload == self.first:
                continue
            with self.assertRaises(ValueError):
                servicio.leer_csv(payload)
        invalid = self.components['v1'].copy()
        invalid.loc[0, 'err'] = 0
        with self.assertRaises(ValueError):
            servicio.leer_csv(invalid.to_csv(index=False).encode())

    def test_sb2_pairs_required_and_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            servicio.agrupar_archivos([('HIP_v1.csv', self.first)], 'SB2')
        with self.assertRaises(ValueError):
            servicio.agrupar_archivos([('HIP.csv', self.first), ('HIP.csv', self.second)], 'SB1')
        result = servicio.agrupar_archivos([('HIP_v1.csv', self.first), ('HIP_v2.csv', self.second)], 'SB2')
        self.assertEqual(set(result['HIP']), {'v1', 'v2'})

    def test_security_block_stops_after_first_attempt_and_preserves_record(self):
        components = {'v1': self.components['v1']}
        priors = pd.DataFrame([dict(sistema='*', tipo='SB1', parametro=p,
            familia='flat', p1='', p2='', unidad='') for p in ['T0', 'V0', 'K']])
        cfg, *_ = servicio.preparar(components, 'year_besselian', 'blocked',
            'SB1', 'fix_eP', 30., 1000., 199.4, .3, np.nan, priors)
        model = Mock()
        model.optimize.side_effect = RuntimeError('[WinError 4551] Application control blocked this file')
        with TemporaryDirectory(prefix='orbix_interface_test_') as directory:
            root = Path(directory)
            (root/'entrada').mkdir()
            components['v1'].to_csv(root/'entrada/v1.csv', index=False)
            with patch('orbix.modelos.compilar', return_value=model):
                result = ajuste.correr(cfg, 'fix_eP', priors,
                    ajuste.Rutas(root, servicio.ROOT/'modelos'), ajuste.Ajustes(), log=lambda _: None)
            self.assertEqual(model.optimize.call_count, 1)
            self.assertEqual(result['registro']['estado'], 'fallo')
            self.assertTrue(result['registro']['fallo_entorno'])
            self.assertTrue((root/'resultados/draws/blocked_SB1_fix_eP/inicios_lbfgs.csv').exists())
            model.sample.assert_not_called()

    def test_preliminary_checkpoint_resumes_without_repeating_optimization(self):
        components = {'v1': self.components['v1']}
        priors = pd.DataFrame([dict(sistema='*', tipo='SB1', parametro=p,
            familia='flat', p1='', p2='', unidad='') for p in ['T0', 'V0', 'K']])
        cfg, *_ = servicio.preparar(components, 'year_besselian', 'stages',
            'SB1', 'fix_eP', 30., 1000., 199.4, .3, np.nan, priors)
        model = Mock()
        model.optimize.return_value.optimized_params_dict = dict(
            lp__=-20., P=.546, T=1990.676, T0=.08, e=.3, w=1.5,
            w_deg=85.9, V0=-14., K=14.)
        with TemporaryDirectory(prefix='orbix_stages_') as directory:
            root = Path(directory)
            (root/'entrada').mkdir()
            components['v1'].to_csv(root/'entrada/v1.csv', index=False)
            routes = ajuste.Rutas(root, servicio.ROOT/'modelos')
            settings = ajuste.Ajustes(n_inicios_SB1=2)
            with patch('orbix.modelos.compilar', return_value=model):
                updates = Mock()
                first = ajuste.correr(cfg, 'fix_eP', priors, routes, settings,
                                     etapa='preliminar', progreso=updates)
                self.assertEqual([c.args[1:] for c in updates.call_args_list],
                                 [(0, 0), (0, 2), (1, 2), (2, 2)])
                self.assertEqual(first['registro']['estado'], 'preliminar', first['registro']['motivo'])
                model.sample.assert_not_called()
                model.sample.side_effect = RuntimeError('sampling reached')
                second = ajuste.correr(cfg, 'fix_eP', priors, routes, settings, etapa='muestreo')
            self.assertIn('sampling reached', second['registro']['motivo'])
            self.assertEqual(model.optimize.call_count, 2)
            model.sample.assert_called_once()
            self.assertEqual(len(model.sample.call_args.kwargs['inits']), settings.cadenas)

    def test_all_schemes_use_current_amplitudes_and_explicit_fixed_values(self):
        for kind in ['SB1', 'SB2']:
            for scheme in modelos.ESQUEMAS:
                components = self.components if kind == 'SB2' else {'v1': self.components['v1']}
                parameters = ['P', 'e', 'T0', 'V0'] + (['K'] if kind == 'SB1' else ['K1', 'K2'])
                priors = pd.DataFrame([dict(sistema='*', tipo=kind, parametro=p, familia='flat', p1='', p2='', unidad='') for p in parameters])
                cfg, info, hyper, effective = servicio.preparar(components, 'year_besselian', 'HIP89000', kind, scheme, 30., 1000., 199.4, .3, np.nan, priors)
                self.assertEqual(info['P_init_fuente'], 'lomb-scargle')
                self.assertEqual(set(p for p in effective.parametro if p.startswith('K')), set(parameters[4:]))
                self.assertNotIn('q', effective.parametro.tolist())
                fix_p, fix_e = modelos.fijos(scheme)
                self.assertEqual(np.isfinite(cfg.P_lit), fix_p)
                self.assertEqual(np.isfinite(cfg.e_lit), fix_e)
                if fix_p:
                    self.assertAlmostEqual(cfg.P_lit, 199.4)
                self.assertTrue(all(hyper[p+'_max'] > 0 for p in parameters[4:]))


if __name__ == '__main__':
    unittest.main()
