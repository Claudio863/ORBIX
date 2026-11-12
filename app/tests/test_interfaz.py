"""Widget wiring and fixed-parameter controls; no new scientific fits."""
import unittest
from streamlit.testing.v1 import AppTest
from app.servicio import ROOT


class ControlesTests(unittest.TestCase):
    def test_four_schemes_for_both_types(self):
        app = AppTest.from_file(str(ROOT/'app/App.py')).run(timeout=30)
        for kind in ['SB1', 'SB2']:
            app.radio[0].set_value(kind).run()
            for scheme, fixed_p, fixed_e in [('P y e libres', False, False),
                    ('P fijo', True, False), ('e fija', False, True), ('P y e fijados', True, True)]:
                next(s for s in app.selectbox if s.label == 'Parámetros fijados').set_value(scheme).run()
                self.assertFalse(app.exception)
                labels = [widget.label for widget in app.number_input]
                self.assertEqual('Período fijado (días)' in labels, fixed_p)
                self.assertEqual('Excentricidad fijada' in labels, fixed_e)
                labels = [widget.label for widget in app.selectbox]
                self.assertEqual('Período P' in labels, not fixed_p)
                self.assertEqual('Excentricidad e' in labels, not fixed_e)
                self.assertEqual('Semiamplitud K₂' in labels, kind == 'SB2')
                self.assertTrue(next(b for b in app.button if b.label == 'Ejecutar optimización del sistema').disabled)

if __name__ == '__main__':
    unittest.main()
