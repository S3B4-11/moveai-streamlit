"""Regresiones de evidencia temporal, cobertura y salida parcial sin forzar verde."""
import copy,unittest,io
from pathlib import Path
from unittest.mock import patch
import numpy as np
from press_quality import phase_reading,sustained_seconds,confirm_deviations,finalize_press
from press_analysis import analyze_press
from decision_policy import apply_policy
from test_press import press_fixture

class PressQualityTests(unittest.TestCase):
    def test_good_average_does_not_replace_missing_peak_phase(self):
        mask=np.ones(24,bool);mask[11]=False;mask[13]=False
        self.assertGreater(mask.mean(),.8)
        phases,good=phase_reading(mask,np.arange(24),0,12,24)
        self.assertFalse(good);self.assertEqual(phases['fase_superior']['measured'],1)
    def test_good_average_does_not_replace_missing_return(self):
        mask=np.ones(24,bool);mask[-3:]=False
        self.assertGreater(mask.mean(),.8)
        phases,good=phase_reading(mask,np.arange(24),0,12,24)
        self.assertFalse(good);self.assertEqual(phases['retorno']['measured'],1)
    def test_very_short_deviation_needs_confirmation(self):
        c=dict(key='press_extension',name='Extensión del codo',status='fuera_de_rango')
        self.assertTrue(confirm_deviations([c],{'press_extension':[False,True,True,False]},np.arange(4)/8))
        self.assertEqual(c['status'],'limite');self.assertEqual(c['evidence_status'],'not_sustained')
    def test_continuous_deviation_keeps_error(self):
        c=dict(key='press_extension',name='Extensión del codo',status='fuera_de_rango')
        self.assertFalse(confirm_deviations([c],{'press_extension':[True,True,True,False]},np.arange(4)/8))
        self.assertEqual(c['status'],'fuera_de_rango')
    def test_two_fragments_cannot_be_joined_as_persistent_evidence(self):
        self.assertLess(sustained_seconds([True]*4,[0,.125,1.,1.125]),.25)
    def partial(self):
        r=analyze_press(press_fixture());rep=r['repetitions'][0]
        r['repetitions']=[copy.deepcopy(rep) for _ in range(5)]
        for i,p in enumerate(r['repetitions'],1):p['number']=i
        r['repetitions'][-1].update(color='amarillo',reason='Faltan medidas fiables en retorno.',measured_fraction=.4)
        r['model']={'available':False,'reason':'Criterios del press.'}
        return finalize_press(apply_policy(r))
    def test_partial_passing_cycles_do_not_turn_the_video_green(self):
        r=self.partial();self.assertEqual(r['status'],'unreadable');self.assertTrue(r['partial_assessment'])
        self.assertEqual(r['assessment_summary']['assessed_complete_cycles'],4)
        self.assertIn('4 de 5',r['reason']);self.assertEqual(r['counts']['verde'],4)
    def test_reading_exposes_wrist_and_camera_guidance(self):
        s=press_fixture();s['visibility'][:,[15,16]]=0
        r=analyze_press(s);r['model']={'available':False}
        r=finalize_press(apply_policy(r))
        self.assertEqual(r['reading']['joint_coverage']['muñeca derecha'],0)
        self.assertTrue(any('manos' in x for x in r['capture_guidance']))
    def test_ui_preserves_partial_results_under_global_abstention(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        r=self.partial()
        class Upload(io.BytesIO):name='press.mp4'
        app=AppTest.from_file(str(Path(__file__).parent/'streamlit_app.py'),default_timeout=20)
        app.secrets['app_password']='test-pass';app.session_state['authenticated']=True
        with patch.object(st,'file_uploader',return_value=Upload(b'partial')),patch('moveai_core.evaluate',return_value=r),patch('video_overlay.render_video',side_effect=RuntimeError('test')),patch('logging.exception'):
            app.run();next(x for x in app.selectbox if x.label=='Ejercicio').select('shoulder_press').run()
            next(x for x in app.button if x.label=='Analizar video').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(x.value=='No se pudo evaluar' for x in app.warning))
            self.assertTrue(any('4 de 5 ciclos completos' in x.value for x in app.markdown))
            app.session_state['workspace'].close()
if __name__=='__main__':unittest.main()
