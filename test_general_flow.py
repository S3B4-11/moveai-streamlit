"""Regresiones de configuración entre videos y salida binaria legible."""
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from analysis import analyze
from decision_policy import apply_policy
from test_moveai import fixture

class GeneralFlowTests(unittest.TestCase):
    def test_readable_model_scores_are_binary_without_an_uncertainty_band(self):
        for p,label in [(.9,'correct'),(.65,'correct'),(.51,'correct'),(.5,'correct'),(.45,'correct'),(.1,'incorrect')]:
            r=analyze(fixture(),'elbow_flexion');r['model']={'available':True,'p_correct':p}
            self.assertEqual(apply_policy(r)['status'],label)
    def test_weak_model_signal_does_not_overrule_completed_passing_criteria(self):
        r=analyze(fixture(),'elbow_flexion');r['model']={'available':True,'p_correct':.42}
        r=apply_policy(r)
        self.assertEqual(r['status'],'correct');self.assertEqual(r['decision_source'],'criterios_y_modelo')
    def test_one_unmeasured_segment_does_not_erase_a_passing_majority(self):
        import copy
        r=analyze(fixture(),'elbow_flexion');rep=r['repetitions'][0]
        r['repetitions']=[copy.deepcopy(rep) for _ in range(5)]
        r['repetitions'][-1].update(color='amarillo',reason='Faltan medidas fiables del inicio, pico o retorno.',measured_fraction=.5)
        r['model']={'available':True,'p_correct':.42}
        r=apply_policy(r);self.assertEqual(r['status'],'correct')
        self.assertEqual(r['counts']['amarillo'],1)
    def test_a_measured_unresolved_criterion_is_not_silently_ignored(self):
        import copy
        r=analyze(fixture(),'elbow_flexion');rep=r['repetitions'][0]
        r['repetitions']=[copy.deepcopy(rep) for _ in range(5)]
        r['repetitions'][-1].update(color='amarillo',reason='Se necesita otra medida.',measured_fraction=.9,visible_fraction=.9)
        r['model']={'available':True,'p_correct':.42}
        self.assertEqual(apply_policy(r)['status'],'incorrect')

    def test_small_deviation_within_existing_tolerance_is_accepted(self):
        r=analyze(fixture(peak=105),'elbow_flexion');r['model']={'available':False}
        self.assertTrue(any(c['status']=='limite' for rep in r['repetitions'] for c in rep['checks']))
        r=apply_policy(r)
        self.assertEqual(r['status'],'correct',r['reason'])
        self.assertTrue(any(rep.get('tolerance_applied') for rep in r['repetitions']))

    def test_clear_incomplete_range_remains_wrong_even_with_a_positive_model(self):
        r=analyze(fixture(peak=60),'elbow_flexion');r['model']={'available':True,'p_correct':.99}
        r=apply_policy(r);self.assertEqual(r['status'],'incorrect');self.assertTrue(r['errors'])
        self.assertEqual(r['errors'][0]['key'],'rango_incompleto')

    def test_occlusion_cannot_be_hidden_by_a_positive_score(self):
        s=fixture();s['visibility'][:]=0
        r=analyze(s,'elbow_flexion');r['model']={'available':True,'p_correct':.99}
        self.assertEqual(apply_policy(r)['status'],'unreadable')

    def test_person_selection_and_side_do_not_leak_to_a_new_video(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        class Upload(io.BytesIO):
            name='video.mp4'
            def __init__(self,data):super().__init__(data)
        current=Upload(b'first video')
        r=analyze(fixture(),'elbow_flexion');r['model']={'available':False,'reason':'criterios'};apply_policy(r)
        app=AppTest.from_file(str(Path(__file__).parent/'streamlit_app.py'),default_timeout=20)
        app.secrets['app_password']='test-pass';app.session_state['authenticated']=True
        with patch.object(st,'file_uploader',side_effect=lambda *a,**k:current),patch('pose_selection.selection_preview',return_value=None),patch('moveai_core.evaluate',return_value=r) as evaluate,patch('video_overlay.render_video',side_effect=RuntimeError('omitido')),patch('logging.exception'):
            app.run()
            next(x for x in app.checkbox if x.label.startswith('Elegir persona')).check().run()
            next(x for x in app.slider if x.label=='Posición horizontal (%)').set_value(30).run()
            next(x for x in app.selectbox if x.label=='Extremidad').select('izquierda').run()
            next(x for x in app.button if x.label=='Analizar video').click().run()
            self.assertEqual(evaluate.call_args.kwargs['person_target'],[.3,.4])
            current=Upload(b'second video with same filename')
            app.run()
            self.assertFalse(next(x for x in app.checkbox if x.label.startswith('Elegir persona')).value)
            self.assertEqual(next(x for x in app.selectbox if x.label=='Extremidad').value,'automatica')
            self.assertIsNone(app.session_state['workspace'].result)
            next(x for x in app.button if x.label=='Analizar video').click().run()
            self.assertIsNone(evaluate.call_args.kwargs['person_target'])
            self.assertFalse(app.exception)
            self.assertTrue(any(x.value=='Ejercicio realizado correctamente' for x in app.success))
            settings=next(x for x in app.expander if x.label=='Ajustar ejercicio y toma')
            self.assertFalse(settings.proto.expanded)
            app.session_state['workspace'].close()

if __name__=='__main__':unittest.main()
