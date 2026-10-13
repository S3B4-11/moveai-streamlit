"""No mezcla personas ni convierte una postura parcial en técnica correcta."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from analysis import analyze
from decision_policy import apply_policy
from posture_review import add_posture_observations,reading_summary
from pose_selection import select_person,validate_target
from test_feedback import squat_fixture
from video_overlay import VideoOverlay,COLORS
from test_video_overlay import coloured_pixels

def pose_at(x,y,visible=True):
    return [SimpleNamespace(x=x,y=y,visibility=1. if visible else .1,presence=1.) for _ in range(33)]

def static_sample(pattern=None,wide=False):
    sample=squat_fixture(pattern,wide=wide);frame=10;n=24
    for key in ['world','image','visibility']:
        sample[key]=np.repeat(sample[key][frame:frame+1],n,axis=0)
    sample.update(times=np.arange(n)/8,detected=np.ones(n,bool),ambiguous=np.zeros(n,bool),duration=n/8)
    return sample

def posture_result(sample):
    r=analyze(sample,'squat');r['model']={'available':False,'reason':'criterios'}
    return apply_policy(reading_summary(add_posture_observations(r)))

class PersonPostureTests(unittest.TestCase):
    def test_explicit_target_ignores_larger_background_person(self):
        poses=[pose_at(.3,.4),pose_at(.65,.4),pose_at(.3,.8)]
        person,ambiguous,center=select_person(poses,[.3,.4])
        self.assertEqual(person,0);self.assertFalse(ambiguous)
        person,ambiguous,_=select_person(poses,[.3,.4],np.array([.32,.43]))
        self.assertEqual(person,0);self.assertFalse(ambiguous)
    def test_ambiguous_or_lost_target_never_selects_another_person(self):
        for poses in [[pose_at(.29,.4),pose_at(.31,.4)],[pose_at(.8,.4)],[pose_at(.3,.4,False)]]:
            index,_,_=select_person(poses,[.3,.4],np.array([.3,.4]))
            self.assertIsNone(index)
    def test_invalid_target_is_rejected(self):
        for target in [[-1,.5],[.5,2],[float('nan'),.5],[.5]]:
            with self.assertRaises(ValueError):validate_target(target)
    def test_static_error_is_red_without_a_complete_repetition(self):
        for pattern in ['inward','outward']:
            r=posture_result(static_sample(pattern))
            self.assertFalse(r['repetitions'])
            self.assertEqual(r['status'],'incorrect',r['reason'])
            self.assertTrue(r['posture_observations'])
            self.assertTrue(all(e['repetition'] is None and e['scope']=='postura' for e in r['errors']))
            self.assertEqual(sum(r['counts'].values()),0)
    def test_static_valid_or_wide_stance_cannot_certify_the_entire_exercise(self):
        for wide in [False,True]:
            r=posture_result(static_sample(wide=wide))
            self.assertFalse(r['errors']);self.assertEqual(r['status'],'uncertain')
            self.assertEqual(r['decision_source'],'postura_parcial')
    def test_hidden_feet_never_turn_static_pose_into_error(self):
        sample=static_sample('inward');sample['visibility'][:,[27,28,29,30,31,32]]=0
        r=posture_result(sample);self.assertFalse(r['errors']);self.assertEqual(r['status'],'unreadable')
    def test_static_overlay_marks_only_the_observed_fault(self):
        r=posture_result(static_sample('inward'));r['sample']['image']*=.65
        frame=VideoOverlay(r,640).draw(np.full((640,640,3),30,np.uint8),1.)
        self.assertGreater(coloured_pixels(frame,COLORS['rojo']),20)
        self.assertEqual(coloured_pixels(frame,COLORS['verde']),0)
    def test_missing_return_does_not_paint_an_observed_valid_posture_red(self):
        r=posture_result(static_sample());r['sample']['image']*=.65
        r['repetitions']=[dict(start_time=0.,end_time=3.,number=1,side='derecha',complete=False,color='amarillo',
                              checks=[dict(key='extension',status='fuera_de_rango')])]
        frame=VideoOverlay(r,640).draw(np.full((640,640,3),30,np.uint8),1.)
        self.assertEqual(coloured_pixels(frame,COLORS['rojo']),0)
        self.assertEqual(coloured_pixels(frame,COLORS['verde']),0)
        self.assertGreater(coloured_pixels(frame,COLORS['neutro']),20)
    def test_selection_changes_invalidate_the_previous_result(self):
        from web_session import SessionWorkspace
        w=SessionWorkspace()
        try:
            w.set_input(b'video','.mp4','squat','automatica',80,person_target=[.3,.4])
            w.result={'title':'anterior'}
            w.set_input(b'video','.mp4','squat','automatica',80,person_target=[.7,.4])
            self.assertIsNone(w.result)
        finally:w.close()
    def test_session_upgrade_preserves_pending_video_and_reviews(self):
        from web_session import SessionWorkspace
        w=SessionWorkspace();w.set_input(b'video','.mp4','squat','automatica',80)
        w.reviews.mkdir();review=w.reviews/'review.json';review.write_text('pending')
        del w.supports_person_selection
        upgraded=SessionWorkspace.upgrade(w)
        try:
            self.assertEqual(upgraded.video.read_bytes(),b'video')
            self.assertEqual(review.read_text(),'pending')
            upgraded.set_input(b'video','.mp4','squat','automatica',80,person_target=[.3,.4])
            self.assertEqual(review.read_text(),'pending')
        finally:upgraded.close()
    def test_explicit_selection_never_uses_an_incompatible_trained_classifier(self):
        from moveai_core import evaluate
        s=squat_fixture();s['person_selection']={'target':[.3,.4]}
        with patch('moveai_core.extract_selected_video',return_value=s),patch('moveai_core.predict') as predict:
            r=evaluate('video.mp4','squat',rotation=0,person_target=[.3,.4])
        predict.assert_not_called();self.assertFalse(r['model']['available']);self.assertFalse(r['sample']['trainable'])
    def test_streamlit_selection_and_posture_error_need_no_repetition(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        class Upload(io.BytesIO):
            name='postura.mp4'
            def __init__(self):super().__init__(b'selected posture video')
        r=posture_result(static_sample('inward'))
        app=AppTest.from_file(str(Path(__file__).parent/'streamlit_app.py'),default_timeout=20)
        app.secrets['app_password']='test-pass';app.session_state['authenticated']=True
        with patch.object(st,'file_uploader',return_value=Upload()),patch('pose_selection.selection_preview',return_value=None),patch('moveai_core.evaluate',return_value=r) as evaluate,patch('video_overlay.render_video',side_effect=RuntimeError('omitido')),patch('logging.exception'):
            app.run();next(x for x in app.checkbox if x.label.startswith('Elegir persona')).check().run()
            next(x for x in app.slider if x.label=='Posición horizontal (%)').set_value(30).run()
            next(x for x in app.button if x.label=='Analizar video').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(evaluate.call_args.kwargs['person_target'],[.3,.4])
            self.assertTrue(any(x.value=='Ejercicio mal ejecutado' for x in app.error))
            self.assertTrue(any('Qué corregir' in x.value for x in app.markdown))
            self.assertTrue(any('no se cuenta como una repetición' in x.value for x in app.caption))
            self.assertFalse(any(x.label=='Ver una repetición' for x in app.selectbox))
            app.session_state['workspace'].close()

if __name__=='__main__':unittest.main()
