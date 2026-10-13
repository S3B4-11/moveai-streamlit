"""Errores observables, aperturas válidas y decisiones coherentes."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from analysis import analyze
from decision_policy import apply_policy
from squat_alignment import add_alignment,alignment_checks
from test_moveai import fixture
from video_orientation import inferred_rotation,rotate_video
from frontal_quality import repair_frontal_projection
from feedback_presentation import APP_REVISION, POLICY_VERSION

def squat_fixture(pattern=None,wide=False,diagonal=False):
    t=np.arange(0,6.5,1/8);n=len(t);phase=np.sin(np.pi*np.clip(((t-.25)%3)/2.5,0,1))
    phase[(((t-.25)%3)>2.5)|(t<.25)]=0
    f=np.deg2rad(105*phase/2);world=np.zeros((n,33,3));image=np.zeros((n,33,2))
    hips=.90-.42*np.cos(f);knee_y=.90-.21*np.cos(f)
    image[:,11]=np.column_stack([np.full(n,.34),hips-.28]);image[:,12]=np.column_stack([np.full(n,.66),hips-.28])
    image[:,23]=np.column_stack([np.full(n,.38),hips]);image[:,24]=np.column_stack([np.full(n,.62),hips])
    world[:,11]=[-.2,-.7,0];world[:,12]=[.2,-.7,0];world[:,23]=[-.12,0,0];world[:,24]=[.12,0,0]
    for side,(h,k,a,heel,toe),sign in [('izquierda',(23,25,27,29,31),-1),('derecha',(24,26,28,30,32),1)]:
        ankle=.5+sign*(.34 if wide else .18);hip=.5+sign*.12
        baseline=(hip+ankle)/2
        x=np.full(n,baseline)
        if wide:x+=(ankle-baseline)*phase
        if pattern=='inward' and side=='izquierda':x+=.18*phase
        if pattern=='outward' and side=='derecha':x+=.20*phase
        image[:,k]=np.column_stack([x,knee_y]);image[:,a]=[ankle,.90]
        image[:,heel]=[ankle-sign*(.01 if wide else .002),.91];image[:,toe]=[ankle+sign*(.08 if wide else .002),.98]
        world[:,k]=np.column_stack([np.full(n,sign*.18),.45*np.cos(f),-.45*np.sin(f)])
        world[:,a]=np.column_stack([np.full(n,sign*.24),.9*np.cos(f),np.zeros(n)])
        world[:,heel]=world[:,a]+[0,.02,.01];world[:,toe]=world[:,a]+[sign*.04,.01,-.08]
    if diagonal:
        world[:,12,2]=.45;world[:,24,2]=.35
    return dict(world=world,image=image,visibility=np.ones((n,33)),times=t,detected=np.ones(n,bool),ambiguous=np.zeros(n,bool),
                coverage=1.,duration=float(t[-1]),pose_version='fixture',extractor_id='fixture',pipeline='moveai_rgb_v4',previews=[])

def decided(sample,exercise='squat',model=None):
    result=analyze(sample,exercise);result['model']=model or {'available':False,'reason':'criterios'}
    repair_frontal_projection(result)
    return apply_policy(add_alignment(result))

class FeedbackTests(unittest.TestCase):
    def test_ui_never_labels_measured_pass_as_correct_under_model_rejection(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from test_video_overlay import visual_fixture
        class Upload(io.BytesIO):
            name='sentadilla.mp4'
            def __init__(self):super().__init__(b'video test')
        for p,status in [(.34,'incorrect'),(.35,'incorrect')]:
            r=visual_fixture();r['quality']['ready']=True
            r['model']={'available':True,'p_correct':p,'scope':'Prueba de presentación'}
            apply_policy(r)
            self.assertEqual(r['status'],status)
            app=AppTest.from_file(str(Path(__file__).parent/'streamlit_app.py'),default_timeout=20)
            app.secrets['app_password']='test-pass';app.session_state['authenticated']=True
            with patch.object(st,'file_uploader',return_value=Upload()),patch('moveai_core.evaluate',return_value=r),patch('video_overlay.render_video',side_effect=RuntimeError('omitido')),patch('logging.exception'):
                app.run();next(x for x in app.button if x.label=='Analizar video').click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any(x.label=='Sin fallos medidos' for x in app.metric))
                self.assertFalse(any(x.label=='Correctas' for x in app.metric))
                selector=next(x for x in app.selectbox if x.label=='Ver una repetición')
                self.assertTrue(all('Sin fallos medidos' in x for x in selector.options))
                self.assertFalse(any('No se pudo evaluar' in x.value for x in app.warning))
                self.assertTrue(any(APP_REVISION in x.value for x in app.caption))
                if status=='incorrect':self.assertTrue(any(x.value=='Ejercicio mal ejecutado' for x in app.error))
                else:self.assertTrue(any(x.value=='Resultado no concluyente' for x in app.warning))
                app.session_state['workspace'].close()

    def test_update_invalidates_previous_session_result(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).parent/'streamlit_app.py'),default_timeout=20)
        app.secrets['app_password']='test-pass';app.session_state['authenticated']=True
        app.run();workspace=app.session_state['workspace']
        workspace.result={'color':'amarillo','title':'No se pudo evaluar'}
        app.session_state['_app_revision']='anterior'
        app.run()
        self.assertFalse(app.exception);self.assertIsNone(workspace.result)
        self.assertTrue(any(x.value=='Tu resultado aparecerá aquí.' for x in app.info))
        workspace.close()

    def test_old_engine_response_is_rejected_with_reboot_instruction(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        class Upload(io.BytesIO):
            name='video.mp4'
            def __init__(self):super().__init__(b'old engine test')
        app=AppTest.from_file(str(Path(__file__).parent/'streamlit_app.py'),default_timeout=20)
        app.secrets['app_password']='test-pass';app.session_state['authenticated']=True
        with patch.object(st,'file_uploader',return_value=Upload()),patch('moveai_core.evaluate',return_value={'feedback_version':'moveai_feedback_v5'}),patch('video_overlay.render_video') as render:
            app.run();next(x for x in app.button if x.label=='Analizar video').click().run()
            self.assertFalse(app.exception);render.assert_not_called()
            self.assertTrue(any('Reboot app' in x.value for x in app.error))
            self.assertIsNone(app.session_state['workspace'].result)
            app.session_state['workspace'].close()

    def test_current_video_classifier_does_not_localize_bad_repetitions(self):
        from test_video_overlay import visual_fixture,coloured_pixels
        from video_overlay import VideoOverlay,COLORS
        for p in [.34,.35]:
            r=visual_fixture();r['quality']['ready']=True;r['model']={'available':True,'p_correct':p}
            apply_policy(r)
            frame=VideoOverlay(r,360).draw(np.full((640,360,3),30,np.uint8),r['repetitions'][0]['peak_time'])
            self.assertEqual(coloured_pixels(frame,COLORS['verde']),0)
            self.assertEqual(coloured_pixels(frame,COLORS['rojo']),0)
            self.assertGreater(coloured_pixels(frame,COLORS['neutro']),20)

    def test_inward_and_outward_are_different_and_detected_bilaterally(self):
        for pattern,name in [('inward','hacia_dentro'),('outward','hacia_fuera')]:
            sample=squat_fixture(pattern)
            result=decided(sample,model={'available':True,'p_correct':.97})
            self.assertEqual(result['status'],'incorrect',result['reason'])
            self.assertEqual(result['decision_source'],'error_observable')
            self.assertTrue(any(c.get('pattern')==name for r in result['repetitions'] for c in r.get('alignment_checks',[])))
            self.assertTrue(all(e['consequence'] for e in result['errors']))
            self.assertLess(result['criterion_score'],1.)
    def test_normal_and_wide_stance_are_not_automatically_errors(self):
        for wide in [False,True]:
            result=decided(squat_fixture(wide=wide),model={'available':True,'p_correct':.9})
            self.assertEqual(result['status'],'correct',result['reason'])
            self.assertFalse(result['errors'])
    def test_diagonal_view_and_hidden_feet_cannot_invent_alignment_errors(self):
        for sample in [squat_fixture('inward',diagonal=True),squat_fixture('inward')]:
            if not sample['world'][0,12,2]:sample['visibility'][:,[29,30,31,32]]=0
            checks=alignment_checks(sample,0,len(sample['times']))
            self.assertFalse(any(c['status']=='fuera_de_rango' for c in checks))
    def test_single_frame_spike_is_not_an_error(self):
        sample=squat_fixture();sample['image'][12,25,0]+=.3
        checks=alignment_checks(sample,0,len(sample['times']))
        self.assertFalse(any(c['status']=='fuera_de_rango' for c in checks))
    def test_outward_motion_following_toes_is_not_a_varus_label(self):
        sample=squat_fixture('outward');sample['image'][:,32,0]+=.12
        checks=alignment_checks(sample,0,len(sample['times']))
        self.assertFalse(any(c.get('pattern')=='hacia_fuera' for c in checks))
    def test_clear_error_beats_positive_model_for_curl(self):
        r=decided(fixture(peak=60),'elbow_flexion',{'available':True,'p_correct':.99})
        self.assertEqual((r['status'],r['color']),('incorrect','rojo'))
    def test_model_negative_is_not_called_unreadable(self):
        r=decided(fixture(),'elbow_flexion',{'available':True,'p_correct':.35})
        self.assertEqual(r['status'],'incorrect');self.assertFalse(r['errors'])
        self.assertIn('no permiten localizar',r['reason'])
    def test_partial_edge_does_not_erase_complete_good_repetitions(self):
        r=analyze(fixture(),'elbow_flexion');partial=copy.deepcopy(r['repetitions'][0])
        partial.update(complete=False,color='amarillo',reason='Falta el inicio completo.')
        r['repetitions'].insert(0,partial);r['model']={'available':False}
        r=apply_policy(r);self.assertEqual(r['status'],'correct');self.assertEqual(r['counts']['amarillo'],1)
    def test_occlusion_is_not_classified_as_bad_technique(self):
        sample=fixture();sample['visibility'][:]=0
        r=decided(sample,'elbow_flexion',{'available':False})
        self.assertEqual(r['status'],'unreadable');self.assertFalse(r['errors'])
    def test_borderline_valid_motion_is_separate_from_no_data(self):
        r=analyze(fixture(peak=110),'elbow_flexion')
        for rep in r['repetitions']:rep['color']='amarillo'
        r['quality']['ready']=True;r['model']={'available':True,'p_correct':.6}
        r=apply_policy(r);self.assertEqual(r['status'],'correct');self.assertEqual(r['title'],'Ejercicio realizado correctamente')
    def test_orientation_uses_body_and_legs_not_just_trunk_lean(self):
        sample=squat_fixture();self.assertEqual(inferred_rotation(sample),0)
        # Ejes del cuerpo girados; misma confianza y sin modificar los datos originales.
        rotated=copy.deepcopy(sample);p=rotated['image'];rotated['image']=np.stack([-p[:,:,1],p[:,:,0]],axis=-1)
        self.assertEqual(inferred_rotation(rotated),-90)
        leaning=squat_fixture();leaning['image'][:,[11,12],0]+=.35
        self.assertEqual(inferred_rotation(leaning),0)
    def test_model_inputs_remain_unchanged_by_feedback(self):
        from camera_model import features
        sample=squat_fixture('inward');before,_=features(sample,'squat')
        r=decided(sample);after,_=features(r['sample'],'squat')
        np.testing.assert_array_equal(before,after)
    def test_frontal_projection_is_not_mistaken_for_unreadable_depth(self):
        raw=analyze(squat_fixture(),'squat')
        self.assertFalse(raw['quality']['ready'])
        self.assertTrue(repair_frontal_projection(raw));self.assertTrue(raw['quality']['ready'])
        self.assertTrue(all(r['color']=='verde' for r in raw['repetitions']))
    def test_frontal_adapter_never_removes_occlusion_or_segment_instability(self):
        for reason in ['Los segmentos estimados cambian demasiado de longitud.','La extremidad activa queda oculta.']:
            raw=analyze(squat_fixture(),'squat')
            for t in raw['tracks']:t['reason']=reason
            self.assertFalse(repair_frontal_projection(raw));self.assertFalse(raw['quality']['ready'])
    def test_upside_down_orientation_is_stable_around_angle_wrap(self):
        sample=squat_fixture();sample['image']=-sample['image'];sample['image'][:,[11,12],0]+=np.sin(sample['times'])[:,None]*.001
        self.assertEqual(inferred_rotation(sample),180)

if __name__=='__main__':unittest.main()
