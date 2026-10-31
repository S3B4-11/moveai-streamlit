"""Regresiones de press, oclusión, fases, ruta API e interfaz."""
import copy,io,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from geometry import SCHEMA
from press_analysis import analyze_press
from decision_policy import apply_policy

def press_fixture(peak=168,elbow_top=10,elbow_bottom=90,lean=0):
    times=np.arange(0,7,1/8);n=len(times);image=np.zeros((n,33,2));image[:]=[.5,.6]
    for j,x,y in [(11,.4,.4),(12,.6,.4),(23,.4,.65),(24,.6,.65)]:image[:,j]=[x,y]
    phase=((times-.3)%3)/2.5;motion=np.sin(np.pi*np.clip(phase,0,1));motion[(phase>1)|(times<.3)]=0
    arm=np.deg2rad(80+(peak-80)*motion);flex=np.deg2rad(elbow_bottom+(elbow_top-elbow_bottom)*motion)
    for s,e,w,sign in [(11,13,15,-1),(12,14,16,1)]:
        image[:,e]=image[:,s]+np.column_stack([sign*.13*np.sin(arm),.13*np.cos(arm)])
        image[:,w]=image[:,e]+np.column_stack([sign*.13*np.sin(arm+flex),.13*np.cos(arm+flex)])
    if lean:
        for i,phi in enumerate(np.deg2rad(lean*motion)):
            rot=np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
            for s,e,w,h in [(11,13,15,23),(12,14,16,24)]:image[i,[s,e,w]]=(image[i,[s,e,w]]-image[i,h])@rot.T+image[i,h]
    return dict(image=image,world=np.dstack([image,np.zeros((n,33))]),visibility=np.ones((n,33)),times=times,
                detected=np.ones(n,bool),ambiguous=np.zeros(n,bool),coverage=1.,duration=7.,previews=[],
                pose_version='fixture',extractor_id='fixture',pipeline=SCHEMA)

def result(s=None,**kwargs):
    r=analyze_press(press_fixture() if s is None else s,**kwargs);r['model']={'available':False,'reason':'Criterios visibles del press.'};return apply_policy(r)

class PressTests(unittest.TestCase):
    def test_initial_bent_elbows_are_normal(self):
        r=result();self.assertEqual(r['status'],'correct',r['repetitions']);self.assertGreaterEqual(r['counts']['verde'],2)
        self.assertTrue(all(x['side']=='ambos' for x in r['repetitions']))
    def test_extension_is_required_at_the_top(self):
        r=result(press_fixture(elbow_top=75));self.assertEqual(r['status'],'incorrect');self.assertIn('press_extension',[e['key'] for e in r['errors']])
    def test_elevation_requires_overhead_range(self):
        r=result(press_fixture(peak=112));self.assertEqual(r['status'],'incorrect');self.assertIn('press_elevacion',[e['key'] for e in r['errors']])
    def test_elbows_need_dynamic_extension(self):
        r=result(press_fixture(elbow_top=70,elbow_bottom=75));self.assertEqual(r['status'],'incorrect');self.assertIn('press_cambio_codo',[e['key'] for e in r['errors']])
    def test_trunk_swing(self):
        r=result(press_fixture(lean=45));self.assertEqual(r['status'],'incorrect');self.assertIn('compensacion_tronco',[e['key'] for e in r['errors']])
    def test_end_card_preserves_clear_cycles(self):
        s=press_fixture()
        for k in ('image','world','visibility','detected','ambiguous'):
            tail=np.repeat(s[k][-1:],32,axis=0);tail[:]=np.nan if k in ('image','world') else 0;s[k]=np.concatenate([s[k],tail])
        s['times']=np.arange(len(s['image']))/8;s['coverage']=float(s['detected'].mean());self.assertEqual(result(s)['status'],'correct')
    def test_missing_wrist_cannot_be_overridden(self):
        s=press_fixture();s['visibility'][:,[15,16]]=0;self.assertEqual(result(s)['status'],'unreadable')
    def test_internal_occlusion_blocks_green(self):
        s=press_fixture();s['visibility'][5:18,[15,16]]=0;s['visibility'][29:42,[15,16]]=0;self.assertNotEqual(result(s)['status'],'correct')
    def test_cutoff_does_not_prove_bad_range(self):
        s=press_fixture()
        for k in ('image','world','visibility','times','detected','ambiguous'):s[k]=s[k][:12]
        r=result(s);self.assertNotEqual(r['status'],'correct');self.assertFalse(r['errors'])
    def test_static_has_no_repetition(self):
        s=press_fixture()
        for k in ('image','world'):s[k][:]=s[k][0]
        r=result(s);self.assertEqual(r['status'],'unreadable');self.assertEqual(r['repetitions'],[])
    def test_hidden_selected_side_not_silently_replaced(self):
        s=press_fixture();s['visibility'][:,[11,13,15,23]]=0
        self.assertEqual(result(copy.deepcopy(s),side='derecha')['status'],'correct');self.assertEqual(result(s,side='izquierda')['status'],'unreadable')
    def test_foreshortened_projection_can_use_coherent_world(self):
        s=press_fixture()
        for e,w in [(13,15),(14,16)]:s['image'][:,w]=s['image'][:,e]+.001
        r=result(s);self.assertEqual(r['status'],'correct')
        self.assertTrue(all(t['source']=='3D estimada' for t in r['tracks']))
    def test_unstable_depth_cannot_replace_missing_projection(self):
        s=press_fixture()
        for e,w in [(13,15),(14,16)]:
            s['image'][:,w]=s['image'][:,e]+.001
            s['world'][::2,w,2]=10
        self.assertEqual(result(s)['status'],'unreadable')
    def test_readable_image_is_not_overruled_by_bad_depth(self):
        s=press_fixture();s['world'][:,:,2]=np.arange(len(s['times']))[:,None]*10
        self.assertTrue(all(t['source']=='2D observable' for t in result(s)['tracks']))
    def test_registry_replaces_the_public_exercise(self):
        from exercise_registry import EXERCISES,TARGETS
        self.assertNotIn('shoulder_abduction',EXERCISES);self.assertIn('shoulder_press',EXERCISES);self.assertEqual(TARGETS['shoulder_press'],150)
    def test_goal_validation(self):
        for v in [40,90,180,float('nan')]:
            with self.assertRaises(ValueError):analyze_press(press_fixture(),target=v)
    def test_api_skips_legacy_and_unapproved_models(self):
        from moveai_core import evaluate,diagnostic
        with patch('moveai_core.extract_video',return_value=press_fixture()),patch('moveai_core.predict') as model,patch('moveai_core.analyze') as legacy:
            r=evaluate('fixture.mp4','shoulder_press',rotation=0)
            self.assertEqual(r['status'],'correct');model.assert_not_called();legacy.assert_not_called()
            self.assertFalse(r['model']['available']);self.assertFalse(r['sample']['trainable']);json.dumps(diagnostic(r),allow_nan=False)
    def test_old_key_requires_reselection(self):
        from moveai_core import evaluate
        with patch('moveai_core.extract_video',return_value=press_fixture()):
            with self.assertRaisesRegex(ValueError,'Selecciona Press'):evaluate('fixture.mp4','shoulder_abduction',rotation=0)
    def test_ui_press_route(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        r=result()
        class Upload(io.BytesIO):name='press.mp4'
        app=AppTest.from_file(str(Path(__file__).parent/'streamlit_app.py'),default_timeout=20)
        app.secrets['app_password']='test-pass';app.session_state['authenticated']=True
        with patch.object(st,'file_uploader',return_value=Upload(b'press')),patch('moveai_core.evaluate',return_value=r) as evaluate,patch('video_overlay.render_video',side_effect=RuntimeError('test')),patch('logging.exception'):
            app.run();next(x for x in app.selectbox if x.label=='Ejercicio').select('shoulder_press').run()
            self.assertEqual(next(x for x in app.slider if x.label=='Elevación objetivo del brazo (grados)').value,150)
            next(x for x in app.button if x.label=='Analizar video').click().run()
            self.assertEqual(evaluate.call_args.args[1],'shoulder_press');self.assertFalse(app.exception)
            self.assertTrue(any(x.value=='Ejercicio realizado correctamente' for x in app.success));app.session_state['workspace'].close()
if __name__=='__main__':unittest.main()
