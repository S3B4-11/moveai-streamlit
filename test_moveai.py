"""Regresiones del fallo de extremidad, visibilidad, modelos y sesiones."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from analysis import analyze
from camera_model import features,raw_score,fuse,predict,FEATURE_NAMES
from feedback import save_review,export_reviews,review_rows
from geometry import SCHEMA
from rgb_dataset import parse_label
from web_session import SessionWorkspace

def fixture(exercise="elbow_flexion",peak=125,bilateral=False):
    times=np.arange(0,6.5,1/12);n=len(times)
    world=np.zeros((n,33,3),float)
    for j,v in {11:[-.2,-1,0],12:[.2,-1,0],23:[-.15,0,0],24:[.15,0,0],
                13:[-.2,-.5,0],14:[.2,-.5,0],15:[-.2,0,0],16:[.2,0,0],
                25:[-.15,.5,0],26:[.15,.5,0],27:[-.15,1,0],28:[.15,1,0]}.items():world[:,j]=v
    position=(times-.25)%3;phase=np.sin(np.pi*np.clip(position/2.5,0,1));phase[(position>2.5)|(times<.25)]=0
    f=np.deg2rad(peak*phase)
    for side in (["izquierda","derecha"] if bilateral else ["derecha"]):
        s,e,w,h,k,a=(11,13,15,23,25,27) if side=="izquierda" else (12,14,16,24,26,28)
        segment=np.column_stack([.5*np.sin(f),.5*np.cos(f),np.zeros(n)])
        if exercise=="elbow_flexion":world[:,w]=world[:,e]+segment
        elif exercise=="shoulder_abduction":world[:,e]=world[:,s]+segment;world[:,w]=world[:,e]+segment
        else:world[:,a]=world[:,k]+segment
    return dict(world=world,image=world[:,:,:2].copy(),visibility=np.ones((n,33)),times=times,
                detected=np.ones(n,bool),ambiguous=np.zeros(n,bool),coverage=1.,duration=float(times[-1]),
                pose_version="fixture",extractor_id="fixture",pipeline=SCHEMA,previews=[])

class MotionTests(unittest.TestCase):
    def test_active_arm_complete(self):
        sample=fixture();sample["visibility"][:,[11,13,15]]=0
        r=analyze(sample,"elbow_flexion")
        self.assertEqual(r["color"],"verde");self.assertEqual(len(r["repetitions"]),2)
        self.assertEqual([t["side"] for t in r["tracks"]],["derecha"])
    def test_incomplete_range(self):
        r=analyze(fixture(peak=60),"elbow_flexion")
        self.assertEqual(r["color"],"rojo")
        self.assertTrue(all(rep["checks"][0]["status"]=="fuera_de_rango" for rep in r["repetitions"]))
    def test_static_not_green(self):self.assertEqual(analyze(fixture(peak=0),"elbow_flexion")["color"],"amarillo")
    def test_cutoff_not_green(self):
        sample=fixture()
        for k in ["world","image","visibility","times","detected","ambiguous"]:sample[k]=sample[k][:18]
        self.assertEqual(analyze(sample,"elbow_flexion")["color"],"amarillo")
    def test_occluded_not_green(self):
        sample=fixture();sample["visibility"][10:26,16]=0
        self.assertNotEqual(analyze(sample,"elbow_flexion")["color"],"verde")
    def test_bad_depth_not_false_error(self):
        sample=fixture();sample["world"][:,16,2]=2
        r=analyze(sample,"elbow_flexion");self.assertEqual(r["color"],"verde")
        self.assertEqual(r["tracks"][0]["source"],"2D observable")
    def test_bilateral_count(self):
        r=analyze(fixture(bilateral=True),"elbow_flexion")
        self.assertEqual(len(r["repetitions"]),2);self.assertTrue(all(x["side"]=="ambos" for x in r["repetitions"]))
    def test_personal_goal_and_features(self):
        sample=fixture("shoulder_abduction",peak=95)
        self.assertEqual(analyze(sample,"shoulder_abduction",target=90)["color"],"verde")
        self.assertEqual(analyze(sample,"shoulder_abduction",target=150)["color"],"rojo")
        first,_=features(sample,"shoulder_abduction");sample["target"]=150
        second,_=features(sample,"shoulder_abduction");np.testing.assert_array_equal(first,second)
        self.assertEqual(sample["target"],150)
        self.assertEqual(len(first),len(FEATURE_NAMES))
    def test_large_trunk_compensation(self):
        sample=fixture();theta=np.deg2rad(45*np.sin(np.pi*np.clip(((sample["times"]-.25)%3)/2.5,0,1)))
        for i,t in enumerate(theta):
            rot=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
            sample["image"][i,[11,12,13,14,15,16]]=sample["image"][i,[11,12,13,14,15,16]]@rot.T
        r=analyze(sample,"elbow_flexion");self.assertEqual(r["color"],"rojo")
        self.assertTrue(any(c["key"]=="compensacion_tronco" and c["status"]=="fuera_de_rango" for rep in r["repetitions"] for c in rep["checks"]))
    def test_initial_hidden_torso_not_false_compensation(self):
        sample=fixture();sample["visibility"][:4,[23,24]]=0
        sample["image"][:4,[23,24],1]=-2
        r=analyze(sample,"elbow_flexion")
        self.assertEqual(r["color"],"verde")
    def test_implausible_torso_not_technique_error(self):
        sample=fixture();sample["image"][:,[11,12],1]=0
        sample["image"][:,[11,12],0]-=1
        self.assertEqual(analyze(sample,"elbow_flexion")["color"],"amarillo")

class PipelineTests(unittest.TestCase):
    def test_conflicting_evidence(self):
        self.assertEqual(fuse("verde",.1,True),("amarillo","evidencia_en_conflicto"))
        self.assertEqual(fuse("rojo",.9,True,True),("amarillo","evidencia_en_conflicto"))
        self.assertEqual(fuse("rojo",.8,True,False),("verde","modelo_rgb"))
        self.assertEqual(fuse("verde",.6,True),("verde","criterios_observables"))
        self.assertEqual(fuse("verde",.35,True),("amarillo","modelo_incierto"))
    def test_old_models_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"elbow_flexion";p.mkdir();(p/"modelo.json").write_text(json.dumps({"schema":"old_kinect"}))
            self.assertFalse(predict(fixture(),"elbow_flexion",d)["available"])
    def test_labels_and_exercises(self):
        row=parse_label("bicep_curl/good/front/subject_001_bicep_curl_good_front.mp4")
        self.assertEqual((row["subject"],row["error"]),("mendeley_rgb:1",0))
        self.assertIsNone(parse_label("shoulder_press/bad/front/subject_001.mp4"))
        self.assertIsNone(parse_label("squat/good/bad/subject_001.mp4"))
    def test_portable_models_and_subject_split(self):
        from train_camera import candidates,portable,model_score,group_folds
        rng=np.random.default_rng(7);x=rng.normal(size=(60,len(FEATURE_NAMES)));y=(x[:,0]+x[:,1]>0).astype(int)
        groups=np.repeat(np.arange(10).astype(str),6)
        for train,test in group_folds(x,y,groups,3):self.assertFalse(set(groups[train])&set(groups[test]))
        for name in ["random_forest","extra_trees","svm_c1","logistic_c01"]:
            model=candidates(1)[name];model.fit(x,y);export=portable(model)
            np.testing.assert_allclose([raw_score(export,v) for v in x[:10]],model_score(model,x[:10]),atol=2e-6)
            ensemble=dict(kind="ensemble",members=[export,export],member_calibration=dict(coefficient=1.2,intercept=-.3))
            expected=1/(1+np.exp(-(1.2*model_score(model,x[:10])-.3)))
            np.testing.assert_allclose([raw_score(ensemble,v) for v in x[:10]],expected,atol=2e-6)
        first=candidates(2)["random_forest"];second=candidates(3)["random_forest"]
        first.fit(x,y);second.fit(x,y)
        model=dict(kind="ensemble",members=[portable(first),portable(second)],score_calibration=dict(coefficient=2.3,intercept=-.4))
        mean_score=(model_score(first,x[:10])+model_score(second,x[:10]))/2
        expected=1/(1+np.exp(-(2.3*mean_score-.4)))
        np.testing.assert_allclose([raw_score(model,v) for v in x[:10]],expected,atol=2e-6)
    def test_reviews_roundtrip_and_consent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);video=root/"video.mp4";video.write_bytes(b"fixture")
            sample=fixture();analyze(sample,"elbow_flexion")
            with self.assertRaises(ValueError):save_review(video,sample,"elbow_flexion","P1",True,"coach",False,True,root/"reviews")
            save_review(video,sample,"elbow_flexion","P1",True,"coach",True,True,root/"reviews")
            archive=export_reviews(root/"reviews");rows=review_rows(archive,"fixture")
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]["error"],0)
            np.testing.assert_array_equal(rows[0]["sample"]["world"],sample["world"])
            with self.assertRaises(ValueError):review_rows(archive,"another-extractor")
    def test_session_invalidation_isolation(self):
        a,b=SessionWorkspace(),SessionWorkspace()
        try:
            a.set_input(b"abc",".mp4","squat","automatica",80);a.result={"color":"verde"}
            a.set_input(b"abc",".mp4","squat","automatica",80);self.assertIsNotNone(a.result)
            a.set_input(b"abc",".mp4","squat","automatica",120);self.assertIsNone(a.result)
            self.assertIsNone(b.video);a.set_input(None,".mp4","squat","automatica",80);self.assertIsNone(a.video)
        finally:a.close();b.close()
    def test_streamlit_login_and_results(self):
        from streamlit.testing.v1 import AppTest
        from moveai_core import diagnostic
        app=AppTest.from_file(str(Path(__file__).parent/"streamlit_app.py"),default_timeout=20)
        app.secrets["app_password"]="test-pass"
        app.run();self.assertFalse(app.exception)
        app.text_input[0].set_value("test-pass");app.button[0].click().run();self.assertFalse(app.exception)
        workspace=app.session_state["workspace"];workspace.result=analyze(fixture(),"elbow_flexion")
        workspace.result["model"]={"available":False,"reason":"criterios"}
        with patch.object(SessionWorkspace,"set_input",return_value=None):app.run()
        self.assertFalse(app.exception);self.assertTrue(len(app.success)>0)
        json.dumps(diagnostic(workspace.result),allow_nan=False)
        workspace.result["model"]={"available":True,"p_correct":.9,"scope":"Estimación de cámara; prueba de interfaz.","metrics":{}}
        with patch.object(SessionWorkspace,"set_input",return_value=None):app.run()
        self.assertFalse(app.exception)

if __name__=="__main__":unittest.main()
