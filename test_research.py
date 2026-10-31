"""Pruebas de integración: máscaras, identidad, pesos y las tres redes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_moveai import fixture
from neural_features import encode,Normalizer,EDGES
from neural_models import build,fit,logits,save_bundle,load_bundle,probability
from research_experiment import grouped,classic,portable_classic,score_classic,calibrate
from camera_model import raw_score
from download_research import label

class ResearchTests(unittest.TestCase):
    def test_same_input_shapes_and_gradients(self):
        x,m=encode(fixture());self.assertEqual(x.shape,(64,33,10));self.assertTrue(np.isfinite(x).all())
        torch.set_num_threads(2)
        for family in ["gru","lstm","stgcn"]:
            model=build(family,dict(hidden=32,dropout=.2));out=model(torch.from_numpy(x[None]))
            self.assertEqual(tuple(out.shape),(1,));out.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    def test_hidden_points_stay_masked(self):
        sample=fixture();sample["visibility"][:,16]=0;sample["world"][:,16]=np.nan
        x,m=encode(sample);self.assertFalse(m[:,16,:5].any());self.assertFalse(x[:,16,:5].any())
        self.assertFalse(x[:,14,8].any())
    def test_ambiguity_not_imputed(self):
        sample=fixture();sample["ambiguous"][:]=True;x,m=encode(sample)
        self.assertFalse(m[:,:,:5].any());self.assertFalse(x[:,:,5].any())
    def test_normalizer_training_only_and_mask(self):
        x,m=encode(fixture());base=Normalizer().fit(x[None],m[None]);before=copy.deepcopy(base.as_dict())
        large=x.copy();large[:,:,:5]*=1000;base.transform(large[None],m[None]);self.assertEqual(before,base.as_dict())
        m[:,16,:]=False;self.assertFalse(base.transform(x,m)[:,16,:].any())
    def test_no_subject_leak_three_roles(self):
        y=np.tile([0,1,0,1],15);groups=np.repeat(np.arange(15).astype(str),4)
        for a,b in grouped(y,groups,3,42):
            d,c=grouped(y[a],groups[a],3,91)[0]
            roles=[set(groups[a[d]]),set(groups[a[c]]),set(groups[b])]
            self.assertTrue(all(not roles[i]&roles[j] for i in range(3) for j in range(i)))
    def test_safe_weights_roundtrip_and_corruption(self):
        x,m=encode(fixture());n=Normalizer().fit(x[None],m[None]);config=dict(hidden=32,dropout=.2)
        for family in ["gru","lstm","stgcn"]:
            with tempfile.TemporaryDirectory() as folder:
                model=build(family,config);model.eval();expected=logits(model,n.transform(x[None],m[None]))
                save_bundle(folder,model,family,config,n,dict(coefficient=1,intercept=0))
                restored,norm,meta=load_bundle(folder);np.testing.assert_allclose(expected,logits(restored,norm.transform(x[None],m[None])),atol=1e-6)
                path=Path(folder)/"weights.npz";path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaises(ValueError):load_bundle(folder)
    def test_baselines_portable_equal(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(60,184));y=(x[:,0]>0).astype(int)
        for family,params in [("random_forest",dict(n_estimators=10,max_depth=4,min_samples_leaf=2)),
            ("extra_trees",dict(n_estimators=10,max_depth=4,min_samples_leaf=2)),("svm",dict(C=1)),("logistic",dict(C=1))]:
            model=classic(family,params,42);model.fit(x,y);portable=portable_classic(model)
            np.testing.assert_allclose(score_classic(model,x[:10]),[raw_score(portable,v) for v in x[:10]],atol=1e-6)
    def test_original_page_bytes(self):
        reference=json.loads((Path(__file__).parent/"preservation.json").read_text())
        for name,digest in reference.items():self.assertEqual(hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest(),digest,name)
    def test_unapproved_router_fallback(self):
        from inference_router import predict
        with patch("neural_runtime.predict",return_value=dict(available=False,reason="not approved")),patch("inference_router.rgb4_predict",return_value=dict(available=True,p_error=.2)):
            self.assertEqual(predict(fixture(),"elbow_flexion")["p_error"],.2)
    def test_four_official_dataset_labels(self):
        for name,exercise in [("bicep_curl","elbow_flexion"),("squat","squat"),("push_up","push_up"),("shoulder","shoulder_press")]:
            parsed=label(f"Dataset Exercise Quality-wise/{name}/bad/side/subject_006_{name}_bad_side.mp4")
            self.assertEqual((parsed["exercise"],parsed["error"],parsed["subject"]),(exercise,1,"mendeley_rgb:6"))
        self.assertIsNone(label("squat/bad/good/subject_002.mp4"))
    def test_fit_best_checkpoint_finite(self):
        xs=[];ms=[];ys=[]
        for i in range(16):
            x,m=encode(fixture(peak=60 if i%2 else 125));xs.append(x);ms.append(m);ys.append(i%2)
        x,m,y=np.array(xs),np.array(ms),np.array(ys)
        model,norm,info=fit("gru",dict(hidden=32,dropout=.2,lr=.001),x,m,y,np.arange(12),np.arange(12,16),max_epochs=2,patience=2)
        self.assertTrue(np.isfinite(logits(model,norm.transform(x,m))).all());self.assertIn(info["best_epoch"],[1,2])

if __name__=="__main__":unittest.main()
