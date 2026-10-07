import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from feedback import save_review,export_reviews,load_reviews,import_reviews
from geometry import features,feature_names,rule_assessment
from train import splits,threshold


def skeleton():
    c=np.zeros((64,33,3),np.float32)
    for k,p in {11:[-.2,-1,0],12:[.2,-1,0],23:[-.15,0,0],24:[.15,0,0],
      13:[-.2,-.5,0],14:[.2,-.5,0],15:[-.2,0,0],16:[.2,0,0],
      25:[-.15,.5,0],26:[.15,.5,0],27:[-.15,1,0],28:[.15,1,0]}.items():c[:,k]=p
    return c


class PipelineTests(unittest.TestCase):
    def test_feature_rotation_and_scale_invariance(self):
        c=skeleton();c[:,26,2]=np.sin(np.linspace(0,np.pi,64))*.3
        theta=.7;r=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        for ex in ['squat','inline_lunge','elbow_flexion','shoulder_abduction']:
            a=features(c,ex);b=features((c@r)*3+4,ex)
            self.assertEqual(len(a),len(feature_names(ex)));np.testing.assert_allclose(a,b,atol=.02)

    def test_no_movement_is_not_evaluable(self):
        r=rule_assessment(skeleton()[:,:,:2],np.ones((64,33)),'elbow_flexion','perfil')
        self.assertFalse(r['evaluable'])

    def test_wrong_view_does_not_validate_primary(self):
        r=rule_assessment(skeleton()[:,:,:2],np.ones((64,33)),'elbow_flexion','frente')
        self.assertEqual(r['checks'][0]['status'],'no_evaluable')

    def test_one_hidden_joint_rejects_measurement(self):
        v=np.ones((64,33));v[:,14]=.1
        r=rule_assessment(skeleton()[:,:,:2],v,'elbow_flexion','perfil')
        self.assertEqual(r['checks'][0]['status'],'no_evaluable')

    def test_curl_full_and_partial_ranges(self):
        for max_flex,expected in [(125,'en_rango'),(60,'fuera_de_rango')]:
            c=skeleton()[:,:,:2];f=np.radians(max_flex*np.sin(np.linspace(0,np.pi,64)))
            c[:,16]=c[:,14]+np.column_stack([.5*np.sin(f),.5*np.cos(f)])
            r=rule_assessment(c,np.ones((64,33)),'elbow_flexion','perfil','derecha',120)
            self.assertTrue(r['evaluable']);self.assertEqual(r['checks'][0]['status'],expected)

    def test_shoulder_goal_is_configurable(self):
        c=skeleton()[:,:,:2];f=np.radians(95*np.sin(np.linspace(0,np.pi,64)))
        direction=np.column_stack([.5*np.sin(f),.5*np.cos(f)])
        c[:,14]=c[:,12]+direction;c[:,16]=c[:,14]+direction
        a=rule_assessment(c,np.ones((64,33)),'shoulder_abduction','frente','derecha',90)
        b=rule_assessment(c,np.ones((64,33)),'shoulder_abduction','frente','derecha',150)
        self.assertEqual(a['checks'][0]['status'],'en_rango');self.assertEqual(b['checks'][0]['status'],'fuera_de_rango')

    def test_lunge_uses_front_leg_not_mean_of_both(self):
        c=skeleton()[:,:,:2];c[:,26,0]+=.5*np.sin(np.linspace(0,np.pi,64))
        r=rule_assessment(c,np.ones((64,33)),'inline_lunge','perfil','derecha',85)
        self.assertEqual(r['checks'][0]['status'],'en_rango')
        self.assertFalse(any(a['key']=='asimetria' for a in r['checks']))

    def test_subjects_never_overlap(self):
        y=np.tile([0,1],12);g=np.repeat(np.arange(6),4);x=np.zeros((24,4))
        for a,b in splits(x,y,g):self.assertFalse(set(g[a]) & set(g[b]))

    def test_review_consent_duplicates_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);v=root/'test.mp4';v.write_bytes(b'test video')
            sample={'world':skeleton(),'image':skeleton()[:,:,:2],'visibility':np.ones((64,33)), 'trainable':True}
            args=[v,sample,'elbow_flexion','p01',False,[], 'revisor',True,True,root/'data']
            bad=args.copy();bad[7]=False
            with self.assertRaises(ValueError):save_review(*bad)
            first=save_review(*args);self.assertEqual(first,save_review(*args))
            self.assertEqual(len(load_reviews(root/'data')),1)
            altered=args.copy();altered[3]='p02'
            with self.assertRaises(ValueError):save_review(*altered)
            archive=export_reviews(root/'data');import_reviews(archive,root/'imported')
            items=load_reviews(root/'imported');self.assertEqual(len(items),1)
            self.assertEqual(items[0]['error'],1);self.assertEqual(items[0]['tags'],[])
            np.testing.assert_array_equal(items[0]['coords'],skeleton())

    def test_unknown_subtype_not_invented(self):
        # Etiquetar 'incorrecta' sin un subtipo está permitido y se conserva.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'v.mp4';p.write_bytes(b'v')
            sample={'world':skeleton(),'image':skeleton()[:,:,:2],'visibility':np.ones((64,33)), 'trainable':True}
            save_review(p,sample,'squat','same_friend',False,[], 'human',True,True,Path(d)/'root')
            self.assertEqual(load_reviews(Path(d)/'root')[0]['tags'],[])

if __name__=='__main__':unittest.main()
