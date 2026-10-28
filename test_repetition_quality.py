"""Una portada no invalida ciclos legibles; una oclusión dentro sí."""
import copy
import unittest
import numpy as np
from test_moveai import fixture
from analysis import analyze
from decision_policy import apply_policy
from repetition_quality import repair_repetition_quality

class RepetitionQualityTests(unittest.TestCase):
    def sample(self,peak=125):
        s=fixture(peak=peak)
        for key in ('world','image','visibility','times','detected','ambiguous'):
            v=s[key];tail=np.repeat(v[-1:],40,axis=0)
            if key=='times':tail=v[-1]+np.arange(1,41)/12
            elif key in ('world','image'):tail[:]=np.nan
            elif key in ('visibility','detected','ambiguous'):tail[:]=0
            s[key]=np.concatenate([v,tail])
        s['coverage']=float(s['detected'].mean());return s
    def run_case(self,s):
        r=analyze(s,'elbow_flexion',side='derecha');r['model']={'available':False}
        return apply_policy(repair_repetition_quality(r))
    def test_end_card_does_not_erase_readable_repetitions(self):
        r=self.run_case(self.sample());self.assertEqual(r['status'],'correct')
        self.assertTrue(r['repetition_quality_repair']);self.assertFalse(r['quality']['ready'])
        self.assertFalse(r['sample']['trainable'])
    def test_visible_bad_range_still_red_with_end_card(self):
        r=self.run_case(self.sample(60));self.assertEqual(r['status'],'incorrect')
        self.assertTrue(any(e['key']=='rango_incompleto' for e in r['errors']))
    def test_internal_occlusion_remains_unassessed(self):
        s=self.sample();s['visibility'][8:23,16]=0
        r=self.run_case(s)
        self.assertTrue(any(x['color']=='amarillo' for x in r['repetitions']))
    def test_good_input_is_unchanged(self):
        r=analyze(fixture(),'elbow_flexion');before=copy.deepcopy(r['repetitions'])
        repair_repetition_quality(r);self.assertEqual(before,r['repetitions'])
    def test_all_hidden_cannot_become_green(self):
        s=self.sample();s['visibility'][:]=0
        self.assertEqual(self.run_case(s)['status'],'unreadable')

if __name__=='__main__':unittest.main()
