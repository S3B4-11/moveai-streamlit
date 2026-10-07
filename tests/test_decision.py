import copy
import unittest
from unittest.mock import patch
import numpy as np
from moveai_core import evaluate, markdown_result
from test_pipeline import skeleton


def curl_sample(max_flex=125):
    world = skeleton()
    flex = np.radians(max_flex * np.sin(np.linspace(0, np.pi, 64)))
    # Movimiento fuera del plano frontal: la 3D conserva la flexión; su proyección no.
    world[:,16] = world[:,14] + np.column_stack([np.zeros(64), .5*np.cos(flex), -.5*np.sin(flex)])
    return {'world':world, 'image':world[:,:,:2].copy(), 'visibility':np.ones((64,33)),
            'coverage':1., 'multiple_people':0, 'pose_version':'fixture', 'thumbnail':None}


def detector(error=False, near=False, available=True):
    return {'available':True, 'preliminary_available':available, 'usable':False,
            'camera_validated':False, 'error':error, 'near_threshold':near,
            'p_error':.8 if error else .1, 'reason':'Cámara pendiente de validación.'}


class DecisionTests(unittest.TestCase):
    def run_case(self, sample=None, learned=None, **options):
        sample = copy.deepcopy(curl_sample() if sample is None else sample)
        learned = copy.deepcopy(detector() if learned is None else learned)
        with patch('moveai_core.extract_video', return_value=sample), \
             patch('moveai_core.predict_model', return_value=learned), \
             patch('moveai_core.predict_subtypes', return_value={'suggestions':[]}):
            return evaluate('clip.mp4', 'elbow_flexion', options.get('view','frente'),
                            options.get('side','automatica'), 120)

    def test_frontal_measure_missing_does_not_hide_preliminary_model(self):
        result = self.run_case()
        self.assertFalse(result['measurement_ready'])
        self.assertTrue(result['prediction_ready'])
        self.assertEqual(result['decision_source'],'model_preliminary')
        self.assertIn('parece correcta',result['title'])
        self.assertFalse(result['learned']['camera_validated'])
        self.assertFalse(result['learned']['usable'])
        self.assertIn('requiere vista de perfil',markdown_result(result))

    def test_incorrect_model_result_also_appears_without_invented_subtype(self):
        result = self.run_case(learned=detector(error=True))
        self.assertIn('posible ejecución incorrecta',result['title'])
        self.assertEqual(result['deviations'],[])

    def test_near_threshold_is_not_reported_as_correct(self):
        result = self.run_case(learned=detector(near=True))
        self.assertEqual(result['decision_source'],'model_uncertain')
        self.assertIn('No concluyente',result['title'])

    def test_hidden_wrist_blocks_preliminary_classifier(self):
        sample = curl_sample();sample['visibility'][:,16]=.2
        result = self.run_case(sample)
        self.assertFalse(result['prediction_ready'])
        self.assertFalse(result['sample']['trainable'])
        self.assertIn('articulaciones',result['decision_reason'])

    def test_low_detection_coverage_blocks_model_and_measures(self):
        sample = curl_sample();sample['coverage']=.5
        result = self.run_case(sample)
        self.assertFalse(result['prediction_ready'])
        self.assertTrue(all(c['status']=='no_evaluable' for c in result['rules']['checks']))

    def test_second_person_blocks_model(self):
        sample = curl_sample();sample['multiple_people']=1
        result = self.run_case(sample)
        self.assertFalse(result['prediction_ready'])
        self.assertIn('más de una persona',result['decision_reason'])

    def test_static_pose_does_not_become_correct_even_if_detector_votes_correct(self):
        result = self.run_case(curl_sample(0))
        self.assertFalse(result['prediction_ready'])
        self.assertIn('No concluyente',result['title'])

    def test_wrong_side_has_a_specific_message_and_auto_selects_moving_arm(self):
        wrong = self.run_case(side='izquierda')
        self.assertFalse(wrong['prediction_ready'])
        self.assertIn('derecha sí se mueve',wrong['decision_reason'])
        auto = self.run_case()
        self.assertEqual(auto['sample']['side'],'derecha')
        self.assertTrue(auto['prediction_ready'])

    def test_visible_frontal_clip_can_be_reviewed_for_training(self):
        result = self.run_case()
        self.assertFalse(result['measurement_ready'])
        self.assertTrue(result['sample']['trainable'])

    def test_out_of_domain_does_not_become_a_positive_verdict(self):
        model=detector(available=False);model.update(out_of_domain=True,reason='Fuera del rango de entrenamiento.')
        result=self.run_case(learned=model)
        self.assertFalse(result['prediction_ready'])
        self.assertIn('No concluyente',result['title'])
        self.assertIn('Fuera del rango',result['decision_reason'])

    def test_measured_error_keeps_priority_over_positive_model(self):
        sample=curl_sample(60)
        # Vista lateral: colocar el movimiento en el plano medible, conservando los ángulos.
        flex=np.radians(60*np.sin(np.linspace(0,np.pi,64)))
        sample['image'][:,16]=sample['image'][:,14]+np.column_stack([.5*np.sin(flex),.5*np.cos(flex)])
        result=self.run_case(sample,view='perfil')
        self.assertEqual(result['decision_source'],'measures')
        self.assertIn('Revisar técnica',result['title'])
        self.assertEqual(result['deviations'][0]['key'],'rango_incompleto')


if __name__=='__main__':unittest.main()
