import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import LogisticRegression

from moveai_core import evaluate
from runtime_model import predict_model
from score_calibration import calibrated_score, sigmoid_probability
from calibrate_scores import stored_folds
from test_decision import curl_sample, detector


ROOT = Path(__file__).resolve().parents[1] / 'modelos_rapidos'


def calibrated_detector(p_correct=.9, supported=True):
    learned = detector(error=p_correct < .5)
    learned['calibration'] = {'available': True, 'p_correct': p_correct, 'p_error': 1-p_correct,
                              'green_threshold': .75, 'red_threshold': .75,
                              'green_supported': supported, 'camera_calibrated': False}
    return learned


class SemaphoreTests(unittest.TestCase):
    def run_case(self, learned=None, sample=None, view='frente'):
        with patch('moveai_core.extract_video', return_value=copy.deepcopy(sample or curl_sample())), \
             patch('moveai_core.predict_model', return_value=copy.deepcopy(learned or calibrated_detector())), \
             patch('moveai_core.predict_subtypes', return_value={'suggestions': []}):
            return evaluate('clip.mp4', 'elbow_flexion', view, 'automatica', 120)

    def test_exact_75_percent_is_green_even_without_profile_measure(self):
        result = self.run_case(calibrated_detector(.75))
        self.assertEqual(result['color'], 'verde')
        self.assertIn('clasificada como correcta', result['title'])
        self.assertFalse(result['measurement_ready'])
        self.assertFalse(result['learned']['camera_validated'])

    def test_just_below_cut_is_yellow(self):
        result = self.run_case(calibrated_detector(.74999))
        self.assertEqual(result['color'], 'amarillo')
        self.assertNotIn('clasificada como correcta', result['title'])

    def test_exact_75_percent_incorrect_is_red(self):
        result = self.run_case(calibrated_detector(.25))
        self.assertEqual(result['color'], 'rojo')
        self.assertIn('clasificada como incorrecta', result['title'])
        self.assertEqual(result['deviations'], [])

    def test_new_calibrated_cut_replaces_the_old_detector_cut(self):
        learned = calibrated_detector(.8)
        learned.update(error=True, near_threshold=True, threshold=.17, p_error=.20)
        self.assertEqual(self.run_case(learned)['color'], 'verde')

    def test_quality_failures_never_receive_green_despite_high_score(self):
        cases = [curl_sample(), curl_sample(), curl_sample(), curl_sample(0)]
        cases[0]['coverage'] = .5
        cases[1]['multiple_people'] = 1
        cases[2]['visibility'][:, 16] = .2
        for sample in cases:
            with self.subTest(coverage=sample['coverage'], people=sample['multiple_people']):
                result = self.run_case(calibrated_detector(.99), sample)
                self.assertEqual(result['color'], 'amarillo')
                self.assertFalse(result['prediction_ready'])
                self.assertTrue(result['decision_reason'])

    def test_out_of_domain_never_receives_green(self):
        learned = calibrated_detector(.99)
        learned.update(out_of_domain=True, preliminary_available=False, reason='Fuera del rango de entrenamiento.')
        self.assertEqual(self.run_case(learned)['color'], 'amarillo')

    def profile_sample(self, flexion):
        sample = curl_sample(flexion)
        flex = np.radians(flexion * np.sin(np.linspace(0, np.pi, 64)))
        sample['image'][:, 16] = sample['image'][:, 14] + np.column_stack([.5*np.sin(flex), .5*np.cos(flex)])
        return sample

    def test_measured_incomplete_range_overrides_high_correct_score(self):
        result = self.run_case(calibrated_detector(.99), self.profile_sample(60), 'perfil')
        self.assertEqual(result['color'], 'rojo')
        self.assertEqual(result['decision_source'], 'measures')

    def test_borderline_measure_prevents_green(self):
        result = self.run_case(calibrated_detector(.99), self.profile_sample(115), 'perfil')
        self.assertEqual(result['color'], 'amarillo')
        self.assertEqual(result['decision_source'], 'measures_borderline')

    def test_unsupported_green_or_missing_calibration_never_is_green(self):
        self.assertEqual(self.run_case(calibrated_detector(.99, supported=False))['color'], 'amarillo')
        self.assertEqual(self.run_case(detector())['color'], 'amarillo')


class CalibrationArtifactTests(unittest.TestCase):
    def test_portable_calibration_matches_sklearn_on_real_scores(self):
        for directory in ROOT.iterdir():
            with self.subTest(exercise=directory.name):
                model = json.loads((directory / 'modelo.json').read_text())
                calibration = json.loads((directory / 'calibracion_score.json').read_text())
                with np.load(directory / 'predicciones_oof.npz', allow_pickle=False) as scores:
                    p, y, groups = scores['prob'], scores['y'], scores['subject']
                    reference = LogisticRegression(C=1e6, max_iter=2000).fit(p[:, None], y)
                    portable = sigmoid_probability(p, calibration['coefficient'], calibration['intercept'])
                    np.testing.assert_allclose(portable, reference.predict_proba(p[:, None])[:, 1], atol=1e-12)
                    result = calibrated_score(model, .2, directory / 'calibracion_score.json')
                    self.assertTrue(result['available'])
                    self.assertFalse(result['camera_calibrated'])
                    for train, test in stored_folds(model, groups, y):
                        self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_changed_weights_cannot_reuse_old_calibration(self):
        directory = ROOT / 'elbow_flexion'
        model = json.loads((directory / 'modelo.json').read_text())
        model['trees'][0]['p_error'][-1] += .001
        result = calibrated_score(model, .2, directory / 'calibracion_score.json')
        self.assertFalse(result['available'])
        self.assertIn('modelo cambió', result['reason'])

    def test_corrupt_calibration_returns_reason_instead_of_crashing(self):
        directory = ROOT / 'squat'
        model = json.loads((directory / 'modelo.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'calibracion_score.json'
            path.write_text('{broken')
            result = calibrated_score(model, .2, path)
            self.assertFalse(result['available'])
            self.assertTrue(result['reason'])

    def test_real_runtime_returns_complementary_calibrated_probabilities(self):
        for exercise in ['squat', 'inline_lunge', 'shoulder_abduction', 'elbow_flexion']:
            with self.subTest(exercise=exercise):
                prediction = predict_model(curl_sample()['world'], exercise, root=ROOT, view='frente', target=120)
                calibration = prediction['calibration']
                self.assertTrue(calibration['available'])
                self.assertAlmostEqual(calibration['p_correct'] + calibration['p_error'], 1.)
                self.assertFalse(prediction['camera_validated'])


if __name__ == '__main__':
    unittest.main()
