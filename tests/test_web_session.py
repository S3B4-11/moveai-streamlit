import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from web_session import SessionWorkspace, model_summary
from moveai_core import evaluate
from test_pipeline import skeleton


class WebSessionTests(unittest.TestCase):
    def test_review_rerun_keeps_result_but_input_changes_invalidate_it(self):
        workspace = SessionWorkspace()
        self.addCleanup(workspace.close)
        args = (b'video one', '.mp4', 'elbow_flexion', 'perfil', 'derecha', 120)
        workspace.set_input(*args)
        workspace.result = {'title': 'first result'}
        workspace.set_input(*args)
        self.assertIsNotNone(workspace.result)
        for index, value in [(0, b'video two'), (2, 'squat'), (3, 'frente'), (4, 'izquierda'), (5, 90)]:
            workspace.set_input(*args)
            workspace.result = {'title': 'old result'}
            changed = list(args)
            changed[index] = value
            workspace.set_input(*changed)
            self.assertIsNone(workspace.result)

    def test_sessions_do_not_share_files_or_reviews(self):
        first, second = SessionWorkspace(), SessionWorkspace()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        for workspace in (first, second):
            workspace.set_input(b'same video', '.mp4', 'squat', 'perfil', 'derecha', 80)
        self.assertNotEqual(first.reviews, second.reviews)
        self.assertNotEqual(first.video, second.video)
        first.clear_video()
        self.assertTrue(second.video.exists())

    def test_unvalidated_detector_cannot_replace_observed_criteria(self):
        coords = skeleton()
        flex = np.radians(125 * np.sin(np.linspace(0, np.pi, 64)))
        coords[:, 16, :2] = coords[:, 14, :2] + np.column_stack([.5*np.sin(flex), .5*np.cos(flex)])
        sample = {'world': coords, 'image': coords[:, :, :2], 'visibility': np.ones((64, 33)),
                  'coverage': 1., 'multiple_people': 0, 'pose_version': 'test', 'thumbnail': None}
        with patch('moveai_core.extract_video', return_value=sample), patch('moveai_core.predict_model',
                   return_value={'available': True, 'usable': False, 'error': True}):
            result = evaluate('unused.mp4', 'elbow_flexion', 'perfil', 'derecha', 120)
        self.assertTrue(result['measurement_ready'])
        self.assertEqual(result['title'], 'Sin desviaciones en los criterios medidos')

    def test_shipped_models_are_all_camera_pending(self):
        root = Path(__file__).resolve().parents[1] / 'modelos_rapidos'
        rows = model_summary(root)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row['Cámara validada'] == 'Pendiente' for row in rows))


if __name__ == '__main__':
    unittest.main()
