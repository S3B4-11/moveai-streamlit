"""Archivos de una sola sesión; los resultados se invalidan al cambiar la entrada."""
from pathlib import Path
import hashlib
import json
import tempfile


class SessionWorkspace:
    def __init__(self):
        self._temporary = tempfile.TemporaryDirectory(prefix='moveai_session_')
        self.root = Path(self._temporary.name)
        self.video = None
        self.digest = None
        self.analysis_key = None
        self.result = None

    @property
    def reviews(self):
        return self.root / 'reviews'

    def set_input(self, video_bytes, suffix, exercise, view, side, target):
        if not video_bytes:
            self.clear_video()
            return
        if len(video_bytes) > 150 * 1024 * 1024:
            raise ValueError('El video supera 150 MB. Recórtalo a una repetición.')
        digest = hashlib.sha256(video_bytes).hexdigest()
        key = (digest, exercise, view, side, float(target))
        if key != self.analysis_key:
            self.result = None
        if digest != self.digest:
            self.clear_video()
            suffix = suffix.lower() if suffix.lower() in ('.mp4', '.mov', '.avi', '.webm', '.m4v') else '.mp4'
            self.video = self.root / ('repetition' + suffix)
            self.video.write_bytes(video_bytes)
            self.digest = digest
        self.analysis_key = key

    def clear_video(self):
        if self.video:
            self.video.unlink(missing_ok=True)
        self.video = self.digest = self.analysis_key = self.result = None

    def close(self):
        self.result = None
        self._temporary.cleanup()


def model_summary(root):
    rows = []
    from geometry import EXERCISES
    for exercise, label in EXERCISES.items():
        path = Path(root) / exercise / 'modelo.json'
        if not path.exists():
            continue
        model = json.loads(path.read_text())
        metrics = model['metrics']
        rows.append({'Ejercicio': label,
                     'Accuracy balanceada': f"{100 * metrics['balanced_accuracy']:.1f}%",
                     'Errores detectados': f"{100 * metrics['sensitivity']:.1f}%",
                     'Correctas sin alarma': f"{100 * metrics['specificity']:.1f}%",
                     'Cámara validada': 'Sí' if model.get('camera_validated') else 'Pendiente'})
    return rows
