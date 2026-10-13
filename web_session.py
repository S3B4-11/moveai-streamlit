"""Separación de videos, revisiones y resultados entre sesiones."""
from pathlib import Path
import hashlib
import tempfile

class SessionWorkspace:
    def __init__(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="moveai_")
        self.root=Path(self.temporary.name);self.video=None;self.digest=None;self.key=None;self.result=None
        self.rendered_video=None;self.render_error=None;self.normalized_video=None
        self.supports_person_selection=True
    @classmethod
    def upgrade(cls,existing):
        """Conserva el video y las revisiones al recargar desde una clase anterior."""
        upgraded=cls();upgraded.temporary.cleanup()
        upgraded.__dict__.update(existing.__dict__)
        upgraded.supports_person_selection=True
        return upgraded
    @property
    def reviews(self):return self.root/"reviews"
    def clear_result(self):
        self.result=None;self.render_error=None
        if self.rendered_video:self.rendered_video.unlink(missing_ok=True)
        self.rendered_video=None
        if self.normalized_video and self.normalized_video.is_relative_to(self.root):self.normalized_video.unlink(missing_ok=True)
        self.normalized_video=None
    def set_input(self,data,suffix,exercise,side,target,rotation="auto",person_target=None):
        if not data:
            if self.video:self.video.unlink(missing_ok=True)
            self.clear_result();self.video=self.digest=self.key=None;return
        if len(data)>150*1024*1024:raise ValueError("El video supera 150 MB.")
        selected=None if person_target is None else tuple(float(v) for v in person_target)
        digest=hashlib.sha256(data).hexdigest();key=(digest,exercise,side,float(target),str(rotation),selected)
        if key!=self.key:self.clear_result()
        if digest!=self.digest:
            if self.video:self.video.unlink(missing_ok=True)
            ext=suffix.lower() if suffix.lower() in [".mp4",".mov",".avi",".webm",".m4v"] else ".mp4"
            self.video=self.root/("video"+ext);self.video.write_bytes(data);self.digest=digest
        self.key=key
    def close(self):self.clear_result();self.temporary.cleanup()
