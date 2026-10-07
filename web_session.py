"""Separación de videos, revisiones y resultados entre sesiones."""
from pathlib import Path
import hashlib
import tempfile

class SessionWorkspace:
    def __init__(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="moveai_")
        self.root=Path(self.temporary.name);self.video=None;self.digest=None;self.key=None;self.result=None
    @property
    def reviews(self):return self.root/"reviews"
    def set_input(self,data,suffix,exercise,side,target):
        if not data:
            if self.video:self.video.unlink(missing_ok=True)
            self.video=self.digest=self.key=self.result=None;return
        if len(data)>150*1024*1024:raise ValueError("El video supera 150 MB.")
        digest=hashlib.sha256(data).hexdigest();key=(digest,exercise,side,float(target))
        if key!=self.key:self.result=None
        if digest!=self.digest:
            if self.video:self.video.unlink(missing_ok=True)
            ext=suffix.lower() if suffix.lower() in [".mp4",".mov",".avi",".webm",".m4v"] else ".mp4"
            self.video=self.root/("video"+ext);self.video.write_bytes(data);self.digest=digest
        self.key=key
    def close(self):self.result=None;self.temporary.cleanup()
