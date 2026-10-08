"""Regresiones de visibilidad, video reproducible y aislamiento de sesión."""
import json
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np
from analysis import analyze
from test_moveai import fixture
from video_overlay import COLORS,VideoOverlay,render_video
from web_session import SessionWorkspace

def visual_fixture(peak=125):
    result=analyze(fixture(peak=peak),"elbow_flexion")
    # Coordenadas de dibujo dentro de un fotograma de prueba vertical.
    result["sample"]["image"]=result["sample"]["image"]*np.array([.2,.3])+np.array([.23,.55])
    return result

def coloured_pixels(frame,color):
    rgb=np.asarray(color[::-1]);delta=np.max(abs(frame[110:].astype(int)-rgb),axis=2)
    return int(np.count_nonzero(delta<15))

class OverlayTests(unittest.TestCase):
    def test_correct_incorrect_and_static_colours(self):
        for peak,color in [(125,"verde"),(60,"rojo"),(0,"amarillo")]:
            result=visual_fixture(peak)
            time=result["repetitions"][0]["peak_time"] if result["repetitions"] else 1.
            frame=VideoOverlay(result,360).draw(np.full((640,360,3),30,np.uint8),time)
            self.assertGreater(coloured_pixels(frame,COLORS[color]),20)

    def test_occlusion_and_large_gaps_do_not_draw_a_pose(self):
        result=visual_fixture();result["sample"]["detected"][:]=False
        overlay=VideoOverlay(result,360)
        self.assertIsNone(overlay.pose_at(1.)[0])
        frame=overlay.draw(np.full((640,360,3),30,np.uint8),1.)
        self.assertEqual(coloured_pixels(frame,COLORS["verde"]),0)
        result=visual_fixture();result["sample"]["times"][20:]+=1.
        overlay=VideoOverlay(result,360)
        self.assertIsNone(overlay.pose_at(2.)[0])

    def test_conflicting_model_cannot_paint_certainty_in_green(self):
        result=visual_fixture();result.update(color="amarillo",decision_source="evidencia_en_conflicto")
        frame=VideoOverlay(result,360).draw(np.full((640,360,3),30,np.uint8),result["repetitions"][0]["peak_time"])
        self.assertEqual(coloured_pixels(frame,COLORS["verde"]),0)
        self.assertGreater(coloured_pixels(frame,COLORS["amarillo"]),20)

    def test_generated_video_is_invalidated_and_sessions_are_isolated(self):
        a,b=SessionWorkspace(),SessionWorkspace()
        try:
            a.set_input(b"source",".mp4","squat","automatica",100)
            a.result={"color":"verde"};a.rendered_video=a.root/"analizado.mp4";a.rendered_video.write_bytes(b"A")
            b.rendered_video=b.root/"analizado.mp4";b.rendered_video.write_bytes(b"B")
            a.set_input(b"source",".mp4","squat","automatica",100)
            self.assertIsNotNone(a.result);self.assertTrue(a.rendered_video.exists())
            old=a.rendered_video
            a.set_input(b"source",".mp4","squat","automatica",120)
            self.assertIsNone(a.result);self.assertIsNone(a.rendered_video);self.assertFalse(old.exists())
            self.assertEqual(b.rendered_video.read_bytes(),b"B")
        finally:a.close();b.close()

    def test_streamlit_generates_once_and_keeps_analysis_if_render_fails(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        class Upload(io.BytesIO):
            name="ejercicio.mp4"
            def __init__(self):super().__init__(b"source video")
        def encoded(source,result,destination):
            Path(destination).write_bytes(b"rendered video");return Path(destination)
        result=visual_fixture();result["model"]={"available":False,"reason":"criterios"}
        app=AppTest.from_file(str(Path(__file__).parent/"streamlit_app.py"),default_timeout=20)
        app.secrets["app_password"]="test-pass";app.session_state["authenticated"]=True
        with patch.object(st,"file_uploader",return_value=Upload()),patch("moveai_core.evaluate",return_value=result),patch("video_overlay.render_video",side_effect=encoded) as render:
            app.run()
            button=next(x for x in app.button if x.label=="Analizar video")
            button.click().run()
            self.assertFalse(app.exception);self.assertEqual(render.call_count,1)
            self.assertEqual(len(app.get("video")),2)
            self.assertIn("Descargar video analizado",[x.proto.label for x in app.get("download_button")])
            app.run();self.assertEqual(render.call_count,1)
            workspace=app.session_state["workspace"]
            previous=workspace.rendered_video
            with patch("video_overlay.render_video",side_effect=RuntimeError("prueba")),patch("logging.exception"):
                next(x for x in app.button if x.label=="Analizar video").click().run()
            self.assertFalse(app.exception);self.assertIsNotNone(workspace.result)
            self.assertIsNone(workspace.rendered_video);self.assertFalse(previous.exists())
            self.assertTrue(any("resultado del análisis sigue disponible" in x.value for x in app.info))
            workspace.close()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg requerido para la exportación")
    def test_mp4_h264_preserves_duration_and_audio(self):
        import cv2
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/"input.avi";sound=root/"audio.wav";source=root/"input.mp4"
            writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*"MJPG"),12.,(360,640))
            self.assertTrue(writer.isOpened())
            for _ in range(78):writer.write(np.full((640,360,3),30,np.uint8))
            writer.release()
            with wave.open(str(sound),"wb") as f:
                f.setparams((1,2,8000,0,"NONE","not compressed"));f.writeframes(np.zeros(52000,dtype=np.int16).tobytes())
            subprocess.run([shutil.which("ffmpeg"),"-hide_banner","-loglevel","error","-y","-i",str(raw),"-i",str(sound),
                            "-c:v","libx264","-threads","2","-pix_fmt","yuv420p","-c:a","aac",str(source)],check=True)
            output=render_video(source,visual_fixture(),root/"output.mp4")
            info=json.loads(subprocess.check_output([shutil.which("ffprobe"),"-v","error","-show_streams","-show_format","-of","json",str(output)]))
            streams={s["codec_type"]:s for s in info["streams"]}
            self.assertEqual(streams["video"]["codec_name"],"h264")
            self.assertEqual(streams["video"]["pix_fmt"],"yuv420p")
            self.assertEqual(streams["audio"]["codec_name"],"aac")
            self.assertAlmostEqual(float(info["format"]["duration"]),6.5,delta=.12)
            cap=cv2.VideoCapture(str(output));cap.set(cv2.CAP_PROP_POS_MSEC,1500);ok,frame=cap.read();cap.release()
            self.assertTrue(ok);self.assertGreater(coloured_pixels(frame,COLORS["verde"]),10)
            self.assertFalse(list(root.glob("render_*")))

if __name__=="__main__":unittest.main()
