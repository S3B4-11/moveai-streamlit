"""Video anotado con poses ya extraídas; exportación H.264 para el navegador."""
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from geometry import JOINTS
from feedback_presentation import repetition_color, repetition_label

COLORS={"verde":(34,197,94),"rojo":(239,68,68),"amarillo":(245,158,11),"neutro":(203,213,225)}
LABELS={"verde":"Ejecución correcta","rojo":"Necesita corrección","amarillo":"No se pudo evaluar","neutro":"Entre repeticiones"}

class VideoOverlay:
    def __init__(self,result,width):
        self.result=result;sample=result["sample"]
        self.times=np.asarray(sample["times"],float)
        self.points=np.asarray(sample["image"],float)
        self.visibility=np.asarray(sample["visibility"],float)
        self.detected=np.asarray(sample["detected"],bool)
        self.ambiguous=np.asarray(sample.get("ambiguous",np.zeros(len(self.times))),bool)
        n=len(self.times)
        if (not n or self.points.shape!=(n,33,2) or self.visibility.shape!=(n,33)
            or self.detected.shape!=(n,) or self.ambiguous.shape!=(n,)
            or not np.isfinite(self.times).all() or np.any(np.diff(self.times)<=0)):
            raise ValueError("Las coordenadas del video no son válidas.")
        self.interval=float(np.median(np.diff(self.times))) if n>1 else .125
        self.interval=min(.2,max(.02,self.interval))
        self.sides=[t["side"] for t in result.get("tracks",[]) if t["side"] in JOINTS]
        if not self.sides:self.sides=list(JOINTS)
        size=max(12,min(26,round(width/20)))
        try:self.font=ImageFont.truetype("DejaVuSans.ttf",size)
        except OSError:self.font=ImageFont.load_default()
        self.font_size=size

    def pose_at(self,time):
        j=int(np.searchsorted(self.times,time))
        nearest=min(max(j,0),len(self.times)-1)
        if j>0 and abs(self.times[j-1]-time)<=abs(self.times[nearest]-time):nearest=j-1
        if abs(self.times[nearest]-time)>self.interval*.55:return None,None
        if not self.detected[nearest] or self.ambiguous[nearest]:return None,None
        if 0<j<len(self.times):
            a,b=j-1,j;gap=self.times[b]-self.times[a]
            if (gap<=self.interval*1.6 and self.detected[a] and self.detected[b]
                and not self.ambiguous[a] and not self.ambiguous[b]):
                fraction=float((time-self.times[a])/gap)
                return ((1-fraction)*self.points[a]+fraction*self.points[b],
                        np.minimum(self.visibility[a],self.visibility[b]))
        return self.points[nearest],self.visibility[nearest]

    def repetition_at(self,time):
        candidates=[r for r in self.result["repetitions"] if r["start_time"]<=time<=r["end_time"]]
        if not candidates:return None
        return max(candidates,key=lambda r:{"verde":0,"amarillo":1,"rojo":2}[r["color"]])

    def draw(self,frame,time):
        import cv2
        height,width=frame.shape[:2];points,visibility=self.pose_at(time)
        rep=self.repetition_at(time)
        color=rep["color"] if rep else "neutro" if self.result["repetitions"] else "amarillo"
        if rep and self.result.get('feedback_version'):
            color=repetition_color(rep,self.result)
        video_only=bool(rep and self.result.get("decision_source")=="modelo_binario" and not self.result.get("errors"))
        banner_color="rojo" if video_only else color
        # Un clasificador de video no localiza un fallo en una articulación.
        # Cuando contradice los criterios, el video evita marcar certeza en verde.
        source=self.result.get("decision_source")
        if rep and (source in ("modelo_incierto","evidencia_en_conflicto")
                    or (source=="modelo_rgb" and self.result["color"]=="rojo" and color=="verde")):
            color="amarillo"
        banner_color="rojo" if video_only else color
        drawn=0
        if points is not None:
            def bone(a,b,rgb):
                if not np.isfinite(visibility[[a,b]]).all() or np.min(visibility[[a,b]])<.55 or not np.isfinite(points[[a,b]]).all():return False
                xy=np.rint(points[[a,b]]*height).astype(int)
                if np.any(xy[:,0]<0) or np.any(xy[:,0]>=width) or np.any(xy[:,1]<0) or np.any(xy[:,1]>=height):return False
                p,q=tuple(xy[0]),tuple(xy[1]);weight=max(2,round(height/180))
                cv2.line(frame,p,q,(20,25,30),weight+3,cv2.LINE_AA)
                cv2.line(frame,p,q,tuple(reversed(rgb)),weight,cv2.LINE_AA)
                for v in (p,q):
                    cv2.circle(frame,v,weight+2,(20,25,30),-1,cv2.LINE_AA)
                    cv2.circle(frame,v,weight,tuple(reversed(rgb)),-1,cv2.LINE_AA)
                return True
            for a,b in [(11,12),(23,24),(11,23),(12,24)]:bone(a,b,COLORS["neutro"])
            alignment={c['side']:c for c in rep.get('alignment_checks',[])} if rep else {}
            sides=list(alignment) if alignment else list(JOINTS) if rep and rep["side"]=="ambos" else [rep["side"]] if rep else self.sides
            for side in sides:
                s,e,w,h,k,a=JOINTS[side]
                chain=(h,k,a) if self.result["exercise"] in ("squat","inline_lunge") else (s,e,w)
                side_color="neutro" if video_only else color
                if side in alignment and not video_only:
                    side_color={"fuera_de_rango":"rojo","en_rango":color,"no_evaluable":"amarillo"}[alignment[side]['status']]
                    if any(c['status']=='fuera_de_rango' and not c['key'].startswith('alineacion_rodilla_') for c in rep['checks']):side_color='rojo'
                    if color=='rojo' and not self.result.get('errors'):side_color='rojo'
                for u,v in zip(chain,chain[1:]):drawn+=int(bone(u,v,COLORS[side_color]))
        if not drawn:color=banner_color="amarillo"
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(image)
        padding=max(8,round(width/40));top=padding
        box_height=self.font_size*2+padding*3
        draw.rounded_rectangle((padding,top,width-padding,top+box_height),radius=padding,fill=(15,20,28))
        first="MOVEAI"+(" · Video" if video_only else f' · Rep. {rep["number"]}' if rep else "")+f" · {int(time)//60:02d}:{int(time)%60:02d}"
        draw.text((padding*2,top+padding),first,font=self.font,fill=(215,222,230))
        label=LABELS[banner_color]
        if drawn and rep and self.result.get('feedback_version') and not video_only:
            label=repetition_label(rep,self.result)
        if drawn and banner_color=='amarillo' and self.result.get('status')=='uncertain':label='Resultado no concluyente'
        draw.text((padding*2,top+padding*2+self.font_size),label,font=self.font,fill=COLORS[banner_color])
        return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

def render_video(source,result,destination):
    import cv2
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg:raise RuntimeError("Falta FFmpeg. Incluye ffmpeg en packages.txt y reinicia la app.")
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination:raise ValueError("El video original y el resultado deben tener rutas diferentes.")
    destination.parent.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(source));writer=None
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not np.isfinite(fps) or fps<=0 or frames<1 or frames/fps>60.5:
            raise ValueError("No se pudo leer un video de hasta 60 segundos.")
        ok,frame=cap.read()
        if not ok:raise ValueError("No se pudo leer el primer fotograma.")
        raw_h,raw_w=frame.shape[:2];scale=min(1.,720/max(raw_h,raw_w))
        width=max(2,2*round(raw_w*scale/2));height=max(2,2*round(raw_h*scale/2));out_fps=min(24.,fps)
        overlay=VideoOverlay(result,width)
        with tempfile.TemporaryDirectory(prefix="render_",dir=destination.parent) as temporary:
            temporary=Path(temporary);raw=temporary/"frames.avi";output=temporary/"analizado.mp4"
            writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*"MJPG"),out_fps,(width,height))
            if not writer.isOpened():raise RuntimeError("No se pudo preparar el video anotado.")
            index=count=0
            while ok and index/fps<=60.5:
                time=index/fps
                if time+1e-7>=count/out_fps:
                    resized=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                    writer.write(overlay.draw(resized,time));count+=1
                index+=1;ok,frame=cap.read()
            writer.release();writer=None
            if not count:raise ValueError("No se encontraron fotogramas para exportar.")
            command=[ffmpeg,"-hide_banner","-loglevel","error","-y","-i",str(raw),"-i",str(source),
                     "-map","0:v:0","-map","1:a:0?","-c:v","libx264","-preset","veryfast","-crf","23",
                     "-threads","2","-pix_fmt","yuv420p","-c:a","aac","-b:a","128k","-movflags","+faststart",
                     "-map_metadata","-1","-t",f"{count/out_fps:.6f}",str(output)]
            encoded=subprocess.run(command,capture_output=True,timeout=120)
            if encoded.returncode or not output.exists() or output.stat().st_size<1000:
                raise RuntimeError("No se pudo exportar el video para el navegador.")
            output.replace(destination)
        return destination
    finally:
        if writer is not None:writer.release()
        cap.release()
