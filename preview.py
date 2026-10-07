"""Fotograma anotado de la repetición, sin volver a ejecutar la pose."""
import io
from PIL import Image,ImageDraw
from geometry import JOINTS

def annotated(sample,rep):
    if not sample.get("previews"):return None
    index,data=min(sample["previews"],key=lambda pair:abs(sample["times"][pair[0]]-rep["peak_time"]))
    image=Image.open(io.BytesIO(data)).convert("RGB");draw=ImageDraw.Draw(image)
    sides=list(JOINTS) if rep["side"]=="ambos" else [rep["side"]]
    points=sample["image"][index]
    for side in sides:
        s,e,w,h,k,a=JOINTS[side]
        chains=[(h,k,a)] if sample.get("exercise") in ("squat","inline_lunge") else [(s,e,w)]
        for joints in chains:
            for i,j in zip(joints,joints[1:]):
                if sample["visibility"][index,[i,j]].min()<.55:continue
                p=tuple(points[i]*image.height);q=tuple(points[j]*image.height)
                draw.line([p,q],fill="#30cde0",width=4)
                for x,y in [p,q]:draw.ellipse((x-4,y-4,x+4,y+4),fill="#30cde0")
    return image
