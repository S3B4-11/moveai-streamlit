"""Respeta la orientación del archivo antes de extraer o dibujar la pose.

Los celulares pueden almacenar píxeles apaisados con una matriz de rotación.
OpenCV 4.11 no la aplica por defecto. Se materializa una sola vez con FFmpeg,
sin cambiar pose.py ni la firma de los clasificadores entrenados.
"""
from pathlib import Path
import json
import math
import shutil
import subprocess
import uuid
import numpy as np

ORIENTATION_VERSION = 'moveai_orientation_v2'


def video_metadata(source):
    """Lee la rotación de presentación (antihoraria), no la postura humana."""
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        raise RuntimeError('Falta FFprobe. Instala ffmpeg mediante packages.txt.')
    command = [ffprobe, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries',
               'stream=width,height,duration:stream_tags=rotate:stream_side_data=rotation:format=duration',
               '-of', 'json', str(source)]
    try:
        info = json.loads(subprocess.run(command, check=True, capture_output=True,
                                         text=True, timeout=20).stdout)
        stream = info['streams'][0]
        width, height = int(stream['width']), int(stream['height'])
        # La matriz de presentación prevalece sobre la etiqueta antigua.
        rotation = next((float(s['rotation']) for s in stream.get('side_data_list', [])
                         if 'rotation' in s), None)
        if rotation is None:
            rotation = float(stream.get('tags', {}).get('rotate', 0))
        durations = []
        for value in (stream.get('duration'), info.get('format', {}).get('duration')):
            try:
                number = float(value)
            except (ValueError, TypeError):
                continue
            if math.isfinite(number):durations.append(number)
        duration = max(durations, default=0.)
        if not math.isfinite(rotation) or min(width, height) <= 0:
            raise ValueError('Metadatos inválidos.')
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError) as error:
        raise ValueError('No se pudo leer la orientación de este video.') from error
    if duration > 60.5:
        raise ValueError('Recorta el video a un máximo de 60 segundos.')
    # La interfaz informa giros horarios; FFprobe informa antihorarios.
    clockwise = (-rotation + 180) % 360 - 180
    if abs(clockwise) < .01:
        clockwise = 0.
    return dict(width=width, height=height, duration=duration,
                metadata_rotation=clockwise)


def _transcode_upright(source, additional_rotation=0):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('Falta FFmpeg para corregir la orientación del video.')
    source = Path(source)
    destination = source.parent / ('moveai_upright_' + uuid.uuid4().hex + '.mp4')
    transform = {0: '', 90: 'transpose=1,', -90: 'transpose=2,',
                 180: 'hflip,vflip,'}[additional_rotation]
    filters = transform + 'pad=ceil(iw/2)*2:ceil(ih/2)*2'
    # FFmpeg aplica primero la matriz del celular. El giro manual es adicional.
    # Se eliminan los metadatos para que ningún reproductor vuelva a girarlo.
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-autorotate', '-i', str(source), '-map', '0:v:0', '-map', '0:a:0?',
               '-vf', filters, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
               '-threads', '2', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
               '-movflags', '+faststart', '-map_metadata', '-1',
               '-metadata:s:v:0', 'rotate=0', str(destination)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        if video_metadata(destination)['metadata_rotation'] != 0:
            raise RuntimeError('No se pudo eliminar la rotación del video preparado.')
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def prepare_video(source, rotation='auto'):
    """Automática respeta el archivo; no adivina giros por flexionar el tronco."""
    if rotation not in ('auto', 0, 90, -90, 180):
        raise ValueError('Orientación de video inválida.')
    metadata = video_metadata(source)
    manual = 0 if rotation == 'auto' else int(rotation)
    output = Path(source)
    if metadata['metadata_rotation'] or manual:
        output = _transcode_upright(source, manual)
    return output, dict(metadata, manual_rotation=manual,
                        normalized=output != Path(source), version=ORIENTATION_VERSION)

def inferred_rotation(sample):
    p=np.asarray(sample['image'],float);vis=np.asarray(sample['visibility'],float)
    valid=np.asarray(sample['detected'],bool)&~np.asarray(sample.get('ambiguous',np.zeros(len(p))),bool)
    joints=[11,12,23,24,27,28]
    valid &= (vis[:,joints].min(axis=1)>=.6)&np.isfinite(p[:,joints]).all(axis=(1,2))
    if valid.sum()<8:return 0
    shoulders=(p[:,11]+p[:,12])/2;hips=(p[:,23]+p[:,24])/2;feet=(p[:,27]+p[:,28])/2
    def degrees(v):return np.degrees(np.arctan2(v[:,0],v[:,1]))
    body=degrees(feet-shoulders)[valid];legs=degrees(feet-hips)[valid];torso=degrees(hips-shoulders)[valid]
    radians=np.deg2rad(body)
    median=float(np.degrees(np.arctan2(np.median(np.sin(radians)),np.median(np.cos(radians)))))
    rotation=int(round(median/90))*90
    if rotation not in (-90,90,180,-180):return 0
    def distance(a):return abs((a-rotation+180)%360-180)
    if (np.mean(distance(body)<20)<.85 or np.mean(distance(legs)<35)<.8 or np.mean(distance(torso)<35)<.8):return 0
    return 180 if abs(rotation)==180 else rotation

def rotate_video(source,rotation):
    if rotation not in (0,90,-90,180):raise ValueError('Orientación de video inválida.')
    return _transcode_upright(source,rotation)
