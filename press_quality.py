"""Lectura por fases y resumen parcial del press. No reduce umbrales angulares."""
from collections import Counter
import numpy as np

MIN_PHASE_SAMPLES=2

def phase_reading(valid,arm,start,peak,end):
    """Exige datos en inicio, fase superior y retorno, además del promedio."""
    a,b,p=int(start),int(end),int(peak)
    mask=np.asarray(valid,bool);curve=np.asarray(arm,float)
    edge=max(2,min(4,(b-a)//4))
    indices={'inicio':np.arange(a,min(p,a+edge-1)+1),
             'fase_superior':np.arange(max(a,p-1),min(b,p+2)),
             'retorno':np.arange(max(p+1,b-edge),b)}
    summary={}
    for name,idx in indices.items():
        good=mask[idx]&np.isfinite(curve[idx])
        summary[name]=dict(samples=int(len(idx)),measured=int(good.sum()),
                           coverage=float(good.mean()) if len(idx) else 0.)
    readable=all(x['measured']>=MIN_PHASE_SAMPLES for x in summary.values())
    return summary,bool(readable)

def finalize_press(result):
    """Conserva el juicio conservador y muestra cuánto se pudo observar."""
    sample=result['sample'];reps=result['repetitions']
    complete=[r for r in reps if r['complete']]
    assessed=[r for r in complete if r['color'] in ('verde','rojo')]
    unresolved=[r for r in complete if r['color']=='amarillo']
    result['assessment_summary']=dict(total_cycles=len(reps),complete_cycles=len(complete),
        assessed_complete_cycles=len(assessed),unassessed_complete_cycles=len(unresolved),
        incomplete_cycles=sum(not r['complete'] for r in reps),
        assessed_fraction=len(assessed)/len(complete) if complete else 0.,
        passing_cycles=sum(r['color']=='verde' for r in assessed),
        flagged_cycles=sum(r['color']=='rojo' for r in assessed),
        scope='Cobertura de lectura y criterios del press; no porcentaje de acierto.')
    result['partial_assessment']=bool(assessed and unresolved)
    result['unassessed_reasons']=dict(Counter(r['reason'] for r in reps if r['color']=='amarillo'))
    vis=np.asarray(sample['visibility']);det=np.asarray(sample['detected'],bool)
    names={11:'hombro izquierdo',12:'hombro derecho',13:'codo izquierdo',14:'codo derecho',
           15:'muñeca izquierda',16:'muñeca derecha',23:'cadera izquierda',24:'cadera derecha'}
    reading=result.setdefault('reading',{})
    reading['joint_coverage']={name:float(np.mean(det&(vis[:,j]>=.55))) for j,name in names.items()}
    views=Counter(t['view_observed'] for t in result['tracks'])
    view=views.most_common(1)[0][0] if views else 'desconocida'
    guidance=[]
    if view=='perfil':guidance.append('Prueba una toma frontal o diagonal: de perfil pueden superponerse los brazos y acortarse los segmentos en la imagen.')
    if any(reading['joint_coverage'][name]<.8 for name in ('muñeca izquierda','muñeca derecha')):
        guidance.append('Mantén las manos y los codos dentro del encuadre durante toda la subida y la bajada.')
    if not complete:guidance.append('Graba desde la posición de hombros hasta el retorno, con la cámara fija y sin cortes en el recorrido.')
    result['capture_guidance']=guidance
    if result['color']=='amarillo':
        if assessed:
            result['reason']=f'Se evaluaron {len(assessed)} de {len(complete)} ciclos completos. Faltan medidas fiables en los restantes; consulta el resultado de cada repetición.'
        elif reps:
            result['reason']=next((r['reason'] for r in reps if r['complete']),reps[0]['reason'])
        else:
            result['reason']='No se identificó un ciclo de press con posición de hombros, subida y retorno legibles.'
    return result

def sustained_seconds(mask,times):
    """Duración entre la primera y última muestra consecutiva, sin interpolar."""
    times=np.asarray(times,float);mask=np.asarray(mask,bool)
    best=0.;start=None
    dt=float(np.median(np.diff(times))) if len(times)>1 else .125
    for i,ok in enumerate(mask):
        if not ok:start=None;continue
        if start is None or (i and times[i]-times[i-1]>max(.2,dt*1.6)):start=i
        best=max(best,float(times[i]-times[start]))
    return best

def confirm_deviations(checks,signals,times):
    """Una desviación breve se muestra, pero no se convierte en un fallo firme."""
    unresolved=False
    for c in checks:
        if c['status']!='fuera_de_rango':continue
        support=np.asarray(signals[c['key']],bool)
        duration=sustained_seconds(support,times)
        c.update(evidence_duration_s=round(duration,3),required_duration_s=.25)
        if duration<.25:
            unresolved=True
            c.update(status='limite',evidence_status='not_sustained',
                     advice='Se observa una desviación breve en '+c['name'].lower()+'. Repite la toma para comprobar si se mantiene durante el movimiento.')
    return unresolved
