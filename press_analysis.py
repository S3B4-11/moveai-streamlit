"""Press con mancuernas: criterios observables específicos, sin clasificador.

Versión de reglas press_rgb_v2. Trabaja en proyección 2D sobre articulaciones
visibles; puede usar coordenadas world estimadas cuando la proyección se
acorta y las longitudes y trayectorias 3D superan controles de coherencia.
No mide carga, fuerza, escápula o curvatura lumbar. Referencias y
bandas son parámetros de ingeniería, no umbrales clínicos ni probabilidades.
Las variables de los clasificadores originales permanecen intactas.
"""
import numpy as np
from geometry import JOINTS, angle, q, smooth, criterion, torso_size
from motion import camera_view, repetitions
from press_quality import phase_reading, confirm_deviations

RULE_VERSION='press_rgb_v2'
TARGET=150
MIN_VISIBLE=.8

def make_track(sample,side):
    s,e,w,h,_,_=JOINTS[side]
    xy=np.asarray(sample['image'],float);vis=np.asarray(sample['visibility'])
    detected=np.asarray(sample['detected'],bool)&~np.asarray(sample.get('ambiguous',np.zeros(len(xy))),bool)
    joints=[s,e,w,h]
    seen=detected&(np.min(vis[:,joints],axis=1)>=.55)&np.isfinite(xy[:,joints]).all((1,2))
    torso=torso_size(xy)
    # Un segmento proyectado casi en un punto no permite medir el ángulo.
    plausible=np.isfinite(torso)&(torso>.01)
    for a,b in [(s,e),(e,w),(s,h)]:
        length=np.linalg.norm(xy[:,a]-xy[:,b],axis=1)
        ref=q(length[seen],75)
        plausible &= length>=.18*np.maximum(torso,1e-6)
        if ref:plausible &= (length>=.45*ref)&(length<=1.8*ref)
    arm=angle(xy[:,e],xy[:,s],xy[:,h])
    elbow=180-angle(xy[:,s],xy[:,e],xy[:,w])
    valid=seen&plausible&np.isfinite(arm)&np.isfinite(elbow)
    valid &= ~np.r_[False,(abs(np.diff(arm))>55)|(abs(np.diff(elbow))>55)]
    source='2D observable';variation=None
    rack2=valid&(arm>=40)&(arm<=125)&(elbow>=45)
    if 'world' in sample:
        world=np.asarray(sample['world'],float)
        good=seen&np.isfinite(world[:,joints]).all((1,2))
        variations=[]
        for a,b in [(s,e),(e,w)]:
            length=np.linalg.norm(world[:,a]-world[:,b],axis=1);ref=q(length[good],50)
            if ref is None or ref<=1e-6:good[:]=False;continue
            variations.append(float(np.std(length[good])/ref))
            good &= (length>=.55*ref)&(length<=1.6*ref)
        variation=max(variations,default=1.)
        arm3=angle(world[:,e],world[:,s],world[:,h])
        flex3=180-angle(world[:,s],world[:,e],world[:,w])
        good &= np.isfinite(arm3)&np.isfinite(flex3)
        good &= ~np.r_[False,(abs(np.diff(arm3))>55)|(abs(np.diff(flex3))>55)]
        rack3=good&(arm3>=40)&(arm3<=125)&(flex3>=45)
        # No se sustituye la visibilidad con profundidad inventada. Sólo se
        # cambia de espacio si la imagen acorta segmentos, falta la posición
        # de hombros en 2D y world conserva suficiente información coherente.
        use3=bool(variation<=.25 and good.sum()>=8 and rack3.sum()>=3
                  and (rack2.sum()<3 or (valid.mean()<.7 and good.sum()>valid.sum())))
        if use3:arm,elbow,valid,source=arm3,flex3,good,'3D estimada'
    times=np.asarray(sample['times']);dt=float(np.median(np.diff(times))) if len(times)>1 else .125
    radius=max(1,round(.12/max(dt,1e-3)))
    primary=smooth(arm,valid,radius);flex=smooth(elbow,valid,radius)
    # El apoyo inicial es la posición de hombros con codo flexionado.
    # No se usa como retorno el reposo con los brazos al costado del cuerpo.
    rack=valid&(arm>=40)&(arm<=125)&(elbow>=45)
    rack_angle=q(arm[rack],20)
    floor=float(np.clip(rack_angle if rack_angle is not None else 75,50,100))
    cycle=np.where(valid,np.maximum(primary,floor),np.nan)
    wrist=q(xy[:,w,0],50);mid=q((xy[:,11,0]+xy[:,12,0])/2,50)
    screen='izquierda de la imagen' if wrist is not None and mid is not None and wrist<mid else 'derecha de la imagen'
    return dict(side=side,screen_side=screen,joints=joints,series=primary,cycle_series=cycle,valid=valid,
                elbow_series=flex,source=source,bone_variation=variation,view_observed=camera_view(xy),
                pose_visibility=float(seen.mean()),visibility=float(valid.mean()),reason='',
                range=(q(primary,95) or 0)-(q(primary,5) or 0),rack_reference=floor,
                rack_observed=bool(rack.sum()>=3))

def check(key,name,value,reference,direction,tolerance,weight,advice,consequence):
    c=criterion(key,name,value,reference,direction,tolerance,weight,advice)
    c.update(source='press_2d',tolerance=float(tolerance),consequence=consequence)
    return c

def assess_press(sample,t,rep,target):
    a,b,p=rep['start'],rep['end'],rep['peak'];s,e,w,h,_,_=JOINTS[t['side']]
    arm=t['series'][a:b];flex=t['elbow_series'][a:b];valid=t['valid'][a:b]
    times=sample['times'];xy=sample['image'][a:b];visible=sample['visibility'][a:b]
    top=valid&(arm>=((q(arm[valid],90) or 180)-12))
    # Medidas en las dos fases; un codo flexionado al inicio es normal.
    elbow_top=q(flex[top],50) if top.sum()>=2 else None
    arm_top=q(arm[valid],95)
    local_peak=p-a;edge=max(2,min(4,len(arm)//4))
    tail=np.arange(len(arm))>=max(local_peak+1,len(arm)-edge)
    bottom=valid&((arm<=t['rack_reference']+25))
    bent=q(flex[bottom],90) if bottom.sum()>=2 else None
    extension=(bent-elbow_top) if bent is not None and elbow_top is not None else None
    return_arm=q(arm[tail&valid],50)
    return_flex=q(flex[tail&valid],50)
    torso=xy[:,s]-xy[:,h];base=valid&(np.arange(len(arm))<=min(local_peak,edge-1))
    ref=np.nanmedian(torso[base],axis=0) if base.any() else None
    lean=None;angles=np.full(len(arm),np.nan)
    if ref is not None:
        angles=angle(torso,np.zeros_like(torso),np.broadcast_to(ref,torso.shape))
        lean=q(angles[valid],90)
    checks=[
      check('press_elevacion','Elevación del brazo en el press',arm_top,target-10,'min',10,4,
        'Eleva las mancuernas por encima de la cabeza dentro de tu recorrido cómodo. Revisa la meta si necesitas otra variante.',
        'Un recorrido menor modifica el estímulo respecto al objetivo elegido; no demuestra una lesión.'),
      check('press_extension','Extensión del codo arriba',elbow_top,30,'max',12,3,
        'Completa la extensión controlada de los codos al finalizar la subida, sin forzar ni golpear el bloqueo.',
        'Mantener los codos muy flexionados arriba cambia el recorrido y el esfuerzo respecto al press seleccionado.'),
      check('press_cambio_codo','Cambio de flexión del codo',extension,30,'min',10,1,
        'Parte con los codos flexionados a la altura de los hombros y extiéndelos durante la subida.',
        'Mover el brazo sin el empuje del codo corresponde a otro patrón y cambia el ejercicio evaluado.'),
      check('press_retorno_brazo','Retorno a la altura de los hombros',return_arm,115,'max',10,1,
        'Baja de forma controlada hasta la posición de hombros antes de iniciar el siguiente empuje.',
        'Un retorno corto reduce la amplitud del ciclo respecto al objetivo elegido.'),
      check('press_retorno_codo','Flexión del codo durante el retorno',return_flex,45,'min',12,1,
        'Permite que los codos se flexionen al bajar las mancuernas hacia los hombros.',
        'Un retorno sin flexión del codo altera el patrón del press y su recorrido.'),
      check('compensacion_tronco','Cambio visible del tronco',lean,20,'max',8,2,
        'Mantén estable el tronco durante el empuje. Revisa la carga si necesitas balancearte para subir.',
        'El balanceo puede redistribuir el esfuerzo hacia el tronco. Esta medida no identifica una lesión lumbar.')]
    for c in checks:c['source']='press_2d' if c['key']=='compensacion_tronco' or t['source']=='2D observable' else 'press_3d_estimada'
    support={
        'press_elevacion':top&(arm<target-20),
        'press_extension':top&(flex>42),
        'press_cambio_codo':top&(bottom.sum()>=3),
        'press_retorno_brazo':tail&valid&(arm>125),
        'press_retorno_codo':tail&valid&(flex<33),
        'compensacion_tronco':valid&(angles>28)}
    short_evidence=confirm_deviations(checks,support,times[a:b])
    total=sum(c['weight'] for c in checks);measured=sum(c['weight'] for c in checks if c['status']!='no_evaluable')
    score=sum(c['weight'] for c in checks if c['status']=='en_rango')/measured if measured else 0
    coverage=measured/total
    seen=(np.min(visible[:,t['joints']],axis=1)>=.55)&sample['detected'][a:b]
    phase_quality,phases_ok=phase_reading(t['valid'],t['series'],a,p,b)
    clear=bool(seen.mean()>=MIN_VISIBLE and valid.mean()>=MIN_VISIBLE and valid[local_peak] and phases_ok)
    failed=[c for c in checks if c['status']=='fuera_de_rango']
    color='amarillo'
    if not rep['complete']:reason=rep['reason']
    elif not phases_ok:reason='Faltan medidas fiables en '+', '.join(k for k,v in phase_quality.items() if v['measured']<2)+'.'
    elif not clear or coverage<1:reason='Faltan medidas fiables de hombro, codo, muñeca o tronco en esta repetición.'
    elif failed:color='rojo';reason='Revisar: '+', '.join(c['name'].lower() for c in failed)+'.'
    elif short_evidence:reason='Hay una desviación breve sin suficiente continuidad para confirmarla.'
    elif any(c['status']=='limite' for c in checks):reason='Hay un criterio cerca del límite.'
    else:color='verde';reason='Subida, extensión del codo, retorno y tronco cumplen los criterios visibles del press.'
    return dict(rep,side=t['side'],screen_side=t['screen_side'],color=color,reason=reason,checks=checks,
                criterion_score=score,criterion_coverage=coverage,peak_angle=arm_top,angle_source=t['source'],
                measured_fraction=float(valid.mean()),visible_fraction=float(seen.mean()),phase_quality=phase_quality,
                start_time=float(times[a]),end_time=float(times[b-1]),peak_time=float(times[p]),quality_scope='repetition')

def analyze_press(sample,side='automatica',target=None):
    if side not in (*JOINTS,'automatica'):raise ValueError('Extremidad desconocida.')
    target=float(TARGET if target is None else target)
    if not np.isfinite(target) or not 120<=target<=170:raise ValueError('La meta del press debe estar entre 120 y 170 grados de elevación del brazo.')
    if len(sample['times'])<8:raise ValueError('No se leyó una secuencia suficiente para el press.')
    tracks=[make_track(sample,s) for s in JOINTS if side=='automatica' or s==side]
    best=max((t['range']*t['visibility'] for t in tracks),default=0)
    active=[t for t in tracks if t['rack_observed'] and t['range']>=20 and t['range']*t['visibility']>=max(12,.3*best)]
    reps=[]
    for t in active:
        reps.extend(assess_press(sample,t,r,target) for r in repetitions(t['cycle_series'],sample['times']))
    reps.sort(key=lambda r:(r['start_time'],r['side']))
    grouped=[]
    for r in reps:
        other=next((g for g in reversed(grouped) if g['side']!=r['side'] and g['side']!='ambos'
                    and abs(g['peak_time']-r['peak_time'])<.4),None)
        if other is None:grouped.append(r)
        else:
            worst=max([other,r],key=lambda a:{'verde':0,'amarillo':1,'rojo':2}[a['color']])
            combined=dict(worst,side='ambos',screen_side='ambos lados',components=[dict(other),r])
            # La observación de un fallo legible en un lado no se borra al
            # fusionarlo con una oclusión o un tramo cortado del otro lado.
            if worst['complete'] and worst['color']=='rojo':combined['observable_error']=True
            grouped[grouped.index(other)]=combined
    for i,r in enumerate(grouped,1):r['number']=i
    complete=[r for r in grouped if r['complete']]
    ready=bool(complete and all(r['visible_fraction']>=MIN_VISIBLE and r['measured_fraction']>=MIN_VISIBLE and r['criterion_coverage']==1 for r in complete))
    sample.update(exercise='shoulder_press',side=side,target=target,trainable=False,assessment_version=RULE_VERSION)
    return dict(exercise='shoulder_press',target=target,tracks=active or tracks,repetitions=grouped,
                counts={c:sum(r['color']==c for r in grouped) for c in ('verde','rojo','amarillo')},
                quality=dict(ready=ready,coverage=float(sample['coverage']),frames=len(sample['times'])),
                sample=sample,color='amarillo',title='No se pudo evaluar',reason='Falta un ciclo completo de press con hombro, codo y muñeca visibles.',
                criterion_score=None,decision_source='criterios_observables',assessment_version=RULE_VERSION,
                assessment_scope='Press de hombros con mancuernas, de pie o sentado; criterios sobre puntos visibles, 2D prioritario y world estimado bajo controles. Sin clasificador supervisado habilitado.')
