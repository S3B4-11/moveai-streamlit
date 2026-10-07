"""Geometría compartida entre entrenamiento e inferencia. No requiere TensorFlow."""
import numpy as np

EXERCISES = {'squat': 'Sentadilla', 'inline_lunge': 'Zancada',
             'shoulder_abduction': 'Elevación lateral / abducción de hombro',
             'elbow_flexion': 'Curl de bíceps'}
SCHEMA = 'angles_v4'
TAGS = {'squat': ['rango_incompleto', 'rodilla_hacia_dentro', 'asimetria'],
        'inline_lunge': ['rango_incompleto', 'rodilla_hacia_dentro', 'compensacion_tronco'],
        'shoulder_abduction': ['rango_incompleto', 'compensacion_tronco', 'elevacion_hombro', 'codo_flexionado'],
        'elbow_flexion': ['rango_incompleto', 'compensacion_tronco']}


def angle(a, b, c):
    u, v = np.asarray(a)-b, np.asarray(c)-b
    den = np.linalg.norm(u, axis=-1)*np.linalg.norm(v, axis=-1)
    out = np.degrees(np.arccos(np.clip(np.sum(u*v, axis=-1)/np.maximum(den, 1e-8), -1, 1)))
    return np.where(den > 1e-6, out, np.nan)


def signals(coords):
    """Ángulos 3D invariantes a la traslación, escala y giro global del sensor."""
    c = np.asarray(coords, dtype=float)
    if c.ndim != 3 or c.shape[1:] != (33, 3):
        raise ValueError('Se esperan coordenadas (T,33,3).')
    hip = (c[:, 23]+c[:, 24])/2
    sh = (c[:, 11]+c[:, 12])/2
    up = sh-hip
    # Usamos un eje corporal inicial para no aprender la orientación de la cámara.
    axis = np.nanmedian(up[:max(2, len(c)//8)], axis=0)
    axis /= max(np.linalg.norm(axis), 1e-6)
    lean = np.degrees(np.arccos(np.clip(up@axis/np.maximum(np.linalg.norm(up, axis=1), 1e-8), -1, 1)))
    result = {'trunk_change': lean}
    for side, s, e, w, h, k, a in [('left',11,13,15,23,25,27), ('right',12,14,16,24,26,28)]:
        result['knee_'+side] = 180-angle(c[:,h], c[:,k], c[:,a])
        result['elbow_'+side] = 180-angle(c[:,s], c[:,e], c[:,w])
        result['arm_'+side] = angle(c[:,e], c[:,s], c[:,h])
        result['hip_'+side] = 180-angle(c[:,s], c[:,h], c[:,k])
    lat=c[:,23]-c[:,24]
    width=np.linalg.norm(lat,axis=1)
    lat=lat/np.maximum(width[:,None],1e-6)
    medial=[]
    for h,k,a,sign in [(23,25,27,1),(24,26,28,-1)]:
        va=c[:,a]-c[:,h];vk=c[:,k]-c[:,h]
        frac=np.clip(np.sum(vk*axis,axis=1)/np.where(abs(va@axis)>1e-6,va@axis,np.nan),0,1)
        expected=c[:,h]+frac[:,None]*va
        medial.append(sign*np.sum((expected-c[:,k])*lat,axis=1)/np.maximum(width,1e-6))
    result['knee_medial']=np.maximum(*medial)
    result['knee_asymmetry'] = abs(result['knee_left']-result['knee_right'])
    result['arm_asymmetry'] = abs(result['arm_left']-result['arm_right'])
    return result


def feature_names(exercise):
    names = ['trunk_change']
    if exercise in ('squat', 'inline_lunge'):
        names += ['knee_active','knee_other','hip_active','hip_other','knee_asymmetry','knee_medial']
    else:
        names += ['elbow_active','elbow_other','arm_active','arm_other','arm_asymmetry']
    stats = ['q10','q50','q90','range','speed','peak_phase']
    return [f'{name}_{stat}' for name in names for stat in stats]+['motion_peak_target_ratio','motion_range_target_ratio']


def features(coords, exercise, target=None):
    s = signals(coords)
    pair = ('knee', 'hip') if exercise in ('squat','inline_lunge') else ('elbow','arm')
    activity = 'arm' if exercise == 'shoulder_abduction' else pair[0]
    left_range = np.nanpercentile(s[activity+'_left'],90)-np.nanpercentile(s[activity+'_left'],10)
    right_range = np.nanpercentile(s[activity+'_right'],90)-np.nanpercentile(s[activity+'_right'],10)
    active, other = ('left','right') if left_range > right_range else ('right','left')
    values = [s['trunk_change']]
    for name in pair:
        values += [s[name+'_'+active],s[name+'_'+other]]
    values += [s['knee_asymmetry' if pair[0]=='knee' else 'arm_asymmetry']]
    if pair[0]=='knee':values += [s['knee_medial']]
    output=[]
    for a in values:
        q=np.nanpercentile(a,[10,50,90])
        output.extend([*q, q[2]-q[0], np.nanmedian(abs(np.diff(a))), float(np.mean(np.flatnonzero(a>=q[2]-1e-6)))/max(len(a)-1,1)])
    target=float(target or {'squat':80,'inline_lunge':85,'shoulder_abduction':150,'elbow_flexion':120}[exercise])
    q=np.nanpercentile(s[activity+'_'+active],[10,90])
    output.extend([q[1]/target,(q[1]-q[0])/target])
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def rule_assessment(image_coords, visibility, exercise, view='perfil', side='derecha', target=None):
    """Medidas 2D en el plano solicitado. Las bandas son configurables, no etiquetas médicas."""
    c=np.asarray(image_coords, float);v=np.asarray(visibility,float)
    s,e,w,h,k,a=(12,14,16,24,26,28) if side=='derecha' else (11,13,15,23,25,27)
    checks=[]
    def add(key,name,series,joints,plane,reference,lower=True,tolerance=10,advice=''):
        valid=np.isfinite(series)
        valid &= np.min(v[:,joints],axis=1)>=0.55
        visible=float(valid.mean())
        evaluable=visible>=0.8 and view==plane
        value=float(np.percentile(series[valid],90)) if valid.any() else None
        status='no_evaluable'
        reason=''
        if visible<.8:reason=f'Articulaciones visibles en {visible:.0%} de los fotogramas; se requiere 80%.'
        elif view!=plane:reason=f'Esta medida 2D requiere vista de {plane}; la configuración indica {view}.'
        if evaluable and value is not None:
            good=value>=reference if lower else value<=reference
            borderline=value>=reference-tolerance if lower else value<=reference+tolerance
            status='en_rango' if good else 'limite' if borderline else 'fuera_de_rango'
        checks.append({'key':key,'name':name,'value':None if value is None else round(value,2),
                       'reference':reference,'direction':'>=' if lower else '<=',
                       'status':status,'view':plane,'visibility':visible,'advice':advice,'reason':reason})
    if exercise in ('squat','inline_lunge'):
        kl=180-angle(c[:,23],c[:,25],c[:,27]);kr=180-angle(c[:,24],c[:,26],c[:,28])
        series=(kl+kr)/2 if exercise=='squat' else (kr if side=='derecha' else kl)
        target=float(target or (80 if exercise=='squat' else 85))
        joints=[23,24,25,26,27,28] if exercise=='squat' else [h,k,a]
        add('rango_incompleto','Flexión de rodilla' + (' delantera' if exercise=='inline_lunge' else ''),
            series,joints,'perfil',target,advice='Ajusta el recorrido al objetivo acordado; usa apoyo si hace falta.')
        # Desplazamiento medial respecto al segmento cadera-tobillo, en frente.
        inward=[]
        for hh,kk,aa in [(23,25,27),(24,26,28)]:
            denom=c[:,aa,1]-c[:,hh,1]
            f=np.clip((c[:,kk,1]-c[:,hh,1])/np.where(abs(denom)>1e-6,denom,np.nan),0,1)
            expected=c[:,hh,0]+f*(c[:,aa,0]-c[:,hh,0])
            mid=(c[:,23,0]+c[:,24,0])/2
            sign=np.sign(c[:,hh,0]-mid)
            width=np.linalg.norm(c[:,23]-c[:,24],axis=1)
            inward.append((expected-c[:,kk,0])*sign/np.maximum(width,1e-6)*100)
        add('rodilla_hacia_dentro','Desplazamiento de rodilla hacia dentro (%)',np.maximum(*inward),
            [23,24,25,26,27,28],'frente',15,False,10,'Mantén la rodilla siguiendo la dirección del pie.')
        if exercise=='squat':
            add('asimetria','Diferencia de flexión entre piernas',abs(kl-kr),[23,24,25,26,27,28],
                'frente',15,False,10,'Comprueba una distribución pareja del apoyo.')
        motion=series
    else:
        arm=angle(c[:,e],c[:,s],c[:,h]);elbow=180-angle(c[:,s],c[:,e],c[:,w])
        motion=arm if exercise=='shoulder_abduction' else elbow
        target=float(target or (90 if exercise=='shoulder_abduction' else 120))
        add('rango_incompleto','Amplitud del brazo' if exercise=='shoulder_abduction' else 'Flexión del codo',
            motion,[s,e,h] if exercise=='shoulder_abduction' else [s,e,w],
            'frente' if exercise=='shoulder_abduction' else 'perfil',target,
            advice='Completa el recorrido objetivo sin impulsar el peso con el tronco.')
        if exercise=='shoulder_abduction':
            add('codo_flexionado','Flexión del codo durante la elevación',elbow,[s,e,w],
                'frente',20,False,15,'Mantén la flexión acordada del codo durante la elevación.')
    hip=(c[:,23]+c[:,24])/2;sh=(c[:,11]+c[:,12])/2;trunk=sh-hip
    ref=np.nanmedian(trunk[:max(2,len(c)//8)],axis=0)
    change=np.degrees(np.arccos(np.clip(trunk@ref/(np.linalg.norm(trunk,axis=1)*max(np.linalg.norm(ref),1e-6)), -1,1)))
    # La inclinación hacia delante es normal en sentadilla: no la llamamos lesión lumbar.
    if exercise!='squat':
        add('compensacion_tronco','Cambio de inclinación del tronco',change,[11,12,23,24],
            'frente' if exercise=='shoulder_abduction' else 'perfil',15,False,10,
            'Reduce la carga o usa apoyo para mantener estable el tronco.')
    required=[h,k,a] if exercise=='inline_lunge' else [23,24,25,26,27,28] if exercise=='squat' else [s,e,w,h]
    seen=np.min(v[:,required],axis=1)>=.55
    q=np.percentile(motion[seen],[10,90]) if seen.any() else [0,0]
    movement=float(q[1]-q[0])
    primary=checks[0]
    quality=bool(primary['status']!='no_evaluable' and movement>=20)
    reason=primary['reason']
    if not reason and movement<20:
        reason=f'El recorrido proyectado varía solo {movement:.1f}°. Puede faltar movimiento, estar elegido otro lado o verse el movimiento fuera del plano de la cámara.'
    return {'checks':checks,'movement':round(movement,1),'evaluable':quality,
            'reason':'' if quality else reason,
            'target':target,'view':view,'side':side}
