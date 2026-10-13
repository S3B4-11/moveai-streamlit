"""Alineación frontal observable; no diagnostica valgo/varo estructural.

Los umbrales son criterios de ingeniería para un patrón pronunciado y persistente,
no puntos de corte clínicos. La apertura de pies/rodillas por sí sola no es un fallo.
"""
import numpy as np
from geometry import signals

SIDES={"izquierda":(23,25,27,29,31),"derecha":(24,26,28,30,32)}
SOURCE="https://pmc.ncbi.nlm.nih.gov/articles/PMC6050697/"

def longest_run(values):
    best=run=0
    for value in values:
        run=run+1 if value else 0;best=max(best,run)
    return best

def alignment_checks(sample,start,end):
    p=np.asarray(sample["image"],float);world=np.asarray(sample["world"],float)
    vis=np.asarray(sample["visibility"],float);times=np.asarray(sample["times"],float)
    seen=np.asarray(sample["detected"],bool)&~np.asarray(sample.get("ambiguous",np.zeros(len(times))),bool)
    body=[11,12,23,24,27,28]
    body_seen=seen&(np.min(vis[:,body],axis=1)>=.6)&np.isfinite(p[:,body]).all(axis=(1,2))
    shoulder=(p[:,11]+p[:,12])/2;hip=(p[:,23]+p[:,24])/2;ankle=(p[:,27]+p[:,28])/2
    torso=np.linalg.norm(hip-shoulder,axis=1);down=ankle-shoulder
    norm=np.linalg.norm(down,axis=1);down=down/np.maximum(norm[:,None],1e-8)
    lateral=np.column_stack([down[:,1],-down[:,0]])
    shoulder_width=abs(np.sum((p[:,12]-p[:,11])*lateral,axis=1))
    hip_width=abs(np.sum((p[:,24]-p[:,23])*lateral,axis=1))
    wshoulder=world[:,12]-world[:,11];whip=world[:,24]-world[:,23]
    def yaw(v):return np.degrees(np.arctan2(abs(v[:,2]),abs(v[:,0])+1e-8))
    frontal=body_seen&(torso>.08)&(shoulder_width/np.maximum(torso,1e-8)>.48)&(hip_width/np.maximum(torso,1e-8)>.22)
    frontal &= np.isfinite(wshoulder).all(axis=1)&np.isfinite(whip).all(axis=1)&(yaw(wshoulder)<30)&(yaw(whip)<35)
    sl=slice(start,end);dt=float(np.median(np.diff(times))) if len(times)>1 else .125
    minimum=max(3,int(np.ceil(.35/max(dt,.02))))
    if np.mean(frontal[sl])<.8:return []
    flex=signals(world);output=[]
    for side,(h,k,a,heel,toe) in SIDES.items():
        joints=[h,k,a,heel,toe]
        valid=frontal&(np.min(vis[:,joints],axis=1)>=.6)&np.isfinite(p[:,joints]).all(axis=(1,2))
        leg=np.linalg.norm(p[:,h]-p[:,k],axis=1)+np.linalg.norm(p[:,k]-p[:,a],axis=1)
        valid &= leg>.12
        # Distancia a la línea cadera-tobillo, usando el eje del cuerpo.
        ha=p[:,a]-p[:,h];hk=p[:,k]-p[:,h]
        fraction=np.clip(np.sum(hk*ha,axis=1)/np.maximum(np.sum(ha*ha,axis=1),1e-8),0,1)
        reference=p[:,h]+fraction[:,None]*ha
        side_sign=np.sign(np.sum((p[:,h]-hip)*lateral,axis=1));valid &= abs(side_sign)>.5
        deviation=np.sum((p[:,k]-reference)*lateral,axis=1)*side_sign/np.maximum(leg,1e-8)
        projected=np.sum(p[:,[a,heel,toe]]*lateral[:,None,:],axis=2)*side_sign[:,None]
        knee=np.sum(p[:,k]*lateral,axis=1)*side_sign
        outside=np.maximum(0,knee-projected.max(axis=1))/np.maximum(leg,1e-8)
        inside=np.maximum(0,projected.min(axis=1)-knee)/np.maximum(leg,1e-8)
        # Una rodilla puede abrirse correctamente siguiendo la dirección del pie.
        # Se exige también un desacuerdo persistente entre ambas direcciones.
        foot=p[:,toe]-p[:,heel];shin=p[:,k]-p[:,a]
        forward=np.sum(foot*down,axis=1);shin_height=-np.sum(shin*down,axis=1)
        foot_side=np.sum(foot*lateral,axis=1)*side_sign
        shin_side=np.sum(shin*lateral,axis=1)*side_sign
        foot_angle=np.degrees(np.arctan2(foot_side,forward))
        shin_angle=np.degrees(np.arctan2(shin_side,shin_height))
        gap=shin_angle-foot_angle
        foot_length=np.linalg.norm(foot,axis=1)
        valid &= (forward>.03*leg)&(shin_height>.08*leg)&(foot_length<.65*leg)&(foot_length>.02*leg)
        standing=valid&np.isfinite(flex['knee_'+side])&(flex['knee_'+side]<35)
        baseline=float(np.median(deviation[standing])) if standing.sum()>=minimum else None
        baseline_gap=float(np.median(gap[standing])) if standing.sum()>=minimum else 0.
        delta=deviation-(baseline if baseline is not None else 0.)
        threshold=.14 if baseline is not None else .25
        absolute=.16 if baseline is not None else .25
        phase=valid&np.isfinite(flex['knee_'+side])&(flex['knee_'+side]>=35)
        key='alineacion_rodilla_'+side
        check=dict(key=key,name='Alineación de rodilla '+side,value=None,reference=.35,direction='max',
                   unit='s de desviación persistente',status='no_evaluable',weight=2,side=side,joints=joints,
                   advice='Graba de frente con ambas rodillas y los pies visibles.',consequence='',source=SOURCE,
                   baseline_available=baseline is not None,observed_fraction=float(np.mean(valid[sl])))
        if np.mean(valid[sl])<.8 or phase[sl].sum()<minimum:
            output.append(check);continue
        inward=phase&(delta < -threshold)&(deviation < -absolute)&(inside>.06)&(gap<-25)&((gap-baseline_gap)<-12)
        outward=phase&(delta > threshold)&(deviation > absolute)&(outside>.06)&(gap>25)&((gap-baseline_gap)>12)
        inward_run=longest_run(inward[sl]);outward_run=longest_run(outward[sl])
        direction='hacia_dentro' if inward_run>=minimum and inward_run>=outward_run else 'hacia_fuera' if outward_run>=minimum else None
        check.update(value=round(max(inward_run,outward_run)*dt,3),status='en_rango',
                     advice='La rodilla sigue el apoyo observado del pie.',
                     lateral_change_percent=round(float(np.percentile(abs(delta[sl][phase[sl]]),90))*100,1),
                     foot_direction_difference=round(float(np.percentile(abs(gap[sl][phase[sl]]),90)),1))
        if direction:
            label='hacia dentro' if direction=='hacia_dentro' else 'hacia fuera'
            check.update(status='fuera_de_rango',pattern=direction,name=f'Rodilla {side} desviada {label}',
                         advice='Busca que la rodilla siga la dirección del pie durante el descenso. Revisa la postura y la carga con un profesional.',
                         consequence='Una desviación repetida con carga puede aumentar o redistribuir la carga en la rodilla. Puede favorecer sobrecarga o molestias; este video no permite identificar una lesión.',
                         evidence_frames=max(inward_run,outward_run))
        output.append(check)
    return output

def add_alignment(result):
    if result['exercise']!='squat':return result
    for rep in result['repetitions']:
        checks=alignment_checks(result['sample'],rep['start'],rep['end'])
        rep['alignment_checks']=checks;rep['checks'].extend(checks)
        total=sum(c['weight'] for c in rep['checks'])
        measured=sum(c['weight'] for c in rep['checks'] if c['status']!='no_evaluable')
        passed=sum(c['weight'] for c in rep['checks'] if c['status']=='en_rango')
        rep.update(criterion_score=passed/measured if measured else 0.,criterion_coverage=measured/total if total else 0.)
        failed=[c for c in checks if c['status']=='fuera_de_rango']
        if failed:
            rep.update(color='rojo',reason='; '.join(c['name'] for c in failed)+'.',observable_error=True,
                       affected_sides=[c['side'] for c in failed])
    complete=[r for r in result['repetitions'] if r['complete']]
    result['criterion_score']=float(np.mean([r['criterion_score'] for r in complete])) if complete else None
    return result
