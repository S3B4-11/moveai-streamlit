"""Criterios de cada repetición y objetivo personal, sin diagnóstico de lesión."""
import numpy as np
from geometry import EXERCISES,TARGETS,JOINTS,SCHEMA,q,signals,trunk_change,criterion
from motion import active_tracks,repetitions

def visible(sample,joints,a,b):
    return (np.min(sample["visibility"][a:b,joints],axis=1)>=.55)&sample["detected"][a:b]

def assess(sample,exercise,t,rep,target):
    a,b=rep["start"],rep["end"];side=t["side"];s,e,w,h,k,ankle=JOINTS[side]
    series=t["series"][a:b];valid=np.isfinite(series)
    peak=q(series,95);base=q(t["cycle_series"][a:b],10)
    name="Flexión del codo" if exercise=="elbow_flexion" else "Elevación del brazo" if exercise=="shoulder_abduction" else "Flexión de rodilla"
    checks=[criterion("rango_incompleto",name,peak,target-10,"min",12,4,
                       "Completa el recorrido acordado sin forzarlo. Ajusta la meta si practicas otra variante.")]
    if exercise=="elbow_flexion":
        checks.append(criterion("extension","Retorno del codo",base,35,"max",12,1,"Retorna de forma controlada a la posición de inicio."))
        mask=visible(sample,[s,e,h],a,b);arm=signals(sample["image"])["arm_"+side][a:b]
        drift=(q(arm[mask],90)-q(arm[mask],10)) if mask.mean()>=.8 else None
        checks.append(criterion("movimiento_brazo","Estabilidad visible del brazo",drift,25,"max",12,2,
                                "Evita adelantar el codo para impulsar la carga."))
    if exercise!="squat":
        full_mask=visible(sample,[11,12,23,24],0,len(sample["times"]))
        mask=full_mask[a:b];lean=trunk_change(sample["image"],a,b,full_mask)
        checks.append(criterion("compensacion_tronco","Cambio visible de inclinación del tronco",
                                q(lean[mask],90) if mask.mean()>=.8 else None,20,"max",8,2,
                                "Mantén estable el tronco y revisa la carga si necesitas balancearte."))
    if exercise=="shoulder_abduction":
        mask=visible(sample,[s,e,w],a,b);flex=signals(sample["world"])["elbow_"+side][a:b]
        checks.append(criterion("codo_flexionado","Flexión del codo al elevar",q(flex[mask],90) if mask.mean()>=.8 else None,
                                35,"max",15,1,"Mantén la flexión de codo acordada para esta variante."))
    if exercise=="squat":
        checks.append(criterion("extension","Retorno de rodilla",base,40,"max",12,1,"Retorna de forma controlada a la postura inicial."))
        if t["view_observed"]!="perfil":
            sig=signals(sample["world"]);mask=visible(sample,[23,24,25,26,27,28],a,b)
            diff=abs(sig["knee_izquierda"][a:b]-sig["knee_derecha"][a:b])
            checks.append(criterion("asimetria","Diferencia estimada entre piernas",q(diff[mask],90) if mask.mean()>=.8 else None,
                                    25,"max",10,2,"Revisa el apoyo y la participación de ambas piernas."))
    total=sum(c["weight"] for c in checks);measurable=sum(c["weight"] for c in checks if c["status"]!="no_evaluable")
    passed=sum(c["weight"] for c in checks if c["status"]=="en_rango")
    score=passed/measurable if measurable else 0;coverage=measurable/total
    failures=[c for c in checks if c["status"]=="fuera_de_rango"];color="amarillo"
    if not rep["complete"]:reason=rep["reason"]
    elif visible(sample,t["joints"],a,b).mean()<.8 or valid.sum()<max(3,.25*len(valid)) or not np.isfinite(t["series"][rep["peak"]]):
        reason="Faltan medidas fiables del inicio, pico o retorno."
    elif t["reason"]:reason=t["reason"]
    elif coverage<.75:reason="Faltan criterios observables para calificar."
    elif failures:
        color="rojo";reason="Revisar: "+", ".join(c["name"].lower() for c in failures)+"."
    elif any(c["status"]=="limite" for c in checks):reason="Hay un criterio cerca del límite."
    elif checks[0]["status"]=="en_rango" and score>=.75:
        color="verde";reason="Recorrido y compensaciones observables en rango."
    else:reason="Falta cumplimiento de los criterios medidos."
    return dict(rep,side=side,screen_side=t["screen_side"],color=color,reason=reason,checks=checks,
                criterion_score=score,criterion_coverage=coverage,peak_angle=peak,angle_source=t["source"],
                measured_fraction=float(valid.mean()),visible_fraction=float(visible(sample,t["joints"],a,b).mean()),
                start_time=float(sample["times"][a]),end_time=float(sample["times"][b-1]),peak_time=float(sample["times"][rep["peak"]]))

def analyze(sample,exercise,side="automatica",target=None):
    if exercise not in EXERCISES or side not in (*JOINTS,"automatica"):raise ValueError("Ejercicio o extremidad desconocidos.")
    target=float(TARGETS[exercise] if target is None else target)
    if not 40<=target<=170:raise ValueError("La meta debe estar entre 40 y 170 grados.")
    tracks=active_tracks(sample,exercise,side);result=[]
    for t in tracks:
        result.extend(assess(sample,exercise,t,r,target) for r in repetitions(t["cycle_series"],sample["times"]))
    result.sort(key=lambda r:(r["start_time"],r["side"]));grouped=[]
    for rep in result:
        if grouped and exercise in ("elbow_flexion","shoulder_abduction") and grouped[-1]["side"]!=rep["side"] and abs(grouped[-1]["peak_time"]-rep["peak_time"])<.3:
            other=grouped.pop();worst=max([other,rep],key=lambda r:{"verde":0,"amarillo":1,"rojo":2}[r["color"]])
            grouped.append(dict(worst,side="ambos",screen_side="ambos lados",components=[other,rep]))
        else:grouped.append(rep)
    for n,r in enumerate(grouped,1):r["number"]=n
    counts={c:sum(r["color"]==c for r in grouped) for c in ["verde","rojo","amarillo"]}
    complete=[r for r in grouped if r["complete"]]
    if not grouped:color,title,reason="amarillo","No se encontró una repetición completa","Graba desde el inicio hasta el retorno con la extremidad activa visible."
    elif counts["rojo"]:color,title,reason="rojo","Revisar "+str(counts["rojo"])+" repetición(es)","Se muestran el criterio y la repetición que requieren revisión."
    elif counts["amarillo"]:color,title,reason="amarillo","Hay repeticiones que necesitan revisión","Hay medidas incompletas o cercanas al límite; revisa las repeticiones señaladas."
    else:color,title,reason="verde","Correcta en los criterios evaluados","Las repeticiones cumplen el recorrido acordado y las compensaciones observables."
    ready=bool(complete and tracks and all(t["pose_visibility"]>=.8 and not t["reason"] for t in tracks))
    sample.update(exercise=exercise,side=side,target=target,trainable=ready,pipeline=SCHEMA)
    return dict(color=color,title=title,reason=reason,exercise=exercise,target=target,tracks=tracks,repetitions=grouped,
                counts=counts,quality=dict(ready=ready,coverage=sample["coverage"],frames=len(sample["times"])),
                criterion_score=float(np.mean([r["criterion_score"] for r in complete])) if complete else None,
                sample=sample,decision_source="criterios_observables")
