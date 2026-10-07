"""Modelos RGB portables; ningún clasificador Kinect participa en la app."""
import hashlib
import json
from pathlib import Path
import numpy as np
from geometry import SCHEMA,TARGETS,JOINTS,signals,trunk_change,torso_size,q
from analysis import analyze
from motion import repetitions

SIGNALS=["elbow","arm","hip","knee"]
QUANTILES=[5,25,50,75,95]
FEATURE_NAMES=["primary_q10","primary_q50","primary_q90","primary_q95","primary_range",
 "primary_speed","peak_target_ratio","range_target_ratio","trunk_q90","arm_drift_q90",
 "elbow_q90","hip_q90","available_fraction","reading_fraction"]
FEATURE_NAMES += [f"{space}_{s}_q{n}" for space in ["world","image"] for s in SIGNALS for n in QUANTILES]
FEATURE_NAMES += ["view_width_ratio","world_shoulder_depth","hip_width_ratio","world_trunk_median",
 "world_trunk_q90","image_trunk_median","image_trunk_q90","image_hip_vertical_range",
 "image_hip_horizontal_range","stance_ratio","hip_knee_height","rep_peak_median","rep_peak_q10",
 "rep_peak_variation","rep_rom_median","rep_arm_world","rep_arm_image","rep_trunk_world",
 "rep_trunk_image","rep_ascent_descent_ratio","active_speed_q75","active_acceleration_q90"]
WAVE_KEYS=["primary","arm_world","arm_image","elbow_world","hip_world","knee_world","trunk_world","trunk_image","knee_alignment_image"]
FEATURE_NAMES += [f"phase_{key}_{n}" for key in WAVE_KEYS for n in range(11)]+[f"phase_variation_{key}" for key in WAVE_KEYS]

def phase_features(sample,tracks,exercise,sw,si,trunk_visible):
    """Forma temporal de ciclos, alineando inicio/pico/retorno entre personas."""
    waves={key:[] for key in WAVE_KEYS};times=sample["times"];grid=np.linspace(0,1,11)
    for t in tracks:
        side=t["side"];s,e,w,h,k,ankle=JOINTS[side]
        for rep in repetitions(t["cycle_series"],times):
            if not rep["complete"]:continue
            a,b,p=rep["start"],rep["end"],rep["peak"];tt=times[a:b]
            phase=np.where(tt<=times[p],.5*(tt-times[a])/max(times[p]-times[a],.01),
                           .5+.5*(tt-times[p])/max(times[b-1]-times[p],.01))
            signals_here={"primary":t["series"][a:b]/TARGETS[exercise],
                          "trunk_world":trunk_change(sample["world"],a,b,trunk_visible)/45,
                          "trunk_image":trunk_change(sample["image"],a,b,trunk_visible)/45}
            image=sample["image"][a:b]
            width=abs(image[:,25,0]-image[:,26,0])/np.maximum(abs(image[:,27,0]-image[:,28,0]),.08*torso_size(image))
            mask=(sample["visibility"][a:b,[25,26,27,28]].min(axis=1)>=.55)&sample["detected"][a:b]
            width[~mask]=np.nan;signals_here["knee_alignment_image"]=width
            for key,sig,ids in [("arm_world",sw,[s,e,h]),("arm_image",si,[s,e,h]),
                                ("elbow_world",sw,[s,e,w]),("hip_world",sw,[s,h,k]),("knee_world",sw,[h,k,ankle])]:
                signal=key.split("_")[0];v=sig[signal+"_"+side][a:b].copy()/180
                mask=(sample["visibility"][a:b,ids].min(axis=1)>=.55)&sample["detected"][a:b]
                v[~mask]=np.nan;signals_here[key]=v
            for key,v in signals_here.items():
                valid=np.isfinite(v)
                if valid.sum()>=3:
                    # Remuestreo para el clasificador; no rellena las medidas
                    # ni convierte una repetición ocluida en evaluable.
                    waves[key].append(np.interp(grid,phase[valid],v[valid]))
    return [float(v) for key in WAVE_KEYS for v in (np.median(waves[key],axis=0) if waves[key] else np.zeros(11))]+[
            float(np.mean(np.std(waves[key],axis=0))) if waves[key] else 0. for key in WAVE_KEYS]

def signature():
    from pose import asset
    h=hashlib.sha256(asset().read_bytes())
    for name in ["geometry.py","motion.py","analysis.py","pose.py","camera_model.py"]:
        h.update(name.encode());h.update((Path(__file__).parent/name).read_text().replace("\r\n","\n").encode())
    return h.hexdigest()

def features(sample,exercise):
    # La meta personal no altera la etiqueta de técnica que aprendió el detector.
    result=analyze(dict(sample),exercise,target=TARGETS[exercise]);tracks=result["tracks"]
    sw,si=signals(sample["world"]),signals(sample["image"]);values=[];speeds=[];arms=[];elbows=[];hips=[]
    for t in tracks:
        side=t["side"];v=t["series"];values.extend(v[np.isfinite(v)])
        speed=abs(np.diff(v))/np.maximum(np.diff(sample["times"]),1e-3);speeds.extend(speed[np.isfinite(speed)])
        arms.append((q(sw["arm_"+side],90) or 0)-(q(sw["arm_"+side],10) or 0))
        elbows.append(q(sw["elbow_"+side],90) or 0);hips.append(q(sw["hip_"+side],90) or 0)
    trunk_visible=(np.min(sample["visibility"][:,[11,12,23,24]],axis=1)>=.55)&sample["detected"]
    trunks=[q(trunk_change(sample["world"],r["start"],r["end"],trunk_visible),90) or 0 for r in result["repetitions"]]
    quantiles=[q(values,n) or 0 for n in (10,50,90,95)]
    coverage=np.mean([r["criterion_coverage"] for r in result["repetitions"]]) if result["repetitions"] else 0
    x=[*quantiles,quantiles[2]-quantiles[0],q(speeds,50) or 0,quantiles[3]/TARGETS[exercise],
       (quantiles[2]-quantiles[0])/TARGETS[exercise],max(trunks,default=0),max(arms,default=0),
       max(elbows,default=0),max(hips,default=0),coverage,min((t["visibility"] for t in tracks),default=0)]
    for space,sig in [("world",sw),("image",si)]:
        for signal in SIGNALS:
            vals=[]
            for t in tracks:
                s,e,w,h,k,a=JOINTS[t["side"]]
                ids=[s,e,w] if signal=="elbow" else [s,e,h] if signal=="arm" else [s,h,k] if signal=="hip" else [h,k,a]
                mask=np.min(sample["visibility"][:,ids],axis=1)>=.55
                vals.extend(sig[signal+"_"+t["side"]][mask])
            x += [q(vals,n) or 0 for n in QUANTILES]
    image,world=sample["image"],sample["world"];ti=np.maximum(torso_size(image),1e-6)
    x += [q(np.linalg.norm(image[:,11]-image[:,12],axis=1)/ti,50) or 0,
          q(abs(world[:,11,2]-world[:,12,2]),50) or 0,q(np.linalg.norm(image[:,23]-image[:,24],axis=1)/ti,50) or 0]
    for coords in [world,image]:
        spine=(coords[:,11]+coords[:,12]-coords[:,23]-coords[:,24])/2
        tilt=np.degrees(np.arccos(np.clip(-spine[:,1]/np.maximum(np.linalg.norm(spine,axis=1),1e-6),-1,1)))
        x += [q(tilt,50) or 0,q(tilt,90) or 0]
    center=(image[:,23]+image[:,24])/2;scale=q(ti,50) or 1
    x += [((q(center[:,axis],95) or 0)-(q(center[:,axis],5) or 0))/scale for axis in [1,0]]
    x += [q(np.linalg.norm(image[:,25]-image[:,26],axis=1)/np.maximum(np.linalg.norm(image[:,27]-image[:,28],axis=1),.05*ti),50) or 0,
          q(((world[:,23,1]+world[:,24,1])-(world[:,25,1]+world[:,26,1]))/2,90) or 0]
    peaks=[];roms=[];aw=[];ai=[];tw=[];tr=[];ratios=[];active_speed=[];acc=[]
    for t in tracks:
        side=t["side"];series=t["series"];cycle=t["cycle_series"]
        for rep in repetitions(cycle,sample["times"]):
            if not rep["complete"]:continue
            a,b=rep["start"],rep["end"];peak=q(series[a:b],95)
            if peak is None:continue
            peaks.append(peak);roms.append(peak-(q(cycle[a:b],10) or 0))
            for sig,bag in [(sw,aw),(si,ai)]:
                v=sig["arm_"+side][a:b];bag.append((q(v,95) or 0)-(q(v,5) or 0))
            for coords,bag in [(world,tw),(image,tr)]:bag.append(q(trunk_change(coords,a,b,trunk_visible),90) or 0)
            times=sample["times"];ratios.append((times[rep["peak"]]-times[a])/max(times[b-1]-times[rep["peak"]],.2))
            dt=np.maximum(np.diff(times[a:b]),1e-3);speed=abs(np.diff(cycle[a:b]))/dt
            active_speed.extend(speed[np.isfinite(speed)])
            if len(speed)>1:
                v=abs(np.diff(speed))/dt[1:];acc.extend(v[np.isfinite(v)])
    x += [q(peaks,50) or 0,q(peaks,10) or 0,float(np.std(peaks))/max(q(peaks,50) or 0,1) if peaks else 0,
          q(roms,50) or 0,*[q(bag,50) or 0 for bag in [aw,ai,tw,tr,ratios]],q(active_speed,75) or 0,q(acc,90) or 0]
    x += phase_features(sample,tracks,exercise,sw,si,trunk_visible)
    if len(x)!=len(FEATURE_NAMES):raise ValueError("Cambió el esquema de características.")
    return np.nan_to_num(np.asarray(x,np.float32)),result

def raw_score(model,x):
    if model["kind"]=="ensemble":
        if "score_calibration" in model:
            c=model["score_calibration"];mean=np.mean([raw_score(m,x) for m in model["members"]])
            return float(1/(1+np.exp(-np.clip(c["coefficient"]*mean+c["intercept"],-40,40))))
        c=model["member_calibration"]
        return float(np.mean([1/(1+np.exp(-np.clip(c["coefficient"]*raw_score(m,x)+c["intercept"],-40,40))) for m in model["members"]]))
    if model["kind"]=="svm":
        z=(x-np.asarray(model["mean"]))/model["scale"]
        distance=np.sum((np.asarray(model["support_vectors"])-z)**2,axis=1)
        return float(np.dot(model["dual_coefficients"],np.exp(-model["gamma"]*distance))+model["intercept"])
    if model["kind"]=="linear":
        z=(x-np.asarray(model["mean"]))/model["scale"]
        return float(np.dot(model["coefficients"],z)+model["intercept"])
    probabilities=[]
    for tree in model["trees"]:
        n=0
        while tree["left"][n]!=-1:
            n=tree["left"][n] if x[tree["feature"][n]]<=tree["threshold"][n] else tree["right"][n]
        probabilities.append(tree["p_error"][n])
    return float(np.mean(probabilities))

def predict(sample,exercise,root=None):
    path=Path(root or Path(__file__).parent/"camera_modelos")/exercise/"modelo.json"
    unavailable=dict(available=False,reason="La calificación se basa en los criterios observables de esta grabación.")
    if not path.exists():return unavailable
    m=json.loads(path.read_text())
    if (m.get("schema")!=SCHEMA or m.get("input_domain")!="rgb_video" or m.get("feature_names")!=FEATURE_NAMES
        or m.get("signature")!=signature() or m.get("extractor_id")!=sample.get("extractor_id")
        or not m.get("approved",False)):return unavailable
    x,result=features(sample,exercise)
    if not result["quality"]["ready"]:
        return dict(unavailable,reason="La lectura de pose necesita revisión; no se muestra una probabilidad del detector.")
    if np.mean((x<np.asarray(m["feature_low"]))|(x>np.asarray(m["feature_high"])))>.25:
        return dict(unavailable,reason="La grabación está fuera del rango evaluado del detector; se usan los criterios observables.")
    p=raw_score(m,x)
    return dict(available=True,p_correct=1-p,p_error=p,metrics=m["metrics"],
                scope="Estimación RGB evaluada por persona en el dataset; no garantiza acierto en toda cámara.")

def hard_failure(result):
    return any(c["key"]!="rango_incompleto" and c["status"]=="fuera_de_rango"
               for r in result["repetitions"] if r["color"]=="rojo" for c in r["checks"])

def fuse(color,p_correct,ready,hard=False):
    if not ready:return color,"criterios_observables"
    if p_correct>=.75:return ("amarillo","evidencia_en_conflicto") if hard else ("verde","modelo_rgb")
    if p_correct<=.25:return ("amarillo","evidencia_en_conflicto") if color=="verde" else ("rojo","modelo_rgb")
    if color=="verde" and p_correct>=.5:return "verde","criterios_observables"
    return "amarillo","modelo_incierto"
