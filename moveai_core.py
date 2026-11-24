"""API pública: video, repeticiones y evidencia de cámara."""
from analysis import analyze
from pose import extract_video
try:
    from inference_router import predict
    ROUTER_CONTEXT=True
except ImportError:
    from camera_model import predict
    ROUTER_CONTEXT=False
from squat_alignment import add_alignment
from decision_policy import apply_policy
from repetition_quality import repair_repetition_quality
from video_orientation import prepare_video
from frontal_quality import repair_frontal_projection,predict_resolved_frontal
from posture_review import add_posture_observations,reading_summary
from pose_selection import extract_selected_video,validate_target

def evaluate(path,exercise,side="automatica",target=None,rotation="auto",person_target=None):
    person_target=validate_target(person_target).tolist() if person_target is not None else None
    def extract(video,selection):
        return extract_video(video) if selection is None else extract_selected_video(video,selection)
    # El navegador ya presenta el video según su matriz de orientación.
    # Se prepara esa misma imagen ANTES de ejecutar MediaPipe.
    analysis_video,orientation=prepare_video(path,rotation)
    manual=orientation['manual_rotation']
    turn=(orientation['metadata_rotation']+manual+180)%360-180
    selected=person_target
    if selected is not None and manual:
        x,y=selected
        selected={90:[1-y,x],-90:[y,1-x],180:[1-x,1-y]}[manual]
    try:
        return _evaluate_prepared(analysis_video,exercise,side,target,selected,person_target,
                                  orientation,turn,extract)
    except Exception:
        if orientation['normalized']:analysis_video.unlink(missing_ok=True)
        raise

def _evaluate_prepared(analysis_video,exercise,side,target,selected,person_target,orientation,turn,extract):
    sample=extract(analysis_video,selected)
    result=analyze(sample,exercise,side,target)
    repair_frontal_projection(result)
    if person_target is None:
        model=predict(result["sample"],exercise,quality_context=result) if ROUTER_CONTEXT else predict(result['sample'],exercise)
        if not model['available']:model=predict_resolved_frontal(result['sample'],exercise,result,model)
    else:
        model=dict(available=False,reason='Se revisan criterios visibles de la persona seleccionada. El clasificador global todavía no está validado para este modo de seguimiento.')
        result['sample']['trainable']=False
    result["model"]=model
    result.update(analysis_video=str(analysis_video),orientation_rotation=turn,
                  video_orientation=orientation)
    return apply_policy(reading_summary(add_posture_observations(add_alignment(repair_repetition_quality(result)))))

def diagnostic(result):
    import math
    import numpy as np
    payload={k:v for k,v in result.items() if k not in ["sample","tracks","analysis_video"]}
    payload["tracks"]=[{k:v for k,v in t.items() if k not in ["series","cycle_series","valid"]} for t in result["tracks"]]
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    return clean(payload)
