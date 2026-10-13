"""Observaciones de alineación independientes de la segmentación de repeticiones."""
import numpy as np
from squat_alignment import alignment_checks

def add_posture_observations(result):
    result['posture_observations']=[]
    if result['exercise']!='squat' or result['quality']['ready']:return result
    sample=result['sample'];times=np.asarray(sample['times'],float)
    if len(times)<8:return result
    dt=float(np.median(np.diff(times)));window=max(6,int(np.ceil(1./dt)));hop=max(1,window//2)
    observations=[]
    for start in range(0,len(times)-window+1,hop):
        end=start+window
        checks=[c for c in alignment_checks(sample,start,end) if c['status']!='no_evaluable']
        if not checks:continue
        # Estos tramos no se cuentan como repeticiones.
        signature=tuple((c['side'],c['status'],c.get('pattern')) for c in checks)
        if observations and observations[-1]['signature']==signature and times[start]<=observations[-1]['end_time']+dt*1.5:
            observations[-1]['end_time']=float(times[end-1]);observations[-1]['end']=end
            for previous,current in zip(observations[-1]['checks'],checks):
                if (current['value'] or 0)>(previous['value'] or 0):previous.update(current)
        else:
            observations.append(dict(start=start,end=end,start_time=float(times[start]),end_time=float(times[end-1]),
                                     checks=checks,signature=signature,scope='alineacion_visible'))
    for observation in observations:observation.pop('signature')
    result['posture_observations']=observations
    return result

def reading_summary(result):
    sample=result['sample'];visibility=np.asarray(sample['visibility']);detected=np.asarray(sample['detected'],bool)
    leg_names={23:'cadera izquierda',24:'cadera derecha',25:'rodilla izquierda',26:'rodilla derecha',
               27:'tobillo izquierdo',28:'tobillo derecho',29:'talón izquierdo',30:'talón derecho',31:'punta izquierda',32:'punta derecha'}
    summary=dict(detected_fraction=float(np.mean(detected)),
                 ambiguous_fraction=float(np.mean(sample.get('ambiguous',np.zeros(len(detected))))),
                 selected_person=bool(sample.get('person_selection')))
    if result['exercise']=='squat':
        summary['joint_coverage']={name:float(np.mean(detected&(visibility[:,joint]>=.6))) for joint,name in leg_names.items()}
    result['reading']=summary
    return result
