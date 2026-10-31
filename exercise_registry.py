"""Ejercicios de la interfaz; conserva intactas las firmas RGB anteriores."""
from geometry import EXERCISES as LEGACY_EXERCISES, TARGETS as LEGACY_TARGETS
EXERCISES={k:v for k,v in LEGACY_EXERCISES.items() if k!='shoulder_abduction'}
EXERCISES={'elbow_flexion':EXERCISES['elbow_flexion'],'squat':EXERCISES['squat'],
           'shoulder_press':'Press de hombros','inline_lunge':EXERCISES['inline_lunge']}
TARGETS={k:v for k,v in LEGACY_TARGETS.items() if k!='shoulder_abduction'}
TARGETS['shoulder_press']=150
