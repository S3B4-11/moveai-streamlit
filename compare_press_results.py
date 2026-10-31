"""Auditoría offline; las etiquetas nunca intervienen en la inferencia de la app."""
import argparse,json
from collections import Counter
from pathlib import Path

def load(path):
    data=json.loads(Path(path).read_text());rows=data.get('rows',data.get('videos',[]))
    result={r['file']:r for r in rows}
    if len(result)!=len(rows) or not rows:raise ValueError('Conjunto vacío o archivos duplicados.')
    if any(r['status']=='processing_error' for r in rows):raise ValueError('Hay errores de procesamiento sin resolver.')
    return result

def compare(old,new):
    if set(old)!=set(new):raise ValueError('No se comparan los mismos videos.')
    if any(old[k]['label']!=new[k]['label'] for k in old):raise ValueError('Las etiquetas cambiaron entre versiones.')
    def summarize(rows):
        counts=Counter((r['label'],r['status']) for r in rows.values())
        return dict(n=len(rows),matches=counts['correct','correct']+counts['incorrect','incorrect'],
                    false_alerts=counts['correct','incorrect'],false_acceptances=counts['incorrect','correct'],
                    unreadable=sum(r['status']=='unreadable' for r in rows.values()),
                    states=dict(Counter(r['status'] for r in rows.values())))
    new_alerts=[k for k,r in new.items() if r['label']=='correct' and r['status']=='incorrect' and old[k]['status']!='incorrect']
    new_acceptances=[k for k,r in new.items() if r['label']=='incorrect' and r['status']=='correct' and old[k]['status']!='correct']
    return dict(scope='Comparación de desarrollo sobre videos ya utilizados; no prueba independiente de generalización.',
                baseline=summarize(old),candidate=summarize(new),new_false_alerts=new_alerts,new_false_acceptances=new_acceptances,
                no_new_observed_misclassifications=not new_alerts and not new_acceptances,
                transitions=dict(Counter(old[k]['status']+' -> '+r['status'] for k,r in new.items())))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('--output',default='comparison.json');a=p.parse_args()
    result=compare(load(a.baseline),load(a.candidate));Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
