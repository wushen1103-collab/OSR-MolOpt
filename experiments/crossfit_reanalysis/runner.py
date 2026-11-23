"""Single-CPU cross-fitting reanalysis. No docking or GPU dependency."""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
 os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,copy,hashlib,json,sys,time,subprocess
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
sys.path.append(str(ROOT.parents[1]))
from osrmolopt import models,interventions,decoupling
INPUT=ROOT/'inputs';OUTPUT=ROOT/'outputs'
def load(name):return json.loads((INPUT/name).read_text())
def atomic(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def fingerprint():
 h=hashlib.sha256(Path(__file__).read_bytes())
 for p in sorted(INPUT.glob('*.json')):h.update(p.name.encode());h.update(p.read_bytes())
 for p in sorted((ROOT/'code/osrmolopt').glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
 return h.hexdigest()
def features(task,crossfit):
 # All label-derived predictors, scaling and descriptors for a donor calibration
 # row are constructed without that row. Outer response units remain six-shot.
 x,yt,ym=models.task_arrays(task,task['cal']+task['evaluation'],'vina_only')
 if crossfit:
  for i,row in enumerate(task['cal']):
   five=copy.deepcopy(task);five['cal']=[r for j,r in enumerate(task['cal']) if j!=i]
   x[i]=models.task_arrays(five,[row],'vina_only')[0][0]
 return x,yt,ym
def fit_choices(tasks,i,cache):
 held=decoupling.label_blind_selection_task(tasks[i]);rows=held['evaluation']
 xh,yt,ym=models.task_arrays(held,held['cal']+rows,'vina_only')
 choices={'Vina-MO':decoupling.vina_pick(held)}
 for label,kind in [('Theil-Sen','theilsen'),('Ridge','ridge')]:
  pt,pm=models.local_predictions(held,rows,kind);choices[label]=models.select(rows,pt,pm)
 donor=[name for name in cache if name!=held['task']]
 x=np.vstack([cache[n][0] for n in donor]);targets=[np.concatenate([cache[n][k] for n in donor]) for k in [1,2]]
 for family in ['hist','ridge']:
  predictions=[]
  for y in targets:
   if family=='hist':model=HistGradientBoostingRegressor(max_iter=120,learning_rate=.04,max_leaf_nodes=10,min_samples_leaf=15,l2_regularization=3.,random_state=20261117)
   else:model=make_pipeline(StandardScaler(),Ridge(alpha=10.,solver='lsqr'))
   model.fit(x,y);predictions.append(model.predict(xh))
  pt,pm=predictions
  if family=='hist':
   choices['Global Hist']=models.select(rows,pt[6:],pm[6:])
   pt,pm=models.adapt(pt,pm,yt,ym,'bias50');choices['HIST-SAFE']=models.select(rows,pt[6:],pm[6:])
  else:choices['Global Ridge']=models.select(rows,pt[6:],pm[6:])
 return choices
def checkpoint(path,fp,compute,fault=False):
 if path.exists():
  d=json.loads(path.read_text())
  if d['fingerprint']==fp:return d['result'],True
 value=compute()
 if fault:
  atomic(path.with_name(path.stem+'.uncommitted.json'),{'fingerprint':fp,'result':value})
  raise RuntimeError('Intentional fault before checkpoint commit')
 atomic(path,{'fingerprint':fp,'result':value});return value,False
def summary(rows):
 d=decoupling.summarize(rows);n=len(rows);rng=np.random.default_rng(20261117);ix=rng.integers(n,size=(20000,n))
 for field in ['dual_success','target','margin','severe']:
  v=np.asarray([r[field] for r in rows],float)
  d[field+'_95_ci']=np.quantile(v[ix].mean(axis=1),[.025,.975]).tolist()
 return d
def summarize_view(raw,clipped,picks,level,retrospective=False):
 methods=list(next(iter(picks['original'].values())))
 result={'level':level,'conditions':{}}
 for label,training,evaluation in [('A','original',raw),('B',level,raw),('C','original',clipped),('D',level,clipped)]:
  outcomes={m:[dict(decoupling.evaluate_pick(t,picks['original' if m=='Vina-MO' else training][t['task']][m]),task=t['task']) for t in evaluation] for m in methods}
  result['conditions'][label]={'methods':{m:summary(rs) for m,rs in outcomes.items()},'paired_vs_vina':{m:decoupling.paired_effect(rs,outcomes['Vina-MO']) for m,rs in outcomes.items()},'oracle_feasible_tasks':decoupling.oracle_count(evaluation),'task_results':outcomes}
 a=result['conditions']['A']['task_results']['HIST-SAFE'];b=result['conditions']['B']['task_results']['HIST-SAFE']
 result['B_minus_A']=decoupling.paired_effect(b,a);result['transition']=decoupling.transition(a,b)
 result['changed_choices']=sum((picks['original'][t['task']]['HIST-SAFE']['row_id'],picks['original'][t['task']]['HIST-SAFE']['orientation'])!=(picks[level][t['task']]['HIST-SAFE']['row_id'],picks[level][t['task']]['HIST-SAFE']['orientation']) for t in raw)
 return result
def one_smoke(fault=False):
 tasks=load('tasks_primary.json');cache={t['task']:features(t,True) for t in tasks}
 # Stronger than checking predictor coefficients: changing the omitted response
 # cannot change any feature of its donor calibration row.
 task=copy.deepcopy(tasks[0]);before=features(task,True)[0][0];task['cal'][0]['delta_target']+=999;task['cal'][0]['delta_margin']-=999
 np.testing.assert_allclose(before,features(task,True)[0][0],rtol=0,atol=0)
 # Evaluation labels must not affect a held choice.
 a=fit_choices(tasks,0,cache);changed=copy.deepcopy(tasks);changed[0]['evaluation'][0]['delta_target']+=999
 assert fit_choices(changed,0,cache)==a
 value,resumed=checkpoint(OUTPUT/'resource_checks'/os.environ.get('OSRMOL_AUDIT_PATH','single')/'fold.json',fingerprint(),lambda:a,fault)
 return {'status':'pass','resumed':resumed,'held_row_feature_label_invariance':True,'evaluation_label_invariance':True,'cpu_threads':1,'GPU':False}
def run(full=False):
 original=load('tasks_original.json');primary=load('tasks_primary.json');small=load('tasks_reduced_pool_sensitivity.json')
 q25=float(np.quantile([p for t in original for r in t['cal']+t['evaluation'] for p in r['potencies']],.25))
 fp=fingerprint();views={'original39_crossfit':original,'primary21_crossfit':primary,'disjoint_reduced_pool':small,
 'Ki_only':[t for t in primary if t['assay_ic50']==0], 'IC50_only':[t for t in primary if t['assay_ic50']==1]}
 allresult={'status':'complete','fingerprint':fp,'q25_floor':q25,'views':{},'GPU':False,'cpu_threads':1}
 reused=0
 for view,tasks in views.items():
  picks={};leveltasks={}
  levels=[('original',None),('floor5',5.),('q25',q25)]
  if view=='primary21_crossfit':
   slots=[p for t in original for r in t['cal']+t['evaluation'] for p in r['potencies']]
   levels += [('q50',float(np.quantile(slots,.5))),('q75',float(np.quantile(slots,.75)))]
  for level,floor in levels:
   transformed=interventions.censor_tasks(tasks,floor);leveltasks[level]=transformed
   cache={t['task']:features(t,True) for t in transformed};picks[level]={}
   for i,t in enumerate(transformed):
    value,hit=checkpoint(OUTPUT/'checkpoints'/view/level/f'{i:02d}.json',fp,lambda:fit_choices(transformed,i,cache));reused+=int(hit);picks[level][t['task']]=value
   print(json.dumps({'view':view,'level':level,'tasks':len(tasks)}),flush=True)
  experiments=[summarize_view(tasks,leveltasks[level],picks,level) for level in ['floor5','q25']]
  curves=[]
  for level,floor in levels:
   evaluation=leveltasks[level];methods=list(next(iter(picks[level].values())))
   outcomes={m:[dict(decoupling.evaluate_pick(t,picks['original' if m=='Vina-MO' else level][t['task']][m]),task=t['task']) for t in evaluation] for m in methods}
   slots=np.array([p for t in tasks for r in t['cal']+t['evaluation'] for p in r['potencies']])
   curves.append({'level':level,'floor':floor,'individual_measurement_censor_rate':float(np.mean(slots<floor)) if floor is not None else 0.,'oracle_feasible_tasks':decoupling.oracle_count(evaluation),'task_results':outcomes,'summaries':{m:summary(rs) for m,rs in outcomes.items()}})
  allresult['views'][view]={'tasks':len(tasks),'experiments':experiments,'picks':picks,'joint_curve':curves}
  atomic(OUTPUT/(view+'.json'),allresult['views'][view])
 allresult['resumed_folds']=reused
 import platform,importlib.metadata
 allresult['environment']={'python':platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn']}}
 atomic(OUTPUT/'crossfit_results.json',allresult)
 print(json.dumps({'status':'complete','views':{k:v['tasks'] for k,v in allresult['views'].items()},'resumed':reused}),flush=True)
def bounded(mode,timeout,extra=[]):
 p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),mode]+extra,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 try:out,_=p.communicate(timeout=timeout);return p.returncode,out
 except subprocess.TimeoutExpired:p.kill();out,_=p.communicate();return 124,out
def audit():
 i=0
 while (OUTPUT/'resource_checks'/f'audit_{i}').exists():i+=1
 os.environ['OSRMOL_AUDIT_PATH']=f'audit_{i}'
 code,out=bounded('_smoke',120,['--fault']);assert code not in [0,124] and not (OUTPUT/'resource_checks'/f'audit_{i}'/'fold.json').exists(),out
 code,out=bounded('_smoke',120);assert code==0,out
 code,out=bounded('_smoke',120);assert code==0 and '"resumed": true' in out,out
 start=time.monotonic();code,out=bounded('_hang',2);assert code==124 and time.monotonic()-start<10
 d={'status':'pass','one_task':'pass','failure_injection':'pass','recovery':'pass','resume':'pass','timeout':'pass','GPU':False,'cpu_threads':1};atomic(OUTPUT/'resource_audit.json',d);print(json.dumps(d))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','run','_smoke','_hang']);p.add_argument('--fault',action='store_true');args=p.parse_args()
 if args.mode=='_hang':time.sleep(30)
 elif args.mode=='_smoke':print(json.dumps(one_smoke(args.fault)))
 elif args.mode=='audit':audit()
 else:
  assert json.loads((OUTPUT/'resource_audit.json').read_text())['status']=='pass'
  run()
