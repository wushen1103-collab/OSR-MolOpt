"""Reconstruct the corrected records and optionally compare regenerated choices."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from osrmolopt import decoupling,interventions

def read(folder,name):return json.loads((ROOT/folder/name).read_text(encoding='utf-8'))
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--refitted',type=Path)
 args=parser.parse_args()
 expected=read('expected','crossfit_results.json')
 primary=read('inputs','tasks_primary.json')
 identities=read('inputs','compound_identity_audit.json')
 sources=read('inputs','selected_source_rows.json')
 views={'primary21_crossfit':primary,'original39_crossfit':read('inputs','tasks_original.json'),
        'disjoint_reduced_pool':read('inputs','tasks_reduced_pool_sensitivity.json'),
        'Ki_only':[t for t in primary if t['assay_ic50']==0],
        'IC50_only':[t for t in primary if t['assay_ic50']==1]}
 checked=0
 for name,tasks in views.items():
  result=expected['views'][name]
  for t in tasks:
   for r in t['cal']+t['evaluation']:
    p=r['potencies'];np.testing.assert_allclose([p[1]-p[0],p[1]-p[0]-(p[3]-p[2])],[r['delta_target'],r['delta_margin']],atol=1e-10,rtol=0)
  for experiment in result['experiments']:
   clipped=interventions.censor_tasks(tasks,5. if experiment['level']=='floor5' else expected['q25_floor'])
   for cell,record in experiment['conditions'].items():
    evaluation=tasks if cell in ['A','B'] else clipped
    learning='original' if cell in ['A','C'] else experiment['level']
    assert record['oracle_feasible_tasks']==decoupling.oracle_count(evaluation)
    for method,outcomes in record['task_results'].items():
     for t,outcome in zip(evaluation,outcomes):
      pick=result['picks']['original' if method=='Vina-MO' else learning][t['task']][method]
      actual=decoupling.evaluate_pick(t,pick)
      assert outcome['task']==t['task']
      for key in ['row_id','orientation','dual_success','severe']:assert actual[key]==outcome[key]
      np.testing.assert_allclose([actual['target'],actual['margin']],[outcome['target'],outcome['margin']],atol=1e-12,rtol=0);checked+=1
     summary=decoupling.summarize(outcomes)
     for key,value in summary.items():assert np.isclose(value,record['methods'][method][key])
     ix=np.random.default_rng(20261117).integers(len(tasks),size=(20000,len(tasks)))
     for field in ['dual_success','target','margin','severe']:
      values=np.asarray([r[field] for r in outcomes],float)
      np.testing.assert_allclose(np.quantile(values[ix].mean(axis=1),[.025,.975]),record['methods'][method][field+'_95_ci'],atol=1e-12,rtol=0)
     assert decoupling.paired_effect(outcomes,record['task_results']['Vina-MO'])==record['paired_vs_vina'][method]
   assert decoupling.paired_effect(experiment['conditions']['B']['task_results']['HIST-SAFE'],experiment['conditions']['A']['task_results']['HIST-SAFE'])==experiment['B_minus_A']
  for curve in result['joint_curve']:
   evaluation=interventions.censor_tasks(tasks,curve['floor'])
   assert curve['oracle_feasible_tasks']==decoupling.oracle_count(evaluation)
   for method,outcomes in curve['task_results'].items():
    for t,outcome in zip(evaluation,outcomes):
     pick=result['picks']['original' if method=='Vina-MO' else curve['level']][t['task']][method]
     actual=decoupling.evaluate_pick(t,pick)
     assert actual['row_id']==outcome['row_id'] and actual['orientation']==outcome['orientation']
     assert actual['dual_success']==outcome['dual_success'] and actual['severe']==outcome['severe']
     np.testing.assert_allclose([actual['target'],actual['margin']],[outcome['target'],outcome['margin']],atol=1e-12,rtol=0);checked+=1
 for task in views['disjoint_reduced_pool']:
  cal={sources[r['row_id']][k] for r in task['cal'] for k in ['Cpd1_CHEMBLID','Cpd2_CHEMBLID']}
  evaluation={sources[r['row_id']][k] for r in task['evaluation'] for k in ['Cpd1_CHEMBLID','Cpd2_CHEMBLID']}
  for key in ['raw_id','parent_id','standard_inchi_key','standardized_parent_inchi_key']:
   assert all(identities[mid][key] for mid in cal|evaluation)
   assert not {identities[mid][key] for mid in cal}&{identities[mid][key] for mid in evaluation}
  assert len({sources[r['row_id']]['sheet'] for r in task['cal']+task['evaluation']})==1
 compared=0
 if args.refitted:
  regenerated=json.loads(args.refitted.read_text(encoding='utf-8'))
  for view,record in expected['views'].items():
   for level,task_picks in record['picks'].items():
    for task,methods in task_picks.items():
     for method,pick in methods.items():
      actual=regenerated['views'][view]['picks'][level][task][method]
      assert (actual['row_id'],actual['orientation'])==(pick['row_id'],pick['orientation']),(view,level,task,method)
      compared+=1
 print(json.dumps({'status':'pass','reconstructed_outcomes':checked,'identity_levels':4,'primary_tasks':21,'mean_gain_intervals':'pass','refitted_choices_compared':compared}))

if __name__=='__main__':main()
