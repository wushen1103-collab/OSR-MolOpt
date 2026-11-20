"""Export long-form numerical result tables from the archived records."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from osrmolopt.data_io import load
OUT=ROOT/'outputs/tables'
OUT.mkdir(parents=True,exist_ok=True)


def table(name,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


record=load('data/records/censoring_decoupling_experiment.json')
rows=[]
for experiment in record['experiments']:
    for condition,values in experiment['conditions'].items():
        for method in ['HIST-SAFE','Vina-MO']:
            rows.append({'level':experiment['level'],'floor':experiment['floor'],'condition':condition,
                         'learning_records':values['training_labels'],'evaluation_records':values['evaluation_labels'],
                         'method':method,**values[method]})
table('censoring_decoupling.csv',rows)
record=load('data/records/davis_external_blind_results.json')
rows=[{**row,'method':method} for method,results in record['task_results'].items() for row in results]
table('external_task_outcomes.csv',rows)
record=load('data/records/interval_censoring_audit.json')
rows=[]
for method,values in record['method_results'].items():
    for row in values['records']:
        for endpoint in ['target','margin']:
            lower,upper=row[endpoint+'_interval']
            rows.append({'method':method,'task':row['task'],'row_id':row['row_id'],'orientation':row['orientation'],
                         'endpoint':endpoint,'lower':lower,'upper':upper,'sign':row[endpoint+'_status']})
table('external_endpoint_bounds.csv',rows)
print(json.dumps({'status':'pass','tables':3,'output':'outputs/tables'}))
