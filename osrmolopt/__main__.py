"""Command-line entry point with bounded execution and explicit CPU limits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Set before importing scientific libraries. This process uses one CPU thread.
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def bounded(command, timeout, expect_failure=False):
    started = time.monotonic()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    try:
        out, _ = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        out, _ = child.communicate()
        return {'returncode':124,'timed_out':True,'seconds':time.monotonic()-started,'output':out}
    return {'returncode':child.returncode,'timed_out':False,'seconds':time.monotonic()-started,'output':out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','smoke','refit','_smoke','_refit','_hang'])
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    parser.add_argument('--timeout',type=float,default=1800)
    parser.add_argument('--inject-fault',action='store_true')
    parser.add_argument('--include-response-curve',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command == '_hang':
        time.sleep(30)
        return
    if args.command == 'verify':
        from .verify import run
        result = run()
        (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result))
        return
    if args.command == '_smoke':
        from .refit import smoke
        _, resumed = smoke(args.output,inject_fault=args.inject_fault)
        print(json.dumps({'status':'pass','resumed':resumed,'gpu_used':False}))
        return
    if args.command == '_refit':
        from .refit import run
        run(args.output,response_curve=args.include_response_curve)
        return
    if args.command == 'refit':
        command=[sys.executable,'-m','osrmolopt','_refit','--output',str(args.output)]
        if args.include_response_curve:
            command.append('--include-response-curve')
        result=bounded(command,args.timeout)
        print(result['output'],end='')
        if result['returncode']:
            raise SystemExit(result['returncode'])
        return
    if args.command == 'smoke':
        probe=args.output/'control_check'
        probe.mkdir(exist_ok=True)
        # Each invocation gets a fresh fault-test destination, without deleting prior work.
        i=0
        while (probe/f'fault_{i}').exists():i+=1
        fault=probe/f'fault_{i}'
        command=[sys.executable,'-m','osrmolopt','_smoke','--output',str(fault)]
        first=bounded(command+['--inject-fault'],90)
        assert first['returncode'] not in [0,124] and not (fault/'smoke.json').exists(), first
        recovered=bounded(command,90)
        assert recovered['returncode']==0 and (fault/'smoke.json').exists(),recovered
        resumed=bounded(command,90)
        assert resumed['returncode']==0 and '"resumed": true' in resumed['output'],resumed
        timeout=bounded([sys.executable,'-m','osrmolopt','_hang','--output',str(probe)],2)
        assert timeout['timed_out'] and timeout['seconds']<10,timeout
        result={'status':'pass','one_task_fit':'pass','fault_before_commit':'pass','recovery':'pass',
                'checkpoint_resume':'pass','timeout_exit':'pass','gpu_used':False,'cpu_threads':1}
        (args.output/'resource_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
