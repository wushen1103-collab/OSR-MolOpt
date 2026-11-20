"""Explicit single-endpoint CPU docking; no batch or GPU launch."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default='davis_docking')
    parser.add_argument('--row-id',required=True)
    parser.add_argument('--compound-index',type=int,choices=[1,2],default=1)
    parser.add_argument('--target-index',type=int,choices=[1,2],default=1)
    parser.add_argument('--vina',required=True)
    parser.add_argument('--prepare-receptor',required=True)
    parser.add_argument('--prepare-ligand',required=True)
    parser.add_argument('--require-tool-hashes',action='store_true')
    args=parser.parse_args()
    if not args.protocol.replace('_','').isalnum():parser.error('Invalid protocol name')
    protocol=json.loads((ROOT/'data/protocols'/f'{args.protocol}.json').read_text())
    from osrmolopt import docking
    for name,tool in [('vina',args.vina),('prepare_receptor',args.prepare_receptor),('prepare_ligand',args.prepare_ligand)]:
        resolved=shutil.which(tool) or tool
        if not Path(resolved).is_file():parser.error(f'Missing executable for {name}')
        checksum=hashlib.sha256(Path(resolved).read_bytes()).hexdigest()
        expected=protocol['tools'][name]['sha256']
        if args.require_tool_hashes and checksum!=expected:parser.error(f'Tool hash mismatch: {name}')
        protocol['tools'][name]['path']=str(Path(resolved).resolve())
    unpacked=ROOT/'artifacts/structures'
    unpacked.mkdir(parents=True,exist_ok=True)
    with tarfile.open(ROOT/'data/docking/structures.tar.gz') as archive:
        archive.extractall(unpacked,filter='data')
    for structure in protocol['structures'].values():
        name=Path(structure['coordinate_path']).name
        structure['coordinate_path']='artifacts/structures/'+name
    protocol['docking']['cpu_per_endpoint']=1
    row=next((row for row in protocol['panel_rows'] if row['row_id']==args.row_id),None)
    if row is None:parser.error('Row not present in the fixed protocol')
    endpoint=next(e for e in docking.endpoints_for([row]) if e['compound_index']==args.compound_index and e['target_index']==args.target_index)
    target,compound=endpoint['target_id'],endpoint['compound_id']
    receptor=docking.prepare_receptor(target,protocol)
    ligand=docking.prepare_ligand(compound,protocol['compound_smiles'][compound],protocol)
    result=docking.dock_endpoint(endpoint,{target:receptor},{compound:ligand},protocol)
    print(json.dumps({'status':result['status'],'endpoint_key':result['endpoint_key'],'score':result['score'],'seed':result['seed']}))


if __name__=='__main__':main()
