"""Archived CPU docking preparation and scoring, with portable output paths."""
from __future__ import annotations
import os, json, hashlib, re, subprocess
from pathlib import Path
from collections import defaultdict
import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'artifacts/docking'
AA3 = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','MSE'}
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)

def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 2147483646 + 1

def run(command: list[str], timeout: int, log: Path) -> subprocess.CompletedProcess:
    log.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False, env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    log.write_text('COMMAND=' + json.dumps(command) + '\nSTDOUT\n' + done.stdout + '\nSTDERR\n' + done.stderr)
    if done.returncode != 0:
        raise RuntimeError(f'command failed rc={done.returncode}; log={log}')
    return done

def residue_match(residue, spec: dict) -> bool:
    return residue.name == spec['ligand']['resname'] and str(residue.seqid) == spec['ligand']['resid']

def prepare_receptor(target_id: str, protocol: dict) -> dict:
    outdir = WORK / 'receptors' / target_id
    status_path = outdir / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('status') == 'complete':
            return status
    spec = protocol['structures'][target_id]
    source = ROOT / spec['coordinate_path']
    structure = gemmi.read_structure(str(source))
    native = None
    for chain in structure[0]:
        if chain.name == spec['ligand']['chain']:
            for residue in chain:
                if residue_match(residue, spec):
                    native = np.asarray([[a.pos.x, a.pos.y, a.pos.z] for a in residue if a.element.name.upper() != 'H'], dtype=float)
                    break
    if native is None or len(native) == 0:
        raise RuntimeError(f'native ligand not found for {target_id}')
    center = native.mean(0)
    size = np.clip(np.ptp(native, axis=0) + 12.0, 18.0, 30.0)
    model = structure[0]
    wanted = set(spec['target_chains'])
    for i in range(len(model) - 1, -1, -1):
        if model[i].name not in wanted:
            del model[i]
    for chain in model:
        for i in range(len(chain) - 1, -1, -1):
            if chain[i].name.upper() not in AA3:
                del chain[i]
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    if len(model) > len(alphabet):
        raise RuntimeError(f'too many retained chains for PDB normalization: {target_id}')
    chain_id_map = {}
    for index, chain in enumerate(model):
        original = chain.name
        normalized = alphabet[index]
        chain_id_map[original] = normalized
        chain.name = normalized
    stripped_input_hydrogens = 0
    for chain in model:
        for residue in chain:
            for atom_index in range(len(residue) - 1, -1, -1):
                if residue[atom_index].element.name.upper() in {'H', 'D'}:
                    del residue[atom_index]
                    stripped_input_hydrogens += 1
    disulfide_cutoff = float(protocol['receptor_preparation']['disulfide_cutoff_angstrom'])
    cysteine_sg = []
    for chain in model:
        for residue in chain:
            if residue.name.upper() != 'CYS':
                continue
            sg = next((atom for atom in residue if atom.name.strip().upper() == 'SG'), None)
            if sg is not None:
                cysteine_sg.append((chain.name, str(residue.seqid), sg.pos))
    disulfide_pairs = []
    disulfide_residues = set()
    for i, left in enumerate(cysteine_sg):
        for right in cysteine_sg[i + 1:]:
            distance = float(left[2].dist(right[2]))
            if distance <= disulfide_cutoff:
                left_id, right_id = (f'{left[0]}:{left[1]}', f'{right[0]}:{right[1]}')
                disulfide_pairs.append({'left': left_id, 'right': right_id, 'sg_distance': distance})
                disulfide_residues.update((left_id, right_id))
    histidine_templates = []
    for chain in model:
        for residue in chain:
            if residue.name.upper() == 'HIS':
                histidine_templates.append(f'{chain.name}:{residue.seqid}=HIE')
    disulfide_templates = [f'{residue_id}=CYX' for residue_id in sorted(disulfide_residues)]
    while len(structure) > 1:
        del structure[-1]
    outdir.mkdir(parents=True, exist_ok=True)
    clean_pdb = outdir / 'receptor_clean.pdb'
    structure.write_pdb(str(clean_pdb))
    pdbqt = outdir / 'receptor.pdbqt'
    base = outdir / 'receptor'
    tool = protocol['tools']['prepare_receptor']['path']
    command = [tool, '--read_pdb', str(clean_pdb), '-o', str(base), '-p', str(pdbqt), '-a']
    if protocol['receptor_preparation'].get('default_altloc'):
        command.extend(['--default_altloc', protocol['receptor_preparation']['default_altloc']])
    residue_templates = histidine_templates + disulfide_templates
    if residue_templates:
        command.extend(['--set_template', ','.join(residue_templates)])
    command.extend(['--box_center', *[f'{x:.3f}' for x in center], '--box_size', *[f'{x:.3f}' for x in size]])
    run(command, 120, outdir / 'prepare.log')
    status = {'status': 'complete', 'target_id': target_id, 'source_path': spec['coordinate_path'], 'source_sha256': spec['coordinate_sha256'], 'clean_pdb': str(clean_pdb.relative_to(ROOT)), 'clean_pdb_sha256': sha(clean_pdb), 'pdbqt': str(pdbqt.relative_to(ROOT)), 'pdbqt_sha256': sha(pdbqt), 'center': center.tolist(), 'size': size.tolist(), 'chain_id_map': chain_id_map, 'chain_normalization': 'retained chains mapped in model order to A-Z,a-z,0-9', 'histidine_templates': histidine_templates, 'disulfide_templates': disulfide_templates, 'disulfide_pairs': disulfide_pairs, 'stripped_input_hydrogens': stripped_input_hydrogens}
    atomic_json(status_path, status)
    return status

def prepare_ligand(compound_id: str, smiles: str, protocol: dict) -> dict:
    outdir = WORK / 'ligands' / compound_id
    status_path = outdir / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('status') == 'complete':
            return status
    input_mol = Chem.MolFromSmiles(smiles)
    if input_mol is None:
        raise RuntimeError(f'invalid SMILES for {compound_id}')
    fragments = list(Chem.GetMolFrags(input_mol, asMols=True, sanitizeFrags=True))
    if not fragments:
        raise RuntimeError(f'SMILES has no molecular fragments for {compound_id}')
    ranked = sorted(fragments, key=lambda fragment: (any((atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())), fragment.GetNumHeavyAtoms(), Chem.MolToSmiles(fragment, canonical=True)), reverse=True)
    parent = ranked[0]
    parent_smiles = Chem.MolToSmiles(parent, canonical=True)
    mol = Chem.AddHs(parent)
    params = AllChem.ETKDGv3()
    params.randomSeed = stable_seed(compound_id)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f'ETKDG failed for {compound_id}')
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    outdir.mkdir(parents=True, exist_ok=True)
    sdf = outdir / 'ligand.sdf'
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()
    pdbqt = outdir / 'ligand.pdbqt'
    tool = protocol['tools']['prepare_ligand']['path']
    run([tool, '-i', str(sdf), '-o', str(pdbqt)], 120, outdir / 'prepare.log')
    status = {'status': 'complete', 'compound_id': compound_id, 'input_smiles': smiles, 'parent_smiles': parent_smiles, 'input_fragment_count': len(fragments), 'fragment_policy': 'prefer carbon-containing fragment, then heavy atoms, then canonical SMILES', 'sdf': str(sdf.relative_to(ROOT)), 'sdf_sha256': sha(sdf), 'pdbqt': str(pdbqt.relative_to(ROOT)), 'pdbqt_sha256': sha(pdbqt)}
    atomic_json(status_path, status)
    return status

def pose_coords(path: Path) -> np.ndarray:
    points = []
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith(('ATOM  ', 'HETATM')):
            try:
                points.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                pass
    if not points:
        raise RuntimeError(f'pose contains no coordinates: {path}')
    return np.asarray(points)

def score(path: Path) -> float:
    match = re.search('REMARK VINA RESULT:\\s+([-+0-9.]+)', path.read_text(errors='replace'))
    if not match:
        raise RuntimeError(f'Vina score missing: {path}')
    return float(match.group(1))

def contacts(clean_pdb: Path, pose: Path, cutoff: float, residue_types: list[str]) -> dict:
    structure = gemmi.read_structure(str(clean_pdb))
    ligand = pose_coords(pose)
    touched = defaultdict(set)
    minimum = {}
    for chain in structure[0]:
        for residue in chain:
            atoms = np.asarray([[a.pos.x, a.pos.y, a.pos.z] for a in residue if a.element.name.upper() != 'H'], dtype=float)
            if not len(atoms):
                continue
            d = float(np.sqrt(((atoms[:, None, :] - ligand[None, :, :]) ** 2).sum(2)).min())
            key = f'{chain.name}:{residue.seqid}:{residue.name}'
            minimum[key] = d
            if d <= cutoff:
                touched[residue.name].add(key)
    return {'counts_by_residue_type': {aa: len(touched.get(aa, set())) for aa in residue_types}, 'contact_residue_count': sum((len(x) for x in touched.values())), 'minimum_distances': minimum}

def dock_endpoint(endpoint: dict, receptors: dict, ligands: dict, protocol: dict) -> dict:
    key = endpoint['endpoint_key']
    outdir = WORK / 'endpoints' / key
    status_path = outdir / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('status') == 'complete':
            return status
    outdir.mkdir(parents=True, exist_ok=True)
    receptor = receptors[endpoint['target_id']]
    ligand = ligands[endpoint['compound_id']]
    pose = outdir / 'pose.pdbqt'
    seed = stable_seed(key)
    d = protocol['docking']
    command = [protocol['tools']['vina']['path'], '--receptor', str(ROOT / receptor['pdbqt']), '--ligand', str(ROOT / ligand['pdbqt']), '--center_x', str(receptor['center'][0]), '--center_y', str(receptor['center'][1]), '--center_z', str(receptor['center'][2]), '--size_x', str(receptor['size'][0]), '--size_y', str(receptor['size'][1]), '--size_z', str(receptor['size'][2]), '--scoring', d['scoring'], '--cpu', str(d['cpu_per_endpoint']), '--exhaustiveness', str(d['exhaustiveness']), '--num_modes', str(d['num_modes']), '--energy_range', str(d['energy_range']), '--seed', str(seed), '--out', str(pose), '--verbosity', '1']
    run(command, d['timeout_seconds'], outdir / 'vina.log')
    contact = contacts(ROOT / receptor['clean_pdb'], pose, protocol['contact_features']['cutoff_angstrom'], protocol['contact_features']['residue_types'])
    status = {'status': 'complete', **endpoint, 'seed': seed, 'score': score(pose), 'pose': str(pose.relative_to(ROOT)), 'pose_sha256': sha(pose), 'contacts': contact}
    atomic_json(status_path, status)
    return status

def endpoints_for(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        for ci, compound in ((1, row['Cpd1_CHEMBLID']), (2, row['Cpd2_CHEMBLID'])):
            for ti, target in ((1, row['T1[CHEMBLID]']), (2, row['T2[CHEMBLID]'])):
                safe_row = row['row_id'].replace(':', '_')
                output.append({'endpoint_key': f'{safe_row}__c{ci}__t{ti}', 'row_id': row['row_id'], 'compound_index': ci, 'target_index': ti, 'compound_id': compound, 'target_id': target})
    return output
