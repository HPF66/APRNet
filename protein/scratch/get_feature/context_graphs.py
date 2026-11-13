"""Structure preparation independent of labels, embeddings and data partitions."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, FastMMCIFParser, MMCIF2Dict, ShrakeRupley, is_aa
from Bio.PDB.DSSP import make_dssp_dict
from Bio.Data.PDBData import residue_sasa_scales
from protein.utils.graph_config import SOURCE, GRAPH, NODE_DIM, EDGE_DIM, atomic, sha
from protein.scratch.get_feature.backbone_graphs import extract_atoms, table

OLD_GRAPH = SOURCE / 'structure_features/residue_graph_v1_20261008'
STRUCTURE_ROOTS = ()
DSSP_EXECUTABLE = None
from protein.utils.structure_paths import resolve_structure


def dssp_program():
    return DSSP_EXECUTABLE or shutil.which('mkdssp') or shutil.which('dssp')


SS_CODES = 'HBEGITS-'
MAX_ASA = residue_sasa_scales['Sander']


def as_list(value):
    return value if isinstance(value, list) else [value]


def assembly_groups(path, sample):
    """Accept only coordinates supported by an explicit identity assembly operation."""
    if sample['assembly_mode'].startswith('biological_assembly'):
        return [set(sample['target_chain_ids'].split(';'))], 'prebuilt_biological_assembly'
    groups = []
    try:
        if path.suffix.lower() == '.cif':
            cif = MMCIF2Dict.MMCIF2Dict(str(path))
            op_ids = as_list(cif.get('_pdbx_struct_oper_list.id', []))
            identity = set()
            for i, op_id in enumerate(op_ids):
                matrix = np.array([[float(as_list(cif[f'_pdbx_struct_oper_list.matrix[{a}][{b}]'])[i]) for b in range(1, 4)] for a in range(1, 4)])
                vector = np.array([float(as_list(cif[f'_pdbx_struct_oper_list.vector[{a}]'])[i]) for a in range(1, 4)])
                if np.allclose(matrix, np.eye(3), atol=1e-5) and np.allclose(vector, 0, atol=1e-5):
                    identity.add(str(op_id))
            labels = as_list(cif.get('_atom_site.label_asym_id', []))
            auths = as_list(cif.get('_atom_site.auth_asym_id', labels))
            mapping = defaultdict(set)
            for label, auth in zip(labels, auths):
                mapping[label].add(auth)
            ids = as_list(cif.get('_pdbx_struct_assembly_gen.assembly_id', []))
            operators = as_list(cif.get('_pdbx_struct_assembly_gen.oper_expression', []))
            chains = as_list(cif.get('_pdbx_struct_assembly_gen.asym_id_list', []))
            first = ids[0] if ids else None
            for aid, expression, chain_list in zip(ids, operators, chains):
                if aid == first and expression.strip('() ') in identity:
                    groups.append(set().union(*(mapping[c] for c in chain_list.split(',') if c in mapping)))
        else:
            active = False
            chains = set()
            matrices = defaultdict(list)
            for line in path.read_text(errors='replace').splitlines():
                if not line.startswith('REMARK 350'):
                    continue
                if 'BIOMOLECULE:' in line:
                    active = line.split('BIOMOLECULE:', 1)[1].strip().split(',')[0] == '1'
                if not active:
                    continue
                if 'APPLY THE FOLLOWING TO CHAINS:' in line:
                    chains = {c.strip() for c in line.split('CHAINS:', 1)[1].split(',') if c.strip()}
                elif 'AND CHAINS:' in line:
                    chains.update(c.strip() for c in line.split('CHAINS:', 1)[1].split(',') if c.strip())
                match = re.search(r'BIOMT([123])\s+(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)', line)
                if match:
                    row, operation = int(match[1]), match[2]
                    if row == 1:
                        matrices[operation] = []
                    matrices[operation].append([float(match[k]) for k in range(3, 7)])
                    if row == 3 and len(matrices[operation]) == 3:
                        transform = np.asarray(matrices[operation])
                        if np.allclose(transform[:, :3], np.eye(3), atol=1e-5) and np.allclose(transform[:, 3], 0, atol=1e-5):
                            groups.append(chains.copy())
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return [], 'assembly_metadata_unresolved:' + type(error).__name__
    return groups, 'identity_operation_in_first_biological_assembly' if groups else 'no_verified_assembly_contact'


def annotation(model, path):
    residues = {(chain.id, str(residue.id[1]) + residue.id[2].strip()): residue
                for chain in model for residue in chain if is_aa(residue, standard=False)}
    result = {}
    dssp_error = None
    (GRAPH / '_temporary').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dssp_', dir=GRAPH / '_temporary') as directory:
        output = Path(directory) / 'structure.dssp'
        try:
            program = dssp_program()
            if not program:
                raise FileNotFoundError('DSSP/mkdssp not found; using masked SS and SASA fallback')
            process = subprocess.run([program, '--output-format=dssp', str(path), str(output)],
                                     capture_output=True, text=True, timeout=45)
            if process.returncode or not output.is_file():
                raise RuntimeError(process.stderr[-160:])
            entries, _ = make_dssp_dict(str(output))
            for (chain_id, residue_id), value in entries.items():
                key = (chain_id, str(residue_id[1]) + residue_id[2].strip())
                if key not in residues:
                    continue
                residue = residues[key]
                if residue.resname not in MAX_ASA:
                    continue
                rsa = np.clip(float(value[2]) / MAX_ASA[residue.resname], 0., 1.)
                result[key] = (value[1] if value[1] in SS_CODES else '-', rsa, True)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError) as error:
            dssp_error = type(error).__name__ + ':' + str(error)[:160]
    # DSSP can fail on incomplete backbone structures. Their coordinates remain
    # usable; exposure is independently computed and secondary structure is masked.
    need_fallback = any(key not in result for key, residue in residues.items() if all(a in residue for a in ('N', 'CA', 'C')))
    sasa_error = None
    if need_fallback:
        try:
            for chain in list(model):
                for residue in list(chain):
                    if not is_aa(residue, standard=False):
                        chain.detach_child(residue.id)
            ShrakeRupley(n_points=50).compute(model, level='R')
            for key, residue in residues.items():
                if key not in result and residue.resname in MAX_ASA and all(a in residue for a in ('N', 'CA', 'C')):
                    result[key] = (None, np.clip(float(residue.sasa) / MAX_ASA[residue.resname], 0., 1.), True)
        except (ValueError, KeyError, AttributeError, ZeroDivisionError) as error:
            sasa_error = type(error).__name__ + ':' + str(error)[:160]
    return result, dssp_error, sasa_error


def contact_graph(xyz, positions, instances, chain_ids, groups, cutoff=10., neighbors=16):
    pairs = {}
    for instance in range(len(chain_ids)):
        nodes = np.flatnonzero(instances == instance)
        if not len(nodes):
            continue
        distances, nearest = cKDTree(xyz[nodes]).query(xyz[nodes], k=min(len(nodes), neighbors + 1))
        if distances.ndim == 1:
            distances, nearest = distances[:, None], nearest[:, None]
        by_position = {int(positions[node]): int(node) for node in nodes}
        for local_target, (ds, ns) in enumerate(zip(distances, nearest)):
            target = int(nodes[local_target])
            for distance, source_local in zip(ds, ns):
                source = int(nodes[source_local])
                if source != target and distance <= cutoff:
                    pairs[(source, target)] = 1
            # Backbone continuity is always retained, even in crowded fibrils.
            for delta in (-1, 1):
                source = by_position.get(int(positions[target]) + delta)
                if source is not None and 2.8 <= np.linalg.norm(xyz[source] - xyz[target]) <= 4.5:
                    pairs[(source, target)] = 0
    verified = {i: {g for g, chains in enumerate(groups) if chain_ids[i] in chains} for i in range(len(chain_ids))}
    if any(len(chains) > 1 for chains in groups) and len(xyz):
        tree = cKDTree(xyz)
        for target, close in enumerate(tree.query_ball_point(xyz, cutoff)):
            eligible = [source for source in close if instances[source] != instances[target]
                        and verified[int(instances[source])] & verified[int(instances[target])]]
            eligible.sort(key=lambda source: (round(float(np.linalg.norm(xyz[source] - xyz[target])), 5), source))
            for source in eligible[:neighbors]:
                pairs[(source, target)] = 2
    keys = sorted(pairs)
    edges = np.array(keys, dtype=np.int32).reshape(-1, 2).T
    attrs = np.zeros((len(keys), EDGE_DIM), np.float32)
    if len(keys):
        source, target = edges
        distances = np.linalg.norm(xyz[source] - xyz[target], axis=1)
        attrs[:, :16] = np.exp(-((distances[:, None] - np.linspace(0., 20., 16)) / 1.25) ** 2)
        types = np.array([pairs[key] for key in keys])
        attrs[np.arange(len(keys)), 16 + types] = 1.
        same = instances[source] == instances[target]
        displacement = np.where(same, positions[source] - positions[target], 0)
        attrs[:, 19] = np.tanh(displacement / 32.)
        attrs[:, 20] = np.log1p(np.abs(displacement)) / 8.
        attrs[:, 21] = distances / 20.
    return edges, attrs


def build(task):
    sample, rows, sequences = task
    sample_id = sample['sample_id']
    output = GRAPH / f'{sample_id}.npz'
    metadata = GRAPH / f'{sample_id}.json'
    path = resolve_structure(sample, SOURCE, STRUCTURE_ROOTS)
    if output.exists() and metadata.exists():
        record = json.loads(metadata.read_text())
        if record['structure_sha256'] != sha(path):
            raise ValueError(f'Structure changed for {sample_id}; choose a fresh graph output directory')
        if record.get('annotation_policy') == 'dssp_or_sasa_v2' and record.get('dssp_program') == dssp_program():
            return record
    parser = PDBParser(QUIET=True) if path.suffix.lower() == '.pdb' else FastMMCIFParser(QUIET=True)
    model = next(parser.get_structure(sample_id, str(path)).get_models())
    groups, assembly_status = assembly_groups(path, sample)
    with np.load(OLD_GRAPH / f'{sample_id}.npz') as old:
        old_values = {key: np.array(old[key]) for key in ('node_seq', 'positions', 'node', 'instance')}
    sids = list(dict.fromkeys(sample['sequence_ids'].split(';')))
    xyz_parts = []
    keys = []
    instance_to_seq = []
    for instance, row in enumerate(rows):
        atoms, present, mismatch = extract_atoms(model[row['chain_id']], row, sequences[row['sequence_id']])
        selected = old_values['instance'] == instance
        pos = old_values['positions'][selected]
        assert np.all(present[pos, 1] & ~mismatch[pos])
        xyz_parts.append(atoms[pos, 1])
        key_by_position = dict(zip(map(lambda x: int(x) - 1, row['coordinate_seqres_positions'].split(',')), row['coordinate_residue_keys'].split(',')))
        keys.extend((row['chain_id'], key_by_position[int(position)]) for position in pos)
        instance_to_seq.append(sids.index(row['sequence_id']))
    xyz = np.concatenate(xyz_parts)
    annotations, dssp_error, sasa_error = annotation(model, path)
    node = np.zeros((len(xyz), NODE_DIM), np.float32)
    node[:, :11] = old_values['node']
    for index, key in enumerate(keys):
        if key not in annotations:
            continue
        ss, rsa, rsa_valid = annotations[key]
        if ss is not None:
            node[index, 11 + SS_CODES.index(ss)] = 1.
            node[index, 19] = 1.
        node[index, 20] = rsa
        node[index, 21] = float(rsa_valid)
    edge_index, edge = contact_graph(xyz, old_values['positions'], old_values['instance'],
                                     [r['chain_id'] for r in rows], groups)
    assert np.isfinite(node).all() and np.isfinite(edge).all()
    value = {key: old_values[key] for key in ('node_seq', 'positions', 'instance')}
    value.update(node=node, edge_index=edge_index, edge=edge,
                 instance_to_seq=np.asarray(instance_to_seq, np.int32))
    with output.with_suffix('.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **value)
    output.with_suffix('.npz.tmp').replace(output)
    record = dict(sample_id=sample_id, file=output.name, sequence_ids=sids,
                  nodes=len(node), edges=edge.shape[0], instances=len(rows),
                  secondary_structure_nodes=int(node[:, 19].sum()), exposure_nodes=int(node[:, 21].sum()),
                  verified_interchain_edges=int(edge[:, 18].sum()),
                  assembly_status=assembly_status, verified_chain_groups=[sorted(g) for g in groups],
                  dssp_error=dssp_error, dssp_program=dssp_program(), annotation_policy='dssp_or_sasa_v2',
                  sasa_error=sasa_error, structure_sha256=sha(path))
    atomic(metadata, record)
    return record


def main(workers):
    GRAPH.mkdir(parents=True, exist_ok=True)
    (GRAPH / '_temporary').mkdir(exist_ok=True)
    protected = {str(SOURCE / name): sha(SOURCE / name) for name in ('labels.tsv', 'chains.tsv', 'sequences.tsv', 'provenance.tsv')}
    old_manifest = json.loads((OLD_GRAPH / 'manifest.json').read_text())
    assert old_manifest['validation'] == 'passed'
    sequences = {r['sequence_id']: r['sequence'] for r in table(SOURCE / 'sequences.tsv')}
    chains = defaultdict(list)
    for row in table(SOURCE / 'chains.tsv'):
        chains[row['sample_id']].append(row)
    samples = table(SOURCE / 'labels.tsv')
    tasks = [(sample, chains[sample['sample_id']], sequences) for sample in samples]
    records = []
    errors = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build, task): task[0]['sample_id'] for task in tasks}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as error:
                errors.append({'sample_id': futures[future], 'error': repr(error)})
            if (len(records) + len(errors)) % 20 == 0:
                progress = dict(stage='building_structure_features', complete=len(records), total=len(tasks),
                                errors=errors, elapsed_seconds=time.time() - started)
                atomic(GRAPH / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    assert not errors, errors
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    records.sort(key=lambda record: record['sample_id'])
    atomic(GRAPH / 'manifest.json', dict(validation='passed', samples=len(records), records=records,
        nodes=sum(r['nodes'] for r in records), edges=sum(r['edges'] for r in records),
        chain_instances=sum(r['instances'] for r in records),
        secondary_structure_nodes=sum(r['secondary_structure_nodes'] for r in records),
        exposure_nodes=sum(r['exposure_nodes'] for r in records),
        samples_with_verified_interchain_edges=sum(r['verified_interchain_edges'] > 0 for r in records),
        feature_schema='backbone_geometry_11 + DSSP_onehot_8 + SS_valid + RSA + RSA_valid',
        edge_schema='distance_RBF_16 + backbone/intrachain/interchain_onehot_3 + signed_sequence_distance + log_sequence_distance + CA_distance',
        max_neighbors=16, cutoff_angstrom=10., node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        selected_model='first coordinate model', protected_input_sha256=protected,
        label_or_experimental_method_features=False, elapsed_seconds=time.time() - started))
    print(json.dumps({'event': 'structure_features_complete', 'samples': len(records), 'seconds': time.time() - started}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    main(parser.parse_args().workers)
