"""Build primary per-sample/per-chain graphs without changing any original inputs."""
import csv, hashlib, json, os, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, FastMMCIFParser, is_aa
from Bio.SeqUtils import seq1
from protein.scratch.get_feature.geometry import chain_geometry
from protein.utils.structure_paths import resolve_structure
STRUCTURE_ROOTS = ()

DATA=Path(__file__).resolve().parents[2] / 'data'
CACHE=DATA/'structure_features/backbone'
CUSTOM={'MSE':'M','SEC':'U','PYL':'O','HYP':'P'}
def table(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def extract_atoms(chain,row,sequence):
    atoms=np.zeros((len(sequence),3,3),np.float32);present=np.zeros((len(sequence),3),bool)
    mismatches=np.zeros(len(sequence),bool)
    residues=defaultdict(list)
    for residue in chain:
        if residue.id[0]!='W':residues[str(residue.id[1])+str(residue.id[2]).strip()].append(residue)
    for key,position,expected in zip(row['coordinate_residue_keys'].split(','),
                                   map(int,row['coordinate_seqres_positions'].split(',')), row['coordinate_sequence']):
        choices=[]
        for option in residues[key]:
            for variant in list(option.child_dict.values()) if option.is_disordered()==2 else [option]:
                aa=seq1(variant.resname,custom_map=CUSTOM,undef_code='X')
                if aa==expected or (expected=='X' and not is_aa(variant,standard=True)):choices.append(variant)
        assert choices,(row['sample_id'],row['chain_id'],key)
        def priority(r):
            return ('CA' in r,(r['CA'].get_occupancy() or 0.) if 'CA' in r else 0.,r.id[0]==' ')
        residue=max(choices,key=priority);i=position-1
        actual=seq1(residue.resname,custom_map=CUSTOM,undef_code='X')
        mismatches[i]=expected!='X' and actual!='X' and actual!=sequence[i]
        for slot,name in enumerate(('N','CA','C')):
            if name in residue:
                atoms[i,slot]=residue[name].coord;present[i,slot]=True
    assert np.isfinite(atoms).all()
    return atoms,present,mismatches

def build(task):
    sample,rows,sequences=task;sample_id=sample['sample_id']
    path=resolve_structure(sample, DATA, STRUCTURE_ROOTS);output=CACHE/f'{sample_id}.npz'
    parser=PDBParser(QUIET=True) if path.suffix.lower()=='.pdb' else FastMMCIFParser(QUIET=True)
    model=next(parser.get_structure(sample_id,str(path)).get_models())
    assert sample['selected_model']=='1'
    sids=list(dict.fromkeys(sample['sequence_ids'].split(';')))
    node_seq,node_pos,node_attr,edges,edge_attr,ranks,instance=[],[],[],[],[],[],[]
    diagnostics=[];offset=0
    for ci,row in enumerate(rows):
        sid=row['sequence_id'];sequence=sequences[sid]
        atoms,present,mismatch=extract_atoms(model[row['chain_id']],row,sequence)
        graph=chain_geometry(atoms,present,mismatch)
        n=len(graph['positions'])
        node_seq.extend([sids.index(sid)]*n);node_pos.extend(graph['positions']);node_attr.append(graph['node'])
        edges.append(graph['edge_index']+offset);edge_attr.append(graph['edge']);ranks.append(graph['edge_rank'])
        instance.extend([ci]*n);offset+=n
        e=graph['edge_index'];pos=graph['positions'];seq_edges=graph['edge'][:,18]>.5
        assert np.all(np.abs(pos[e[0,seq_edges]]-pos[e[1,seq_edges]])==1)
        diagnostics.append(dict(chain_id=row['chain_id'],sequence_id=sid,length=len(sequence),
                                nodes=n,edges=e.shape[1],mismatch_masked=int(mismatch.sum()),
                                no_ca=int((~present[:,1]).sum())))
    value=dict(node_seq=np.asarray(node_seq,np.int16),positions=np.asarray(node_pos,np.int32),
               node=np.concatenate(node_attr),edge_index=np.concatenate(edges,axis=1),
               edge=np.concatenate(edge_attr),edge_rank=np.concatenate(ranks),
               instance=np.asarray(instance,np.int16))
    for key in ('node','edge'):assert np.isfinite(value[key]).all()
    with output.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,**value)
    output.with_suffix('.npz.tmp').replace(output)
    return dict(sample_id=sample_id,file=output.name,sequence_ids=sids,chains=diagnostics,
                nodes=len(node_pos),edges=value['edge_index'].shape[1],structure_sha256=sha(path))


def initialize_worker(data, cache, structure_roots):
    """Pass runtime paths explicitly, including with spawn/forkserver workers."""
    global DATA, CACHE, STRUCTURE_ROOTS
    DATA, CACHE = Path(data), Path(cache)
    STRUCTURE_ROOTS = tuple(map(Path, structure_roots))


def main(workers=8):
    CACHE.mkdir(parents=True,exist_ok=True)
    protected={str(DATA/name):sha(DATA/name) for name in ('labels.tsv','chains.tsv','sequences.tsv','provenance.tsv')}
    sequences={r['sequence_id']:r['sequence'] for r in table(DATA/'sequences.tsv')}
    chains=defaultdict(list)
    for row in table(DATA/'chains.tsv'):chains[row['sample_id']].append(row)
    tasks=[(r,chains[r['sample_id']],sequences) for r in table(DATA/'labels.tsv')]
    records=[];start=time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                             initargs=(DATA, CACHE, STRUCTURE_ROOTS)) as pool:
        for record in pool.map(build,tasks,chunksize=4):
            records.append(record)
            if len(records)%100==0:print(json.dumps({'completed':len(records),'total':len(tasks)}),flush=True)
    assert all(sha(Path(p))==s for p,s in protected.items())
    atomic(CACHE/'manifest.json',dict(validation='passed',records=records,protected_input_sha256=protected,
        samples=len(records),chain_instances=sum(len(r['chains']) for r in records),
        nodes=sum(r['nodes'] for r in records),edges=sum(r['edges'] for r in records),
        first_model_only=True,inter_chain_edges=False,mismatch_coordinates_masked=True,
        full_sequence_positions_preserved=True,max_neighbors=32,max_distance=16.,elapsed_seconds=time.time()-start))
    print(json.dumps({'event':'graphs_complete','samples':len(records),'elapsed_seconds':time.time()-start}),flush=True)
if __name__=='__main__':main()
