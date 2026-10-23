"""Frozen full-chain ESM-2 extraction with checked cache reuse and resumability."""
from pathlib import Path
import argparse,csv,json,hashlib,os,time,fcntl,shutil
import torch,esm

MODELS={'650m':('esm2_t33_650M_UR50D',1280,33)}
def sha(s):return hashlib.sha256(s.encode('ascii')).hexdigest()
def atomic_json(path,d):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,path)
def load_tensor(p,shape):
    x=torch.load(p,map_location='cpu',weights_only=True)
    if not isinstance(x,torch.Tensor) or tuple(x.shape)!=shape or x.dtype!=torch.float32 or not torch.isfinite(x).all():
        raise ValueError(f'Invalid residue tensor {p}')
    return x.contiguous()
def save_tensor(p,x):
    tmp=p.with_name(p.name+'.tmp');torch.save(x,tmp);os.replace(tmp,p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output-root',required=True)
    ap.add_argument('--model',choices=MODELS,default='650m');ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--reuse-dir');args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=Path(args.dataset);out=Path(args.output_root)/('esm2_'+args.model);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.extraction.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model_name,dim,layer=MODELS[args.model]
    with (dataset/'sequences.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    assert len(rows)==len({r['sequence_id'] for r in rows})
    for r in rows:assert len(r['sequence'])==int(r['sequence_length'])
    sequence_digest=hashlib.sha256(''.join(r['sequence_id']+'\t'+r['sequence']+'\n' for r in rows).encode()).hexdigest()
    records={};pending=[];reused=0;already=0;computed=0;rejected_reuse=[]
    existing_manifest=out/'manifest.json';existing_records={}
    if existing_manifest.exists():
        d=json.loads(existing_manifest.read_text())
        assert d['model']==model_name and d['sequence_table_digest']==sequence_digest
        existing_records={r['sequence_id']:r for r in d['records']}
    reuse={};reuse_dir=Path(args.reuse_dir) if args.reuse_dir else None
    if reuse_dir and (reuse_dir/'manifest.json').exists():
        d=json.loads((reuse_dir/'manifest.json').read_text())
        if d.get('model')==model_name and d.get('representation_layer')==layer and d.get('dtype')=='float32':
            reuse={r['sequence_id']:r for r in d['records']}
    def record(r,source):
        return {'sequence_id':r['sequence_id'],'length':len(r['sequence']),'sequence_sha256':sha(r['sequence']),
            'file':r['sequence_id']+'.pkl','source':source}
    def metadata(status):
        return {'status':status,'model':model_name,'representation_layer':layer,'hidden_dim':dim,'dtype':'float32',
           'format':'torch.save raw CPU Tensor in .pkl, residue order [L,D], BOS/EOS excluded',
           'dataset_directory':str(dataset),'input_sequence_table':str(dataset/'sequences.tsv'),
           'sequence_table_digest':sequence_digest,'input_record_count':len(rows),
           'sequence_policy':'exact input sequence; ESM alphabet tokens; no trimming or truncation',
           'long_sequence_policy':'full chain in one forward pass, long chains batched individually; ESM-2 rotary positions',
           'inference_precision':'float32, TF32 disabled, no autocast','device':args.device,
           'reused_verified_cache_files':reused,'resumed_verified_files':already,'newly_extracted_sequences':computed,
           'rejected_reuse_records':rejected_reuse,'records':[records[r['sequence_id']] for r in rows if r['sequence_id'] in records]}
    def update(status='running'):
        atomic_json(out/'manifest.json',metadata(status));atomic_json(out/'progress.json',{
            'status':status,'model':args.model,'completed':len(records),'total':len(rows),'reused':reused,'new':computed,
            'pid':os.getpid(),'last_update_unix':time.time()})
    for r in rows:
        sid=r['sequence_id'];length=len(r['sequence']);p=out/(sid+'.pkl');old=existing_records.get(sid)
        if p.exists() and old and old['sequence_sha256']==sha(r['sequence']):
            load_tensor(p,(length,dim));records[sid]=old;already+=1;continue
        cached=reuse.get(sid)
        if cached and cached['sequence_sha256']==sha(r['sequence']) and cached['length']==length:
            try:
                origin=reuse_dir/cached['file'];load_tensor(origin,(length,dim));shutil.copyfile(origin,p)
                assert p.stat().st_nlink==1 and not p.is_symlink()
                records[sid]=record(r,'verified_existing_cache_copy');reused+=1;continue
            except Exception as e:rejected_reuse.append({'sequence_id':sid,'error':str(e)})
        pending.append(r)
    update();print(json.dumps({'model':args.model,'reused':reused,'already':already,'remaining':len(pending),'total':len(rows)}),flush=True)
    if not pending:update('complete');return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    device=torch.device(args.device)
    model,alphabet=getattr(esm.pretrained,model_name)();model.eval().to(device)
    model.requires_grad_(False);convert=alphabet.get_batch_converter()
    allowed=set(alphabet.all_toks)
    for r in pending:
        if any(aa not in allowed for aa in r['sequence']):raise ValueError('unsupported residue token '+r['sequence_id'])
    pending.sort(key=lambda r:(len(r['sequence']),r['sequence_id']))
    def infer(batch):
        nonlocal computed
        try:
            _,_,tokens=convert([(r['sequence_id'],r['sequence']) for r in batch]);tokens=tokens.to(device)
            with torch.inference_mode():result=model(tokens,repr_layers=[layer],return_contacts=False)['representations'][layer]
            outputs=[result[i,1:len(r['sequence'])+1].detach().to('cpu',dtype=torch.float32).contiguous() for i,r in enumerate(batch)]
            del result,tokens
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch)==1:raise
            middle=len(batch)//2;infer(batch[:middle]);infer(batch[middle:]);return
        for r,x in zip(batch,outputs):
            expected=(len(r['sequence']),dim)
            assert tuple(x.shape)==expected and torch.isfinite(x).all()
            p=out/(r['sequence_id']+'.pkl');save_tensor(p,x)
            load_tensor(p,expected);records[r['sequence_id']]=record(r,'new_frozen_full_chain_inference');computed+=1
        update()
        if computed%10<len(batch) or len(records)==len(rows):
            print(json.dumps({'model':args.model,'completed':len(records),'total':len(rows),'new':computed,'last_length':len(batch[-1]['sequence'])}),flush=True)
    batch=[]
    for r in pending:
        if len(r['sequence'])>1022:
            if batch:infer(batch);batch=[]
            infer([r]);continue
        batch.append(r)
        if len(batch)==args.batch_size:infer(batch);batch=[]
    if batch:infer(batch)
    assert len(records)==len(rows)
    assert {p.name for p in out.glob('SEQ-*.pkl')}=={r['sequence_id']+'.pkl' for r in rows}
    for r in rows:
        p=out/(r['sequence_id']+'.pkl');load_tensor(p,(len(r['sequence']),dim));assert p.stat().st_nlink==1
    update('complete');print(json.dumps({'status':'complete','model':args.model,'total':len(rows),'reused':reused,'new':computed,'output':str(out)}),flush=True)

if __name__=='__main__':main()
