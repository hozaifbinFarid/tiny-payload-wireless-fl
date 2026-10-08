from pathlib import Path
import json, hashlib, csv, math
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits
from PIL import Image
import fitz

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audit'
OUT.mkdir(exist_ok=True)
threadpool_limits(2)
ERRORS = []
COUNTS = Counter()
DETAILS = []

def check(condition, message):
    if not condition: ERRORS.append(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)

def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def metrics(w, data, architecture):
    d, c = data['X'].shape[1], len(np.unique(data['y']))
    widths = [d,c] if architecture=='logistic' else [d,128,64,c]
    ids=data['test']; a=data['X'][ids]; pos=0
    for i,(di,do) in enumerate(zip(widths[:-1],widths[1:])):
        mat=w[pos:pos+di*do].reshape(di,do); pos+=di*do
        bias=w[pos:pos+do];pos+=do
        a=a@mat+bias
        if i<len(widths)-2:a=np.maximum(a,0)
    check(pos==len(w),'model dimension')
    y=data['y'][ids]; pred=a.argmax(1); correct=pred==y
    cm=np.zeros((c,c),dtype=np.int64);np.add.at(cm,(y,pred),1)
    z=a-a.max(axis=1,keepdims=True)
    logp=z-np.log(np.exp(z).sum(axis=1,keepdims=True))
    sub={str(int(s)):float(correct[data['subject'][ids]==s].mean()) for s in np.unique(data['subject'][ids])}
    return dict(accuracy=float(correct.mean()),loss=float(-logp[np.arange(len(ids)),y].mean()),
                balanced_accuracy=float((cm.diagonal()/cm.sum(axis=1)).mean()),
                worst_subject_accuracy=min(sub.values()),per_subject=sub,confusion=cm.tolist())

def paired(frame,metric,groupcols):
    rows=[]
    for key, block in frame.groupby(groupcols):
        key=key if isinstance(key,tuple) else (key,)
        p=block.pivot(index='seed',columns='method',values=metric)
        for m in p.columns:
            if m=='fixed_k': continue
            delta=(p[m]-p['fixed_k']).dropna();n=len(delta)
            se=delta.std(ddof=1)/np.sqrt(n);half=stats.t.ppf(.975,n-1)*se
            pv=float(stats.ttest_1samp(delta,0).pvalue) if se>0 else (1.0 if delta.mean()==0 else 0.0)
            rows.append(dict(zip(groupcols,key))|dict(method=m,reference='fixed_k',n_paired_seeds=n,
                mean_difference=delta.mean(),ci95_low=delta.mean()-half,ci95_high=delta.mean()+half,p_value=pv))
    return pd.DataFrame(rows)

def holm(ps):
    ps=np.asarray(ps);idx=np.argsort(ps);out=np.empty(len(ps))
    out[idx]=np.minimum(1,np.maximum.accumulate(ps[idx]*(len(ps)-np.arange(len(ps)))))
    return out

all_test=[];all_history=[];all_diag=[];all_match=[];all_sub=[];cms=[];metarecs=[]
for study in sorted((ROOT/'extracted').glob('*/study_*')):
    short={'TinyPayloadFL2':'HAR_subject','TinyPayloadFL_noniid':'HAR_Dirichlet','TinyPayloadFL_pamap2':'PAMAP2_subject'}[study.parent.name]
    print('Inspecting',short,flush=True)
    manifests={}
    for path in sorted(study.parent.rglob('*')):
        if not path.is_file():continue
        ext=path.suffix;COUNTS[ext]+=1
        if ext=='.json':manifests[str(path)]=json.loads(path.read_text())
        elif ext=='.csv':
            df=pd.read_csv(path);check(not df.columns.duplicated().any(),str(path)+' duplicate columns')
        elif ext=='.npz':
            with np.load(path,allow_pickle=False) as z:
                for k in z.files:
                    a=z[k]
                    if a.dtype.kind in 'fci':check(np.isfinite(a).all(),str(path)+' nonfinite '+k)
                    elif k=='metadata_json':json.loads(str(a.item()))
            side=path.with_suffix('.sha256.json')
            if side.exists():
                COUNTS['checkpoint_sha_checks']+=1
                check(sha(path)==json.loads(side.read_text())['sha256'],str(path)+' sha mismatch')
        elif ext=='.png':
            with Image.open(path) as im:im.load();check(im.width>0,str(path))
        elif ext=='.pdf':
            with fitz.open(path) as pdf:
                for pg in pdf:pg.get_text();pg.get_pixmap(matrix=fitz.Matrix(.2,.2))
        elif ext=='.md':path.read_text()
        else:ERRORS.append('Unhandled file '+str(path))
    sm=json.loads((study/'study_manifest.json').read_text());cfg=sm['config'];dm=sm['data_manifest']
    check(dm==json.loads((study/'data_manifest.json').read_text()),short+' data manifest conflict')
    check(sm['data_id']==digest(dm),short+' data id')
    ignored={'mount_drive','backend','run_experiments','run_diagnostics','final_test','gpu_limit_hours','gpu_reservation_seconds','max_run_wall_seconds','keep_checkpoints','strict_backend_resume','diagnostic_draws','diagnostic_batches'}
    science={k:v for k,v in cfg.items() if k not in ignored}
    check(sm['study_id']==digest({'data':sm['data_id'],'code':sm['code_id'],'config':science}),short+' study identity')
    plan=json.loads((study/'frozen_plan.json').read_text())
    check(json.loads((study/'test_opened.json').read_text())['plan_sha256']==sha(study/'frozen_plan.json'),short+' frozen plan hash')
    dp=next((study.parent/'data').glob('*.npz'))
    with np.load(dp,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    for part in ['train','val','test']:
        key={'val':'validation'}.get(part,part)
        check(hashlib.sha256(data[part].tobytes()).hexdigest()==dm[key+'_index_hash'],short+' '+part+' indices')
    check(not set(data['train'])&set(data['test']) and not set(data['train'])&set(data['val']) and not set(data['val'])&set(data['test']),short+' overlap')
    for p,q in [('train','val'),('train','test'),('val','test')]:
        check(not set(data['subject'][data[p]])&set(data['subject'][data[q]]),short+' subject overlap')
    client_ids=dm['training_subjects'];clients={s:data['train'][data['subject'][data['train']]==s] for s in client_ids}
    if cfg.get('dirichlet_alpha',0)>0:
        rng=np.random.default_rng(cfg['split_seed']+99991);cs={s:[] for s in client_ids}
        for cl in np.unique(data['y'][data['train']]):
            ix=data['train'][data['y'][data['train']]==cl].copy();rng.shuffle(ix)
            props=rng.dirichlet(np.full(len(client_ids),cfg['dirichlet_alpha']))
            cuts=(np.cumsum(props[:-1])*len(ix)).astype(int)
            for s,chunk in zip(client_ids,np.split(ix,cuts)):
                if len(chunk):cs[s].append(chunk)
        clients={s:np.concatenate(v) for s,v in cs.items() if v}
    clstats=[]
    for s,ids in clients.items():
        counts=np.bincount(data['y'][ids],minlength=dm['classes'])
        clstats.append(dict(study=short,client=s,n=len(ids),class_counts=counts.tolist(),classes_present=int((counts>0).sum())))
    (OUT/f'{short}_clients.json').write_text(json.dumps(clstats,indent=2))
    hist=[];tests=[];subs=[];runspec=[];max_loss_error=0
    for rp in sorted((study/'runs').iterdir()):
        rs=json.loads((rp/'run_spec.json').read_text());sp=rs['spec'];runspec.append(canonical(sp))
        check(rs['data']==dm and rs['code_id']==sm['code_id'] and rs['config']==cfg,str(rp)+' run spec')
        check(rs['identity']==digest({'spec':sp,'data':dm,'cfg':science,'code':sm['code_id']}),str(rp)+' run identity')
        latest=json.loads((rp/'latest.json').read_text());cp=rp/latest['file']
        check(sha(cp)==latest['sha256'] and latest['round']==600,str(rp)+' latest')
        with np.load(cp,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata_json'].item()));w=z['w'].copy()
        check(meta['status']=='completed' and meta['round']==600,str(rp)+' not completed')
        check(meta['spec']==sp and meta['identity']==rs['identity'],str(rp)+' checkpoint spec')
        for old in sorted(rp.glob('state_*.npz')):
            with np.load(old,allow_pickle=False) as z:om=json.loads(str(z['metadata_json'].item()))
            check(om['history']==meta['history'][:len(om['history'])],str(old)+' history prefix')
        tm=json.loads((rp/'test_metrics.json').read_text())
        check(tm['weights_sha256']==hashlib.sha256(w.tobytes()).hexdigest() and tm['data_id']==sm['data_id'],str(rp)+' final test association')
        calc=metrics(w,data,sp['model']);recorded=tm['metrics']
        check(calc['confusion']==recorded['confusion'],str(rp)+' confusion matrix mismatch')
        check(calc['per_subject']==recorded['per_subject'],str(rp)+' subject accuracy mismatch')
        for k in ['accuracy','balanced_accuracy','worst_subject_accuracy']:check(abs(calc[k]-recorded[k])<1e-12,str(rp)+' '+k)
        max_loss_error=max(max_loss_error,abs(calc['loss']-recorded['loss']))
        check(abs(calc['loss']-recorded['loss'])<2e-5,str(rp)+' loss')
        tests.append(sp|dict(test_accuracy=recorded['accuracy'],test_loss=recorded['loss'],test_balanced_accuracy=recorded['balanced_accuracy'],
                            test_worst_subject_accuracy=recorded['worst_subject_accuracy'],last_round=meta['round'],validation_best_round=meta['best_round'],
                            deadline_seconds=meta['deadline_seconds'])|meta['meters'])
        hist += [sp|h|{'run_status':meta['status']} for h in meta['history']]
        subs += [sp|{'subject':int(s),'test_accuracy':v} for s,v in recorded['per_subject'].items()]
        cms.append({'study':short,**sp,'confusion':calc['confusion']})
        metarecs.append(dict(study=short,run=rp.name,**sp,parameter_count=len(w),backend=meta['backend_id']))
        if (rp/'probe_calibration.json').exists():
            cal=json.loads((rp/'probe_calibration.json').read_text())
            check(len(cal['records'])==len(client_ids)*cfg['probe_rounds'],str(rp)+' probes')
            check(cal['identity']==rs['identity'],str(rp)+' probe identity')
    check(sorted(runspec)==sorted(map(canonical,plan['runs'])),short+' plan completeness')
    test=pd.DataFrame(tests);history=pd.DataFrame(hist);subjects=pd.DataFrame(subs)
    keys=['family','model','channel','k','method','seed']
    for df,file,extra in [(test,'final_test_metrics.csv',[]),(history,'validation_history.csv',['round']),(subjects,'per_subject_test_metrics.csv',['subject'])]:
        old=pd.read_csv(study/'tables'/file)
        a=df.sort_values(keys+extra).reset_index(drop=True);b=old.sort_values(keys+extra).reset_index(drop=True)
        for k in old.columns:
            if pd.api.types.is_numeric_dtype(b[k]):check(np.allclose(a[k],b[k],equal_nan=True,rtol=1e-10,atol=1e-10),short+' CSV '+file+' '+k)
            else:check(a[k].fillna('').equals(b[k].fillna('')),short+' CSV '+file+' '+k)
    diag=pd.concat([pd.DataFrame(json.loads(p.read_text())) for p in sorted((study/'diagnostics').glob('*.json'))],ignore_index=True)
    archived_diag=pd.read_csv(study/'tables/bias_diagnostics.csv')
    a=diag.sort_values(['k','target','method']).reset_index(drop=True);b=archived_diag.sort_values(['k','target','method']).reset_index(drop=True)
    for c in a:
        if pd.api.types.is_numeric_dtype(b[c]):check(np.allclose(a[c],b[c],equal_nan=True,atol=1e-12),short+' diagnostic '+c)
        else:check(a[c].equals(b[c]),short+' diagnostic '+c)
    for cost in ['uplink_bytes','total_wire_bytes','sim_seconds']:
        recomputed=[]
        for key,bl in history[history.family=='main'].groupby(['model','channel','k','seed']):
            upper=bl.groupby('method')[cost].max().min()
            for method,h in bl.groupby('method'):
                hh=h[h[cost]<=upper].sort_values('round').iloc[-1]
                recomputed.append(dict(zip(['model','channel','k','seed'],key))|dict(method=method,cost_metric=cost,common_budget=upper,observed_round=hh['round'],validation_accuracy=hh['val_accuracy'],validation_worst_subject=hh['val_worst_subject']))
        mm=pd.DataFrame(recomputed);old=pd.read_csv(study/'tables'/f'matched_{cost}.csv')
        a=mm.sort_values(['k','seed','method']).reset_index(drop=True);b=old.sort_values(['k','seed','method']).reset_index(drop=True)
        for c in ['common_budget','observed_round','validation_accuracy','validation_worst_subject']:check(np.allclose(a[c],b[c],atol=1e-10),short+' matched '+cost+' '+c)
        all_match.append(mm.assign(study=short))
    pdiff=paired(test,'test_accuracy',['family','model','channel','k'])
    old=pd.read_csv(study/'tables/final_test_paired_differences.csv')
    for c in ['mean_difference','ci95_low','ci95_high']:
        a=pdiff.sort_values(['family','k','method'])[c];b=old.sort_values(['family','k','method'])[c]
        check(np.allclose(a,b,atol=1e-12),short+' paired '+c)
    all_test.append(test.assign(study=short));all_history.append(history.assign(study=short));all_diag.append(diag.assign(study=short));all_sub.append(subjects.assign(study=short))
    DETAILS.append(dict(study=short,study_id=sm['study_id'],code_id=sm['code_id'],versions=sm['versions'],data_manifest=dm,
                        config=cfg,plan_created=plan['created'],test_opened=json.loads((study/'test_opened.json').read_text()),
                        completion=json.loads((study/'completion.json').read_text()),gpu_ledger=json.loads((study.parent/'gpu_budget.json').read_text()),
                        n_files=sum(1 for p in study.parent.rglob('*') if p.is_file()),max_recomputed_test_loss_error=max_loss_error,
                        client_sizes=[x['n'] for x in clstats],test_subject_counts={str(int(s)):int((data['subject'][data['test']]==s).sum()) for s in np.unique(data['subject'][data['test']])},
                        class_test_counts=np.bincount(data['y'][data['test']],minlength=dm['classes']).tolist()))
    print(short,'264 final models independently evaluated; max loss error',max_loss_error,'errors so far',len(ERRORS),flush=True)

test=pd.concat(all_test,ignore_index=True);history=pd.concat(all_history,ignore_index=True);diag=pd.concat(all_diag,ignore_index=True);matched=pd.concat(all_match,ignore_index=True)
for df,name in [(test,'verified_final_test'),(history,'verified_history'),(diag,'verified_diagnostics'),(matched,'verified_matched_cost'),(pd.concat(all_sub),'verified_per_subject'),(pd.DataFrame(metarecs),'run_inventory')]:df.to_csv(OUT/f'{name}.csv',index=False)
pairedtest=paired(test,'test_accuracy',['study','family','model','channel','k'])
pairedtest['holm_p_main45']=np.nan
mask=pairedtest.family=='main';pairedtest.loc[mask,'holm_p_main45']=holm(pairedtest.loc[mask,'p_value'])
pairedtest['holm_p_robustness6']=np.nan;pairedtest.loc[~mask,'holm_p_robustness6']=holm(pairedtest.loc[~mask,'p_value'])
pairedtest.to_csv(OUT/'test_paired_recomputed.csv',index=False)
matchedpairs=paired(matched,'validation_accuracy',['study','model','channel','k','cost_metric'])
matchedpairs['holm_p_all135']=holm(matchedpairs.p_value)
matchedpairs.to_csv(OUT/'matched_paired_recomputed.csv',index=False)
(OUT/'verified_confusion_matrices.json').write_text(json.dumps(cms,indent=2))
(OUT/'study_details.json').write_text(json.dumps(DETAILS,indent=2))
(OUT/'audit_status.json').write_text(json.dumps({'files':dict(COUNTS),'errors':ERRORS,'final_models_recomputed':len(test)},indent=2))
print(json.dumps({'files':dict(COUNTS),'errors':ERRORS,'final_models_recomputed':len(test)},indent=2),flush=True)
