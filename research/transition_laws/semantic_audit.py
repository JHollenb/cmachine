"""Re-decode authoritative answer IDs with the pinned fast tokenizer, no model."""
import argparse,hashlib,json,re
from pathlib import Path
from tokenizers import Tokenizer
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(job,source):
    r=json.loads((job/'report.json').read_text());files={}
    for name,wanted in r['model']['tokenizer_files'].items():
        got=sha(source/name)
        if got!=wanted:raise ValueError('Tokenizer source identity: '+name)
        files[name]=got
    tokenizer=Tokenizer.from_file(str(source/'tokenizer.json'));decoded=[];scores={}
    cases={c['id']:c for c in json.loads((job/'raw/heldout-inputs.json').read_text())}
    # Recompute both arithmetic sequences separately from supplied labels.
    if cases['h2']['expected']!=[19*24,19*24+18] or cases['h3']['expected']!=[27*13,27*13-26]:raise ValueError('Arithmetic truth')
    for label,data in r['runs'].items():
        reported={(x['case_id'],x['turn']):x['answer'] for x in data['metrics']['turns']}
        checks=[]
        for path in data['driver']['paths']:
            for i,t in enumerate(path['turns']):
                text=tokenizer.decode(t['answer_ids'],skip_special_tokens=True)
                if text!=reported[(path['case_id'],i)]:raise ValueError('Worker decode mismatch')
                decoded.append({'label':label,'case_id':path['case_id'],'turn':i,'answer_ids':t['answer_ids'],'text':text,'matches_worker':True})
                if not label.startswith('heldout'):continue
                c=cases[path['case_id']]
                if c['format']=='prose':continue
                expected=c['expected'][i];value=text.strip()
                if c['format']=='word':correct=value.casefold().rstrip('.!')==expected.casefold()
                else:correct=bool(re.fullmatch(r'[+-]?\d+',value)) and int(value)==expected
                checks.append({'case_id':path['case_id'],'turn':i,'text':text,'expected':expected,'strict_correct':correct})
        if checks:scores[label]={'correct':sum(c['strict_correct'] for c in checks),'count':len(checks),'checks':checks}
    out={'schema':'transition-laws-semantic-audit-v1','report_sha256':sha(job/'report.json'),'tokenizer_hashes':files,'model_execution':False,'all_decodes_match_worker':True,'decoded_turns':len(decoded),'structured_scores':scores,'decoded':decoded,'scope':'Deterministic tokenizer reconstruction and literal structured-task scoring; no automatic prose quality or general semantic certification.'}
    (job/'analysis/semantic-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'decoded_turns':len(decoded),'scores':{k:{a:v for a,v in x.items() if a!='checks'} for k,x in scores.items()}}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('tokenizer',type=Path);a=p.parse_args();audit(a.job,a.tokenizer)
