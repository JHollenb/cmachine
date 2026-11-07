"""Additive recovery of a completed C assay whose final summary used a wrong key."""
import argparse,base64,gzip,hashlib,io,json,sys,tarfile,types
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root):
    root=Path(root);report=json.loads((root/'report.json').read_text());job=json.loads((root/'job.json').read_text());submission=json.loads((root/'submission.json').read_text())
    if job['job_id']!=submission['job_id'] or sha(root/'logs.txt')!=job['result']['artifacts']['job_log']['sha256']:raise ValueError('Scheduler log seal differs')
    error=report.get('error','')
    if "KeyError: 'advances'" not in error or 'summary=' not in error or 'driver' not in report or job['state']!='failed':raise ValueError('Recovery applies only to the declared completed-C summary-key failure')
    for name,digest in submission['config']['sources'].items():
        if sha(root/'payload'/name)!=digest:raise ValueError('Executed source differs')
    if sha(root/'payload/source-manifest.json')!=submission['config']['source_manifest_sha256']:raise ValueError('Source manifest differs')
    archive_rows={};logged=[];archives=0
    # The authenticated job-log archive covers native reference arrays too;
    # the failed worker never reached its ordinary raw_files manifest.
    for line in (root/'logs.txt').read_text().splitlines():
        if line.startswith('COMPOSITION_RESULT_GZIP_B64='):logged.append(json.loads(gzip.decompress(base64.b64decode(line.split('=',1)[1]))))
        if not line.startswith('COMPOSITION_FILES_TGZ_B64='):continue
        archives+=1
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(line.split('=',1)[1])) ,mode='r:gz') as archive:
            for member in archive.getmembers():
                name=Path(member.name)
                if not member.isfile() or name.is_absolute() or '..' in name.parts or name.parts[0] not in ('raw','payload'):raise ValueError('Invalid archive member')
                data=archive.extractfile(member).read();digest=hashlib.sha256(data).hexdigest()
                if sha(root/name)!=digest or (root/name).stat().st_size!=len(data):raise ValueError('Collected file differs from sealed log archive')
                archive_rows[str(name)]={'sha256':digest,'bytes':len(data)}
    if logged!=[report] or archives!=1 or json.loads((root/'raw/report.json').read_text())!=report or json.loads((root/'raw/driver.json').read_text())!=report['driver']:raise ValueError('Failed report/archive/C driver join differs')
    wrapper=report['saturn_receipt'];receipt=dict(wrapper['receipt']);fingerprint=receipt.pop('fingerprint')
    if hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=fingerprint or receipt['parent']!=wrapper['before'] or receipt['result']!=wrapper['after']:raise ValueError('Whole C receipt differs')
    original=(root/'payload/analysis.py').read_text();old="sum(p['advances'] for p in driver['paths'])";new="sum(p['count'] for p in driver['paths'])"
    if original.count(old)!=2:raise ValueError('Correction recipe differs')
    corrected=original.replace(old,new);directory=root/'recovery';directory.mkdir(exist_ok=True);fixed=directory/'analysis-corrected.py'
    if fixed.exists() and fixed.read_text()!=corrected:raise ValueError('Recovery source drift')
    fixed.write_text(corrected)
    # Bind helpers to the executed sources; only two summary field accesses change.
    sys.path.insert(0,str((root/'payload').resolve()))
    module=types.ModuleType('recovered_query_analysis');module.__file__=str(fixed);exec(compile(corrected,str(fixed),'exec'),module.__dict__)
    analysis=module.analyze(root/'raw',root/'payload/bank',report['driver'],report['texts'])
    result={'schema':'tcm-query-consumer-analysis-recovery-v1','original_report_sha256':sha(root/'report.json'),'scheduler_status':job['state'],'original_error':error,'correction':'Exactly two final-summary generator accesses change driver path advances to count; computation/selection/evaluation code unchanged.','executed_analyzer_sha256':sha(root/'payload/analysis.py'),'corrected_analyzer_sha256':sha(fixed),'models_evaluated':0,'c_process_completed_before_postprocessing_failure':True,'analysis':analysis,'archive_files':archive_rows,'scope':'Additive saved-array reconstruction. Original failed report/raw/payload/logs remain unchanged; no model rerun or threshold change.','terminal_status':'not-requested'}
    target=root/'analysis-recovery.json';data=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if target.exists() and json.loads(target.read_text())!=result:raise ValueError('Recovery result drift')
    target.write_text(data)
    verification={'schema':'tcm-query-consumer-qualified-verification-v1','report_sha256':sha(root/'report.json'),'scheduler_status':'failed','c_process_complete':True,'scheduler_log_seal':True,'source_files_verified':len(submission['config']['sources']),'raw_files_verified':sum(name.startswith('raw/') for name in archive_rows),'archive_files_verified':len(archive_rows),'whole_invocation_receipt':True,'analysis_recomputed':True,'analysis_recovery_sha256':sha(target),'correction_limited_to_two_summary_keys':True,'models_evaluated':0,'terminal_status':'not-requested'}
    (root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n');(root/'observations.jsonl').write_text(''.join(json.dumps({**row,'schema':'tcm-recovered-query-observation-v1','model_id':report['model']['identity'],'report_sha256':sha(root/'report.json'),'analysis_recovery_sha256':sha(target)},sort_keys=True,allow_nan=False)+'\n' for row in analysis['observations']))
    print(json.dumps(verification));return verification
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');verify(p.parse_args().root)
