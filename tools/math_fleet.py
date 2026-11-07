"""Stage a small, frozen math panel and submit exactly one admitted CPU worker."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
ROOT=Path(__file__).resolve().parents[1]
DOMAINS=ROOT.parent

def submit():
    from mrun.client.submit import launch
    from math_worker import PINS
    with tempfile.TemporaryDirectory(prefix='tcm-math-stage-') as tmp:
        stage=Path(tmp);shutil.copytree(ROOT/'src',stage/'src');shutil.copy2(ROOT/'Makefile',stage/'Makefile')
        (stage/'tools').mkdir();(stage/'inputs').mkdir();(stage/'sources').mkdir()
        for name in ('bundle.py','program.py','inspect_program.py','intake_worker.py','program_chat.py','mamba_chat_source.py','math_program.py','math_worker.py'):
            shutil.copy2(ROOT/'tools'/name,stage/'tools'/name)
        experiments=DOMAINS/'model-experiments/experiments'
        shutil.copy2(experiments/'2026-10-16-learned-matrix-algorithm/law.py',stage/'sources/law.py')
        shutil.copy2(experiments/'2026-10-17-learned-matrix-c-scale/frozen-rules.json',stage/'inputs/exact-rules.json')
        for seed in (137,149,163):shutil.copy2(experiments/f'2026-10-30-architecture-matched-learning/exports/seed{seed}-bilinear7.json',stage/f'inputs/seed{seed}-bilinear7.json')
        for version in ('v1','v2a'):
            target=stage/'sources'/version;target.mkdir()
            for name in ('concept_graph.py','scheduler.py','goals.py','train.py'):
                shutil.copy2(experiments/f'2026-11-02-concept-graph-{version}'/name,target/name)
        shutil.copy2(experiments/'2026-11-02-concept-graph-v1/native_store.py',stage/'sources/v1/native_store.py')
        native=DOMAINS/'spectral-phase-format-research/domains/ml/spectral-native/spectral_native'
        (stage/'spectral_native').mkdir();(stage/'spectral_native/__init__.py').write_text('')
        for name in ('knowledge.py','fieldlm_world.py','fieldlm_world2.py','role_chart_field.py','field_fabric.py'):
            shutil.copy2(native/name,stage/'spectral_native'/name)
        for name,directory in [('field_store.py','2026-10-30-fieldlm-mathematical-concepts'),('native_dimensions.py','2026-10-30-fieldlm-mathematical-metaconcepts')]:
            shutil.copy2(experiments/directory/name,stage/'sources'/name)
        for name in ('concepts.py','algebra.py'):
            shutil.copy2(experiments/'2026-10-30-fieldlm-mathematical-concepts'/name,stage/'sources'/name)
        sources={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file()}
        config={'schema':'tcm-math-intake-v1','mode':'math','sources':sources,'device':'cpu','backend':'c-fp32-cpu','dtype':'fp32',
            'cpu_threads':1,'max_retries':0,'model_artifacts':PINS,
            'python_hash_seed':0,
            'geometry':'5 frozen cells x272 source rows+swap+3 own recurrence steps; two frozen scorer versions;2 learned concepts+13 supplied laws;3 goals x6 scheduler controls x6rounds;48applications/arm;768nodes;1024matches/round',
            'work_contract':'FieldLM native projected read audit; exact immutable-query C memoization; batch each scoring frontier; no training, model acquisition download, nested jobs, or automatic retries'}
        (stage/'source-manifest.json').write_text(json.dumps(config,indent=2)+'\n')
        job=launch('env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --offline --project {env} python tools/math_worker.py math',experiment='cmachine-model-intake-math',payload=stage,
            config=config,needs={},model=None,prefer_host='beast',pin='beast',env_alias='mrun',timeout_s=600,detach=True,
            preflight=True,guarded=True,retry_on_kill=False,
            note='Frozen math integration:5 tiny cells,21/23-feature scorers,live FieldLM reads+18 bounded search arms;one CPU thread,48applications/6rounds/768nodes each;no training/CUDA/retries;stop after panel')
        target=ROOT/'results'/job;target.mkdir(parents=True);shutil.copytree(stage,target/'payload')
        (target/'submission.json').write_text(json.dumps({'job_id':job,'config':config},indent=2)+'\n')
    print(json.dumps({'job_id':job,'result_directory':str(target)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('submit','collect'));parser.add_argument('job',nargs='?');args=parser.parse_args()
    if args.action=='submit':submit()
    else:
        import contextlib
        import io
        from intake_fleet import collect
        with contextlib.redirect_stdout(io.StringIO()):collect(args.job)
        target=ROOT/'results'/args.job;job=json.loads((target/'job.json').read_text())
        report=json.loads((target/'report.json').read_text()) if (target/'report.json').exists() else {}
        result=report.get('result',{})
        print(json.dumps({'job_id':args.job,'state':job['state'],'status':report.get('status'),'error':report.get('error'),
            'result_directory':str(target),'summary':{key:result.get(key) for key in
                ('search_completed','search_errors','lookup_calls','lookup_all_equal','scorer_batches','scorer_max_abs','C_calls')}}))
