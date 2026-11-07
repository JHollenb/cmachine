"""Stage a small, frozen math panel and submit exactly one admitted CPU worker."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
ROOT=Path(__file__).resolve().parents[1]
DOMAINS=ROOT.parent

def submit(mode='math'):
    from mrun.client.submit import launch
    from math_worker import PINS
    with tempfile.TemporaryDirectory(prefix='tcm-math-stage-') as tmp:
        stage=Path(tmp);shutil.copytree(ROOT/'src',stage/'src');shutil.copy2(ROOT/'Makefile',stage/'Makefile')
        (stage/'tools').mkdir();(stage/'inputs').mkdir();(stage/'sources').mkdir()
        for name in ('bundle.py','program.py','inspect_program.py','intake_worker.py','program_chat.py','mamba_chat_source.py','math_program.py','math_worker.py'):
            shutil.copy2(ROOT/'tools'/name,stage/'tools'/name)
        if mode in ('math-causal','math-capability'):
            for name in ('math_debugger.py','math_causal_worker.py'):
                shutil.copy2(ROOT/'tools'/name,stage/'tools'/name)
            public=stage/'public/saturn_pub';public.mkdir(parents=True)
            for name in ('__init__.py','core.py','contracts.py','values.py'):
                shutil.copy2(DOMAINS/'saturn-pub/src/saturn_pub'/name,public/name)
        if mode=='math-capability':
            for name in ('math_capability.py','math_capability_worker.py'):
                shutil.copy2(ROOT/'tools'/name,stage/'tools'/name)
            shutil.copy2(DOMAINS/'manalysis/src/manalysis/bench/conditional_family.py',stage/'sources/conditional_family.py')
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
        config={'schema':'tcm-math-intake-v1','mode':mode,'sources':sources,'device':'cpu','backend':'c-fp32-cpu','dtype':'fp32',
            'cpu_threads':1,'max_retries':0,'model_artifacts':PINS,
            'python_hash_seed':0,
            'geometry':'5 frozen cells x272 source rows+swap+3 own recurrence steps; two frozen scorer versions;2 learned concepts+13 supplied laws;3 goals x6 scheduler controls x6rounds;48applications/arm;768nodes;1024matches/round',
            'work_contract':'FieldLM native projected read audit; exact immutable-query C memoization; batch each scoring frontier; no training, model acquisition download, nested jobs, or automatic retries'}
        if mode=='math-causal':
            config.update(evidence_budget_bytes=256*1024**2,
                geometry='3 goals;two frozen baselines each;all reached V2a choice cuts;V1 whole-batch swaps on recipient parent;scalar sham/untouched/A-B-A/permutation;<=64 donor-recipient single exchanges;alternative batch+8singleton controls;1/2/4/8 doses;reciprocal V1 suffix',
                work_contract='Original search prefix recomputation, same caps, Saturn StateCut/choice Acts;exact executable+bundle+input C memo;all replay/native audit work charged;no training/retries/nested jobs')
        if mode=='math-capability':
            config.update(evidence_budget_bytes=256*1024**2,
                geometry='12discoverygoals,32direct C neuronablations,discovery-only rankings;24testgoals x23policies;6fresh follow-up goals x7policies+2post-selection leads;6native tie-order controls;602arms;6rounds,8choices,48total,768nodes,1024matches;16numericalrows/program',
                work_contract='Prepare frozen scorers once;original host DictIndex search;C scores and numerical consumers;Saturn first-choice reconstruction;canonical manalysis coalition catalog;no training/retries/nested jobs')
        (stage/'source-manifest.json').write_text(json.dumps(config,indent=2)+'\n')
        worker={'math':'math_worker.py','math-causal':'math_causal_worker.py','math-capability':'math_capability_worker.py'}[mode]
        job=launch('env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --offline --project {env} python tools/'+worker+' '+mode,experiment='cmachine-model-intake-'+mode,payload=stage,
            config=config,needs={},model=None,prefer_host='beast',pin='beast',env_alias='mrun',timeout_s=600,detach=True,
            preflight=True,guarded=True,retry_on_kill=False,
            note='Frozen capability follow-up to job-2f36448406a4:repair constant-left matmul coverage;12discoverygoals+32Cablations;seal unchanged masks,23policies on24known-family assay rows,6fresh follow-up rows x7policies+2selected leads+6native tie-order controls;602arms;original DictIndex search,Cscoring+16row consumers,Saturn cuts;1CPUthread,6rounds/8choices/48total/768nodes;600s lease;no training/CUDA/retries;stop after panel' if mode=='math-capability' else 'Frozen same-parent causal audit:3goals,2baselines,6round scans,<=64singleton exchanges,alternativebatch+8singletons,sham/A-B-A/permutation/dose/reciprocal controls;original consumer prefix recomputation+native audit+stateless C memo;all replay/native audit work charged;1CPU thread,48selections/6rounds/768nodes,600s lease;no training/CUDA/retries;stop after panel' if mode=='math-causal' else 'Frozen math integration:5 tiny cells,21/23-feature scorers,live FieldLM reads+18 bounded search arms;one CPU thread,48applications/6rounds/768nodes each;no training/CUDA/retries;stop after panel')
        target=ROOT/'results'/job;target.mkdir(parents=True);shutil.copytree(stage,target/'payload')
        (target/'submission.json').write_text(json.dumps({'job_id':job,'config':config},indent=2)+'\n')
    print(json.dumps({'job_id':job,'result_directory':str(target)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('submit','collect'));parser.add_argument('job',nargs='?');parser.add_argument('--mode',choices=('math','math-causal','math-capability'),default='math');args=parser.parse_args()
    if args.action=='submit':submit(args.mode)
    else:
        import contextlib
        import io
        from intake_fleet import collect
        with contextlib.redirect_stdout(io.StringIO()):collect(args.job)
        target=ROOT/'results'/args.job;job=json.loads((target/'job.json').read_text())
        report=json.loads((target/'report.json').read_text()) if (target/'report.json').exists() else {}
        result=report.get('result',{})
        keys=('goals','variants','completed','errors','physical_C_calls','logical_C_calls','scorer_max_abs','clock_confounded') if report.get('mode')=='math-capability' else ('branches','replays','physical_C_calls','logical_C_calls','field_all_equal','scorer_max_abs','potentially_clock_confounded_attempts') if report.get('mode')=='math-causal' else ('search_completed','search_errors','lookup_calls','lookup_all_equal','scorer_batches','scorer_max_abs','C_calls')
        print(json.dumps({'job_id':args.job,'state':job['state'],'status':report.get('status'),'error':report.get('error'),
            'result_directory':str(target),'summary':{key:result.get(key) for key in keys}}))
