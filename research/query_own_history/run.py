"""Authenticated small payload, measured C reservation and exact-job collection."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'query_horizon/runs/job-309bf320370d'
EXPECTED = '3870a2522c9986179ba1ac2e71736091c233889cf4804cd5f99a7120fb7d0db8'
LIMIT = 256*1024**2
spec = importlib.util.spec_from_file_location('composition_collection', HERE.parent / 'composition/run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def stage():
    base.checked(PREVIOUS / 'report.json', EXPECTED)
    report = json.loads((PREVIOUS / 'report.json').read_text())
    if json.loads((PREVIOUS / 'verification.json').read_text())['report_sha256'] != EXPECTED:
        raise ValueError('Prior verification differs')
    out = HERE / 'staged' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') / 'payload'
    out.mkdir(parents=True)
    frozen = json.loads((PREVIOUS / 'payload/source-manifest.json').read_text())
    for name, digest in frozen['sources'].items():
        if name.startswith(('runtime/', 'public/')) or name == 'saturn_bridge.py':
            source = PREVIOUS / 'payload' / name
            base.checked(source, digest)
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ('driver.c', 'memory.h', 'distribution.h', 'Makefile', 'reader.py', 'metrics.py', 'analysis.py', 'worker.py', 'run.py', 'PLAN.md', 'MATH.md', 'test_metrics.py', 'test_memory.c', 'test_distribution.c'):
        shutil.copy2(HERE / name, out / name)
    def raw(name, target=None):
        description = report['raw_files'][name]
        base.checked(PREVIOUS / 'raw' / name, description['sha256'])
        shutil.copy2(PREVIOUS / 'raw' / name, out / (target or name))
        return description
    programs = {}
    for kind in ('basis', 'q', 'k', 'v'):
        name = f'program128-{kind}.f32'
        programs[name] = raw(name)
    programs['query-fit.f32'] = raw('selected-query.f32', 'query-fit.f32')
    if programs['query-fit.f32']['sha256'] != programs['program128-q.f32']['sha256'] or report['source_selection']['selected_index'] != 5:
        raise ValueError('Frozen dose2 binding differs')
    raw('source-traces.json', 'historical-traces.json')
    raw('source-metrics-5.json', 'historical-metrics.json')
    inputs = json.loads((PREVIOUS / 'raw/source-input-manifest.json').read_text())
    base.checked(PREVIOUS / 'raw/source-input-manifest.json', report['raw_files']['source-input-manifest.json']['sha256'])
    for row in inputs['sources']:
        row['native_eos_token_ids'] = report['model']['native_eos_token_ids']
    (out / 'input-manifest.json').write_text(json.dumps(inputs, indent=2, sort_keys=True)+'\n')
    historical = {}
    for c in range(3):
        historical[str(c)] = {}
        for role, label, fields in [('compact', 'proposal5', ('q.f32', 'k.f32', 'v.f32', 'attention.f32', 'tops.i32')), ('native', 'native', ('norm.f32', 'q.f32', 'k.f32', 'v.f32', 'attention.f32', 'tops.i32'))]:
            for field in fields:
                name = f'source{c}-{label}-{field}'
                description = report['raw_files'][name]
                base.checked(PREVIOUS / 'raw' / name, description['sha256'])
                historical[str(c)][role+':'+field] = description
    lineage = {'schema': 'tcm-own-history-lineage-v1', 'previous_job': 'job-309bf320370d', 'previous_report_sha256': EXPECTED, 'query_image_index': 5, 'query_dose': 2, 'query_image_sha256': programs['query-fit.f32']['sha256'], 'program_files': programs, 'historical_arrays': historical, 'new_fit_forwards': 0, 'scope': 'Same image, sources, horizon, model and memory; only history-generation condition changes.'}
    (out / 'lineage.json').write_text(json.dumps(lineage, indent=2, sort_keys=True)+'\n')
    (out / 'baseline.json').write_text(json.dumps({'model': report['model'], 'previous_report_sha256': EXPECTED}, indent=2, sort_keys=True)+'\n')
    files = {str(p.relative_to(out)): base.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'source-manifest.json').write_text(json.dumps({'schema': 'tcm-own-history-payload-v1', 'sources': files, 'raw_archive_limit_bytes': LIMIT}, indent=2)+'\n')
    print(json.dumps({'payload': str(out), 'sources': len(files), 'payload_bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}))


def submit(payload):
    from mrun.client.submit import launch
    payload = Path(payload).resolve()
    manifest = json.loads((payload / 'source-manifest.json').read_text())
    for name, digest in manifest['sources'].items():
        base.checked(payload / name, digest)
    config = {'schema': 'tcm-own-history-execution-v1', 'backend': 'hf', 'candidate_backend': 'c-cublas-fp32+cpu-fp64-centroid', 'device': 'cuda', 'dtype': 'fp32', 'cpu_threads': 1, 'max_seq_len': 512, 'backend_options': {'physical_batch_size': 1}, 'max_retries': 0, 'sources': manifest['sources'], 'source_manifest_sha256': base.sha(payload / 'source-manifest.json'), 'raw_archive_limit_bytes': LIMIT, 'reservation_source': 'Measured sameC/model/capacity512 history job-b141eadfe952 RSS4224.3MB/VRAM6380MB; job-309bf320370d same6144MB reservation succeeded. No simultaneous sessions except unchanged16-token sham. Guard6144*1.1=6758.4MB fits Beast; no source-wide host logit cache.', 'physical_work': 'one resident C modelCPU1B1;32sham;3archivednative histories xcompact/native +3compact-generated histories xcompact/native;12shorttrajectories,max4*451+32advances;candidate seals before independently built shadows;full-clockC/Pythonaudits;stopfixedpanel', 'scheduling_contract': {'gpu_model_loads': 1, 'branch_execution': 'in_process', 'physical_batch_size': 1, 'max_retries': 0, 'collection_policy': 'direct_job_only'}}
    job = launch('uv run --offline --project {env} python worker.py', experiment='c-state-machine-query-own-history', payload=payload, model='qwen2.5-1.5b-instruct', prefer_host='beast', pin='beast', needs={'cuda': True}, config=config, env_alias='mrun', ram_mb=6144, vram_mb=6144, cpu_threads=1, est_wall_s=180, timeout_s=840, detach=True, retry_on_kill=False, preflight=True, note='Fixedquery historycut:3same32+followup32sources;nativeIDs andcompactownIDs each evaluatedbycompactandfreshnative,12shortpaths+32sham;oneCmodelCPU1B1;candidate seal before shadow;no fit/newtargets;stopfixedpanel')
    root = HERE / 'runs' / job
    root.mkdir(parents=True, exist_ok=True)
    (root / 'submission.json').write_text(json.dumps({'job_id': job, 'payload': str(payload), 'config': config}, indent=2)+'\n')
    print(json.dumps({'job_id': job, 'result_directory': str(root)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('stage', 'submit', 'collect', 'verify'))
    p.add_argument('target', nargs='?')
    a = p.parse_args()
    if a.action == 'stage':
        stage()
    elif a.action == 'submit':
        submit(a.target)
    elif a.action == 'collect':
        base.LIMIT = LIMIT
        base.collect(a.target, HERE / 'runs')
    else:
        base.verify(a.target)
