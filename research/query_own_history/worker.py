"""One finite resident C model; sealed autonomous candidates and native shadows."""
import base64
import gzip
import hashlib
import io
import json
import os
import platform
import selectors
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'results'
LIMIT = 256*1024**2
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def experiment():
    sys.path.insert(0, str(ROOT / 'public'))
    from saturn_bridge import invoke
    from metrics import audit_pair
    from analysis import analyze
    from transformers import AutoTokenizer
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    for name, digest in manifest['sources'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Payload source differs: '+name)
    model = json.loads((ROOT / 'baseline.json').read_text())['model']
    tokroot = Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    for name, digest in model['tokenizer_files'].items():
        if sha(tokroot / name) != digest:
            raise ValueError('Tokenizer differs')
    tokenizer = AutoTokenizer.from_pretrained(tokroot, local_files_only=True)
    for name in ('lineage.json', 'historical-traces.json', 'historical-metrics.json', 'input-manifest.json'):
        shutil.copy2(ROOT / name, RAW / name)
    for path in sorted(ROOT.glob('*.f32')):
        shutil.copy2(path, RAW / path.name)
    inputs = json.loads((RAW / 'input-manifest.json').read_text())
    old = json.loads((RAW / 'historical-traces.json').read_text())['sources']
    for c, row in enumerate(inputs['sources']):
        prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt_text']}], tokenize=False, add_generation_prompt=True)
        if tokenizer.encode(prefix, add_special_tokens=False) != row['prompt_ids']:
            raise ValueError('Prompt token identity differs')
        extended = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt_text']}, {'role': 'assistant', 'content': ''}, {'role': 'user', 'content': inputs['followup_text']}], tokenize=False, add_generation_prompt=True)
        if not extended.startswith(prefix) or tokenizer.encode(extended[len(prefix):], add_special_tokens=False) != row['followup_ids']:
            raise ValueError('Followup template identity differs')
        for stem, ids in [(f'source{c}', row['prompt_ids']), (f'source{c}-followup', row['followup_ids']), (f'source{c}-native-history', old[c]['consumed_ids'])]:
            (RAW / (stem+'.txt')).write_text(' '.join(map(str, ids))+'\n')
    maxima = [len(s['prompt_ids'])+len(s['followup_ids'])+64 for s in inputs['sources']]
    payload_bytes = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'results' not in p.parts)
    existing = sum(p.stat().st_size for p in RAW.iterdir() if p.is_file())
    bound = 4*sum(maxima)*27140 + 48*151936*4 + 24*98464 + payload_bytes + existing + 32*1024**2
    if max(maxima) >= 512 or bound > LIMIT:
        raise ValueError('Source/capacity/archive bound differs')
    report = {'schema': 'tcm-query-own-history-experiment-v1', 'model': model, 'lineage': json.loads((RAW / 'lineage.json').read_text()), 'source_inputs': inputs, 'evidence_capacity_preflight': {'maxima': maxima, 'raw_archive_bound_bytes': bound, 'limit_bytes': LIMIT, 'physical_advances_bound': 4*sum(maxima)+32, 'scratch_streams_bound_bytes': 2*max(maxima)*151936*4}, 'mechanics_status': 'in-progress', 'instrument_status': 'in-progress', 'trend_status': 'observations', 'terminal_status': 'not-requested', 'environment': {'platform': platform.platform(), 'python': sys.version, 'job_id': os.environ.get('MRUN_JOB_ID')}}
    (RAW / 'partial-report.json').write_text(json.dumps(report)+'\n')
    build = subprocess.run(['make'], cwd=ROOT, capture_output=True, text=True, timeout=120)
    (RAW / 'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:
        raise RuntimeError('C build failed: '+build.stderr[-10000:])
    report['binary_sha256'] = sha(ROOT / 'build/query-program')
    args = [str(ROOT / 'build/query-program'), model['path'], str(RAW)]
    def controller(_):
        audits, lines = [], []
        deadline = time.monotonic()+600
        with (RAW / 'driver.stderr.txt').open('w') as err:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
            poll = selectors.DefaultSelector()
            poll.register(process.stdout, selectors.EVENT_READ)
            try:
                while True:
                    if time.monotonic() > deadline:
                        raise RuntimeError('Finite C process deadline exceeded')
                    if not poll.select(1):
                        if process.poll() is not None:
                            break
                        continue
                    line = process.stdout.readline()
                    if not line:
                        if process.poll() is not None:
                            break
                        continue
                    lines.append(line)
                    if not line.startswith('PAIR_READY ') or len(line) > 256:
                        raise ValueError('Unexpected driver protocol')
                    index = int(line.split()[1])
                    if index != len(audits) or index >= 6:
                        raise ValueError('Pair order differs')
                    audits.append(audit_pair(RAW, index))
                    print(f'pair{index}: full-clock C/Python agreement; futureKL={audits[-1]["future_mean_kl_nats"]:.9g}', flush=True)
                    for name in ('candidate-logits.tmp', 'native-logits.tmp'):
                        (RAW / name).unlink()
                    process.stdin.write('\n')
                    process.stdin.flush()
                code = process.wait(timeout=5)
                if code or len(audits) != 6:
                    raise RuntimeError(f'C process {code}: '+(RAW / 'driver.stderr.txt').read_text()[-10000:])
            finally:
                poll.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                process.stdin.close()
                process.stdout.close()
                (RAW / 'driver.stdout.txt').write_text(''.join(lines))
        return json.loads((RAW / 'driver.json').read_text())
    report['driver'], report['saturn_receipt'] = invoke(args, None, {'model_digest': model['content_sha256'], 'binary_sha256': report['binary_sha256'], 'source_manifest_sha256': sha(ROOT / 'source-manifest.json'), 'input_manifest_sha256': sha(RAW / 'input-manifest.json'), 'lineage_sha256': sha(RAW / 'lineage.json'), 'scope': 'Whole finite C invocation; fixed image/memory, native/compact history generators crossed with independently built native/compact evaluators; candidate seals precede shadows'}, controller=controller)
    if report['driver']['model_digest'] != model['content_sha256'] or report['driver']['model_identity'] != model['identity']:
        raise ValueError('Model identity differs')
    texts = {}
    for pair in report['driver']['pairs']:
        name = pair['evaluators'][0]['name']
        trace = json.loads((RAW / f'{name}-trace.json').read_text())
        texts[name] = [tokenizer.decode(t['answer_ids'], skip_special_tokens=True) for t in trace['turns']]
    report['texts'] = texts
    (RAW / 'partial-report.json').write_text(json.dumps(report)+'\n')
    report['analysis'] = analyze(RAW, ROOT / 'bank', report['driver'], texts)
    report['raw_files'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(RAW.iterdir()) if p.is_file() and p.name != 'partial-report.json'}
    report.update(mechanics_status='complete', instrument_status='complete')
    return report


def emit():
    files = []
    for directory, prefix in ((RAW, 'raw'), (ROOT / 'runtime', 'payload/runtime'), (ROOT / 'public', 'payload/public')):
        files += [(p, prefix+'/'+str(p.relative_to(directory))) for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    files += [(p, 'payload/'+p.name) for p in sorted(ROOT.iterdir()) if p.is_file()]
    if sum(p.stat().st_size for p, _ in files) > LIMIT:
        raise ValueError('Archive cap exceeded')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    print('COMPOSITION_FILES_TGZ_B64='+base64.b64encode(buffer.getvalue()).decode(), flush=True)


if __name__ == '__main__':
    RAW.mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        report = experiment()
        code = 0
    except Exception:
        report = json.loads((RAW / 'partial-report.json').read_text()) if (RAW / 'partial-report.json').exists() else {}
        report.update(error=traceback.format_exc(), mechanics_status='incomplete')
        code = 1
    for name in ('candidate-logits.tmp', 'native-logits.tmp'):
        if (RAW / name).exists():
            (RAW / name).unlink()
    report['worker_seconds'] = time.monotonic()-started
    data = json.dumps(report, sort_keys=True, allow_nan=False).encode()
    (RAW / 'report.json').write_bytes(data+b'\n')
    print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(data, mtime=0)).decode(), flush=True)
    emit()
    raise SystemExit(code)
