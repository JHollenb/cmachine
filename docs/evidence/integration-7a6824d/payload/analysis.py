"""Exact engineering parity and separately preserved semantic observations."""
from pathlib import Path
import ast,json,re
from support import sha
HERE=Path(__file__).resolve().parent
def analyze(raw,driver,manifest,published=True):
    raw=Path(raw);origin=json.loads((HERE/'origin.json').read_text());request=json.loads((raw/'request.json').read_text());observations=[]
    if request['git_commit']!=manifest['git_commit'] or request['source_manifest_sha256']!=sha(HERE/'source-manifest.json') or driver['git_commit']!=manifest['git_commit'] or request['binary_sha256']!=driver['binary_sha256']:raise ValueError('Commit/source request join differs')
    actual=[x['command'] for x in driver['protocol']]+[x['command'] for x in driver['mechanics']]+[x['command'] for x in driver['clients']]
    if actual!=request['execution_commands'] or len(actual)!=12:raise ValueError('Executed panel membership differs')
    by_name={x['name']:x for x in driver['protocol']}
    if set(by_name)!=set(('reference-native','native','full-byte','bounded','bounded-fp32')):raise ValueError('Protocol policy membership differs')
    reference=by_name['reference-native']['result'];native=by_name['native']['result']
    if reference!=native or len(native['turns'])!=10:raise ValueError('Original versus integrated native execution differs')
    observations.append({'kind':'native-commit-parity','metrics':{'own_turns':10,'prefill_and_final_logits':20,'valid_cache_and_controls_exact':True,'source_commit':manifest['git_commit'],'reference_commit':manifest['native_reference_commit']}})
    protocols=[]
    for name in ('full-byte','bounded','bounded-fp32'):
        entry=by_name[name];p=entry['result'];policy=entry['policy'];prior=origin['historical'][policy];ready=p['ready'];capacity=768 if policy=='full-byte' else 112;state=2535428+60216*capacity;profile=1 if name=='bounded-fp32' else 2;model_bytes=6211573780 if profile==1 else 3124434964
        if ready['cache_policy']!=policy or ready['weight_profile']!=profile or ready['resident_capacity']!=capacity or ready['capacity']!=768 or ready['gpu_session_bytes']!=state or ready['tracked_gpu_bytes']!=model_bytes+state:raise ValueError('Activated memory/weight profile differs')
        if len(p['turns'])!=10 or len(p['refusals'])!=11 or not p['exact_sibling_replay'] or any(not x['state_unchanged'] for x in p['refusals']):raise ValueError('Protocol lifecycle coverage differs')
        if len(p['generation_checks'])!=1 or not p['generation_checks'][0]['state_equal_except_marker_id'] or not p['generation_checks'][0]['repeated_eos_empty_and_nonmutating']:raise ValueError('Pending-token ownership differs')
        for t in p['turns'][:8]:
            old=next(x for x in prior['turns'] if x['input_index']==t['input_index'])
            if t['generated']['ids']!=old['answer_ids'] or t['prefill']['logits_sha256']!=old['prefill']['sha256'] or t['final']['logits_sha256']!=old['final']['sha256'] or t['final']['position']!=old['final_position']:raise ValueError('Historical own-token/readout parity differs')
        if p['delayed_root']['cache_sha256']!=prior['root_sha256'] or p['delayed_root']['position']!=prior['root_position']:raise ValueError('Historical compressed state differs')
        if p['reset']['position'] or p['reset']['resident_rows'] or p['reset']['stack_depth'] or p['reset']['evictions']:raise ValueError('Reset retained reachable state')
        trace=json.loads((raw/(name+'-protocol.json')).read_text())
        if trace[0]['reply']!=ready or sum('error' in x['reply'] for x in trace)!=11:raise ValueError('Raw protocol join differs')
        row={'name':name,'historical_turns':8,'historical_full_logits':16,'state_gpu_bytes':state,'model_and_state_gpu_bytes':model_bytes+state,'nonmutating_refusals':11,'exact_replay':True,'split_generation_and_eos':True};protocols.append(row);observations.append({'kind':'integrated-policy-parity','metrics':row})
    if len(driver['mechanics'])!=2:raise ValueError('Real mechanics membership differs')
    m={x['name']:x['result'] for x in driver['mechanics']}
    if not m['native']['writer_parent_lineage_checks'] or not m['native']['valid_kv_and_full_logits_exact'] or m['native']['stack_limit']!=64 or not all(m['bounded'][k] for k in ('occupied_sample_abort','cache_logits_rng_lineage_restored','exact_child_replay','same_layer_retry_exact')) or m['bounded']['state_gpu_bytes']!=9279620:raise ValueError('Real-model merged lifecycle differs')
    clients=[]
    for x in driver['clients']:
        name=x['name'];text=(raw/(name+'.stdout.txt')).read_text();answers=re.findall(r'model> ([\s\S]*?)(?=you> |\Z)',text);resets=text.count('[recomputed prompt after token-prefix change; saved parent and nested frames invalidated]');ready=json.loads(text.splitlines()[0])
        if x['returncode'] or sha(raw/(name+'.stdin.txt'))!=x['stdin_sha256'] or sha(raw/(name+'.stdout.txt'))!=x['stdout_sha256'] or answers!=origin['client_answers'][name] or resets:raise ValueError('Historical text-client behavior or continuous history differs')
        if ready.get('cache_policy','native')!=x['policy']:raise ValueError('Text client policy differs')
        states=[]
        for line in text.splitlines():
            i=line.find("{'event': 'state'")
            if i>=0:states.append(ast.literal_eval(line[i:]))
        if x['policy']!='native' and (not states or any(s['gpu_state_bytes']!=ready['gpu_session_bytes'] for s in states)):raise ValueError('Text client state allocation grew')
        row={'name':name,'answers':answers,'prompt_recomputations':resets,'state_gpu_bytes':ready['gpu_session_bytes'],'maximum_position':max((s['position'] for s in states),default=0)};clients.append(row);observations.append({'kind':'chat-behavior-observation','metrics':row})
    install=driver['installation'];installed=json.loads((raw/'installed-files.json').read_text());expected={'source/'+n:r['sha256'] for n,r in manifest['sources'].items()};expected.update({'source/source-manifest.json':sha(HERE/'source-manifest.json'),'build/tcm-cuda':driver['binary_sha256'],'build/tcm-bounded-chat':driver['binary_sha256'],'tools/chat.py':sha(HERE/'tools/chat.py')})
    if installed['files']!=expected or installed['git_commit']!=manifest['git_commit'] or install['git_commit']!=manifest['git_commit'] or sha(raw/'installed-files.json')!=install['manifest_sha256'] or install['published']!=published or install['service_started']:raise ValueError('Commit-bound installation differs')
    return {'summary':{'source_commit':manifest['git_commit'],'native_reference_commit':manifest['native_reference_commit'],'native_reference_exact':True,'protocol':protocols,'real_mechanics':m,'clients':clients,'installation':install,'scope':'integrated execution parity; historical bounded recall failures preserved; no new semantic or speed claim'},'observations':observations}
