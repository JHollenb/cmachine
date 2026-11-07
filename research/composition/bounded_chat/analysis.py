"""Saved protocol/client engineering checks and separately retained text outcomes."""
from pathlib import Path
import json,re,hashlib
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def analyze(raw,bank,driver,texts,installation_stage='published'):
    raw=Path(raw);origin=json.loads((HERE/'origin.json').read_text());observations=[];protocol=[];clients=[];declared=json.loads((raw/'request.json').read_text())['execution_commands']
    if driver['schema']!='tcm-bounded-chat-driver-v1' or len(driver['protocol'])!=2 or len(driver['clients'])!=5:raise ValueError('Persistent panel membership differs')
    actual=[p['command'] for p in driver['protocol']]+[p['command'] for p in driver['clients']]
    if actual!=declared:raise ValueError('Executed command list differs')
    for entry in driver['protocol']:
        policy=entry['policy'];p=entry['result'];prior=origin['historical'][policy];ready=p['ready'];c=112 if policy=='bounded' else 768;expected=2535428+60216*c
        if ready['cache_policy']!=policy or ready['capacity']!=768 or ready['resident_capacity']!=c or ready['gpu_session_bytes']!=expected or ready['tracked_gpu_bytes']!=3124434964+expected or ready['stack_limit']!=2 or ready['legacy_checkpoint']:raise ValueError('Activated protocol/resource profile differs')
        if len(p['turns'])!=10 or [t['input_index'] for t in p['turns']]!=[0,3,1,2,4,5,6,7,4,4] or len(p['refusals'])!=11 or not p['exact_sibling_replay']:raise ValueError('Protocol mechanics coverage differs')
        checks=p['generation_checks']
        if len(checks)!=1 or not checks[0]['state_equal_except_marker_id'] or not checks[0]['repeated_eos_empty_and_nonmutating']:raise ValueError('Repeated generation coverage differs')
        for t in p['turns'][:8]:
            ix=t['input_index'];historical=next(x for x in prior['turns'] if x['input_index']==ix);gen=t['generated'];before=t['prefill'];after=t['final'];delivered=t['delivered_state']
            if gen['ids']!=historical['answer_ids'] or before['logits_sha256']!=historical['prefill']['sha256'] or after['logits_sha256']!=historical['final']['sha256'] or after['position']!=historical['final_position'] or not delivered['delivered'] or delivered['pending_token']!=gen['ids'][-1] or after['delivered'] or after['position']!=delivered['position']+1:raise ValueError('Historical readout or consumed-token ownership differs')
        if p['delayed_root']['cache_sha256']!=prior['root_sha256'] or p['delayed_root']['position']!=prior['root_position'] or any(not x['state_unchanged'] for x in p['refusals']) or p['reset']['position'] or p['reset']['stack_depth'] or p['reset']['evictions'] or p['reset']['resident_rows']:raise ValueError('Persistent lifecycle/root differs')
        trace=json.loads((raw/(policy+'-protocol.json')).read_text())
        if trace[0]['reply']!=ready or len([x for x in trace if 'error' in x['reply']])!=11:raise ValueError('Raw protocol/refusal join differs')
        metrics={'policy':policy,'historical_turns_matched':8,'historical_prefill_and_final_logits_matched':16,'state_gpu_bytes':expected,'nonmutating_refusals':len(p['refusals']),'exact_sibling_replay':True,'split_generation_and_repeated_eos':True,'delayed_root_position':p['delayed_root']['position']};protocol.append(metrics);observations.append({'kind':'persistent-chat-mechanics','case':policy,'metrics':metrics})
    for entry in driver['clients']:
        name=entry['name'];stdout=(raw/(name+'.stdout.txt')).read_text();stdin=(raw/(name+'.stdin.txt')).read_text();stderr=(raw/(name+'.stderr.txt')).read_text()
        if entry['returncode'] or sha(raw/(name+'.stdout.txt'))!=entry['stdout_sha256'] or sha(raw/(name+'.stdin.txt'))!=entry['stdin_sha256']:raise ValueError('Text-client custody/exit differs')
        resets=stdout.count('[recomputed prompt after token-prefix change; saved parent and nested frames invalidated]');replies=re.findall(r'model> ([\s\S]*?)(?=you> |\Z)',stdout);states=[]
        # State is intentionally ordinary client printing of a Python dict.
        import ast
        for line in stdout.splitlines():
            pos=line.find("{'event': 'state'")
            if pos>=0:states.append(ast.literal_eval(line[pos:]))
        first=stdout.splitlines()[0];ready=json.loads(first)
        if ready.get('cache_policy','native')!=entry['policy']:raise ValueError('Client requested policy not active')
        if entry['policy']!='native' and (not states or any(x['resident_capacity']!=ready['resident_capacity'] or x['gpu_state_bytes']!=ready['gpu_session_bytes'] for x in states)):raise ValueError('Client live allocation changed')
        metrics={'name':name,'policy':entry['policy'],'answer_texts':replies,'prompt_recomputations':resets,'state_reports':len(states),'maximum_position':max([x['position'] for x in states],default=0),'state_gpu_bytes':ready['gpu_session_bytes'],'native_model_weights_and_forward_retained':True};clients.append(metrics);observations.append({'kind':'interactive-client-observation','case':name,'metrics':metrics})
    install=driver['installation']
    if (installation_stage=='published' and (not install['alias_updated'] or not install['stable_current_unchanged'])) or (installation_stage=='prepared' and install['alias_updated']) or install['service_started'] or install['binary_sha256']!=driver['binary_sha256'] or install['client_sha256']!=sha(HERE/'chat.py') or install['source_manifest_sha256']!=sha(HERE/'source-manifest.json'):raise ValueError('Opt-in installation receipt differs')
    installed=json.loads((raw/'installed-files.json').read_text());sources=json.loads((HERE/'source-manifest.json').read_text())['sources'];expected={'source/'+n:h for n,h in sources.items()};expected.update({'source/source-manifest.json':install['source_manifest_sha256'],'build/tcm-bounded-chat':install['binary_sha256'],'build/tcm-cuda':install['stable_native_sha256'],'tools/chat.py':install['client_sha256']})
    if sha(raw/'installed-files.json')!=install['installed_manifest_sha256'] or installed['files']!=expected or len(expected)!=install['installed_file_count'] or installed['stable_source']!=install['stable_path']:raise ValueError('Installed-file custody differs')
    return {'summary':{'protocol':protocol,'clients':clients,'installation':install,'scope':'opt-in persistent execution and historical parity; bounded recall accuracy remains unestablished; native current unchanged'},'observations':observations}
