"""Model-free tests of the actual text client, transport and policy authority."""
from pathlib import Path
import json,os,subprocess,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent
TOKENIZER='''class Tokenizer:
    def apply_chat_template(self,messages,**kw):return [1,2,3]
    def decode(self,ids,**kw):return "ok"
class AutoTokenizer:
    @staticmethod
    def from_pretrained(*args,**kw):return Tokenizer()
'''
SERVER='''#!/usr/bin/env python3
import sys,json,os
args=sys.argv
def value(k,default):return args[args.index(k)+1] if k in args else default
policy=os.environ.get("TASK_CHAT_READY_POLICY",value("--cache-policy","native"))
ready={"event":"ready","capacity":int(value("--context","768")),"stack_limit":2 if policy!="native" else 64,"legacy_checkpoint":policy=="native"}
if policy!="native":ready.update(cache_policy=policy,anchors=int(value("--anchors","16")),recent=int(value("--recent","64")),reservoir=int(value("--reservoir","32")))
print(json.dumps(ready),flush=True)
frames=[];counter=0
for line in sys.stdin:
    text=line.strip()
    with open(os.environ["TASK_CHAT_TEST_LOG"],"a") as f:f.write(json.dumps({"args":args,"command":text})+"\\n")
    if text=="Q":break
    if text.startswith("A "):reply={"event":"appended"}
    elif text.startswith("G "):reply={"event":"generated","ids":[4,5]}
    elif text=="P":
        counter+=1;frames.append(counter);reply={"event":"pushed","frame":counter}
    elif text.startswith(("B ","K ")):frames.pop();reply={"event":"changed"}
    elif text=="S":reply={"event":"state","stack_depth":len(frames)}
    else:reply={"event":"other"}
    print(json.dumps(reply),flush=True)
'''
class Client(unittest.TestCase):
    def run_client(self,args,transcript,env_extra=None):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'transformers.py').write_text(TOKENIZER);binary=p/'server';binary.write_text(SERVER);binary.chmod(0o755);log=p/'commands.jsonl';env={**os.environ,'PYTHONPATH':str(p),'TASK_CHAT_TEST_LOG':str(log),**(env_extra or {})};r=subprocess.run([sys.executable,str(HERE/'chat.py'),'--binary',str(binary),'--model',str(p/'fake.tcm'),'--tokenizer',str(p),'--context','768','--new-tokens','8',*args],input=transcript,capture_output=True,text=True,env=env,timeout=20);rows=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else [];return r,rows
    def test_bounded_client_enforces_snapshot_budget_and_legacy_commands(self):
        r,rows=self.run_client(['--cache-policy','bounded'],'hello\n/push\n/push\n/push\n/rollback\n/commit\n/state\n/save\n/rewind\n/quit\n');self.assertEqual(r.returncode,0,r.stderr);commands=[x['command'] for x in rows];self.assertEqual(commands.count('P'),2);self.assertEqual([x for x in commands if x.startswith(('B ','K '))],['B 2','K 1']);self.assertNotIn('C',commands);self.assertNotIn('R',commands);self.assertIn('Nested frame limit reached',r.stdout);self.assertIn('legacy /save is unavailable',r.stdout);self.assertIn('--cache-policy',rows[0]['args'])
    def test_native_remains_default_and_legacy_ready_is_accepted(self):
        r,rows=self.run_client([],'hello\n/quit\n');self.assertEqual(r.returncode,0,r.stderr);self.assertNotIn('--cache-policy',rows[0]['args']);self.assertIn('model> ok',r.stdout)
    def test_wrong_activated_policy_refuses(self):
        r,rows=self.run_client(['--cache-policy','bounded'],'hello\n',{'TASK_CHAT_READY_POLICY':'native'});self.assertNotEqual(r.returncode,0);self.assertIn('did not activate the requested cache policy',r.stderr);self.assertFalse(rows)
    def test_explicit_binary_cannot_silently_activate_bounded_under_native(self):
        r,rows=self.run_client([],'hello\n',{'TASK_CHAT_READY_POLICY':'bounded'});self.assertNotEqual(r.returncode,0);self.assertIn('did not activate the requested cache policy',r.stderr);self.assertFalse(rows)
    def test_invalid_bounded_geometry_refuses_before_tokenizer_or_process(self):
        r,rows=self.run_client(['--cache-policy','bounded','--recent','0'],'hello\n');self.assertEqual(r.returncode,2);self.assertIn('positive recent rows',r.stderr);self.assertFalse(rows)
if __name__=='__main__':unittest.main()
