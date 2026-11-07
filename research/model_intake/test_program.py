"""Tiny generated organisms only; pretrained comparisons run through mrun."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools"))
from program import Program, lower_fx, lower_mrun
from inspect_program import inspect
import intake_worker
from program_chat import verify_tokenizer_binding


class PhaseToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq=nn.Parameter(torch.tensor([-1.3,2.7]))
        self.reference=nn.Parameter(torch.tensor([-4.,9.]))
        self.decoder=nn.Linear(2,3)
    def forward(self,x):
        a=x[:,0].float().unsqueeze(-1)
        phase=(a*self.freq.unsqueeze(0)) % (2*np.pi)
        ref=torch.remainder(self.reference,2*np.pi)
        gate=(1+torch.cos(phase-ref.unsqueeze(0)))/2
        return self.decoder(gate)


class ProgramMechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["make","-C",str(ROOT)],check=True,capture_output=True)
        torch.set_num_threads(1)

    def run_c(self,path,rows,*flags,check=True):
        root=path.parent
        np.asarray(rows,dtype=np.float32).tofile(root/"input.f32")
        result=subprocess.run([os.environ.get("TCM_TEST_EXECUTABLE",str(ROOT/"build/tcm")),"program",str(path),"--input",str(root/"input.f32"),"--output",str(root/"logits.f32"),*flags],capture_output=True,text=True)
        if check:self.assertEqual(result.returncode,0,result.stderr)
        return result,np.fromfile(root/"logits.f32",dtype=np.float32) if (root/"logits.f32").exists() else None

    def test_fx_fresh_rows_negative_remainder_and_trace(self):
        torch.manual_seed(2);model=PhaseToy().eval()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"toy.tcm";mapping=lower_fx(model,1,"1"*64,path)
            rows=[[-8.],[-1.],[0.],[11.]]
            result,actual=self.run_c(path,rows,"--trace",str(Path(tmp)/"trace.f32"))
            with torch.no_grad():expected=model(torch.tensor(rows)).numpy()
            np.testing.assert_allclose(actual.reshape(expected.shape),expected,atol=2e-6,rtol=2e-6)
            self.assertEqual(len(np.fromfile(Path(tmp)/"trace.f32",dtype=np.float32)),len(rows)*sum(v["length"] for v in mapping["values"]))
            self.assertFalse(json.loads(result.stdout)["framework_fallback"])
            symbols=inspect(mapping,trace=Path(tmp)/"trace.f32",row=2,job_id="toy")
            self.assertEqual(symbols["symbols"][0]["observation"]["min"],0)
            self.assertEqual(symbols["symbols"][0]["consumers"],[0])
            self.assertEqual(symbols["symbols"][-1]["producer"]["kind"],"linear")
            mapping["values"][0]["length"]=2
            with self.assertRaisesRegex(ValueError,"fingerprint"):
                inspect(mapping)

    def test_mamba_canonical_graph_state_and_own_tokens(self):
        from transformers import MambaConfig, MambaForCausalLM
        torch.manual_seed(3)
        config=MambaConfig(hidden_size=4,state_size=2,num_hidden_layers=2,expand=2,conv_kernel=3,time_step_rank=1,vocab_size=7)
        model=MambaForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"source";model.save_pretrained(source)
            path=Path(tmp)/"toy.tcm";mapping=lower_mrun(source,path)
            result,actual=self.run_c(path,[[1],[2],[3]],"--generate","4","--replay")
            with torch.no_grad():expected=model(torch.tensor([[1,2,3]]),use_cache=False).logits[0].numpy()
            np.testing.assert_allclose(actual.reshape(-1,7)[:3],expected,atol=2e-5,rtol=2e-5)
            record=json.loads(result.stdout)
            self.assertTrue(record["replay_exact"])
            self.assertEqual(record["consumed_rows"],6)
            tokens=[1,2,3];selected=[]
            with torch.no_grad():
                for _ in range(4):
                    token=int(model(torch.tensor([tokens]),use_cache=False).logits[0,-1].argmax());selected.append(token);tokens.append(token)
            self.assertEqual(record["generated_tokens"],selected)
            self.assertEqual(record["pending_token"],selected[-1])
            self.assertEqual(len(mapping["state_slots"]),4)
            parameters={p["source_name"]:p for p in mapping["parameters"]}
            self.assertEqual(parameters["token_embedding.weight"]["bundle_name"],parameters["lm_head.weight"]["bundle_name"])
            self.assertTrue(all(o["source"].startswith("mrun://ir/") for o in mapping["instructions"]))

    def test_target_rejects_unimplemented_fx(self):
        class Unknown(nn.Module):
            def forward(self,x):return torch.sin(x)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"unimplemented FX"):
                lower_fx(Unknown(),1,"2"*64,Path(tmp)/"bad.tcm")
            self.assertFalse((Path(tmp)/"bad.tcm").exists())

    def test_native_rejects_bad_slice_before_execution(self):
        p=Program("3"*64,1);p.output=p.emit("slice",2,offset=3)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"bad.tcm";p.save(path)
            result,_=self.run_c(path,[[1]],check=False)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("shape or state",result.stderr)

    def test_semantic_vocab_excludes_physical_padding(self):
        p=Program("4"*64,1,token_mode=True,vocab=3)
        w=p.parameter("embedding",torch.tensor([[1.,0.],[0.,1.],[1.,1.],[100.,100.]]))
        hidden=p.emit("embedding",2,w=w)
        p.output=p.emit("linear",4,a=hidden,w=w)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"padded.tcm";mapping=p.save(path)
            result,_=self.run_c(path,[[2]],"--generate","2","--replay")
            self.assertTrue(all(t<3 for t in json.loads(result.stdout)["generated_tokens"]))
            result,_=self.run_c(path,[[3]],check=False)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("semantic token domain",result.stderr)

    def test_late_source_change_never_publishes_bundle(self):
        p=Program("5"*64,1);p.output=p.emit("affine",1,scale=1)
        def changed():raise RuntimeError("source changed")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"guarded.tcm"
            with self.assertRaisesRegex(RuntimeError,"source changed"):
                p.save(path,source_guard=changed)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_fx_identity_includes_program_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=lower_fx(PhaseToy(),1,"6"*64,Path(tmp)/"a.tcm",source_record={"code":"a"})
            b=lower_fx(PhaseToy(),1,"6"*64,Path(tmp)/"b.tcm",source_record={"code":"b"})
            self.assertNotEqual(a["instructions"][0]["source"],b["instructions"][0]["source"])

    def test_failed_optional_arm_preserves_completed_baseline(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(intake_worker,"RAW",Path(tmp)):
            baseline={"source":"known-source","status":"completed","tokens":[1,2]}
            def broken():raise RuntimeError("injected optional arm failure")
            arm=intake_worker.optional_arm("toy",baseline,broken)
            self.assertEqual(arm["status"],"error")
            self.assertIn("injected optional arm failure",arm["error"])
            self.assertEqual(json.loads((Path(tmp)/"toy-baseline-summary.json").read_text()),baseline)
            self.assertEqual(json.loads((Path(tmp)/"toy-intervention-summary.json").read_text()),arm)
            self.assertEqual(baseline["status"],"completed")

    def test_tokenizer_binding_rejects_auxiliary_changes_and_model_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source";source.mkdir()
            assets={"tokenizer_config.json":b"{}","emoji.json":b'{"a":"b"}'}
            for name,data in assets.items():(source/name).write_bytes(data)
            origin={"source_fingerprint":"7"*64,"source_files":[
                {"path":name,"byte_count":len(data),"sha256":hashlib.sha256(data).hexdigest(),
                 "role":"tokenizer-config" if name.startswith("tokenizer") else "metadata"}
                for name,data in assets.items()]}
            p=Program("7"*64,1,origin=origin);p.output=p.emit("affine",1,scale=1)
            path=root/"model.tcm";p.save(path)
            bound=verify_tokenizer_binding(path,source)
            self.assertIn("emoji.json",bound["assets"])
            (source/"emoji.json").write_bytes(b'{"a":"c"}')
            with self.assertRaisesRegex(ValueError,"asset differs"):
                verify_tokenizer_binding(path,source)
            (source/"emoji.json").write_bytes(assets["emoji.json"])
            (source/"tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"unbound"):
                verify_tokenizer_binding(path,source)
            (source/"tokenizer.json").unlink()
            other=Program("8"*64,1);other.output=other.emit("affine",1,scale=2);other.save(root/"other.tcm")
            path.write_bytes((root/"other.tcm").read_bytes())
            with self.assertRaisesRegex(ValueError,"model and source map differ"):
                verify_tokenizer_binding(path,source)

if __name__=="__main__":unittest.main()
