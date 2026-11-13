"""Native CLI against an ephemeral supplied 87-tensor ISA bundle, not weights
from any trained model. This verifies the real loader/observer/report path.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bundle", ROOT/"tools/bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def fixture(path):
    rng = np.random.default_rng(351)
    tensors = {}

    def tensor(name, shape, constant=None):
        value = np.full(shape, constant, dtype="<f4") if constant is not None else (rng.normal(size=shape)*.01).astype("<f4")
        tensors[name] = ("float32", shape, value.tobytes())

    def linear(name, width, out, bias=True):
        tensor(name+".weight", (out, width))
        if bias:
            tensor(name+".bias", (out,))

    for name, width, out in [("decoder.0",120,256),("decoder.2",256,256),("dest",256,8),
                             ("src",256,8),("query",272,32),("offset_head",256,8),("inp",293,48),
                             ("flag_query",256,64),("reg_out",48,8),("imm_out",48,1),("store_out",48,2),
                             ("flags.0",500,128),("flags.2",128,4),("width_head",256,3),("length",256,15),
                             ("syscall",256,1),("store_flag",256,1)]:
        linear(name,width,out)
    tensor("pos", (64,16)); tensor("ins_pos", (122,16))
    for layer in range(3):
        base = f"mixers.{layer}"
        linear(base+".in_proj",48,192); linear(base+".select",96,9,False)
        linear(base+".dt_proj",1,96); linear(base+".out_proj",96,48)
        linear(f"post.{layer}.1",48,96); linear(f"post.{layer}.3",96,48)
        tensor(base+".conv.weight",(96,1,3)); tensor(base+".conv.bias",(96,))
        tensor(base+".a_log",(96,4),-.7); tensor(base+".d",(96,))
        tensor(f"post.{layer}.0.weight",(48,),1); tensor(f"post.{layer}.0.bias",(48,))
    assert len(tensors) == 87
    epsilon = struct.unpack("<I", struct.pack("<f",1e-5))[0]
    bundle.write_bundle(path,1,hashlib.sha256(b"supplied-isa-fixture-v1").hexdigest(),[48,3,epsilon],tensors)


class IsaCircuitCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["make","-C",str(ROOT),"isa-circuit"],check=True,capture_output=True)

    def test_full_native_cli_root_recipe_and_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"supplied.tcm"; fixture(path)
            result=subprocess.run([ROOT/"build/tcm-isa-circuit",path,"--bytes","4889d8","--source","8",
                                   "--receiver","31","--cut","31","--dose","0"],check=True,capture_output=True,text=True)
            rows=[json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[0]["schema"],"tcm-isa-scan-inspection-v1")
            self.assertEqual(rows[0]["cross_cut_rank_bound_per_channel"],4)
            account=rows[1]
            total=sum(account[k] for k in ["copy_prior","head_bias","carry","gelu_affine","gelu_nonlinear","mlp_remainder","readout_remainder"])
            self.assertAlmostEqual(total,account["native_logit"],places=13)
            self.assertEqual(rows[2]["native_register_logit_delta"],0)
            self.assertTrue(rows[2]["parent_restored"])
            self.assertEqual(rows[2]["model_digest"],rows[0]["model_digest"])

    def test_invalid_cli_inputs_refuse_before_loading_model(self):
        for args in [["--bytes","xyz"],["--bytes","00","--dose","nan"],["--bytes","00","--channel","96"]]:
            result=subprocess.run([ROOT/"build/tcm-isa-circuit","absent",*args],capture_output=True)
            self.assertEqual(result.returncode,2)


if __name__=="__main__":
    unittest.main()
