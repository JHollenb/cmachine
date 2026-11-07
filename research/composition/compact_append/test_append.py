"""Small actual-runtime lifecycle/codec checks and executed payload imports."""
from pathlib import Path
import os,subprocess,sys,tempfile,unittest
import numpy as np
from analysis import encode,response
HERE=Path(__file__).resolve().parent
class AppendMechanics(unittest.TestCase):
    def test_actual_runtime_layout_append_abort_rollback(self):
        with tempfile.TemporaryDirectory() as temp:
            binary=Path(temp)/"mechanics"
            cmd=["cc","-O1","-g","-std=c11","-ffp-contract=off","-fsanitize=address,undefined","-Iruntime","-include","codec.h","cache_mechanics.c","runtime/qwen.c","runtime/model.c","gpu_stub.c","-lm","-o",str(binary)]
            build=subprocess.run(cmd,cwd=HERE,capture_output=True,text=True);self.assertEqual(build.returncode,0,build.stderr);result=subprocess.run([str(binary)],cwd=HERE,capture_output=True,text=True,env={**os.environ,"ASAN_OPTIONS":"detect_leaks=0"});self.assertEqual(result.returncode,0,result.stderr);self.assertIn("abort, replay",result.stdout)
    def test_vector_codec_signed_ties_and_zero(self):
        x=np.zeros((2,128),dtype=np.float32);x[0,:5]=[127,.5,-.5,1.5,-1.5];c,s,d=encode(x);np.testing.assert_array_equal(c[0,:5],[127,1,-1,2,-2]);np.testing.assert_array_equal(s,[1,1]);np.testing.assert_array_equal(d[1],0)
    def test_current_row_participates_in_response(self):
        q=np.zeros((12,128));k=np.zeros((2,2,128));v=np.zeros_like(k);v[:,1]=2;np.testing.assert_array_equal(response(q,k,v)[:,1:],1)
    def test_receipt_bridge_import_closure(self):
        env={**os.environ,"PYTHONPATH":str(HERE/"public")+os.pathsep+str(HERE)};subprocess.run([sys.executable,"-c","from saturn_bridge import invoke; assert callable(invoke)"],cwd=HERE,env=env,capture_output=True,check=True)
if __name__=="__main__":unittest.main()
