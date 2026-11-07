"""Small model-free checks of the actual byte C program and numerical account."""
from pathlib import Path
import os,subprocess,sys,tempfile,unittest
import numpy as np
from analysis import byte_read,response
HERE=Path(__file__).resolve().parent

class ByteProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);binary=cls.root/"smoke"
        subprocess.run(["cc","-O2","-std=c11","-ffp-contract=off","-I",str(HERE),str(HERE/"byte_smoke.c"),"-lm","-o",str(binary)],check=True,capture_output=True)
        subprocess.run([str(binary),str(cls.root/"bank.bin"),str(cls.root/"prediction.f64")],check=True,capture_output=True)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_extent_and_away_from_zero_ties(self):
        b=byte_read(self.root/"bank.bin");self.assertEqual((self.root/"bank.bin").stat().st_size,17984);np.testing.assert_array_equal(b["codes"][0,0,0,:5],[127,1,-1,2,-2])
    def test_zero_values_decode_exactly(self):
        b=byte_read(self.root/"bank.bin");np.testing.assert_array_equal(b["values"][0,0],np.zeros(128))
    def test_actual_C_predictor_matches_independent_response(self):
        b=byte_read(self.root/"bank.bin");q=np.tile(np.asarray([(d%7-3)*np.float32(.02) for d in range(128)],dtype=np.float32),(12,1));actual=np.fromfile(self.root/"prediction.f64",dtype="<f8").reshape(12,129);expected=response(q,b["keys"],b["values"]);np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14)
    def test_softmax_mean_bound_on_signed_extreme(self):
        eps=.7;p=np.asarray([1/(1+np.exp(eps)),np.exp(eps)/(1+np.exp(eps))]);r=p*np.exp([eps,-eps]);r/=r.sum();self.assertAlmostEqual(float(np.abs(p-r).sum()),2*np.tanh(eps/2),places=14)
    def test_actual_receipt_bridge_import_closure(self):
        root=HERE;env={**os.environ,"PYTHONPATH":str(root/"public")+os.pathsep+str(root)}
        subprocess.run([sys.executable,"-c","from saturn_bridge import invoke; import saturn_pub; assert callable(invoke)"],cwd=root,env=env,check=True,capture_output=True)

if __name__=="__main__":unittest.main()
