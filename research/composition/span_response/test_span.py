"""Small model-free constructor, visibility, math and payload checks."""
import math,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from analysis import bank_read,response,mix
HERE=Path(__file__).resolve().parent

class SpanPrograms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);exe=cls.root/"smoke"
        subprocess.run(["cc","-O2","-std=c11","-ffp-contract=off","-I"+str(HERE),str(HERE/"bank_smoke.c"),"-lm","-o",str(exe)],check=True,capture_output=True)
        cls.result=subprocess.run([str(exe),str(cls.root)],check=True,capture_output=True,text=True)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_original_operands_not_retained(self):self.assertIn("predictions unchanged",self.result.stdout)
    def test_exact34_and_program_extents(self):
        b=bank_read(self.root/"fixture-34.bin")
        np.testing.assert_array_equal(b["keys"],np.fromfile(self.root/"original-keys.f32",dtype="<f4").reshape(2,34,128));np.testing.assert_array_equal(b["values"],np.fromfile(self.root/"original-values.f32",dtype="<f4").reshape(2,34,128));np.testing.assert_array_equal(b["counts"],1)
        self.assertEqual((self.root/"fixture-8.bin").stat().st_size,16480);self.assertEqual((self.root/"fixture-1.bin").stat().st_size,2088)
    def test_counts_and_partitions(self):
        for m in (1,8,34):
            b=bank_read(self.root/f"fixture-{m}.bin");members=np.fromfile(self.root/f"members-{m}.u64",dtype="<u8").reshape(2,m)
            for head in range(2):
                union=0
                for count,mask in zip(b["counts"][head],members[head]):
                    self.assertEqual(union&int(mask),0);self.assertEqual(int(count),int(mask).bit_count());union|=int(mask)
                self.assertEqual(union,(1<<34)-1)
    def test_bank_only_response_agrees_with_C(self):
        q=np.fromfile(self.root/"query.f32",dtype="<f4").reshape(12,128)
        for m in (1,8,34):
            b=bank_read(self.root/f"fixture-{m}.bin");c=np.fromfile(self.root/f"response-{m}.f64",dtype="<f8").reshape(12,129)
            np.testing.assert_allclose(response(q,b["keys"],b["values"],b["counts"]),c,rtol=1e-12,atol=1e-12)
    def test_jensen_for_ideal_memory_only_means(self):
        q=np.fromfile(self.root/"query.f32",dtype="<f4").reshape(12,128);keys=np.fromfile(self.root/"original-keys.f32",dtype="<f4").reshape(2,34,128);values=np.zeros_like(keys);logs=[]
        for m in (1,8,34):
            members=np.fromfile(self.root/f"members-{m}.u64",dtype="<u8").reshape(2,m);b=bank_read(self.root/f"fixture-{m}.bin");means=np.asarray([[np.mean(keys[h,[(int(mask)&(1<<i))!=0 for i in range(34)]].astype(float),axis=0) for mask in members[h]] for h in range(2)])
            logs.append(response(q,means,np.zeros_like(means),b["counts"])[:,0])
        self.assertTrue(np.all(logs[0]<=logs[1]+1e-12));self.assertTrue(np.all(logs[1]<=logs[2]+1e-12))
    def test_weighted_entries_share_global_normalizer(self):
        q=np.zeros((12,128));k=np.zeros((2,1,128));v=np.ones((2,1,128));span=response(q,k,v,np.full((2,1),3));complement=response(q,k,2*v,np.full((2,1),7));np.testing.assert_allclose(mix(complement,span),1.7)
    def test_receipt_bridge_import_closure(self):
        sys.path.insert(0,str(HERE/"public"));from saturn_bridge import invoke
        self.assertTrue(callable(invoke))

if __name__=="__main__":unittest.main()
