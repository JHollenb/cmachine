from pathlib import Path
import os,subprocess,tempfile,unittest
HERE=Path(__file__).resolve().parent
class BoundedMechanics(unittest.TestCase):
    def test_ties_away_codec_and_constant_value_consumer(self):
        import numpy as np
        from analysis import State,codec
        row=np.zeros(128,dtype=np.float32);row[:6]=[127,np.nextafter(np.float32(.5),np.float32(0)),.5,-.5,63.5,-63.5];q,s=codec(row);self.assertEqual(q[:6].tolist(),[127,0,1,-1,64,-64]);self.assertEqual(float(s),1)
        state=State(2,4,3)
        for p in range(30):state.append(np.ones((3,2,2,128),dtype=np.float32))
        np.testing.assert_allclose(state.context(np.zeros((12,128)),0),1,atol=1e-14)
    def test_actual_session_guards_and_ring_under_sanitizers(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'mechanics';args=['cc','-O1','-g','-std=c11','-ffp-contract=off','-fsanitize=address,undefined','-Iruntime','mechanics.c','runtime/qwen.c','runtime/model.c','gpu_stub.c','weight_stub.c','-lm','-o',str(binary)]
            r=subprocess.run(args,cwd=HERE,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);r=subprocess.run([str(binary)],capture_output=True,text=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'});self.assertEqual(r.returncode,0,r.stderr);self.assertIn('accepted-sample abort',r.stdout)
    def test_reservoir_does_not_duplicate_recent_or_anchors(self):
        import numpy as np
        from analysis import State
        s=State(2,4,3);row=np.arange(1536,dtype=np.float32).reshape(3,2,2,128)
        for p in range(400):
            s.append(row);valid=s.pos[s.pos>=0].tolist();self.assertEqual(len(valid),len(set(valid)));expected=list(range(max(2,p-3),p+1)) if p>=2 else [];expected=[-1]*(4-len(expected))+expected;self.assertEqual(sorted(s.pos[2:6].tolist()),expected)
            if p>=5:self.assertEqual(s.n,p-5)
if __name__=='__main__':unittest.main()
