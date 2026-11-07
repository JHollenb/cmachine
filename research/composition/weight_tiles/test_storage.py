from pathlib import Path
import os,subprocess,tempfile,unittest,sys
import numpy as np
HERE=Path(__file__).resolve().parent
class StorageMechanics(unittest.TestCase):
    def test_public_panel_receipt_binds_complete_commands(self):
        sys.path.insert(0,str(HERE/'public'))
        from panel_bridge import invoke,CPanel
        commands=[['c-machine','model','source','correction','recall','prose','count',f'out{i}',str(i)] for i in range(3)];request={'execution_commands':commands};result,wrapper=invoke(commands,None,request,lambda _: {'profiles':[0,1,2]})
        self.assertEqual(result['profiles'],[0,1,2]);self.assertEqual(wrapper['request']['execution_commands'],commands);self.assertIn('three sequential',wrapper['scope'])
        with self.assertRaises(ValueError):CPanel(commands[:2],request,lambda _: {})
    def test_actual_runtime_with_owned_embedding_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'mechanics';args=['cc','-O1','-g','-std=c11','-ffp-contract=off','-fsanitize=address,undefined','-Iruntime','-include','codec.h','cache_mechanics.c','runtime/qwen.c','runtime/model.c','gpu_stub.c','weight_stub.c','-lm','-o',str(binary)]
            build=subprocess.run(args,cwd=HERE,capture_output=True,text=True);self.assertEqual(build.returncode,0,build.stderr);r=subprocess.run([str(binary)],capture_output=True,text=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'});self.assertEqual(r.returncode,0,r.stderr)
    def test_all_finite_bf16_encodings_widen_without_bit_loss(self):
        bits=np.arange(65536,dtype=np.uint16);finite=(bits&0x7f80)!=0x7f80;fp32=bits.astype(np.uint32)<<np.uint32(16);values=fp32.view(np.float32);self.assertTrue(np.isfinite(values[finite]).all());np.testing.assert_array_equal((values.view(np.uint32)>>16).astype(np.uint16),bits)
if __name__=='__main__':unittest.main()
