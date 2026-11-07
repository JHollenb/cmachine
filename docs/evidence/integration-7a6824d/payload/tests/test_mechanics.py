"""Model-free integrity and transition regressions; scientific runs use mrun."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from bundle import write_bundle


class Mechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["make","-C",str(ROOT)],check=True,capture_output=True)

    def test_checksum_and_extent_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"tiny.tcm"
            write_bundle(path,1,"0"*64,[48,3],{"test":("float32",(2,),struct.pack("<ff",1,2))})
            result=subprocess.run([ROOT/"build/tcm","inspect",path],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)["tensors"],1)
            raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
            result=subprocess.run([ROOT/"build/tcm","inspect",path],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("SHA256",result.stderr)
            # A correctly checksummed descriptor still cannot read beyond EOF.
            raw[-1]^=1
            struct.pack_into("<Q",raw,176+104,len(raw)+8)
            raw[8:40]=hashlib.sha256(raw[40:]).digest();path.write_bytes(raw)
            result=subprocess.run([ROOT/"build/tcm","inspect",path],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("extent",result.stderr)

    def test_atomic_commit_and_phase_contract(self):
        source=r'''
#include "machine.h"
#include <assert.h>
#include <math.h>
#include <string.h>
int main(void) {
    TcmModel bundle={0};IsaModel model={0};model.bundle=&bundle;
    IsaSession session;isa_session_init(&session,&model,0);
    session.state.phase=2;session.state.row=64;
    session.state.arch.registers[0]=123;session.state.arch.rip=UINT64_MAX;
    IsaArch before=session.state.arch;
    assert(isa_commit(&session)==-1);
    assert(!memcmp(&before,&session.state.arch,sizeof(before)));
    session.state.arch.rip=0;before=session.state.arch;
    session.state.output.syscall=NAN;
    assert(isa_commit(&session)==-1);
    assert(!memcmp(&before,&session.state.arch,sizeof(before)));
    assert(isa_begin(&session,0,0)==-1);
    IsaCheckpoint checkpoint;isa_checkpoint(&session,&checkpoint);
    checkpoint.state.row=31;checkpoint.state.phase=2;
    assert(isa_restore(&session,&checkpoint)==-1);
    session.state.output.syscall=0;
    assert(isa_commit(&session)==0);
    assert(session.state.arch.rip==1);
    float bits[64]={0};bits[63]=1;assert(isa_bits(bits)==(UINT64_C(1)<<63));
    float equal[3]={2,2,1};assert(isa_argmax(equal,3)==0);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"test.c";path.write_text(source);binary=Path(tmp)/"test"
            subprocess.run(["cc","-std=c11","-ffp-contract=off","-I",str(ROOT/"src"),str(path),str(ROOT/"src/isa.c"),str(ROOT/"src/model.c"),"-lm","-o",str(binary)],check=True,capture_output=True)
            subprocess.run([binary],check=True,capture_output=True)


if __name__=="__main__":unittest.main()
