"""Bound evidence extraction by local submission authority, not remote output."""
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import intake_fleet


class CollectionBudget(unittest.TestCase):
    def collect(self,local_budget,remote_budget,size):
        archive=io.BytesIO()
        with tarfile.open(fileobj=archive,mode='w:gz') as output:
            member=tarfile.TarInfo('raw/toy.bin');member.size=size
            output.addfile(member,io.BytesIO(b'x'*size))
        import gzip
        report={'status':'completed','config':{'evidence_budget_bytes':remote_budget}}
        logs=('TCM_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(json.dumps(report).encode())).decode()+'\n'+
              'TCM_FILES_TGZ_B64='+base64.b64encode(archive.getvalue()).decode()+'\n').encode()
        class Api:
            def json(self,*args):return {'state':'succeeded'}
            def request(self,*args,**kwargs):return None,logs,None
        module=types.ModuleType('mrun.client.api');module.Api=Api
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'results/toy';target.mkdir(parents=True)
            (target/'submission.json').write_text(json.dumps({'config':{'evidence_budget_bytes':local_budget}}))
            with patch.object(intake_fleet,'ROOT',root),patch.dict(sys.modules,{'mrun.client.api':module}),contextlib.redirect_stdout(io.StringIO()):
                intake_fleet.collect('toy')
            return (target/'raw/toy.bin').read_bytes()

    def test_remote_output_cannot_increase_local_budget(self):
        with self.assertRaisesRegex(ValueError,'evidence budget exceeded'):self.collect(2,256*1024**2,3)
    def test_declared_budget_and_absolute_ceiling(self):
        self.assertEqual(self.collect(3,1,3),b'xxx')
        for invalid in (True,0,256*1024**2+1):
            with self.assertRaisesRegex(ValueError,'invalid evidence budget'):self.collect(invalid,1,1)


if __name__=='__main__':unittest.main()
