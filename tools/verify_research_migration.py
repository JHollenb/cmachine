"""Verify the relocated immutable payload/raw/report bytes without a model."""
from pathlib import Path
import gzip,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def verify():
    receipt=json.loads((ROOT/'docs/evidence/research-migration.json').read_text());root=ROOT/'research';manifest=ROOT/receipt['manifest'];assert sha(manifest)==receipt['manifest_sha256'];data=gzip.decompress(manifest.read_bytes());assert hashlib.sha256(data).hexdigest()==receipt['uncompressed_manifest_sha256'];rows=[json.loads(x) for x in data.splitlines()];checked=total=0
    assert Path(receipt['source']).resolve()==root.resolve()
    for r in rows:
        if 'sha256' not in r:continue
        p=Path(r['path']);protected=('payload' in p.parts or 'raw' in p.parts or ('runs' in p.parts and p.name=='report.json'))
        if protected:
            current=root/p
            if sha(current)!=r['sha256'] or current.stat().st_size!=r['bytes']:raise ValueError('Historical evidence differs: '+str(p))
            checked+=1;total+=r['bytes']
    out={'schema':'cmachine-migration-verification-v1','manifest_sha256':receipt['manifest_sha256'],'inventory_records':len(rows),'protected_files_verified':checked,'protected_bytes_verified':total,'legacy_address_resolves_to_canonical_research':True,'scope':'pre-move frozen payload,raw,and historical report bytes; live plans and new evidence are outside this check'}
    (ROOT/'docs/evidence/migration-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':verify()
