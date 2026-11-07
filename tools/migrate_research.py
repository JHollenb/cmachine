"""Move the local research tree atomically, preserving the legacy address.

Raw evidence stays local. A full byte manifest is kept in research/migration;
the small receipt can be committed without publishing model/result payloads.
"""
from pathlib import Path
import gzip,hashlib,json,os

ROOT=Path(__file__).resolve().parents[1]

def migrate():
    source=ROOT.parent/'saturn/cmachine';target=ROOT/'research'
    if source.is_symlink() or target.exists():
        raise SystemExit('Migration already exists; inspect its receipt rather than repeating it')
    before=source.stat();source.rename(target)
    try:source.symlink_to(os.path.relpath(target,source.parent),target_is_directory=True)
    except BaseException:
        target.rename(source);raise
    after=target.stat()
    assert (before.st_dev,before.st_ino)==(after.st_dev,after.st_ino)
    assert source.resolve()==target.resolve()
    inventory=[];total=0
    for p in sorted(target.rglob('*')):
        name=str(p.relative_to(target))
        if p.is_symlink():inventory.append({'path':name,'symlink':os.readlink(p)})
        elif p.is_file():
            first=p.stat();h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1024**2),b''):h.update(b)
            last=p.stat()
            if (first.st_size,first.st_mtime_ns)!=(last.st_size,last.st_mtime_ns):
                raise RuntimeError('Concurrent mutation during manifest: '+name)
            inventory.append({'path':name,'bytes':first.st_size,'sha256':h.hexdigest()});total+=first.st_size
    data=''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in inventory).encode()
    manifest=target/'migration/tree.jsonl.gz';manifest.parent.mkdir(exist_ok=True);manifest.write_bytes(gzip.compress(data,mtime=0))
    receipt={'schema':'cmachine-research-migration-v1','source':str(source),'destination':str(target),'legacy_link':os.readlink(source),'atomic_same_filesystem_rename':True,'same_directory_inode':True,'files_and_links':len(inventory),'file_bytes':total,'manifest':'research/migration/tree.jsonl.gz','manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'uncompressed_manifest_sha256':hashlib.sha256(data).hexdigest(),'historical_bytes_rewritten':False}
    out=ROOT/'docs/evidence/research-migration.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':migrate()
