"""Content-bound opt-in installation; stable current is never rewritten."""
from pathlib import Path
import hashlib,json,os,shutil,uuid
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
BASE=Path('/mnt/big/saturn-cmachine')
def prepare(root):
    root=Path(root);stable=(BASE/'current').resolve(strict=True);native=stable/'build/tcm-cuda';native_sha=sha(native);binary_sha=sha(root/'build/tcm-bounded-chat');client_sha=sha(root/'chat.py');source_sha=sha(root/'source-manifest.json');key=hashlib.sha256((binary_sha+client_sha+source_sha+native_sha).encode()).hexdigest()[:16];target=BASE/'experimental'/('bounded-chat-'+key);alias=BASE/'bounded-current';previous=os.readlink(alias) if alias.is_symlink() else None
    if alias.exists() and not alias.is_symlink():raise ValueError('Opt-in alias exists as an ordinary path')
    expected={}
    for n in json.loads((root/'source-manifest.json').read_text())['sources']:expected['source/'+n]=sha(root/n)
    expected['source/source-manifest.json']=source_sha;expected.update({'build/tcm-bounded-chat':binary_sha,'build/tcm-cuda':native_sha,'tools/chat.py':client_sha})
    manifest={'schema':'tcm-bounded-chat-installed-files-v1','files':expected,'stable_source':str(stable),'stable_native_sha256':native_sha,'source_manifest_sha256':source_sha}
    if target.exists():
        if json.loads((target/'install-manifest.json').read_text())!=manifest:raise ValueError('Existing content installation differs')
    else:
        temporary=target.with_name(target.name+'.preparing-'+uuid.uuid4().hex);temporary.mkdir(parents=True)
        for relative in expected:
            dest=temporary/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            if relative.startswith('source/'):src=root/relative[7:]
            elif relative=='build/tcm-cuda':src=native
            elif relative=='tools/chat.py':src=root/'chat.py'
            else:src=root/relative
            shutil.copy2(src,dest)
        (temporary/'install-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');temporary.rename(target)
    for n,h in expected.items():
        if sha(target/n)!=h:raise ValueError('Installed byte differs: '+n)
    return {'schema':'tcm-bounded-chat-install-v1','path':str(target),'alias':str(alias),'previous_alias':previous,'stable_path':str(stable),'stable_native_sha256':native_sha,'binary_sha256':binary_sha,'client_sha256':client_sha,'source_manifest_sha256':source_sha,'installed_manifest_sha256':sha(target/'install-manifest.json'),'installed_file_count':len(expected),'alias_updated':False,'service_started':False}
def publish(receipt):
    alias=Path(receipt['alias']);now=os.readlink(alias) if alias.is_symlink() else None
    if alias.exists() and not alias.is_symlink():raise ValueError('Opt-in alias changed into an ordinary path')
    if now!=receipt['previous_alias'] or (BASE/'current').resolve()!=Path(receipt['stable_path']) or sha(Path(receipt['stable_path'])/'build/tcm-cuda')!=receipt['stable_native_sha256']:raise ValueError('Installation pointer changed concurrently')
    temporary=alias.with_name(alias.name+'.next-'+uuid.uuid4().hex);temporary.symlink_to(receipt['path']);os.replace(temporary,alias);receipt.update(alias_updated=True,stable_current_unchanged=True)
    if alias.resolve()!=Path(receipt['path']):raise ValueError('Opt-in link did not resolve')
    return receipt
