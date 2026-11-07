"""Versioned canonical installation with reversible legacy alias publication."""
from pathlib import Path
import json,os,shutil,uuid
from support import sha
ALIASES=(Path('/mnt/big/cmachine/current'),Path('/mnt/big/saturn-cmachine/current'),Path('/mnt/big/saturn-cmachine/bounded-current'))
def prepare(root,manifest):
    root=Path(root);binary=sha(root/'build/tcm-cuda');target=ALIASES[0].parent/'builds'/(manifest['git_commit'][:16]+'-'+binary[:12]);previous={}
    for a in ALIASES:
        if a.exists() and not a.is_symlink():raise ValueError('Installation alias is an ordinary path')
        previous[str(a)]=os.readlink(a) if a.is_symlink() else None
    files={'source/'+n:r['sha256'] for n,r in manifest['sources'].items()};files.update({'source/source-manifest.json':sha(root/'source-manifest.json'),'build/tcm-cuda':binary,'build/tcm-bounded-chat':binary,'tools/chat.py':sha(root/'tools/chat.py')})
    record={'schema':'cmachine-installed-files-v1','git_commit':manifest['git_commit'],'git_tree':manifest['git_tree'],'files':files}
    if not target.exists():
        temp=target.with_name(target.name+'.preparing-'+uuid.uuid4().hex);temp.mkdir(parents=True)
        for name in files:
            dest=temp/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if name=='build/tcm-bounded-chat':dest.symlink_to('tcm-cuda');continue
            shutil.copy2(root/(name[7:] if name.startswith('source/') else name),dest)
        (temp/'install-manifest.json').write_text(json.dumps(record,indent=2)+'\n');temp.rename(target)
    if json.loads((target/'install-manifest.json').read_text())!=record:raise ValueError('Existing installed manifest differs')
    for n,h in files.items():
        if sha(target/n)!=h:raise ValueError('Installed file differs: '+n)
    return {'schema':'cmachine-install-v1','path':str(target),'git_commit':manifest['git_commit'],'binary_sha256':binary,'client_sha256':sha(root/'tools/chat.py'),'manifest_sha256':sha(target/'install-manifest.json'),'file_count':len(files),'previous_aliases':previous,'published':False,'service_started':False}
def publish(record):
    changed=[]
    try:
        for text,old in record['previous_aliases'].items():
            a=Path(text);current=os.readlink(a) if a.is_symlink() else None
            if current!=old or (a.exists() and not a.is_symlink()):raise ValueError('Installation alias changed concurrently')
        for text,old in record['previous_aliases'].items():
            a=Path(text);a.parent.mkdir(parents=True,exist_ok=True);temp=a.with_name(a.name+'.next-'+uuid.uuid4().hex);temp.symlink_to(record['path']);os.replace(temp,a);changed.append((a,old))
        if any(Path(a).resolve()!=Path(record['path']) for a in record['previous_aliases']):raise ValueError('Published installation alias differs')
    except BaseException:
        for a,old in reversed(changed):
            if a.is_symlink() and os.readlink(a)==record['path']:
                if old is None:a.unlink()
                else:
                    temp=a.with_name(a.name+'.restore-'+uuid.uuid4().hex);temp.symlink_to(old);os.replace(temp,a)
        raise
    record['published']=True;return record
