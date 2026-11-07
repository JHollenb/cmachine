"""Saved-operand query × key-bank crosses at each fixed writer/history."""
import argparse
import json
from pathlib import Path
import numpy as np
from numerics import full_read
from results_analysis import load,sha,summary,square


def analyze(root):
    root=Path(root);raw=root/'raw';report=json.loads((root/'report.json').read_text())
    paths={p['name']:p for p in report['driver']['paths']};rows=[]
    for task in range(6):
        for writer in (0,1):
            native_name=f'case{task}-replay-'+('10' if writer else '00')
            cheap_name=f'case{task}-'+('own-11' if writer else 'replay-01')
            native,cheap=paths[native_name],paths[cheap_name];n=native['count']
            if native['consumed_ids']!=cheap['consumed_ids']:raise ValueError('Reader crosses need common IDs')
            arrays={}
            for name in (native_name,cheap_name):
                arrays[name]={'q':load(raw,name+'-q.f32',(n,1536)),
                              'k':load(raw,name+'-k.f32',(n,2,128)),
                              'v':load(raw,name+'-v.f32',(n,2,128))}
            a,b=arrays[native_name],arrays[cheap_name]
            if not np.array_equal(a['v'],b['v']):raise ValueError('Values must be reader-independent')
            for turn in native['turns']:
                for stage in ('prefill','final'):
                    t=turn[stage]['position']-1;v=a['v'][:t+1]
                    # Here square's first axis is Q and its second is K, not
                    # the writer/reader policy axes used by results_analysis.
                    corners=[full_read(a['q'][t],a['k'][:t+1],v),
                             full_read(b['q'][t],a['k'][:t+1],v),
                             full_read(a['q'][t],b['k'][:t+1],v),
                             full_read(b['q'][t],b['k'][:t+1],v)]
                    s=square(corners)
                    rows.append({'task':task,'writer':'compiled' if writer else 'native',
                                 'turn':turn['turn'],'stage':stage,'position':t+1,
                                 'query_only_relative':s['writer_relative'],
                                 'key_only_relative':s['reader_relative'],
                                 'both_relative':s['joint_relative'],
                                 'query_key_interaction_relative':s['interaction_relative'],
                                 'query_key_cross_scaled':s['writer_reader_cross_scaled'],
                                 'gram_scaled':s['gram_scaled'],'closure_max_abs':s['closure_max_abs'],
                                 'query_operand_relative':float(np.linalg.norm(b['q'][t]-a['q'][t])/np.linalg.norm(a['q'][t])),
                                 'key_bank_relative':float(np.linalg.norm(b['k'][:t+1]-a['k'][:t+1])/np.linalg.norm(a['k'][:t+1]))})
    grouped={writer:{key:summary(r[key] for r in rows if r['writer']==writer)
                     for key in ('query_only_relative','key_only_relative','both_relative',
                                 'query_key_interaction_relative','query_operand_relative','key_bank_relative')}
             for writer in ('native','compiled')}
    return {'schema':'operand-composition-saved-reader-cross-v1','report_sha256':sha(root/'report.json'),
            'analysis_source_sha256':sha(__file__),'summary':grouped,'rows':rows,
            'scope':'CPU64 attention equations on actual saved operands at matching histories and fixed writer; current query/key crosses do not run a new native future.',
            'models_evaluated':0,'terminal_status':'not-requested'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();result=analyze(a.root)
    out=a.root/'analysis';out.mkdir(exist_ok=True)
    (out/'reader-attribution.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
