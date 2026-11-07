"""Additive FP64/fsum correction to FP32-denominator diagnostics; no model calls."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(a,b):
    a=np.asarray(a,dtype=float).reshape(-1);b=np.asarray(b,dtype=float).reshape(-1)
    numerator=math.sqrt(math.fsum(float(x-y)**2 for x,y in zip(a,b)))
    denominator=math.sqrt(math.fsum(float(x)**2 for x in b))
    return numerator/max(denominator,1e-30)

def correct(root):
    root=Path(root);report=json.loads((root/'report.json').read_text());paths={p['name']:p for p in report['driver']['paths']};rows=[]
    for old in report['analysis']['query_operand_comparisons']:
        name=old['candidate'];native=f'case{old["task"]}-native';n=old['positions'];row=dict(old)
        for port,key,width in (('q','query_relative_error',1536),('attention','attention_relative_error',1536),('output','layer_output_relative_error',1536)):
            values=[]
            for label in (name,native):
                file=f'{label}-{port}.f32';record=report['raw_files'][file];path=root/'raw'/file
                if sha(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise ValueError('Raw metric input differs')
                values.append(np.fromfile(path,dtype='<f4').reshape(paths[label]['count'],width)[:n])
            row[key]=relative(*values)
        rows.append(row)
    result={'schema':'tcm-query-fp64-metric-correction-v1','report_sha256':sha(root/'report.json'),'helper_sha256':sha(__file__),'models_evaluated':0,'scope':'Additive correction only: original query/attention/output relative metrics used an incoming-FP32 reference norm. These recompute numerator and denominator in FP64 with math.fsum. Original report and strict refusal remain unchanged. Full-logit comparisons and source fitting are unaffected.','rows':rows}
    (root/'fp64-query-metrics.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows),'models_evaluated':0,'report_sha256':result['report_sha256']}));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');correct(p.parse_args().root)
