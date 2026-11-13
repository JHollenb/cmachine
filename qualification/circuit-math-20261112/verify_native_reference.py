"""Compare complete tiny native trajectories with the original committed core.

Requires the base Git object; no model, NumPy, CUDA or external service.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = "8822497"
FIXTURE = r'''
#define main lifecycle_main
#include "state_mechanics.c"
#undef main
int main(void){
    QwenModel m;fixture(&m);QwenSession s;assert(!qwen_session_init(&s,&m,16));
    for(int token=2;token<10;token++){
        assert(!qwen_push(&s,token));
        assert(fwrite(&s.position,sizeof(int),1,stdout)==1);
        assert(fwrite(&s.pending,sizeof(int),1,stdout)==1);
        assert(fwrite(s.logits,4,m.vocab,stdout)==(size_t)m.vocab);
        for(int layer=0;layer<m.layers;layer++)for(int head=0;head<m.kvheads;head++){
            size_t off=((size_t)layer*m.kvheads+head)*s.capacity*m.head;
            size_t count=(size_t)s.position*m.head;
            assert(fwrite(s.keys+off,4,count,stdout)==count);
            assert(fwrite(s.values+off,4,count,stdout)==count);
        }
    }
    qwen_session_free(&s);fixture_free(&m);return 0;
}
'''


def run(command):
    return subprocess.run(["rtk", "proxy", *command], cwd=ROOT, check=True, capture_output=True).stdout


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / "fixture.c"
        source.write_text(FIXTURE)
        paths = []
        base_hashes = {}
        for name in ("src/qwen.c", "tests/gpu_stub.c"):
            data = run(["git", "show", f"{BASE}:{name}"])
            path = tmp / Path(name).name
            path.write_bytes(data)
            paths.append(str(path))
            base_hashes[name] = hashlib.sha256(data).hexdigest()
        outputs = []
        for lane, sources in (("base", paths), ("current", ["src/qwen.c", "tests/gpu_stub.c"])):
            binary = tmp / lane
            run(["cc", "-O1", "-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                 "-ffp-contract=off", "-fsanitize=address,undefined", "-Isrc", "-Itests",
                 "src/model.c", *sources, str(source), "-lm", "-o", str(binary)])
            outputs.append(run([str(binary)]))
        assert outputs[0] == outputs[1], "native trajectory changed"
        print(json.dumps({"base": BASE, "base_source_sha256": base_hashes,
                          "observations": "eight native steps: full logits, valid all-layer KV, position and pending token",
                          "bytes": len(outputs[0]), "byte_identical": True,
                          "sha256": hashlib.sha256(outputs[0]).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
