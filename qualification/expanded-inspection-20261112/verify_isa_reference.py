"""Compare old native ISA core with new observed core on a supplied fixture."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = "c4e366d"
SOURCE = r'''
#include "isa_fixture.h"
#include <stdio.h>
static void observer(void *ctx,unsigned row,unsigned layer,IsaPort port,const float *data,size_t n){
    (void)row;(void)layer;(void)port;assert(data&&n);(*(size_t*)ctx)++;
}
int main(void){
    IsaFixture f;isa_fixture_init(&f);IsaSession s;IsaArch arch={0};arch.registers[0]=123;
    isa_session_init(&s,&f.model,&arch);size_t calls=0;s.observer=observer;s.observer_context=&calls;
    const unsigned char bytes[]={0x48,0x89,0xd8};
    for(int instruction=0;instruction<3;instruction++){
        assert(!isa_begin(&s,bytes,sizeof(bytes)));
        while(s.state.phase==1){assert(!isa_advance(&s));assert(fwrite(&s.state,1,sizeof(s.state),stdout)==sizeof(s.state));}
        assert(!isa_commit(&s));assert(fwrite(&s.state,1,sizeof(s.state),stdout)==sizeof(s.state));
    }
    fprintf(stderr,"%zu\n",calls);isa_fixture_free(&f);return 0;
}
'''


def run(command):
    return subprocess.run(["rtk", "proxy", *command], cwd=ROOT, check=True, capture_output=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); fixture = tmp/"fixture.c"; fixture.write_text(SOURCE)
        old = run(["git", "show", f"{BASE}:src/isa.c"]).stdout
        old_source = tmp/"isa-base.c"; old_source.write_bytes(old)
        results = []
        for name, source in [("base", str(old_source)), ("observed", "src/isa.c")]:
            binary = tmp/name
            run(["cc", "-O1", "-g", "-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                 "-ffp-contract=off", "-fsanitize=address,undefined", "-Isrc", "-Itests",
                 "src/model.c", source, str(fixture), "-lm", "-o", str(binary)])
            results.append(run([str(binary)]))
        assert results[0].stdout == results[1].stdout, "native state changed"
        calls = int(results[1].stderr.strip()); assert calls == 3*64*3*16
        print(json.dumps({"base": BASE, "base_isa_sha256": hashlib.sha256(old).hexdigest(),
                          "scope": "three supplied instructions, every full bit-row state and architectural commit",
                          "state_byte_identical": True, "observed_port_calls": calls,
                          "bytes_compared": len(results[0].stdout),
                          "state_stream_sha256": hashlib.sha256(results[0].stdout).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
