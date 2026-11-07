"""Source-bound offline evidence sweep for the attention-read rows."""
from pathlib import Path
import argparse,importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("whole_turn_evidence",HERE.parent/"whole_turn/evidence.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("job");a=p.parse_args();module.evidence(a.job,here=HERE)
