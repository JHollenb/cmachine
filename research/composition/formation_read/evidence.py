"""Reuse the scoped composition evidence projection for this result root."""
from pathlib import Path
import argparse,importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("whole_turn_evidence",HERE.parent/"whole_turn/evidence.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("job");args=parser.parse_args();module.evidence(args.job,here=HERE)
