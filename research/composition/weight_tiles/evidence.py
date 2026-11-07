"""Standard offline evidence sweep over verified weight-panel observations."""
from pathlib import Path
import argparse,importlib.util
HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('whole_turn_evidence',HERE.parent/'whole_turn/evidence.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');a=p.parse_args();m.evidence(a.job,here=HERE)
