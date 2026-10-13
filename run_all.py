"""
run_all.py - run the whole pipeline after the data has been downloaded (ingest.py universe + download).

    python run_all.py            # every step
    python run_all.py --from 4   # resume from step 4

Stops at the first failing step. Each step can also be run on its own.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("pandas reference (ground truth)", ["pandas_reference.py", "BTCUSDT", "TNSRUSDT", "VIRTUALUSDT"]),
    ("CSV -> Parquet", ["ingest.py", "parquet"]),
    ("CSV vs Parquet benchmark", ["benchmark.py"]),
    ("Spark metrics + validation", ["metrics.py"]),
    ("signal decay + quintiles", ["analysis.py"]),
    ("skew diagnosis and fix", ["skew_scaling.py", "skew"]),
    ("scaling curve", ["skew_scaling.py", "scale"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", type=int, default=1, help="step number to start from")
    start = parser.parse_args().start

    t_all = time.time()
    for i, (name, args) in enumerate(STEPS, 1):
        if i < start:
            continue
        print(f"\n{'=' * 70}\nSTEP {i}/{len(STEPS)}: {name}\n{'=' * 70}", flush=True)
        t0 = time.time()
        code = subprocess.run([sys.executable, str(ROOT / args[0]), *args[1:]], cwd=ROOT).returncode
        if code != 0:
            sys.exit(f"\nSTEP {i} ({name}) failed with exit code {code}. Fix it, then: python run_all.py --from {i}")
        print(f"--- step {i} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nALL STEPS DONE in {(time.time() - t_all) / 60:.1f} min. Results in results/, charts in results/figures/")


if __name__ == "__main__":
    main()
