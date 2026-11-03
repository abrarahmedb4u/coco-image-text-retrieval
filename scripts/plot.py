"""Draw the figures for one step (or all) from the saved results files. Does not load PyTorch.

Run from the repo root:  python scripts/plot.py 03      (or: python scripts/plot.py all)
Steps 03-07 call this automatically at the end, in a separate process.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plots import PLOTS  # noqa: E402


def main():
    steps = sys.argv[1:] or ["all"]
    if steps == ["all"]:
        steps = list(PLOTS)
    for step in steps:
        PLOTS[step]()


if __name__ == "__main__":
    main()
