"""Usage: python check_data.py [data_dir] [loads...]   e.g.  python check_data.py data 0 1 2 3"""
import os, sys
from data import CLASSES, mat_path
d = sys.argv[1] if len(sys.argv) > 1 else "data"
loads = [int(x) for x in sys.argv[2:]] or [0, 1, 2, 3]
missing = [(n, b + L, L) for n, b, _ in CLASSES for L in loads if not os.path.exists(mat_path(d, b, L))]
total = len(CLASSES) * len(loads)
if not missing:
    print(f"All {total} files found in '{d}'.")
else:
    print(f"Missing {len(missing)} of {total}: " + ", ".join(f"{fn}.mat ({n}, {L}HP)" for n, fn, L in missing))
