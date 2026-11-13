#!/usr/bin/env python3
"""Golden test: C++ features must equal Python features (tolerance 1e-4)."""
import csv
import sys

py_path = sys.argv[1] if len(sys.argv) > 1 else "data/features_py.csv"
cpp_path = sys.argv[2] if len(sys.argv) > 2 else "data/features_cpp.csv"
with open(py_path) as a, open(cpp_path) as b:
    ra, rb = list(csv.DictReader(a)), list(csv.DictReader(b))
if len(ra) != len(rb):
    sys.exit(f"FAIL: window count differs: python={len(ra)} cpp={len(rb)}")
names = [k for k in rb[0].keys() if k != "end_frame"]
worst = (0.0, None, None)
for i, (x, y) in enumerate(zip(ra, rb)):
    if x["end_frame"] != y["end_frame"]:
        sys.exit(f"FAIL: window {i} ends at different frames")
    for n in names:
        d = abs(float(x[n]) - float(y[n]))
        if d > worst[0]:
            worst = (d, i, n)
print(f"{len(ra)} windows x {len(names)} features compared; max |diff| = {worst[0]:.2e}"
      + (f" (window {worst[1]}, {worst[2]})" if worst[1] is not None else ""))
if worst[0] > 1e-4:
    sys.exit("FAIL: train/serve skew detected")
print("PASS: C++ and Python features match")
