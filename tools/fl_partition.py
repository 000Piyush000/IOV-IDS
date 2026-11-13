#!/usr/bin/env python3
"""
Step 6a: split the window dataset into NON-IID client datasets for
federated learning, plus a shared held-out test set.

  python3 tools/fl_partition.py --features data/features_py.csv --clients 10 --mec 2

Non-IID on purpose (that's the realistic, hard case for federated learning):
  - each OBU (vehicle) sees benign traffic + only ONE attack type
  - each MEC (edge node) sees benign traffic + a slice of EVERY attack type
So no single vehicle can learn all attacks alone -- they need each other.

Labels are binary for the federated head: 1 = attack, 0 = benign.
Standardization stats come from the training split only (no test leakage).
Output: data/fl/client<i>.csv, data/fl/test.csv  (columns: class,y,f0..f38)
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_detector import block_split, load   # same leakage-safe split


def write(path, X, cls):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "y"] + [f"f{i}" for i in range(X.shape[1])])
        for xi, c in zip(X, cls):
            w.writerow([c, 0 if c == "BENIGN" else 1] + [f"{v:.6f}" for v in xi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_py.csv")
    ap.add_argument("--out-dir", default="data/fl")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--mec", type=int, default=2, help="last N clients are MEC nodes")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    X, y, _ = load(args.features)
    tr, te = block_split(len(y), 40, 0.2, 42)
    mean, std = X[tr].mean(0), X[tr].std(0)
    std[std < 1e-9] = 1.0
    Xs = (X - mean) / std
    rng = np.random.default_rng(args.seed)

    attacks = sorted(str(c) for c in set(y[tr]) if c != "BENIGN")
    n_obu = args.clients - args.mec
    holders = {a: [] for a in attacks}            # attack class -> clients that see it
    for i in range(n_obu):
        holders[attacks[i % len(attacks)]].append(i)
    for i in range(n_obu, args.clients):
        for a in attacks:
            holders[a].append(i)

    parts = [[] for _ in range(args.clients)]
    benign = rng.permutation(tr[y[tr] == "BENIGN"])
    for i, chunk in enumerate(np.array_split(benign, args.clients)):
        parts[i].extend(chunk)
    for a in attacks:
        idx = rng.permutation(tr[y[tr] == a])
        for cid, chunk in zip(holders[a], np.array_split(idx, len(holders[a]))):
            parts[cid].extend(chunk)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"attack classes: {attacks}")
    for i, p in enumerate(parts):
        p = np.array(sorted(p))
        role = "MEC" if i >= n_obu else "OBU"
        seen = sorted(str(c) for c in set(y[p]) - {"BENIGN"})
        write(f"{args.out_dir}/client{i}.csv", Xs[p], y[p])
        print(f"client{i} ({role}): {len(p):5d} windows, attacks seen: {seen}")
    write(f"{args.out_dir}/test.csv", Xs[te], y[te])
    print(f"test: {len(te)} windows (all classes) -> {args.out_dir}/test.csv")


if __name__ == "__main__":
    main()
