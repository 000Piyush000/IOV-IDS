#!/usr/bin/env python3
"""
Python twin of common/features.cpp. Builds the TRAINING dataset:
replay.log + replay_labels.csv -> features_py.csv (39 features + window label).

Window label: if >= 10% of the frames in the window are attack frames,
the window gets that attack's class; otherwise BENIGN.

  python3 tools/features.py --log data/replay.log --labels data/replay_labels.csv \
      --out data/features_py.csv
"""
import argparse
import csv
import math
from collections import Counter, deque

NAMES = (["n_unique_ids", "id_entropy", "top_id_ratio", "frac_id_zero", "mean_id",
          "std_id", "min_id", "max_id", "id_change_rate", "mean_dlc"]
         + [f"mean_b{i}" for i in range(8)] + [f"std_b{i}" for i in range(8)]
         + [f"chg_b{i}" for i in range(8)]
         + ["payload_entropy", "frac_zero_bytes", "frac_ff_bytes", "mean_hamming",
            "max_repeat_run"])
assert len(NAMES) == 39
MAX_ID = 2047.0


def entropy(counter, total):
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def pstd(s, sq, n):
    if n == 0:
        return 0.0
    m = s / n
    return math.sqrt(max(0.0, sq / n - m * m))


def extract(w):
    """w: list of (can_id:int, data:bytes) oldest -> newest."""
    n = len(w)
    ids = [fr[0] for fr in w]
    id_cnt = Counter(ids)
    xs = [i / MAX_ID for i in ids]
    s, sq = sum(xs), sum(x * x for x in xs)
    changes, run, max_run = 0, 1, 1
    for i in range(1, n):
        if ids[i] != ids[i - 1]:
            changes += 1
            run = 1
        else:
            run += 1
            max_run = max(max_run, run)
    f = [float(len(id_cnt)), entropy(id_cnt, n), max(id_cnt.values()) / n,
         id_cnt.get(0, 0) / n, s / n, pstd(s, sq, n), min(ids) / MAX_ID,
         max(ids) / MAX_ID, changes / (n - 1) if n > 1 else 0.0,
         sum(len(fr[1]) for fr in w) / n / 8.0]

    means, stds = [], []
    for b in range(8):
        vals = [fr[1][b] / 255.0 for fr in w if len(fr[1]) > b]
        c = len(vals)
        bs, bsq = sum(vals), sum(v * v for v in vals)
        means.append(bs / c if c else 0.0)
        stds.append(pstd(bs, bsq, c))
    f += means + stds

    last, comparisons, bit_diffs = {}, 0, 0
    byte_changes = [0] * 8
    for cid, data in w:
        if cid in last:
            comparisons += 1
            prev = last[cid]
            for b in range(8):
                x = (data[b] if b < len(data) else 0) ^ (prev[b] if b < len(prev) else 0)
                if x:
                    byte_changes[b] += 1
                bit_diffs += bin(x).count("1")
        last[cid] = data
    f += [bc / comparisons if comparisons else 0.0 for bc in byte_changes]

    allb = [b for _, d in w for b in d[:8]]
    nb = len(allb)
    f += [entropy(Counter(allb), nb),
          allb.count(0) / nb if nb else 0.0,
          allb.count(0xFF) / nb if nb else 0.0,
          bit_diffs / (64.0 * comparisons) if comparisons else 0.0,
          max_run / n]
    return f


def read_log(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if "#" not in line:
                continue
            frame = line.split()[-1]
            cid, hexdata = frame.split("#")
            yield int(cid, 16), bytes.fromhex(hexdata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/replay.log")
    ap.add_argument("--labels", default="data/replay_labels.csv")
    ap.add_argument("--out", default="data/features_py.csv")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--attack-frac", type=float, default=0.10)
    args = ap.parse_args()

    with open(args.labels) as fh:
        labels = [row["label"] for row in csv.DictReader(fh)]

    win = deque(maxlen=args.window)
    lab_win = deque(maxlen=args.window)
    since = 0
    counts = Counter()
    with open(args.out, "w", newline="") as fo:
        wr = csv.writer(fo)
        wr.writerow(["end_frame"] + NAMES + ["label"])
        for idx, fr in enumerate(read_log(args.log)):
            win.append(fr)
            lab_win.append(labels[idx])
            since += 1
            if len(win) == args.window and since >= args.stride:
                since = 0
                attacks = Counter(l for l in lab_win if l != "BENIGN")
                label = "BENIGN"
                if attacks:
                    cls, c = attacks.most_common(1)[0]
                    if c >= args.attack_frac * args.window:
                        label = cls
                counts[label] += 1
                wr.writerow([idx] + [f"{v:.6f}" for v in extract(list(win))] + [label])
    print(f"windows={sum(counts.values())} -> {args.out}")
    for k, v in sorted(counts.items()):
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
