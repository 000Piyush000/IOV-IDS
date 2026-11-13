#!/usr/bin/env python3
"""
Convert CICIoV2024 CSV files into a candump-format log that `canplayer` can
replay onto vcan0, plus a labels file (ground truth) for later evaluation.

CICIoV2024 CSVs have NO timestamps: each row is one CAN frame
(ID, DATA_0..DATA_7, label, category, specific_class). So we generate the
timing ourselves at a fixed bus rate (default 4000 frames/s, like a real car).

Scenario: benign background traffic, with attack segments inserted.
During an attack segment, attack frames are interleaved with benign frames
(an attacker injects into a live bus; real ECUs keep talking).

Usage:
  python3 tools/csv_to_canlog.py --benign data/decimal_benign.csv \
      --attack data/decimal_DoS.csv --attack data/decimal_spoofing-RPM.csv \
      --benign-chunk 20000 --attack-chunk 5000 --rate 4000 \
      --out data/replay.log --labels data/replay_labels.csv
"""
import argparse
import csv
import sys


def find_cols(header):
    """Locate ID, DATA_0..7 and label columns, tolerating naming differences."""
    norm = [h.strip().lower() for h in header]
    try:
        id_col = norm.index("id")
    except ValueError:
        sys.exit(f"No 'ID' column found. Header was: {header}")
    data_cols = []
    for i in range(8):
        name = f"data_{i}"
        if name not in norm:
            sys.exit(f"No '{name}' column found. Header was: {header}")
        data_cols.append(norm.index(name))
    label_col = None
    for cand in ("specific_class", "category", "label"):
        if cand in norm:
            label_col = norm.index(cand)
            break
    return id_col, data_cols, label_col


def read_frames(path, limit=None):
    """Yield (can_id:int, data:bytes, label:str) from a CICIoV2024 CSV."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        id_col, data_cols, label_col = find_cols(header)
        n = 0
        for row in reader:
            if not row:
                continue
            can_id = int(float(row[id_col]))
            data = bytes(int(float(row[c])) & 0xFF for c in data_cols)
            label = row[label_col].strip() if label_col is not None else "UNKNOWN"
            yield can_id, data, label
            n += 1
            if limit and n >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign", required=True)
    ap.add_argument("--attack", action="append", default=[],
                    help="attack CSV (repeatable)")
    ap.add_argument("--benign-chunk", type=int, default=20000,
                    help="benign frames between attack segments")
    ap.add_argument("--attack-chunk", type=int, default=5000,
                    help="attack frames per attack segment")
    ap.add_argument("--rate", type=float, default=4000.0, help="frames/s")
    ap.add_argument("--iface", default="vcan0")
    ap.add_argument("--start-ts", type=float, default=1700000000.0)
    ap.add_argument("--out", default="data/replay.log")
    ap.add_argument("--labels", default="data/replay_labels.csv")
    args = ap.parse_args()

    benign = read_frames(args.benign)
    dt = 1.0 / args.rate
    ts = args.start_ts
    count = {}

    with open(args.out, "w") as log, open(args.labels, "w", newline="") as lab:
        lw = csv.writer(lab)
        lw.writerow(["frame_idx", "timestamp", "can_id", "label"])
        idx = 0

        def emit(frame):
            nonlocal ts, idx
            can_id, data, label = frame
            log.write(f"({ts:.6f}) {args.iface} {can_id:03X}#{data.hex().upper()}\n")
            lw.writerow([idx, f"{ts:.6f}", f"{can_id:03X}", label])
            count[label] = count.get(label, 0) + 1
            ts += dt
            idx += 1

        def benign_frames(n):
            for _ in range(n):
                fr = next(benign, None)
                if fr is None:
                    return False
                emit(fr)
            return True

        # warm-up benign segment
        benign_frames(args.benign_chunk)
        for path in args.attack:
            # attack segment: interleave 1 attack frame with 1 benign frame
            for afr in read_frames(path, limit=args.attack_chunk):
                emit(afr)
                bfr = next(benign, None)
                if bfr is not None:
                    emit(bfr)
            benign_frames(args.benign_chunk)  # recovery / normal traffic

    total = sum(count.values())
    print(f"Wrote {total} frames ({total / args.rate:.1f} s at {args.rate:.0f} fps)")
    for k, v in sorted(count.items()):
        print(f"  {k:20s} {v}")


if __name__ == "__main__":
    main()
