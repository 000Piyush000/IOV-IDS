#!/usr/bin/env python3
"""
Step 4a: train the window classifier and export it to ONNX.

  python3 tools/train_detector.py --features data/features_py.csv --out models/ids.onnx

1. Leakage-safe BLOCK-WISE split. Windows overlap by 50%, so neighbouring
   windows share frames. A random split would put near-duplicates in train
   AND test (inflated accuracy). We cut the timeline into blocks of
   consecutive windows, assign whole blocks to train/test, and drop one
   window on each side of every block boundary (purge gap).
2. Standardize with TRAIN statistics only; oversample rare attack classes.
3. Train a compact MLP (39 -> 64 -> 32 -> n_classes).
4. Export ONNX with normalization + softmax INSIDE the graph, so the C++
   agent feeds raw features and gets probabilities (no skew possible).
5. Check ONNX Runtime output == scikit-learn output.
"""
import argparse
import csv
import os

import numpy as np
import onnx
try:
    import onnxruntime as ort   # optional: only for the parity check
except ImportError:
    ort = None
from onnx import TensorProto, helper, numpy_helper
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    names = [k for k in rows[0].keys() if k not in ("end_frame", "label")]
    X = np.array([[float(r[n]) for n in names] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows])
    return X, y, names


def block_split(n, block, test_frac, seed):
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    test_blocks = set(rng.choice(n_blocks, size=max(1, int(n_blocks * test_frac)), replace=False))
    block_of = np.arange(n) // block
    is_test = np.array([b in test_blocks for b in block_of])
    # purge: drop windows adjacent to a train/test boundary (they share frames)
    keep = np.ones(n, dtype=bool)
    for i in range(1, n):
        if is_test[i] != is_test[i - 1]:
            keep[i - 1] = keep[i] = False
    tr = np.where(keep & ~is_test)[0]
    te = np.where(keep & is_test)[0]
    return tr, te


def oversample(X, y, seed):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    idx = []
    for c, k in zip(classes, counts):
        ci = np.where(y == c)[0]
        idx.append(ci)
        if k < target:
            idx.append(rng.choice(ci, size=target - k, replace=True))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def export_onnx(clf, mean, std, n_features, path):
    """Build the ONNX graph by hand from the trained weights:
    probs = softmax( relu(relu(((x-mean)/std) W1 + b1) W2 + b2) W3 + b3 )"""
    inits = [numpy_helper.from_array(mean.astype(np.float32), "mean"),
             numpy_helper.from_array(std.astype(np.float32), "std")]
    nodes = [helper.make_node("Sub", ["features", "mean"], ["centered"]),
             helper.make_node("Div", ["centered", "std"], ["h0"])]
    prev = "h0"
    n_layers = len(clf.coefs_)
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        inits += [numpy_helper.from_array(W.astype(np.float32), f"W{i}"),
                  numpy_helper.from_array(b.astype(np.float32), f"b{i}")]
        nodes += [helper.make_node("MatMul", [prev, f"W{i}"], [f"mm{i}"]),
                  helper.make_node("Add", [f"mm{i}", f"b{i}"], [f"z{i}"])]
        if i < n_layers - 1:
            nodes.append(helper.make_node("Relu", [f"z{i}"], [f"h{i + 1}"]))
            prev = f"h{i + 1}"
        else:
            nodes.append(helper.make_node("Softmax", [f"z{i}"], ["probs"], axis=1))
    n_out = clf.coefs_[-1].shape[1]
    graph = helper.make_graph(
        nodes, "iov_ids_mlp",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [None, n_out])],
        inits)
    # opset 13 / IR 8: loads on any ONNX Runtime >= 1.10
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_py.csv")
    ap.add_argument("--out", default="models/ids.onnx")
    ap.add_argument("--block", type=int, default=40, help="windows per split block")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", action="append", default=[],
                    help="attack class to hold out of TRAINING (simulates a zero-day)")
    args = ap.parse_args()

    X, y, names = load(args.features)
    tr, te = block_split(len(y), args.block, args.test_frac, args.seed)
    if args.exclude:
        tr = np.array([i for i in tr if y[i] not in args.exclude])
        te = np.array([i for i in te if y[i] not in args.exclude])
        print(f"ZERO-DAY SIMULATION: {args.exclude} removed from training (model has never seen it)")
    print(f"windows={len(y)}  train={len(tr)}  test={len(te)}  (purged {len(y) - len(tr) - len(te)})")

    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    std[std < 1e-9] = 1.0                       # constant features: avoid divide-by-zero
    Xtr, ytr = oversample((X[tr] - mean) / std, y[tr], args.seed)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                        max_iter=300, random_state=args.seed)
    clf.fit(Xtr, ytr)

    pred = clf.predict((X[te] - mean) / std)
    print("\n=== Test set (held-out blocks) ===")
    print(classification_report(y[te], pred, digits=4, zero_division=0))
    print("macro-F1:", round(f1_score(y[te], pred, average="macro"), 4))
    print("Confusion matrix (rows=true, cols=pred):", [str(c) for c in clf.classes_])
    print(confusion_matrix(y[te], pred, labels=clf.classes_))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    export_onnx(clf, mean, std, X.shape[1], args.out)
    with open(os.path.join(os.path.dirname(args.out) or ".", "classes.txt"), "w") as fh:
        fh.write("\n".join(clf.classes_) + "\n")

    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nONNX saved: {args.out} ({size_kb:.1f} KB), classes: {[str(c) for c in clf.classes_]}")
    if ort is None:
        print("(onnxruntime not installed in Python: skipped parity check; C++ bench still works)")
        return
    # parity: ONNX Runtime must reproduce scikit-learn's probabilities
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    p_onnx = sess.run(["probs"], {"features": X[te].astype(np.float32)})[0]
    p_sk = clf.predict_proba((X[te] - mean) / std)
    diff = np.abs(p_onnx - p_sk).max()
    agree = (p_onnx.argmax(1) == p_sk.argmax(1)).mean()
    print(f"ONNX vs sklearn: max |prob diff| = {diff:.2e}, argmax agreement = {agree * 100:.2f}%")


if __name__ == "__main__":
    main()
