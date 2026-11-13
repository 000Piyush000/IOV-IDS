# Real-Time Federated Intrusion Detection for the Internet of Vehicles

A C++17 system that detects attacks on in-vehicle **CAN bus** traffic in real time,
learns collaboratively across vehicles with **federated learning** (no raw data
shared), flags **zero-day** attacks, and produces **LLM incident reports whose every
claim is validated** against detector evidence.

Dataset: **CICIoV2024** (benign + DoS, GAS, RPM, SPEED, STEERING_WHEEL spoofing).

## Architecture

```
 CICIoV2024 CSV ─► csv_to_canlog.py ─► canplayer ─► vcan0..vcanN (Linux SocketCAN)
                                                        │
                       ┌────────────────────────────────▼───────────────┐
 VEHICLE AGENT (C++)   │ SocketCAN raw socket → ring-buffer window (100) │
 one per vehicle       │ → 39 features → ONNX Runtime classifier          │
                       │ → zero-day scorer (entropy / Hamming z-score)    │
                       └────────────────────────────────┬───────────────┘
                                                        │ TCP, length-prefixed binary protocol
                       ┌────────────────────────────────▼───────────────┐
 EDGE SERVER (C++)     │ single-thread epoll event loop                   │
                       │ alert aggregation → alerts.jsonl (evidence)      │
                       │ FedAvg coordinator for federated learning        │
                       └────────────────────────────────┬───────────────┘
                                                        │
 LLM REPORTER (Python) │ evidence → Llama 3 (local, Ollama) → JSON draft   │
                       │ → validator checks every claim → advisory report  │
                       │   (0 automatic actions)                           │
```

## Components

| Path | What it does |
|---|---|
| `common/frame_source.hpp` | `IFrameSource` interface: SocketCAN or log file (testable without hardware) |
| `agent/socketcan_source.cpp` | Raw CAN socket: `socket(PF_CAN)`, `ioctl(SIOCGIFINDEX)`, `bind`, `read` |
| `common/sliding_window.hpp` | O(1) ring buffer, window 100 frames, stride 50 |
| `common/features.cpp` / `tools/features.py` | 39 features in C++ and Python, golden-tested identical |
| `tools/train_detector.py` | Leakage-safe block split, MLP, ONNX export with normalization inside the graph |
| `common/detector.cpp` | ONNX Runtime wrapper (1 thread, no allocation in hot path, PIMPL) |
| `common/protocol.cpp` | Binary framing `[len][type][payload]`, big-endian, partial-read safe |
| `server/edge_server.cpp` | epoll server: N vehicles in one thread; FedAvg coordinator |
| `fl/fl_client.cpp` | Federated client: local SGD on own non-IID data, sends only 40 parameters |
| `common/zero_day.hpp` | Welford baseline + k-sigma anomaly score, label-free |
| `tools/llm_report.py` | Evidence → LLM → validator → advisory Markdown report |
| `tests/` | GoogleTest (C++) + unittest (Python); CI runs both, plus an ASan/UBSan build |

## Build & run

```bash
./tools/get_onnxruntime.sh                     # one-time
cmake -S . -B build && cmake --build build -j
ctest --test-dir build                         # C++ unit tests
python3 -m unittest discover -s tests -p 'test_*.py'

python3 tools/csv_to_canlog.py --benign data/decimal_benign.csv --attack data/decimal_DoS.csv ...
python3 tools/features.py && python3 tools/train_detector.py
./build/feature_dump data/replay.log data/features_cpp.csv && python3 tools/compare_features.py
./build/bench_infer data/replay.log models/ids.onnx models/classes.txt data/features_py.csv
./tools/run_fleet.sh 5                         # live fleet on vcan
python3 tools/fl_partition.py && ./tools/run_fl.sh   # federated learning
python3 tools/llm_report.py                    # LLM incident report (needs Ollama)
```

## Results (CICIoV2024, Ubuntu VM, 4 vCPU)

| Metric | Value |
|---|---|
| Feature extraction + inference per window | p50 ≈ 19 µs, p99 ≈ 55–77 µs |
| ONNX inference alone | p50 ≈ 2.8 µs |
| Offline throughput (one core) | ≈ 1.1 M frames/s |
| Live fleet | 5 vehicles × ~4,000 frames/s ≈ 20 K frames/s |
| Classifier (held-out blocks) | macro-F1 = 1.0 |
| Zero-day (SPEED excluded from training) | 100 % flagged, 0 % false positives on benign |
| C++ vs Python features | identical on every window (max diff 0) |

CICIoV2024 attacks are injected at high rate with fixed or extreme payloads, so they are
highly separable. Stealthy low-rate attacks are the open problem.

## Design decisions

- **Train/serve skew prevented** by a golden test (C++ vs Python features) and by
  baking normalization + softmax into the ONNX graph.
- **No leakage**: windows overlap, so the split is block-wise with a purge gap.
- **Detection never depends on the network**: the agent keeps detecting if the server
  is down and reconnects with backoff.
- **LLM is advisory only**: it drafts from aggregated evidence, every claim is validated,
  and failed claims are flagged, never hidden. Nothing is executed automatically.
