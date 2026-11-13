#!/bin/bash
# Federated learning demo: edge server as FedAvg coordinator + N vehicle clients,
# each training only on its own non-IID data.
#   ./tools/run_fl.sh [N=10] [rounds=20] [port=9001]
N=${1:-10}; ROUNDS=${2:-20}; PORT=${3:-9001}
cd "$(dirname "$0")/.."
mkdir -p reports/fl
./build/edge_server -p $PORT -o reports/fl/alerts.jsonl \
    --fl-clients $N --fl-rounds $ROUNDS --fl-test data/fl/test.csv &
SERVER=$!
sleep 0.5
for ((v=0; v<N; v++)); do
  role=obu; [ $v -ge $((N-2)) ] && role=mec
  ./build/fl_client -i $v -r $role -S 127.0.0.1:$PORT \
      --train data/fl/client$v.csv --test data/fl/test.csv > reports/fl/client$v.log 2>&1 &
done
wait $(jobs -p | grep -v "^$SERVER$")    # wait for all clients to finish
sleep 0.5
kill -INT $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
echo; echo "== client 0 (OBU, saw only one attack type):"; tail -8 reports/fl/client0.log
