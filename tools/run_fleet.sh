#!/bin/bash
# Fleet simulation: N vehicles, each with its own vcan bus, agent and replay,
# all reporting to one edge server. Ctrl+C stops everything.
#   ./tools/run_fleet.sh [N=10] [replay=data/replay.log] [port=9000]
N=${1:-10}; LOG=${2:-data/replay.log}; PORT=${3:-9000}
cd "$(dirname "$0")/.."
mkdir -p reports/fleet

sudo modprobe vcan
for ((v=0; v<N; v++)); do
  sudo ip link add dev vcan$v type vcan 2>/dev/null
  sudo ip link set up vcan$v
done

cleanup() {
  trap '' INT TERM                      # ignore further Ctrl+C while cleaning up
  echo; echo "== stopping fleet"
  pkill -INT -f "build/agent vcan" 2>/dev/null
  pkill canplayer 2>/dev/null
  sleep 1
  pkill -INT -f "build/edge_server -p $PORT" 2>/dev/null
  sleep 1
  pkill -KILL -f "build/agent vcan" 2>/dev/null   # anything still alive
  pkill -KILL -f "build/edge_server -p $PORT" 2>/dev/null
  echo "== done. agent logs: reports/fleet/agentN.log  alerts: reports/alerts.jsonl"
  exit 0
}
trap cleanup INT TERM

./build/edge_server -p $PORT &
sleep 0.5
for ((v=0; v<N; v++)); do
  role=obu; [ $v -ge $((N-2)) ] && role=mec
  ./build/agent vcan$v -i $v -r $role -S 127.0.0.1:$PORT > reports/fleet/agent$v.log 2>&1 &
done
sleep 1
echo "== replaying $LOG on $N vehicle buses. Ctrl+C to stop."
for ((v=0; v<N; v++)); do
  canplayer vcan$v=vcan0 -I "$LOG" &
done
while true; do sleep 1; done          # keep running until Ctrl+C
