#!/bin/bash
# Kill the given (own, resumable) PIDs if free disk on / drops below $2 MB (default 3300).
PIDS="$1"; MIN_MB=${2:-3300}
while true; do
  alive=0; for p in $PIDS; do kill -0 $p 2>/dev/null && alive=1; done
  [ $alive = 0 ] && exit 0
  free=$(df -m / | awk 'NR==2{print $4}')
  if [ "$free" -lt "$MIN_MB" ]; then
    echo "$(date +%H:%M:%S) GUARD: ${free} MB free < ${MIN_MB} -> SIGTERM $PIDS ($(sysctl -n vm.swapusage))"
    for p in $PIDS; do kill $p 2>/dev/null; done; exit 0
  fi
  sleep 3
done
