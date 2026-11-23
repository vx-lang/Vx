#!/bin/zsh
# Runs vec_loops.vx with two release builds of vxc, alternating them, and writes runs.csv and
# one log for each run.
#
# Usage: run.sh <vxc of main> <vxc of the PR> <rounds>
# Run it from the root of a Vx checkout, after `source config.local`.
set -e
base=$1
pr=$2
rounds=$3
dir=$(dirname $0)
src=$dir/vec_loops.vx
mkdir -p $dir/logs
echo "round,compiler,loop,microseconds,result,load_1min" > $dir/runs.csv
for r in $(seq 1 $rounds); do
  # Alternate which compiler goes first in each round.
  if (( r % 2 == 1 )); then order=(main pr); else order=(pr main); fi
  for c in $order; do
    if [[ $c == main ]]; then vxc=$base; else vxc=$pr; fi
    # Wait until the 1-minute load average is below 6. The program uses one core, and the M2 Pro
    # has 6 performance cores. A virtual machine kept the load near 5 during these runs.
    while (( $(sysctl -n vm.loadavg | awk '{print $2}') > 6.0 )); do sleep 5; done
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    log=$dir/logs/round${r}_$c.txt
    $vxc -O3 $src > $log 2>&1
    grep -E '^(fill|sum)_' $log | while read -r loop us result; do
      echo "$r,$c,$loop,$us,$result,$load" >> $dir/runs.csv
    done
  done
done
