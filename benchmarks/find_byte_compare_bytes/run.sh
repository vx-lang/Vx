#!/bin/bash
# Runs bytes.vx <runs> times and writes runs.csv, setup.txt and one log for each run into
# <output folder>. Before each run it waits until the 1-minute load average is below LOAD_MAX
# (default 2), and it records the load in runs.csv.
#
# Usage: run.sh <vxc> <runs> <output folder>
# Run it from the root of a Vx checkout, after `source config.local`.
set -e
vxc=$1
runs=$2
out=$3
src=$(dirname "$0")/bytes.vx
load_max=${LOAD_MAX:-2}
mkdir -p "$out/logs"

load() {
  if [[ $(uname) == Darwin ]]; then sysctl -n vm.loadavg | awk '{print $2}'; else cut -d' ' -f1 /proc/loadavg; fi
}

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "system: $(uname -srm)"
  if [[ $(uname) == Darwin ]]; then
    echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
    echo "os: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  else
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "c library: $(ldd --version 2>&1 | head -1)"
  fi
  echo "vxc: $($vxc --version 2>&1 | head -1)"
  echo "vx commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "llc: $(llc --version 2>&1 | grep -m1 -i 'llvm version' | sed 's/^ *//')"
  echo "command: $vxc -O2 $src"
} > "$out/setup.txt"

echo "run,load_1min,bytes,round,routine,gb_per_s" > "$out/runs.csv"
for r in $(seq 1 "$runs"); do
  while (( $(echo "$(load) > $load_max" | bc) )); do sleep 5; done
  l=$(load)
  log=$out/logs/run$r.txt
  if ! "$vxc" -O2 "$src" > "$log" 2>&1; then
    echo "run $r: vxc failed, see $log"
    exit 1
  fi
  grep -E '^[0-9]+,' "$log" | sed "s/^/$r,$l,/" >> "$out/runs.csv"
  if grep -q 'wrong result' "$log"; then echo "run $r: a routine gave a wrong result, see $log"; fi
done

# The median, the slowest and the fastest of each routine at each size, over every round of
# every run, and the median of `find_byte` and `compare_bytes` divided by the median of C's.
python3 -I - "$out/runs.csv" <<'PY' | tee "$out/summary.txt"
import csv, statistics, sys
rates = {}
for row in csv.DictReader(open(sys.argv[1])):
    rates.setdefault((int(row['bytes']), row['routine']), []).append(float(row['gb_per_s']))
names = ['find_byte', 'memchr', 'find_loop', 'compare_bytes', 'memcmp', 'compare_loop']
print('GB/s: median (slowest-fastest)')
print('| bytes | ' + ' | '.join(names) + ' | find_byte / memchr | compare_bytes / memcmp |')
print('|---' * (len(names) + 3) + '|')
for n in sorted({k[0] for k in rates}):
    med = {r: statistics.median(rates[(n, r)]) for r in names}
    cells = [f'{med[r]:.1f} ({min(rates[(n, r)]):.1f}-{max(rates[(n, r)]):.1f})' for r in names]
    print(f'| {n} | ' + ' | '.join(cells) +
          f' | {med["find_byte"] / med["memchr"]:.2f} | {med["compare_bytes"] / med["memcmp"]:.2f} |')
PY
