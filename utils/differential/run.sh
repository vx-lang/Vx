#!/usr/bin/env bash
#===- run.sh - the same mistake, in Vx and in CUDA -----------------------===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Runs each pair under pairs/ and records, for both toolchains: does it compile,
# does it run, is the answer right, and the failure text verbatim.
#
# Usage:
#   ./run.sh                # every pair
#   ./run.sh 01 03          # pairs whose directory name starts with these
#   ./run.sh --list         # what is here, no compiling
#
# The Vx half runs anywhere. The CUDA half needs a GPU and nvcc; without them the
# CUDA columns are recorded as "skipped" rather than silently passing.
#
#===----------------------------------------------------------------------===#

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VXC="${VXC:-$REPO/target/debug/vxc}"
NVCC="${NVCC:-nvcc}"
# Compile for the part the results are about. Without this nvcc picks a default
# old enough to warn about its own deprecation, and a table of run-time behaviour
# built from code compiled for another architecture invites the obvious question.
NVCC_ARCH="${NVCC_ARCH:-sm_80}"

if [[ "${1:-}" == "--list" ]]; then
  for d in "$HERE"/pairs/*/; do
    ( . "$d/pair.env"; printf '%-34s %-8s %s\n' "$(basename "$d")" "$E_CODE" "$TITLE" )
  done
  exit 0
fi

# The Vx half is a compile-time refusal and says the same thing on any machine, so
# there is no reason to carry a compiler to the GPU box to re-derive it. `vxc` is
# also built for the host that built it -- a macOS arm64 binary will not run on a
# Linux pod -- so on the pod this is the mode to use.
CUDA_ONLY=no
if [[ "${1:-}" == "--cuda-only" ]]; then
  CUDA_ONLY=yes
  shift
fi

# Fail here rather than halfway through. A missing `vxc` exits 127, and 127 is
# non-zero, so a runner that reads "non-zero means refused" would record a clean
# refusal for every pair and never touch a compiler.
if [[ "$CUDA_ONLY" == no ]]; then
  if [[ ! -x "$VXC" ]]; then
    echo "no vxc at $VXC" >&2
    echo "build it, set VXC=..., or run with --cuda-only to do the GPU half alone." >&2
    exit 2
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is needed to escape the evidence into results.json" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HERE/results/$STAMP"
mkdir -p "$OUT"

# What card is present, so a pair that names one can refuse to run on another.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[[ -z "$GPU_NAME" ]] && GPU_NAME="none"
HAVE_NVCC=no
command -v "$NVCC" >/dev/null 2>&1 && HAVE_NVCC=yes

{
  echo "utc: $STAMP"
  echo "gpu: $GPU_NAME"
  echo "nvcc: $HAVE_NVCC"
  echo "nvcc_arch: $NVCC_ARCH"
  command -v "$NVCC" >/dev/null 2>&1 && "$NVCC" --version | tail -2
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/driver: /'
} > "$OUT/environment.txt"

echo "results -> $OUT"
echo "gpu: $GPU_NAME   nvcc: $HAVE_NVCC"
echo

rows=()

for d in "$HERE"/pairs/*/; do
  name="$(basename "$d")"

  # A bare argument list selects pairs by name prefix.
  if [[ $# -gt 0 ]]; then
    match=no
    for want in "$@"; do
      [[ "$name" == "$want"* ]] && match=yes
    done
    [[ "$match" == no ]] && continue
  fi

  TITLE=""; VX_SOURCE=""; VX_ARGS=""; E_CODE=""; EXPECT_CUDA=""; EXPECT_GPU=""
  # Most pairs are "Vx refuses, CUDA does not". Pair 06 is the other way round, so
  # which outcome counts as correct is per-pair rather than assumed.
  EXPECT_VX="refused"
  # shellcheck disable=SC1091
  . "$d/pair.env"

  cell="$OUT/$name"
  mkdir -p "$cell"
  echo "== $name — $TITLE"

  # ---- the Vx half -------------------------------------------------------
  vx_verdict="not-run-here"
  vx_evidence=""
  if [[ "$CUDA_ONLY" == yes ]]; then
    echo "   vx:   not run (--cuda-only); the refusal is machine-independent and recorded elsewhere"
  else
    # Expected to REFUSE. A zero exit here is the interesting failure: it means the
    # program the pair is built around is no longer rejected.
    # shellcheck disable=SC2086
    "$VXC" $VX_ARGS "$REPO/$VX_SOURCE" > "$cell/vx.stdout" 2> "$cell/vx.stderr"
    vx_rc=$?
    # `Error[E6003] at 8:13: ...` and a bare `Error: ...` are both real shapes, so
    # match the line rather than the code -- an uncoded diagnostic is exactly what
    # pair 02 is about, and a pattern requiring a code would drop it.
    vx_evidence="$(grep -hE '^Error' "$cell/vx.stdout" "$cell/vx.stderr" 2>/dev/null | head -1)"
    if [[ $vx_rc -eq 0 ]]; then
      # Correct for a pair that expects admission, alarming for one that does not.
      if [[ "$EXPECT_VX" == "accepted" ]]; then vx_verdict="accepted"; else vx_verdict="ACCEPTED"; fi
    elif [[ -z "$vx_evidence" ]]; then
      # Non-zero with nothing that looks like a diagnostic. A crash, a missing
      # machine file, a compiler that is not there. Not a refusal, whatever the
      # exit code says.
      vx_verdict="NO-DIAGNOSTIC(rc=$vx_rc)"
    else
      vx_verdict="refused"
    fi
    echo "   vx:   $vx_verdict  ${vx_evidence:-<no Error line>}"
  fi

  # ---- the CUDA half -----------------------------------------------------
  cuda_compiles="skipped"; cuda_runs="skipped"; cuda_evidence=""; cuda_rc=""

  if [[ "$HAVE_NVCC" == no ]]; then
    cuda_evidence="nvcc not present"
  elif [[ -n "$EXPECT_GPU" && "$GPU_NAME" != *"$EXPECT_GPU"* ]]; then
    # Refuse rather than produce a row that looks fine: this pair's bound comes
    # from a specific machine model and means nothing against another card.
    cuda_evidence="pair expects $EXPECT_GPU, found $GPU_NAME"
    echo "   cuda: SKIPPED — $cuda_evidence"
  else
    "$NVCC" -O2 -arch="$NVCC_ARCH" -o "$cell/a.out" "$d/cuda.cu" > "$cell/nvcc.stdout" 2> "$cell/nvcc.stderr"
    nvcc_rc=$?
    if [[ $nvcc_rc -ne 0 ]]; then
      cuda_compiles="no"
      cuda_runs="n/a"
      cuda_evidence="$(head -c 2000 "$cell/nvcc.stderr")"
    else
      cuda_compiles="yes"
      "$cell/a.out" > "$cell/run.stdout" 2> "$cell/run.stderr"
      cuda_rc=$?
      # 128+N is a signal; 139 is SIGSEGV, which is pair 02's expected end.
      if [[ $cuda_rc -eq 0 ]]; then
        cuda_runs="yes"
      elif [[ $cuda_rc -gt 128 ]]; then
        cuda_runs="signal $((cuda_rc - 128))"
      else
        cuda_runs="exit $cuda_rc"
      fi
      cuda_evidence="$(cat "$cell/run.stdout" "$cell/run.stderr" 2>/dev/null | head -c 2000)"
    fi
    echo "   cuda: compiles=$cuda_compiles runs=$cuda_runs"
  fi

  # ---- did the pair behave as the ticket claims? -------------------------
  agrees="unknown"
  if [[ "$cuda_compiles" != "skipped" ]]; then
    case "$EXPECT_CUDA" in
      runtime-failure)
        if [[ "$cuda_compiles" == "yes" && "$cuda_runs" != "yes" ]]; then agrees="yes"; else agrees="NO"; fi ;;
      compile-error)
        if [[ "$cuda_compiles" == "no" ]]; then agrees="yes"; else agrees="NO"; fi ;;
      runtime-success)
        if [[ "$cuda_compiles" == "yes" && "$cuda_runs" == "yes" ]]; then agrees="yes"; else agrees="NO"; fi ;;
    esac
    echo "   pair: expected=$EXPECT_CUDA observed_as_expected=$agrees"
  fi
  echo

  rows+=("$(printf '{"pair":"%s","title":"%s","e_code":"%s","vx_verdict":"%s","vx_evidence":%s,"cuda_compiles":"%s","cuda_runs":"%s","cuda_expected":"%s","as_expected":"%s","cuda_evidence":%s}' \
    "$name" "$TITLE" "$E_CODE" "$vx_verdict" \
    "$(printf '%s' "$vx_evidence" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
    "$cuda_compiles" "$cuda_runs" "$EXPECT_CUDA" "$agrees" \
    "$(printf '%s' "$cuda_evidence" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")")
done

{
  printf '{\n  "utc": "%s",\n  "gpu": "%s",\n  "pairs": [\n' "$STAMP" "$GPU_NAME"
  for i in "${!rows[@]}"; do
    printf '    %s' "${rows[$i]}"
    [[ $i -lt $(( ${#rows[@]} - 1 )) ]] && printf ','
    printf '\n'
  done
  printf '  ]\n}\n'
} > "$OUT/results.json"

echo "wrote $OUT/results.json"
