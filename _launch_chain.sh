#!/bin/bash
# 1) wait for cache regeneration (our code, inherited caches deleted 2026-08-11)
# 2) smoke every new scene (2 rounds x 20 iters, distortion + mixed_area)
# 3) gate on full-test-set n_views, then launch the real sweeps
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
for p in 381493 381494 381495; do
  until ! ps -p "$p" > /dev/null 2>&1; do sleep 15; done
done
echo "=== regen finished $(date) ==="
grep -h "FAIL" log/regen_*.log 2>/dev/null && { echo "REGEN HAD FAILURES -- aborting"; exit 1; }
for s in lego ficus mic; do
  echo -n "$s cues: "; ls -d /mnt/data1/samk/NEU/sorted_dataset/$s/policy/mesh_milo/*/*/ 2>/dev/null | xargs -n1 basename | sort -u | tr '\n' ' '; echo
done
echo -n "drjohnson precapture: "
for d in mesh_texture mesh_depth test_mesh_texture test_mesh_depth; do
  echo -n "$d=$(ls /mnt/data1/samk/NEU/sorted_dataset/milo_meshes/drjohnson-dw50/$d 2>/dev/null | wc -l) "
done; echo

export POLICIES_OVERRIDE="distortion mixed_area"
for s in lego ficus mic drjohnson; do
  echo "=== SMOKE $s $(date +%H:%M:%S) ==="
  bash _smoke_rd_sweep.sh "$s" 0
done
echo "=== SMOKE PHASE DONE $(date) ==="

GATE_OK=1
for s in lego ficus mic drjohnson; do
  for pol in distortion mixed_area; do
    f="output/_smoke_rd/$s/${pol}_2000_occlusion/results_lmg.json"
    n=$(conda run -n lmg python -c "
import json
try:
    d=json.load(open('$f')); k=sorted(d)[-1]; print(d[k]['n_views'])
except Exception: print(0)
" 2>/dev/null | tail -1)
    echo "gate: $s/$pol n_views=${n:-0}"
    [ "${n:-0}" -gt 1 ] || { echo "GATE FAIL: $s/$pol"; GATE_OK=0; }
  done
done
[ "$GATE_OK" = 1 ] || { echo "SMOKE GATE FAILED -- nothing launched"; exit 1; }

echo "=== GATE PASSED, LAUNCHING $(date) ==="
TS=$(date +%Y%m%d_%H%M)
nohup bash -c "bash exp_rd_sweep.sh lego 0 && bash exp_schedule_ablation.sh curves_hotdog_ship 0"  > "log/chain_gpu0_${TS}.log" 2>&1 &
echo "GPU0_CHAIN=$! (lego -> curves_hotdog_ship)"
nohup bash -c "bash exp_rd_sweep.sh ficus 1 && bash exp_rd_sweep.sh mic 1"                          > "log/chain_gpu1_${TS}.log" 2>&1 &
echo "GPU1_CHAIN=$! (ficus -> mic)"
nohup bash -c "bash exp_rd_sweep.sh bicycle 2 && bash exp_schedule_ablation.sh curves_bicycle 2"    > "log/chain_gpu2_${TS}.log" 2>&1 &
echo "GPU2_CHAIN=$! (bicycle -> curves_bicycle)"
nohup bash exp_rd_sweep.sh drjohnson 3                                                             > "log/chain_gpu3_${TS}.log" 2>&1 &
echo "GPU3_CHAIN=$! (drjohnson)"
echo "=== LAUNCHED $(date) ==="
