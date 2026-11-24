#!/usr/bin/env bash
# Overnight pipeline: scaling benchmark -> train 4 duration-guided models ->
# attention-free prefix ablation (Transformer) -> evaluate -> figures.
# Safe to re-run: finished stages are skipped, training resumes from last.pt.
cd "$(dirname "$0")"
DATA=${SSM_DATA:-C:/ssm_data}
STEPS=${STEPS:-40000}
mkdir -p results "$DATA/runs" "$DATA/runs_prefix"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
run() {  # run <label> <command...>
  local label=$1; shift
  echo "$(stamp) STAGE $label"
  if "$@"; then echo "$(stamp) DONE $label"; else echo "$(stamp) FAILED $label"; fi
}

if [ ! -f results/scaling.json ]; then
  run bench_scaling bash -c "python bench_scaling.py > results/bench_scaling.log 2>&1"
fi

for k in ${KINDS:-s4 transformer lstm mamba}; do
  if [ -f "$DATA/runs/$k/final.pt" ]; then echo "$(stamp) SKIP $k (done)"; continue; fi
  extra=""
  # Pure-PyTorch Mamba is several times slower per step than S4 here: same LR
  # schedule, 4 micro-batches per step (same frames per step), stop early.
  if [ "$k" = "mamba" ]; then extra="--accum 2 --stop_at ${MAMBA_STOP:-16000}"; fi
  run "train $k" bash -c "python train.py --kind $k --steps $STEPS $extra >> $DATA/runs/$k.out 2>&1"
done

# Ablation: the attention-free prefix design (no alignment). S4 was run to 12k
# steps and failed to align; give the Transformer the same 12k steps.
if [ ! -f "$DATA/runs_prefix/transformer/final.pt" ]; then
  run "train transformer_prefix" bash -c "python train.py --kind transformer --variant prefix --steps $STEPS \
    --stop_at 12000 --out $DATA/runs_prefix/transformer >> $DATA/runs_prefix/transformer.out 2>&1"
fi

run evaluate bash -c "python evaluate.py > results/eval.log 2>&1"
run evaluate_prefix bash -c "python evaluate.py --runs $DATA/runs_prefix --kinds s4,transformer --tag _prefix \
  --n 50 > results/eval_prefix.log 2>&1"
run figures bash -c "python make_figures.py > results/figures.log 2>&1"
echo "$(stamp) ALL DONE"
