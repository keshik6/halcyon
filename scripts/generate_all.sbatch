#!/usr/bin/env bash
#SBATCH --job-name=halcyon-t2i
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# GPIC Test-50K samples for every Halcyon model x branch, one array task each
# on 4 GPUs, ready for gpic-eval. Submit from the repo root:
#
#   mkdir -p logs
#   PROMPTS=/path/to/gpic_test50k.jsonl sbatch scripts/generate_all.sbatch
#   PROMPTS=... sbatch --array=6,7 scripts/generate_all.sbatch   # only Halcyon-1.6B
#
# Output: $OUT_ROOT/<repo>/<variant>_cfg<scale>/images/{row:06d}_0.png
# Reruns resume; a finished task exits immediately.
set -euo pipefail

PROMPTS="${PROMPTS:?set PROMPTS to the GPIC Test-50K caption JSONL}"
OUT_ROOT="${OUT_ROOT:-outputs/t2i/gpic_test50k}"
ORG="${ORG:-radicalnumerics}"
SEED="${SEED:-1234}"

# index: HF repo | branch | variant | cfg   (baseline = paper baseline, main = I2T 10%)
# cfg = paper Table 16 (lowest gFD on GPIC Val-50K). baseline: p_drop 0.1;
# i2t10: p_drop 0.0, eta 0.1. Each task checks its branch exists before starting.
RUNS=(
  "Halcyon-0.1B  | baseline      | baseline | 7.5"
  "Halcyon-0.1B  | main          | i2t10    | 7.5"
  "Halcyon-0.25B | baseline      | baseline | 6.0"
  "Halcyon-0.25B | main          | i2t10    | 6.0"
  "Halcyon-0.52B | baseline      | baseline | 6.0"
  "Halcyon-0.52B | main          | i2t10    | 6.0"
  "Halcyon-1.6B  | baseline      | baseline | 5.0"
  "Halcyon-1.6B  | main          | i2t10    | 5.2"
  "Halcyon-4B    | baseline      | baseline | 4.5"
  "Halcyon-4B    | main          | i2t10    | 4.6"
)

cd "${SLURM_SUBMIT_DIR:-.}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate halcyon

task="${SLURM_ARRAY_TASK_ID:?submit with sbatch (array task id missing)}"
IFS='|' read -r repo rev variant cfg <<< "${RUNS[$task]}"
repo=$(echo $repo); rev=$(echo $rev); variant=$(echo $variant); cfg=$(echo $cfg)

python -c '
import sys
from huggingface_hub import list_repo_refs
repo, rev = sys.argv[1:]
branches = [b.name for b in list_repo_refs(repo).branches]
if rev not in branches:
    sys.exit(f"{repo} has no branch {rev!r}; branches: {branches}")
' "$ORG/$repo" "$rev"

out="$OUT_ROOT/$repo/${variant}_cfg$cfg"
n_prompts=$(grep -c . "$PROMPTS")
done_imgs=$( (find "$out/images" -name '*.png' 2>/dev/null || true) | wc -l)
if (( done_imgs >= n_prompts )); then
  echo "skip $out ($done_imgs images)"
  exit 0
fi

echo "task $task: $ORG/$repo@$rev ($variant) cfg $cfg -> $out on GPUs ${CUDA_VISIBLE_DEVICES:-?}"
python generate_images.py \
  --model "$ORG/$repo" --revision "$rev" \
  --prompts "$PROMPTS" \
  --cfg-scale "$cfg" --top-k 0 --top-p 0 --temperature 1.0 --seed "$SEED" \
  --out-dir "$out"
echo "$(find "$out/images" -name '*.png' | wc -l) / $n_prompts images in $out/images"
