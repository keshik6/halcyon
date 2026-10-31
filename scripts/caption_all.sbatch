#!/usr/bin/env bash
#SBATCH --job-name=halcyon-i2t
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# Caption the GPIC Test-50K images with each Halcyon I2T-10% model (branch main),
# one array task per size on 4 GPUs. Submit from the repo root:
#
#   mkdir -p logs
#   IMAGES=/path/to/image_dir sbatch scripts/caption_all.sbatch
#   IMAGES=... sbatch --array=3 scripts/caption_all.sbatch   # only Halcyon-1.6B
#
# Output: $OUT_ROOT/<repo>/captions.jsonl (merged, input order) + per-shard files.
set -euo pipefail

IMAGES="${IMAGES:?set IMAGES to an image directory}"
OUT_ROOT="${OUT_ROOT:-outputs/i2t/gpic_test50k}"
ORG="${ORG:-radicalnumerics}"
BRANCH="${BRANCH:-main}"
SEED="${SEED:-1234}"
TEMPERATURE="${TEMPERATURE:-1.0}"

MODELS=(Halcyon-0.1B Halcyon-0.25B Halcyon-0.52B Halcyon-1.6B Halcyon-4B)

cd "${SLURM_SUBMIT_DIR:-.}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate halcyon

repo="${MODELS[${SLURM_ARRAY_TASK_ID:?submit with sbatch (array task id missing)}]}"
out="$OUT_ROOT/$repo"
if [[ -s "$out/captions.jsonl" ]]; then
  echo "skip $out ($(wc -l < "$out/captions.jsonl") captions)"
  exit 0
fi

echo "$ORG/$repo@$BRANCH: $IMAGES -> $out on GPUs ${CUDA_VISIBLE_DEVICES:-?}"
python caption_images.py \
  --model "$ORG/$repo" --revision "$BRANCH" \
  --images "$IMAGES" \
  --temperature "$TEMPERATURE" --top-k 0 --top-p 0 --seed "$SEED" \
  --out-dir "$out" > /dev/null
echo "$(wc -l < "$out/captions.jsonl") captions in $out/captions.jsonl"
head -3 "$out/captions.jsonl" | python -c 'import json,sys; [print(json.loads(l)["image"], "->", json.loads(l)["caption"]) for l in sys.stdin]'
