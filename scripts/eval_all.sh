#!/usr/bin/env bash
# Run gpic-eval (DINOv2) on every image folder under $OUT_ROOT and collect a CSV.
#
#   REF=/path/to/test_stats.npz PROMPTS=/path/to/gpic_test50k.jsonl \
#     bash scripts/eval_all.sh
#
# Needs a conda env "gpic-eval" with gpic-eval installed
# (https://github.com/keshik6/gpic). Expects
# $OUT_ROOT/<model>/<variant>_cfg<scale>/images/ (from generate_all.sbatch). Each run's JSON lands in <run>/out_gpic-eval/; runs that
# already have one are skipped. CSV: $OUT_ROOT/gpic_eval_results.csv
set -euo pipefail

OUT_ROOT="$(realpath "${OUT_ROOT:-outputs/t2i/gpic_test50k}")"
REF="${REF:?set REF to the GPIC Test-50K reference stats (.npz)}"
CSV="${CSV:-$OUT_ROOT/gpic_eval_results.csv}"
PROMPTS="${PROMPTS:?set PROMPTS to the caption JSONL the images came from}"
EXPECTED="${EXPECTED:-$(grep -c . "$PROMPTS")}"  # one image per prompt

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpic-eval

ref_name=$(basename "$REF" .npz)
for img_dir in "$OUT_ROOT"/*/*/images; do
  run=$(dirname "$img_dir")
  json="$run/out_gpic-eval/images_${ref_name}.json"
  n=$(find "$img_dir" -name '*.png' | wc -l)
  if [[ -f "$json" ]]; then
    echo "skip $run (done)"
    continue
  fi
  if (( n < EXPECTED )); then
    echo "skip $run (only $n / $EXPECTED images)" >&2
    continue
  fi
  echo "=== $run ($n images)"
  # gpic-eval writes out_gpic-eval/<basename>_<ref>.json into the cwd, so run it
  # from inside each run dir; no trailing slash, or the basename is empty.
  (cd "$run" && gpic-eval "$img_dir" "$REF" --models dino --batch_size 256 --num_workers 4)
done

python - "$OUT_ROOT" "$ref_name" "$CSV" <<'EOF'
import csv, glob, json, os, sys
root, ref, out = sys.argv[1:]
keys = ["fd", "fd_mu", "fd_sigma", "precision", "recall", "density", "coverage", "mmd"]
rows = []
for path in sorted(glob.glob(f"{root}/*/*/out_gpic-eval/images_{ref}.json")):
    run = os.path.dirname(os.path.dirname(path))
    model, setting = run.split(os.sep)[-2:]
    variant, cfg = setting.rsplit("_cfg", 1)
    dino = json.load(open(path))["dinov2"]
    n = len(glob.glob(f"{run}/images/*.png"))
    rows.append({"model": model, "variant": variant, "cfg": cfg, "n_images": n,
                 **{f"gfd" if k == "fd" else k: dino.get(k) for k in keys}})
with open(out, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["model"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\n{len(rows)} runs -> {out}")
for r in rows:
    print(f"{r['model']:<14} {r['variant']:<9} cfg {r['cfg']:<4} gFD {r['gfd']:.3f}")
EOF
