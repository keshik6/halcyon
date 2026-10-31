
<h1 align="center">Halcyon</h1>
<p align="center"><strong>Text-to-Image Generative Modeling Without Language Pretraining</strong></p>

<p align="center">
  <a href="https://keshik6.github.io/">Keshigeyan Chandrasegaran</a><sup>1,2</sup>,
  <a href="https://zhaoyue-zephyrus.github.io/">Yue Zhao</a><sup>1</sup>,
  <a href="https://www.linkedin.com/in/athms">Armin W. Thomas</a><sup>2</sup>,
  <a href="https://web.stanford.edu/~eadeli/">Ehsan Adeli</a><sup>1</sup>,<br>
  <a href="https://jiajunwu.com/">Jiajun Wu</a><sup>1</sup>,
  <a href="https://www.niebles.net/">Juan Carlos Niebles</a><sup>1</sup>,
  <a href="https://profiles.stanford.edu/fei-fei-li">Li Fei-Fei</a><sup>1</sup>,
  <a href="https://zymrael.github.io/">Michael Poli</a><sup>1,2</sup>
</p>

<p align="center">
  <sup>1</sup> Stanford University &nbsp;&nbsp; <sup>2</sup> Radical Numerics
</p>

<p align="center">
  <a href="https://cs.stanford.edu/~keshik/halcyon">🌎 Project Page</a> |
  <a href="https://arxiv.org/abs/2610.26863">📄 Paper</a> |
  <a href="https://huggingface.co/collections/radicalnumerics/halcyon">🤗 Models</a> |
  <a href="https://www.radicalnumerics.ai/blog/halcyon">✍️ Blog</a>
</p>


<p align="center">
  <img src="assets/halcyon.webp" width="100%">
</p>


## 📣 News

- **[2026-10-30]: Inference code and demo notebook released.**
- **[2026-10-15]: Halcyon paper and model weights released.**


## About this code

This repository provides inference code for **Halcyon**, a family of early-fusion multimodal autoregressive models trained entirely from scratch, without language pretraining.

Halcyon uses a decoder-only Transformer trained on [GPIC](https://github.com/keshik6/gpic). The model uses a joint vocabulary with 16,384 text tokens and 64,000 image tokens. All components, including the text tokenizer, image tokenizer, and generative model, are trained from scratch. The released models are base models (no post-training).

## Setup

```bash
git clone https://github.com/keshik6/halcyon.git
cd halcyon
conda env create -f environment.yml
conda activate halcyon
```

This installs vLLM 0.24.0, PyTorch 2.11.0 (CUDA 13.0 runtime, so the NVIDIA driver must support CUDA 13.0), and Transformers 5.10.1. Models and tokenizers are downloaded from the Hugging Face Hub automatically.

## Text-to-image

```bash
# One or more captions
python generate_images.py --caption "A red fox sits in fresh snow at dawn, soft golden light on its fur and frosted pine trees behind it."

# A JSONL file with a "caption" field per line (or a .txt file, one caption per line)
python generate_images.py --prompts examples/prompts.jsonl
```

Images are written to `outputs/t2i/images/{row:06d}_{sample}.png`, alongside the raw image codes (`codes_NNN.bin`) and a manifest containing the caption and seed of every sample (`manifest_NNN.jsonl`).

## Image-to-text

```bash
# A single image
python caption_images.py --images examples/sample.png

# A directory of images
python caption_images.py --images /path/to/image_dir
```

Captions are written to `outputs/i2t/captions.jsonl`. Images of any resolution are accepted; each image is center-cropped to the largest square, resized to 256×256, and tokenized.

## Demo notebook

[`demo_notebooks/halcyon_demo.ipynb`](demo_notebooks/halcyon_demo.ipynb) walks through model setup, tokenization, and step-by-step sampling for text-to-image generation and image captioning using Halcyon-4B. The notebook runs on a single H100 GPU.

## Models

Each model repository provides two branches:

- `baseline`: Standard text-to-image (T2I) training with 10% caption dropout.
- `main`: Replaces 10% caption dropout with image-to-text (I2T) supervision, enabling image captioning under the same pretraining budget.

By default, `generate_images.py` uses `baseline`, while `caption_images.py` uses `main`. Both scripts default to `Halcyon-1.6B`. Use `--model` to select a different model size and `--revision` to select a different branch.

| Model | CFG scale (`baseline` / `main`) | gFD `baseline` | gFD `main` |
|---|---|---|---|
| [`radicalnumerics/Halcyon-0.1B`](https://huggingface.co/radicalnumerics/Halcyon-0.1B) | 7.5 / 7.5 | 144.16 | 143.89 |
| [`radicalnumerics/Halcyon-0.25B`](https://huggingface.co/radicalnumerics/Halcyon-0.25B) | 6.0 / 6.0 | 72.55 | 70.60 |
| [`radicalnumerics/Halcyon-0.52B`](https://huggingface.co/radicalnumerics/Halcyon-0.52B) | 6.0 / 6.0 | 48.02 | 48.39 |
| [`radicalnumerics/Halcyon-1.6B`](https://huggingface.co/radicalnumerics/Halcyon-1.6B) | 5.0 / 5.2 | 35.28 | 36.00 |
| [`radicalnumerics/Halcyon-4B`](https://huggingface.co/radicalnumerics/Halcyon-4B) | 4.5 / 4.6 | 28.42 | 29.30 |

gFD (DINOv2 Fréchet distance; lower is better) is evaluated on GPIC Test-50K at 256×256 resolution using this repository's inference code. We generate one image per caption with seed 1234, temperature 1.0, no top-k or top-p filtering, and the CFG scales shown in the table.

Text and image tokenizers are available at [`radicalnumerics/Halcyon-Tokenizers`](https://huggingface.co/radicalnumerics/Halcyon-Tokenizers).

## Evaluation

Evaluate generated images with the `gpic-eval` package from [keshik6/gpic](https://github.com/keshik6/gpic):

```bash
gpic-eval outputs/t2i/images /path/to/test_stats.npz --models dino --batch_size 256
```

The `scripts/` directory contains slurm scripts.

- `generate_all.sbatch`: Generate images with all 10 model/branch combinations on GPIC Test-50K.
- `caption_all.sbatch`: Caption an image directory using every `main` model.
- `eval_all.sh`: Run `gpic-eval` on each output directory and collect the results in a CSV.

## Notes

The models and results reported in the paper were obtained using our internal training and inference stack. This repository is a faithful port of the inference pipeline, with minor numerical differences.

Generation and captioning use all available GPUs, with one process per GPU. Restrict GPU usage with `CUDA_VISIBLE_DEVICES` or `--num-gpus`. Generation resumes from the last completed sample when rerun with the same `--out-dir`. Run either script with `--help` for all available options.


## AI Usage

Claude Opus 5.5 was used to assist with porting the inference code and writing the demo notebook. The ported implementation was verified by the authors through parity tests and gFD evaluation.


## Acknowledgments

This codebase builds on several open-source projects:

- [GPIC](https://github.com/keshik6/gpic) for training data and evaluation.
- [vLLM](https://github.com/vllm-project/vllm) for efficient batched inference.
- [NVIDIA Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) for the image tokenizer architecture.
- [Qwen3](https://github.com/QwenLM/Qwen3) for the LLM architecture.
- [Hugging Face Transformers](https://github.com/huggingface/transformers) and [Tokenizers](https://github.com/huggingface/tokenizers).


## Citation

```bibtex
@article{chandrasegaran2026halcyon,
  title   = {Text-to-Image Generative Modeling Without Language Pretraining},
  author  = {Chandrasegaran, Keshigeyan and Zhao, Yue and Thomas, Armin W. and Adeli, Ehsan and Wu, Jiajun and Niebles, Juan Carlos and Fei-Fei, Li and Poli, Michael},
  journal = {arXiv preprint arXiv:2610.26863},
  year    = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
