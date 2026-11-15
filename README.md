# HDDC

This repository provides the code for **Vision-Language Model Guided Heterogeneous Degradation Divide-and-Conquer for Underwater Image Enhancement**.

## 1. Code Structure

```text
HDDC_github/
├── prompt_tuning/               # Prompt embedding fine-tuning
│   ├── model.py                 # Learnable Text 1 embeddings
│   ├── data.py                  # Images and ranking annotations
│   ├── loss.py                  # Comparison pairs for the ranking loss
│   ├── train.py                 # Fine-tuning entry point
│   ├── config.yaml              # Fine-tuning configuration
│   ├── annotations.example.json # Annotation format example
│   └── weights/                 # Released prompt weights
├── hddc/                        # Underwater image enhancement
│   ├── model.py                 # Full HDDC network
│   ├── block.py                 # Experts and degradation fusion modules
│   ├── dcw.py                   # DCW computation and dynamic routing
│   ├── ddm.py                   # DDM generation
│   ├── data.py                  # Images and DDMs
│   ├── loss.py                  # Enhancement loss
│   ├── utils.py                 # Configuration, checkpoint, and PSNR utilities
│   ├── train.py                 # Training entry point
│   ├── inference.py             # Inference entry point
│   ├── configs/                 # Training, inference, and DDM configurations
│   └── weights/                 # Released enhancement weights
├── data/                        # Training and test data
│   ├── train_dataset/
│   │   ├── raw/                 # Input images
│   │   ├── reference/           # Paired reference images
│   │   └── ddm/                 # Four DDMs for each input image
│   └── test_dataset/
│       ├── raw/
│       ├── reference/
│       └── ddm/
└── requirements.txt
```

Run all commands below from the repository root.

## 2. Installation

Install the dependencies in a Linux environment with Python 3.10 and CUDA-enabled PyTorch:

```bash
python -m pip install -r requirements.txt --no-build-isolation
```

## 3. Inference

### 3.1 Prepare CLIP Weights

By default, we use OpenCLIP `ViT-B-32` with the `laion2b_s34b_b79k` pretrained weights from [Hugging Face](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/tree/main).

To download the weights in advance, run:

```bash
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"
```

The default cache directory is `~/.cache/huggingface/hub/`. DDM generation, inference, and training reuse this cache. If you skip this step, the weights are downloaded automatically when you first run the DDM generation command in Section 3.3.

### 3.2 Prepare the Two HDDC Checkpoints

Download both files and place them at the following paths:

| Checkpoint | Download | Local path |
|---|---|---|
| Enhancement weights `hddc.pth` | [Download](https://github.com/tksr-hitomi/HDDC/releases/download/weights-v1.0.0/hddc.pth) | `hddc/weights/hddc.pth` |
| Prompt weights `ranking_prompts.pth` | [Download](https://github.com/tksr-hitomi/HDDC/releases/download/weights-v1.0.0/ranking_prompts.pth) | `prompt_tuning/weights/ranking_prompts.pth` |

### 3.3 Prepare Images and Generate DDMs

Place the input images in `data/test_dataset/raw/`, then generate their DDMs:

```bash
python -m hddc.ddm \
  --raw-dir ./data/test_dataset/raw \
  --ddm-dir ./data/test_dataset/ddm
```

DDMs are saved to `data/test_dataset/ddm/`.

### 3.4 Run Inference

```bash
python -m hddc.inference --config hddc/configs/inference.yaml
```

Enhanced images are saved to `hddc/outputs/inference/test_dataset/`. You can change the input and output paths in `hddc/configs/inference.yaml`.

## 4. Training

### 4.1 Train HDDC with the Released Prompt Weights

Prepare the CLIP weights as described above. Place `ranking_prompts.pth` in `prompt_tuning/weights/` and prepare paired training and test data:

```text
data/
├── train_dataset/
│   ├── raw/          # Input images
│   ├── reference/    # Paired reference images
│   └── ddm/          # Four DDMs for each input image
└── test_dataset/
    ├── raw/
    ├── reference/
    └── ddm/
```

Input and reference filenames must match without their extensions. Generate DDMs for both datasets as described in Section 3.3.

Set the data paths in `hddc/configs/train.yaml`, then run:

```bash
python -m hddc.train --config hddc/configs/train.yaml
```

Training results are saved to `hddc/outputs/train/`.

### 4.2 Fine-Tune Prompt Embeddings (Optional)

Prepare the CLIP weights, fine-tuning images, and a ranking annotation JSON file. See `prompt_tuning/annotations.example.json` for the annotation format.

```bash
python -m prompt_tuning.train \
  --annotations /path/to/annotations.json \
  --image-root /path/to/images \
  --output-dir ./prompt_tuning/outputs
```

Fine-tuning results are saved to `prompt_tuning/outputs/`.

To use the new prompt weights, update `clip.prompt_checkpoint` in both the HDDC training and inference configurations to the same checkpoint path.
