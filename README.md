# Orientation-Agnostic Mouse Brain Extractor

A MONAI/PyTorch brain-mask training and inference framework designed for rodent MRI data where image orientation cannot be trusted.

The guiding rule is:

> **Never canonicalize and save.**  
> The output mask must remain in the exact voxel/header space of the input NIfTI.

Internally, inference can rotate arrays, ensemble predictions, and invert them back. The saved mask should overlay the original input image directly, including whatever strange affine/header conventions came with the file.

---

## Why this exists

Many rodent MRI skull-stripping pipelines quietly assume some combination of:

- canonical orientation
- reliable NIfTI headers
- predictable anatomical axis order
- consistent scanner/export conventions

Those assumptions often fail.

This project instead treats orientation uncertainty as a first-class problem.

The model is trained with handedness-preserving 3D cube rotations, and inference can use test-time augmentation over the 24 proper rotations of the cube. Predictions are inverted back to native voxel space and averaged as probabilities.

---

## Core design

### Input channels

The network uses two input channels:

1. normalized image intensity
2. normalized gradient magnitude

The gradient channel helps the model learn boundary/edge information that can survive contrast changes better than raw intensity alone.

### Output

Binary segmentation:

- `0` = background
- `1` = brain

### Orientation handling

This project uses **rotation equivariance**, not rotation invariance.

That distinction matters:

- Bad: "the output is identical no matter how the input is rotated"
- Good: "if the input rotates, the predicted mask rotates with it"

Inference works by:

1. applying a proper 3D rotation to the image array
2. running the model
3. applying softmax
4. inverting the probability map back to original voxel space
5. averaging probabilities across rotations
6. taking the final argmax
7. saving the mask in original input voxel/header space

Only handedness-preserving rotations are used: the 24 proper cube rotations.

---

## Files

- `train.py`  
  Train a standard MONAI `SwinUNETR` with two-channel input and random proper cube rotations.

- `inference.py`  
  Run native-space, orientation-agnostic inference with `none`, `fast`, `medium`, `full`, or `adaptive` TTA.

- `benchmark_configs.py`  
  Compare patch sizes and TTA modes using existing checkpoints.

---

## Installation

Create an environment with PyTorch, MONAI, nibabel, scipy, numpy, and tqdm.

Example:

```bash
conda create -n oa_mbe python=3.10 -y
conda activate oa_mbe

# Install the appropriate PyTorch build for your CUDA version.
# Example only:
pip install torch torchvision torchaudio

pip install monai nibabel scipy numpy tqdm
```

Optional:

```bash
pip install wandb
```

---

## Training

Example:

```bash
python train.py \
  --img_dir /path/to/images \
  --mask_dir /path/to/masks \
  --model_ckpt_save_dir ./checkpoints/best_model_96.pt \
  --patch_size 96 \
  --training_epochs 120 \
  --batch_size 2 \
  --device cuda
```

### Notes

The default model input patch is isotropic:

```text
96 x 96 x 96
```

Low-resolution images smaller than the patch size are padded during training/inference.

To try a larger patch later:

```bash
python train.py \
  --img_dir /path/to/images \
  --mask_dir /path/to/masks \
  --model_ckpt_save_dir ./checkpoints/best_model_128.pt \
  --patch_size 128 \
  --training_epochs 120 \
  --batch_size 1 \
  --device cuda
```

Larger patches may improve anatomical context but increase VRAM and reduce batch diversity.

---

## Inference

Adaptive TTA is the recommended default:

```bash
python inference.py \
  --model_ckpt_save_dir ./checkpoints/best_model_96.pt \
  --inference_dir /path/to/images \
  --output_dir ./predicted_masks \
  --patch_size 96 \
  --tta adaptive \
  --device cuda
```

CPU inference is supported:

```bash
python inference.py \
  --model_ckpt_save_dir ./checkpoints/best_model_96.pt \
  --inference_dir /path/to/images \
  --output_dir ./predicted_masks_cpu \
  --patch_size 96 \
  --tta adaptive \
  --device cpu \
  --sw_batch_size 1
```

### TTA modes

```text
none      1 orientation
fast      6 orientations
medium    12 orientations
full      all 24 proper rotations
adaptive  starts small and adds rotations until stable, up to 24
```

### Adaptive TTA

Adaptive mode starts with a small number of rotations, then adds more if the probability map remains uncertain or the hard mask changes between batches.

Important options:

```bash
--adaptive_initial 4
--adaptive_batch 4
--adaptive_min_rotations 8
--adaptive_ambiguity_threshold 0.002
--adaptive_dice_threshold 0.995
```

A conservative run:

```bash
python inference.py \
  --model_ckpt_save_dir ./checkpoints/best_model_96.pt \
  --inference_dir /path/to/images \
  --tta full \
  --patch_size 96
```

---

## Physical volume warnings

Inference can warn if the final mask volume is outside a plausible range.

Example for mouse brain:

```bash
python inference.py \
  --model_ckpt_save_dir ./checkpoints/best_model_96.pt \
  --inference_dir /path/to/images \
  --tta adaptive \
  --min_volume_mm3 300 \
  --max_volume_mm3 700
```

For rat brain, use a larger range or disable the check.

The volume check is a warning mechanism, not a hard failure.

---

## Benchmarking patch size and TTA mode

After training separate checkpoints, compare compute and quality:

```bash
python benchmark_configs.py \
  --images /path/to/images \
  --masks /path/to/masks \
  --model96 ./checkpoints/best_model_96.pt \
  --model128 ./checkpoints/best_model_128.pt \
  --inference_script ./inference.py \
  --tta_modes fast adaptive full \
  --output_csv benchmark_results.csv \
  --device cuda
```

This produces a CSV with timing and optional quality metrics.

---

## Output guarantee

The output mask is saved:

- in the original input array shape
- with the original input affine
- with a copied input header
- without canonicalizing the NIfTI

The final mask should overlay the input image directly.

---

## Current limitations

- This is a v1 implementation.
- Training assumes paired image/mask files can be matched by filename substring.
- Adaptive TTA thresholds may need tuning.
- `benchmark_configs.py` uses a lightweight Hausdorff approximation in voxel units, not a full HD95 implementation.
- Rat brain support should work structurally, but volume priors must be changed or disabled.

---

## Philosophy

This project trades some compute for robustness.

Instead of trying to guess or repair orientation before masking, it asks the model the same question from multiple physically valid orientations and combines the answers in native space.

That avoids the chicken-and-egg problem:

> "I need a decent mask to determine orientation, but I need orientation to get a decent mask."

The goal is to make the first mask good enough without trusting orientation first.
