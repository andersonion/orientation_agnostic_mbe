#!/usr/bin/env python3
"""
train.py

Orientation-agnostic rodent brain extraction training.

Design goals:
  - No canonical orientation requirement.
  - No coordinate / positional channels.
  - Two-channel input:
      channel 0 = normalized image
      channel 1 = normalized gradient magnitude
  - Random handedness-preserving cube rotations during training.
  - Binary brain mask output.
  - Standard MONAI SwinUNETR.
  - Patch-based training with spatial padding for small inputs.

Expected training data:
  --img_dir  contains image NIfTI files (*.nii.gz)
  --mask_dir contains matching mask NIfTI files (*.nii.gz)

Matching is intentionally permissive but safe:
  image basename -> remove "_T2.nii.gz" or ".nii.gz"
  mask is first file in mask_dir containing that base id and "mask".

Example:
  python train.py \
    --img_dir /path/to/images \
    --mask_dir /path/to/masks \
    --model_ckpt_save_dir ./checkpoints/best_model.pt \
    --patch_size 96 \
    --training_epochs 120 \
    --batch_size 2
"""

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, ConfusionMatrixMetric
from monai.networks.nets import SwinUNETR
from monai.optimizers import WarmupCosineSchedule
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    KeepLargestConnectedComponent,
    LoadImaged,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandScaleIntensityd,
    SpatialPadd,
)


# -----------------------------
# Proper cube rotations
# -----------------------------

def _permutation_parity(perm: Tuple[int, int, int]) -> int:
    inv = 0
    for i in range(3):
        for j in range(i + 1, 3):
            if perm[i] > perm[j]:
                inv += 1
    return 1 if inv % 2 == 0 else -1


def proper_cube_rotations() -> List[Dict[str, object]]:
    """
    Return the 24 handedness-preserving rotations of a cube.

    The transform acts on spatial dimensions only, with tensors shaped:
      [C, X, Y, Z] or [B, C, X, Y, Z]

    Each rotation is represented as:
      perm: a permutation of spatial axes
      flips: spatial axes to flip AFTER permutation, expressed in permuted-axis indices

    determinant = parity(perm) * product(flip signs) = +1
    """
    rots = []
    perms = [
        (0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0),
    ]

    for perm in perms:
        parity = _permutation_parity(perm)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    det = parity * sx * sy * sz
                    if det != 1:
                        continue
                    flips = tuple(i for i, s in enumerate((sx, sy, sz)) if s == -1)
                    rots.append({"perm": perm, "flips": flips})

    if len(rots) != 24:
        raise RuntimeError(f"Expected 24 proper rotations; got {len(rots)}")

    return rots


PROPER_ROTATIONS = proper_cube_rotations()


def apply_spatial_rotation_tensor(x: torch.Tensor, rot: Dict[str, object]) -> torch.Tensor:
    """
    Apply proper cube rotation to a tensor.

    Supports:
      [C, X, Y, Z]
      [B, C, X, Y, Z]
    """
    perm = tuple(rot["perm"])
    flips = tuple(rot["flips"])

    if x.ndim == 4:
        spatial_dims = (1, 2, 3)
        permuted = x.permute(0, 1 + perm[0], 1 + perm[1], 1 + perm[2])
        flip_dims = tuple(1 + f for f in flips)
    elif x.ndim == 5:
        spatial_dims = (2, 3, 4)
        permuted = x.permute(0, 1, 2 + perm[0], 2 + perm[1], 2 + perm[2])
        flip_dims = tuple(2 + f for f in flips)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")

    if flip_dims:
        permuted = torch.flip(permuted, dims=flip_dims)

    return permuted.contiguous()


class RandProperCubeRotationd:
    """
    MONAI-style dictionary transform applying one random handedness-preserving
    cube rotation to all specified keys.

    This is intentionally array/tensor-only and does not modify affine/header.
    Training is done in voxel/index space.
    """

    def __init__(self, keys: List[str], prob: float = 1.0):
        self.keys = keys
        self.prob = prob
        self.rots = PROPER_ROTATIONS

    def __call__(self, data):
        d = dict(data)
        if random.random() > self.prob:
            return d

        rot = random.choice(self.rots)

        for key in self.keys:
            arr = d[key]
            if not torch.is_tensor(arr):
                arr = torch.as_tensor(arr)
            d[key] = apply_spatial_rotation_tensor(arr, rot)

        return d


# -----------------------------
# Gradient channel
# -----------------------------

def add_gradient_channel(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Add normalized gradient magnitude as channel 1.

    Input:
      [B, 1, X, Y, Z]

    Output:
      [B, 2, X, Y, Z]
    """
    if x.ndim != 5 or x.shape[1] != 1:
        raise ValueError(f"Expected image tensor [B,1,X,Y,Z], got {tuple(x.shape)}")

    img = x[:, 0]

    gx = torch.zeros_like(img)
    gy = torch.zeros_like(img)
    gz = torch.zeros_like(img)

    gx[:, 1:-1, :, :] = (img[:, 2:, :, :] - img[:, :-2, :, :]) * 0.5
    gy[:, :, 1:-1, :] = (img[:, :, 2:, :] - img[:, :, :-2, :]) * 0.5
    gz[:, :, :, 1:-1] = (img[:, :, :, 2:] - img[:, :, :, :-2]) * 0.5

    grad = torch.sqrt(gx * gx + gy * gy + gz * gz + eps)

    # Per-volume robust-ish normalization.
    flat = grad.flatten(start_dim=1)
    mean = flat.mean(dim=1).view(-1, 1, 1, 1)
    std = flat.std(dim=1).view(-1, 1, 1, 1).clamp_min(eps)
    grad = (grad - mean) / std

    return torch.cat([x, grad.unsqueeze(1)], dim=1)


# -----------------------------
# Data helpers
# -----------------------------

def image_id_from_name(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith("_T2.nii.gz"):
        return base[:-len("_T2.nii.gz")]
    if base.endswith(".nii.gz"):
        return base[:-len(".nii.gz")]
    if base.endswith(".nii"):
        return base[:-len(".nii")]
    return os.path.splitext(base)[0]


def find_data_pairs(img_dir: str, mask_dir: str) -> List[Dict[str, str]]:
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))

    data_dicts = []
    for img in image_files:
        base = os.path.basename(img)
        if base.startswith("._") or "mask" in base.lower() or "normcoord" in base.lower():
            continue

        img_id = image_id_from_name(img)

        candidates = []

        for m in mask_files:

            mask_base = os.path.basename(m).lower()

            # Require explicit delimiter boundary.
            #
            # VALID:
            #   A123_mask.nii.gz
            #   A123_brain_mask.nii.gz
            #
            # INVALID:
            #   XA123_mask.nii.gz
            #   A1234_mask.nii.gz
            #

            if not mask_base.startswith(f"{img_id.lower()}_"):
                continue

            if "mask" not in mask_base:
                continue

            candidates.append(m)

        if not candidates:
            print(f"[WARN] No mask found for image: {img}")
            continue

        if len(candidates) > 1:
            print(f"[WARN] Multiple masks found for {img}; using first: {candidates[0]}")

        data_dicts.append({"image": img, "label": candidates[0]})

    return data_dicts


def split_train_val(data_dicts: List[Dict[str, str]], val_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(data_dicts)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    if not train_files:
        raise RuntimeError("No training files after split. Provide more data or lower val_fraction.")

    return train_files, val_files


def build_transforms(patch_size: Tuple[int, int, int], rotate_prob: float):
    keys = ["image", "label"]

    train_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        NormalizeIntensityd(keys="image", nonzero=True),
        SpatialPadd(keys=keys, spatial_size=patch_size),
        RandProperCubeRotationd(keys=keys, prob=rotate_prob),
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=2,
            allow_smaller=False,
        ),
        RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.7, 1.5)),
        RandScaleIntensityd(keys="image", prob=0.2, factors=0.2),
        RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.1),
    ])

    val_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        NormalizeIntensityd(keys="image", nonzero=True),
    ])

    return train_transforms, val_transforms


def make_model(patch_size: Tuple[int, int, int], feature_size: int, device: str):
    model = SwinUNETR(
        img_size=patch_size,
        in_channels=2,
        out_channels=2,
        feature_size=feature_size,
        use_checkpoint=True,
        spatial_dims=3,
    )
    return model.to(device)


def load_weights_flexibly(model: torch.nn.Module, weights_path: str, device: str, strict: bool = False):
    if not weights_path:
        return

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    weights = checkpoint.get("model_weights", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    missing, unexpected = model.load_state_dict(weights, strict=strict)
    print(f"[INFO] Loaded weights from: {weights_path}")
    if missing:
        print(f"[INFO] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[INFO] Unexpected keys: {len(unexpected)}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train orientation-agnostic rodent brain extractor.")
    parser.add_argument("--img_dir", required=True, help="Directory containing input image NIfTIs.")
    parser.add_argument("--mask_dir", required=True, help="Directory containing binary mask NIfTIs.")
    parser.add_argument("--model_ckpt_save_dir", required=True, help="Path to save best model checkpoint.")
    parser.add_argument("--pretrained_weights", default="", help="Optional pretrained weights.")
    parser.add_argument("--strict_load", action="store_true", help="Strictly load pretrained weights.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--training_epochs", type=int, default=120)
    parser.add_argument("--patch_size", type=int, default=96, help="Isotropic patch size.")
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--rotate_prob", type=float, default=1.0, help="Probability of random proper cube rotation.")
    parser.add_argument("--cache_rate", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="orientation-agnostic-mbe")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_last", action="store_true", help="Also save last epoch checkpoint next to best checkpoint.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    ckpt_path = Path(args.model_ckpt_save_dir)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["patch_size_tuple"] = patch_size
    config["in_channels"] = 2
    config["out_channels"] = 2

    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb requested but not importable.")
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"oa-mbe-ps{args.patch_size}",
            config=config,
        )
    else:
        class DummyWandb:
            @staticmethod
            def log(*a, **k): pass
            @staticmethod
            def finish(): pass
        globals()["wandb"] = DummyWandb()

    data_dicts = find_data_pairs(args.img_dir, args.mask_dir)
    if len(data_dicts) < 2:
        raise RuntimeError(f"Need at least 2 image/mask pairs; found {len(data_dicts)}")

    train_files, val_files = split_train_val(data_dicts, args.val_fraction, args.seed)
    print(f"[INFO] Training samples: {len(train_files)}")
    print(f"[INFO] Validation samples: {len(val_files)}")
    print(f"[INFO] Patch size: {patch_size}")
    print(f"[INFO] Device: {args.device}")

    train_transforms, val_transforms = build_transforms(patch_size, args.rotate_prob)

    train_ds = CacheDataset(train_files, transform=train_transforms, cache_rate=args.cache_rate)
    val_ds = CacheDataset(val_files, transform=val_transforms, cache_rate=args.cache_rate)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = make_model(patch_size, args.feature_size, args.device)
    if args.pretrained_weights:
        load_weights_flexibly(model, args.pretrained_weights, args.device, strict=args.strict_load)

    loss_function = DiceFocalLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=True,
        gamma=2.0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10, t_total=args.training_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    msd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")
    sens_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name="sensitivity",
        compute_sample=True,
        reduction="mean",
    )

    post_pred = Compose([
        AsDiscrete(argmax=True, to_onehot=2),
        KeepLargestConnectedComponent(applied_labels=[1]),
    ])
    post_label = AsDiscrete(to_onehot=2)

    best_val_dice = -float("inf")
    best_val_hd95 = float("inf")

    metadata_path = ckpt_path.with_suffix(ckpt_path.suffix + ".json")
    metadata_path.write_text(json.dumps(config, indent=2, default=str) + "\n")

    for epoch in range(1, args.training_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.training_epochs} [Train]")
        for batch in pbar:
            step += 1

            x = batch["image"].to(args.device)
            y = batch["label"].to(args.device)

            x2 = add_gradient_channel(x)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                outputs = model(x2)
                if outputs.shape[2:] != y.shape[2:]:
                    raise RuntimeError(
                        f"Training output/label spatial mismatch: "
                        f"outputs={tuple(outputs.shape)}, labels={tuple(y.shape)}"
                    )
                loss = loss_function(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            pbar.set_postfix({"loss": float(loss.item())})

        mean_train_loss = epoch_loss / max(step, 1)
        scheduler.step()
        epoch_seconds = time.time() - t0

        wandb.log({
            "train/loss": mean_train_loss,
            "train/lr": scheduler.get_last_lr()[0],
            "train/epoch_seconds": epoch_seconds,
        }, step=epoch)

        if epoch % args.val_interval != 0:
            continue

        model.eval()
        val_loss_epoch = 0.0

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.training_epochs} [Val]")
            for batch in vbar:
                val_x = batch["image"].to(args.device)
                val_y = batch["label"].to(args.device)
                val_x2 = add_gradient_channel(val_x)

                with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                    val_outputs = sliding_window_inference(
                        inputs=val_x2,
                        roi_size=patch_size,
                        sw_batch_size=4,
                        predictor=model,
                        overlap=0.5,
                    )

                    if val_outputs.shape[2:] != val_y.shape[2:]:
                        raise RuntimeError(
                            f"Validation output/label spatial mismatch: "
                            f"outputs={tuple(val_outputs.shape)}, labels={tuple(val_y.shape)}"
                        )

                    loss = loss_function(val_outputs, val_y)

                val_loss_epoch += float(loss.item())

                val_outputs_list = decollate_batch(val_outputs)
                val_labels_list = decollate_batch(val_y)

                val_outputs_clean = [post_pred(i) for i in val_outputs_list]
                val_labels_clean = [post_label(i) for i in val_labels_list]

                dice_metric(y_pred=val_outputs_clean, y=val_labels_clean)
                hd95_metric(y_pred=val_outputs_clean, y=val_labels_clean)
                msd_metric(y_pred=val_outputs_clean, y=val_labels_clean)
                sens_metric(y_pred=val_outputs_clean, y=val_labels_clean)

        mean_dice = float(dice_metric.aggregate().item())
        mean_hd95 = float(hd95_metric.aggregate().item())
        mean_msd = float(msd_metric.aggregate().item())
        mean_sens = float(sens_metric.aggregate()[0].item())
        mean_val_loss = val_loss_epoch / max(len(val_loader), 1)

        dice_metric.reset()
        hd95_metric.reset()
        msd_metric.reset()
        sens_metric.reset()

        print(
            f"Val => Dice: {mean_dice:.4f} | HD95: {mean_hd95:.4f} | "
            f"MSD: {mean_msd:.4f} | Sens: {mean_sens:.4f} | loss: {mean_val_loss:.4f}"
        )

        wandb.log({
            "val/loss": mean_val_loss,
            "val/dice": mean_dice,
            "val/hd95": mean_hd95,
            "val/msd": mean_msd,
            "val/sensitivity": mean_sens,
        }, step=epoch)

        # Primary: Dice. Tie-breaker: HD95.
        is_best = (mean_dice > best_val_dice) or (
            math.isclose(mean_dice, best_val_dice) and mean_hd95 < best_val_hd95
        )

        if is_best:
            best_val_dice = mean_dice
            best_val_hd95 = mean_hd95
            torch.save(model.state_dict(), ckpt_path)
            print(f"[*] New best model saved: {ckpt_path}")
            print(f"    Best Dice={best_val_dice:.4f}, HD95={best_val_hd95:.4f}")

        if args.save_last:
            last_path = ckpt_path.with_name(ckpt_path.stem + "_last" + ckpt_path.suffix)
            torch.save(model.state_dict(), last_path)

    wandb.finish()
    print("[DONE] Fine-tuning complete.")


if __name__ == "__main__":
    main()
