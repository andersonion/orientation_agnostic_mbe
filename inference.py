#!/usr/bin/env python3
"""
inference.py

Orientation-agnostic rodent brain extraction inference.

Design goals:
  - Never canonicalize input or saved output.
  - Internal rotations are array-space only.
  - Final mask is inverted back into original voxel array space.
  - Output NIfTI preserves input affine/header as much as nibabel allows.
  - Two-channel input:
      channel 0 = normalized image
      channel 1 = normalized gradient magnitude
  - Adaptive, fast, or full 24 proper-rotation TTA.
  - Largest-connected-component cleanup by default.
  - Optional physical volume warning.

Example:
  python inference.py \
    --model_ckpt_save_dir ./checkpoints/best_model.pt \
    --inference_dir /path/to/images \
    --output_dir ./predicted_masks \
    --patch_size 96 \
    --tta adaptive
"""

import argparse
import csv
import glob
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR


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
    Apply proper cube rotation to tensor.

    Supports:
      [C, X, Y, Z]
      [B, C, X, Y, Z]
    """
    perm = tuple(rot["perm"])
    flips = tuple(rot["flips"])

    if x.ndim == 4:
        y = x.permute(0, 1 + perm[0], 1 + perm[1], 1 + perm[2])
        flip_dims = tuple(1 + f for f in flips)
    elif x.ndim == 5:
        y = x.permute(0, 1, 2 + perm[0], 2 + perm[1], 2 + perm[2])
        flip_dims = tuple(2 + f for f in flips)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")

    if flip_dims:
        y = torch.flip(y, dims=flip_dims)

    return y.contiguous()


def invert_spatial_rotation_tensor(y: torch.Tensor, rot: Dict[str, object]) -> torch.Tensor:
    """
    Invert apply_spatial_rotation_tensor.

    If forward is:
      y = flip(permute(x))

    inverse is:
      x = inverse_permute(flip(y))
    """
    perm = tuple(rot["perm"])
    flips = tuple(rot["flips"])

    if y.ndim == 4:
        if flips:
            y = torch.flip(y, dims=tuple(1 + f for f in flips))
        inv_perm = [0, 0, 0]
        for new_axis, old_axis in enumerate(perm):
            inv_perm[old_axis] = new_axis
        x = y.permute(0, 1 + inv_perm[0], 1 + inv_perm[1], 1 + inv_perm[2])
    elif y.ndim == 5:
        if flips:
            y = torch.flip(y, dims=tuple(2 + f for f in flips))
        inv_perm = [0, 0, 0]
        for new_axis, old_axis in enumerate(perm):
            inv_perm[old_axis] = new_axis
        x = y.permute(0, 1, 2 + inv_perm[0], 2 + inv_perm[1], 2 + inv_perm[2])
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(y.shape)}")

    return x.contiguous()


def get_rotation_subset(mode: str) -> List[Dict[str, object]]:
    """
    Return rotations for fixed modes.

    identity is always first.
    """
    rots = PROPER_ROTATIONS

    # Ensure identity first.
    identity_idx = None
    for i, r in enumerate(rots):
        if tuple(r["perm"]) == (0, 1, 2) and tuple(r["flips"]) == ():
            identity_idx = i
            break

    if identity_idx is None:
        raise RuntimeError("Identity rotation not found.")

    rots = [rots[identity_idx]] + rots[:identity_idx] + rots[identity_idx + 1:]

    if mode == "none":
        return rots[:1]
    if mode == "fast":
        return rots[:6]
    if mode == "medium":
        return rots[:12]
    if mode in ("full", "adaptive"):
        return rots

    raise ValueError(f"Unknown tta mode: {mode}")


# -----------------------------
# Image preprocessing
# -----------------------------

def robust_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    data = arr.astype(np.float32, copy=False)
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("Image has no finite voxels.")

    vals = data[finite]
    nonzero = vals[np.abs(vals) > eps]
    use = nonzero if nonzero.size > 100 else vals

    mean = float(use.mean())
    std = float(use.std())
    if std < eps:
        std = 1.0

    out = (data - mean) / std
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def gradient_magnitude_channel(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    gx, gy, gz = np.gradient(img.astype(np.float32, copy=False))
    grad = np.sqrt(gx * gx + gy * gy + gz * gz + eps).astype(np.float32, copy=False)

    finite = np.isfinite(grad)
    if finite.any():
        vals = grad[finite]
        mean = float(vals.mean())
        std = float(vals.std())
        if std < eps:
            std = 1.0
        grad = (grad - mean) / std
        grad[~finite] = 0.0
    else:
        grad[:] = 0.0

    return grad.astype(np.float32, copy=False)


def make_two_channel_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    img = robust_normalize(arr)
    grad = gradient_magnitude_channel(img)
    stacked = np.stack([img, grad], axis=0)  # [2, X, Y, Z]
    return torch.from_numpy(stacked).unsqueeze(0).to(device=device, dtype=torch.float32)


def pad_to_min_spatial(x: torch.Tensor, min_size: Tuple[int, int, int]):
    """
    Pad [B,C,X,Y,Z] so each spatial dim >= min_size.
    Returns padded tensor and original spatial shape.
    """
    orig_shape = tuple(x.shape[2:])
    pads = []
    for current, target in reversed(list(zip(orig_shape, min_size))):
        total = max(0, target - current)
        before = total // 2
        after = total - before
        pads.extend([before, after])

    if any(pads):
        x = torch.nn.functional.pad(x, pads, mode="constant", value=0)

    return x, orig_shape


def crop_to_shape(x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Center-crop [B,C,X,Y,Z] back to shape.
    """
    slices = []
    for dim, target in zip(x.shape[2:], shape):
        if dim == target:
            slices.append(slice(None))
        elif dim > target:
            start = (dim - target) // 2
            slices.append(slice(start, start + target))
        else:
            raise ValueError(f"Cannot crop dim {dim} to larger target {target}")

    return x[(slice(None), slice(None), *slices)]


# -----------------------------
# Model/inference helpers
# -----------------------------

def make_model(patch_size: Tuple[int, int, int], feature_size: int, device: str):
    model = SwinUNETR(
        in_channels=2,
        out_channels=2,
        feature_size=feature_size,
        use_checkpoint=True,
        spatial_dims=3,
    )
    return model.to(device)


def load_model(model, ckpt_path: str, device: str):
    checkpoint = torch.load(ckpt_path, map_location=device)
    weights = checkpoint.get("model_weights", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def infer_one_orientation(
    model,
    native_input: torch.Tensor,
    rot: Dict[str, object],
    patch_size: Tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    device: str,
):
    """
    Rotate native two-channel input, run sliding-window inference,
    softmax, and invert probability map back to native voxel space.

    Returns:
      prob_native [1,2,X,Y,Z]
    """
    x_rot = apply_spatial_rotation_tensor(native_input, rot)
    x_rot, orig_rot_shape = pad_to_min_spatial(x_rot, patch_size)

    with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        logits_rot = sliding_window_inference(
            inputs=x_rot,
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )

    logits_rot = crop_to_shape(logits_rot, orig_rot_shape)

    prob_rot = torch.softmax(logits_rot.float(), dim=1)
    prob_native = invert_spatial_rotation_tensor(prob_rot, rot)

    return prob_native


def ambiguity_fraction(prob_fg: torch.Tensor, low: float = 0.40, high: float = 0.60) -> float:
    """
    Fraction of voxels near decision boundary.
    prob_fg shape: [X,Y,Z] or [1,X,Y,Z]
    """
    p = prob_fg.detach()
    frac = ((p >= low) & (p <= high)).float().mean().item()
    return float(frac)


def dice_between_binary(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.bool()
    b = b.bool()
    inter = (a & b).sum().float()
    denom = a.sum().float() + b.sum().float()
    return float(((2 * inter + eps) / (denom + eps)).item())


def run_tta_inference(
    model,
    native_input: torch.Tensor,
    patch_size: Tuple[int, int, int],
    tta: str,
    sw_batch_size: int,
    overlap: float,
    adaptive_initial: int,
    adaptive_batch: int,
    adaptive_min_rotations: int,
    adaptive_ambiguity_threshold: float,
    adaptive_dice_threshold: float,
    device: str,
):
    rotations = get_rotation_subset(tta)

    if tta != "adaptive":
        accum = None
        per_rotation_seconds = []
        for rot in rotations:
            t0 = time.time()
            prob = infer_one_orientation(model, native_input, rot, patch_size, sw_batch_size, overlap, device)
            per_rotation_seconds.append(time.time() - t0)
            accum = prob if accum is None else accum + prob

        mean_prob = accum / float(len(rotations))
        return mean_prob, {
            "tta_mode": tta,
            "num_rotations": len(rotations),
            "stopped_adaptively": False,
            "ambiguity_fraction": ambiguity_fraction(mean_prob[0, 1]),
            "mean_rotation_seconds": float(np.mean(per_rotation_seconds)) if per_rotation_seconds else 0.0,
        }

    # Adaptive mode.
    max_rots = rotations
    accum = None
    count = 0
    previous_mask = None
    per_rotation_seconds = []
    stopped = False
    final_amb = None
    final_dice = None

    target_first = max(1, adaptive_initial)
    batch_size = max(1, adaptive_batch)
    min_rots = max(1, adaptive_min_rotations)

    while count < len(max_rots):
        if count == 0:
            next_n = min(target_first, len(max_rots))
        else:
            next_n = min(batch_size, len(max_rots) - count)

        for rot in max_rots[count:count + next_n]:
            t0 = time.time()
            prob = infer_one_orientation(model, native_input, rot, patch_size, sw_batch_size, overlap, device)
            per_rotation_seconds.append(time.time() - t0)
            accum = prob if accum is None else accum + prob

        count += next_n
        mean_prob = accum / float(count)
        fg_prob = mean_prob[0, 1]
        current_mask = fg_prob >= 0.5
        final_amb = ambiguity_fraction(fg_prob)

        if previous_mask is None:
            final_dice = None
        else:
            final_dice = dice_between_binary(current_mask, previous_mask)

        # Stop only after at least min_rots and after we have a previous estimate.
        if (
            count >= min_rots
            and final_dice is not None
            and final_amb <= adaptive_ambiguity_threshold
            and final_dice >= adaptive_dice_threshold
        ):
            stopped = True
            break

        previous_mask = current_mask

    mean_prob = accum / float(count)
    return mean_prob, {
        "tta_mode": "adaptive",
        "num_rotations": count,
        "stopped_adaptively": stopped,
        "ambiguity_fraction": float(final_amb if final_amb is not None else ambiguity_fraction(mean_prob[0, 1])),
        "stability_dice": None if final_dice is None else float(final_dice),
        "mean_rotation_seconds": float(np.mean(per_rotation_seconds)) if per_rotation_seconds else 0.0,
    }


# -----------------------------
# Mask postprocessing/output
# -----------------------------

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndi.label(mask.astype(bool))
    if n == 0:
        return mask.astype(np.uint8)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return (labeled == largest).astype(np.uint8)


def voxel_volume_mm3(img: nib.Nifti1Image) -> float:
    zooms = img.header.get_zooms()[:3]
    return float(abs(zooms[0] * zooms[1] * zooms[2]))


def save_mask_like_input(mask: np.ndarray, input_img: nib.Nifti1Image, out_path: str):
    """
    Save mask with input affine/header.

    This intentionally preserves the input affine/header conventions instead of
    canonicalizing. The data array is in original input voxel order.
    """
    hdr = input_img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr["cal_min"] = 0
    hdr["cal_max"] = 1

    out = nib.Nifti1Image(mask.astype(np.uint8), input_img.affine, header=hdr)
    nib.save(out, out_path)


def output_name_for_input(path: str, postfix: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7] + postfix + ".nii.gz"
    if base.endswith(".nii"):
        return base[:-4] + postfix + ".nii.gz"
    return base + postfix + ".nii.gz"


def collect_inputs(inference_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(inference_dir, "*.nii.gz")))
    keep = []
    for f in files:
        b = os.path.basename(f).lower()
        if b.startswith("._"):
            continue
        if "mask" in b or "normcoord" in b or "pred" in b:
            continue
        keep.append(f)
    return keep


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Orientation-agnostic rodent brain extraction inference.")
    parser.add_argument("--model_ckpt_save_dir", required=True, help="Model checkpoint path.")
    parser.add_argument("--inference_dir", required=True, help="Directory containing image NIfTIs.")
    parser.add_argument("--output_dir", default="", help="Output directory. Default: inference_dir/predicted_masks")
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tta", choices=["none", "fast", "medium", "full", "adaptive"], default="adaptive")
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--keep_all_components", action="store_true")
    parser.add_argument("--output_postfix", default="_pred_mask")
    parser.add_argument("--write_prob", action="store_true", help="Also save foreground probability image.")
    parser.add_argument("--min_volume_mm3", type=float, default=0.0, help="Warn if mask volume below this. 0 disables lower warning.")
    parser.add_argument("--max_volume_mm3", type=float, default=0.0, help="Warn if mask volume above this. 0 disables upper warning.")

    # Adaptive TTA controls.
    parser.add_argument("--adaptive_initial", type=int, default=4)
    parser.add_argument("--adaptive_batch", type=int, default=4)
    parser.add_argument("--adaptive_min_rotations", type=int, default=8)
    parser.add_argument("--adaptive_ambiguity_threshold", type=float, default=0.002,
                        help="Stop if fraction of voxels with foreground probability 0.4-0.6 is below this.")
    parser.add_argument("--adaptive_dice_threshold", type=float, default=0.995,
                        help="Stop if hard masks are this stable between TTA batches.")

    args = parser.parse_args()

    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    output_dir = args.output_dir or os.path.join(args.inference_dir, "predicted_masks")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Patch size: {patch_size}")
    print(f"[INFO] TTA mode: {args.tta}")
    print(f"[INFO] Output dir: {output_dir}")
    print("[INFO] Output masks will be saved in original input voxel/header space.")

    model = make_model(patch_size, args.feature_size, args.device)
    model = load_model(model, args.model_ckpt_save_dir, args.device)

    input_files = collect_inputs(args.inference_dir)
    if not input_files:
        raise RuntimeError(f"No valid input NIfTI files found in: {args.inference_dir}")

    metrics_rows = []

    with torch.no_grad():
        for img_path in tqdm(input_files, desc="Images"):
            t_img0 = time.time()
            input_img = nib.load(img_path)
            arr = np.asanyarray(input_img.dataobj).astype(np.float32)

            if arr.ndim != 3:
                print(f"[WARN] Skipping non-3D image: {img_path} shape={arr.shape}")
                continue

            native_input = make_two_channel_tensor(arr, args.device)

            prob, info = run_tta_inference(
                model=model,
                native_input=native_input,
                patch_size=patch_size,
                tta=args.tta,
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
                adaptive_initial=args.adaptive_initial,
                adaptive_batch=args.adaptive_batch,
                adaptive_min_rotations=args.adaptive_min_rotations,
                adaptive_ambiguity_threshold=args.adaptive_ambiguity_threshold,
                adaptive_dice_threshold=args.adaptive_dice_threshold,
                device=args.device,
            )

            fg_prob = prob[0, 1].detach().cpu().numpy().astype(np.float32)
            mask = (fg_prob >= 0.5).astype(np.uint8)

            if not args.keep_all_components:
                mask = keep_largest_component(mask)

            vv = voxel_volume_mm3(input_img)
            mask_volume_mm3 = float(mask.sum()) * vv

            warnings = []
            if args.min_volume_mm3 > 0 and mask_volume_mm3 < args.min_volume_mm3:
                warnings.append(f"below_min_volume_{args.min_volume_mm3:g}")
            if args.max_volume_mm3 > 0 and mask_volume_mm3 > args.max_volume_mm3:
                warnings.append(f"above_max_volume_{args.max_volume_mm3:g}")

            out_name = output_name_for_input(img_path, args.output_postfix)
            out_path = os.path.join(output_dir, out_name)
            save_mask_like_input(mask, input_img, out_path)

            prob_path = ""
            if args.write_prob:
                prob_name = output_name_for_input(img_path, "_fg_prob")
                prob_path = os.path.join(output_dir, prob_name)
                hdr = input_img.header.copy()
                hdr.set_data_dtype(np.float32)
                nib.save(nib.Nifti1Image(fg_prob, input_img.affine, header=hdr), prob_path)

            total_seconds = time.time() - t_img0

            row = {
                "input": img_path,
                "output_mask": out_path,
                "output_prob": prob_path,
                "shape_x": arr.shape[0],
                "shape_y": arr.shape[1],
                "shape_z": arr.shape[2],
                "voxel_volume_mm3": vv,
                "mask_voxels": int(mask.sum()),
                "mask_volume_mm3": mask_volume_mm3,
                "total_seconds": total_seconds,
                "warning": ";".join(warnings),
            }
            row.update(info)
            metrics_rows.append(row)

            if warnings:
                print(f"[WARN] {os.path.basename(img_path)}: {', '.join(warnings)}; volume={mask_volume_mm3:.2f} mm^3")

    csv_path = os.path.join(output_dir, "inference_metrics.csv")
    if metrics_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    print(f"[DONE] Inference complete. Metrics: {csv_path}")


if __name__ == "__main__":
    main()
