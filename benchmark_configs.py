#!/usr/bin/env python3
"""
benchmark_configs.py

Small helper for comparing patch size and TTA settings.

This script does NOT train models for you. It runs inference using existing
checkpoints and, if ground-truth masks are supplied, computes simple quality
metrics.

Use it after training separate checkpoints, e.g. one for 96 and one for 128.

Example:
  python benchmark_configs.py \
    --images /path/to/images \
    --masks /path/to/masks \
    --model96 ./checkpoints/best_96.pt \
    --model128 ./checkpoints/best_128.pt \
    --output_csv benchmark_results.csv \
    --tta_modes fast adaptive full
"""

import argparse
import csv
import os
import subprocess
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice(a, b, eps=1e-8):
    a = a.astype(bool)
    b = b.astype(bool)
    return float((2 * np.logical_and(a, b).sum() + eps) / (a.sum() + b.sum() + eps))


def simple_hd_vox(a, b):
    """
    Lightweight Hausdorff in voxel units using foreground coordinates.
    Not HD95. Intended for quick benchmarking only.
    """
    ca = np.argwhere(a.astype(bool))
    cb = np.argwhere(b.astype(bool))
    if ca.size == 0 or cb.size == 0:
        return float("nan")
    return float(max(directed_hausdorff(ca, cb)[0], directed_hausdorff(cb, ca)[0]))


def image_id(path):
    b = os.path.basename(path)
    if b.endswith(".nii.gz"):
        b = b[:-7]
    elif b.endswith(".nii"):
        b = b[:-4]
    return b.replace("_T2", "")


def find_mask_for_image(img, masks):
    iid = image_id(img)
    matches = [m for m in masks if iid in os.path.basename(m) and "mask" in os.path.basename(m).lower()]
    return matches[0] if matches else ""


def collect_images(d):
    return sorted([
        os.path.join(d, f) for f in os.listdir(d)
        if f.endswith(".nii.gz")
        and "mask" not in f.lower()
        and "pred" not in f.lower()
        and "normcoord" not in f.lower()
        and not f.startswith("._")
    ])


def collect_masks(d):
    return sorted([
        os.path.join(d, f) for f in os.listdir(d)
        if f.endswith(".nii.gz")
        and "mask" in f.lower()
        and not f.startswith("._")
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--masks", default="", help="Optional GT mask dir.")
    p.add_argument("--model96", default="")
    p.add_argument("--model128", default="")
    p.add_argument("--inference_script", default="inference.py")
    p.add_argument("--tta_modes", nargs="+", default=["fast", "adaptive", "full"])
    p.add_argument("--output_csv", default="benchmark_results.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_images", type=int, default=0)
    args = p.parse_args()

    configs = []
    if args.model96:
        configs.append((96, args.model96))
    if args.model128:
        configs.append((128, args.model128))

    if not configs:
        raise RuntimeError("Provide at least --model96 or --model128")

    images = collect_images(args.images)
    if args.max_images > 0:
        images = images[:args.max_images]

    masks = collect_masks(args.masks) if args.masks else []

    rows = []

    with tempfile.TemporaryDirectory(prefix="oa_mbe_bench_") as tmp:
        tmp = Path(tmp)

        for patch_size, model in configs:
            for tta in args.tta_modes:
                outdir = tmp / f"ps{patch_size}_{tta}"
                outdir.mkdir(parents=True, exist_ok=True)

                t0 = time.time()
                cmd = [
                    "python", args.inference_script,
                    "--model_ckpt_save_dir", model,
                    "--inference_dir", args.images,
                    "--output_dir", str(outdir),
                    "--patch_size", str(patch_size),
                    "--tta", tta,
                    "--device", args.device,
                ]
                print("[RUN]", " ".join(cmd))
                proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                elapsed = time.time() - t0

                if proc.returncode != 0:
                    rows.append({
                        "patch_size": patch_size,
                        "tta": tta,
                        "status": "failed",
                        "seconds_total": elapsed,
                        "stdout_tail": proc.stdout[-2000:],
                    })
                    continue

                pred_files = collect_masks(str(outdir))
                pred_by_id = {image_id(f).replace("_pred_mask", ""): f for f in pred_files}

                if masks:
                    dices = []
                    hds = []
                    for img in images:
                        gt = find_mask_for_image(img, masks)
                        pred = pred_by_id.get(image_id(img), "")
                        if not gt or not pred:
                            continue
                        gt_arr = np.asanyarray(nib.load(gt).dataobj) > 0
                        pred_arr = np.asanyarray(nib.load(pred).dataobj) > 0
                        if gt_arr.shape != pred_arr.shape:
                            print(f"[WARN] shape mismatch for {img}: gt={gt_arr.shape}, pred={pred_arr.shape}")
                            continue
                        dices.append(dice(pred_arr, gt_arr))
                        hds.append(simple_hd_vox(pred_arr, gt_arr))

                    rows.append({
                        "patch_size": patch_size,
                        "tta": tta,
                        "status": "ok",
                        "seconds_total": elapsed,
                        "seconds_per_image": elapsed / max(len(images), 1),
                        "mean_dice": float(np.mean(dices)) if dices else "",
                        "mean_hd_vox": float(np.nanmean(hds)) if hds else "",
                        "n_eval": len(dices),
                    })
                else:
                    rows.append({
                        "patch_size": patch_size,
                        "tta": tta,
                        "status": "ok",
                        "seconds_total": elapsed,
                        "seconds_per_image": elapsed / max(len(images), 1),
                        "mean_dice": "",
                        "mean_hd_vox": "",
                        "n_eval": 0,
                    })

    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
