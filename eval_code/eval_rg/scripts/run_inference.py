from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installation
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from lightmedvlm import extract_pred_text, load_json, save_json, load_model_from_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images_root", required=True, help="Root folder that contains the images")
    p.add_argument("--annot_json", required=True, help="Annotation JSON (list of samples with image_path/report)")
    p.add_argument("--out_json", required=True, help="Where to write predictions JSON")
    p.add_argument("--ckpt_path", default=None, help="Checkpoint file (optional). If omitted, auto-picks newest in --checkpoints_dir")
    p.add_argument("--checkpoints_dir", default="lightmedvlm/checkpoints", help="Directory containing checkpoints")
    p.add_argument("--vision_model", default="microsoft/swin-base-patch4-window7-224")
    p.add_argument("--llm_model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--clean_pred", action="store_true", help="Apply extract_pred_text() to model output before saving")
    args = p.parse_args()

    model = load_model_from_checkpoint(
        ckpt_path=args.ckpt_path,
        checkpoints_dir=args.checkpoints_dir,
        vision_model=args.vision_model,
        llm_model=args.llm_model,
        device=args.device,
    )

    data = load_json(args.annot_json)
    assert isinstance(data, list), "Annotation JSON must be a list of samples"

    results = []
    for item in tqdm(data, desc="Infer"):
        img_rel_paths: List[str] = item["image_path"]
        img_paths = [os.path.join(args.images_root, p) for p in img_rel_paths]

        pred = model.inference(img_paths)[0]
        if args.clean_pred:
            pred = extract_pred_text(pred)

        results.append(
            {
                "id": item.get("id"),
                "gt_report": item.get("report") or item.get("gt_report") or "",
                "pred_report": pred,
            }
        )

    save_json(results, args.out_json)
    print(f"Wrote predictions: {args.out_json}")


if __name__ == "__main__":
    main()
