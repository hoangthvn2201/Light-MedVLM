from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installation
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

from lightmedvlm import compute_report_metrics, extract_pred_text, load_json, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred_json", required=True, help="Predictions JSON (list with gt_report/pred_report)")
    p.add_argument("--apply_extract_pred_text", action="store_true", help="Create pred_after via extract_pred_text and evaluate on it")
    p.add_argument("--save_fixed_json", default=None, help="If set, write a copy of JSON with pred_after field")
    args = p.parse_args()

    preds = load_json(args.pred_json)
    assert isinstance(preds, list), "pred_json must be a list"

    gt_list = []
    pred_list = []

    for item in preds:
        gt = item.get("gt_report", "")
        pred = item.get("pred_report", "")

        if args.apply_extract_pred_text:
            pred_after = extract_pred_text(pred)
            item["pred_after"] = pred_after
            pred = pred_after

        gt_list.append(gt)
        pred_list.append(pred)

    if args.save_fixed_json:
        save_json(preds, args.save_fixed_json)

    m = compute_report_metrics(gt_list=gt_list, pred_list=pred_list)
    for k, v in m.as_dict().items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
