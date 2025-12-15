from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def extract_pred_text(pred_report: str) -> str:
    """Extract the main report text from model output.

    The original notebook strips everything before 'FINDINGS :' or 'IMPRESSION :'
    (case-insensitive), if present.
    """
    pred_lower = pred_report.lower()

    if "findings :" in pred_lower:
        match = re.search(r"findings :(.*)", pred_report, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else pred_report

    if "impression :" in pred_lower:
        match = re.search(r"impression :(.*)", pred_report, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else pred_report

    return pred_report


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
