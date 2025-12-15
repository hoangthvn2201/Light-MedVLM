from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installation
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True, help="Hugging Face repo id, e.g. user/model")
    p.add_argument("--local_dir", default="lightmedvlm", help="Local folder to store snapshot")
    args = p.parse_args()

    Path(args.local_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir)
    print(f"Downloaded to: {args.local_dir}")


if __name__ == "__main__":
    main()
