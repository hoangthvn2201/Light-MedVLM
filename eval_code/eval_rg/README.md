# lightmedvlm_eval

Converted from `eval-capstone-3.ipynb` into a runnable folder of Python modules and scripts.

## Structure

- `lightmedvlm/`
  - `model.py`: `LightMedVLM` + `MLP`
  - `checkpoint.py`: checkpoint-safe loading helpers
  - `utils.py`: JSON helpers + `extract_pred_text`
  - `metrics.py`: BLEU/METEOR/ROUGE/BERTScore computation
- `scripts/`
  - `download_model.py`: `snapshot_download` wrapper
  - `run_inference.py`: run model inference on an annotation JSON
  - `evaluate.py`: compute metrics from a predictions JSON
- `requirements.txt`

## Typical workflow

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Download model snapshot (optional)
```bash
python scripts/download_model.py \
  --repo_id huyhoangt2201/lightmedvlm-mimic-phase1-1epochs-reduced \
  --local_dir lightmedvlm
```

### 3) Run inference (IU-Xray style JSON)
Your annotation JSON is expected to be a list with fields:
- `id`
- `image_path`: list of relative image paths
- `report`: ground-truth report text

```bash
python scripts/run_inference.py \
  --images_root /path/to/iu_images \
  --annot_json /path/to/iuxray_test.json \
  --out_json iu_predictions.json \
  --checkpoints_dir lightmedvlm/checkpoints \
  --device cuda \
  --clean_pred
```

If you already know the exact checkpoint file:
```bash
python scripts/run_inference.py --ckpt_path /path/to/checkpoint.ckpt ...
```

### 4) Evaluate predictions
```bash
python scripts/evaluate.py --pred_json iu_predictions.json --apply_extract_pred_text
```
Optionally save a fixed copy (adds `pred_after`):
```bash
python scripts/evaluate.py --pred_json iu_predictions.json --apply_extract_pred_text --save_fixed_json iu_predictions_fixed.json
```
