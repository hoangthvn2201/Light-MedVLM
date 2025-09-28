# Light-MedVLM

## Phase 0: Instruction Finetuning LLM

```bash
python train.py \
    --model_id google/gemma-3-270m \
    --epochs 1 \
    --batch_size 1 \
    --use_lora \
    --push_to_hub \
    --hub_model_id your-username/medical-gemma
```