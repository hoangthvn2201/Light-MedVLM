import argparse
import os
import json
from typing import Optional

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from huggingface_hub import login
from dotenv import load_dotenv
import wandb
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import get_scheduler, EarlyStoppingCallback, TrainerCallback


class LossLoggingCallback(TrainerCallback):
    """Custom callback to save training and validation losses to a text file."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.loss_file = os.path.join(output_dir, "training_losses.txt")
        self.losses = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(self.loss_file, 'w') as f:
            f.write("Epoch\tStep\tTraining_Loss\tValidation_Loss\tLearning_Rate\n")
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logging occurs during training."""
        if logs is not None:
            current_log = {
                'epoch': state.epoch,
                'step': state.global_step,
                'train_loss': logs.get('train_loss', None),
                'eval_loss': logs.get('eval_loss', None),
                'learning_rate': logs.get('learning_rate', None)
            }
            self.losses.append(current_log)
            with open(self.loss_file, 'a') as f:
                f.write(f"{current_log['epoch']:.2f}\t"
                       f"{current_log['step']}\t"
                       f"{current_log['train_loss'] if current_log['train_loss'] else 'N/A'}\t"
                       f"{current_log['eval_loss'] if current_log['eval_loss'] else 'N/A'}\t"
                       f"{current_log['learning_rate'] if current_log['learning_rate'] else 'N/A'}\n")
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of training to save a summary."""
        summary_file = os.path.join(self.output_dir, "training_summary.json")
        summary = {
            'total_epochs': state.epoch,
            'total_steps': state.global_step,
            'final_train_loss': self.losses[-1]['train_loss'] if self.losses else None,
            'final_eval_loss': self.losses[-1]['eval_loss'] if self.losses else None,
            'best_eval_loss': min([log['eval_loss'] for log in self.losses if log['eval_loss'] is not None], default=None),
            'all_losses': self.losses
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Loss logs saved to: {self.loss_file}")
        print(f"Training summary saved to: {summary_file}")


def formatting_prompts_func(examples, tokenizer):
    """Format the dataset examples into chat template format."""
    questions = examples['Question']
    answers = examples['Answer']
    texts = []
    for question, answer in zip(questions, answers):
        if tokenizer.chat_template is not None:
            prompt = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        else: 
            gemma_template_prompt = "<start_of_turn>user\n{question}<end_of_turn><start_of_turn>model\n{answer}<end_of_turn>"
            text = gemma_template_prompt.format(question=question, answer=answer)
            texts.append(text)
    return {"text": texts}


def load_and_prepare_model(args):
    """Load and prepare the model for training."""
    print(f"Loading model: {args.model_id}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=False if args.use_lora else True,
    )

    if args.use_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,  
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",    
            use_gradient_checkpointing="unsloth", 
            random_state=3407,
            use_rslora=False,   
            loftq_config=None, 
        )
    
    return model, tokenizer


def load_and_prepare_dataset(args, tokenizer):
    """Load and prepare the dataset for training."""
    print("Loading dataset...")
    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
    
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer), 
        batched=True, 
        batch_size=8
    )
    
    dataset_split = dataset.train_test_split(test_size=args.test_size)
    train_ds = dataset_split['train']
    test_ds = dataset_split['test']
    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    
    return train_ds, test_ds


def setup_training_args(args):
    """Setup training configuration."""
    return SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True, 
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        
        optim="adamw_8bit", 
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        seed=args.seed,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        report_to="none" if not args.use_wandb else "wandb",
        run_name=args.run_name if args.use_wandb else None,
        output_dir=args.output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a medical LLM")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="google/gemma-3-270m",
                       help="Hugging Face model ID")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--dtype", type=str, default=None,
                       help="Data type for model weights")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit precision")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=32,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                       help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=5,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=5,
                       help="Logging steps")
    parser.add_argument("--seed", type=int, default=3047,
                       help="Random seed")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                       help="Early stopping patience")
    
    # Dataset arguments
    parser.add_argument("--test_size", type=float, default=0.05,
                       help="Test set size ratio")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./sft_trainer",
                       help="Output directory")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for logging")
    
    # Authentication arguments
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Face token")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                       help="Weights & Biases API key")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                       help="Hub model ID (e.g., username/model-name)")
    
    args = parser.parse_args()
    
    load_dotenv()
    
    if args.hf_token is None:
        args.hf_token = os.getenv('HF_TOKEN')
    if args.wandb_api_key is None:
        args.wandb_api_key = os.getenv('WANDB_API_KEY')
    
    if args.run_name is None:
        model_name = args.model_id.split('/')[-1]
        lora_suffix = "LoRA" if args.use_lora else "Full"
        args.run_name = f"{model_name}_{args.epochs}E_{lora_suffix}_BS{args.batch_size}"
    
    if args.push_to_hub and args.hub_model_id is None:
        args.hub_model_id = f"huyhoangt2201/{args.run_name}"
    
    if args.wandb_api_key and args.use_wandb:
        wandb.login(key=args.wandb_api_key)
        wandb.init(project="medical-llm-finetuning", name=args.run_name)
    
    if args.hf_token:
        login(token=args.hf_token)
    
    model, tokenizer = load_and_prepare_model(args)
    
    train_ds, test_ds = load_and_prepare_dataset(args, tokenizer)

    training_args = setup_training_args(args)
    
    loss_callback = LossLoggingCallback(args.output_dir)
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        loss_callback
    ]
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
        callbacks=callbacks,
    )
    
    print("Starting training...")
    trainer.train()
    
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing model to hub: {args.hub_model_id}")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print("Model successfully pushed to Hugging Face Hub!")
    
    if args.use_wandb:
        wandb.finish()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()