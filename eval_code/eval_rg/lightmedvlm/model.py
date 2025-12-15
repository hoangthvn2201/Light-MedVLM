from __future__ import annotations

import re
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    SwinModel,
)
from peft import get_peft_model, LoraConfig, TaskType


class MLP(nn.Module):
    def __init__(self, in_dim: int, inter_dim: int, out_dim: int):
        super().__init__()
        self.hidden_1 = nn.Linear(in_dim, inter_dim)
        self.act = nn.GELU()
        self.hidden_2 = nn.Linear(inter_dim, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.hidden_1(x))
        x = self.dropout(x)
        return self.hidden_2(x)


class LightMedVLM(pl.LightningModule):
    """LightMedVLM LightningModule used for inference in the original notebook.

    Notes:
    - This module focuses on forward + inference; no training step is included.
    - Designed to be loaded from a Lightning checkpoint via `load_from_checkpoint`.
    """

    def __init__(
        self,
        vision_model: str = "microsoft/swin-base-patch4-window7-224",
        llm_model: str = "Qwen/Qwen3-0.6B",
        vis_use_lora: bool = False,
        vis_r: int = 8,
        vis_alpha: int = 16,
        freeze_vm: bool = True,
        llm_use_lora: bool = False,
        llm_r: int = 8,
        llm_alpha: int = 16,
        lora_dropout: float = 0.05,
        low_resource: bool = False,
        max_length: int = 256,
    ):
        super().__init__()

        self.vision_model = vision_model
        self.llm_model = llm_model
        self.vis_use_lora = vis_use_lora
        self.vis_r = vis_r
        self.vis_alpha = vis_alpha
        self.freeze_vm = freeze_vm
        self.llm_use_lora = llm_use_lora
        self.llm_r = llm_r
        self.llm_alpha = llm_alpha
        self.lora_dropout = lora_dropout
        self.low_resource = low_resource
        self.max_length = max_length

        # --- vision encoder ---
        print(f"Loading vision encoder: {self.vision_model}")
        self.vision_encoder = SwinModel.from_pretrained(self.vision_model)
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_model)

        if self.vis_use_lora:
            vis_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.vis_r,
                lora_alpha=self.vis_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["query", "key", "value", "dense"],
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, vis_cfg)

        if self.freeze_vm:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # Projection (Swin hidden size -> LLM hidden size) created later after LLM load.

        # --- LLM ---
        print(f"Loading LLM: {self.llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if self.low_resource else None,
        )

        if self.llm_use_lora:
            llm_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.llm_r,
                lora_alpha=self.llm_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.llm = get_peft_model(self.llm, llm_cfg)

        # Create projector after we know hidden sizes
        vis_hidden = self.vision_encoder.config.hidden_size
        llm_hidden = self.llm.config.hidden_size
        self.projector = MLP(vis_hidden, llm_hidden, llm_hidden)

    def _parse_image(self, image: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def encode_img(self, images: List[Union[str, Image.Image]]) -> Tuple[torch.Tensor, torch.Tensor]:
        pil_images = [self._parse_image(im) for im in images]
        inputs = self.image_processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            out = self.vision_encoder(pixel_values=pixel_values)
            feats = out.last_hidden_state  # [B, N, C]

        feats = self.projector(feats)  # [B, N, H]
        atts = torch.ones(feats.shape[:-1], dtype=torch.long, device=feats.device)  # [B, N]
        return feats, atts

    def prompt_wrap(self, image_embeds: torch.Tensor, atts_img: torch.Tensor, prompt: str):
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = prompt_tokens["input_ids"].to(self.device)
        attention_mask = prompt_tokens["attention_mask"].to(self.device)

        # Text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concatenate [IMG] + text
        inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_img, attention_mask], dim=1)
        return inputs_embeds, attention_mask

    def forward(self, images: List[Union[str, Image.Image]], prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        img_embeds, atts_img = self.encode_img(images)
        inputs_embeds, attention_mask = self.prompt_wrap(img_embeds, atts_img, prompt)
        return inputs_embeds, attention_mask

    def decode(self, token_ids: torch.Tensor) -> str:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text

    @torch.no_grad()
    def inference(
        self,
        image_paths: List[Union[str, Image.Image]],
        question: str = "Please generate a radiology report for this chest x-ray image.",
        beam_size: int = 3,
        do_sample: bool = False,
        min_new_tokens: int = 5,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
    ) -> List[str]:
        prompt = question.strip()
        inputs_embeds, attention_mask = self.forward(image_paths, prompt)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=beam_size,
            do_sample=do_sample,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        out = [self.decode(i) for i in outputs]
        return out
