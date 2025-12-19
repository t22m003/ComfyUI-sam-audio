# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from typing import Tuple

import torch
import transformers

from sam_audio.model.config import T5EncoderConfig


class T5TextEncoder(torch.nn.Module):
    def __init__(self, cfg: T5EncoderConfig):
        super().__init__()
        self.model = transformers.T5EncoderModel.from_pretrained(cfg.name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.name)
        self.pad_mode = cfg.pad_mode
        self.max_length = cfg.max_length

    def forward(self, texts: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=self.pad_mode,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )["last_hidden_state"]

        return res, attention_mask.bool()
