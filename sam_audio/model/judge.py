# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from dataclasses import dataclass
from typing import Optional

import torch
from core.audio_visual_encoder.transformer import BaseModelOutputWithPooling
from core.audio_visual_encoder.transformer import Transformer as PEAVTransformer
from transformers import AutoModel

from .base import BaseModel
from .codec import DACVAEEncoder
from .config import SAMAudioJudgeConfig


@dataclass
class SAMAudioJudgeOutput:
    r"""
    overall (torch.Tensor, optional): Overall score tensor of shape (batch_size, 1).
    recall (torch.Tensor, optional): Recall score tensor of shape (batch_size, 1).
    precision (torch.Tensor, optional): Precision score tensor of shape (batch_size, 1).
    faithfulness (torch.Tensor, optional): Faithfulness score tensor of shape (batch_size, 1).
    text_model_output (BaseModelOutputWithPooling): Output from the text model.
    audio_model_output (BaseModelOutputWithPooling): Output from the audio model.
    """

    overall: Optional[torch.Tensor] = None
    recall: Optional[torch.Tensor] = None
    precision: Optional[torch.Tensor] = None
    faithfulness: Optional[torch.Tensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None


class SAMAudioJudgeModel(BaseModel):
    config_cls = SAMAudioJudgeConfig
    revision = "sam_audio"

    def __init__(self, config: SAMAudioJudgeConfig):
        super().__init__()
        self.config = config
        self.data_proj = torch.nn.Linear(
            config.audio_codec.codebook_dim, config.transformer.hidden_size
        )
        self.audio_codec = DACVAEEncoder(config.audio_codec)
        self.transformer = PEAVTransformer(config.transformer)
        self.finetune_transformer = PEAVTransformer(config.finetune_transformer)
        self.text_model = AutoModel.from_config(config.text_model)
        self.cat_audio_proj = torch.nn.Linear(
            2 * config.transformer.hidden_size, config.bottleneck_dim
        )
        self.text_proj1 = torch.nn.Linear(
            in_features=config.text_model.hidden_size,
            out_features=config.transformer.hidden_size,
            bias=False,
        )
        self.text_proj2 = torch.nn.Linear(
            in_features=config.transformer.hidden_size,
            out_features=config.bottleneck_dim,
        )
        self.layer_norm = torch.nn.LayerNorm(config.bottleneck_dim)
        self.proj_audio_and_text = torch.nn.Linear(
            2 * config.bottleneck_dim, config.bottleneck_dim
        )
        self.finetune_data_proj = torch.nn.Linear(
            config.bottleneck_dim, config.finetune_transformer.hidden_size
        )
        self.head = torch.nn.Linear(
            config.finetune_transformer.hidden_size, 4, bias=False
        )
        self.mean = torch.nn.Parameter(torch.zeros(4, requires_grad=False))
        self.std = torch.nn.Parameter(torch.ones(4, requires_grad=False))

    def _get_text_output(self, input_ids, attention_mask):
        nth_layer = self.config.nth_text_layer
        output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=nth_layer is not None,
        )
        if nth_layer is None:
            text_model_output = output.last_hidden_state
        else:
            text_model_output = output.hidden_states[nth_layer]

        return BaseModelOutputWithPooling(
            last_hidden_state=text_model_output, pooler_output=text_model_output[:, 0]
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # tokenized text
        input_values: torch.Tensor,  # input audio waveform
        separated_values: torch.Tensor,  # separated audio waveform
        attention_mask: Optional[torch.Tensor] = None,  # text attention mask
        padding_mask: Optional[torch.Tensor] = None,  # audio padding mask
    ) -> SAMAudioJudgeOutput:
        text_features = self.text_proj1(
            self._get_text_output(input_ids, attention_mask).pooler_output
        )
        stacked_audios = torch.cat([input_values, separated_values], dim=0)
        stacked_codec_features = self.audio_codec(stacked_audios)
        feature_padding_mask = None
        if padding_mask is not None:
            feature_padding_mask = padding_mask[
                :, :: self.config.audio_codec.hop_length
            ]
        stacked_features = self.transformer(
            self.data_proj(stacked_codec_features.transpose(1, 2)),
            padding_mask=feature_padding_mask,
        )
        input_features, hyp_features = stacked_features.last_hidden_state.chunk(2, 0)
        audio_features = self.cat_audio_proj(
            torch.cat([hyp_features, input_features], dim=2)
        )
        expanded_text = (
            self.layer_norm(self.text_proj2(text_features))
            .unsqueeze(1)
            .expand_as(audio_features)
        )
        audio_and_text = self.proj_audio_and_text(
            torch.cat([audio_features, expanded_text], dim=2)
        )
        finetune_transformer_output = self.finetune_transformer(
            self.finetune_data_proj(audio_and_text), padding_mask=feature_padding_mask
        )
        result = self.head(finetune_transformer_output.last_hidden_state)
        if feature_padding_mask is not None:
            feature_padding_mask = feature_padding_mask.unsqueeze(-1)
        pooled = torch.masked.mean(result, mask=feature_padding_mask, dim=1)
        de_normalized = pooled * self.std + self.mean
        return SAMAudioJudgeOutput(*de_normalized.chunk(4, dim=1))


__all__ = ["SAMAudioJudgeModel", "SAMAudioJudgeOutput"]
