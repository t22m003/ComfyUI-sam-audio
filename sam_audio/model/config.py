# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from typing import Optional, Tuple

import numpy as np
from core.audio_visual_encoder.config import TransformerConfig as PEAVTransformerConfig
from transformers import ModernBertConfig


class DACVAEConfig:
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list[int] = [2, 8, 10, 12],
        latent_dim: int = 1024,
        decoder_dim: int = 1536,
        decoder_rates: list[int] = [12, 10, 8, 2],
        n_codebooks: int = 16,
        codebook_size: int = 1024,
        codebook_dim: int = 128,
        quantizer_dropout: bool = False,
        sample_rate: int = 48_000,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.sample_rate = sample_rate
        self.mean = mean
        self.std = std

    @property
    def hop_length(self):
        return int(np.prod(self.encoder_rates))


class TextEncoderConfig:
    def __init__(self, dim: int = 768):
        self.dim = dim


class T5EncoderConfig(TextEncoderConfig):
    def __init__(
        self,
        name: str = "t5-base",
        max_length: Optional[int] = 512,
        pad_mode: str = "longest",
        dim: int = 768,
    ):
        super().__init__(dim=dim)
        self.name = name
        self.max_length = max_length
        self.pad_mode = pad_mode


class VisionEncoderConfig:
    def __init__(self, dim: int = 1024, batch_size: int = 300):
        self.dim = dim
        self.batch_size = batch_size


class PerceptionEncoderConfig(VisionEncoderConfig):
    def __init__(
        self,
        dim: int = 1024,
        batch_size: int = 300,
        name: str = "PE-Core-L14-336",
        normalize_feature: bool = True,
        interpolation_mode: str = "BICUBIC",
        image_size: int = 336,
    ):
        super().__init__(dim=dim, batch_size=batch_size)
        self.name = name
        self.normalize_feature = normalize_feature
        self.interpolation_mode = interpolation_mode
        self.image_size = image_size


class TransformerConfig:
    def __init__(
        self,
        dim: int = 2048,
        n_heads: int = 16,
        n_layers: int = 16,
        dropout: float = 0.1,
        norm_eps: float = 1.0e-05,
        qk_norm: bool = True,
        fc_bias: bool = False,
        ffn_exp: int = 4,
        ffn_dim_multiplier: int = 1,
        multiple_of: int = 64,
        non_linearity: str = "swiglu",
        use_rope: bool = True,
        max_positions: int = 10000,
        frequency_embedding_dim: int = 256,
        timestep_non_linearity: str = "swiglu",
        t_block_non_linearity: str = "silu",
        t_block_bias: bool = True,
        context_dim: int = 2048,
        context_non_linearity: str = "swiglu",
        context_embedder_dropout: float = 0.0,
        context_norm: bool = False,
        out_channels: int = 256,
        in_channels: Optional[int] = None,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.fc_bias = fc_bias
        self.ffn_exp = ffn_exp
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.non_linearity = non_linearity
        self.use_rope = use_rope
        self.max_positions = max_positions
        self.frequency_embedding_dim = frequency_embedding_dim
        self.timestep_non_linearity = timestep_non_linearity
        self.t_block_non_linearity = t_block_non_linearity
        self.t_block_bias = t_block_bias
        self.context_dim = context_dim
        self.context_non_linearity = context_non_linearity
        self.context_embedder_dropout = context_embedder_dropout
        self.context_norm = context_norm
        self.out_channels = out_channels
        self.in_channels = in_channels


class RankerConfig:
    kind: str


class ImageBindRankerConfig(RankerConfig):
    kind: str = "imagebind"

    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint


class ClapRankerConfig(RankerConfig):
    kind: str = "clap"

    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint


class JudgeRankerConfig(RankerConfig):
    kind: str = "judge"

    def __init__(self, checkpoint_or_model_id: str = "facebook/sam-audio-judge"):
        self.checkpoint_or_model_id = checkpoint_or_model_id


class SoundActivityRankerConfig(RankerConfig):
    kind: str = "sound_activity"

    def __init__(
        self,
        threshold_mode: str = "rel_to_max",
        sil_threshold: float = -40,
        metric: str = "iou",
    ):
        self.threshold_mode = threshold_mode
        self.sil_threshold = sil_threshold
        self.metric = metric


class EnsembleRankerConfig(RankerConfig):
    kind: str = "ensemble"

    def __init__(self, rankers: dict[str, Tuple[RankerConfig, float]]):
        self.rankers = rankers


def parse_ranker_config(config_dict: dict):
    kind = config_dict.pop("kind")
    match kind:
        case ImageBindRankerConfig.kind:
            return ImageBindRankerConfig(**config_dict)
        case ClapRankerConfig.kind:
            return ClapRankerConfig(**config_dict)
        case JudgeRankerConfig.kind:
            return JudgeRankerConfig(**config_dict)
        case SoundActivityRankerConfig.kind:
            return SoundActivityRankerConfig(**config_dict)
        case EnsembleRankerConfig.kind:
            return EnsembleRankerConfig(
                {
                    k: (parse_ranker_config(v), w)
                    for k, (v, w) in config_dict["rankers"].items()
                }
            )


class SAMAudioConfig:
    def __init__(
        self,
        in_channels: int = 768,
        audio_codec=None,
        text_encoder=None,
        vision_encoder=None,
        transformer=None,
        num_anchors: int = 3,
        anchor_embedding_dim: int = 128,
        visual_ranker=None,
        text_ranker=None,
        span_predictor: Optional[str] = "pe-a-frame-large",
    ):
        self.in_channels = in_channels
        self.audio_codec = DACVAEConfig(**(audio_codec or {}))
        self.text_encoder = T5EncoderConfig(**(text_encoder or {}))
        self.vision_encoder = PerceptionEncoderConfig(**(vision_encoder or {}))
        self.transformer = TransformerConfig(**(transformer or {}))
        self.num_anchors = num_anchors
        self.anchor_embedding_dim = anchor_embedding_dim
        self.visual_ranker = (
            None if visual_ranker is None else parse_ranker_config(visual_ranker)
        )
        self.text_ranker = (
            None if text_ranker is None else parse_ranker_config(text_ranker)
        )
        self.span_predictor = span_predictor


class SAMAudioJudgeConfig:
    def __init__(
        self,
        audio_codec: DACVAEConfig = None,
        transformer: PEAVTransformerConfig = None,
        text_model: ModernBertConfig = None,
        finetune_transformer: PEAVTransformerConfig = None,
        nth_text_layer: int = 22,
        bottleneck_dim: int = 256,
    ):
        self.audio_codec = DACVAEConfig(**(audio_codec or {}))
        self.transformer = PEAVTransformerConfig(**(transformer or {}))
        self.text_model = ModernBertConfig(**(text_model or {}))
        self.finetune_transformer = PEAVTransformerConfig(
            **(finetune_transformer or {})
        )
        self.nth_text_layer = nth_text_layer
        self.bottleneck_dim = bottleneck_dim
