# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import torch

from ..model.config import JudgeRankerConfig
from ..model.judge import SAMAudioJudgeModel
from ..processor import SAMAudioJudgeProcessor
from .ranker import Ranker


class JudgeRanker(Ranker):
    def __init__(self, config: JudgeRankerConfig):
        super().__init__()
        self.config = config
        self.model = SAMAudioJudgeModel.from_pretrained(config.checkpoint_or_model_id)
        self.processor = SAMAudioJudgeProcessor.from_pretrained(
            config.checkpoint_or_model_id
        )

    @torch.inference_mode()
    def forward(
        self,
        input_audio: list[torch.Tensor],
        extracted_audio: list[torch.Tensor],
        descriptions: list[str],
        sample_rate: int = 48_000,
        **kwargs,
    ):
        bsz, ncandidates = len(input_audio), len(input_audio[0])
        input_seqs = [x[None] for candidates in input_audio for x in candidates]
        extracted_seqs = [x[None] for candidates in extracted_audio for x in candidates]
        repeated_descriptions = [x for x in descriptions for _ in range(ncandidates)]
        processed = self.processor(
            text=repeated_descriptions,
            input_audio=input_seqs,
            separated_audio=extracted_seqs,
            return_tensors="pt",
            padding=True,
            sampling_rate=sample_rate,
        )
        res = self.model(**processed.to(input_audio[0].device))
        return res.overall.view(bsz, ncandidates)
