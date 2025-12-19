# Mock torchcodec.decoders module for compatibility
# Provides AudioDecoder and VideoDecoder using torchaudio as backend

import torch
import torchaudio


class AudioDecoder:
    """Mock AudioDecoder using torchaudio"""
    
    def __init__(self, source, sample_rate=None, num_channels=None):
        self.source = source
        self.target_sample_rate = sample_rate
        self.num_channels = num_channels
        self._data = None
    
    def get_all_samples(self):
        if self._data is None:
            waveform, sr = torchaudio.load(self.source)
            if self.target_sample_rate and sr != self.target_sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate)
            if self.num_channels and waveform.shape[0] != self.num_channels:
                if self.num_channels == 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
            self._data = waveform
        
        class Result:
            def __init__(self, data):
                self.data = data
        
        return Result(self._data)


class VideoDecoder:
    """Mock VideoDecoder - basic stub"""
    
    def __init__(self, source, dimension_order="NCHW"):
        self.source = source
        self.dimension_order = dimension_order
        self._frames = None
        self._pts = None
    
    def __len__(self):
        # Return a dummy length
        return 0
    
    def __getitem__(self, idx):
        # Return empty tensor for video frames
        return torch.zeros(1, 3, 224, 224)
    
    def get_frames_in_range(self, start, end):
        class Result:
            def __init__(self):
                self.data = torch.zeros(1, 3, 224, 224)
                self.pts_seconds = torch.zeros(1)
        return Result()


__all__ = ["AudioDecoder", "VideoDecoder"]
