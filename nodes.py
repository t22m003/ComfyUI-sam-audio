# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# ComfyUI Custom Node for SAM-Audio

import os
import sys

# Add sam_audio module from this custom node directory to path
_sam_audio_path = os.path.dirname(__file__)
if _sam_audio_path not in sys.path:
    sys.path.insert(0, _sam_audio_path)

# Inject torchcodec.decoders shim before importing sam_audio
# This is needed because the installed torchcodec doesn't have the decoders module
import types
import importlib.util
_shim_path = os.path.join(os.path.dirname(__file__), "torchcodec_shim.py")
_shim_spec = importlib.util.spec_from_file_location("torchcodec_shim", _shim_path)
torchcodec_shim = importlib.util.module_from_spec(_shim_spec)
_shim_spec.loader.exec_module(torchcodec_shim)

if "torchcodec" not in sys.modules:
    torchcodec_module = types.ModuleType("torchcodec")
    torchcodec_module.__spec__ = importlib.util.spec_from_loader("torchcodec", loader=None)
    sys.modules["torchcodec"] = torchcodec_module
if "torchcodec.decoders" not in sys.modules:
    torchcodec_shim.__spec__ = importlib.util.spec_from_loader("torchcodec.decoders", loader=None)
    sys.modules["torchcodec.decoders"] = torchcodec_shim

import torch
import torchaudio
import folder_paths

from sam_audio import SAMAudio, SAMAudioProcessor



# Register sam_audio folder for model directories
SAM_AUDIO_FOLDER = "sam_audio"
folder_paths.folder_names_and_paths[SAM_AUDIO_FOLDER] = (
    [os.path.join(folder_paths.models_dir, SAM_AUDIO_FOLDER)],
    set()  # No specific extensions, we're looking for directories
)


def get_sam_audio_models():
    """Get list of SAM-Audio model directories and HuggingFace model names"""
    models = []
    
    # Add HuggingFace models
    hf_models = [
        "facebook/sam-audio-small",
        "facebook/sam-audio-base",
        "facebook/sam-audio-large",
        "facebook/sam-audio-small-tv",
        "facebook/sam-audio-base-tv",
        "facebook/sam-audio-large-tv",
    ]
    models.extend(hf_models)
    
    # Add local model directories from sam_audio folder
    sam_audio_dir = os.path.join(folder_paths.models_dir, SAM_AUDIO_FOLDER)
    if os.path.exists(sam_audio_dir):
        for item in os.listdir(sam_audio_dir):
            item_path = os.path.join(sam_audio_dir, item)
            # Check if it's a valid SAM-Audio model directory (has config.json and checkpoint.pt)
            if os.path.isdir(item_path):
                config_path = os.path.join(item_path, "config.json")
                checkpoint_path = os.path.join(item_path, "checkpoint.pt")
                if os.path.exists(config_path) and os.path.exists(checkpoint_path):
                    models.append(f"local/{item}")
    
    return models


class SAMAudioModelLoader:
    """Load SAM-Audio model and processor from HuggingFace or local sam_audio directory"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_sam_audio_models(), {"default": "facebook/sam-audio-large"}),
            }
        }
    
    RETURN_TYPES = ("SAM_AUDIO_MODEL", "SAM_AUDIO_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "audio/sam-audio"

    def load_model(self, model_name):
        if model_name.startswith("local/"):
            # Load from local sam_audio directory
            local_name = model_name[6:]  # Remove "local/" prefix
            model_path = os.path.join(folder_paths.models_dir, SAM_AUDIO_FOLDER, local_name)
        else:
            # Load from HuggingFace
            model_path = model_name
        
        model = SAMAudio.from_pretrained(model_path)
        processor = SAMAudioProcessor.from_pretrained(model_path)
        model = model.eval().cuda()
        return (model, processor)


class SAMAudioTextSeparate:
    """Separate audio using text description prompt"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "description": ("STRING", {
                    "default": "A person speaking",
                    "multiline": True,
                }),
                "model": ("SAM_AUDIO_MODEL",),
                "processor": ("SAM_AUDIO_PROCESSOR",),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("target", "residual")
    FUNCTION = "separate"
    CATEGORY = "audio/sam-audio"

    def separate(self, audio, description, model, processor):
        # audio is dict with 'waveform' and 'sample_rate'
        waveform = audio["waveform"]  # (batch, channels, samples)
        sample_rate = audio["sample_rate"]

        # Resample if needed
        if sample_rate != processor.audio_sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, processor.audio_sampling_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        
        # Remove batch dim for processing (take first item)
        audio_tensor = waveform[0]  # (channels, samples)

        batch = processor(
            audios=[audio_tensor],
            descriptions=[description],
        ).to("cuda")

        result = model.separate(batch)

        # result.target and result.residual are lists (batch results)
        target_tensor = result.target[0] if isinstance(result.target, list) else result.target
        residual_tensor = result.residual[0] if isinstance(result.residual, list) else result.residual
        
        target_audio = {
            "waveform": target_tensor.cpu().unsqueeze(0).unsqueeze(0),  # Add batch and channel dims back
            "sample_rate": processor.audio_sampling_rate,
        }
        residual_audio = {
            "waveform": residual_tensor.cpu().unsqueeze(0).unsqueeze(0),
            "sample_rate": processor.audio_sampling_rate,
        }

        return (target_audio, residual_audio)


class SAMAudioVisualSeparate:
    """Separate audio using visual prompt (video + mask)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("IMAGE",),  # Video frames as IMAGE sequence
                "mask": ("MASK",),
                "model": ("SAM_AUDIO_MODEL",),
                "processor": ("SAM_AUDIO_PROCESSOR",),
            },
            "optional": {
                "audio": ("AUDIO",),  # Optional: if not provided, extract from video
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("target", "residual")
    FUNCTION = "separate"
    CATEGORY = "audio/sam-audio"

    def separate(self, video, mask, model, processor, audio=None):
        # video is (frames, height, width, channels) as IMAGE
        # mask is (frames, height, width)
        
        # Process masked video
        masked_video = processor.mask_videos([video], [mask])

        if audio is not None:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if sample_rate != processor.audio_sampling_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, processor.audio_sampling_rate)
                waveform = resampler(waveform)
            
            if waveform.shape[1] > 1:
                waveform = waveform.mean(dim=1, keepdim=True)
            
            audio_tensor = waveform[0]
        else:
            # Use video path for audio extraction (requires video to be path)
            audio_tensor = video

        batch = processor(
            audios=[audio_tensor],
            descriptions=[""],  # Empty description for visual prompting
            masked_videos=masked_video,
        ).to("cuda")

        result = model.separate(batch)

        target_tensor = result.target[0] if isinstance(result.target, list) else result.target
        residual_tensor = result.residual[0] if isinstance(result.residual, list) else result.residual
        
        target_audio = {
            "waveform": target_tensor.cpu().unsqueeze(0).unsqueeze(0),
            "sample_rate": processor.audio_sampling_rate,
        }
        residual_audio = {
            "waveform": residual_tensor.cpu().unsqueeze(0).unsqueeze(0),
            "sample_rate": processor.audio_sampling_rate,
        }

        return (target_audio, residual_audio)


class SAMAudioSpanSeparate:
    """Separate audio using time span prompt"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "description": ("STRING", {
                    "default": "A horn honking",
                    "multiline": True,
                }),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "display": "number",
                }),
                "end_time": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "display": "number",
                }),
                "model": ("SAM_AUDIO_MODEL",),
                "processor": ("SAM_AUDIO_PROCESSOR",),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("target", "residual")
    FUNCTION = "separate"
    CATEGORY = "audio/sam-audio"

    def separate(self, audio, description, start_time, end_time, model, processor):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if sample_rate != processor.audio_sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, processor.audio_sampling_rate)
            waveform = resampler(waveform)

        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        
        audio_tensor = waveform[0]

        # Create anchor with positive span
        anchors = [[["+", start_time, end_time]]]

        batch = processor(
            audios=[audio_tensor],
            descriptions=[description],
            anchors=anchors,
        ).to("cuda")

        result = model.separate(batch)

        target_tensor = result.target[0] if isinstance(result.target, list) else result.target
        residual_tensor = result.residual[0] if isinstance(result.residual, list) else result.residual
        
        target_audio = {
            "waveform": target_tensor.cpu().unsqueeze(0).unsqueeze(0),
            "sample_rate": processor.audio_sampling_rate,
        }
        residual_audio = {
            "waveform": residual_tensor.cpu().unsqueeze(0).unsqueeze(0),
            "sample_rate": processor.audio_sampling_rate,
        }

        return (target_audio, residual_audio)


class SaveAudio:
    """Save audio to file"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "sam_audio_output"}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_audio"
    CATEGORY = "audio/sam-audio"

    def save_audio(self, audio, filename_prefix):
        output_dir = folder_paths.get_output_directory()
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Find unique filename
        counter = 0
        while True:
            filename = f"{filename_prefix}_{counter:05d}.wav"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1

        # Save audio (remove batch dim if present)
        if waveform.dim() == 3:
            waveform = waveform[0]
        torchaudio.save(filepath, waveform, sample_rate)

        return {}


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SAMAudioModelLoader": SAMAudioModelLoader,
    "SAMAudioTextSeparate": SAMAudioTextSeparate,
    "SAMAudioVisualSeparate": SAMAudioVisualSeparate,
    "SAMAudioSpanSeparate": SAMAudioSpanSeparate,
    "SaveAudio": SaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMAudioModelLoader": "SAM-Audio Model Loader",
    "SAMAudioTextSeparate": "SAM-Audio Text Separate",
    "SAMAudioVisualSeparate": "SAM-Audio Visual Separate",
    "SAMAudioSpanSeparate": "SAM-Audio Span Separate",
    "SaveAudio": "Save Audio",
}
