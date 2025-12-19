# ComfyUI SAM-Audio

Custom nodes for using SAM-Audio (Segment Anything Model for Audio) in ComfyUI.

## Features

Separate specific sounds from audio using text, visual (video + mask), or time span prompts.

## Nodes

| Node | Description |
|------|-------------|
| **SAM-Audio Model Loader** | Load SAM-Audio model and processor |
| **SAM-Audio Text Separate** | Separate sound by text description (e.g., "A person speaking") |
| **SAM-Audio Visual Separate** | Separate sound associated with visual objects using video and mask |
| **SAM-Audio Span Separate** | Separate sound by specifying time range (start/end seconds) |
| **Save Audio** | Save separated audio to file |

## Installation

### Prerequisites

1. Clone SAM-Audio repository to ComfyUI directory:
```bash
cd /path/to/ComfyUI
git clone https://github.com/facebookresearch/sam-audio.git
```

2. Install dependencies into ComfyUI's Python environment:
```bash
# Activate your ComfyUI virtual environment (e.g., ~/venv/comfy)
source ~/venv/comfy/bin/activate

# Install core dependencies
pip install einops torchdiffeq audiobox_aesthetics huggingface_hub xformers

# Install dacvae
pip install git+https://github.com/facebookresearch/dacvae.git

# Install perception_models (provides core.audio_visual_encoder)
pip install git+https://github.com/facebookresearch/perception_models@unpin-deps --no-deps

# Install ImageBind (required for Visual Ranker)
pip install git+https://github.com/facebookresearch/ImageBind.git

# Install LAION-CLAP (required for Text Ranker)
pip install laion-clap
```

3. Login to HuggingFace:
```bash
huggingface-cli login
```

4. Request access to [SAM-Audio HuggingFace repo](https://huggingface.co/facebook/sam-audio-large)

### Install Custom Node

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/t22m003/ComfyUI-sam-audio.git
```

> **Note**: This custom node requires the `sam-audio` repository to be present at `ComfyUI/sam-audio`. The node imports the `sam_audio` module from this location.

## Usage

### Text Prompting

1. Load model with `SAM-Audio Model Loader`
2. Load audio file with `Load Audio`
3. Connect to `SAM-Audio Text Separate` and describe the sound to isolate
4. Save result with `Save Audio`

### Span Prompting

1. Load model and audio
2. Use `SAM-Audio Span Separate` and specify start/end seconds
3. Set the time range where the target sound occurs
4. Save result

### Visual Prompting

1. Prepare video frames and mask
2. Connect to `SAM-Audio Visual Separate`
3. Sound associated with the masked object will be separated

## Sample Workflows

Sample workflow files are included in the `examples/` directory:

- **[text_separation_workflow.json](examples/text_separation_workflow.json)** - Separate audio using text description
- **[span_separation_workflow.json](examples/span_separation_workflow.json)** - Separate audio by specifying time range
- **[visual_separation_workflow.json](examples/visual_separation_workflow.json)** - Separate audio associated with visual objects (uses `CLIPSeg` from ComfyUI-Essentials and `Video Combine` from VideoHelperSuite)

To use: Load the workflow JSON file in ComfyUI (drag & drop or File → Load).

## Available Models

### HuggingFace Models

The following models are automatically downloaded from HuggingFace:

- `facebook/sam-audio-small`
- `facebook/sam-audio-base`
- `facebook/sam-audio-large` (recommended)
- `facebook/sam-audio-small-tv` (for Visual Prompting)
- `facebook/sam-audio-base-tv`
- `facebook/sam-audio-large-tv`

### Local Models

To use locally downloaded models, place them in the `models/sam_audio/` directory:

```
ComfyUI/
  models/
    sam_audio/
      my-model/          # Model name (any name)
        config.json      # Required
        checkpoint.pt    # Required
```

Once properly placed, the model will be available as `local/my-model` in the node.

> **Note**: Models downloaded from HuggingFace are typically cached in `~/.cache/huggingface/hub/`. You can copy or symlink `models--facebook--sam-audio-large/snapshots/<hash>/` from that directory to `models/sam_audio/sam-audio-large/` for offline usage.

## License

This custom node uses [SAM-Audio](https://github.com/facebookresearch/sam-audio).

**SAM-Audio is subject to Meta's "SAM License". Key restrictions:**

- Must include SAM License when redistributing
- Must acknowledge use of SAM-Audio in research publications
- Prohibited for military, weapons, or espionage-related use
- Commercial use is allowed (royalty-free)

See [SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE) for details.
