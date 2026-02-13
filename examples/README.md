# Qwen3-VL Examples

This directory contains example scripts demonstrating various features of Qwen3-VL.

## Interleaved Timestep Video Processing

**File:** `interleaved_timestep_video_example.py`

This example demonstrates Qwen3-VL's unique interleaved timestep video processing feature, which allows precise temporal understanding of video content.

### What it shows:

1. **Video Frame Extraction with Timestamps**: How to extract frames from a video while preserving their temporal information
2. **Interleaved Content Creation**: How to create interleaved timestamp-image pairs for model input
3. **API Usage**: Structure for using the Qwen3-VL API with video inputs
4. **Spatial-Temporal Grounding**: Example of precise event localization in videos

### Running the example:

```bash
# Basic demonstration (no dependencies required)
python examples/interleaved_timestep_video_example.py

# For full functionality, install dependencies:
pip install decord numpy pillow

# Then provide a video path in the script
```

### Key Concepts:

- **Text-Timestamp Alignment**: Qwen3-VL uses timestamps to separate video frames, enabling precise event localization
- **Interleaved-MRoPE**: Position encoding that enhances long-horizon video reasoning
- **Timestamp Tokens**: Format `<|t_start|>X.XX<|t_end|>` used to mark temporal positions

### Related Documentation:

- [Interleaved Timestep Video Processing Documentation](../docs/interleaved_timestep_video_processing.md) - Comprehensive guide with bilingual documentation
- [Video Understanding Cookbook](../cookbooks/video_understanding.ipynb) - Interactive notebook with more examples

### Implementation Details:

The core implementation can be found in:
- `qwen-vl-finetune/qwenvl/data/rope2d.py` - Position encoding with `get_rope_index_3()`
- `qwen-vl-utils/src/qwen_vl_utils/vision_process.py` - Video processing functions

For more information, see the [Qwen3-VL Paper](https://arxiv.org/pdf/2511.21631).
