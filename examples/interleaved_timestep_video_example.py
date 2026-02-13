#!/usr/bin/env python3
"""
Example script demonstrating Qwen3-VL's interleaved timestep video processing.

This example shows how to:
1. Extract video frames with timestamps
2. Use interleaved timestamp-image pairs for inference
3. Perform spatial-temporal grounding

Reference: docs/interleaved_timestep_video_processing.md
"""

import os
from typing import List, Tuple, Optional

# Optional imports - script will work without these for demonstration
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def extract_frames_with_timestamps(video_path: str, num_frames: int = 64) -> Tuple[Optional[object], Optional[object]]:
    """
    Extract frames from video with their timestamps.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        
    Returns:
        Tuple of (frames, timestamps)
        - frames: numpy array of shape (num_frames, H, W, C)
        - timestamps: numpy array of shape (num_frames, 2) containing [timestamp, index]
    """
    try:
        from decord import VideoReader, cpu
    except ImportError:
        raise ImportError("Please install decord: pip install decord")
    
    if not HAS_NUMPY:
        raise ImportError("Please install numpy: pip install numpy")
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Uniformly sample frame indices
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    # Extract frames
    frames = vr.get_batch(indices).asnumpy()
    
    # Get timestamp for each frame
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])
    
    return frames, timestamps


def create_interleaved_content(frames: object, timestamps: object, save_dir: str = "/tmp/frames") -> List[dict]:
    """
    Create interleaved timestamp-image content for Qwen3-VL.
    
    Args:
        frames: Numpy array of frames
        timestamps: Numpy array of timestamps
        save_dir: Directory to save temporary frame images
        
    Returns:
        List of content items (text and image) for model input
    """
    if not HAS_PIL:
        raise ImportError("Please install Pillow: pip install Pillow")
        
    os.makedirs(save_dir, exist_ok=True)
    
    content = []
    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        # Add timestamp token
        content.append({
            "type": "text",
            "text": f"<|t_start|>{ts[0]:.2f}<|t_end|>"
        })
        
        # Save frame as image
        frame_path = os.path.join(save_dir, f"frame_{i:04d}.jpg")
        Image.fromarray(frame).save(frame_path)
        
        # Add image
        content.append({
            "type": "image",
            "image": f"file://{frame_path}"
        })
    
    return content


def example_video_qa_with_timestamps(video_path: str, question: str, num_frames: int = 32):
    """
    Example: Video QA using interleaved timestamp-image pairs.
    
    Args:
        video_path: Path to video file
        question: Question to ask about the video
        num_frames: Number of frames to extract
    """
    print(f"Processing video: {video_path}")
    print(f"Question: {question}")
    
    # Extract frames and timestamps
    frames, timestamps = extract_frames_with_timestamps(video_path, num_frames)
    print(f"\nExtracted {len(frames)} frames")
    print(f"Timestamp range: {timestamps[0][0]:.2f}s to {timestamps[-1][0]:.2f}s")
    
    # Create interleaved content
    content = create_interleaved_content(frames, timestamps)
    
    # Add question
    content.append({"type": "text", "text": question})
    
    # Prepare messages for API
    messages = [{"role": "user", "content": content}]
    
    print("\nContent structure created with interleaved timestamps and images")
    print(f"Total content items: {len(content)}")
    print(f"- Timestamp items: {len([c for c in content if c.get('type') == 'text' and '<|t_start|>' in c.get('text', '')])}")
    print(f"- Image items: {len([c for c in content if c.get('type') == 'image'])}")
    
    return messages


def example_spatial_temporal_grounding(video_path: str, num_frames: int = 64):
    """
    Example: Spatial-temporal grounding with precise event localization.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
    """
    prompt = """Localize a series of activity events in the video, output the start and 
end timestamp for each event, and describe each event with sentences. 
Provide the result in JSON format with 'mm:ss.ff' format for time depiction."""
    
    print(f"\n{'='*60}")
    print("Spatial-Temporal Grounding Example")
    print(f"{'='*60}")
    
    return example_video_qa_with_timestamps(video_path, prompt, num_frames)


def example_using_api():
    """
    Example: Using the API with video URL (simplified version without actual API call).
    """
    print(f"\n{'='*60}")
    print("API Usage Example (Structure Only)")
    print(f"{'='*60}")
    
    # This shows the structure, but doesn't make actual API calls
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "https://example.com/video.mp4"},
                {"type": "text", "text": "Describe the events in the video with their timestamps"}
            ]
        }
    ]
    
    print("\nMessage structure for API:")
    print("- Role: user")
    print("- Content:")
    print("  - Video URL: https://example.com/video.mp4")
    print("  - Text prompt: Describe the events in the video with their timestamps")
    
    # To actually use the API:
    # from openai import OpenAI
    # client = OpenAI(api_key="your-api-key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    # response = client.chat.completions.create(model="qwen-vl-max-latest", messages=messages)
    
    return messages


def main():
    """
    Main function demonstrating various use cases.
    """
    print("Qwen3-VL Interleaved Timestep Video Processing Examples")
    print("="*60)
    
    # Example 1: API usage structure
    example_using_api()
    
    # Example 2 & 3: Would require actual video file
    # Uncomment and provide video path to run
    # video_path = "/path/to/your/video.mp4"
    # example_video_qa_with_timestamps(video_path, "What are the main activities in this video?", num_frames=32)
    # example_spatial_temporal_grounding(video_path, num_frames=64)
    
    print("\n" + "="*60)
    print("For more examples and documentation, see:")
    print("- docs/interleaved_timestep_video_processing.md")
    print("- cookbooks/video_understanding.ipynb")
    print("="*60)


if __name__ == "__main__":
    main()
