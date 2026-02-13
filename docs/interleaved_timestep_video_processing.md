# Qwen3-VL交错时间步视频处理 / Interleaved Timestep Video Processing

[中文](#中文文档) | [English](#english-documentation)

---

## 中文文档

### 概述

Qwen3-VL采用了一种创新的**交错时间步（Interleaved Timestep）**方法来处理视频。与传统的绝对时间位置编码不同，Qwen3-VL使用时间戳来分隔视频帧，这使得模型能够更精确地理解视频中的时序信息。

### 核心特性

1. **文本-时间戳对齐（Text-Timestamp Alignment）**: 超越了T-RoPE，实现精确的基于时间戳的事件定位，增强视频时序建模能力
2. **交错MRoPE（Interleaved-MRoPE）**: 在时间、宽度和高度维度上进行全频率分配，使用鲁棒的位置编码，增强长视频推理能力

### 实现位置

交错时间步的核心实现位于：
- **主要代码**: `qwen-vl-finetune/qwenvl/data/rope2d.py` 中的 `get_rope_index_3()` 函数
- **视频处理**: `qwen-vl-utils/src/qwen_vl_utils/vision_process.py` 中的视频读取和处理函数

### 关键代码解析

#### 1. 交错时间步的位置编码 (rope2d.py)

```python
def get_rope_index_3(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    与原始实现不同，Qwen3VL使用时间戳而非绝对时间位置ID。
    
    由于我们使用时间戳来分隔视频，格式如下：
    <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>
    
    因此video_grid_thw也需要相应地拆分。
    """
    if video_grid_thw is not None:
        # 将video_grid_thw按照时间维度展开，每个时间步一个条目
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        # 将时间维度设置为1，因为我们使用时间戳来编码时序信息
        video_grid_thw[:, 0] = 1
    
    # ... 后续的位置编码计算
```

**关键点**:
- 每个视频帧都与一个时间戳配对
- `video_grid_thw[:, 0] = 1` 表示将时间维度设为1，因为时间信息通过时间戳token传递
- 这种设计使得模型能够精确定位视频中的事件

#### 2. 视频帧提取与时间戳 (vision_process.py)

```python
def _read_video_decord(ele: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
    """使用decord.VideoReader读取视频"""
    import decord
    video_path = ele["video"]
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    
    # 计算帧范围
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames, video_fps
    )
    
    # 根据配置决定提取的帧数
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    
    # 线性采样帧索引
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # 转换为TCHW格式
    
    # 计算采样FPS
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    
    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="decord",
    )
    return video, video_metadata, sample_fps
```

### 使用示例

#### 示例1: 使用视频URL（通过API）

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 使用视频URL，自动处理时间戳
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://example.com/video.mp4"},
            {"type": "text", "text": "请描述视频中发生的事件及其时间点"}
        ]
    }
]

response = client.chat.completions.create(
    model="qwen-vl-max-latest",
    messages=messages
)
print(response.choices[0].message.content)
```

#### 示例2: 使用交错的时间戳-图像对

这是Qwen3-VL的独特功能，允许你明确指定每个帧的时间戳：

```python
# 方法1: 使用图像列表和sample_fps
messages = [
    {
        "role": "user", 
        "content": [
            {
                "type": "video",
                "video": [
                    "file:///path/to/frame1.jpg",
                    "file:///path/to/frame2.jpg",
                    "file:///path/to/frame3.jpg",
                    "file:///path/to/frame4.jpg",
                ],
                'sample_fps': '1',  # 采样帧率（每秒帧数），用于确定每帧的时间戳
            },
            {"type": "text", "text": "描述这个视频"},
        ],
    }
]

# 方法2: 使用时间戳token进行精确时间控制
# 格式: <|t_start|>0.0<|t_end|><|vision_start|><image><|vision_end|>
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "<|t_start|>0.0<|t_end|>"},
            {"type": "image", "image": "file:///path/to/frame1.jpg"},
            {"type": "text", "text": "<|t_start|>2.5<|t_end|>"},
            {"type": "image", "image": "file:///path/to/frame2.jpg"},
            {"type": "text", "text": "<|t_start|>5.0<|t_end|>"},
            {"type": "image", "image": "file:///path/to/frame3.jpg"},
            {"type": "text", "text": "在2.5秒时发生了什么？"},
        ]
    }
]
```

#### 示例3: 从视频提取帧和时间戳

```python
from decord import VideoReader, cpu
import numpy as np

def get_video_frames_with_timestamps(video_path, num_frames=64):
    """提取视频帧及其时间戳"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # 均匀采样帧索引
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    # 提取帧
    frames = vr.get_batch(indices).asnumpy()
    
    # 获取每帧的时间戳
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])
    
    return frames, timestamps

# 使用示例
video_path = "path/to/video.mp4"
frames, timestamps = get_video_frames_with_timestamps(video_path, num_frames=64)

print(f"提取了 {len(frames)} 帧")
print(f"时间戳范围: {timestamps[0][0]:.2f}s 到 {timestamps[-1][0]:.2f}s")
```

#### 示例4: 时空定位（Spatial-Temporal Grounding）

```python
# 使用交错的时间戳-图像对进行精确的时空定位
video_url = "https://example.com/video.mp4"
prompt = """定位视频中的一系列活动事件，输出每个事件的开始和结束时间戳，
并用句子描述每个事件。以'mm:ss.ff'格式的时间描述提供JSON格式的结果。"""

# 提取帧和时间戳
frames, timestamps = get_video_frames_with_timestamps(video_url, num_frames=64)

# 构建带时间戳的内容
content = []
for i, (frame, ts) in enumerate(zip(frames, timestamps)):
    # 添加时间戳
    content.append({
        "type": "text", 
        "text": f"<|t_start|>{ts[0]:.2f}<|t_end|>"
    })
    # 添加帧
    content.append({
        "type": "image",
        "image": f"file:///path/to/frame_{i}.jpg"
    })
content.append({"type": "text", "text": prompt})

messages = [{"role": "user", "content": content}]

# 调用API进行推理
response = client.chat.completions.create(
    model="qwen-vl-max-latest",
    messages=messages
)
```

### 配置参数

在处理视频时，可以通过以下参数控制帧提取：

```python
video_config = {
    "type": "video",
    "video": "path/to/video.mp4",
    
    # 帧数控制
    "nframes": 64,  # 直接指定帧数（与fps互斥）
    # 或使用FPS控制
    "fps": 2.0,  # 采样帧率，默认2.0
    "min_frames": 4,  # 最小帧数，默认4
    "max_frames": 768,  # 最大帧数，默认768
    
    # 时间范围控制
    "video_start": 0.0,  # 起始时间（秒）
    "video_end": 10.0,   # 结束时间（秒）
    
    # 分辨率控制
    "min_pixels": 128 * 28 * 28,  # 最小像素数
    "max_pixels": 768 * 28 * 28,  # 最大像素数
}
```

### 技术优势

1. **精确的时间定位**: 通过时间戳token，模型可以精确理解和定位视频中的事件
2. **灵活的时序建模**: 支持非均匀采样和自定义时间间隔
3. **长视频支持**: Interleaved-MRoPE使得模型能够处理更长的视频序列
4. **更好的事件理解**: 时间戳信息帮助模型建立因果关系和时序逻辑

### 参考资料

- **论文**: [Qwen3-VL Technical Report](https://arxiv.org/pdf/2511.21631)
- **示例代码**: `cookbooks/video_understanding.ipynb`
- **核心实现**: `qwen-vl-finetune/qwenvl/data/rope2d.py`

---

## English Documentation

### Overview

Qwen3-VL employs an innovative **Interleaved Timestep** approach for video processing. Unlike traditional absolute time position encoding, Qwen3-VL uses timestamps to separate video frames, enabling the model to understand temporal information in videos more precisely.

### Key Features

1. **Text-Timestamp Alignment**: Goes beyond T-RoPE to achieve precise timestamp-grounded event localization, enhancing video temporal modeling
2. **Interleaved-MRoPE**: Full-frequency allocation across time, width, and height dimensions with robust positional embeddings, improving long-horizon video reasoning

### Implementation Location

The core implementation of interleaved timesteps is located in:
- **Main Code**: `get_rope_index_3()` function in `qwen-vl-finetune/qwenvl/data/rope2d.py`
- **Video Processing**: Video reading and processing functions in `qwen-vl-utils/src/qwen_vl_utils/vision_process.py`

### Key Code Explanation

#### 1. Interleaved Timestep Position Encoding (rope2d.py)

```python
def get_rope_index_3(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Different from the original implementation, Qwen3VL uses timestamps 
    rather than absolute time position IDs.
    
    Since we use timestamps to separate videos, in the format:
    <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>
    
    The video_grid_thw should also be split accordingly.
    """
    if video_grid_thw is not None:
        # Expand video_grid_thw along the time dimension, one entry per timestep
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        # Set time dimension to 1, as temporal information is encoded via timestamp tokens
        video_grid_thw[:, 0] = 1
    
    # ... subsequent position encoding calculations
```

**Key Points**:
- Each video frame is paired with a timestamp
- `video_grid_thw[:, 0] = 1` sets the time dimension to 1, as temporal information is conveyed through timestamp tokens
- This design enables the model to precisely localize events in videos

#### 2. Video Frame Extraction with Timestamps (vision_process.py)

```python
def _read_video_decord(ele: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
    """Read video using decord.VideoReader"""
    import decord
    video_path = ele["video"]
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    
    # Calculate frame range
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames, video_fps
    )
    
    # Determine number of frames to extract based on configuration
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    
    # Sample frame indices linearly
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    
    # Calculate sample FPS
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    
    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="decord",
    )
    return video, video_metadata, sample_fps
```

### Usage Examples

#### Example 1: Using Video URL (via API)

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Use video URL with automatic timestamp handling
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://example.com/video.mp4"},
            {"type": "text", "text": "Describe the events in the video with their timestamps"}
        ]
    }
]

response = client.chat.completions.create(
    model="qwen-vl-max-latest",
    messages=messages
)
print(response.choices[0].message.content)
```

#### Example 2: Using Interleaved Timestamp-Image Pairs

This is a unique feature of Qwen3-VL that allows you to explicitly specify timestamps for each frame:

```python
# Method 1: Using image list with sample_fps
messages = [
    {
        "role": "user", 
        "content": [
            {
                "type": "video",
                "video": [
                    "file:///path/to/frame1.jpg",
                    "file:///path/to/frame2.jpg",
                    "file:///path/to/frame3.jpg",
                    "file:///path/to/frame4.jpg",
                ],
                'sample_fps': '1',  # Frame sampling rate (frames per second) for timestamp determination
            },
            {"type": "text", "text": "Describe this video"},
        ],
    }
]

# Method 2: Using timestamp tokens for precise temporal control
# Format: <|t_start|>0.0<|t_end|><|vision_start|><image><|vision_end|>
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "<|t_start|>0.0<|t_end|>"},
            {"type": "image", "image": "file:///path/to/frame1.jpg"},
            {"type": "text", "text": "<|t_start|>2.5<|t_end|>"},
            {"type": "image", "image": "file:///path/to/frame2.jpg"},
            {"type": "text", "text": "<|t_start|>5.0<|t_end|>"},
            {"type": "image", "image": "file:///path/to/frame3.jpg"},
            {"type": "text", "text": "What happened at 2.5 seconds?"},
        ]
    }
]
```

#### Example 3: Extracting Frames and Timestamps from Video

```python
from decord import VideoReader, cpu
import numpy as np

def get_video_frames_with_timestamps(video_path, num_frames=64):
    """Extract video frames with their timestamps"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Uniformly sample frame indices
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    # Extract frames
    frames = vr.get_batch(indices).asnumpy()
    
    # Get timestamp for each frame
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])
    
    return frames, timestamps

# Usage
video_path = "path/to/video.mp4"
frames, timestamps = get_video_frames_with_timestamps(video_path, num_frames=64)

print(f"Extracted {len(frames)} frames")
print(f"Timestamp range: {timestamps[0][0]:.2f}s to {timestamps[-1][0]:.2f}s")
```

#### Example 4: Spatial-Temporal Grounding

```python
# Use interleaved timestamp-image pairs for precise spatial-temporal grounding
video_url = "https://example.com/video.mp4"
prompt = """Localize a series of activity events in the video, output the start and 
end timestamp for each event, and describe each event with sentences. 
Provide the result in JSON format with 'mm:ss.ff' format for time depiction."""

# Extract frames and timestamps
frames, timestamps = get_video_frames_with_timestamps(video_url, num_frames=64)

# Build content with timestamps
content = []
for i, (frame, ts) in enumerate(zip(frames, timestamps)):
    # Add timestamp
    content.append({
        "type": "text", 
        "text": f"<|t_start|>{ts[0]:.2f}<|t_end|>"
    })
    # Add frame
    content.append({
        "type": "image",
        "image": f"file:///path/to/frame_{i}.jpg"
    })
content.append({"type": "text", "text": prompt})

messages = [{"role": "user", "content": content}]

# Call API for inference
response = client.chat.completions.create(
    model="qwen-vl-max-latest",
    messages=messages
)
```

### Configuration Parameters

When processing videos, you can control frame extraction with these parameters:

```python
video_config = {
    "type": "video",
    "video": "path/to/video.mp4",
    
    # Frame count control
    "nframes": 64,  # Directly specify number of frames (mutually exclusive with fps)
    # Or use FPS control
    "fps": 2.0,  # Sampling frame rate, default 2.0
    "min_frames": 4,  # Minimum number of frames, default 4
    "max_frames": 768,  # Maximum number of frames, default 768
    
    # Time range control
    "video_start": 0.0,  # Start time in seconds
    "video_end": 10.0,   # End time in seconds
    
    # Resolution control
    "min_pixels": 128 * 28 * 28,  # Minimum pixels
    "max_pixels": 768 * 28 * 28,  # Maximum pixels
}
```

### Technical Advantages

1. **Precise Temporal Localization**: Through timestamp tokens, the model can accurately understand and localize events in videos
2. **Flexible Temporal Modeling**: Supports non-uniform sampling and custom time intervals
3. **Long Video Support**: Interleaved-MRoPE enables the model to handle longer video sequences
4. **Better Event Understanding**: Timestamp information helps the model establish causal relationships and temporal logic

### References

- **Paper**: [Qwen3-VL Technical Report](https://arxiv.org/pdf/2511.21631)
- **Example Code**: `cookbooks/video_understanding.ipynb`
- **Core Implementation**: `qwen-vl-finetune/qwenvl/data/rope2d.py`
