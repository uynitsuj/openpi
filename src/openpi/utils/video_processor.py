import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple, Union


def resize_and_pad_video(
    input_path: str,
    output_path: str,
    target_size: int = 224,
    target_aspect_ratio: Optional[Tuple[int, int]] = None,
    encoder: str = "h264_nvenc",
    crf: Optional[int] = None,
    bitrate: Optional[str] = None,
    overwrite: bool = True,
    fps: Optional[Union[int, float, str]] = None,
    frame_stride: Optional[int] = None
) -> bool:
    """
    Resize, pad, and optionally subsample a video using ffmpeg.
    
    Args:
        input_path: Path to input video file
        output_path: Path for output video file
        target_size: Target width/height for square output (default: 224)
        target_aspect_ratio: Optional tuple (width, height) for non-square output
        encoder: Video encoder to use (default: "h264_nvenc", fallback: "libx264")
        crf: Constant Rate Factor for quality (0-51, lower is better quality)
        bitrate: Target bitrate (e.g., "2M", "1500k")
        overwrite: Whether to overwrite existing output file
        fps: Target frame rate (e.g., 15, 30, "30000/1001"). If None, keeps original fps
        frame_stride: Take every Nth frame (e.g., stride=2 takes every other frame).
                     Alternative to fps for precise frame subsampling
    
    Returns:
        bool: True if successful, False otherwise
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If parameters are invalid
    
    Note:
        - If both fps and frame_stride are specified, frame_stride takes precedence
        - frame_stride=2 means take every 2nd frame (halves frame rate)
        - frame_stride=3 means take every 3rd frame (1/3 frame rate)
    """
    
    # Validate inputs
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    
    if frame_stride is not None and frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    
    # Create output directory if it doesn't exist
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle file overwrite
    if output_path_obj.exists() and not overwrite:
        raise FileExistsError(f"Output file exists and overwrite=False: {output_path}")
    
    # Determine target dimensions
    if target_aspect_ratio:
        target_width, target_height = target_aspect_ratio
    else:
        target_width = target_height = target_size
    
    # Build ffmpeg command
    cmd = ["ffmpeg"]
    
    # Input
    cmd.extend(["-i", str(input_path)])
    
    # Build video filter chain
    filters = []
    
    # Temporal subsampling filter
    if frame_stride is not None:
        # Use select filter for precise frame selection
        filters.append(f"select='not(mod(n,{frame_stride}))'")
    elif fps is not None:
        # Use fps filter for frame rate conversion
        filters.append(f"fps={fps}")
    
    # Spatial scaling and padding filters
    if target_aspect_ratio:
        scale_filter = f"scale='if(gt(a,{target_width}/{target_height}),{target_width},-1)':'if(gt(a,{target_width}/{target_height}),-1,{target_height})'"
    else:
        # For square output, maintain aspect ratio and fit within target_size
        scale_filter = f"scale='if(gt(a,1),{target_size},-1)':'if(gt(a,1),-1,{target_size})'"
    
    filters.append(scale_filter)
    
    pad_filter = f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2"
    filters.append(pad_filter)
    
    # Combine all filters
    vf = ",".join(filters)
    cmd.extend(["-vf", vf])
    
    # Video encoding options
    cmd.extend(["-c:v", encoder])
    
    # Quality/bitrate settings
    if crf is not None:
        cmd.extend(["-crf", str(crf)])
    elif bitrate is not None:
        cmd.extend(["-b:v", bitrate])
    
    # If using frame selection, need to handle timestamps
    if frame_stride is not None:
        cmd.extend(["-vsync", "vfr"])  # Variable frame rate to handle selected frames
    
    # Overwrite output file
    if overwrite:
        cmd.append("-y")
    
    # Output
    cmd.append(str(output_path))
    
    try:
        # Run ffmpeg command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return True
        
    except subprocess.CalledProcessError as e:
        # If NVENC fails, try with CPU encoder as fallback
        if encoder == "h264_nvenc" and "h264_nvenc" in str(e.stderr):
            print(f"NVENC encoder failed, falling back to libx264...")
            return resize_and_pad_video(
                input_path=str(input_path),
                output_path=str(output_path),
                target_size=target_size,
                target_aspect_ratio=target_aspect_ratio,
                encoder="libx264",
                crf=crf,
                bitrate=bitrate,
                overwrite=overwrite,
                fps=fps,
                frame_stride=frame_stride
            )
        
        print(f"FFmpeg error: {e.stderr}")
        return False
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False


def get_video_info(input_path: str) -> dict:
    """
    Get basic information about a video file.
    
    Args:
        input_path: Path to video file
        
    Returns:
        dict: Video information including fps, duration, resolution
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
        str(input_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        
        video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
        if not video_stream:
            return {}
        
        info = {
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")),  # Convert fraction to float
            "duration": float(data["format"].get("duration", 0)),
            "frame_count": int(video_stream.get("nb_frames", 0)) if video_stream.get("nb_frames") else None
        }
        
        return info
        
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}


def batch_resize_videos(
    input_dir: str,
    output_dir: str,
    target_size: int = 224,
    file_pattern: str = "*.mp4",
    **kwargs
) -> dict:
    """
    Batch process multiple videos in a directory.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory for output videos
        target_size: Target size for square output
        file_pattern: Glob pattern for input files (default: "*.mp4")
        **kwargs: Additional arguments passed to resize_and_pad_video
    
    Returns:
        dict: Results with 'success' and 'failed' lists
    """
    input_dir_obj = Path(input_dir)
    output_dir_obj = Path(output_dir)
    
    if not input_dir_obj.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    results = {"success": [], "failed": []}
    
    for input_file in input_dir_obj.glob(file_pattern):
        output_file = output_dir_obj / input_file.name
        
        try:
            success = resize_and_pad_video(
                input_path=str(input_file),
                output_path=str(output_file),
                target_size=target_size,
                **kwargs
            )
            
            if success:
                results["success"].append(str(input_file))
            else:
                results["failed"].append(str(input_file))
                
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            results["failed"].append(str(input_file))
    
    return results


# Example usage
if __name__ == "__main__":
    input_video = "/nfs_us/data/oreo_xmi/clean_whiteboard/episode_026MQ7i2jd1QK7qEGyn7ylM1conj9oSJL532dI0apV8/left_camera-images-rgb.mp4"
    
    # Get video info first
    print("Original video info:")
    info = get_video_info(input_video)
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # Example 1: Basic processing without temporal subsampling
    print("1. Basic processing (no temporal subsampling):")
    success = resize_and_pad_video(
        input_path=input_video,
        output_path="test_output_full.mp4",
        target_size=224,
        encoder="h264_nvenc"
    )
    print(f"Success: {success}\n")
    
    # Example 2: Frame stride - every other frame (stride=2)
    print("2. Every other frame (stride=2):")
    success = resize_and_pad_video(
        input_path=input_video,
        output_path="test_output_stride2.mp4",
        target_size=224,
        frame_stride=2,
        encoder="h264_nvenc"
    )
    print(f"Success: {success}\n")
    
    # Example 3: Frame stride - every 3rd frame (stride=3)
    print("3. Every 3rd frame (stride=3):")
    success = resize_and_pad_video(
        input_path=input_video,
        output_path="test_output_stride3.mp4",
        target_size=224,
        frame_stride=3,
        encoder="h264_nvenc"
    )
    print(f"Success: {success}\n")
    
    # Example 4: Specific FPS (15 fps)
    print("4. Convert to 15 fps:")
    success = resize_and_pad_video(
        input_path=input_video,
        output_path="test_output_15fps.mp4",
        target_size=224,
        fps=15,
        encoder="h264_nvenc"
    )
    print(f"Success: {success}\n") 