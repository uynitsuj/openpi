#!/usr/bin/env python3
"""
Detailed debug script to understand frame loading and show pixel differences.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def debug_frame_loading():
    """Debug frame loading with detailed output."""
    
    repo_id = "uynitsuj/hummus_xmi_full_subsample_2_cleaned2"
    episode_idx = 2648
    
    print(f"Debugging frame loading for episode {episode_idx}")
    
    # Test different timestamps
    fps = 15.0
    frame_indices = [0, 1, 2, 3]
    timestamps = [idx / fps for idx in frame_indices]
    
    print(f"Frame indices: {frame_indices}")
    print(f"Timestamps: {[f'{t:.3f}' for t in timestamps]}")
    
    frames_data = []
    
    for i, (frame_idx, timestamp) in enumerate(zip(frame_indices, timestamps)):
        print(f"\n--- Loading frame {frame_idx} (timestamp {timestamp:.3f}) ---")
        
        try:
            # Create dataset for this specific timestamp
            dataset = lerobot_dataset.LeRobotDataset(
                repo_id,
                delta_timestamps={"state": [timestamp]},
            )
            
            # Get the frame data
            frame_data = dataset[episode_idx]
            print(f"  Frame data type: {type(frame_data)}")
            print(f"  Frame data keys: {list(frame_data.keys())}")
            
            # Extract camera data
            camera_mapping = {
                "base_0_rgb": "top_camera-images-rgb",
                "left_wrist_0_rgb": "left_camera-images-rgb", 
                "right_wrist_0_rgb": "right_camera-images-rgb",
            }
            
            for expected_key, actual_key in camera_mapping.items():
                if actual_key in frame_data:
                    images = frame_data[actual_key]
                    print(f"  {actual_key} type: {type(images)}")
                    
                    # Handle different image formats
                    if isinstance(images, torch.Tensor):
                        image = images.numpy()
                    elif isinstance(images, dict) and "bytes" in images:
                        from PIL import Image
                        import io
                        image = Image.open(io.BytesIO(images["bytes"]))
                        image = np.array(image)
                    elif hasattr(images, 'convert'):
                        image = np.array(images)
                    elif isinstance(images, np.ndarray):
                        image = images
                    else:
                        print(f"    Unknown format: {type(images)}")
                        continue
                    
                    print(f"    Image shape: {image.shape}")
                    print(f"    Image dtype: {image.dtype}")
                    print(f"    Image min/max: {image.min():.3f}/{image.max():.3f}")
                    print(f"    Image mean: {image.mean():.3f}")
                    
                    # Resize to 112x112 for comparison
                    if image.shape[:2] != (112, 112):
                        # Handle different image formats
                        if len(image.shape) == 3 and image.shape[0] == 3:
                            # (C, H, W) -> (H, W, C) for PIL
                            image = np.transpose(image, (1, 2, 0))
                        elif len(image.shape) == 3 and image.shape[2] == 3:
                            # (H, W, C) - already correct
                            pass
                        else:
                            print(f"    Unexpected image shape: {image.shape}")
                            continue
                        
                        from PIL import Image
                        pil_image = Image.fromarray((image * 255).astype(np.uint8))
                        image = np.array(pil_image.resize((112, 112))) / 255.0
                    
                    frames_data.append({
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                        'camera': expected_key,
                        'image': image,
                        'mean': image.mean(),
                        'std': image.std(),
                    })
                    
                    print(f"    Resized image mean: {image.mean():.3f}")
                    print(f"    Resized image std: {image.std():.3f}")
                    break  # Only use first camera for now
            
        except Exception as e:
            print(f"  Error loading frame {frame_idx}: {e}")
    
    # Compare frames
    print(f"\n--- Frame Comparison ---")
    if len(frames_data) >= 2:
        for i in range(len(frames_data) - 1):
            frame1 = frames_data[i]
            frame2 = frames_data[i + 1]
            
            diff = np.abs(frame1['image'] - frame2['image'])
            mean_diff = diff.mean()
            max_diff = diff.max()
            
            print(f"  Frame {frame1['frame_idx']} vs {frame2['frame_idx']}:")
            print(f"    Mean difference: {mean_diff:.6f}")
            print(f"    Max difference: {max_diff:.6f}")
            print(f"    Mean values: {frame1['mean']:.3f} vs {frame2['mean']:.3f}")
            
            if mean_diff < 1e-6:
                print(f"    ⚠️  Frames are nearly identical!")
            else:
                print(f"    ✅ Frames are different!")
    
    # Save debug images
    if frames_data:
        save_debug_images([f['image'] for f in frames_data], "debug_frame_loading")
        print(f"\nDebug images saved to debug_frame_loading_debug.png")

def save_debug_images(images, prefix):
    """Save debug images to visualize the frames."""
    try:
        # Convert to uint8 for saving
        images_uint8 = []
        for img in images:
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img_uint8 = (img * 255).astype(np.uint8)
                else:
                    img_uint8 = img.astype(np.uint8)
            else:
                img_uint8 = img.astype(np.uint8)
            images_uint8.append(img_uint8)
        
        # Create a grid of images
        fig, axes = plt.subplots(1, len(images_uint8), figsize=(3*len(images_uint8), 3))
        if len(images_uint8) == 1:
            axes = [axes]
        
        for i, img in enumerate(images_uint8):
            axes[i].imshow(img)
            axes[i].set_title(f"Frame {i}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{prefix}_debug.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Debug images saved to {prefix}_debug.png")
        
    except Exception as e:
        print(f"Error saving debug images: {e}")

if __name__ == "__main__":
    debug_frame_loading() 