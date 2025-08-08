import numpy as np
from pathlib import Path
import cv2
import torch
# from omegaconf import OmegaConf
import os
import logging


def resize_with_center_crop(
    images: np.ndarray,
    height: int,
    width: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resizes an image to a target height and width without distortion by center cropping.

    Args:
        images: Input image(s) with shape (h, w, c) or (b, h, w, c)
        height: Target height
        width: Target width
        interpolation: OpenCV interpolation method (default: cv2.INTER_LINEAR)

    Returns:
        Resized and center-cropped image(s) with shape (height, width, c) or (b, height, width, c)
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # Add batch dimension

    batch_size, cur_height, cur_width, channels = images.shape

    # Calculate scaling ratio to ensure both dimensions are at least as large as target
    # (we'll crop the excess, so we want to scale up to cover the target dimensions)
    ratio = max(height / cur_height, width / cur_width)
    resized_height = int(cur_height * ratio)
    resized_width = int(cur_width * ratio)

    # Process each image in the batch
    cropped_images = np.zeros((batch_size, height, width, channels), dtype=images.dtype)

    for i in range(batch_size):
        # Resize image so that the smaller dimension fits the target
        resized_img = cv2.resize(images[i], (resized_width, resized_height), interpolation=interpolation)

        # Calculate crop offsets to center the crop
        crop_h0 = (resized_height - height) // 2
        crop_w0 = (resized_width - width) // 2

        # Ensure we don't go out of bounds
        crop_h0 = max(0, crop_h0)
        crop_w0 = max(0, crop_w0)
        crop_h1 = min(resized_height, crop_h0 + height)
        crop_w1 = min(resized_width, crop_w0 + width)

        # Extract the center crop
        cropped_img = resized_img[crop_h0:crop_h1, crop_w0:crop_w1]

        # Handle edge case where crop might be smaller than target (shouldn't happen with correct ratio calculation)
        if cropped_img.shape[0] != height or cropped_img.shape[1] != width:
            cropped_img = cv2.resize(cropped_img, (width, height), interpolation=interpolation)

        cropped_images[i] = cropped_img

    # Remove batch dimension if it wasn't in the input
    if not has_batch_dim:
        cropped_images = cropped_images[0]

    return cropped_images


def resize_with_pad(
    images: np.ndarray,
    height: int,
    width: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resizes an image to a target height and width without distortion by padding with black.

    Args:
        images: Input image(s) with shape (h, w, c) or (b, h, w, c)
        height: Target height
        width: Target width
        interpolation: OpenCV interpolation method (default: cv2.INTER_LINEAR)

    Returns:
        Resized and padded image(s) with shape (height, width, c) or (b, height, width, c)
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # Add batch dimension

    batch_size, cur_height, cur_width, channels = images.shape

    # Calculate scaling ratio to maintain aspect ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Process each image in the batch
    resized_images = np.zeros((batch_size, resized_height, resized_width, channels), dtype=images.dtype)

    for i in range(batch_size):
        resized_images[i] = cv2.resize(images[i], (resized_width, resized_height), interpolation=interpolation)

    # Calculate padding amounts
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Determine padding value based on dtype
    if images.dtype == np.uint8:
        pad_value = 0
    elif images.dtype == np.float32:
        pad_value = -1.0
    else:
        pad_value = 0

    # Apply padding
    padded_images = np.pad(
        resized_images,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )

    # Remove batch dimension if it wasn't in the input
    if not has_batch_dim:
        padded_images = padded_images[0]

    return padded_images


def matrix_to_position_quaternion(matrix):
    """Convert 4x4 transformation matrix to position and quaternion (wxyz)."""
    position = matrix[:3, 3]
    
    # Extract rotation matrix and convert to quaternion
    R = matrix[:3, :3]
    
    # Convert rotation matrix to quaternion (wxyz format)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    
    return position, np.array([qw, qx, qy, qz])

def depth_color_to_pointcloud(
    depth: np.ndarray, img: np.ndarray, intrinsics: np.ndarray, subsample_factor: int = 2, far_clip: float = 20.0, near_clip: float = 0.015
) -> np.ndarray:
    """Convert depth and rgb image to points."""
    H, W = depth.shape
    H = H // subsample_factor
    W = W // subsample_factor
    depth = depth[::subsample_factor, ::subsample_factor]
    img = img[::subsample_factor, ::subsample_factor]

    # Scale intrinsics to match subsampled image
    intrinsics = intrinsics.copy()
    intrinsics[0, 0] /= subsample_factor  # fx
    intrinsics[1, 1] /= subsample_factor  # fy
    intrinsics[0, 2] /= subsample_factor  # cx
    intrinsics[1, 2] /= subsample_factor  # cy

    # Create meshgrid of pixel coordinates using numpy
    j, i = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    pixels = np.stack([j.flatten(), i.flatten()], axis=-1).astype(np.float32)

    # Get z values for all pixels
    z = depth.reshape(-1)

    # Calculate x,y coordinates for all pixels in parallel
    x = (pixels[:, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (pixels[:, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1]

    # Stack x,y,z coordinates
    points = np.stack([x, y, z], axis=-1)

    # Get colors for all pixels
    colors = img.reshape(-1, img.shape[-1])[:, :3] / 255.0

    # Filter out NaN and infinite values and zeros and z values beyond 20m
    valid_mask = (
        ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1) & (points[:, 2] < far_clip) & (points[:, 2] > near_clip)
    )

    return points[valid_mask], colors[valid_mask]

def disparity_to_depth(disparity: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
    """Convert disparity to depth using stereo baseline and focal length."""
    # Avoid division by zero
    disparity = np.where(disparity > 0, disparity, np.inf)
    depth = baseline * focal_length / disparity
    return depth.astype(np.float32)

def process_stereo_pair(img0: np.ndarray, img1: np.ndarray, model, intrinsics: np.ndarray, 
                       distortion: np.ndarray, baseline: float, valid_iters: int = 32, undistort: bool = True, 
                       max_height: int = 360) -> np.ndarray:
    """Process stereo pair to generate depth map with memory management and optional downsizing."""
    from FoundationStereo.core.utils.utils import InputPadder
    
    # Store original image dimensions
    orig_height, orig_width = img0.shape[:2]
    
    # Downsize images if height exceeds max_height
    scale_factor = 1.0
    if orig_height > max_height:
        scale_factor = max_height / orig_height
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        
        # print(f"Downsizing images from {orig_width}x{orig_height} to {new_width}x{new_height} (scale: {scale_factor:.3f})")
        
        img0_resized = cv2.resize(img0, (new_width, new_height), interpolation=cv2.INTER_AREA)
        img1_resized = cv2.resize(img1, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Scale intrinsics to match resized images
        intrinsics_scaled = intrinsics.copy()
        intrinsics_scaled[0, 0] *= scale_factor  # fx
        intrinsics_scaled[1, 1] *= scale_factor  # fy
        intrinsics_scaled[0, 2] *= scale_factor  # cx
        intrinsics_scaled[1, 2] *= scale_factor  # cy
    else:
        img0_resized = img0
        img1_resized = img1
        intrinsics_scaled = intrinsics

    # Undistort images if requested
    if undistort:
        img0_undist = cv2.undistort(img0_resized, intrinsics_scaled, distortion)
        img1_undist = cv2.undistort(img1_resized, intrinsics_scaled, distortion)
    else:
        img0_undist = img0_resized
        img1_undist = img1_resized
    
    # Initialize tensors to None for proper cleanup
    img0_tensor = None
    img1_tensor = None
    img0_padded = None
    img1_padded = None
    disp = None
    
    try:
        # Convert to torch tensors
        img0_tensor = torch.as_tensor(img0_undist).cuda().float()[None].permute(0,3,1,2)
        img1_tensor = torch.as_tensor(img1_undist).cuda().float()[None].permute(0,3,1,2)
        
        # Pad images
        padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
        img0_padded, img1_padded = padder.pad(img0_tensor, img1_tensor)
        
        # Free unpadded tensors immediately
        del img0_tensor, img1_tensor
        img0_tensor = img1_tensor = None
        
        # Run inference
        with torch.cuda.amp.autocast(True):
            disp = model.forward(img0_padded, img1_padded, iters=valid_iters, test_mode=True)
        
        # Free padded tensors immediately after inference
        del img0_padded, img1_padded
        img0_padded = img1_padded = None
        
        # Unpad and convert to numpy
        disp = padder.unpad(disp.float())
        disp_numpy = disp.data.cpu().numpy().reshape(img0_undist.shape[:2])
        
        # Free disparity tensor
        del disp
        disp = None
        
        # Force GPU memory cleanup
        torch.cuda.empty_cache()
        
        # If we resized the images, resize the depth back to original size
        if scale_factor != 1.0:
            depth_resized = cv2.resize(disp_numpy, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
            # Scale depth values back to original focal length
            depth_resized = depth_resized / scale_factor
        else:
            depth_resized = disp_numpy
            
        # Convert disparity to depth using original intrinsics
        depth = disparity_to_depth(depth_resized, baseline, intrinsics[0, 0])

        return depth
        
    except Exception as e:
        print(f"Error in stereo processing: {e}")
        # Clean up tensors in case of error
        for tensor in [img0_tensor, img1_tensor, img0_padded, img1_padded, disp]:
            if tensor is not None:
                del tensor
        torch.cuda.empty_cache()
        raise
    
    finally:
        # Final cleanup to ensure all tensors are freed
        for tensor in [img0_tensor, img1_tensor, img0_padded, img1_padded, disp]:
            if tensor is not None:
                del tensor
        torch.cuda.empty_cache()

def load_foundation_stereo_model(ckpt_dir: str):
    """Load Foundation Stereo model from checkpoint."""
    from FoundationStereo.core.foundation_stereo import FoundationStereo
    
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'

    args = OmegaConf.create(cfg)
    logging.info(f"Loading Foundation Stereo with args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()
    
    return model
    # except ImportError:
    #     logging.warning("FoundationStereo not available. Stereo processing disabled.")
    #     return None
    # except Exception as e:
    #     logging.error(f"Failed to load Foundation Stereo model: {e}")
    #     return None

def extract_video_frames(video_path: Path, max_frames: int = None) -> np.ndarray:
    """Extract frames from MP4 video and return as numpy array."""
    print(f"Extracting frames from {video_path.name}...")
    
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1
        
        # Limit frames if specified
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    
    if frames:
        frames_array = np.array(frames)
        print(f"  Extracted {len(frames)} frames with shape {frames_array.shape}")
        return frames_array
    else:
        print(f"  No frames extracted from {video_path.name}")
        return np.array([])

def load_ZED_intrinsics(intrinsics_path: str, camera_name: str = "LEFT_CAM_VGA") -> np.ndarray:
    """Load ZED intrinsics from file."""
    fx, fy, cx, cy = None, None, None, None
    k1, k2, p1, p2, k3 = None, None, None, None, None

    with open(intrinsics_path, 'r') as f:
        line_iter = iter(f.readlines())
        for line in line_iter:
            i = 0
            if camera_name in line:
                while fx is None or fy is None or cx is None or cy is None or k1 is None or k2 is None or p1 is None or p2 is None or k3 is None:
                    line = next(line_iter)
                    if 'fx' in line:    
                        fx = float(line.split('=')[1].strip())
                    if 'fy' in line:
                        fy = float(line.split('=')[1].strip())
                    if 'cx' in line:
                        cx = float(line.split('=')[1].strip())
                    if 'cy' in line:
                        cy = float(line.split('=')[1].strip())
                    if 'k1' in line:
                        k1 = float(line.split('=')[1].strip())
                    if 'k2' in line:
                        k2 = float(line.split('=')[1].strip())
                    if 'p1' in line:
                        p1 = float(line.split('=')[1].strip())
                    if 'p2' in line:
                        p2 = float(line.split('=')[1].strip())
                    if 'k3' in line:
                        k3 = float(line.split('=')[1].strip())   
                i += 1
                if i > 10:
                    break
                        
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        distortion = np.array([k1, k2, p1, p2, k3])
    return intrinsics, distortion