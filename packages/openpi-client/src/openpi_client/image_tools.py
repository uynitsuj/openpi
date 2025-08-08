import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def resize_with_center_crop(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_center_crop for multiple images using PIL. Resizes a batch of images to a target height
    and width without distortion by center cropping.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized and center-cropped images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([np.array(_resize_with_center_crop(Image.fromarray(im), height, width, method=method)) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

def _resize_with_center_crop(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_center_crop for one image using PIL. Resizes an image to a target height and
    width without distortion by cropping the center of the image.
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.
    
    # Calculate scaling ratio to ensure both dimensions are at least as large as target
    # (we'll crop the excess, so we want to scale up to cover the target dimensions)
    ratio = max(height / cur_height, width / cur_width)
    resized_width = int(cur_width * ratio)
    resized_height = int(cur_height * ratio)
    
    # Resize image so that the smaller dimension fits the target
    resized_image = image.resize((resized_width, resized_height), resample=method)
    
    # Calculate crop offsets to center the crop
    crop_w0 = (resized_width - width) // 2
    crop_h0 = (resized_height - height) // 2
    
    # Ensure we don't go out of bounds
    crop_w0 = max(0, crop_w0)
    crop_h0 = max(0, crop_h0)
    crop_w1 = min(resized_width, crop_w0 + width)
    crop_h1 = min(resized_height, crop_h0 + height)
    
    # Extract the center crop using PIL's crop method (left, upper, right, lower)
    cropped_image = resized_image.crop((crop_w0, crop_h0, crop_w1, crop_h1))
    
    # Handle edge case where crop might be smaller than target (shouldn't happen with correct ratio calculation)
    if cropped_image.size != (width, height):
        cropped_image = cropped_image.resize((width, height), resample=method)
    
    assert cropped_image.size == (width, height)
    return cropped_image