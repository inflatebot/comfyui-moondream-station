import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image.

    ComfyUI images are [B,H,W,C] tensors, float32, 0-1 range, NHWC format.
    This function takes the first image from the batch.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Convert from float [0,1] to uint8 [0,255]
    image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np, mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI image tensor.

    Returns a [1,H,W,C] tensor, float32, 0-1 range.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_np = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np)

    # Add batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    return tensor


def create_empty_image(width: int = 64, height: int = 64) -> torch.Tensor:
    """Create an empty (black) image tensor.

    Returns a [1,H,W,C] tensor of zeros.
    """
    return torch.zeros(1, height, width, 3, dtype=torch.float32)


def crop_image(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Crop a PIL image to the given box.

    Args:
        image: PIL Image
        box: (x1, y1, x2, y2) in pixel coordinates

    Returns:
        Cropped PIL Image
    """
    return image.crop(box)


def normalize_to_pixel_coords(
    box: dict, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Convert normalized bounding box to pixel coordinates.

    Args:
        box: Dict with x_min, y_min, x_max, y_max in [0,1] range
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        (x1, y1, x2, y2) in pixel coordinates
    """
    x1 = int(box["x_min"] * image_width)
    y1 = int(box["y_min"] * image_height)
    x2 = int(box["x_max"] * image_width)
    y2 = int(box["y_max"] * image_height)
    return (x1, y1, x2, y2)


def combine_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Compute the bounding box that contains all input boxes.

    Args:
        boxes: List of (x1, y1, x2, y2) tuples

    Returns:
        Combined (x1, y1, x2, y2) bounding box
    """
    if not boxes:
        return (0, 0, 64, 64)

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return (x1, y1, x2, y2)
