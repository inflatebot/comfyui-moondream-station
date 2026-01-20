import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from svgelements import SVG, Path

    HAS_SVGELEMENTS = True
except ImportError:
    HAS_SVGELEMENTS = False


def create_mask_from_boxes(
    boxes: list[tuple[int, int, int, int]], width: int, height: int
) -> torch.Tensor:
    """Create a binary mask from bounding boxes.

    Args:
        boxes: List of (x1, y1, x2, y2) in pixel coordinates
        width: Mask width
        height: Mask height

    Returns:
        Mask tensor of shape [1, H, W], float32, 0-1 range
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for box in boxes:
        draw.rectangle(box, fill=255)

    mask_np = np.array(mask).astype(np.float32) / 255.0
    return torch.from_numpy(mask_np).unsqueeze(0)


def create_empty_mask(width: int = 64, height: int = 64) -> torch.Tensor:
    """Create an empty (black) mask tensor.

    Returns:
        Mask tensor of shape [1, H, W], all zeros
    """
    return torch.zeros(1, height, width, dtype=torch.float32)


def svg_path_to_mask(
    svg_path: str, width: int, height: int, fallback_box: tuple[int, int, int, int] = None
) -> torch.Tensor:
    """Convert an SVG path string to a binary mask.

    Args:
        svg_path: SVG path data string (d attribute)
        width: Mask width
        height: Mask height
        fallback_box: Bounding box to use if SVG parsing fails

    Returns:
        Mask tensor of shape [1, H, W], float32, 0-1 range
    """
    if not HAS_SVGELEMENTS:
        if fallback_box:
            return create_mask_from_boxes([fallback_box], width, height)
        return create_empty_mask(width, height)

    try:
        # Parse the SVG path
        path = Path(svg_path)

        # Create a mask image
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # Convert path segments to polygon points
        points = []
        for segment in path.segments():
            # Get the end point of each segment
            if hasattr(segment, "end"):
                end = segment.end
                points.append((end.real, end.imag))

        if len(points) >= 3:
            draw.polygon(points, fill=255)
        elif fallback_box:
            draw.rectangle(fallback_box, fill=255)

        mask_np = np.array(mask).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np).unsqueeze(0)

    except Exception:
        if fallback_box:
            return create_mask_from_boxes([fallback_box], width, height)
        return create_empty_mask(width, height)


def combine_masks(masks: list[torch.Tensor]) -> torch.Tensor:
    """Combine multiple masks into one using max (union).

    Args:
        masks: List of mask tensors of shape [1, H, W]

    Returns:
        Combined mask tensor of shape [1, H, W]
    """
    if not masks:
        return create_empty_mask()

    combined = masks[0].clone()
    for mask in masks[1:]:
        combined = torch.maximum(combined, mask)

    return combined


def stack_masks(masks: list[torch.Tensor]) -> torch.Tensor:
    """Stack multiple masks into a batch.

    Args:
        masks: List of mask tensors of shape [1, H, W]

    Returns:
        Stacked mask tensor of shape [B, H, W]
    """
    if not masks:
        return create_empty_mask()

    return torch.cat(masks, dim=0)
