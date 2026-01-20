import json
import torch
from ..utils.client_wrapper import MoondreamClientWrapper
from ..utils.image_utils import (
    tensor_to_pil,
    pil_to_tensor,
    create_empty_image,
    crop_image,
    normalize_to_pixel_coords,
    combine_boxes,
)
from ..utils.mask_utils import (
    create_mask_from_boxes,
    create_empty_mask,
    svg_path_to_mask,
    combine_masks,
    stack_masks,
)


class MoondreamCaption:
    """Generate a caption for an image using Moondream."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "client": ("MOONDREAM_CLIENT",),
                "length": (["short", "normal", "long"],),
            },
            "optional": {
                "settings": ("MOONDREAM_SETTINGS",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "Moondream"

    def caption(
        self,
        image: torch.Tensor,
        client: MoondreamClientWrapper,
        length: str = "normal",
        settings: dict = None,
    ):
        pil_image = tensor_to_pil(image)
        caption = client.caption(pil_image, length=length, settings=settings)

        # Handle different response formats
        if isinstance(caption, dict):
            caption = caption.get("caption", str(caption))

        return (str(caption),)


class MoondreamQuery:
    """Ask a question about an image using Moondream."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "client": ("MOONDREAM_CLIENT",),
                "query": (
                    "STRING",
                    {
                        "default": "What is in this image?",
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "reasoning": ("BOOLEAN", {"default": False}),
                "structured_output": ("BOOLEAN", {"default": False}),
                "settings": ("MOONDREAM_SETTINGS",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "answer_structured", "reasoning_trace")
    FUNCTION = "query"
    CATEGORY = "Moondream"

    def query(
        self,
        image: torch.Tensor,
        client: MoondreamClientWrapper,
        query: str,
        reasoning: bool = False,
        structured_output: bool = False,
        settings: dict = None,
    ):
        pil_image = tensor_to_pil(image)

        # Modify query for structured output
        actual_query = query
        if structured_output:
            actual_query = f'A JSON array with keys: {query}.'

        # Add reasoning instruction if enabled
        if reasoning:
            actual_query = f"Think step by step and explain your reasoning. {actual_query}"

        response = client.query(pil_image, actual_query, settings=settings)

        # Handle different response formats
        if isinstance(response, dict):
            answer = response.get("answer", str(response))
            reasoning_trace = response.get("reasoning", "")
        else:
            answer = str(response)
            reasoning_trace = ""

        # Try to extract reasoning from the answer if not provided separately
        if reasoning and not reasoning_trace:
            # Look for common reasoning patterns
            if "Therefore" in answer or "So," in answer:
                parts = answer.rsplit("Therefore", 1)
                if len(parts) == 2:
                    reasoning_trace = parts[0].strip()
                    answer = "Therefore" + parts[1]

        # For structured output, try to parse as JSON
        answer_structured = ""
        if structured_output:
            try:
                parsed = json.loads(answer)
                answer_structured = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                start = answer.find("[")
                end = answer.rfind("]") + 1
                if start != -1 and end > start:
                    try:
                        parsed = json.loads(answer[start:end])
                        answer_structured = json.dumps(parsed, indent=2)
                    except json.JSONDecodeError:
                        answer_structured = answer

        return (answer, answer_structured, reasoning_trace)


class MoondreamPoint:
    """Find points for objects in an image using Moondream."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "client": ("MOONDREAM_CLIENT",),
                "target": (
                    "STRING",
                    {
                        "default": "object",
                        "multiline": False,
                    },
                ),
            },
            "optional": {
                "settings": ("MOONDREAM_SETTINGS",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("points",)
    FUNCTION = "point"
    CATEGORY = "Moondream"

    def point(
        self,
        image: torch.Tensor,
        client: MoondreamClientWrapper,
        target: str,
        settings: dict = None,
    ):
        pil_image = tensor_to_pil(image)
        result = client.point(pil_image, target)

        # Convert result to JSON string
        if isinstance(result, dict):
            points_json = json.dumps(result, indent=2)
        else:
            points_json = json.dumps({"points": result}, indent=2)

        return (points_json,)


class MoondreamDetect:
    """Detect and crop objects in an image using Moondream."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "client": ("MOONDREAM_CLIENT",),
                "target": (
                    "STRING",
                    {
                        "default": "object",
                        "multiline": False,
                    },
                ),
            },
            "optional": {
                "combine": ("BOOLEAN", {"default": False}),
                "settings": ("MOONDREAM_SETTINGS",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cropped", "mask")
    FUNCTION = "detect"
    CATEGORY = "Moondream"

    def detect(
        self,
        image: torch.Tensor,
        client: MoondreamClientWrapper,
        target: str,
        combine: bool = False,
        settings: dict = None,
    ):
        pil_image = tensor_to_pil(image)
        width, height = pil_image.size

        result = client.detect(pil_image, target)

        # Extract objects from result
        if isinstance(result, dict):
            objects = result.get("objects", [])
        else:
            objects = []

        # No detections - return empty image and mask
        if not objects:
            empty_image = create_empty_image(64, 64)
            empty_mask = create_empty_mask(width, height)
            return (empty_image, empty_mask)

        # Convert normalized boxes to pixel coordinates
        pixel_boxes = []
        for obj in objects:
            if "x_min" in obj:
                box = normalize_to_pixel_coords(obj, width, height)
            elif "bbox" in obj:
                bbox = obj["bbox"]
                box = normalize_to_pixel_coords(
                    {
                        "x_min": bbox[0],
                        "y_min": bbox[1],
                        "x_max": bbox[2],
                        "y_max": bbox[3],
                    },
                    width,
                    height,
                )
            else:
                continue
            pixel_boxes.append(box)

        if not pixel_boxes:
            empty_image = create_empty_image(64, 64)
            empty_mask = create_empty_mask(width, height)
            return (empty_image, empty_mask)

        if combine:
            # Combine all boxes into one bounding box
            combined_box = combine_boxes(pixel_boxes)
            cropped_pil = crop_image(pil_image, combined_box)
            cropped_tensor = pil_to_tensor(cropped_pil)
            mask = create_mask_from_boxes([combined_box], width, height)
        else:
            # Create separate crops for each box
            crops = []
            for box in pixel_boxes:
                cropped_pil = crop_image(pil_image, box)
                cropped_tensor = pil_to_tensor(cropped_pil)
                crops.append(cropped_tensor)

            # Stack crops into a batch (need to resize to same size first)
            # For simplicity, we'll just return all crops at their original sizes
            # This requires finding the max dimensions and padding
            if len(crops) == 1:
                cropped_tensor = crops[0]
            else:
                # Find max dimensions
                max_h = max(c.shape[1] for c in crops)
                max_w = max(c.shape[2] for c in crops)

                # Pad all crops to max dimensions
                padded_crops = []
                for crop in crops:
                    h, w = crop.shape[1], crop.shape[2]
                    if h < max_h or w < max_w:
                        padded = torch.zeros(1, max_h, max_w, 3, dtype=crop.dtype)
                        padded[:, :h, :w, :] = crop
                        padded_crops.append(padded)
                    else:
                        padded_crops.append(crop)

                cropped_tensor = torch.cat(padded_crops, dim=0)

            # Create mask with all boxes
            mask = create_mask_from_boxes(pixel_boxes, width, height)

        return (cropped_tensor, mask)


class MoondreamSegment:
    """Segment objects in an image using Moondream (Cloud API only)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "client": ("MOONDREAM_CLIENT",),
                "target": (
                    "STRING",
                    {
                        "default": "object",
                        "multiline": False,
                    },
                ),
            },
            "optional": {
                "multi_object": ("BOOLEAN", {"default": False}),
                "settings": ("MOONDREAM_SETTINGS",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment"
    CATEGORY = "Moondream"

    def segment(
        self,
        image: torch.Tensor,
        client: MoondreamClientWrapper,
        target: str,
        multi_object: bool = False,
        settings: dict = None,
    ):
        # Check if segmentation is supported
        if not client.supports_segment():
            raise RuntimeError(
                "Segmentation is only available with the cloud API. "
                "Please provide an API key to MoondreamClient."
            )

        pil_image = tensor_to_pil(image)
        width, height = pil_image.size

        if multi_object:
            # First detect all objects, then segment each with a spatial hint
            detect_result = client.detect(pil_image, target)

            if isinstance(detect_result, dict):
                objects = detect_result.get("objects", [])
            else:
                objects = []

            if not objects:
                return (create_empty_mask(width, height),)

            masks = []
            for obj in objects:
                # Get center point of the bounding box as spatial hint
                if "x_min" in obj:
                    center_x = (obj["x_min"] + obj["x_max"]) / 2
                    center_y = (obj["y_min"] + obj["y_max"]) / 2
                elif "bbox" in obj:
                    bbox = obj["bbox"]
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                else:
                    continue

                # Segment with the center point as hint
                try:
                    result = client.segment(
                        pil_image, target, points=[(center_x, center_y)]
                    )
                    mask = self._parse_segment_result(result, width, height, obj)
                    masks.append(mask)
                except Exception:
                    # Fall back to bounding box mask
                    if "x_min" in obj:
                        box = normalize_to_pixel_coords(obj, width, height)
                    else:
                        bbox = obj["bbox"]
                        box = normalize_to_pixel_coords(
                            {
                                "x_min": bbox[0],
                                "y_min": bbox[1],
                                "x_max": bbox[2],
                                "y_max": bbox[3],
                            },
                            width,
                            height,
                        )
                    masks.append(create_mask_from_boxes([box], width, height))

            if not masks:
                return (create_empty_mask(width, height),)

            # Combine all masks
            combined_mask = combine_masks(masks)
            return (combined_mask,)

        else:
            # Single object segmentation
            result = client.segment(pil_image, target)
            mask = self._parse_segment_result(result, width, height)
            return (mask,)

    def _parse_segment_result(
        self, result, width: int, height: int, fallback_obj: dict = None
    ) -> torch.Tensor:
        """Parse segmentation result and convert to mask."""
        # Handle different response formats
        if isinstance(result, dict):
            # Check for SVG path
            if "svg" in result:
                svg_path = result["svg"]
                fallback_box = None
                if fallback_obj:
                    if "x_min" in fallback_obj:
                        fallback_box = normalize_to_pixel_coords(
                            fallback_obj, width, height
                        )
                return svg_path_to_mask(svg_path, width, height, fallback_box)

            # Check for mask data
            if "mask" in result:
                mask_data = result["mask"]
                if isinstance(mask_data, str):
                    # SVG path
                    return svg_path_to_mask(mask_data, width, height)

            # Check for path attribute
            if "path" in result:
                return svg_path_to_mask(result["path"], width, height)

        # Fallback to empty mask
        return create_empty_mask(width, height)
