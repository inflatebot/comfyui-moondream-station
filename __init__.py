"""ComfyUI custom nodes for Moondream 3 vision-language model."""

from .nodes import (
    MoondreamClient,
    MoondreamLoader,
    MoondreamSettings,
    MoondreamCaption,
    MoondreamQuery,
    MoondreamPoint,
    MoondreamDetect,
    MoondreamSegment,
)

NODE_CLASS_MAPPINGS = {
    "MoondreamClient": MoondreamClient,
    "MoondreamLoader": MoondreamLoader,
    "MoondreamSettings": MoondreamSettings,
    "MoondreamCaption": MoondreamCaption,
    "MoondreamQuery": MoondreamQuery,
    "MoondreamPoint": MoondreamPoint,
    "MoondreamDetect": MoondreamDetect,
    "MoondreamSegment": MoondreamSegment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoondreamClient": "Moondream Client",
    "MoondreamLoader": "Moondream Loader",
    "MoondreamSettings": "Moondream Settings",
    "MoondreamCaption": "Moondream Caption",
    "MoondreamQuery": "Moondream Query",
    "MoondreamPoint": "Moondream Point",
    "MoondreamDetect": "Moondream Detect",
    "MoondreamSegment": "Moondream Segment",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
