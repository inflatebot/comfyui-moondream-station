from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MoondreamClientWrapper:
    """Wrapper around moondream client/model with metadata."""

    client: Any
    comfy_model: Any = None  # MoondreamModelWrapper for ComfyUI integration
    has_api_key: bool = False
    is_local: bool = False
    device: Optional[str] = None
    _unloaded: bool = field(default=False, repr=False)

    def supports_segment(self) -> bool:
        """Check if this client supports segmentation (cloud API only)."""
        return self.has_api_key

    def unload(self) -> None:
        """Mark the model as unloaded.

        Note: Actual memory management is handled by ComfyUI's model management
        system. This method exists for API compatibility.
        """
        if self._unloaded:
            return

        if not self.is_local:
            # Nothing to unload for API clients
            self._unloaded = True
            return

        # ComfyUI handles actual unloading via the comfy_model wrapper
        # Just mark as unloaded for tracking purposes
        self._unloaded = True

    def is_loaded(self) -> bool:
        """Check if the model is still loaded."""
        return not self._unloaded

    def caption(self, image, length: str = "normal", settings: Optional[dict] = None):
        """Generate a caption for the image."""
        kwargs = {}
        if settings:
            if "max_tokens" in settings:
                kwargs["max_tokens"] = settings["max_tokens"]

        return self.client.caption(image, length=length, **kwargs)

    def query(self, image, question: str, settings: Optional[dict] = None):
        """Answer a question about the image."""
        kwargs = {}
        if settings:
            if "max_tokens" in settings:
                kwargs["max_tokens"] = settings["max_tokens"]

        return self.client.query(image, question, **kwargs)

    def detect(self, image, target: str):
        """Detect objects in the image."""
        try:
            return self.client.detect(image, target)
        except KeyError as e:
            raise RuntimeError(
                f"Moondream detect API returned unexpected response format (missing key: {e}). "
                "This may indicate a version mismatch between the moondream client library "
                "and your Moondream Station server. Try updating both to the latest version."
            ) from e

    def point(self, image, target: str):
        """Find points for objects in the image."""
        try:
            return self.client.point(image, target)
        except KeyError as e:
            raise RuntimeError(
                f"Moondream point API returned unexpected response format (missing key: {e}). "
                "This may indicate a version mismatch between the moondream client library "
                "and your Moondream Station server. Try updating both to the latest version."
            ) from e

    def segment(self, image, target: str, points: Optional[list] = None):
        """Segment objects in the image (cloud API only)."""
        if not self.supports_segment():
            raise RuntimeError(
                "Segmentation is only available with the cloud API. "
                "Please provide an API key to MoondreamClient."
            )

        kwargs = {}
        if points:
            kwargs["points"] = points

        try:
            return self.client.segment(image, target, **kwargs)
        except KeyError as e:
            raise RuntimeError(
                f"Moondream segment API returned unexpected response format (missing key: {e}). "
                "This may indicate a version mismatch between the moondream client library "
                "and the Moondream cloud API."
            ) from e
