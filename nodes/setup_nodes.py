import torch
import comfy.model_management as mm
from ..utils.client_wrapper import MoondreamClientWrapper


class MoondreamClient:
    """Connect to a Moondream cloud API or local Moondream Station endpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Cloud API key (optional)",
                    },
                ),
                "endpoint": (
                    "STRING",
                    {
                        "default": "http://localhost:2020/v1",
                        "multiline": False,
                        "placeholder": "Moondream Station endpoint",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MOONDREAM_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "create_client"
    CATEGORY = "Moondream"

    def create_client(self, api_key: str = "", endpoint: str = "http://localhost:2020/v1"):
        import moondream as md

        api_key = api_key.strip()
        endpoint = endpoint.strip()

        has_api_key = bool(api_key)

        if has_api_key:
            # Use cloud API
            client = md.vl(api_key=api_key)
        elif endpoint:
            # Use local endpoint (Moondream Station)
            client = md.vl(endpoint=endpoint)
        else:
            raise ValueError(
                "Either api_key or endpoint must be provided. "
                "For cloud API, provide an api_key. "
                "For local inference, provide a Moondream Station endpoint."
            )

        wrapper = MoondreamClientWrapper(
            client=client,
            has_api_key=has_api_key,
            is_local=False,
            device=None,
        )

        return (wrapper,)


class MoondreamLoader:
    """Load a Moondream model locally using Transformers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo": (
                    "STRING",
                    {
                        "default": "moondream/moondream3-preview",
                        "multiline": False,
                    },
                ),
                "load_device": (["gpu", "cpu"],),
            },
        }

    RETURN_TYPES = ("MOONDREAM_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "load_model"
    CATEGORY = "Moondream"

    def load_model(self, repo: str, load_device: str = "gpu"):
        from transformers import AutoConfig, AutoModelForCausalLM

        # Validate model architecture
        config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        architectures = getattr(config, "architectures", []) or []

        is_moondream = any(
            "Moondream" in arch or "HfMoondream" in arch for arch in architectures
        )
        if not is_moondream:
            raise ValueError(
                f"Model {repo} does not appear to be a Moondream model. "
                f"Architectures: {architectures}"
            )

        # Determine target device
        if load_device == "gpu":
            if torch.cuda.is_available():
                device = mm.get_torch_device()
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        offload_device = mm.unet_offload_device()
        dtype = torch.float16 if device.type != "cpu" else torch.float32

        # Load model to offload device (CPU) initially
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(offload_device)

        # Create a wrapper that mimics the moondream client interface
        local_client = LocalMoondreamModel(model, device, offload_device)

        wrapper = MoondreamClientWrapper(
            client=local_client,
            has_api_key=False,
            is_local=True,
            device=str(device),
        )

        return (wrapper,)


class LocalMoondreamModel:
    """Wrapper to make a local Transformers model have the same interface as md.Client."""

    def __init__(self, model, device: torch.device, offload_device: torch.device):
        self.model = model
        self.device = device
        self.offload_device = offload_device

    def _move_to_device(self):
        """Move model to inference device before running."""
        self.model.to(self.device)

    def _offload(self):
        """Move model back to offload device and clear cache."""
        self.model.to(self.offload_device)
        mm.soft_empty_cache()

    def caption(self, image, length: str = "normal", **kwargs):
        """Generate a caption for the image."""
        self._move_to_device()
        try:
            return self.model.caption(image, length=length)
        finally:
            self._offload()

    def query(self, image, question: str, **kwargs):
        """Answer a question about the image."""
        self._move_to_device()
        try:
            return self.model.query(image, question)
        finally:
            self._offload()

    def detect(self, image, target: str):
        """Detect objects in the image."""
        self._move_to_device()
        try:
            return self.model.detect(image, target)
        finally:
            self._offload()

    def point(self, image, target: str):
        """Find points for objects in the image."""
        self._move_to_device()
        try:
            return self.model.point(image, target)
        finally:
            self._offload()


class MoondreamSettings:
    """Configure generation settings for Moondream inference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 4096, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("MOONDREAM_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "create_settings"
    CATEGORY = "Moondream"

    def create_settings(
        self, temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 512
    ):
        settings = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        return (settings,)

