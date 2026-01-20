import torch
import comfy.model_management
from ..utils.client_wrapper import MoondreamClientWrapper
from ..utils.comfy_model_wrapper import MoondreamModelWrapper

# Cache for loaded local models
_model_cache: dict[str, MoondreamClientWrapper] = {}


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
                "keep_loaded": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MOONDREAM_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "load_model"
    CATEGORY = "Moondream"

    def load_model(
        self, repo: str, load_device: str = "gpu", keep_loaded: bool = True
    ):
        from transformers import AutoConfig, AutoModelForCausalLM

        cache_key = f"{repo}_{load_device}"

        # Always check cache first - return cached model if available
        if cache_key in _model_cache:
            cached = _model_cache[cache_key]
            if not keep_loaded:
                # Remove from cache so it won't be kept after this use
                del _model_cache[cache_key]
            return (cached,)

        # Clear cache - ComfyUI will handle actual unloading
        _model_cache.clear()

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

        # Determine device
        if load_device == "gpu":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = "cpu"

        # Load model - Moondream handles tokenization internally
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map={"": device},
        )

        # Wrap and register with ComfyUI's model management
        load_dev = torch.device(device)
        offload_dev = torch.device("cpu")
        comfy_wrapper = MoondreamModelWrapper(model, load_dev, offload_dev)
        comfy.model_management.load_models_gpu([comfy_wrapper])

        # Create a wrapper that mimics the moondream client interface
        local_client = LocalMoondreamModel(model, device)

        wrapper = MoondreamClientWrapper(
            client=local_client,
            comfy_model=comfy_wrapper,
            has_api_key=False,
            is_local=True,
            device=device,
        )

        # Cache if requested
        if keep_loaded:
            _model_cache[cache_key] = wrapper

        return (wrapper,)


class LocalMoondreamModel:
    """Wrapper to make a local Transformers model have the same interface as md.Client."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    def caption(self, image, length: str = "normal", **kwargs):
        """Generate a caption for the image."""
        # Moondream model has a built-in caption method
        return self.model.caption(image, length=length)

    def query(self, image, question: str, **kwargs):
        """Answer a question about the image."""
        # Moondream model has a built-in query method
        return self.model.query(image, question)

    def detect(self, image, target: str):
        """Detect objects in the image."""
        # Moondream model has a built-in detect method
        return self.model.detect(image, target)

    def point(self, image, target: str):
        """Find points for objects in the image."""
        # Moondream model has a built-in point method
        return self.model.point(image, target)


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


