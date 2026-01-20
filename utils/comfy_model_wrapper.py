"""Wrapper to integrate Moondream models with ComfyUI's model management system."""

import torch
import comfy.model_management


class MoondreamModelWrapper:
    """Minimal wrapper to integrate with ComfyUI's model management.

    This implements the interface required by ComfyUI's LoadedModel class,
    allowing the model to be automatically unloaded when VRAM is needed.
    """

    def __init__(self, model, load_device: torch.device, offload_device: torch.device):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.parent = None
        self._size = None
        self._current_device = load_device
        self._loaded_memory = self.model_size()

    def model_size(self) -> int:
        """Return total bytes of all model parameters."""
        if self._size is None:
            self._size = comfy.model_management.module_size(self.model)
        return self._size

    def loaded_size(self) -> int:
        """Return bytes currently loaded on the load device."""
        return self._loaded_memory

    def model_patches_to(self, device):
        """Move model patches to device (no-op for Moondream)."""
        pass

    def model_patches_models(self):
        """Return list of sub-models from patches (empty for Moondream)."""
        return []

    def lowvram_patch_counter(self):
        """Return lowvram patch counter."""
        return 0

    def current_loaded_device(self):
        """Return the device where the model is currently loaded."""
        return self._current_device

    def model_dtype(self):
        """Return the model's dtype."""
        return next(self.model.parameters()).dtype

    def partially_load(self, device_to, memory_to_load=0, force_patch_weights=False):
        """Move model weights to the specified device and return bytes loaded.

        For simplicity, this loads everything rather than partially.
        """
        self.model.to(device_to)
        self._current_device = device_to
        self._loaded_memory = self.model_size()
        return self._loaded_memory

    def partially_unload(self, device_to, memory_to_free=0):
        """Move model weights to the specified device and return bytes freed.

        For simplicity, this unloads everything rather than partially.
        """
        self.model.to(device_to)
        self._current_device = device_to
        freed = self._loaded_memory
        self._loaded_memory = 0
        return freed

    def detach(self, unpatch_weights=True):
        """Full cleanup - move model to offload device."""
        self.model.to(self.offload_device)
        self._current_device = self.offload_device
        self._loaded_memory = 0

    def is_clone(self, other):
        """Check if this wrapper refers to the same model as another."""
        return hasattr(other, 'model') and self.model is other.model
