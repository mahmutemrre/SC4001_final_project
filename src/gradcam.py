"""
Grad-CAM visualisation utilities.

Generates class-activation heatmaps for CNN models to show which spatial
regions influence predictions most (Selvaraju et al., 2017).  For the ViT
model, attention rollout is used instead.
"""
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Grad-CAM for CNN models
# ---------------------------------------------------------------------------
class GradCAM:
    """Compute Grad-CAM for a given CNN model and target convolutional layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(self, x, class_idx=None):
        """Return Grad-CAM heatmap of shape (H, W) for input x (1, C, H, W)."""
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward()

        # Global-average-pool the gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx


# ---------------------------------------------------------------------------
# Attention rollout for ViT
# ---------------------------------------------------------------------------
def attention_rollout(model, x):
    """Compute attention rollout for SimplViT and return (H, W) map.

    Hooks into every MultiheadAttention layer, multiplies attention
    matrices across layers, and reshapes the CLS token's attention over
    patches into a spatial map.
    """
    attentions = []

    def hook_fn(module, input, output):
        # nn.MultiheadAttention returns (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            attentions.append(output[1].detach())  # (B, N, N)

    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(hook_fn))

    # Need attention weights — set need_weights=True temporarily
    orig_flags = []
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            orig_flags.append(getattr(module, 'need_weights', True))
            module.need_weights = True

    model.eval()
    with torch.no_grad():
        _ = model(x)

    # Restore flags and remove hooks
    i = 0
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            module.need_weights = orig_flags[i]
            i += 1
    for h in hooks:
        h.remove()

    if not attentions:
        return np.zeros((7, 7))  # fallback

    # Rollout: multiply attention matrices layer by layer
    result = attentions[0].squeeze(0).cpu().numpy()  # (N, N)
    for attn in attentions[1:]:
        attn_np = attn.squeeze(0).cpu().numpy()
        result = attn_np @ result

    # CLS token's attention over patch tokens (skip CLS→CLS)
    cls_attn = result[0, 1:]   # (num_patches,)
    num_patches = len(cls_attn)
    grid_size = int(np.sqrt(num_patches))
    cam = cls_attn.reshape(grid_size, grid_size)

    # Normalise to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


# ---------------------------------------------------------------------------
# Helper: get the target convolutional layer for each CNN model
# ---------------------------------------------------------------------------
def get_target_layer(model):
    """Return the last convolutional layer suitable for Grad-CAM."""
    # Walk backwards through named modules to find the last Conv2d
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv
