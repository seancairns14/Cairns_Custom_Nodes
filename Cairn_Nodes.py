import logging
from random import random
from easy_nodes import (
    NumberInput,
    ComfyNode,
    MaskTensor,
    StringInput,
    ImageTensor,
    Choice,
)
import easy_nodes
import torch
import comfy.sample
import comfy.utils
import latent_preview


# Multiple outputs can be returned by annotating with tuple[].
# Pass return_names if you want to give them labels in ComfyUI.
@ComfyNode("Example category", color="#0066cc", bg_color="#ffcc00", return_names=["Below", "Above"])
def threshold_image(image: ImageTensor, threshold_value: float = NumberInput(0.5, 0, 1, 0.0001, display="slider")) -> tuple[MaskTensor, MaskTensor]:
    """Returns separate masks for values above and below the threshold value."""
    mask_below = torch.any(image < threshold_value, dim=-1).squeeze(-1)
    
    logging.info(f"Number of pixels below threshold: {mask_below.sum()}")
    logging.info(f"Number of pixels above threshold: {(~mask_below).sum()}")
    
    return mask_below.float(), (~mask_below).float()
