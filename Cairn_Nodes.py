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

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )


@ComfyNode(category="Image Processing",
           display_name="K-Sample Image with Latent",
           description="Uses common_ksampler to sample images from latent space.",
           color="#336699")
def k_sample_with_latent(model, seed: int, steps: int = 50, cfg: float = 7.5, 
                         sampler_name: str = "ddim", scheduler: str = "default",
                         positive: list = [], negative: list = [], latent: dict = {},
                         denoise: float = 1.0, disable_noise: bool = False) -> ImageTensor:
    """
    This node applies K-sampling to a latent image tensor using common_ksampler.
    """
    
    # Use the provided function to perform the latent space sampling
    result_latent = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                    positive, negative, latent, denoise, disable_noise)
    
    # Extract the generated latent image from the output
    latent_image = result_latent[0]["samples"]
    
    # Convert latent space back into image tensor
    output_image = model.decode(latent_image)
    
    # Show the resulting image
    easy_nodes.show_image(output_image)
    
    return output_image

