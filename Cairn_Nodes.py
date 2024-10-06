import logging
from random import random
from easy_nodes import (
    NumberInput,
    ComfyNode,
    MaskTensor,
    StringInput,
    ImageTensor,
    Choice,
    ModelTensor,
    ConditioningTensor,
)
import easy_nodes
import torch
import comfy.sample
import comfy.utils
import latent_preview


# Define the common k-sampler function
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    # Denoising process using the model
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
        denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
        force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed
    )

    out = latent.copy()
    out["samples"] = samples
    return (out, )

# Define the node class using @ComfyNode decorator
@ComfyNode(category="Sampling", display_name="KSampler", description="Denoises a latent image using a provided model and conditioning", color="#44AA88")
class KSampler:
    """
    Uses the provided model, positive and negative conditioning to denoise the latent image.
    """
    
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "model": ModelTensor({"tooltip": "The model used for denoising the input latent."}),
                "seed": NumberInput(0, 0, 0xffffffffffffffff, display="slider", step=1, tooltip="Random seed used for noise generation."),
                "steps": NumberInput(20, 1, 10000, display="slider", step=1, tooltip="Number of denoising steps."),
                "cfg": NumberInput(8.0, 0.0, 100.0, step=0.1, tooltip="Classifier-Free Guidance scale for controlling creativity."),
                "sampler_name": comfy.samplers.KSampler.SAMPLERS,
                "scheduler": comfy.samplers.KSampler.SCHEDULERS,
                "positive": ConditioningTensor({"tooltip": "Conditioning for desired attributes in the output image."}),
                "negative": ConditioningTensor({"tooltip": "Conditioning for undesired attributes in the output image."}),
                "latent_image": ConditioningTensor({"tooltip": "Latent image to denoise."}),
                "denoise": NumberInput(1.0, 0.0, 1.0, step=0.01, tooltip="Degree of denoising to apply. Lower values preserve more initial image structure."),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)

    # The function to process the sampling
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise
        )

