import torch
from easy_nodes import ComfyNode, NumberInput, ModelTensor, ConditioningTensor, LatentTensor, ImageTensor
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


@ComfyNode()
def create_random_image(width: int=NumberInput(128, 128, 1024), 
                        height: int=NumberInput(128, 128, 1024),
                        model: ModelTensor=None) -> ImageTensor:
    
    return torch.rand((1, height, width, 3))



