import torch
from easy_nodes import ComfyNode, NumberInput, ModelTensor, ConditioningTensor, LatentTensor
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

# Define the node using @ComfyNode decorator
@ComfyNode(category="Sampling", display_name="KSampler", description="Denoises a latent image using a provided model and conditioning", color="#44AA88")
def ksampler(
    model: ModelTensor,
    seed: int = NumberInput(0, 0, 0xffffffffffffffff, display="slider", step=1),
    steps: int = NumberInput(20, 1, 10000, display="slider", step=1,
    cfg: float = NumberInput(8.0, 0.0, 100.0, step=0.1),
    sampler_name: str = comfy.samplers.KSampler.SAMPLERS,
    scheduler: str = comfy.samplers.KSampler.SCHEDULERS,
    positive: ConditioningTensor = None,  # Default value provided
    negative: ConditioningTensor = None,  # Default value provided
    latent_image: LatentTensor = None,    # Default value provided
    denoise: float = NumberInput(1.0, 0.0, 1.0, step=0.01)
) -> LatentTensor:
    """
    Uses the provided model, positive and negative conditioning to denoise the latent image.
    """
    return common_ksampler(
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise
    )
