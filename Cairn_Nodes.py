import torch
from easy_nodes import ComfyNode, NumberInput, ModelTensor, ConditioningTensor, LatentTensor, ImageTensor, Choice
import comfy.samplers
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
def Cairns_ksample(model: ModelTensor=None, 
                   seed: int = NumberInput(0, 0, 0xffffffffffffffff, step=1),
                   steps: int = NumberInput(20, 1, 10000, step=1),
                   cfg: float = NumberInput(8.0, 0.0, 100.0, step=0.1),
                   sampler_name: str = Choice(comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SAMPLERS[0]),
                   scheduler_name: str = Choice(comfy.samplers.KSampler.SCHEDULERS, comfy.samplers.KSampler.SCHEDULERS[0]),
                   positive: ConditioningTensor = None,  
                   negative: ConditioningTensor = None,  
                   latent_image: LatentTensor = None,    
                   denoise: float = NumberInput(1.0, 0.0, 1.0, step=0.01)
                   ) -> ImageTensor:

    # Set default values for ConditioningTensors and LatentTensor if they are None
    if latent_image is None:
        latent_image = comfy.sample.create_random_latent(model.width, model.height, seed)

    if positive is None:
        positive = ConditioningTensor.default_positive()

    if negative is None:
        negative = ConditioningTensor.default_negative()

    # Prepare the latent dictionary for common_ksampler
    latent_dict = {"samples": latent_image}

    # Call the common ksampler function
    output_latent = common_ksampler(
        model=model,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler_name,
        positive=positive,
        negative=negative,
        latent=latent_dict,
        denoise=denoise
    )

    # Convert the latent to an ImageTensor (depending on your implementation of conversion)
    output_image = comfy.utils.latent_to_image(output_latent[0]["samples"])

    return output_image
