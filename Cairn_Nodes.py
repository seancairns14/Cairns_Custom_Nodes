import torch
from easy_nodes import (
    NumberInput,
    ComfyNode,
    MaskTensor,
    StringInput,
    ImageTensor,
    Choice,
    LatentTensor,
    ModelTensor,
    ConditioningTensor
)
import easy_nodes
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


class RepeatPipe:
    def __init__(self) -> None:
        self.model = None  # Expected to be a ModelTensor
        self.pos = None    # Expected to be a ConditioningTensor
        self.neg = None    # Expected to be a ConditioningTensor
        self.latent = None # Expected to be a LatentTensor
        self.vae = None    # Expected to be a VAE

# Register the class as a pipeline type
easy_nodes.register_type(RepeatPipe, "PIPELINE")
easy_nodes.create_field_setter_node(RepeatPipe)

# Define a custom node that will create and return a RepeatPipe instance
@ComfyNode()
def RepeatPipe_IN(model=None, pos=None, neg=None, latent=None, vae=None) -> list[RepeatPipe]:
    """
    A node that creates and returns an instance of the RepeatPipe class.
    
    Args:
        model: The model tensor to be used in the pipeline.
        pos: Positive conditioning tensor.
        neg: Negative conditioning tensor.
        latent: Latent tensor.
        vae: VAE for the pipeline.
    
    Returns:
        RepeatPipe: An instance of the RepeatPipe class with set fields.
    """
    # Instantiate the RepeatPipe
    pipe = RepeatPipe()

    # Set the attributes based on the input arguments (optional)
    pipe.model = model
    pipe.pos = pos
    pipe.neg = neg
    pipe.latent = latent
    pipe.vae = vae

    return [pipe]  # Return the pipeline object



def ensure_defaults(model, latent_image, positive, negative, seed):
    if latent_image is None:
        latent_image = comfy.sample.create_random_latent(model.width, model.height, seed)
    if positive is None:
        positive = ConditioningTensor.default_positive()
    if negative is None:
        negative = ConditioningTensor.default_negative()
    return latent_image, positive, negative


@ComfyNode()
def Cairns_ksample(model: ModelTensor = None,
                   seed: int = NumberInput(0, 0, 0xffffffffffffffff, step=1),
                   steps: int = NumberInput(20, 1, 10000, step=1),
                   cfg: float = NumberInput(8.0, 0.0, 100.0, step=0.1),
                   sampler_name: str = Choice(comfy.samplers.KSampler.SAMPLERS),
                   scheduler_name: str = Choice(comfy.samplers.KSampler.SCHEDULERS),
                   positive: ConditioningTensor = None,  
                   negative: ConditioningTensor = None,  
                   latent_image: LatentTensor = None,
                   denoise: float = NumberInput(1.0, 0.0, 1.0, step=0.01)) -> LatentTensor:

    latent_image, positive, negative = ensure_defaults(model, latent_image, positive, negative, seed)

    return common_ksampler(
        model=model, seed=seed, steps=steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler_name,
        positive=positive, negative=negative, latent=latent_image,
        denoise=denoise
    )


@ComfyNode()
def repeat_ksample(repeat_pipe: list[RepeatPipe] = None,
                   seed: int = NumberInput(0, 0, 0xffffffffffffffff, step=1),
                   steps: int = NumberInput(20, 1, 10000, step=1),
                   cfg: float = NumberInput(8.0, 0.0, 100.0, step=0.1),
                   sampler_name: str = Choice(comfy.samplers.KSampler.SAMPLERS),
                   scheduler_name: str = Choice(comfy.samplers.KSampler.SCHEDULERS),   
                   denoise: float = NumberInput(1.0, 0.0, 1.0, step=0.01)) -> list[LatentTensor]:

    if repeat_pipe[0] is None:
        raise ValueError("RepeatPipe must be provided.")

    model = repeat_pipe[0].model
    positive = repeat_pipe[0].pos
    negative = repeat_pipe[0].neg
    latent_image = repeat_pipe[0].latent

    latent_image, positive, negative = ensure_defaults(model, latent_image, positive, negative, seed)

    return common_ksampler(
        model=model, seed=seed, steps=steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler_name,
        positive=positive, negative=negative, latent=latent_image,
        denoise=denoise
    )
