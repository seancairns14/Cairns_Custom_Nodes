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
    ConditioningTensor,
    AnyType
    
)

import easy_nodes
import comfy.samplers
import comfy.utils
import latent_preview
import comfy.sd

import re



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

def text_encode(clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([[cond, output]], )


def extract_prompt_list(input_str: str):
    # Regular expression pattern to match the strings inside the brackets and quotes
    pattern = r"'([^']*)'"
    
    # Use re.findall to extract all matches (each string inside the quotes)
    prompts = re.findall(pattern, input_str)
    
    # Now, for each prompt, split by commas outside the quotes
    result = [prompt.split(',') for prompt in prompts]
    
    return result

def concat_prompt_lists(list1, list2):
    # Concatenate both lists
    combined_list = list1 + list2
    
    # Join all elements of the combined list into a single string
    combined_prompt = ' '.join(combined_list)  # You can change the separator if needed
    
    return combined_prompt


def decode(vae, samples):
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images, )


class RepeatPipe:
    def __init__(self) -> None:
        self.model = None  # Expected to be a ModelTensor
        self.pos = None    # Expected to be a ConditioningTensor
        self.neg = None    # Expected to be a ConditioningTensor
        self.latent = None # Expected to be a LatentTensor
        self.vae = None    # Expected to be a VAE
        self.clip = None
        self.prompt = None
        self.image = None

# Register the class as a pipeline type
easy_nodes.register_type(RepeatPipe, "PIPELINE")
easy_nodes.create_field_setter_node(RepeatPipe)

# Define a custom node that will create and return a RepeatPipe instance
@ComfyNode()
def RepeatPipe_IN(model: ModelTensor=None, pos: ConditioningTensor=None, neg: ConditioningTensor=None, 
                  latent: LatentTensor=None, vae: comfy.sd.VAE=None, clip: AnyType=None, prompt: str=None, image: ImageTensor = None) -> list[RepeatPipe]:
    
    # Instantiate the RepeatPipe
    pipe = RepeatPipe()

    # Set the attributes based on the input arguments (optional)
    pipe.model = model
    pipe.pos = pos
    pipe.neg = neg
    pipe.latent = latent
    pipe.vae = vae
    pipe.clip = clip
    pipe.prompt = prompt
    pipe.image = image

    pipe = [pipe]
    if not isinstance(pipe, list) or not all(isinstance(item, RepeatPipe) for item in pipe):
        raise ValueError(f"RepeatPipe must be a list of RepeatPipe objects. Instead, got {type(pipe)} with elements of type {[type(item) for item in pipe]}.")

    


    return pipe  # Return the pipeline object


@ComfyNode()
def RepeatPipe_OUT(model: ModelTensor=None, pos: ConditioningTensor=None, neg: ConditioningTensor=None, 
                  latent: LatentTensor=None, vae: comfy.sd.VAE=None, clip: AnyType=None, prompt: str=None, image: ImageTensor = None) -> list[RepeatPipe]:
    
    # Instantiate the RepeatPipe
    pipe = RepeatPipe()

    # Set the attributes based on the input arguments (optional)
    pipe.model = model
    pipe.pos = pos
    pipe.neg = neg
    pipe.latent = latent
    pipe.vae = vae
    pipe.clip = clip
    pipe.prompt = prompt
    pipe.image = image

    pipe = [pipe]
    if not isinstance(pipe, list) or not all(isinstance(item, RepeatPipe) for item in pipe):
        raise ValueError(f"RepeatPipe must be a list of RepeatPipe objects. Instead, got {type(pipe)} with elements of type {[type(item) for item in pipe]}.")

    


    return pipe  # Return the pipeline object



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
def repeat_ksample(repeat_pipes: list[RepeatPipe] = None,
                   seed: int = NumberInput(0, 0, 0xffffffffffffffff, step=1),
                   steps: int = NumberInput(20, 1, 10000, step=1),
                   cfg: float = NumberInput(8.0, 0.0, 100.0, step=0.1),
                   sampler_name: str = Choice(comfy.samplers.KSampler.SAMPLERS),
                   scheduler_name: str = Choice(comfy.samplers.KSampler.SCHEDULERS),   
                   denoise: float = NumberInput(1.0, 0.0, 1.0, step=0.01),
                   text: str = StringInput("Example: ['This prompt, is, one prompt', 'This is, another']")) -> list[LatentTensor]:

    if repeat_pipes is None or len(repeat_pipes) == 0:
        raise ValueError(f"RepeatPipe must be provided. Instead {type(repeat_pipes)} was provided.")
    new_pipes = []
    for pipe in repeat_pipes:

        model = pipe
        positive = pipe.pos
        negative = pipe.neg
        latent_image = pipe.latent
        vae = pipe.vae
        clip = pipe.clip
        prompt = pipe.prompt 
        image = pipe.image

        latent_image, positive, negative = ensure_defaults(model, latent_image, positive, negative, seed)

        text_list = extract_prompt_list(text)
        prompt_list = extract_prompt_list(prompt)

        for p in prompt_list:
            for t in text_list:
                new_prompt = concat_prompt_lists(p, t)
                new_positive = text_encode(clip=clip, text=new_prompt)
                new_latent = common_ksampler(
                    model=model, seed=seed, steps=steps, cfg=cfg,
                    sampler_name=sampler_name, scheduler=scheduler_name,
                    positive=new_positive, negative=negative, latent=latent_image,
                    denoise=denoise
                )
                new_pipe = RepeatPipe()
                new_pipe.model = model
                new_pipe.pos = new_positive
                new_pipe.neg = negative
                new_pipe.latent = new_latent
                new_pipe.vae = vae
                new_pipe.clip = clip
                new_pipe.prompt = new_prompt
                new_pipe.image = decode(vae=new_pipe.vae, samples=new_latent)
                new_pipes.append(new_pipe)


    return new_pipes
