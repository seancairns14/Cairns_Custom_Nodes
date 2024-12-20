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
from nodes import SaveImage
import folder_paths
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageOps, ImageSequence, ImageFile
from comfy.cli_args import args

import re
import json
import os
import numpy as np
import random

import logging


logging.basicConfig(
    level=logging.DEBUG,  # Log all messages of level DEBUG or higher
    format='%(asctime)s - %(levelname)s - %(message)s'  # Customize log message format
)

# Create a logger
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.DEBUG)  # Log all levels from DEBUG and above


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
    return out

def text_encode(clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([[cond, output]], )

def braces_to_list(input_str: str):
    """
    Extracts strings enclosed in curly braces from the input string,
    removes trailing commas, and trims whitespace.

    Args:
        input_str (str): The input string, e.g., "{string}, {string}, {string}".

    Returns:
        list: A list of cleaned strings, e.g., ["word, this, word", "another, test, word"].
    """
    # Use regex to find all content inside curly braces
    matches = re.findall(r"\{(.*?)\}", input_str)
    # Process each match: strip whitespace, remove trailing commas
    return [match.strip().rstrip(',') for match in matches if match.strip()]




def string_to_list(input_str: str):
    """
    Converts a comma-separated string into a list of trimmed strings,
    ensuring no spaces at the start or end of each string and no commas within strings.

    Args:
        input_str (str): The input string, e.g., "this, is, a, test".

    Returns:
        list: A list of strings, e.g., ['this', 'is', 'a', 'test'].
    """
    # Split the input string by commas, trim spaces, and remove commas within strings
    return [item.strip().replace(',', '') for item in input_str.split(',') if item.strip()]


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
def RepeatPipe_IN(model: ModelTensor, pos: ConditioningTensor, neg: ConditioningTensor, 
                  latent: LatentTensor, vae: comfy.sd.VAE, clip: AnyType, prompt: str, image: ImageTensor) -> list[RepeatPipe]:
    logging.info("Starting RepeatPipe_IN function")


    # Instantiate the RepeatPipe
    pipe = RepeatPipe()
    logging.info("Created new RepeatPipe instance")

    # Set the attributes based on the input arguments
    pipe.model = model
    pipe.pos = pos
    pipe.neg = neg
    pipe.latent = latent
    pipe.vae = vae
    pipe.clip = clip
    pipe.prompt = prompt
    pipe.image = image

    # Wrap the pipe in a list
    pipe = [pipe]
    logging.info("Wrapped RepeatPipe instance in a list")

    # Validate the output
    if not isinstance(pipe, list):
        logging.error("Output is not a list. Got: %s", type(pipe))
        raise ValueError(f"RepeatPipe must be a list of RepeatPipe objects. Instead, got {type(pipe)}.")
    if not all(isinstance(item, RepeatPipe) for item in pipe):
        item_types = [type(item) for item in pipe]
        logging.error("Not all elements in the list are RepeatPipe objects. Element types: %s", item_types)
        raise ValueError(f"RepeatPipe must be a list of RepeatPipe objects. Instead, got a list with elements of type {item_types}.")

    logging.info("Validated output: List of RepeatPipe objects")
    logging.info("Successfully created and returned a list of RepeatPipe objects with %d item(s)", len(pipe))
    
    return pipe  # Return the pipeline object


@ComfyNode()
def RepeatPipe_OUT(repeat_pipes: list[RepeatPipe]) -> list[ImageTensor]:
    return [pipe.image for pipe in repeat_pipes if pipe.image is not None] 



def ensure_defaults(model, latent_image, positive, negative, seed):
    if latent_image is None:
        latent_image = comfy.sample.create_random_latent(model.width, model.height, seed)
    if positive is None:
        positive = ConditioningTensor.default_positive()
    if negative is None:
        negative = ConditioningTensor.default_negative()
    return latent_image, positive, negative




@ComfyNode()
def Cairns_ksample(repeat_pipes: list[RepeatPipe],
                   seed: int = NumberInput(0, 0, 0xffffffffffffffff, step=1),
                   steps: int = NumberInput(20, 1, 10000, step=1),
                   cfg: float = NumberInput(8.0, 0.0, 100.0, step=0.1),
                   sampler_name: str = Choice(comfy.samplers.KSampler.SAMPLERS),
                   scheduler_name: str = Choice(comfy.samplers.KSampler.SCHEDULERS),  
                   denoise: float = NumberInput(1.0, 0.0, 1.0, step=0.01)) -> LatentTensor:
    
    pipe = repeat_pipes[0]
    model = pipe.model
    positive = pipe.pos
    negative = pipe.neg
    latent_image = pipe.latent

    latent_image, positive, negative = ensure_defaults(model, latent_image, positive, negative, seed)

    return common_ksampler(
        model=model, seed=seed, steps=steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler_name,
        positive=positive, negative=negative, latent=latent_image,
        denoise=denoise
    )


@ComfyNode()
def repeat_ksample(repeat_pipes: list[RepeatPipe],
                   seed: int = NumberInput(0, 0, 0xffffffffffffffff, step=1),
                   steps: int = NumberInput(20, 1, 10000, step=1),
                   cfg: float = NumberInput(8.0, 0.0, 100.0, step=0.1),
                   sampler_name: str = Choice(comfy.samplers.KSampler.SAMPLERS),
                   scheduler_name: str = Choice(comfy.samplers.KSampler.SCHEDULERS),   
                   denoise: float = NumberInput(1.0, 0.0, 1.0, step=0.01),
                   text: str = StringInput("Example: {This prompt, is, one prompt}, {This is, another}")) -> list[RepeatPipe]:
    
    logging.info("Starting repeat_ksample function")

    # Check input validity
    if repeat_pipes is None or len(repeat_pipes) == 0:
        logging.error("Invalid repeat_pipes: %s", type(repeat_pipes))
        raise ValueError(f"RepeatPipe must be provided. Instead {type(repeat_pipes)} was provided.")
    
    logging.info("Received %d repeat_pipes", len(repeat_pipes))
    logging.info("Seed: %d, Steps: %d, CFG: %.2f, Sampler: %s, Scheduler: %s, Denoise: %.2f",
                  seed, steps, cfg, sampler_name, scheduler_name, denoise)
    
    new_pipes = []
    
    for pipe_idx, pipe in enumerate(repeat_pipes):
        logging.info("Processing pipe %d: %s", pipe_idx, pipe)

        # Extract attributes
        model = pipe.model
        positive = pipe.pos
        negative = pipe.neg
        latent_image = pipe.latent
        vae = pipe.vae
        clip = pipe.clip
        prompt = pipe.prompt 
        
        logging.info("Pipe %d - Model: %s, Prompt: %s", pipe_idx, model, prompt)

        # Ensure defaults
        latent_image, positive, negative = ensure_defaults(model, latent_image, positive, negative, seed)

        # Extract and iterate prompts
        text_list = braces_to_list(text)
        prompt_list = string_to_list(prompt)

        # Add checks to ensure these lists aren't empty
        if not text_list:
            logging.info("Text list is empty: %s", text_list)
            continue  # Skip this iteration if text_list is empty
        if not prompt_list:
            logging.info("Prompt list is empty: %s", prompt_list)
            continue  # Skip this iteration if prompt_list is empty

        logging.info("text list %s, and prompt list: %s", text_list, prompt_list)
        for text_idx, t in enumerate(text_list):
            t_list = string_to_list(t)
            new_prompt = concat_prompt_lists(prompt_list, t_list)
            
            logging.info("Pipe %d - Text %d: Concatenated Prompt: %s", pipe_idx, text_idx, new_prompt)

            # Encode positive text
            new_positive = text_encode(clip=clip, text=new_prompt)

            # Generate latent tensor
            new_latent = common_ksampler(
                model=model, seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler_name,
                positive=new_positive, negative=negative, latent=latent_image,
                denoise=denoise
            )
            logging.info("Pipe %d - Generated new latent tensor - Text %d", pipe_idx, text_idx)

            # Create new pipe
            new_pipe = RepeatPipe()
            new_pipe.model = model
            new_pipe.pos = new_positive
            new_pipe.neg = negative
            new_pipe.latent = new_latent
            new_pipe.vae = vae
            new_pipe.clip = clip
            new_pipe.prompt = new_prompt
            new_pipe.image = decode(vae=new_pipe.vae, samples=new_latent)
            logging.info("Pipe %d - Created new RepeatPipe with decoded image for Text %d", pipe_idx, text_idx)

            new_pipes.append(new_pipe)

    logging.info("Generated %d new pipes", len(new_pipes))
    return new_pipes


@ComfyNode(is_output_node=True, color="#006600")
def preview_pipe_images(repeat_pipe: list[RepeatPipe]) -> list[RepeatPipe]:
    """
    Displays preview images from the given list of RepeatPipe objects.
    This node uses easy_nodes.show_image for displaying images.
    """
    # Loop through the RepeatPipe objects
    for pipe in repeat_pipe:
        image = getattr(pipe, 'image', None)  # Get the image from the pipe
        if image is not None:
            easy_nodes.show_image(image)  # Display the image

    return repeat_pipe
