from diffusers import StableDiffusionPipeline
import torch

model_key="stabilityai/stable-diffusion-2-1-base"
precision_t = torch.float16
device = "cuda"
# Create model
pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t)

prompt='a DSLR photo of a tiger dressed as a doctor'