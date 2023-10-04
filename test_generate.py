from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
import os
import math
from tqdm import trange

model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
prompt_with_=prompt.replace(" ","_")
sample_num=3000
sqrt_sample_num=math.ceil(math.sqrt(sample_num))

result_dir=f"/home1/jo_891/data1/diffusion_generation/train/{prompt_with_}"
os.makedirs(result_dir,exist_ok=True)
print(f"result_dir: {result_dir}")
# (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
# batch_size = num_images_per_prompt
# num_channels_latents = self.unet.config.in_channels

guidance_scale_list=[3.,4.,5.]
# randomly change sequece of guidance_scale_list
np.random.shuffle(guidance_scale_list)

latent_shape=(1,pipe.unet.config.in_channels,pipe.unet.config.sample_size,pipe.unet.config.sample_size)

latent=Image.open("/home1/jo_891/data1/stable-dreamfusion/horse_source.jpg").convert("RGB")
# convert to tensor and rescale to [-1,1]
latent=np.array(latent)/255*2-1
latent=latent.transpose(2,0,1)
latent=torch.from_numpy(latent).unsqueeze(0).to("cuda").to(torch.float16) # (1,3,512,512)

with torch.no_grad():
    posterior = pipe.vae.encode(latent).latent_dist
    latent = posterior.sample() * pipe.vae.config.scaling_factor
    




with torch.no_grad():
    for guidance_scale in guidance_scale_list:
        for sample_idx in trange(sample_num//3):
            noise=torch.randn_like(latent)
            t=np.random.uniform(0.62,0.85)
            t=(1000*torch.tensor([t])).long()
            latent_noisy=pipe.scheduler.add_noise(latent, noise, t)

            image = pipe(prompt,num_images_per_prompt=1,
                        latents=latent_noisy,
                        guidance_scale=guidance_scale,
                        num_inference_steps=30,
                        ).images[0]

            while os.path.exists(os.path.join(result_dir,f"{sample_idx}.png")):
                sample_idx+=1
            image.save(os.path.join(result_dir,f"{sample_idx}_t_{t.item()}.png"))