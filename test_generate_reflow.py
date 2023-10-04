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
sample_num=2000

torch.manual_seed(3)
np.random.seed(3)

result_dir=f"/home1/jo_891/data1/diffusion_generation_sd_reflow/train"
os.makedirs(result_dir,exist_ok=True)
print(f"result_dir: {result_dir}")
# (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
# batch_size = num_images_per_prompt
# num_channels_latents = self.unet.config.in_channels

latent=Image.open("/home1/jo_891/data1/stable-dreamfusion/horse_source.jpg").convert("RGB")
# convert to tensor and rescale to [-1,1]
latent=np.array(latent)/255*2-1
latent=latent.transpose(2,0,1)
latent=torch.from_numpy(latent).unsqueeze(0).to("cuda").to(torch.float16) # (1,3,512,512)

with torch.no_grad():
    posterior = pipe.vae.encode(latent).latent_dist
    latent = posterior.sample() * pipe.vae.config.scaling_factor

latent_shape=(1,pipe.unet.config.in_channels,pipe.unet.config.sample_size,pipe.unet.config.sample_size)


with torch.no_grad():
    for sample_idx in trange(sample_num):
        noise=randn_tensor(latent_shape,dtype=torch.float16,device="cuda")
        t=np.random.uniform(0.65,0.9)
        t=(1000*torch.tensor([t])).long()
        latent_org=pipe.scheduler.add_noise(latent, noise, t)
        latent_opt = pipe(prompt,num_images_per_prompt=1,
                    latents=latent_org,
                    guidance_scale=4.,
                    num_inference_steps=30,
                    output_type='latent',
                    ).images[0]
        idx=0
        while os.path.exists(os.path.join(result_dir,f"{str(idx).zfill(7)}_opt.npy")):
            idx+=1
        np.save(os.path.join(result_dir,f"{str(idx).zfill(7)}_opt.npy"),latent_opt.detach().cpu().numpy())
        np.save(os.path.join(result_dir,f"{str(idx).zfill(7)}_org.npy"),latent_org.squeeze().detach().cpu().numpy())