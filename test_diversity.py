from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
from diffusers.utils import randn_tensor
import os
import math

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
sample_num=81
sqrt_sample_num=math.ceil(math.sqrt(sample_num))
result_dir="cfg_results_diversity"
os.makedirs(result_dir,exist_ok=True)
# (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
# batch_size = num_images_per_prompt
# num_channels_latents = self.unet.config.in_channels

guidance_scale_list=[10.,50.,100.]
guidance_rescale_list=[0.,0.3,0.5,0.7,1.]

# guidance_scale=[3.,5.]
# guidance_rescale=[0.,0.2]

latent_shape=(1,pipe.unet.config.in_channels,pipe.unet.config.sample_size,pipe.unet.config.sample_size)

for guidance_scale in guidance_scale_list:
    for guidance_rescale in guidance_rescale_list:

        images=[]

        for sample_idx in range(sample_num):

            latents = randn_tensor(latent_shape, device="cuda", dtype=torch.float16)

        
            image = pipe(prompt,num_images_per_prompt=1,
                        latents=latents,
                        guidance_scale=guidance_scale,
                        guidance_rescale=guidance_rescale).images[0]
            images.append(image)

        # convert images to numpy arrays
        images = [np.array(img) for img in images]

        image_empty=np.zeros((sqrt_sample_num*images[0].shape[0],
                            sqrt_sample_num*images[0].shape[1],
                            3),
                            dtype=np.uint8)

        for i in range(sqrt_sample_num):
            for j in range(sqrt_sample_num):
                image_empty[i*images[0].shape[0]:(i+1)*images[0].shape[0],
                            j*images[0].shape[1]:(j+1)*images[0].shape[1],:]=images[i*sqrt_sample_num+j]


        # concat images which are in image and are PIL images
        image = Image.fromarray(image_empty)

        # annotation for row and column

        # save image
        image.save(os.path.join(result_dir,f"guidance_scale_{guidance_scale}_rescale_{guidance_rescale}.png"))




    # Path: test_diversity.py