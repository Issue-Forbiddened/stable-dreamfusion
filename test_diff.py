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

result_dir="cfg_results_diffcfg"
os.makedirs(result_dir,exist_ok=True)
# (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
# batch_size = num_images_per_prompt
# num_channels_latents = self.unet.config.in_channels

guidance_scale_list=[3.,5.,10.,50.,100.]
guidance_rescale_list=[0.]

# guidance_scale=[3.,5.]
# guidance_rescale=[0.,0.2]

num_samples_per_rescale = 20  # 以20为例

latent_shape=(num_samples_per_rescale,pipe.unet.config.in_channels,pipe.unet.config.sample_size,pipe.unet.config.sample_size)
latents = randn_tensor(latent_shape, device="cuda", dtype=torch.float16)


for guidance_rescale in guidance_rescale_list:

    final_images = []

    for sample_idx in range(num_samples_per_rescale):

        images=[]

        for guidance_scale in guidance_scale_list:

            image = pipe(prompt, num_images_per_prompt=1,
                         latents=latents[sample_idx:sample_idx+1],  
                         guidance_scale=guidance_scale).images[0]
            images.append(image)

        # images is a list of PIL images, concat them horizontally
        total_width = sum([img.width for img in images])
        max_height = max(img.height for img in images)

        combined_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

        final_images.append(combined_image)

    # Now, stack the final_images vertically
    max_width = max(img.width for img in final_images)
    total_height = sum([img.height for img in final_images])

    combined_final_image = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for img in final_images:
        combined_final_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the concatenated image
    filename = f"prompt_{prompt}_guidance_rescale_{guidance_rescale}.jpg"
    filepath = os.path.join(result_dir, filename)
    combined_final_image.save(filepath)