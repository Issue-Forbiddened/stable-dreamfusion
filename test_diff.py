from diffusers import StableDiffusionPipeline,UNet2DConditionModel
import torch
import numpy as np
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
import os
import math
from freeu import Free_UNetModel
import pdb
# model_id = "runwayml/stable-diffusion-v1-5"
model_id="stabilityai/stable-diffusion-2-1-base"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

freeu_params = {'b1': 1., 'b2': 1., 's1': 1., 's2': 1.}
pipe.unet=Free_UNetModel(unet=pipe.unet,**freeu_params)


prompt = "a photo of an astronaut riding a horse on mars"

result_dir="cfg_results_diffcfg_freeu/searchbs"
os.makedirs(result_dir,exist_ok=True)
# (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
# batch_size = num_images_per_prompt
# num_channels_latents = self.unet.config.in_channels

guidance_scale_list=[3.,5.,10.,20.,50.,100.]
b_list=[0.6,0.7,0.8,0.9,1.0,1.1,1.2]
s_list=[0.1,0.2,0.3,0.4,0.5]

# guidance_scale=[3.,5.]
# guidance_rescale=[0.,0.2]

num_samples_per_rescale = 10  # 以20为例

latent_shape=(num_samples_per_rescale,pipe.unet.config.in_channels,pipe.unet.config.sample_size,pipe.unet.config.sample_size)
latents = randn_tensor(latent_shape, device="cuda", dtype=torch.float16)

# b1: 1.1, b2: 1.2, s1: 0.9, s2: 0.2


level_one_name='guidance_scale'
level_two_name='b'
level_three_name='s'

level_one_iter=guidance_scale_list
level_two_iter=b_list
level_three_iter=s_list


for level_one in level_one_iter:

    for sample_idx in range(num_samples_per_rescale):
        final_images = []
        for level_two in level_two_iter:

            images=[]

            for level_three in level_three_iter:
                freeu_params['b1']=level_two
                freeu_params['b2']=level_two+0.05
                freeu_params['s2']=level_three
                freeu_params['s1']=level_three+0.6

                pipe.unet.set_params(**freeu_params)

                image = pipe(prompt, num_images_per_prompt=1,
                             latents=latents[sample_idx:sample_idx+1],  
                            # latents=latents[0:0+1], 
                            guidance_scale=level_one,
                            num_inference_steps=10,
                            ).images[0]
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
        filename = f"prompt_{prompt}"
        filename += f"_{level_one_name}_{level_one}"
        filename += f"_{level_two_name}"
        filename += f"_{level_three_name}"
        filename += f"sample_{sample_idx}"
        filename += ".png"

        filepath = os.path.join(result_dir, filename)
        combined_final_image.save(filepath)