from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
from diffusers.utils import randn_tensor
import os

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
sample_num=3
result_dir="cfg_results"
os.makedirs(result_dir,exist_ok=True)
# (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
# batch_size = num_images_per_prompt
# num_channels_latents = self.unet.config.in_channels

guidance_scale=[float(i) for i in range(10,110,10)]
guidance_rescale=[0.,0.2,0.4,0.6,0.8,1.]

# guidance_scale=[3.,5.]
# guidance_rescale=[0.,0.2]

latent_shape=(1,pipe.unet.config.in_channels,pipe.unet.config.sample_size,pipe.unet.config.sample_size)

for sample_idx in range(sample_num):

    latents = randn_tensor(latent_shape, device="cuda", dtype=torch.float16)

    images=[]
    for i in range(len(guidance_scale)):
        for j in range(len(guidance_rescale)):
            image = pipe(prompt,num_images_per_prompt=1,
                        latents=latents,
                        guidance_scale=guidance_scale[i],
                        guidance_rescale=guidance_rescale[j]).images[0]
            images.append(image)

    # convert images to numpy arrays
    images = [np.array(img) for img in images]

    image_empty=np.zeros((len(guidance_scale*images[0].shape[0]),
                        len(guidance_rescale*images[0].shape[1]),
                        3),dtype=np.uint8)

    for i in range(len(guidance_scale)):
        for j in range(len(guidance_rescale)):
            image_empty[i*images[0].shape[0]:(i+1)*images[0].shape[0],
                        j*images[0].shape[1]:(j+1)*images[0].shape[1],:]=images[i*len(guidance_rescale)+j]




    # concat images which are in image and are PIL images
    image = Image.fromarray(image_empty)

    # annotation for row and column
    annotation_row = [str(i) for i in guidance_scale]
    annotation_col = [str(i) for i in guidance_rescale]


    # save image
    image.save(os.path.join(result_dir,f"sample_{sample_idx}.png"))

    # save annotation
    with open(os.path.join(result_dir,f"sample_{sample_idx}_annotation.txt"), "w") as f:
        f.write("\n".join(['row_guidance_scale:']+annotation_row))
        f.write("\n")
        f.write("\n".join(['col_guidance_rescale:']+annotation_col))
        f.write("\n")




