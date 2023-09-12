from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
from diffusers.utils import randn_tensor
import os
import math

# DDIMScheduler
from diffusers import DDIMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers

model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "trump figure"

# source_image_path='/home1/jo_891/data1/stable-dreamfusion/trial_trump_w_lora_anneal_correct_sum/validation/df_ep0100_0001_rgb.png' #fully trained
phase='end'
if phase=='beginning':
    source_image_path='/home1/jo_891/data1/stable-dreamfusion/trial_trump_w/validation/df_ep0023_0001_rgb.png' # beginning
elif phase=='middle':
    source_image_path='/home1/jo_891/data1/stable-dreamfusion/trial_trump_w/validation/df_ep0060_0001_rgb.png' # middle
elif phase=='end':
    source_image_path='/home1/jo_891/data1/stable-dreamfusion/trial_trump_w/validation/df_ep0098_0001_rgb.png' # end
source_image=Image.open(source_image_path)
source_image=source_image.resize((512,512))
source_image=np.array(source_image)
source_image=source_image.transpose(2,0,1)
source_image=torch.from_numpy(source_image).unsqueeze(0).to("cuda").half()

guidance_scale=10.

# transfer to (-1.1) from (0,255)
source_image=source_image/127.5-1.0
result_dir="lora_results_diversity"
os.makedirs(result_dir,exist_ok=True)

vae=pipe.vae
vae=vae.to("cuda")

vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet

lora=False
if lora:
    lora_ckpt='/home1/jo_891/data1/stable-dreamfusion/trial_trump_w_lora_anneal_correct_sum/checkpoints/df.pth'
    lora_ckpt=torch.load(lora_ckpt)
    lora_ckpt=lora_ckpt['guidance_lora']
    lora_attn_procs={}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim).to("cuda")


    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    lora_layers.load_state_dict(lora_ckpt)

scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", torch_dtype=torch.float16)

del pipe

def encode(vae,imgs):
    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample() * vae.config.scaling_factor
    return latents

def decode(vae,denoised_latents):
    with torch.no_grad():
        image = vae.decode(denoised_latents / vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)*255
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0,2,3,1).float().numpy().astype(np.uint8)[0]
        return image
latents=encode(vae,source_image)

images=[]

for i in range(2,1000,100):
    # t = torch.randint(2, 998, (latents.shape[0],), dtype=torch.long, device=torch.device("cuda"))
    t = torch.tensor([i], dtype=torch.long, device=torch.device("cuda"))

    inputs = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, return_tensors='pt')
    embeddings = text_encoder(inputs.input_ids.to(t.device))[0]

    with torch.no_grad():
        # add noise
        noise = torch.randn_like(latents)
        latents_noisy = scheduler.add_noise(latents, noise, t)
        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        noise_pred = unet(latent_model_input, tt,encoder_hidden_states=embeddings.repeat(latent_model_input.shape[0],1,1)).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        alphas = scheduler.alphas.to(latents)
        total_timesteps = 1000 - 1 + 1
        index = total_timesteps - t.to(latents.device) - 1 
        b = len(noise_pred)
        a_t = alphas[index].reshape(b,1,1,1).to(latents.device)
        sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(latents.device)                
        denoised_latents = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0

        grad=noise_pred-noise


    # decode latents,latents_noisy and denoised_latents to images and concat them
    image_latents=decode(vae,latents)
    image_latents_noisy=decode(vae,latents_noisy)
    image_denoised_latents=decode(vae,denoised_latents)
    image_grad=decode(vae,grad)
    concated_image=np.concatenate((image_latents,image_latents_noisy,image_denoised_latents,image_grad),axis=1)

    images.append(concated_image)

    # # convert images to PIL images, which is (batch_size, height, width, num_channels)
    # image = Image.fromarray(concated_image)


    # # save image
    # image.save(os.path.join(result_dir,f"{t.int().item()}.png"))

images=np.concatenate(images,axis=0)

# convert images to PIL images, which is (batch_size, height, width, num_channels)
image = Image.fromarray(images)

# save image
image.save(os.path.join(result_dir,f"trump_figure_scale_{guidance_scale}_{'lora_' if lora else ''}_{phase}.png"))