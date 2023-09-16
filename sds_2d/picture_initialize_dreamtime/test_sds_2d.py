import os 
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm  import tqdm

prompt = "donald trump"
latent_num=1

model_key="stabilityai/stable-diffusion-2-1-base"
precision_t = torch.float16
device = "cuda"
# Create model
pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t)

pipe.to(device)

vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet.to(device)


for p in unet.parameters():
    p.requires_grad = False

scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=precision_t)

min_step, max_step = 2,998

use_dreamtime=True
dreamtime_m12s12=[800,500,300,100] if use_dreamtime else None

def dreamtime_w(t,dreamtime_m12s12=[800,500,300,100]):
    if not torch.is_tensor(t):
        t=torch.tensor(t).to(device)
    if dreamtime_m12s12 is not None:
        m1=dreamtime_m12s12[0]
        m2=dreamtime_m12s12[1]
        s1=dreamtime_m12s12[2]
        s2=dreamtime_m12s12[3]
    else:
        return 1.
    if t > m1:
        return torch.exp(-(t - m1)**2 / (2 * (s1**2))).to(device)
    elif m2 <= t <= m1:
        return torch.ones(t.shape)
    elif t < m2:
        return torch.exp(-(t - m2)**2 / (2 * (s2**2))).to(device)

if True:
    dreamtime_t2index=lambda t: (t-min_step).long() if torch.is_tensor(t) else int(t-min_step)
    dreamtime_index2t=lambda i: torch.tensor(i).to(device)+min_step if not torch.is_tensor(i) else i+min_step
    dreamtime_w_sum=sum([dreamtime_w(t) for t in range(min_step,max_step+1)])
    dreamtime_p=dreamtime_w_normalized=lambda t: dreamtime_w(t)/dreamtime_w_sum
    dreamtime_p_list=torch.tensor([dreamtime_p(t) for t in range(min_step,max_step)]).to(device)
    dreamtime_p_t2Tsum=lambda t: dreamtime_p_list[dreamtime_t2index(t):].sum()
    dreamtime_p_t2Tsum_lookup=torch.tensor([dreamtime_p_t2Tsum(t) for t in range(min_step,max_step+1)]).to(device)

    dreamtime_optimal_t=lambda train_ratio: dreamtime_index2t((dreamtime_p_t2Tsum_lookup-train_ratio).abs().argmin())

def get_t(train_ratio,batch_size,dreamtime_m12s12=None):
    if dreamtime_m12s12 is None:
        return torch.randint(min_step, max_step + 1, (batch_size,), dtype=torch.long, device=device)
    else:
        return (dreamtime_optimal_t(train_ratio)).repeat(batch_size)


def get_latent_codes_as_param(latent_num=latent_num):
    return torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.rand((latent_num,4, 64,64), device=device, dtype=torch.float32)))
    # return torch.nn.Parameter(torch.zeros((latent_num,4, 64,64), device=device, dtype=torch.float32))

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def _get_w(t,alphas, dreamtime_m12s12=None):
    if dreamtime_m12s12 is None:
        return (1 - alphas[t])
    else:
        return torch.ones_like((1 - alphas[t])) # much better than using w = (1 - alphas[t]), indicating no robustness to w(t) using dreamtime
alphas=scheduler.alphas.to(device)
get_w=lambda t: _get_w(t,alphas,dreamtime_m12s12=dreamtime_m12s12)

with torch.no_grad():
    text_inputs = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, return_tensors='pt')
    embeddings = text_encoder(text_inputs.input_ids.to(device))[0].repeat(2*latent_num,1,1)

# latents=get_latent_codes_as_param(latent_num=latent_num)

source_image_path='/home1/jo_891/data1/stable-dreamfusion/trial_trump_dreamtime_w/validation/df_ep0065_0007_rgb.png'
source_image=Image.open(source_image_path)
source_image=source_image.resize((512,512))
source_image=np.array(source_image)
source_image=source_image.transpose(2,0,1)
source_image=torch.from_numpy(source_image).unsqueeze(0).to("cuda").to(torch.float16)

def encode(imgs):
    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample() * vae.config.scaling_factor
    return latents

latents=encode(source_image)

latents=torch.nn.Parameter(latents.to(torch.float32))

optim=torch.optim.Adam([latents],lr=0.01,weight_decay=0.,betas=(0.9,0.999),eps=1e-8)

scaler = torch.cuda.amp.GradScaler(enabled= True if precision_t == torch.float16 else False)

def train_step(latents,text_embedding,guidance_scale=100.,guidance_rescale=0.,train_ratio=1.,grad_scale=1.,dreamtime_m12s12=None,save_guidance_path=None):
    t=get_t(train_ratio,batch_size=latents.shape[0],dreamtime_m12s12=dreamtime_m12s12)

    with torch.no_grad():
        # add noise
        noise = torch.randn_like(latents)
        latents_noisy = scheduler.add_noise(latents, noise, t)
        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        noise_pred = unet(latent_model_input, tt, encoder_hidden_states=text_embedding).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        if guidance_rescale:
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_pos, guidance_rescale)
    w = get_w(t)
    grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
    grad = torch.nan_to_num(grad)
    
    if save_guidance_path is not None:
        save_guidance(latents,latents_noisy,noise_pred,noise,save_guidance_path,t)

    targets = (latents - grad).detach()
    loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

    return loss


def decode_latents(latents):
    with torch.no_grad():

        latents = 1 / vae.config.scaling_factor * latents

        imgs = vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

def save_guidance(latents,latents_noisy,noise_pred,noise,save_guidance_path,t,as_latent=False):
    with torch.no_grad():
        pred_rgb_512 = decode_latents(latents)

        # visualize predicted denoised image
        # The following block of code is equivalent to `predict_start_from_noise`...
        # see zero123_utils.py's version for a simpler implementation.
        alphas = scheduler.alphas.to(latents)
        total_timesteps = max_step - min_step + 1
        index = total_timesteps - t.to(latents.device) - 1 
        b = len(noise_pred)
        a_t = alphas[index].reshape(b,1,1,1).to(device)
        sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(device)                
        pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
        result_hopefully_less_noisy_image = decode_latents(pred_x0.to(latents.type(precision_t)))

        # visualize noisier image
        result_noisier_image = decode_latents(latents_noisy.to(pred_x0).type(precision_t))

        diff_latent_image=decode_latents((noise_pred-noise).to(pred_x0).type(precision_t))

        zero_latent_image=decode_latents(torch.zeros_like(latents).to(pred_x0).type(precision_t))


        viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image, diff_latent_image,zero_latent_image],dim=0)
        save_image(viz_images, save_guidance_path)

total_step=10000

save_path_dir=os.path.join("sds_2d",'picture_initialize_dreamtime')
os.makedirs(save_path_dir,exist_ok=True)
# set process bar
pbar = tqdm(total=total_step)

# use tensorboard

writer = SummaryWriter(save_path_dir)

# save this code to save_path_dir
os.system(f"cp {__file__} {save_path_dir}")


for i in (range(0,total_step)):
    optim.zero_grad()
    with torch.cuda.amp.autocast(enabled= True if precision_t == torch.float16 else False):
        loss=train_step(latents,embeddings,guidance_scale=100.,guidance_rescale=0.,
                        train_ratio=i/total_step,grad_scale=1.,dreamtime_m12s12=dreamtime_m12s12,
                        save_guidance_path=os.path.join(save_path_dir,f"{i}.png") if i%100==0 else None)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    pbar.update(1)
    pbar.set_description(f"loss: {loss.item():.4f}")
    writer.add_scalar("loss",loss.item(),i)
    # add image saved by save_guidance
    if i%100==0:
        writer.add_image("guidance",torch.tensor(np.array(Image.open(os.path.join(save_path_dir,f"{i}.png"))).transpose(2,0,1))/255.,i)


