from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98],dreamtime_m12s12=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)   
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.dreamtime_m12s12=dreamtime_m12s12

        if self.dreamtime_m12s12 is not None:
            self.dreamtime_t2index=lambda t: (t-self.min_step).long() if torch.is_tensor(t) else int(t-self.min_step)
            self.dreamtime_index2t=lambda i: torch.tensor(i).to(self.device)+self.min_step if not torch.is_tensor(i) else i+self.min_step
            self.dreamtime_w_sum=sum([self.dreamtime_w(t) for t in range(self.min_step,self.max_step+1)])
            self.dreamtime_p=self.dreamtime_w_normalized=lambda t: self.dreamtime_w(t)/self.dreamtime_w_sum
            self.dreamtime_p_list=torch.tensor([self.dreamtime_p(t) for t in range(self.min_step,self.max_step)]).to(self.device)
            self.dreamtime_p_t2Tsum=lambda t: self.dreamtime_p_list[self.dreamtime_t2index(t):].sum()
            self.dreamtime_p_t2Tsum_lookup=torch.tensor([self.dreamtime_p_t2Tsum(t) for t in range(self.min_step,self.max_step+1)]).to(self.device)

            self.dreamtime_optimal_t=lambda train_ratio: self.dreamtime_index2t((self.dreamtime_p_t2Tsum_lookup-train_ratio).abs().argmin())


        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings


    def rescale_noise_cfg(self,noise_cfg, noise_pred_text, guidance_rescale=0.0):
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

    def dreamtime_w(self,t):
        if not torch.is_tensor(t):
            t=torch.tensor(t).to(self.device)
        if self.dreamtime_m12s12 is not None:
            m1=self.dreamtime_m12s12[0]
            m2=self.dreamtime_m12s12[1]
            s1=self.dreamtime_m12s12[2]
            s2=self.dreamtime_m12s12[3]
        else:
            return 1.
        if t > m1:
            return torch.exp(-(t - m1)**2 / (2 * (s1**2))).to(self.device)
        elif m2 <= t <= m1:
            return torch.ones(t.shape)
        elif t < m2:
            return torch.exp(-(t - m2)**2 / (2 * (s2**2))).to(self.device)

    def get_t(self,train_ratio,batch_size):
        if self.dreamtime_m12s12 is None:
            return torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
        else:
            return (self.dreamtime_optimal_t(train_ratio)).repeat(batch_size)
    def get_w(self,t):
        if self.dreamtime_m12s12 is None:
            return (1 - self.alphas[t])
        else:
            return torch.ones_like((1 - self.alphas[t])) # much better than using w = (1 - self.alphas[t]), indicating no robustness to w(t) using dreamtime

    def save_guidance(self,pred_rgb,latents,latents_noisy,noise_pred,noise,save_guidance_path,t,as_latent=False):
        with torch.no_grad():
            if as_latent:
                pred_rgb_512 = self.decode_latents(latents)
            else:
                pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)

            # visualize predicted denoised image
            # The following block of code is equivalent to `predict_start_from_noise`...
            # see zero123_utils.py's version for a simpler implementation.
            alphas = self.scheduler.alphas.to(latents)
            total_timesteps = self.max_step - self.min_step + 1
            index = total_timesteps - t.to(latents.device) - 1 
            b = len(noise_pred)
            a_t = alphas[index].reshape(b,1,1,1).to(self.device)
            sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
            sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
            pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
            result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

            # visualize noisier image
            result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

            diff_latent_image=self.decode_latents((noise_pred-noise).to(pred_x0).type(self.precision_t))


            # TODO: also denoise all-the-way

            # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
            viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image, diff_latent_image],dim=0)
            save_image(viz_images, save_guidance_path)


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None,guidance_rescale=0.,train_ratio=1.):

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!aa
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # MYMARK
        t=self.get_t(train_ratio,latents.shape[0])

        # if train_ratio<0.5:
        #     guidance_scale=100.*(0.5-train_ratio)*2+10.*train_ratio*2
        # else:
        #     guidance_scale=10.*2*(1-train_ratio)+100.*(train_ratio-0.5)*2

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
            if guidance_rescale:
                noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_pos, guidance_rescale)

        # w(t), sigma_t^2
        w = self.get_w(t)
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            self.save_guidance(pred_rgb,latents,latents_noisy,noise_pred,noise,save_guidance_path,t,as_latent)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss
    

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts       

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * (1 + K))
            tt = torch.cat([t] * (1 + K))
            unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)            

        # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1 
                b = len(noise_pred)
                a_t = alphas[index].reshape(b,1,1,1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))



                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                save_image(viz_images, save_guidance_path)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss


    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from copy import deepcopy
import pdb
class StableDiffusion_LoRA(StableDiffusion):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98], dreamtime_m12s12=None):
        super().__init__(device, fp16, vram_O, sd_version, hf_key, t_range, dreamtime_m12s12)
 
        # Set correct lora layers
        lora_attn_procs = {}
        self.lora_params=[]

        self.lora_unet=deepcopy(self.unet)
        
        for name in self.lora_unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.lora_unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.lora_unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.lora_unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.lora_unet.config.block_out_channels[block_id]
            lora_layers=LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim).to(self.device)
            lora_layers.requires_grad_(True)
            lora_attn_procs[name] = lora_layers
            self.lora_params.extend(lora_layers.parameters()) 

        self.lora_unet.set_attn_processor(lora_attn_procs)
        # self.lora_layers = AttnProcsLayers(self.lora_unet.attn_processors)
        # self.lora_layers=self.lora_layers.requires_grad_(True)
        self.lora_layers=nn.ParameterList(self.lora_params)
        self.lora_unet.requires_grad_(True)

    def get_t(self,train_ratio,batch_size):
        if self.dreamtime_m12s12 is None:
            if train_ratio<0.3:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
            else:
                t = torch.randint(self.min_step, (self.max_step + 1 + self.min_step)//2, (batch_size,), dtype=torch.long, device=self.device)
        else:
            t = (self.dreamtime_optimal_t(train_ratio)).repeat(batch_size)
        return t

    def get_w(self,t):
        return (1 - self.alphas[t])

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                    save_guidance_path:Path=None,guidance_rescale=0.,train_ratio=1.):

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)


        # predict the noise residual with unet, NO grad!

        t=self.get_t(train_ratio,latents.shape[0])


        # with torch.no_grad():
            # add noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        noise_pred_pretrained = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
        
        # perform guidance (high scale from paper!)
        noise_pred_uncond_pretrained, noise_pred_pos_pretrained = noise_pred_pretrained.chunk(2)
        noise_pred_pretrained = noise_pred_uncond_pretrained + guidance_scale * (noise_pred_pos_pretrained - noise_pred_uncond_pretrained)
        if guidance_rescale:
            noise_pred_pretrained = self.rescale_noise_cfg(noise_pred_pretrained, noise_pred_pos_pretrained, guidance_rescale)

        noise_pred_lora=self.lora_unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond_lora, noise_pred_pos_lora = noise_pred_lora.chunk(2)
        noise_pred_lora = noise_pred_uncond_lora + guidance_scale * (noise_pred_pos_lora - noise_pred_uncond_lora)
        if guidance_rescale:
            noise_pred_lora = self.rescale_noise_cfg(noise_pred_lora, noise_pred_pos_lora, guidance_rescale)

        # w(t), sigma_t^2
        w = self.get_w(t)

        grad = grad_scale * w[:, None, None, None] * (noise_pred_pretrained - noise_pred_lora.detach())
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            self.save_guidance(pred_rgb,latents,latents_noisy,noise_pred_pretrained,noise_pred_lora,save_guidance_path,t,as_latent)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        loss_lora=0.5 * F.mse_loss(noise_pred_lora.float(), noise.float(), reduction='sum') / latents.shape[0]
        # loss_lora=F.mse_loss(noise_pred_lora.float(), noise.float(), reduction='mean')  

        loss=loss+loss_lora



        return loss
        


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




