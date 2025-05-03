from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import torch
import torch.nn as nn
from tqdm import tqdm

from helpers import *


class OverlayIllusion(nn.Module):

    def __init__(self, checkpoint_path, device='cuda', pipe=None, backlight=2.0, Lz=2.0):
        super().__init__()
        self.device = torch.device(device)
        
        if pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(
                checkpoint_path,
                scheduler=DDIMScheduler.from_pretrained(checkpoint_path, subfolder="scheduler"),
                torch_dtype=torch.float,
                use_safetensors=True,
                requires_safety_checker=False,
                safety_checker=None,
            )
        
        self.pipe = pipe
        self.vae = pipe.vae.to(self.device) 
        self.tokenizer = pipe.tokenizer                    
        self.text_encoder = pipe.text_encoder.to(self.device) 
        self.unet = pipe.unet.to(self.device) 
        self.scheduler = pipe.scheduler

        self.uncond_text = ""
        self.checkpoint_path = checkpoint_path

        self.backlight = backlight
        self.Lz = Lz

    def sample(self, prompts: list[str], height=512, width=512, num_steps=80, guidance_scale=7.5):

        text_embeddings = []
        for prompt in prompts:
            embedding = self.get_text_embedding(self.pipe, prompt)
            text_embeddings.append(embedding)

        # Create 5 Latents
        latents_shape = (5, self.pipe.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=self.device)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            # Prepare Latents 
            latent_inputs = torch.cat([latents] * 2)
            latent_inputs = self.scheduler.scale_model_input(latent_inputs, t)

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_inputs, 
                    t, 
                    encoder_hidden_states=text_embeddings).sample

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Get Clean Images Prediction
            clean_images_pred = get_clean_sample(
                model=self.pipe, 
                sample=latents, 
                model_output=noise_pred,
                timestep=t,
                pred_type="epsilon"
                )
            
            # Constraint Clean Images Using Burgert Formula
            Ta, Tb, Tc, Td, Tz = clean_images_pred.chunk(5)
            clean_images_constraint = torch.stack(f(
                Ta=Ta, Tb=Tb, Tc=Tc, Td=Td, Tz=Tz, 
                Lz=self.Lz, 
                backlight=self.backlight
                ), dim=0)
            
            noise_constraint = clean_images_constraint - clean_images_pred

            # Step with Noise Constraint
            latents = self.scheduler.step(
                model_output=noise_constraint,
                timestep=t,
                sample=latents,
            ).prev_sample

        images = get_images(self.pipe, latents=latents)
        return images


if __name__ == "__main__":

    Overlay = OverlayIllusion()

    
