from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image


class FlipIllusion(nn.Module):

    def __init__(self, checkpoint_path, device='cuda', pipe=None):
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

    def get_text_embedding(self, prompts, negative_prompt=None):
        """
        Embed Fucking text into Token Space
        """
        
        if isinstance(prompts,str):
            prompts = [prompts]
    
        text_input = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.tokenizer.model_max_length, 
            truncation=True, return_tensors='pt').input_ids.to(self.device)
        
        uncond_input = self.tokenizer(
            [self.uncond_text] * len(prompts), 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt').input_ids.to(self.device)
    
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.to(self.device))[0]
            uncond_embeddings = self.text_encoder(uncond_input.to(self.device))[0]
    
        output_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
        return output_embeddings
    
    @torch.no_grad()
    def get_images(self, latents: list):
        images = []
        for latent in latents:
            latent_image = latent / 0.18215
            image = self.vae.decode(latent_image).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = (image * 255).round().to(torch.uint8)
            image = image.permute(0, 2, 3, 1).squeeze(0)
            images.append(Image.fromarray(image.cpu().numpy()))
        return tuple(images)
    
    def sample(self, prompts: list[str], height=512, width=512, num_steps=80, guidance_scale=7.5):

        text_embeddings = []
        for prompt in prompts:
            embedding = self.get_text_embedding(prompt)
            text_embeddings.append(embedding)

        latents_shape = (1, self.pipe.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=self.device)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            if i % 2 == 0:
                # Rotate 180 degrees every iteration
                latents = torch.rot90(latents, 2, [2, 3])
            latent_input = torch.cat([latents] * 2)

            latent_input = self.scheduler.scale_model_input(latent_input, t)
            
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_input, 
                    t, 
                    encoder_hidden_states=text_embeddings[i%2]).sample
                
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if i % 2 == 0:
                latents = torch.rot90(latents, 2, [2, 3])
                noise_pred = torch.rot90(noise_pred, 2, [2, 3])

            # Step the latent with the rotated noise (denoises the lower half)
            latents = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latents,
            ).prev_sample

        image = self.get_images([latents])
        return image

if __name__ == '__main__':
    diffusion = FlipIllusion(checkpoint_path="./stable-diffusion-v1-5")

    prompts = [
        "Old man",
        "Camp fire"
    ]

    images = diffusion.sample(prompts)
    for i, image in enumerate(images):
        file = f"image_{i}.png"
        image.save(file)
