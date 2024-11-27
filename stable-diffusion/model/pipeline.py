import numpy as np
import torch
from tqdm import tqdm

from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt: str,
             uncond_prompt: str, # i.e., negative prompt, or empty string
             input_image=None,
             strength: float=0.8, do_cfg=True, cfg_scale=7.5,
             sampler_name='ddpm', n_inference_steps=50, models={},
             seed=None, idle_device=None, tokenizer=None, device=None):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError('Strength must be < 0 < 1')

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models['clip']
        clip.to(device)

        if do_cfg:
            # convert the prompt into tokens using tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            # bs, seq_len
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # bs, seq_len -> bs, seq_len, dim
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # bs, seq_len -> bs, seq_len, dim
            uncond_context = clip(uncond_tokens)

            # combine 2 prompts
            # (2, seq_len, dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])

        else:
            # convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # combine one prompt only
            # 1, 77, 768
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise NotImplementedError(f'Unknown sampler: {sampler_name}')

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # image to image
        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # H, W, Channel
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # H, W, C -> bs, H, W, C
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # bs, H, W, C -> bs, C, H, W
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)

            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(device)

        # text to image
        else:
            # start with random noise N(0, I) to generate an image
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # 999, 998, ... 0 | with 1000 time steps
        # 1000, 980, 960, ..., 0 | with 50 time steps, the process starts at 1000 and makes steps of 20, i.e. 1000/50
        # each time steps indicates a noise level
        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep, in enumerate(timesteps):
            # 1, 320
            # use the sin and cos to define the timesteps (transformer)
            time_embedding = get_time_embedding(timestep).to(device)
            # bs, 4, latents_h, latents_w
            model_input = latents

            # classifier free guidance
            if do_cfg:
                # bs, 4, latents_h, latents_w -> 2 * bs, 4, latents_h, latents_w
                model_input = model_input.repeat(2, 1, 1, 1)

            # predict noise by the U-Net
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # combine conditional & unconditional outputs
                output_conditional, output_unconditional = model_output.chunk(2)
                model_output = cfg_scale * (output_conditional - output_unconditional) + output_unconditional

            # Remove noise predicted by the U-Net
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # bs, C, H, W -> bs, H, W, C
        images = images.permute(0, 2, 3, 1)
        images = images.to('cpu', torch.uint8).numpy()

        return images[0]


def rescale(x, old_range, new_range, clamp: bool=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)

    return x

def get_time_embedding(timestep):
    # 160,
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # 1, 160
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # 1, 320
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
