import torch
import numpy as np
from tqdm import tqdm

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt: str, uncond_prompt: str, input_image=None,
             strength: float=0.8, do_cfg=True, cfg_scale=7.5,
             sampler_name='ddpm', n_inference_steps=50, models={},
             seed=None, idle_device=None, tokenizer=None, device=None):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError('Strength must be < 0 < 1')

        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

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
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise NotImplementedError(f'Unknown sampler: {sampler_name}')

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # H, W, Channel
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # H, W, C -> bs, H, W, C
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # bs, H, W, C -> bs, C, H, W
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

