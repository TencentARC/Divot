import torch
from functools import partial
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.utils_diffusion import make_beta_schedule, rescale_zero_terminal_snr

class DynamiCrafter(nn.Module):
    def __init__(self, video_tokenizer, diffusion_model, first_stage_model, image_proj_model, unconditional_guidance_scale=7.5, num_frames=5, image_size=256, fs=3, ddim_steps=50, ddim_eta=1.0, timestep_spacing='uniform', guidance_rescale=0.7, v_posterior=0., channels=4, scale_factor=0.18215) -> None:
        super().__init__()
        self.video_tokenizer = video_tokenizer.to(torch.float32)
        self.diffusion_model = diffusion_model
        self.first_stage_model = first_stage_model
        self.image_proj_model = image_proj_model
        
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.num_frames = num_frames
        self.image_size = image_size

        self.fs = fs
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.timestep_spacing = timestep_spacing
        self.guidance_rescale = guidance_rescale
        self.v_posterior = v_posterior
        self.channels = channels
        self.scale_factor = scale_factor
        
        self.register_schedule()
        self.share_buffers_with_diffusion_model()
        self.ddim_sampler = DDIMSampler(self.diffusion_model)
        
    
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=0.00085, linear_end=0.012, cosine_s=8e-3):
        
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                    cosine_s=cosine_s)
        betas = rescale_zero_terminal_snr(betas)
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))

        self.register_buffer('sqrt_recip_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
    def share_buffers_with_diffusion_model(self):
        self.diffusion_model.betas = self.betas
        self.diffusion_model.alphas_cumprod = self.alphas_cumprod
        self.diffusion_model.alphas_cumprod_prev = self.alphas_cumprod_prev
        self.diffusion_model.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
        self.diffusion_model.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        self.diffusion_model.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod
        self.diffusion_model.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod
        self.diffusion_model.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod
        self.diffusion_model.posterior_variance = self.posterior_variance
        self.diffusion_model.posterior_log_variance_clipped = self.posterior_log_variance_clipped
        self.diffusion_model.posterior_mean_coef1 = self.posterior_mean_coef1
        self.diffusion_model.posterior_mean_coef2 = self.posterior_mean_coef2
        
        
    def decode_core(self, z):
        b, _, t, _, _ = z.shape
        z = rearrange(z, 'b c t h w -> (b t) c h w')
        reshape_back = True
  
        z = 1. / self.scale_factor * z
        results = self.first_stage_model.decode(z)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results
    
    def forward(self, video_embeds):
        fs = torch.tensor([self.fs] * 1, dtype=torch.long, device=video_embeds.device)
        
        video_embeds = self.image_proj_model(video_embeds)
        cond = video_embeds

        fake_video = torch.zeros(1, self.num_frames, 3, self.image_size, self.image_size, dtype=video_embeds.dtype).to(video_embeds.device)
        uc_video_embeds = self.video_tokenizer(fake_video)

        uc_video_embeds = self.image_proj_model(uc_video_embeds)
        uc = uc_video_embeds

        z0 = None
        cond_mask = None
        cond_z0 = None
        
        h, w = self.image_size // 8, self.image_size // 8
        noise_shape = [1, self.channels, 16, h, w]

        samples, _ = self.ddim_sampler.sample(S=self.ddim_steps,
                                        conditioning=cond,
                                        batch_size=1,
                                        shape=noise_shape[1:],
                                        verbose=False,
                                        unconditional_guidance_scale=self.unconditional_guidance_scale,
                                        unconditional_conditioning=uc,
                                        eta=self.ddim_eta,
                                        cfg_img=None, 
                                        mask=cond_mask,
                                        x0=cond_z0,
                                        fs=fs,
                                        timestep_spacing=self.timestep_spacing,
                                        guidance_rescale=self.guidance_rescale,
                                        ) 
        decode_images = self.decode_core(samples)
        
        return decode_images

    
    @classmethod
    def from_pretrained(cls, video_tokenizer, diffusion_model, first_stage_model, image_proj_model, pretrained_model_path=None, **kwargs):
        model = cls(video_tokenizer=video_tokenizer, diffusion_model=diffusion_model, first_stage_model=first_stage_model, image_proj_model=image_proj_model, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('agent model, missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
            print('agent model, missing keys: ', missing)
            print('unexpected keys:', unexpected)
        return model