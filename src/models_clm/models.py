import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from transformers import LogitsProcessorList
from .generation import AutoVideoTokenGenerationProcessor

IMG_BOI_TOKEN = "<img>"
IMG_EOI_TOKEN = "</img>"
VID_BOI_TOKEN = "<vid>"
VID_EOI_TOKEN = "</vid>"
BOI_TOKEN = "<frame>"
EOI_TOKEN = "</frame>"
FRAME_TOKEN = '<frame_{:05d}>'


def cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss


class ProjectionLayer(nn.Module):

    def __init__(self, num_queries, embed_dim, input_dim, output_dim) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)


def gmm_check_params(means, variances, weights):
    assert (
        means.ndim == 4
    ), "means should have shape (batch_size, num_tokens, k, d), got {}".format(
        means.shape
    )
    assert (
        means.shape == variances.shape
    ), "means and variances should have the same shape, got {} and {}".format(
        means.shape, variances.shape
    )
    assert means.shape[:3] == weights.shape


def gmm_split_params(gmm_params, k, d, var_scale=1.0):
    # Note that returned weights are logits instead of probabilities
    batch_size, num_tokens, _ = gmm_params.shape

    means = gmm_params[..., : k * d].reshape(batch_size, num_tokens, k, d)

    variances = gmm_params[..., k * d : 2 * k * d].reshape(batch_size, num_tokens, k, d)
    variances = torch.clamp(F.softplus(variances), min=1e-5)
    variances = variances * var_scale

    weights = gmm_params[..., 2 * k * d :]

    return means, variances, weights


def compute_gmm_loss(means, variances, weights, targets):
    # Note that the input weights are logits
    # gmm_check_params(means, variances, weights)
    assert means.shape[-1] == targets.shape[-1]

    # Create the Gaussian Mixture Model
    mixture = D.Categorical(logits=weights)
    components = D.Independent(D.Normal(means, torch.sqrt(variances)), 1)
    gmm = D.MixtureSameFamily(mixture, components)

    # Compute the negative log-likelihood and scale it by the dimensionality d
    log_probs = gmm.log_prob(targets)
    nll = -log_probs.mean() / targets.shape[-1]  # Scale NLL by dimension d

    return nll


def gmm_predict(means, weights):
    # Note that the input weights are logits
    weighted_means = torch.einsum("bnkd,bnk->bnd", means, weights.softmax(-1))
    return weighted_means


def gmm_sample_weighted(means, variances, weights):
    # Note that the input weights are logits
    # gmm_check_params(means, variances, weights)

    # Reshape means and variances
    std = torch.sqrt(variances)
    normal_dist = D.Normal(means, std)
    samples = normal_dist.sample()  # Shape: (batch_size, num_tokens, k, d)

    probs = weights.softmax(-1)
    samples = (samples * probs.unsqueeze(-1)).sum(-2)

    return samples


def gmm_sample(means, variances, weights):
    # Note that the input weights are logits
    # gmm_check_params(means, variances, weights)

    batch_size, num_tokens, k, d = means.shape

    # Reshape weights and sample component indices
    weights_flat = weights.view(-1, k)  # Flatten to (batch_size * num_tokens, k)
    mixture_dist = D.Categorical(logits=weights_flat)
    component_indices = mixture_dist.sample()  # Shape: (batch_size * num_tokens,)

    # Reshape means and variances and select based on sampled indices
    means_flat = means.view(-1, k, d)  # Flatten to (batch_size * num_tokens, k, d)
    variances_flat = variances.view(-1, k, d)  # Same as means

    # Use advanced indexing to select the means and variances
    means_selected = means_flat[
        torch.arange(means_flat.size(0)), component_indices
    ]  # Shape: (batch_size * num_tokens, d)
    variances_selected = variances_flat[
        torch.arange(variances_flat.size(0)), component_indices
    ]

    # Compute standard deviations and sample from the normal distributions
    std_selected = torch.sqrt(variances_selected)
    normal_dist = D.Normal(means_selected, std_selected)
    samples_flat = normal_dist.sample()  # Shape: (batch_size * num_tokens, d)

    # Reshape samples back to (batch_size, num_tokens, d)
    samples = samples_flat.view(batch_size, num_tokens, d)

    return samples


def gmm_params_to_predictions(params, k, do_sample=False, var_scale=1.0, weighted_sample=False):
    # The output of GMM loss is GMM params, we need to sample from it
    d = (params.shape[-1] - k) // k // 2
    assert 2 * k * d + k == params.shape[-1], \
        "Invalid GMM params, k = {}, 2 * k * d + k = {}".format(k, params.shape[-1])
    means, variances, weights = gmm_split_params(params, k, d, var_scale=var_scale)
    if do_sample:
        if weighted_sample:
            predictions = gmm_sample_weighted(means, variances, weights)
        else:
            predictions = gmm_sample(means, variances, weights)
    else:
        predictions = gmm_predict(means, weights)

    return predictions


class ContinuousLVLM_Video_Comp_Gen(nn.Module):
    def __init__(self, llm, input_resampler, output_resampler, freeze_input_resampler=False, freeze_output_resampler=False, num_frames=4, num_gmm_kernel=16, lm_loss_scale=1.0, rec_loss_scale=0.0, l1_loss_scale=0.0, gmm_loss_scale=0.0) -> None:
        super().__init__()
        self.llm = llm
        self.input_resampler = input_resampler
        self.output_resampler = output_resampler
        self.freeze_input_resampler = freeze_input_resampler
        self.freeze_output_resampler = freeze_output_resampler
        self.num_frames = num_frames
        self.lm_loss_scale = lm_loss_scale
        self.rec_loss_scale = rec_loss_scale
        self.l1_loss_scale = l1_loss_scale
        self.gmm_loss_scale = gmm_loss_scale

        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()

        self.num_gmm_kernel = num_gmm_kernel

        if self.freeze_input_resampler:
            self.input_resampler = self.input_resampler.requires_grad_(False)

        if self.freeze_output_resampler:
            self.output_resampler = self.output_resampler.requires_grad_(False)


    def forward(self, input_ids, attention_mask, labels, video_embeds, embeds_gen_mask, embeds_cmp_mask, ids_gen_mask,
                ids_cmp_mask):

        input_embeds = self.llm.get_input_embeddings()(input_ids)  # bz x seq_len x dim, 4 x 160 x 4096

        bz, sq, dim = input_embeds.shape

        has_image_input = embeds_cmp_mask.sum().item() > 0
        has_image_output = embeds_gen_mask.sum().item() > 0

        num_clips = video_embeds.shape[1]

        if has_image_input:
            video_embeds_comp = video_embeds.reshape(bz, -1, video_embeds.shape[-1])
            video_embeds_in = self.input_resampler(video_embeds_comp)
            video_embeds_in = video_embeds_in.reshape(bz, num_clips, -1, video_embeds_in.shape[-1])
            input_embeds[ids_cmp_mask] = video_embeds_in[embeds_cmp_mask].reshape(-1, video_embeds_in.shape[-1])
        elif not self.freeze_input_resampler:
            video_embeds_comp_fake = torch.randn(bz, self.input_resampler.num_queries, self.input_resampler.input_dim).to(input_embeds.device, dtype=input_embeds.dtype)
            video_embeds_in_fake = self.input_resampler(video_embeds_comp_fake)
            input_embeds[:, :self.input_resampler.num_queries] = input_embeds[:, :self.input_resampler.num_queries] + 0.0 * video_embeds_in_fake

        output_lm = self.llm(attention_mask=attention_mask,
                             inputs_embeds=input_embeds,
                             labels=labels,
                             output_hidden_states=True,
                             return_dict=True)
        lm_loss = output_lm['loss']

        if has_image_output:
            # only for single-clip generation
            assert num_clips == 1
            embeds_gen_mask = embeds_gen_mask.reshape(-1)
            video_embeds_gen = video_embeds.reshape(bz, -1, video_embeds.shape[-1])
            last_hidden_state = output_lm.hidden_states[-1]
            target_embeds = video_embeds_gen[embeds_gen_mask]
            output_image_embeds = last_hidden_state[ids_gen_mask].view(bz, -1, dim)
            recon_image_embeds = self.output_resampler(output_image_embeds)
            means, variances, weights = gmm_split_params(
                recon_image_embeds, self.num_gmm_kernel, self.output_resampler.embed_dim
            )
            gmm_loss = compute_gmm_loss(
                means, variances, weights, target_embeds
            )
        else:
            if self.freeze_output_resampler:
                 gmm_loss = torch.zeros_like(lm_loss)
            else:
                output_image_embeds_fake = torch.randn(bz, self.output_resampler.num_queries,
                                                self.output_resampler.input_dim).to(input_embeds.device, dtype=input_embeds.dtype)
                recon_image_embeds_fake = self.output_resampler(output_image_embeds_fake)
                target_embeds_fake = torch.randn(bz, self.output_resampler.num_queries,
                                            self.output_resampler.output_dim).to(input_embeds.device, dtype=input_embeds.dtype)
                gmm_loss = self.mse(recon_image_embeds_fake, target_embeds_fake) * 0.0

        total_loss = self.lm_loss_scale * lm_loss + self.gmm_loss_scale * gmm_loss
        return {'total_loss': total_loss, 'lm_loss': lm_loss, 'gmm_loss': gmm_loss}


    def generate(self,
                 tokenizer,
                 input_ids=None,
                 attention_mask=None,
                 ids_gen_mask=None,
                 ids_cmp_mask=None,
                 video_embeds=None,
                 embeds_cmp_mask=None,
                 logits_processor=None,
                 num_vid_in_tokens=100,
                 num_vid_out_tokens=100,
                 var_scale=0.1,
                 temperature=0.7,
                 num_beams=1,
                 max_new_tokens=256,
                 top_p=0.5,
                 for_text=False,
                 eos_token=None,
                 device="cuda"):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(
                AutoVideoTokenGenerationProcessor(tokenizer=tokenizer, num_vid_gen_tokens=num_vid_out_tokens))

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        input_ids = input_ids.to(device=device)
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        bz, sq, dim = input_embeds.shape

        if video_embeds is not None:
            assert embeds_cmp_mask is not None and ids_cmp_mask is not None
            with torch.no_grad():
                num_clips_in = video_embeds.shape[1]
                video_embeds = video_embeds.reshape(bz, -1, video_embeds.shape[-1])
                video_embeds_in = self.input_resampler(video_embeds)
                video_embeds_in = video_embeds_in.reshape(bz, num_clips_in, -1, video_embeds_in.shape[-1])
                input_embeds[ids_cmp_mask] = video_embeds_in[embeds_cmp_mask].reshape(-1, video_embeds_in.shape[-1])

        if for_text:
            generation_config = {
                'temperature': temperature,
                'num_beams': num_beams,
                'max_new_tokens': max_new_tokens,
                'top_p': top_p,
                'do_sample': True
            }

            eos_token_id = tokenizer.encode(eos_token, add_special_tokens=False)[0]
            output = self.llm.generate(input_ids=input_ids,
                                    inputs_embeds=input_embeds,
                                    output_hidden_states=True,
                                    return_dict_in_generate=True,
                                    eos_token_id=eos_token_id,
                                    logits_processor=logits_processor,
                                    **generation_config)
            generate_ids = output.sequences[0][input_ids.shape[1]:]
            generate_text = tokenizer.decode(generate_ids, skip_special_tokens=False)
            return generate_text

        attention_mask = attention_mask.to(device=device)
        output = self.llm(
            inputs_embeds=input_embeds,
            return_dict=True,
            use_cache=True,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        output_embeds = output.hidden_states[-1][ids_gen_mask].unsqueeze(0)
        print(input_embeds.shape, output_embeds.shape)

        vid_gen_feats = self.output_resampler(output_embeds)

        if self.gmm_loss_scale != 0.0:
            vid_gen_feats = gmm_params_to_predictions(vid_gen_feats, self.num_gmm_kernel, do_sample=True, var_scale=var_scale, weighted_sample=False)

        return vid_gen_feats


    @classmethod
    def from_pretrained(cls, llm, input_resampler, output_resampler, pretrained_model_path=None, **kwargs):
        model = cls(llm=llm, input_resampler=input_resampler, output_resampler=output_resampler, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('agent model, missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
            print('agent model, missing keys: ', missing)
            print('unexpected keys:', unexpected)
        return model
