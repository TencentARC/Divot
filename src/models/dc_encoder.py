import torch
import torch.nn as nn
import kornia
import math
from collections import OrderedDict
import open_clip
from einops import rearrange
from timm.models.layers import LayerNorm
from .vit import Block
from functools import partial

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

def autocast(f):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled=True,
                                     dtype=torch.get_autocast_gpu_dtype(),
                                     cache_enabled=torch.is_autocast_cache_enabled()):
            return f(*args, **kwargs)
    return do_autocast

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):
    def encode(self, x):
        return x


class FrozenOpenCLIPImageEmbedderV2(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda",
                 freeze=True, layer="pooled", antialias=True):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'),
                                                            pretrained=version, )
        del model.transformer
        self.model = model
        self.device = device

        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)


    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x.cpu(), self.mean.cpu(), self.std.cpu()).to(x.device)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, preprocess=True, no_dropout=False):
        ## image: b c h w
        z = self.encode_with_vision_transformer(image, preprocess)
        return z

    def encode_with_vision_transformer(self, x, preprocess):
        if preprocess:
            x = self.preprocess(x)

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.model.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.model.visual.grid_size[0], self.model.visual.patch_size[0], self.model.visual.grid_size[1], self.model.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.model.visual.grid_size[0] * self.model.visual.grid_size[1], -1)
            x = self.model.visual.patchnorm_pre_ln(x)
            x = self.model.visual.conv1(x)
        else:
            x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class MotionPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: int = 8,
        width: int = 1024,
        heads: int = 12,
        context_length: int = 16,  # number of frames
        causal: bool = False,  # causal mask in transformer
        zero_init: bool = False,
    ):
        super().__init__()

        self.context_length = context_length

        if causal:
            self.context_length = context_length
            mask = self.build_attention_mask()
        self.transformer = Transformer(width, layers, heads, mask if causal else None)

        self.input_proj = nn.Linear(input_dim, width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, width)
        )
        self.ln_final = LayerNorm(width)
        self.output_proj = nn.Linear(width, output_dim)

        self._init_weights(zero_init=zero_init)

    def _init_weights(self, zero_init: bool = False):
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if zero_init:
            nn.init.eye_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
            nn.init.zeros_(self.positional_embedding)
            for block in self.transformer.resblocks:
                nn.init.zeros_(block.attn.out_proj.weight)
                nn.init.zeros_(block.attn.out_proj.bias)
                nn.init.zeros_(block.mlp.c_proj.weight)
                nn.init.zeros_(block.mlp.c_proj.bias)
            # Skip ln_final as they are already initialized
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor):
        x = self.input_proj(x)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        ## ! This is modified for zero init
        # no residual
        x = self.ln_final(x)
        x = self.output_proj(x)
        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


class PerceiverAttention(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler_DC(nn.Module):

    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        norm_out=True,
        ff_mult=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)

        if norm_out:
            self.norm_out = nn.LayerNorm(output_dim)
        else:
            self.norm_out = nn.Identity()

        self.in_dim = dim
        self.out_dim = output_dim

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ]))

    def forward(self, x):

        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        output_embeds = self.norm_out(latents)

        return output_embeds


class SelfAttention_3D_Pos_Down(nn.Module):
    def __init__(
        self,
        dim=1024,
        num_heads=8,
        transformer_depth=4,
        resampler_depth=6,
        num_queries=4,
        num_frames=5,
        do_normalize=True,
        use_grad_checkpointing=False,
        num_queries_down=64,
        dim_down=256,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads=num_heads
        self.transformer_depth = transformer_depth
        self.resampler_depth = resampler_depth
        self.num_queries = num_queries
        self.num_frames = num_frames
        self.num_frame_tokens = int(num_queries / num_frames)
        self.use_grad_checkpointing = use_grad_checkpointing
        self.do_normalize = do_normalize
        self.num_queries_down = num_queries_down
        self.dim_down = dim_down

        self.blocks = nn.ModuleList([
            Block(
                dim=self.dim, num_heads=self.num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), use_grad_checkpointing=self.use_grad_checkpointing)
            for i in range(self.transformer_depth)])

        scale = self.dim ** -0.5
        self.pos_embed_spatial = nn.Parameter(scale * torch.randn(self.num_frame_tokens, self.dim))
        self.pos_embed_temporal = nn.Parameter(scale * torch.randn(self.num_frames, self.dim))

        if self.do_normalize:
            self.norm = nn.LayerNorm(self.dim)
        else:
            self.norm = nn.Identity()

        self.down_sampler = Resampler_DC(dim=self.dim, depth=resampler_depth, num_queries=num_queries_down, embedding_dim=self.dim, output_dim=self.dim_down)

    def forward(self, x):
        bz = x.shape[0]
        x = x.reshape(bz, self.num_frames, self.num_frame_tokens, -1) + self.pos_embed_spatial.unsqueeze(0).unsqueeze(1)  + self.pos_embed_temporal.unsqueeze(0).unsqueeze(2)
        x = x.reshape(bz, -1, x.shape[-1])

        for blk in self.blocks:
            x = blk(x)

        hidden_embeds = self.norm(x)

        hidden_embeds_down = self.down_sampler(hidden_embeds)

        return hidden_embeds_down

    def proj_embeds(self, x):
        prompt_embeds = self.proj(x)

        return prompt_embeds


class VideoRepreClipTemporalPoolModel(nn.Module):
    def __init__(self, num_frames, input_dim, hidden_size, width, temporal_depth, pool_size, use_norm, transformer_depth, resampler_depth, st_num_tokens, down_num_tokens, resampler_dim, pretrained_model_path=None):
        super().__init__()
        self.vit_model = FrozenOpenCLIPImageEmbedderV2(freeze=True, version='pretrained/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin')

        self.temporal_encoder = MotionPredictor(
            input_dim=input_dim,
            output_dim=hidden_size,
            layers=temporal_depth,
            width=width,
            context_length=num_frames,
            causal=False,
            zero_init=True
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        self.vit_model.eval()
        for param in self.vit_model.parameters():
            param.requires_grad = False

        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(hidden_size)

        self.image_proj_model = SelfAttention_3D_Pos_Down(dim=hidden_size, transformer_depth=transformer_depth, resampler_depth=resampler_depth, num_queries=st_num_tokens, num_frames=num_frames, do_normalize=True, use_grad_checkpointing=False, num_queries_down=down_num_tokens, dim_down=resampler_dim)


    def forward(self, images):
        bz = images.shape[0]
        n_frames = images.shape[1]
        images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
        image_features = self.vit_model(images)[:, 1:]
        image_features = image_features.reshape(bz, n_frames, image_features.shape[1], image_features.shape[2])

        video_features = rearrange(image_features, "b f n c -> (b n) f c", b=bz)

        video_features_temporal = self.temporal_encoder(video_features)
        video_features_temporal = rearrange(video_features_temporal, "(b n) f c -> b f n c", b=bz)
        hw = int(video_features_temporal.size(2) ** 0.5)
        video_features_temporal = rearrange(video_features_temporal , "b f (h w) c -> (b f) c h w", h=hw, w=hw)

        video_features_pool = self.spatial_pool(video_features_temporal)

        video_features_final = rearrange(video_features_pool, "(b f) c h w -> b (f h w) c", b=bz)

        if self.use_norm:
            video_features_final = self.norm(video_features_final)

        video_features_final = self.image_proj_model(video_features_final)

        return video_features_final

    @classmethod
    def from_pretrained(cls, pretrained_model_path=None, **kwargs):
        model = cls(**kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('missing keys: ', missing)
            print('unexpected keys:', unexpected)
            print('unexpected keys can only include vit_model')
        return model
