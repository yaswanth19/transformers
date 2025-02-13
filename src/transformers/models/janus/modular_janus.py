from transformers import PreTrainedModel, GenerationMixin
from .configuration_janus import JanusConfig, JanusGenVisionConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class JanusGenVisionEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        z_channels=256,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(
                    JanusGenVisionResnetBlock(
                        block_in, block_out, dropout=dropout, norm_type=norm_type
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(JanusGenVisionAttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = JanusGenVisionDownsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(
            JanusGenVisionResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )
        self.mid.append(JanusGenVisionAttnBlock(block_in, norm_type=norm_type))
        self.mid.append(
            JanusGenVisionResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )

        # end
        self.norm_out = JanusGenVisionNormalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # end
        h = self.norm_out(h)
        h = janus_gen_vision_nonlinearity(h)
        h = self.conv_out(h)
        return h


class JanusGenVisionDecoder(nn.Module):
    def __init__(
        self,
        z_channels=256,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        out_channels=3,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(
            JanusGenVisionResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )
        self.mid.append(JanusGenVisionAttnBlock(block_in, norm_type=norm_type))
        self.mid.append(
            JanusGenVisionResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(
                    JanusGenVisionResnetBlock(
                        block_in, block_out, dropout=dropout, norm_type=norm_type
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(JanusGenVisionAttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = JanusGenVisionUpsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = JanusGenVisionNormalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(
            block_in, out_channels, kernel_size=3, stride=1, padding=1
        )

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = janus_gen_vision_nonlinearity(h)
        h = self.conv_out(h)
        return h


class JanusGenVisionVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(
                self.embedding.weight.data, p=2, dim=-1
            )
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum("b c h w -> b h w c", z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, torch.einsum("n d -> d n", embedding)
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * janus_gen_vision_compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum("b h w c -> b c h w", z_q)

        return (
            z_q,
            (vq_loss, commit_loss, entropy_loss),
            (perplexity, min_encodings, min_encoding_indices),
        )

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class JanusGenVisionResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        norm_type="group",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = JanusGenVisionNormalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = JanusGenVisionNormalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = janus_gen_vision_nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = janus_gen_vision_nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class JanusGenVisionAttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.norm = JanusGenVisionNormalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def janus_gen_vision_nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def JanusGenVisionNormalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "batch":
        return nn.SyncBatchNorm(in_channels)


class JanusGenVisionUpsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        if x.dtype != torch.float32:
            x = F.interpolate(x.to(torch.float), scale_factor=2.0, mode="nearest").to(
                torch.bfloat16
            )
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.with_conv:
            x = self.conv(x)
        return x


class JanusGenVisionDownsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def janus_gen_vision_compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


class JanusGenVision(PreTrainedModel):
    config_class = JanusGenVisionConfig
    base_model_prefix = "janus_gen_vision"
    main_input_name = "pixel_values"


    def __init__(self, config: JanusGenVisionConfig):
        super().__init__()
        self.config = config
        self.encoder = JanusGenVisionEncoder(
            ch_mult=config.encoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )
        self.decoder = JanusGenVisionDecoder(
            ch_mult=config.decoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )

        self.quantize = JanusGenVisionVectorQuantizer(
            n_e=config.codebook_size,
            e_dim=config.codebook_embed_dim,
            beta=config.commit_loss_beta,
            entropy_loss_ratio=config.entropy_loss_ratio,
            l2_norm=config.codebook_l2_norm,
            show_usage=config.codebook_show_usage,
        )

        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(
            config.codebook_embed_dim, config.z_channels, 1
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


class JanusPreTrainedModel(PreTrainedModel):
    config_class = JanusConfig
    base_model_prefix = "janus"

    def _init_weights(self, module):
        pass

class JanusForConditionalGeneration(JanusPreTrainedModel, GenerationMixin):
    config_class = JanusConfig

    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        pass



__all__ = [
    "JanusGenVision",
    "JanusPreTrainedModel",
    "JanusForConditionalGeneration"
]
import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
    torch_int,
)

from .configuration_janus import JanusVisionEncoderConfig
from ..vit.modeling_vit import ViTPatchEmbeddings
from ..dinov2_with_registers.modeling_dinov2_with_registers  import Dinov2WithRegistersLayerScale, Dinov2WithRegistersDropPath
from ..siglip.modeling_siglip import SiglipEncoder, SiglipVisionTransformer, SiglipVisionModel, SiglipMultiheadAttentionPoolingHead

class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """
    return_indices: torch.jit.Final[bool]

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
            return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(self, x) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.training or self.prob == 0.:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1. - self.prob)))
        keep_indices = torch.argsort(torch.randn(B, L, device=x.device), dim=-1)[:, :num_keep]
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x

class JanusVisionEncoderPatchEmbeddings(ViTPatchEmbeddings):
    pass


class JanusVisionEncoderEmbeddings(nn.Module):
    def __init__(self, config:JanusVisionEncoderConfig, use_special_tokens: bool,):
        super().__init__()

        self.use_special_tokens = use_special_tokens
        if use_special_tokens:
            self.cls_token = nn.Parameter(torch.rand(1,1,config.hidden_size))
            self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))

        # Currently using hidden_drop_rate instead of positional_dropout_rate, is it necessary?
        self.dropout = nn.Dropout(config.hidden_dropout_rate)
        self.patch_embeddings = JanusVisionEncoderPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches

        num_prefix_tokens = config.num_register_tokens + 1
        pos_embed_len = self.num_patches + num_prefix_tokens if use_special_tokens else self.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, pos_embed_len, config.hidden_size) * 0.02)

        # Used to reduce computationality.
        self.patch_dropout = PatchDropout(config.drop_path_rate, num_prefix_tokens) if config.drop_path_rate else nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # Add CLS and Register token embeddings.
        special_token_embeddings = []
        if self.use_special_tokens:
            cls_token_embeddings = self.cls_token.expand((batch_size, -1,-1))
            special_token_embeddings.append(cls_token_embeddings)

            if self.register_tokens.shape[1]:
                register_token_embeddings = self.register_tokens.expand((batch_size, -1,-1))
                special_token_embeddings.append(register_token_embeddings)

        if self.use_special_tokens:
            embeddings = embeddings + self.position_embeddings
            embeddings = torch.cat(special_token_embeddings+[embeddings], dim=1)
        else:
            # embeddings = torch.cat(special_token_embeddings+[embeddings], dim=1)
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)


        # Perform Patch dropout
        embeddings = self.patch_dropout(embeddings)
        return embeddings

# Todo: introduce compatiability for cache
class JanusVisionEncoderAttention(nn.Module):
    """Attention Class for Janus Vision Encoder """
    def __init__(self, config: JanusVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        proj_dropout = config.projection_dropout
        qk_norm = config.use_qk_norm

        # Split the weights manually and checkif getting correct output or not
        self.qkv = nn.Linear(self.embed_dim, 3*self.embed_dim, bias=config.qkv_bias)
        self.projection_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        self.query_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()
        self.key_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()

    def forward(self,hidden_states:torch.Tensor,attention_mask: Optional[torch.Tensor] = None, output_attentions: Optional[torch.Tensor] = None):
        batch_size , seq_len, _ = hidden_states.size()

        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = qkv.unbind(0)

        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        # Is it a bug or deliberate change?
        query_states = query_states * self.scale

        # batch, num head,seqlen,seqlen
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Only apply attention dropout during training.
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (batch_size,1, seq_len, self.head_dim):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, seq_len, self.head_dim)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape( batch_size, seq_len, self.embed_dim )

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)

        outputs = (output, attn_weights) if output_attentions else (output, None)
        return outputs

class JanusVisionEncoderLayerScale(Dinov2WithRegistersLayerScale):
    pass
class JanusVisionEncoderDropPath(Dinov2WithRegistersDropPath):
    pass

class JanusVisionEncoderMLP(nn.Module):
    def __init__(self, config:JanusVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act] # Gelu act
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_rate)
        self.dropout2 = nn.Dropout(config.hidden_dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        return hidden_states

class JanusVisionEncoderLayer(nn.Module):
    def __init__(self,config: JanusVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = JanusVisionEncoderAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)

        self.layer_scale1 = JanusVisionEncoderLayerScale(config) if config.layerscale_value else nn.Identity()
        self.layer_scale2 = JanusVisionEncoderLayerScale(config) if config.layerscale_value else nn.Identity()
        self.drop_path1 = (
            JanusVisionEncoderDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            JanusVisionEncoderDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        self.mlp = JanusVisionEncoderMLP(config)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # Pre-Norm before attention .
        norm_hidden_states = self.layer_norm1(hidden_states)
        attn_output, attn_weights = self.self_attn(
            norm_hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )

        scaled_attn_output = self.layer_scale1(attn_output)
        dropped_attn_output = self.drop_path1(scaled_attn_output)
        hidden_states = hidden_states + dropped_attn_output

        norm_hidden_states = self.layer_norm2(hidden_states)

        mlp_output = self.mlp(norm_hidden_states)

        scaled_mlp_output = self.layer_scale2(mlp_output)
        dropped_mlp_output = self.drop_path2(scaled_mlp_output)
        hidden_states = hidden_states + dropped_mlp_output

        return (hidden_states, attn_weights if output_attentions else None)

# copied from SiglipMultiheadAttentionPoolingHead
# class JanusAttentionPoolLatent(SiglipMultiheadAttentionPoolingHead):
#     pass

class JanusAttentionPoolLatent(nn.Module):
    """ Hugging Face-compatible Attention Pooling with Manual QKV """

    def __init__(self, config: JanusVisionEncoderConfig):
        super().__init__()

        self.latent_len = getattr(config, "latent_len", 1)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mlp_ratio = getattr(config, "mlp_ratio", 4.0)
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention

        # Learnable latent query (probe)
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, self.hidden_size))

        # Linear layers for QKV projection
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.kv_proj = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Normalization & MLP
        self.norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = JanusVisionEncoderMLP(config)

        self.proj_drop = nn.Dropout(getattr(config, "dropout", 0.0))

    def forward(self, hidden_state):
        batch_size, seq_len, _ = hidden_state.shape

        # Expand learnable latent tokens for batch
        q_latent = self.latent.expand(batch_size, -1, -1)  # (B, latent_len, hidden_size)

        # Compute Q projection from latent tokens
        q = self.q_proj(q_latent)  # (B, latent_len, hidden_size)

        # Compute combined KV projection
        kv = self.kv_proj(hidden_state)  # (B, seq_len, 2 * hidden_size)
        k, v = kv.view(batch_size, seq_len, 2, self.num_heads, self.head_dim).unbind(2)
        # Batch_sisxe, num_heads, seq_len, head_dim
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # Reshape Q for multi-head attention (B, N, H) → (B, num_heads, N, head_dim)
        q = q.view(batch_size, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, latent_len, head_dim)

        # Scaled dot-product attention (without `torch.nn.MultiheadAttention`)
        attn_weights = (q @ k.transpose(2,3)) * self.scale  # (B, num_heads, latent_len, seq_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        x = attn_weights @ v  # (B, num_heads, latent_len, head_dim)

        x = x.transpose(1, 2).reshape(batch_size, self.latent_len, self.hidden_size)

        x = self.proj(x)
        x = self.proj_drop(x)

        # Residual connection + MLP
        residual = x
        x = self.norm(x)
        x = residual + self.mlp(x)

        return x[:, 0]


# Copied from siglip encoder
class JanusVisionEncoder(nn.Module):

    def __init__(self,config:JanusVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([JanusVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# Copied from siglip vision transformer
class JanusVisionEncoderTransformer(nn.Module):
    def __init__(self, config: JanusVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embeddings = JanusVisionEncoderEmbeddings(config,use_special_tokens=False)
        self.encoder = JanusVisionEncoder(config)
        self.layenorm = nn.LayerNorm(config.hidden_size)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = JanusAttentionPoolLatent(config)
            # Won't be using as a standalone classifier head hence no num classes

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.layenorm(last_hidden_state)
        pooler_output = self.head(hidden_states)


        if not return_dict:
            return (last_hidden_state) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )