# coding=utf-8
# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from functools import cached_property
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ..dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersDropPath,
    Dinov2WithRegistersLayerScale,
from ...generation import GenerationMixin
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
    torch_int,
)

from ..chameleon.modeling_chameleon import (
    ChameleonForConditionalGeneration,
    ChameleonImageVocabularyMapping,
    ChameleonModel,
    ChameleonPreTrainedModel,
    ChameleonVQVAE,
    ChameleonVQVAEEncoderAttnBlock,
    ChameleonVQVAEEncoderResnetBlock,
    ChameleonVQVAEVectorQuantizer,
    ChameleonVQVAEEncoder,
    ChameleonVQVAEEncoderConvDownsample
)
from .configuration_janus import JanusVisionConfig,JanusConfig,JanusTextConfig, JanusVQVAEConfig
from ..vit.modeling_vit import ViTPatchEmbeddings
from .configuration_janus import JanusConfig, JanusGenVisionConfig, JanusVisionEncoderConfig, JanusGenHeadConfig, \
    JanusGenAlignerConfig


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
                res_block.append(JanusGenVisionResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
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
        self.mid.append(JanusGenVisionResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(JanusGenVisionAttnBlock(block_in, norm_type=norm_type))
        self.mid.append(JanusGenVisionResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = JanusGenVisionNormalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

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
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(JanusGenVisionResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(JanusGenVisionAttnBlock(block_in, norm_type=norm_type))
        self.mid.append(JanusGenVisionResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(JanusGenVisionResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
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
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

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
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
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
                torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                + torch.sum(embedding ** 2, dim=1)
                - 2 * torch.einsum("bd,dn->bn", z_flattened, torch.einsum("n d -> d n", embedding))
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = JanusGenVisionNormalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

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
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

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
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == "batch":
        return nn.SyncBatchNorm(in_channels)


class JanusGenVisionUpsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = F.interpolate(x.to(torch.float), scale_factor=2.0, mode="nearest").to(torch.bfloat16)
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
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

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

    _no_split_modules = [
        "JanusGenVisionEncoder",
        "JanusGenVisionDecoder",
        "JanusGenVisionVectorQuantizer",
        "JanusGenVisionResnetBlock",
        "JanusGenVisionAttnBlock"
    ]

    def __init__(self, config: JanusGenVisionConfig):
        super().__init__(config)
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
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

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


class JanusGenHead(PreTrainedModel):
    config_class = JanusGenHeadConfig
    base_model_prefix = "janus_gen_head"

    def __init__(self, config: JanusGenHeadConfig):
        super().__init__(config)
        self.output_mlp_projector = torch.nn.Linear(config.n_embed, config.image_token_embed)
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(config.image_token_embed, config.image_token_size)

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


class JanusGenAligner(PreTrainedModel):
    config_class = JanusGenAlignerConfig
    base_model_prefix = "janus_gen_aligner"

    def __init__(self, config: JanusGenAlignerConfig):
        super().__init__(config)

        if config.projector_type == "identity":
            modules = nn.Identity()

        elif config.projector_type == "linear":
            modules = nn.Linear(config.input_dim, config.n_embed)

        elif config.projector_type == "mlp_gelu":
            mlp_depth = getattr(config, "depth", 1)
            modules = [nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            modules = nn.Sequential(*modules)

        elif config.projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = getattr(config, "depth", 1)
            self.high_up_proj = nn.Linear(config.input_dim, config.n_embed // 2)
            self.low_up_proj = nn.Linear(config.input_dim, config.n_embed // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

        self.layers = modules

    def forward(self, x_or_tuple: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Args:
            x_or_tuple (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:  if it is a tuple of torch.Tensor,
                then it comes from the hybrid vision encoder, and x = high_res_x, low_res_x);
                otherwise it is the feature from the single vision encoder.

        Returns:
            x (torch.Tensor): [b, s, c]
        """

        if isinstance(x_or_tuple, tuple):
            # config.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x_or_tuple
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        else:
            x = x_or_tuple

        return self.layers(x)


class JanusPreTrainedModel(PreTrainedModel):
    config_class = JanusConfig
    base_model_prefix = "janus"
    _no_split_modules = [
        "JanusGenHead",
        "JanusGenAligner",
    ]

    def _init_weights(self, module):
        pass


class JanusForCausalLM(JanusPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        pass

class JanusForConditionalGeneration(JanusPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.text_model = JanusForCausalLM._from_config(config.text_config)
        self.gen_head = JanusGenHead(config.gen_head_config)
        self.gen_aligner = JanusGenAligner(config.gen_aligner_config)
        self.gen_vision = JanusGenVision(config.gen_vision_config)
        self.gen_embed = torch.nn.Embedding(
            config.gen_vision_config.image_token_size, config.gen_vision_config.n_embed
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **kwargs):
        pass

from ..dinov2_with_registers.modeling_dinov2_with_registers  import Dinov2WithRegistersLayerScale, Dinov2WithRegistersDropPath
from ..siglip.modeling_siglip import SiglipEncoder, SiglipVisionTransformer, SiglipVisionModel, SiglipMultiheadAttentionPoolingHead
from ..llama.modeling_llama import LlamaModel, LlamaForCausalLM

if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward



if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "JanusConfig"
_CHECKPOINT_FOR_DOC = ""


class JanusVisionPatchEmbeddings(ViTPatchEmbeddings):
    pass

# ToDO: Is interpolate pos embeddings required for this model as of now passing?
@dataclass
class JanusVQVAEOutput:
    pass

class JanusVisionEmbeddings(nn.Module):
    def __init__(self, config:JanusVisionConfig):
        super().__init__()

        self.use_special_tokens = config.use_special_tokens
        if self.use_special_tokens:
            self.cls_token = nn.Parameter(torch.rand(1,1,config.hidden_size))
            self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))

        # Currently using hidden_drop_rate instead of positional_dropout_rate, is it necessary?
        self.dropout = nn.Dropout(config.hidden_dropout_rate)
        self.patch_embeddings = JanusVisionPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches

        num_prefix_tokens = config.num_register_tokens + 1
        pos_embed_len = self.num_patches + num_prefix_tokens if self.use_special_tokens else self.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, pos_embed_len, config.hidden_size) * 0.02)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype), interpolate_pos_encoding)

        # Add CLS and Register token embeddings
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
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings

# Todo: introduce compatiability for cache
class JanusVisionAttention(nn.Module):
    """Attention Class for Janus Vision Encoder """
    def __init__(self, config: JanusVisionConfig):
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

        # Batched computation of query, key, value states.
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Permute the dims of qkv vector and unravel it into query, key, value states.
        query_states, key_states, value_states = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        # Is it a bug or deliberate change?
        query_states = query_states * self.scale

        attn_weights = torch.matmul(query_states, key_states.transpose(2,3))

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (batch_size,1, seq_len, self.head_dim):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, seq_len, self.head_dim)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1 ,dtype=torch.float32).to(query_states.dtype)
        # Only apply attention dropout during training.
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape( batch_size, seq_len, self.embed_dim)

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)

        outputs = (output, attn_weights) if output_attentions else (output, None)
        return outputs

class JanusVisionFlashAttention2(JanusVisionAttention):
    """
    JanusVision flash attention module. This module inherits from `JanusVisionAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    is_causal = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        batch_size , seq_len, _ = hidden_states.size()

        # Batched computation of query, key, value states.
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states, key_states, value_states = qkv.permute.unbind(2)
        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (Idefics2VisionRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            seq_len,
            dropout=dropout_rate,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim).contiguous()
        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)

        # In `Flash Attenition` we don't return Attention weights, hence return None.
        return output, None

class JanusVisionSdpaAttention(JanusVisionAttention):
    """
    Janusvision attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `JanusVisionAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    is_causal = False

    # Adapted from transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "JanusVisionModel is using JanusVisionSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        batch_size , seq_len, _ = hidden_states.size()

        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        query_states, key_states, value_states = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if self.is_causal and seq_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)
        return output, None

JANUS_VISION_ATTENTION_CLASSES = {
    "eager": JanusVisionAttention,
    "sdpa": JanusVisionSdpaAttention,
    "flash_attention_2": JanusVisionFlashAttention2,
}


class JanusVisionLayerScale(Dinov2WithRegistersLayerScale):
    pass
class JanusVisionDropPath(Dinov2WithRegistersDropPath):
    pass

class JanusVisionMLP(nn.Module):
    def __init__(self, config:JanusVisionConfig):
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
    def __init__(self,config: JanusVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = JANUS_VISION_ATTENTION_CLASSES[config._attn_implementation](config=config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.layer_scale1 = JanusVisionLayerScale(config) if config.layerscale_value else nn.Identity()
        self.layer_scale2 = JanusVisionLayerScale(config) if config.layerscale_value else nn.Identity()
        self.drop_path1 = (
            JanusVisionDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            JanusVisionDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        self.mlp = JanusVisionMLP(config)

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

class JanusVisionAttentionPoolLatent(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()

        self.latent_len = getattr(config, "latent_len", 1)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mlp_ratio = getattr(config, "mlp_ratio", 4.0)
        self.scale = self.head_dim ** -0.5

        # Learnable latent query (probe)
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, self.hidden_size))

        # Linear layers for QKV projection
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.kv_proj = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Normalization & MLP
        self.norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = JanusVisionMLP(config)

        self.proj_drop = nn.Dropout(getattr(config, "dropout", 0.0))

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape

        # Expand learnable latent tokens for batch
        q_latent = self.latent.expand(batch_size, -1, -1)  # (B, latent_len, hidden_size)

        # Compute Q projection from latent tokens
        query_states = self.q_proj(q_latent)  # (B, latent_len, hidden_size)

        # Compute combined KV projection
        kv = self.kv_proj(hidden_states)
        key_states, value_states = kv.view(batch_size, seq_len, 2, self.num_heads, self.head_dim).unbind(2)

        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        query_states = query_states.view(batch_size, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)  # (B, num_heads, latent_len, head_dim)

        # Validate shape
        if attn_output.size() != (batch_size, self.num_heads, self.latent_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, self.latent_len, self.head_dim)},"
                f" but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, self.latent_len, self.hidden_size)

        output = self.proj(attn_output)
        output = self.proj_drop(output)

        output = output + self.mlp(self.norm(output))

        return output[:, 0]


class JanusVisionEncoder(SiglipEncoder):
    def __init__(self,config:JanusVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([JanusVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class JanusPreTrainedModel():
     """An abstract class to load pretrained weigths"""
     pass

class JanusVisionTransformer(SiglipVisionTransformer,nn.Module):
    def __init__(self, config: JanusVisionConfig):
        nn.Module.__init__()
        self.config = config
        self.post_layernorm = nn.LayerNorm(config.hidden_size)
        self.embeddings = JanusVisionEmbeddings(config)
        self.encoder = JanusVisionEncoder(config)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = JanusVisionAttentionPoolLatent(config)

class JanusVisionAlignerMLP(nn.Module):
    def __init__(self, config:JanusVisionConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.aligner_projection_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(config.aligner_projection_size, config.aligner_projection_size) for _ in range(1, config.num_aligner_hidden_states)
        ])
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states

class JanusImageVocabularyMapping(ChameleonImageVocabularyMapping):
    @cached_property
    def bpe2img_mapping_tensor(self):
        mapping = torch.zeros(max(self.bpe2img.keys()) + 1, dtype=torch.int)
        for k, v in self.bpe2img.items():
            mapping[k] = v
        return mapping

class JanusTextModel(LlamaModel):
    pass
class JanusTextForCausalLM(LlamaForCausalLM, JanusPreTrainedModel, GenerationMixin):
    config_class = JanusTextConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = JanusTextModel(config)

class JanusForConditionalGeneration(JanusPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["language_model.lm_head.weight"]
    _supports_static_cache = False  # `get_image_tokens()`, called when `pixel_values` is passed, is not compilable.

    def __init__(self, config):
        super().__init__(config)
        self.vision_model = JanusVisionTransformer(config.vision_config)
        self.language_model = JanusTextForCausalLM(config.text_config)
        self.aligner = JanusVisionAlignerMLP(config.vision_config)
        self.vqmodel = JanusVQVAE(config.vq_config)
        # self.vocabulary_mapping = JanusImageVocabularyMapping(config.vocabulary_map)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self,input_ids):
        return self.language_model.get_input_embeddings()(input_ids)

    def get_image_embeddings(self,pixel_values):
        image_embeds = self.vision_model(pixel_values)
        image_embeds = self.aligner(image_embeds.last_hidden_state)
        return image_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None:
            image_embeds = self.get_image_embeddings(pixel_values)
            special_image_mask = input_ids == 100581
            image_embeds_flat = image_embeds.reshape(-1, 2048)
            special_image_mask = special_image_mask.unsqueeze(-1).expand(-1, -1, 2048)

            text_embeds = self.get_input_embeddings(input_ids)
            image_embeds = image_embeds.to(text_embeds.device, text_embeds.dtype)
            inputs_embeds = text_embeds.masked_scatter(special_image_mask, image_embeds_flat)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        )

        return outputs

__all__ = ["JanusGenVision",
           "JanusPreTrainedModel",
           "JanusForConditionalGeneration",
           "JanusForCausalLM"
           ]