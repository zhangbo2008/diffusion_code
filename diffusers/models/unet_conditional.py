import functools
import math
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin
from ..modeling_utils import ModelMixin
from .attention import AttentionBlock, SpatialTransformer
from .embeddings import GaussianFourierProjection, get_timestep_embedding
from .resnet import Downsample2D, FirDownsample2D, FirUpsample2D, ResnetBlock2D, Upsample2D
from .unet_new import UNetMidBlock2DCrossAttn, get_down_block, get_up_block


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1, dim2, method="cat"):
        super().__init__()
        # 1x1 convolution with DDPM initialization.
        self.Conv_0 = nn.Conv2d(dim1, dim2, kernel_size=1, padding=0)
        self.method = method


#    def forward(self, x, y):
#        h = self.Conv_0(x)
#        if self.method == "cat":
#            return torch.cat([h, y], dim=1)
#        elif self.method == "sum":
#            return h + y
#        else:
#            raise ValueError(f"Method {self.method} not recognized.")


class TimestepEmbedding(nn.Module):
    def __init__(self, channel, time_embed_dim, act_fn="silu"):
        super().__init__()

        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.act = None
        if act_fn == "silu":
            self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels, flip_sin_to_cos, downscale_freq_shift):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class UNetConditionalModel(ModelMixin, ConfigMixin):
    """
    The full UNet model with attention and timestep embedding. :param in_channels: channels in the input Tensor. :param
    model_channels: base channel count for the model. :param out_channels: channels in the output Tensor. :param
    num_res_blocks: number of residual blocks per downsample. :param attention_resolutions: a collection of downsample
    rates at which
        attention will take place. May be a set, list, or tuple. For example, if this contains 4, then at 4x
        downsampling, attention will be used.
    :param dropout: the dropout probability. :param channel_mult: channel multiplier for each level of the UNet. :param
    conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D. :param num_classes: if specified (as an int), then this
    model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage. :param num_heads: the number of attention
    heads in each attention layer. :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism. :param resblock_updown: use residual blocks
    for up/downsampling. :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size=None,
        in_channels=4,
        out_channels=4,
        num_res_blocks=2,
        dropout=0,
        block_channels=(320, 640, 1280, 1280),
        down_blocks=(
            "UNetResCrossAttnDownBlock2D",
            "UNetResCrossAttnDownBlock2D",
            "UNetResCrossAttnDownBlock2D",
            "UNetResDownBlock2D",
        ),
        downsample_padding=1,
        up_blocks=(
            "UNetResUpBlock2D",
            "UNetResCrossAttnUpBlock2D",
            "UNetResCrossAttnUpBlock2D",
            "UNetResCrossAttnUpBlock2D",
        ),
        resnet_act_fn="silu",
        resnet_eps=1e-5,
        conv_resample=True,
        num_head_channels=8,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
        mid_block_scale_factor=1,
        center_input_sample=False,
        # TODO(PVP) - to delete later at release
        # IMPORTANT: NOT RELEVANT WHEN REVIEWING API
        # ======================================
        # LDM
        attention_resolutions=(4, 2, 1),
        # DDPM
        out_ch=None,
        resolution=None,
        attn_resolutions=None,
        resamp_with_conv=None,
        ch_mult=None,
        ch=None,
        ddpm=False,
        # SDE
        sde=False,
        nf=None,
        fir=None,
        progressive=None,
        progressive_combine=None,
        scale_by_sigma=None,
        skip_rescale=None,
        num_channels=None,
        centered=False,
        conditional=True,
        conv_size=3,
        fir_kernel=(1, 3, 3, 1),
        fourier_scale=16,
        init_scale=0.0,
        progressive_input="input_skip",
        resnet_num_groups=32,
        continuous=True,
        ldm=False,
    ):
        super().__init__()
        # register all __init__ params to be accessible via `self.config.<...>`
        # should probably be automated down the road as this is pure boiler plate code
        self.register_to_config(
            image_size=image_size,
            in_channels=in_channels,
            block_channels=block_channels,
            downsample_padding=downsample_padding,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            down_blocks=down_blocks,
            up_blocks=up_blocks,
            dropout=dropout,
            resnet_eps=resnet_eps,
            conv_resample=conv_resample,
            num_head_channels=num_head_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
            attention_resolutions=attention_resolutions,
            attn_resolutions=attn_resolutions,
            mid_block_scale_factor=mid_block_scale_factor,
            resnet_num_groups=resnet_num_groups,
            center_input_sample=center_input_sample,
        )

        self.ldm = ldm

        # TODO(PVP) - to delete later at release
        # IMPORTANT: NOT RELEVANT WHEN REVIEWING API
        # ======================================
        self.image_size = image_size
        time_embed_dim = block_channels[0] * 4
        # ======================================

        # input
        self.conv_in = nn.Conv2d(in_channels, block_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_steps = Timesteps(block_channels[0], flip_sin_to_cos, downscale_freq_shift)
        timestep_input_dim = block_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.downsample_blocks = nn.ModuleList([])
        self.mid = None
        self.upsample_blocks = nn.ModuleList([])

        # down
        output_channel = block_channels[0]
        for i, down_block_type in enumerate(down_blocks):
            input_channel = output_channel
            output_channel = block_channels[i]
            is_final_block = i == len(block_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=num_res_blocks,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                attn_num_head_channels=num_head_channels,
                downsample_padding=downsample_padding,
            )
            self.downsample_blocks.append(down_block)

        # mid
        self.mid = UNetMidBlock2DCrossAttn(
            in_channels=block_channels[-1],
            dropout=dropout,
            temb_channels=time_embed_dim,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            attn_num_head_channels=num_head_channels,
            resnet_groups=resnet_num_groups,
        )

        # up
        reversed_block_channels = list(reversed(block_channels))
        output_channel = reversed_block_channels[0]
        for i, up_block_type in enumerate(up_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_block_channels[i]
            input_channel = reversed_block_channels[min(i + 1, len(block_channels) - 1)]

            is_final_block = i == len(block_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=num_res_blocks + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                attn_num_head_channels=num_head_channels,
            )
            self.upsample_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = resnet_num_groups if resnet_num_groups is not None else min(block_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_channels[0], num_groups=num_groups_out, eps=resnet_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_channels[0], out_channels, 3, padding=1)

        # ======================== Out ====================

        # =========== TO DELETE AFTER CONVERSION ==========
        # TODO(PVP) - to delete later at release
        # IMPORTANT: NOT RELEVANT WHEN REVIEWING API
        # ======================================
        self.is_overwritten = False
        if ldm:
            num_heads = 8
            num_head_channels = -1
            transformer_depth = 1
            use_spatial_transformer = True
            context_dim = 1280
            legacy = False
            model_channels = block_channels[0]
            channel_mult = tuple([x // model_channels for x in block_channels])
            self.init_for_ldm(
                in_channels,
                model_channels,
                channel_mult,
                num_res_blocks,
                dropout,
                time_embed_dim,
                attention_resolutions,
                num_head_channels,
                num_heads,
                legacy,
                False,
                transformer_depth,
                context_dim,
                conv_resample,
                out_channels,
            )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ) -> Dict[str, torch.FloatTensor]:
        # TODO(PVP) - to delete later at release
        # IMPORTANT: NOT RELEVANT WHEN REVIEWING API
        # ======================================
        if not self.is_overwritten:
            self.set_weights()

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        t_emb = self.time_steps(timesteps)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.downsample_blocks:

            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        skip_sample = None
        for upsample_block in self.upsample_blocks:

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)

        # 6. post-process

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        output = {"sample": sample}

        return output

    # !!!IMPORTANT - ALL OF THE FOLLOWING CODE WILL BE DELETED AT RELEASE TIME AND SHOULD NOT BE TAKEN INTO CONSIDERATION WHEN EVALUATING THE API ###
    # =================================================================================================================================================

    def set_weights(self):
        self.is_overwritten = True
        if self.ldm:
            self.time_embedding.linear_1.weight.data = self.time_embed[0].weight.data
            self.time_embedding.linear_1.bias.data = self.time_embed[0].bias.data
            self.time_embedding.linear_2.weight.data = self.time_embed[2].weight.data
            self.time_embedding.linear_2.bias.data = self.time_embed[2].bias.data

            self.conv_in.weight.data = self.input_blocks[0][0].weight.data
            self.conv_in.bias.data = self.input_blocks[0][0].bias.data

            # ================ SET WEIGHTS OF ALL WEIGHTS ==================
            for i, input_layer in enumerate(self.input_blocks[1:]):
                block_id = i // (self.config.num_res_blocks + 1)
                layer_in_block_id = i % (self.config.num_res_blocks + 1)

                if layer_in_block_id == 2:
                    self.downsample_blocks[block_id].downsamplers[0].conv.weight.data = input_layer[0].op.weight.data
                    self.downsample_blocks[block_id].downsamplers[0].conv.bias.data = input_layer[0].op.bias.data
                elif len(input_layer) > 1:
                    self.downsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])
                    self.downsample_blocks[block_id].attentions[layer_in_block_id].set_weight(input_layer[1])
                else:
                    self.downsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])

            self.mid.resnets[0].set_weight(self.middle_block[0])
            self.mid.resnets[1].set_weight(self.middle_block[2])
            self.mid.attentions[0].set_weight(self.middle_block[1])

            for i, input_layer in enumerate(self.output_blocks):
                block_id = i // (self.config.num_res_blocks + 1)
                layer_in_block_id = i % (self.config.num_res_blocks + 1)

                if len(input_layer) > 2:
                    self.upsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])
                    self.upsample_blocks[block_id].attentions[layer_in_block_id].set_weight(input_layer[1])
                    self.upsample_blocks[block_id].upsamplers[0].conv.weight.data = input_layer[2].conv.weight.data
                    self.upsample_blocks[block_id].upsamplers[0].conv.bias.data = input_layer[2].conv.bias.data
                elif len(input_layer) > 1 and "Upsample2D" in input_layer[1].__class__.__name__:
                    self.upsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])
                    self.upsample_blocks[block_id].upsamplers[0].conv.weight.data = input_layer[1].conv.weight.data
                    self.upsample_blocks[block_id].upsamplers[0].conv.bias.data = input_layer[1].conv.bias.data
                elif len(input_layer) > 1:
                    self.upsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])
                    self.upsample_blocks[block_id].attentions[layer_in_block_id].set_weight(input_layer[1])
                else:
                    self.upsample_blocks[block_id].resnets[layer_in_block_id].set_weight(input_layer[0])

            self.conv_norm_out.weight.data = self.out[0].weight.data
            self.conv_norm_out.bias.data = self.out[0].bias.data
            self.conv_out.weight.data = self.out[2].weight.data
            self.conv_out.bias.data = self.out[2].bias.data

            self.remove_ldm()

    def init_for_ldm(
        self,
        in_channels,
        model_channels,
        channel_mult,
        num_res_blocks,
        dropout,
        time_embed_dim,
        attention_resolutions,
        num_head_channels,
        num_heads,
        legacy,
        use_spatial_transformer,
        transformer_depth,
        context_dim,
        conv_resample,
        out_channels,
    ):
        # TODO(PVP) - delete after weight conversion
        class TimestepEmbedSequential(nn.Sequential):
            """
            A sequential module that passes timestep embeddings to the children that support it as an extra input.
            """

            pass

        # TODO(PVP) - delete after weight conversion
        def conv_nd(dims, *args, **kwargs):
            """
            Create a 1D, 2D, or 3D convolution module.
            """
            if dims == 1:
                return nn.Conv1d(*args, **kwargs)
            elif dims == 2:
                return nn.Conv2d(*args, **kwargs)
            elif dims == 3:
                return nn.Conv3d(*args, **kwargs)
            raise ValueError(f"unsupported dimensions: {dims}")

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        dims = 2
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock2D(
                        in_channels=ch,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        temb_channels=time_embed_dim,
                        eps=1e-5,
                        non_linearity="silu",
                        overwrite_for_ldm=True,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        ),
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample2D(ch, use_conv=conv_resample, out_channels=out_ch, padding=1, name="op")
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = num_head_channels

        if dim_head < 0:
            dim_head = None

        # TODO(Patrick) - delete after weight conversion
        # init to be able to overwrite `self.mid`
        self.middle_block = TimestepEmbedSequential(
            ResnetBlock2D(
                in_channels=ch,
                out_channels=None,
                dropout=dropout,
                temb_channels=time_embed_dim,
                eps=1e-5,
                non_linearity="silu",
                overwrite_for_ldm=True,
            ),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
            ResnetBlock2D(
                in_channels=ch,
                out_channels=None,
                dropout=dropout,
                temb_channels=time_embed_dim,
                eps=1e-5,
                non_linearity="silu",
                overwrite_for_ldm=True,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResnetBlock2D(
                        in_channels=ch + ich,
                        out_channels=model_channels * mult,
                        dropout=dropout,
                        temb_channels=time_embed_dim,
                        eps=1e-5,
                        non_linearity="silu",
                        overwrite_for_ldm=True,
                    ),
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(Upsample2D(ch, use_conv=conv_resample, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(num_channels=model_channels, num_groups=32, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def remove_ldm(self):
        del self.time_embed
        del self.input_blocks
        del self.middle_block
        del self.output_blocks
        del self.out
