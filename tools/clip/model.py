from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from functools import reduce
from operator import mul

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlock_Deep(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, i: int = 0,):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        self.prompt_dropout = nn.Dropout(0.1)
        self.prompt_proj = nn.Linear(3, 768)
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False
        self.n_layer = i
        
        self.back_add = True

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs: torch.Tensor, combine_prompts: bool=False, use_prompt: bool=True, outer_prompt: torch.Tensor=None):
        if combine_prompts:
            if use_prompt:
                if outer_prompt == None:
                    x = inputs[0]
                    compound_prompts_deeper = inputs[1]
                    
                    # # B = x.shape[0]
                    # # if not self.first_layer:
                    # #     deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                    # #                 compound_prompts_deeper[self.n_layer]-1).expand(B, -1, -1))
                    # if not self.first_layer and self.n_layer < compound_prompts_deeper.shape[0]:
                    #     visual_context = compound_prompts_deeper[self.n_layer] - 1
                    #     visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                    #     # x = torch.cat((
                    #     #         x[:1, :, :],
                    #     #         visual_context.to(x.dtype),
                    #     #         x[(1+visual_context.shape[0]):, :, :]
                    #     #     ), dim=0)
                    #     x = torch.concat((
                    #         x[:(x.shape[0] - visual_context.shape[0]), :, :],
                    #         visual_context.to(x.dtype),
                    #         ), dim=0)            
                   
                else: 
                    x = inputs[0]
                    outer_prompt = outer_prompt.type(x.dtype)
                    # if len(outer_prompt.shape) == 2:
                    #     x = torch.concat((x, outer_prompt.unsqueeze(0)), dim=1)
                    # elif len(outer_prompt.shape) == 3:
                    #     x = torch.concat((x, outer_prompt), dim=1)
                    compound_prompts_deeper = outer_prompt
            else:
                    x = inputs[0]
                    compound_prompts_deeper = torch.zeros(3-1, 100, x.shape[2]).cuda().type(x.dtype)
            
            if self.n_layer < compound_prompts_deeper.shape[0]:
            
            # if not self.first_layer and self.n_layer < compound_prompts_deeper.shape[0]:
                visual_context = compound_prompts_deeper[self.n_layer]
                visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                if self.back_add:
                    if self.first_layer:
                        x = torch.cat((
                            x,
                            visual_context.to(x.dtype),
                        ), dim=0)
                    else:
                        x = torch.cat((
                            x[:(x.shape[0] - visual_context.shape[0]), :, :],
                            visual_context.to(x.dtype),
                        ), dim=0)
                    
                else:
                    if self.first_layer:
                        x = torch.cat((
                            x[:1, :, :],
                            visual_context.to(x.dtype),
                            x[1:, :, :]
                        ), dim=0)
                    else:
                        x = torch.cat((
                            x[:1, :, :],
                            visual_context.to(x.dtype),
                            x[(1+visual_context.shape[0]):, :, :]
                        ), dim=0)
                # x = torch.cat((
                #         x[:1, :, :],
                #         visual_context.to(x.dtype),
                #         x[(1+visual_context.shape[0]):, :, :]
                #     ), dim=0)
                # x = torch.concat((
                #     x[:(x.shape[0] - visual_context.shape[0]), :, :],
                #     visual_context.to(x.dtype),
                #     ), dim=0)
                # x = torch.concat((
                #     x,
                #     visual_context.to(x.dtype),
                #     ), dim=0)
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return [x, compound_prompts_deeper]
                    
        else: 
            x = inputs
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
            
        # return [x, compound_prompts_deeper]

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_Deep(width, heads, attn_mask, i) for i in range(layers)])

    def forward(self, x: torch.Tensor, combine_prompts, use_prompt: bool=True, outer_prompt: torch.Tensor=None):
        for block in self.resblocks:
            x = block(x, combine_prompts, use_prompt, outer_prompt)
        # return self.resblocks(x, combine_prompts)
        return x

class Combined_VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, prompt_num: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        # self.prompt_num = prompt_num
        self.prompt_num = 100
        # val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + width))  # noqa
        # if self.config['MODE']['shallow']:
        #     self.vision_prompt = nn.Parameter(torch.zeros(
        #             1, self.prompt_num, width))
        #         # xavier_uniform initialization
        #     nn.init.uniform_(self.vision_prompt.data, -1, 1)
        # else:
            # if self.prompt_config.DEEP:  # noqa
        total_d_layer = 3
        self.vision_prompt = nn.Parameter(torch.zeros(
            total_d_layer, self.prompt_num, width))
        # xavier_uniform initialization
        self.val = math.sqrt(6. / float(3 * reduce(mul, [patch_size,patch_size], 1) + width))
        # nn.init.uniform_(self.vision_prompt.data, -self.val, self.val)
            
    def init_prompt(self):
        self.vision_prompt.data.zero_()
        # nn.init.uniform_(self.vision_prompt.data, -self.val, self.val)        
        # self.deep_vision_prompt.data.zero_()
    
    def forward(self, x: torch.Tensor, use_prompt, outer_prompt):
        self.use_prompt = use_prompt
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        combine_prompts = True
        # if self.config['MODE']['shallow']:
        #     combine_prompts = False
        #     # ##
        #     if use_prompt:
        #         if outer_prompt == None:
        #             # print ('use itself prompt')
        #             vision_prompt = self.vision_prompt.expand(x.shape[0], self.vision_prompt.shape[1], self.vision_prompt.shape[2]).type(x.dtype)
        #             x = torch.concat((x, vision_prompt), dim=1)
        #         else:
        #             # print ('use outer prompt', outer_prompt)
        #             # vision_prompt = outer_prompt.expand(x.shape[0], outer_prompt.shape[1], outer_prompt.shape[2]).type(x.dtype)
        #             outer_prompt = outer_prompt.type(x.dtype)
        #             # x = torch.concat((x, outer_prompt[0].unsqueeze(0)), dim=1)
        #             # if len(outer_prompt.shape) == 2:
        #             #     x = torch.concat((x, outer_prompt.unsqueeze(0)), dim=1)
        #             # elif len(outer_prompt.shape) == 3:
        #             x = torch.concat((x, outer_prompt), dim=1)
        #     else:
        #         # print ('not use prompt')
        #         # 不更新参数
        #         vision_prompt=torch.zeros(x.shape[0],self.vision_prompt.shape[1],self.vision_prompt.shape[2]).cuda().type(x.dtype)
        #         # print('x.shape', x.shape)
        #         x = torch.concat((x, vision_prompt), dim=1)
        #         # print('x.shape', x.shape)
        #     # if self.prompt_num != 0 :
        #     #     vision_prompt = self.vision_prompt.expand(x.shape[0], self.vision_prompt.shape[0], self.vision_prompt.shape[1]).type(x.dtype)
        #     #     x = torch.concat((x, vision_prompt), dim=1)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        # self.deep_vision_prompt = self.deep_vision_prompt.permute(1, 0, 2)
        if combine_prompts:
            # outputs = self.transformer([x, self.vision_prompt], combine_prompts=combine_prompts, use_prompt=True, outer_prompt=outer_prompt)
            outputs = self.transformer([x, self.vision_prompt], combine_prompts=combine_prompts, use_prompt=use_prompt, outer_prompt=outer_prompt)
            x = outputs[0]
        else:
            x = self.transformer(x, combine_prompts=combine_prompts, use_prompt=True, outer_prompt=outer_prompt)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # if self.prompt_num !=0 :
        #     vision_prompt = self.vision_prompt.expand(x.shape[0], self.vision_prompt.shape[0], self.vision_prompt.shape[1]).type(x.dtype)
        #     x = torch.concat((x, vision_prompt), dim=1)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
    
    
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, prompt_num: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.prompt_num = prompt_num
        self.vision_prompt = nn.Parameter(torch.zeros(self.prompt_num, width))
        nn.init.uniform_(self.vision_prompt.data, -1, 1)

    
    def init_prompt(self):
        self.vision_prompt.data.zero_()
        
    def forward(self, x: torch.Tensor, use_prompt, outer_prompt):
        self.use_prompt = use_prompt
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        # x = self.ln_pre(x)
        
        # ##
        if use_prompt:
            if outer_prompt == None:
                # print ('use itself prompt')
                vision_prompt = self.vision_prompt.expand(x.shape[0], self.vision_prompt.shape[0], self.vision_prompt.shape[1]).type(x.dtype)
                x = torch.concat((x, vision_prompt), dim=1)
            else:
                # print ('use outer prompt', outer_prompt)
                # vision_prompt = outer_prompt.expand(x.shape[0], outer_prompt.shape[1], outer_prompt.shape[2]).type(x.dtype)
                outer_prompt = outer_prompt.type(x.dtype)
                # x = torch.concat((x, outer_prompt[0].unsqueeze(0)), dim=1)
                if len(outer_prompt.shape) == 2:
                    x = torch.concat((x, outer_prompt.unsqueeze(0)), dim=1)
                elif len(outer_prompt.shape) == 3:
                    x = torch.concat((x, outer_prompt), dim=1)
        else:
            # print ('not use prompt')
            # 不更新参数
            vision_prompt=torch.zeros(x.shape[0],self.vision_prompt.shape[0],self.vision_prompt.shape[1]).cuda().type(x.dtype)
            # print('x.shape', x.shape)
            x = torch.concat((x, vision_prompt), dim=1)
            # print('x.shape', x.shape)
        # if self.prompt_num != 0 :
        #     vision_prompt = self.vision_prompt.expand(x.shape[0], self.vision_prompt.shape[0], self.vision_prompt.shape[1]).type(x.dtype)
        #     x = torch.concat((x, vision_prompt), dim=1)
        
        x = self.ln_pre(x,)    
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # if self.prompt_num !=0 :
        #     vision_prompt = self.vision_prompt.expand(x.shape[0], self.vision_prompt.shape[0], self.vision_prompt.shape[1]).type(x.dtype)
        #     x = torch.concat((x, vision_prompt), dim=1)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 prompt_num: int
                 ):
        super().__init__()

        self.context_length = context_length
        self.prompt_num = prompt_num
        

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = Combined_VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                prompt_num=prompt_num,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        
        # self.prompt_num = prompt_num
        self.text_prompt = nn.Parameter(torch.zeros(self.prompt_num, transformer_width))
        # self.text_prompt_num = prompt_num
        # self.vision_prompt = nn.Parameter(torch.zeros(self.prompt_num, transformer_width))
        
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
                        

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        # ##
        # mask = torch.empty(self.context_length + self.prompt_num, self.context_length + self.prompt_num)
        mask = torch.empty(self.context_length , self.context_length )
        
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, use_prompt, outer_prompt=None):
        return self.visual(image.type(self.dtype), use_prompt, outer_prompt)
    
    def init_prompt(self):
        self.text_prompt.data.zero_()

    def encode_text(self, text, use_prompt, outer_prompt=None):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        x = x + self.positional_embedding.type(self.dtype)
        # ##
        # if use_prompt:
        #     if outer_prompt == None:
        #         # print ('use itself prompt')
        #         text_prompt = self.text_prompt.expand(x.shape[0], self.text_prompt.shape[0], self.text_prompt.shape[1]).type(x.dtype)
        #         x = torch.concat((x, text_prompt), dim=1)
        #     else:
        #         # print ('use outer prompt', outer_prompt)
        #         outer_prompt = outer_prompt.type(x.dtype)
        #         x = torch.concat((x, outer_prompt.unsqueeze(0)), dim=1)
        # else:
        #     # print ('not use prompt')
        #     # 不更新参数
        #     text_prompt=torch.zeros(x.shape[0],self.text_prompt.shape[0],self.text_prompt.shape[1]).cuda().type(x.dtype)
        #     x = torch.concat((x, text_prompt), dim=1)
            
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, combine_prompts=False,use_prompt=True, outer_prompt=outer_prompt)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, use_prompt):
        # self.use_prompt = use_prompt
        image_features = self.encode_image(image, use_prompt)
        text_features = self.encode_text(text, use_prompt)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    # ##
    prompt_num = state_dict['prompt_num']
    # use_prompt = state_dict['use_prompt']
    # outer_prompt = state_dict['outer_prompt']
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, prompt_num
        
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    # ##
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
