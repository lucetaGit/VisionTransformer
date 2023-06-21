# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import collections.abc
import math
import os
import re
import warnings
from collections import OrderedDict
from functools import partial
from itertools import repeat
import torch
import torch.nn.functional as F
from torch import nn
from .pruning import Threshold_Pruning
from .layers_quant import DropPath, HybridEmbed, PatchEmbed, trunc_normal_
from .ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear
from .utils import load_weights_from_npz

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'vit_base_patch16_224', 'vit_large_patch16_224'
]


log_data = [0, 0]
def check_ratio(x, x_min, x_max):
    b = torch.logical_and(x_min<=x, x<=x_max)
    print(f"{b.sum()}/{b.numel()} = {b.sum()/b.numel()*100:.2f}%")
    log_data[0] += (b.sum()/b.numel()).item()
    log_data[1] += 1


PRUNE = False


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 quant=False,
                 calibrate=False,
                 s=0.0,
                 c=0.0,
                 Attn_Th=0.0,
                 cfg=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qinput = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.qkv = QLinear(dim,
                           dim * 3,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.proj = QLinear(dim,
                            dim,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W)
        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.qact_attn1 = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.log_int_softmax = QIntSoftmax(
            log_i_softmax=cfg.INT_SOFTMAX,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_S,
            calibration_mode=cfg.CALIBRATION_MODE_S,
            observer_str=cfg.OBSERVER_S,
            quantizer_str=cfg.QUANTIZER_S)
        self.log_int_softmax_enabled = cfg.INT_SOFTMAX


        self.pruning = Threshold_Pruning(s, c)


        self.s = s
        self.c = c
        self.Attn_Th = Attn_Th


    def forward(self, x):
        B, N, C = x.shape
        # Batch, 197=(224/16)**2+1, 768=16x16x3

        x = self.qinput(x)
        calibration = self.qinput.calibrate
        # print(f"S: {self.s}")
        # print(f"C: {self.c}")
        # print(f"Th: {self.Th}")



        # if PRUNE and not calibration:
            # x_int = self.qinput.quantizer.quant(x)
            # w_int = self.qkv.quantizer.quant(self.qkv.weight)
            # x_scale = self.qinput.quantizer.scale
            # w_scale = self.qkv.quantizer.scale
            # b_scale = 1/(x_scale*w_scale)
            # b_int = (self.qkv.bias*b_scale).round()

            # fn_mask = lambda x: \
                    # torch.where(x.abs()<16, x, (x/2**4).round() * 2**4)
            # x_masked = fn_mask(x_int)
            # w_masked = fn_mask(w_int)
            # qkv_masked = F.linear(x_masked, w_masked, b_int)
            # bound = torch.quantile(qkv_masked.abs(), 0.3)
            # mask = torch.where(qkv_masked.abs()<bound, 0, 1)

        




        # Float friendly version
        # if PRUNE and not calibration:
        #     x_int = self.qinput.quantizer.quant(x)
        #     w_int = self.qkv.quantizer.quant(self.qkv.weight)
        #     #############################
        #     # check_ratio(x_int, -8, 7)
        #     # check_ratio(w_int, -8, 7)
        #     # check_ratio(x_int, -16, 15)
        #     # check_ratio(w_int, -16, 15)
        #     #############################
        #     fn_mask = lambda x: \
        #             torch.where(x.abs()<16, x, (x/2**4).round() * 2**4)
        #     x_masked = fn_mask(x_int)
        #     w_masked = fn_mask(w_int)

        #     x_fp = self.qinput.quantizer.dequantize(x_masked)
        #     w_fp = self.qkv.quantizer.dequantize(w_masked)
        #     qkv_masked = F.linear(x_fp, w_fp, self.qkv.bias)
        #     qkv_masked = qkv_masked.type(torch.float32)
        #     bound = torch.quantile(qkv_masked.abs(), 0.3)
        #     mask = torch.where(qkv_masked.abs()<bound, 0, 1)

        # if not calibration:
            # x_int = self.qinput.quantizer.quant(x)
            # w_int = self.qkv.quantizer.quant(self.qkv.weight)
            # x_scale = self.qinput.quantizer.scale
            # w_scale = self.qkv.quantizer.scale
            # scale = x_scale*w_scale
            # b_scale = 1/scale
            # b_int = (self.qkv.bias*b_scale).round()

            # shift = 1
            # fn_mask = lambda x: \
                    # torch.where(x.abs()<16, x, (x/2**shift).round() * 2**shift)
            # x_masked = fn_mask(x_int)
            # w_masked = fn_mask(w_int)
            # qkv_masked = F.linear(x_masked, w_masked, b_int)
            # x = scale * qkv_masked
        # else:
            # x = self.qkv(x)


        x = self.qkv(x)

        


        ###########################################
        # if PRUNE and not calibration:
            # x_int = self.qact1.quantizer.quant(x)
            # check_ratio(x_int, -16, 15)
        ###########################################
        x = self.qact1(x)
        # if PRUNE and not calibration:
            # x = x*mask



        qkv = x.reshape(B, N, 3, self.num_heads,
                        C // self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)




        attn = (q @ k.transpose(-2, -1)) * self.scale
        # softthreshold1 = attn * torch.tanh(self.s*(attn-self.Th))
        # softthreshold2 = self.c * torch.tanh(self.s*(attn-self.Th))
        
        # attn = torch.where(attn >= self.Th, softthreshold1, softthreshold2)
        # B, H, N, N = attn.shape      
        # sparsity = len(torch.where(attn < self.Th)[0])/(B*H*N*N)

        attn, sparsity = self.pruning(attn, self.Attn_Th)

        print(f"Sparsity:{sparsity*100}")

        # print("real score:", attn[0][0][0][0])
        attn = self.qact_attn1(attn)


        # print(f"Threshold: {self.Th}")

        if self.log_int_softmax_enabled:
            attn = self.log_int_softmax(attn, self.qact_attn1.quantizer.scale)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        ###########################################
        # if PRUNE and not calibration:
            # x_int = self.qact2.quantizer.quant(x)
            # check_ratio(x_int, -16, 15)
        ###########################################
        x = self.qact2(x)
        x = self.proj(x)
        x = self.qact3(x)
        x = self.proj_drop(x)
        return x
    


# # Custom hook function to extract intermediate activations
# def hook_fn(module, input, output):
#     # Store the intermediate activations in a list or process them as needed
#     intermediate_activations.append(output)
    


# def score_extraction():
#     # Create a list to store intermediate activations
#     intermediate_activations = []
#     target = Attention()
#     # Register the hook on the second convolutional layer (conv2)
#     target_layer = target.qact_attn1
#     target_layer.register_forward_hook(hook_fn)

#     return intermediate_activations

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.qinput = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = QLinear(in_features,
                           hidden_features,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        self.act = act_layer()
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = QLinear(hidden_features,
                           out_features,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        calibration = self.qinput.calibrate
        x = self.qinput(x)



        if PRUNE and not calibration:
            x_int = self.qinput.quantizer.quant(x)
            w_int = self.fc1.quantizer.quant(self.fc1.weight)
            #############################
            # check_ratio(x_int, -8, 7)
            # check_ratio(w_int, -8, 7)
            # check_ratio(x_int, -16, 15)
            # check_ratio(w_int, -16, 15)
            #############################
            fn_mask = lambda x: \
                    torch.where(x.abs()<16, x, (x/2**4).round() * 2**4)
            x_masked = fn_mask(x_int)
            w_masked = fn_mask(w_int)

            x_fp = self.qinput.quantizer.dequantize(x_masked)
            w_fp = self.fc1.quantizer.dequantize(w_masked)
            qkv_masked = F.linear(x_fp, w_fp, self.fc1.bias)
            qkv_masked = qkv_masked.type(torch.float32)
            bound = torch.quantile(qkv_masked.abs(), 0.3)
            mask = torch.where(qkv_masked.abs()<bound, 0, 1)

        x = self.fc1(x)
        # if PRUNE and not calibration:
            # x = x*mask

        x = self.act(x)
        x = self.qact1(x)
        x = self.drop(x)

        if PRUNE and not calibration:
            x_int = self.qinput.quantizer.quant(x)
            w_int = self.fc2.quantizer.quant(self.fc2.weight)
            #############################
            # check_ratio(x_int, -8, 7)
            # check_ratio(w_int, -8, 7)
            # check_ratio(x_int, -16, 15)
            # check_ratio(w_int, -16, 15)
            #############################
            fn_mask = lambda x: \
                    torch.where(x.abs()<16, x, (x/2**4).round() * 2**4)
            x_masked = fn_mask(x_int)
            w_masked = fn_mask(w_int)

            x_fp = self.qinput.quantizer.dequantize(x_masked)
            w_fp = self.fc1.quantizer.dequantize(w_masked)
            qkv_masked = F.linear(x_fp, w_fp, self.fc2.bias)
            qkv_masked = qkv_masked.type(torch.float32)
            bound = torch.quantile(qkv_masked.abs(), 0.3)
            mask = torch.where(qkv_masked.abs()<bound, 0, 1)

        x = self.fc2(x)
        # if PRUNE and not calibration:
            # x = x*mask

        x = self.qact2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 quant=False,
                 calibrate=False,
                s   = 10,
                c   = 1000,
                Attn_Th  = 0.0,
                 cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              s   = s,
                              c   = c,
                              Attn_Th  = Attn_Th,
                              cfg=cfg)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)
        self.norm2 = norm_layer(dim)
        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       quant=quant,
                       calibrate=calibrate,
                       cfg=cfg)
        self.qact4 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)


    def forward(self, x, last_quantizer=None):
        # print(f"S: {self.s}")
        # print(f"C: {self.c}")
        # print(f"Th: {self.Th}")
        
            
            
        x = self.attn(
                self.qact1(self.norm1(x, last_quantizer,
                                      self.qact1.quantizer)))



        x = self.qact2(x + self.drop_path(x))

        x = self.qact4(x + self.drop_path(
            self.mlp(
                self.qact3(
                    self.norm2(x, self.qact2.quantizer,
                               self.qact3.quantizer)))))
        
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 hybrid_backbone=None,
                 norm_layer=None,
                 quant=False,
                 calibrate=False,
                 input_quant=False,
                 s = 0,
                 c = 0,
                 cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.cfg = cfg
        self.input_quant = input_quant
        if input_quant:
            self.qact_input = QAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=in_chans,
                                          embed_dim=embed_dim,
                                          quant=quant,
                                          calibrate=calibrate,
                                          cfg=cfg)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.qact_embed = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.qact_pos = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        

        Attn_Th_Arr = nn.Parameter(torch.zeros(depth))

        self.Attn_Th_Arr = Attn_Th_Arr

        self.blocks= nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  quant=quant,
                  calibrate=calibrate,
                  s   = s,
                  c   = c,
                  Attn_Th = Attn_Th_Arr[i],
                  cfg=cfg) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([
                    ('fc', nn.Linear(embed_dim, representation_size)),
                    ('act', nn.Tanh()),
                ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (QLinear(self.num_features,
                             num_classes,
                             quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_W,
                             calibration_mode=cfg.CALIBRATION_MODE_W,
                             observer_str=cfg.OBSERVER_W,
                             quantizer_str=cfg.QUANTIZER_W)
                     if num_classes > 0 else nn.Identity())
        self.act_out = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.embed_dim, num_classes)
                     if num_classes > 0 else nn.Identity())

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

    def forward_features(self, x):
        B = x.shape[0]

        if self.input_quant:
            x = self.qact_input(x)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.qact_embed(x)
        x = x + self.qact_pos(self.pos_embed)
        x = self.qact1(x)

        x = self.pos_drop(x) 

        for i, blk in enumerate(self.blocks):
            last_quantizer = self.qact1.quantizer if i == 0 else self.blocks[
                i - 1].qact4.quantizer
            x = blk(x, last_quantizer)

        x = self.norm(x, self.blocks[-1].qact4.quantizer,
                      self.qact2.quantizer)[:, 0]
        x = self.qact2(x)
        x = self.pre_logits(x)


        return x

    def forward(self, x):
        # DJS
        print(f"Attn Th Arr {self.Attn_Th_Arr}")
        x = self.forward_features(x)
        x = self.head(x)
        x = self.act_out(x)


        return x



def deit_tiny_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(QIntLayerNorm, eps=1e-6),
        quant=quant,
        calibrate=calibrate,
        input_quant=True,
        cfg=cfg,
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
            map_location='cpu',
            check_hash=True,
        )
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def deit_small_patch16_224(pretrained=False,
                           quant=False,
                           calibrate=False,
                           cfg=None,
                           **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=384,
                              depth=12,
                              num_heads=6,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=True,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def deit_base_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=True,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def vit_base_patch16_224(
                            pretrained=False,
                         quant=False,
                         calibrate=False,
                        s = 10,
                        c = 1000,
                         cfg=None,
                         **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=False,
                              s = 10,
                              c = 1000,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        url = 'https://storage.googleapis.com/vit_models/augreg/' + \
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        load_weights_from_npz(model, url, check_hash=True)
    return model


def vit_large_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=False,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        url = 'https://storage.googleapis.com/vit_models/augreg/' + \
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        load_weights_from_npz(model, url, check_hash=True)
    return model
