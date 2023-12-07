import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from math import sqrt, log
import matplotlib.pyplot as plt

from embed import DataEmbedding_wo_time

import random

def l2norm(t):
    return F.normalize(t, dim = -1)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, attn_bias):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # 将QKV映射到对应的d_q、d_k、d_v维度上，并映射h次
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 计算得到注意力结果
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            attn_bias
        )
        out = out.view(B, L, -1)

        # 将多个头的输出concat后，映射回d_model维度
        return self.out_projection(out), attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, configs=None,
                 attn_scale_init=20):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.enc_in = configs.enc_in
       
        self.scale = scale

    def forward(self, queries, keys, values, attn_mask, attn_bias):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        # Q和K之间点积计算得到注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # 做mask的注意力
            scores.masked_fill_(attn_mask.mask, -np.inf)

        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 3, 1, 2)
            A = self.dropout(torch.softmax(scores * scale + attn_bias, dim=-1))
        else:
            # scale后再做softmax，得到注意力分数
            A = self.dropout(torch.softmax(scores * scale, dim=-1))
        # 分数和values加权求和，得到输出的V矩阵
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, attn_bias=None):
        # 自注意力
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            attn_bias=attn_bias
        )
        # 先dropout，再add（残差连接），再norm（这里是batchnorm？？？）
        x = x + self.dropout(new_x)
        y = x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)  # permute也是因为batchnorm而非layernorm
        # 过MLP层，一维卷积类似于MLP？
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # MLP也有add（残差连接）和norm流程
        y = x + y
        y = self.norm2(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, attn_bias=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                # 如果有conv，那么一层注意力、一层卷积交替进行
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                # 没有conv的话，就只需要做注意力（里面包含了FFN）即可了。
                x, attn = attn_layer(x, attn_mask=attn_mask, attn_bias=attn_bias)
                attns.append(attn)

        # 有norm的话输出前再做一次norm
        # 这里由于batchnorm代替了layernorm，所以要做一次转置？
        if self.norm is not None:
            # x = self.norm(x)
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x, attns

class PatchTST(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs, device):
        super(PatchTST, self).__init__()

        self.enc_in = configs.enc_in
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.label_len = configs.label_len
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.num_heads = configs.n_heads
        self.factor = 3
        self.activation = 'gelu'
        
        # Embedding
        self.enc_embedding = DataEmbedding_wo_time(self.patch_size, 
                                            configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention,
                                      configs=configs),
                                      configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=self.activation
                ) for l in range(configs.e_layers)
            ],
            # norm_layer=torch.nn.LayerNorm(configs.d_model)
            norm_layer=torch.nn.BatchNorm1d(configs.d_model)
        )
        
        self.proj = nn.Linear(configs.d_model * self.patch_num, configs.pred_len, bias=True)
        self.cnt = 0
    
    def forward(self, x_enc, itr):
        B, L, M = x_enc.shape

        # 使用的是普通的Instance Normalization，而非RevIN？
        # 因为没有加可训练参数
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x_enc /= stdev

        x_enc = rearrange(x_enc, 'b l m -> b m l')
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_enc = rearrange(x_enc, 'b m n p -> (b m) n p')

        # 先做embedding
        enc_out = self.enc_embedding(x_enc)

        # 再过encoder层
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # 最后过一个线性的映射层
        enc_out = self.proj(enc_out.reshape(B*M, -1))
        enc_out = rearrange(enc_out, '(b m) l -> b l m', m=M)
        
        # revin
        # 再将之前norm后的数据denorm回来
        enc_out = enc_out[:, -self.pred_len:, :]
        enc_out = enc_out * stdev
        enc_out = enc_out + means

        x_enc = enc_out * stdev
        x_enc = enc_out + means


        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out  # [B, L, D]

