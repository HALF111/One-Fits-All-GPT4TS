import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        # 对patch后的数据再做padding
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                # 加载有预训练参数的GPT2模型
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                # 否则生成一个随机初始化的模型
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            # 只保留我们需要的那些层？
            # 例如论文中为保留3层或6层的GPT模型
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        # 输入输出层
        # 输入做embedding，将各个patch映射到中间维度
        # 输出则是将patch_num个d_model的中间向量战平后、映射到pred_len的预测输出中
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        # 只对layerNorm层和Positional Embedding层做微调，其他层冻结住
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # 将每一层都移到当前的device上，并设置为训练模式
        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
        # 先记录下输入数据的维度为[batch, seq_len, channel]
        B, L, M = x.shape

        # 使用的是普通的Instance Normalization，而非RevIN？
        # 因为没有加可训练参数
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        # 做patch和padding？
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        # 先过输入层
        # 输入层会做embedding，将各个patch映射到中间维度
        outputs = self.in_layer(x)
        # 然后经过GPT2模型
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        # 最后经过输出层
        # 输出层则将patch_num个d_model的中间向量战平后、映射到pred_len的预测输出中
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        # 将之前的参数denorm回来
        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
