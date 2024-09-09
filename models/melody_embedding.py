import os
import json
import yaml
import pickle
import datetime
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from utils.hparams import hparams, set_hparams
from positional_encodings.torch_encodings import PositionalEncoding1D
from keys import *
    
def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)  # param1:词嵌入字典大小； param2：每个词嵌入单词的大小
    nn.init.normal_(m.weight, mean=0,
                    std=embedding_dim ** -0.5)  # 正态分布初始化；e.g.,torch.nn.init.normal_(tensor, mean=0, std=1) 使值服从正态分布N(mean, std)，默认值为0，1
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class MelodyEmbedding(nn.Module):
    def __init__(self, event2word_dict, d_embed, drop_prob=0.1):  # d_embedding = dimension of embedding
        super().__init__()
        
        # self.use_pos = use_pos
        self.token_size = len(event2word_dict['Token'])

        self.total_size = self.token_size
        
        self.p_enc_1d_model = PositionalEncoding1D(d_embed)
                          
        # Embedding init |  
        self.token_emb = Embedding(self.total_size, d_embed, padding_idx=0)
        
        self.drop_out = nn.Dropout(p=drop_prob)


    def forward(self, bar, pos, token, dur, phrase):
        token_embed = self.token_emb(token)
        
        embeds = token_embed
        
        pos_enc = self.p_enc_1d_model(embeds)

        return embeds + pos_enc