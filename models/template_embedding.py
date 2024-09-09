import torch
import torch.nn as nn
from keys import *
from positional_encodings.torch_encodings import PositionalEncoding1D

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0,
                    std=embedding_dim ** -0.5)  
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class TemplateEmbedding(nn.Module):
    def __init__(self, event2word_dict, d_embed, drop_prob=0.1):
        super().__init__()
        
        ## dict sizes
        self.prosody_size = len(event2word_dict['Prosody'])

        self.total_size = self.prosody_size
        
        self.p_enc_1d_model = PositionalEncoding1D(d_embed)
                          
        # Embedding init |  
        self.prosody_emb = Embedding(self.prosody_size, d_embed, padding_idx=0)
        
        self.drop_out = nn.Dropout(p=drop_prob)


    def forward(self, strength, length, phrase):
        prosody_embed = self.prosody_emb(strength)
        
        embeds = prosody_embed
        
        pos_enc = self.p_enc_1d_model(embeds)

        return embeds + pos_enc
