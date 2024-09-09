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
        self.strength_size = len(event2word_dict['Strength'])
        self.length_size = len(event2word_dict['Length'])
        self.phrase_size = len(event2word_dict['Phrase'])

        self.total_size = self.strength_size + self.length_size + self.phrase_size
        
        self.p_enc_1d_model = PositionalEncoding1D(d_embed)
                          
        # Embedding init |  
        self.strength_emb = Embedding(self.strength_size, d_embed, padding_idx=1)
        self.length_emb = Embedding(self.length_size, d_embed, padding_idx=1)
        self.phrase_emb = Embedding(self.phrase_size, d_embed, padding_idx=1)
        
        self.emb_proj = nn.Linear(3 * d_embed, d_embed)
        
        self.drop_out = nn.Dropout(p=drop_prob)


    def forward(self, strength, length, phrase):
        strength_embed = self.strength_emb(strength)
        length_embed = self.length_emb(length)
        phrase_embed = self.phrase_emb(phrase)
        
        embeds = [strength_embed, length_embed, phrase_embed]
        embeds = torch.cat(embeds, -1)
        embeds = self.emb_proj(embeds)
        
        pos_enc = self.p_enc_1d_model(embeds)

        return embeds + pos_enc
