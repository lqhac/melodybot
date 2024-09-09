#!/usr/bin/env python
# coding: utf-8

# In[7]:

import torch
import os
from torch import nn
from tqdm import tqdm
import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
from torch.nn import Parameter
import math
import torch.onnx.operators
import torch.nn.functional as F
from collections import defaultdict
from functools import partial
from utils.infer_utils import temperature_sampling
from template_embedding import TemplateEmbedding
from melody_embedding import MelodyEmbedding
import numpy as np
from transformers import BartForConditionalGeneration, BartModel
from torch.nn import CrossEntropyLoss
from positional_encodings.torch_encodings import PositionalEncoding1D
from transformers import BartConfig
from modules.mumidi_transformer.mumidi_transformer import *

# In[8]:


class MusicTransformer(nn.Module):
    def __init__(self, event2word_dict, word2event_dict, hidden_size, num_heads,
                 enc_layers, dec_layers, dropout, enc_ffn_kernel_size,
                 dec_ffn_kernel_size):
        super(MusicTransformer, self).__init__()
        self.event2word_dict = event2word_dict
        self.word2event_dict = word2event_dict
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.enc_ffn_kernel_size = enc_ffn_kernel_size
        self.dec_ffn_kernel_size = dec_ffn_kernel_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pos_enc = PositionalEncoding1D(self.hidden_size)

        ## embedding layers
        self.src_emb = TemplateEmbedding(event2word_dict=event2word_dict, d_embed=hidden_size, drop_prob=self.dropout)
        self.tgt_emb = MelodyEmbedding(event2word_dict=event2word_dict, d_embed=hidden_size, drop_prob=self.dropout)
        
        self.encoder = FFTBlocks(self.hidden_size, self.enc_layers, use_pos_embed=True,
                                 ffn_kernel_size=self.enc_ffn_kernel_size,
                                 num_heads=self.num_heads)
        self.decoder = MusicTransformerDecoder(
            self.hidden_size, self.dec_layers, self.dropout, 
            out_dim=self.tgt_emb.total_size,
            use_pos_embed=True,
            num_heads=self.num_heads, dec_ffn_kernel_size=self.dec_ffn_kernel_size)
        
        self.lm = nn.Linear(self.hidden_size, self.tgt_emb.total_size)
        

    def forward(self, enc_inputs, dec_inputs):
        cond_embeds = self.src_emb(**enc_inputs)
        tgt_embeds = self.tgt_emb(**dec_inputs)
        
        enc_outputs = self.encoder(cond_embeds)
        dec_outputs, _ = self.decoder(tgt_embeds, enc_outputs)
        # print(dec_outputs.shape)
        return dec_outputs
        # return self.split_dec_outputs(dec_outputs)
        
    
    def split_dec_outputs(self, dec_outputs):
        bar_out_size = self.tgt_emb.bar_size
        pos_out_size = bar_out_size + self.tgt_emb.pos_size
        token_out_size = pos_out_size + self.tgt_emb.token_size
        dur_out_size = token_out_size + self.tgt_emb.dur_size
        phrase_out_size = dur_out_size + self.tgt_emb.phrase_size
        
        # word_out_size = self.lyr_embed.word_size
        # rem_out_size = word_out_size + self.lyr_embed.rem_size
        bar_out = dec_outputs[:, :, : bar_out_size]
        pos_out = dec_outputs[:, :, bar_out_size: pos_out_size]
        token_out = dec_outputs[:, :, pos_out_size: token_out_size]
        dur_out = dec_outputs[:, :, token_out_size: dur_out_size]
        phrase_out = dec_outputs[:, :, dur_out_size: phrase_out_size]
        
        return bar_out, pos_out, token_out, dur_out, phrase_out
    
    def infer_step (self, enc_inputs, dec_inputs_gt, sentence_maxlen, temperature, topk, device, output_dir, midi_name):
        sampling_func = partial(temperature_sampling, temperature=temperature, topk=topk)

        bsz, _ = dec_inputs_gt['token'].shape
        decode_length = sentence_maxlen  # the max number of Tokens in a midi

        dec_inputs = dec_inputs_gt

        tf_steps = dec_inputs_gt['token'].shape[1]  ## number of teacher-forcing steps
        sentence_len = dec_inputs_gt['token'].shape[1]

        is_end = False
        xe = []
        
        output_words = []
        
        sentence_num = 0
        
        for step in tqdm(range(decode_length)):
            cond_embeds = self.src_emb(**enc_inputs)
            tgt_embeds = self.tgt_emb(**dec_inputs)
            
            enc_outputs = self.encoder(cond_embeds)
            dec_outputs, _ = self.decoder(tgt_embeds, enc_outputs)
            
            bar_out, pos_out, token_out, dur_out, phrase_out = self.split_dec_outputs(dec_outputs)
            
            # print(bar_out.shape)

            bar_logits = bar_out[:,-1,:].cpu().squeeze().detach().numpy()
            pos_logits = pos_out[:,-1,:].cpu().squeeze().detach().numpy()
            token_logits = token_out[:,-1,:].cpu().squeeze().detach().numpy()
            dur_logits = dur_out[:,-1,:].cpu().squeeze().detach().numpy()
            phrase_logits = phrase_out[:,-1,:].cpu().squeeze().detach().numpy()
            
            bar_id = sampling_func(logits=bar_logits)
            pos_id = sampling_func(logits=pos_logits)
            token_id = sampling_func(logits=token_logits)
            dur_id = sampling_func(logits=dur_logits)
            phrase_id = sampling_func(logits=phrase_logits)
            
            
            # print(bar_id, pos_id, token_id, dur_id, phrase_id)
            
            if token_id == self.event2word_dict['Pitch'][f"</s>"]:
                print(f"Generation End")
                break
            
            output_words.append((bar_id, pos_id, token_id, dur_id, phrase_id))
            
            dec_inputs = {
                'bar': torch.cat((dec_inputs['bar'], torch.LongTensor([[bar_id]]).to(device)), dim=1),
                'pos': torch.cat((dec_inputs['pos'], torch.LongTensor([[pos_id]]).to(device)), dim=1),
                'token': torch.cat((dec_inputs['token'], torch.LongTensor([[token_id]]).to(device)), dim=1),
                'dur': torch.cat((dec_inputs['dur'], torch.LongTensor([[dur_id]]).to(device)), dim=1),
                'phrase': torch.cat((dec_inputs['phrase'], torch.LongTensor([[phrase_id]]).to(device)), dim=1),
            }
        
        write(words=output_words, 
              output_dir=output_dir, 
              midi_name=midi_name, 
              word2event=self.word2event_dict)
        
        return output_words
    
    def infer (self, enc_inputs, dec_inputs_gt, sentence_maxlen, temperature, topk, device, output_dir, midi_name, tf_steps=1):
        sampling_func = partial(temperature_sampling, temperature=temperature, topk=topk)

        bsz, _ = dec_inputs_gt['token'].shape
        decode_length = sentence_maxlen  # the max number of Tokens in a midi
        
        dec_inputs = {
            'bar': torch.zeros(bsz, decode_length).fill_(0).long().to(device),
            'pos': torch.zeros(bsz, decode_length).fill_(0).long().to(device),
            'token': torch.zeros(bsz, decode_length).fill_(0).long().to(device),
            'dur': torch.zeros(bsz, decode_length).fill_(0).long().to(device),
            'phrase': torch.zeros(bsz, decode_length).fill_(0).long().to(device),
        }


        tf_steps = dec_inputs_gt['token'].shape[1]  ## number of teacher-forcing steps
        sentence_len = dec_inputs_gt['token'].shape[1]

        is_end = False
        xe = []
        output_words = []
        sentence_num = 0
        max_sentence_step = 0
        
        # init dec_inputs with prompt
        for i in range(bsz):
            for seq_idx in range(tf_steps):
                dec_inputs['bar'][i,seq_idx] = dec_inputs_gt['bar'][i,seq_idx]
                dec_inputs['pos'][i,seq_idx] = dec_inputs_gt['pos'][i,seq_idx]
                dec_inputs['token'][i,seq_idx] = dec_inputs_gt['token'][i,seq_idx]
                dec_inputs['dur'][i,seq_idx] = dec_inputs_gt['dur'][i,seq_idx]
                dec_inputs['phrase'][i,seq_idx] = dec_inputs_gt['phrase'][i,seq_idx]
                
        ## teacher forcing
        for step in tqdm(range(tf_steps-1)):
            cond_embeds = self.src_emb(**enc_inputs)
            tgt_embeds = self.tgt_emb(**{k: v[:, step:step + 1] for k, v in dec_inputs.items()})
            
            enc_outputs = self.encoder(cond_embeds)
            dec_outputs, _ = self.decoder(tgt_embeds, enc_outputs)
            
            # bar_out, pos_out, token_out, dur_out, phrase_out = self.split_dec_outputs(dec_outputs)
            
            bar_id = dec_inputs_gt['bar'][:, step+1]
            pos_id = dec_inputs_gt['pos'][:, step+1]
            token_id = dec_inputs_gt['token'][:, step+1]
            dur_id = dec_inputs_gt['dur'][:, step+1]
            phrase_id = dec_inputs_gt['phrase'][:, step+1]
            
            output_words.append(token_id)
            
            max_sentence_step += 1
            
            
        ## state 2
        for step in tqdm(range(max_sentence_step, decode_length)):
            cond_embeds = self.src_emb(**enc_inputs)
            tgt_embeds = self.tgt_emb(**{k: v[:, step:step + 1] for k, v in dec_inputs.items()})
            
            enc_outputs = self.encoder(cond_embeds)
            dec_outputs, _ = self.decoder(tgt_embeds, enc_outputs)
            
            token_out = dec_outputs

            token_logits = token_out.cpu().squeeze().detach().numpy()
            
            token_id = sampling_func(logits=token_logits)
            
            
            if token_id == self.event2word_dict['Token'][f"</s>"]:
                break
            max_sentence_step += 1
            if max_sentence_step >= decode_length:
                break
            
            output_words.append(token_id)
            
            
            dec_inputs['bar'][0, step + 1] = 0
            dec_inputs['pos'][0, step + 1] = 0
            dec_inputs['token'][0, step + 1] = token_id
            dec_inputs['dur'][0, step + 1] = 0
            dec_inputs['phrase'][0, step + 1] = 0
            
            
        
        write(words=output_words, 
              output_dir=output_dir, 
              midi_name=midi_name, 
              word2event=self.word2event_dict)
        
        return output_words


def write(words, output_dir, midi_name, word2event):
    notes_all = []
    markers = []
    bar_cnt = -1
    midi_obj = miditoolkit.midi.parser.MidiFile()
    event_type_list = []
    notes_all = []
    cur_bar, cur_pos = 0, 0
    last = "BOS"
    output_tokens = []
    
    for event in words:
        token_id = event
        token = word2event['Token'][token_id]
        output_tokens.append(token)
        
        if "Bar" in token:
            last = "Bar"
            bar_cnt += 1
        if "Pos" in token:
            last = "Pos"
            cur_pos = int(token.split('_')[-1])
        if "Pitch" in token:
            last = "Pitch"
            cur_pitch = int(token.split('_')[-1])
        if "Dur" in token and last == "Pitch":
            last = "Dur"
            cur_dur = int(token.split('_')[-1])
            time = bar_cnt*480*4 + cur_pos
            notes_all.append(
                Note(pitch=cur_pitch, start=time, end=time+cur_dur, velocity=80)
            )
        
    
    # tempo
    midi_obj.tempo_changes.append(
                TempoChange(tempo=65, time=0))

    # marker
    midi_obj.markers.extend(markers)

    # track
    piano_track = Instrument(0, is_drum=False, name='melody')

    # notes
    piano_track.notes = notes_all
    midi_obj.instruments = [piano_track]

    # save
    tgt_melody_pth = os.path.join(output_dir, f"{midi_name.strip()}.mid")
    
    midi_obj.dump(tgt_melody_pth)

    return output_dir





