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
from utils.infer_utils import temperature_sampling, temperature_sampling_torch, greedy_sampling
from models.template_embedding import TemplateEmbedding
from models.melody_embedding import MelodyEmbedding
import numpy as np
from transformers import BartForConditionalGeneration, BartModel
from torch.nn import CrossEntropyLoss
from positional_encodings.torch_encodings import PositionalEncoding1D
from transformers import BartConfig

# In[8]:


class Bart(BartModel):
    def __init__(self, event2word_dict, word2event_dict, model_pth, hidden_size, num_layers, num_heads, dropout):
        config = BartConfig.from_pretrained("facebook/bart-base")
        super().__init__(config)
        # super(Bart, self).__init__()
        self.event2word_dict = event2word_dict
        self.word2event_dict = word2event_dict
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pos_enc = PositionalEncoding1D(self.hidden_size)

        ## embedding layers
        self.src_emb = TemplateEmbedding(event2word_dict=event2word_dict,  d_embed=hidden_size, drop_prob=self.dropout)
        self.tgt_emb = MelodyEmbedding(event2word_dict=event2word_dict, d_embed=hidden_size, drop_prob=self.dropout)
        
        self.lm = nn.Linear(self.hidden_size, self.tgt_emb.total_size)
        

    def forward(self, enc_inputs, dec_inputs):
        cond_embeds = self.src_emb(**enc_inputs)
        tgt_embeds = self.tgt_emb(**dec_inputs)
        
        outputs = super().forward(inputs_embeds=cond_embeds,
                                  decoder_inputs_embeds=tgt_embeds,)
                                 # labels=dec_inputs['token'])
        
        # Extract the hidden states from the decoder layers
        dec_hidden_states = outputs.last_hidden_state
        # dec_hidden_states = dec_hidden_states.transpose(0,1)
        # print(outputs.decoder_hidden_states)
        dec_outputs = self.lm(dec_hidden_states)
        # model_outputs = self.split_dec_outputs(dec_outputs)
        
        # return self.split_dec_outputs(dec_outputs)
        return dec_outputs
    
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
    
    def infer (self, enc_inputs, dec_inputs_gt, sentence_maxlen, temperature, topk, device, output_dir, midi_name, dec_labels, gt_output_dir, tf_steps=1):
        sampling_func = partial(temperature_sampling_torch, temperature=temperature, topk=topk)
        greed_sampling = partial(greedy_sampling)
        # sampling_func = greed_sampling

        bsz, _ = dec_inputs_gt['token'].shape
        decode_length = sentence_maxlen  # the max number of Tokens in a midi


        tf_steps = dec_inputs_gt['token'].shape[1]  ## number of teacher-forcing steps
        sentence_len = dec_inputs_gt['token'].shape[1]

        dec_inputs = dec_inputs_gt
        
        is_end = False
        xe = []
        
        output_words = []
        if tf_steps == 1:
            output_words.extend(dec_inputs_gt['token'].cpu().squeeze(0).detach().numpy().tolist())
        else:
            output_words.extend(dec_inputs_gt['token'].cpu().squeeze(0).detach().numpy().tolist())
    
        sentence_num = 0
        
        for step in tqdm(range(decode_length)):
            cond_embeds = self.src_emb(**enc_inputs)
            tgt_embeds = self.tgt_emb(**dec_inputs)
            
            outputs = super().forward(inputs_embeds=cond_embeds,
                                      decoder_inputs_embeds=tgt_embeds)
            dec_hidden_states = outputs.last_hidden_state
            dec_outputs = self.lm(dec_hidden_states)
            
            token_out = dec_outputs
            # print(f"token out shape: {token_out.shape}")
            token_logits = token_out[:,-1,:].squeeze() #.cpu().squeeze().detach().numpy()
            # print(f"token logits shape: {token_logits.shape}")
            token_id = int(sampling_func(logits=token_logits).cpu().squeeze().detach().numpy())
            
            output_words.append(token_id)
            
            if token_id == self.event2word_dict['Token'][f"</s>"]:
                break
            
            dec_inputs = {
                'bar': torch.cat((dec_inputs['bar'], torch.LongTensor([[0]]).to(device)), dim=1),
                'pos': torch.cat((dec_inputs['pos'], torch.LongTensor([[0]]).to(device)), dim=1),
                'token': torch.cat((dec_inputs['token'], torch.LongTensor([[token_id]]).to(device)), dim=1),
                'dur': torch.cat((dec_inputs['dur'], torch.LongTensor([[0]]).to(device)), dim=1),
                'phrase': torch.cat((dec_inputs['phrase'], torch.LongTensor([[0]]).to(device)), dim=1),
            }
        
        write(words=output_words, 
              output_dir=output_dir, 
              midi_name=midi_name, 
              word2event=self.word2event_dict)
        
        write(words=list(dec_labels['token'].cpu().squeeze().numpy()), 
              output_dir=gt_output_dir, 
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
    last_time = -1
    
    for idx, event in enumerate(words):
        token_id = event
        token = word2event['Token'][token_id]
        output_tokens.append(token)
        # print(token)
        
        if "Bar" in token:
            bar_cnt += 1
            if "sent" in last:
                markers.append(Marker(time=bar_cnt*480*4, text=token))
            last = token
        if "Pos" in token:
            cur_pos = int(token.split('_')[-1])
            if "sent" in last:
                markers.append(Marker(time=bar_cnt*480*4+cur_pos, text=token))
            last = token
        if "Pitch" in token:
            cur_pitch = int(token.split('_')[-1])
            last = token
        if "Dur" in token and "Pitch" in last:
            cur_dur = int(token.split('_')[-1])
            time = bar_cnt*480*4 + cur_pos
            if time == last_time:
                print(f"| WARNING: overlapping notes at ts {idx}")
            notes_all.append(
                Note(pitch=cur_pitch, start=time, end=time+cur_dur, velocity=80)
            )
            last = token
            last_time = time
        if "sent" in token:
            last = token
    
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
    tgt_remi_pth = os.path.join(output_dir, f"remi_{midi_name.strip()}.txt")
    
    midi_obj.dump(tgt_melody_pth)
    with open(tgt_remi_pth, 'w') as f:
        f.writelines([t.strip()+'\n' for t in output_tokens])
    f.close()

    return output_dir


def write_reorder(words, output_dir, midi_name, word2event):
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
            last = token
            bar_cnt += 1
        if "Pos" in token:
            last = token
            cur_pos = int(token.split('_')[-1])
        if "Dur" in token:
            last = token
            cur_dur = int(token.split('_')[-1])
        if "Pitch" in token:
            last = token
            cur_pitch = int(token.split('_')[-1])
            time = bar_cnt*480*4 + cur_pos
            notes_all.append(
                Note(pitch=cur_pitch, start=time, end=time+cur_dur, velocity=80)
            )
        if "<sent>" in token and "Dur" in last:
            dur = int(last.split('_')[-1])
            markers.append(
                Marker(time=time+dur, text="EOS")
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
    tgt_remi_pth = os.path.join(output_dir, f"remi_{midi_name.strip()}.txt")
    
    midi_obj.dump(tgt_melody_pth)
    with open(tgt_remi_pth, 'w') as f:
        f.writelines([t.strip()+'\n' for t in output_tokens])
    f.close()

    return output_dir




