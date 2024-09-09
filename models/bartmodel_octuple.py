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
from models.template_embedding_octuple import TemplateEmbedding
from models.melody_embedding_octuple import MelodyEmbedding
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
        
        return self.split_dec_outputs(dec_outputs)
    
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
    
    def infer (self, enc_inputs, dec_inputs_gt, decode_length, sentence_maxlen, temperature, topk, device, output_dir, midi_name):
        sampling_func = partial(temperature_sampling_torch, temperature=temperature, topk=topk)
        print(f"Sampling strategy: temperature-{temperature}, top-{topk}")
        # greed_sampling = partial(greedy_sampling)
        greed_sampling = sampling_func
        # greed_sampling = partial(temperature_sampling_torch, temperature=0.98, topk=1)
        conserv_sampling = partial(temperature_sampling_torch, temperature=0.98, topk=3)
        

        bsz, _ = dec_inputs_gt['token'].shape

        tf_steps = dec_inputs_gt['token'].shape[1]  ## number of teacher-forcing steps
        sentence_len = dec_inputs_gt['token'].shape[1]

        dec_inputs = dec_inputs_gt
        
        is_end = False
        xe = []
        
        output_words = []
        bar_prime = dec_inputs_gt['bar'].cpu().squeeze(0).detach().numpy().tolist()
        pos_prime = dec_inputs_gt['pos'].cpu().squeeze(0).detach().numpy().tolist()
        token_prime = dec_inputs_gt['token'].cpu().squeeze(0).detach().numpy().tolist()
        dur_prime = dec_inputs_gt['dur'].cpu().squeeze(0).detach().numpy().tolist()
        phrase_prime = dec_inputs_gt['phrase'].cpu().squeeze(0).detach().numpy().tolist()
        
        past_bar_pos = {}
        ppls = []
        for i in range(dec_inputs_gt['token'].shape[-1]):
            if bar_prime[i] not in past_bar_pos.keys():
                past_bar_pos[bar_prime[i]] = []
            past_bar_pos[bar_prime[i]].append(pos_prime[i])
            output_words.append((
                bar_prime[i], 
                pos_prime[i], 
                token_prime[i], 
                dur_prime[i], 
                phrase_prime[i]
            ))
        
        sentence_num = 0
        past_times = []
        
        
        for step in tqdm(range(sentence_maxlen)):
            # if len(output_words) == decode_length:
                # break
            
            cond_embeds = self.src_emb(**enc_inputs)
            tgt_embeds = self.tgt_emb(**dec_inputs)
            
            outputs = super().forward(inputs_embeds=cond_embeds,
                                      decoder_inputs_embeds=tgt_embeds)
            dec_hidden_states = outputs.last_hidden_state
            dec_outputs = self.lm(dec_hidden_states)
            
            bar_out, pos_out, token_out, dur_out, phrase_out = self.split_dec_outputs(dec_outputs)

            bar_logits = bar_out[:,-1,:].squeeze() #.cpu().squeeze().detach().numpy()
            pos_logits = pos_out[:,-1,:].squeeze() #.cpu().squeeze().detach().numpy()
            token_logits = token_out[:,-1,:].squeeze() #.cpu().squeeze().detach().numpy()
            dur_logits = dur_out[:,-1,:].squeeze() #.cpu().squeeze().detach().numpy()
            phrase_logits = phrase_out[:,-1,:].squeeze() #.cpu().squeeze().detach().numpy()
            
            bar_id = int(greed_sampling(logits=bar_logits).cpu().squeeze().detach().numpy())       
            if bar_id not in past_bar_pos.keys():
                past_bar_pos[bar_id] = []
            else: ## penalty
                for sampled_pos in past_bar_pos[bar_id]:
                    pos_logits[sampled_pos] -= 10000000  ## with decoding constraints
            pos_id = int(conserv_sampling(logits=pos_logits).cpu().squeeze().detach().numpy())
            past_bar_pos[bar_id].append(pos_id)

            token_id = int(sampling_func(logits=token_logits).cpu().squeeze().detach().numpy())
            dur_id = int(sampling_func(logits=dur_logits).cpu().squeeze().detach().numpy())
            phrase_id = int(greed_sampling(logits=phrase_logits).cpu().squeeze().detach().numpy())


            if bar_id == self.event2word_dict['Bar'][f"</s>"] or \
            pos_id == self.event2word_dict['Pos'][f"<pad>"] or \
            token_id == self.event2word_dict['Pitch'][f"<pad>"] or \
            dur_id == self.event2word_dict['Dur'][f"<pad>"] or \
            phrase_id == self.event2word_dict['Phrase'][f"<pad>"]:
                print(f"Decode ends at step {step}")
                break

            
            bar_token, pos_token = self.word2event_dict['Bar'][bar_id], self.word2event_dict['Pos'][pos_id]
            """
            if "_" not in pos_token:
                src_id = enc_inputs['strength'].cpu().squeeze().numpy()[tgt_embeds.shape[1]]
                if self.word2event_dict['Strength'] == '<strong>':
                    pos_id = self.event2word_dict['Pos']['Pos_0']
                elif self.word2event_dict['Strength'] == '<substrong>':
                    pos_id = self.event2word_dict['Pos']['Pos_960']
                else:
                    pos_id = self.event2word_dict['Pos']['Pos_480']
            """
            bar_num, pos_num = int(self.word2event_dict['Bar'][bar_id].split('_')[-1]), int(self.word2event_dict['Pos'][pos_id].split('_')[-1])
            time = bar_num * 480*4 + pos_num
            # print(self.word2event_dict['Bar'][bar_id], self.word2event_dict['Pos'][pos_id], time)
                
            
            if time not in past_times:
                past_times.append(time)
            else:
                print(f"|>Warning: overlapped at {dec_inputs['token'].shape[-1]+1}. <{bar_token}, {pos_token}>")
            
            # if time not in past_times:
            output_words.append((bar_id, pos_id, token_id, dur_id, phrase_id))
            
            # if len(output_words) == decode_length:
                # break
            
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
    
    


def write(words, output_dir, midi_name, word2event):
    notes_all = []
    markers = []
    bar_cnt = -1
    positions = 0
    midi_obj = miditoolkit.midi.parser.MidiFile()
    event_type_list = []
    notes_all = []
    
    for event in words:
        bar_id, pos_id, pitch_id, dur_id, phrase_id = event[0], event[1], event[2], event[3], event[4]
        
        bar = word2event['Bar'][bar_id]
        pos = word2event['Pos'][pos_id]
        pitch = word2event['Pitch'][pitch_id]
        dur = word2event['Dur'][dur_id]
        phrase = word2event['Phrase'][phrase_id]
        
        # print(f"{bar}, {pos}, {pitch}, {dur}, {phrase}")
        
        if ("Bar_" not in bar) or ("Pos_" not in pos) or ("Pitch_" not in pitch) or ("Dur_" not in dur) or (("<true>" not in phrase) and ("<false>" not in phrase)):
            continue
        bar_num = int(bar.split('_')[1])
        pos_num = int(pos.split('_')[1])
        pitch_num = int(pitch.split('_')[1])
        dur_num = int(dur.split('_')[1])
        phrase_bool = True if phrase == '<true>' else False
        
        start = bar_num * 1920 + pos_num
        
        end = start + dur_num
        notes_all.append(
            Note(pitch=pitch_num, start=start, end=end, velocity=80)
        )
        if phrase_bool:
            markers.append(Marker(time=start, text='Phrase'))
        
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




