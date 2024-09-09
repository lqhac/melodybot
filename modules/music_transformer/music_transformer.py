from functools import partial

import torch
from torch import nn
from tqdm import tqdm

from modules.asr.seq2seq import TransformerASRDecoder
from modules.commons.common_layers import Embedding
from modules.fastspeech.tts_modules import FFTBlocks
from utils_structMIDI.mumidi.hparams import hparams
import torch.nn.functional as F

# from ut .music_utils.infer_utils import temperature_sampling_torch


class MuMIDIEmbedding(nn.Module):
    def __init__(self, n_token, d_embed):
        super().__init__()
        self.word_emb = Embedding(n_token, d_embed, padding_idx=0)
        self.vel_emb = Embedding(32, d_embed, padding_idx=0)
        self.dur_emb = Embedding(64, d_embed, padding_idx=0)
        self.bar_embed = Embedding(hparams['bar_voc_size'], d_embed)
        self.pos_embed = Embedding(hparams['pos_voc_size'], d_embed)
        dims_in = 5 * d_embed
        self.token_embed_proj = nn.Linear(dims_in, d_embed)

        self.tempo_embed = Embedding(3, d_embed)
        self.total_tracks = len(hparams['instru2track'].keys())
        dims_in += d_embed
        if hparams['use_style']:
            self.style_embed = Embedding(20, d_embed)
            dims_in += d_embed
        if hparams['use_track_mask']:
            track_embed_size = 32
            self.track_embeds = nn.ModuleList([Embedding(2, track_embed_size) for _ in range(self.total_tracks)])
            dims_in += self.total_tracks * track_embed_size
        self.token_embed_proj_cond = nn.Linear(dims_in, d_embed)

    def forward(self, token, vel, dur, bar, pos, tempo=None, style=None, track_mask=None, cond=False):
        embeds = [self.word_emb(token), self.vel_emb(vel), self.dur_emb(dur),
                  self.bar_embed(bar - bar // hparams['bar_voc_size'] * hparams['bar_voc_size']),
                  self.pos_embed(pos)]
        if cond:
            embeds.append(self.tempo_embed(tempo))
            if hparams['use_style']:
                embeds.append(self.style_embed(style))
            if hparams['use_track_mask']:
                embeds += [self.track_embeds[t](track_mask[:, :, t]) for t in range(self.total_tracks)]
            embeds = torch.cat(embeds, -1)
            embeds = self.token_embed_proj_cond(embeds)

        else:
            embeds = torch.cat(embeds, -1)
            embeds = self.token_embed_proj(embeds)
        return embeds


class MusicTransformerDecoder(TransformerASRDecoder):
    pass


class MusicTransformer(nn.Module):
    def __init__(self, id2token, token2id):
        super().__init__()
        self.hidden_size = hparams['hidden_size']
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.dropout = hparams['dropout']
        self.id2token = id2token
        self.token2id = token2id
        self.num_heads = hparams['num_heads']
        self.dict_size = dict_size = len(self.id2token)
        self.mumidi_embed = MuMIDIEmbedding(dict_size, self.hidden_size)
        self.encoder = FFTBlocks(self.hidden_size, self.enc_layers, use_pos_embed=False,
                                 ffn_kernel_size=hparams['enc_ffn_kernel_size'],
                                 num_heads=self.num_heads)
        self.decoder = MusicTransformerDecoder(
            self.hidden_size, self.dec_layers, self.dropout, dict_size + 32 + 64, use_pos_embed=False,
            num_heads=self.num_heads)

    def forward(self, enc_inputs, dec_inputs):
        cond_embeds = self.mumidi_embed(**enc_inputs, cond=True)  # [B, T_cond, H]
        enc_outputs = self.encoder(cond_embeds)
        tgt_embeds = self.mumidi_embed(**dec_inputs)  # [B, T_tgt, H]
        dec_outputs, _ = self.decoder(tgt_embeds, enc_outputs)
        token_out, vel_out, dur_out = self.split_dec_outputs(dec_outputs)
        return token_out, vel_out, dur_out

    def split_dec_outputs(self, dec_outputs):
        return dec_outputs[:, :, :self.dict_size], \
               dec_outputs[:, :, self.dict_size:self.dict_size + 32], \
               dec_outputs[:, :, self.dict_size + 32: self.dict_size + 32 + 64]

    def infer(self, enc_inputs, n_bars, dec_inputs_gt=None):
        cond_embeds = self.mumidi_embed(**enc_inputs, cond=True)  # [B, T_cond, H]
        enc_outputs = self.encoder(cond_embeds)
        bsz, T_cond, H = cond_embeds.shape
        decode_length = hparams['sentence_maxlen']
        xe_loss = 0
        xe_loss_cnt = 0
        start_token = self.token2id['Bar_0']
        dec_inputs = {
            'token': enc_outputs.new(bsz, decode_length).fill_(start_token).long(),
            'vel': enc_outputs.new(bsz, decode_length).fill_(0).long(),
            'dur': enc_outputs.new(bsz, decode_length).fill_(0).long(),
            'bar': enc_outputs.new(bsz, decode_length).fill_(0).long(),
            'pos': enc_outputs.new(bsz, decode_length).fill_(0).long(),
        }
        dec_output_bars = enc_outputs.new(bsz).fill_(1).long()
        not_end_flag = enc_outputs.new(bsz).fill_(1).bool()
        incremental_state = {}
        step = 0
        sampling_func = partial(
            temperature_sampling_torch, temperature=hparams['temperature'], topk=hparams['topk'])

        for step in tqdm(range(decode_length - 1)):
            tgt_embeds = self.mumidi_embed(**{k: v[:, step:step + 1] for k, v in dec_inputs.items()})
            dec_outputs, attn_logits = self.decoder(tgt_embeds, enc_outputs, incremental_state=incremental_state)
            token_out, vel_out, dur_out = self.split_dec_outputs(dec_outputs)
            if dec_inputs_gt is None:
                for b in range(bsz):
                    dec_inputs['token'][b, step + 1] = sampling_func(logits=token_out[b, -1])
                    dec_inputs['vel'][b, step + 1] = sampling_func(logits=vel_out[b, -1]) \
                        if hparams['use_velocity'] else 31
                    dec_inputs['dur'][b, step + 1] = sampling_func(logits=dur_out[b, -1])
            else:
                if step + 1 == dec_inputs_gt['token'].shape[1]:
                    break
                dec_inputs['token'][:, step + 1] = dec_inputs_gt['token'][:, step + 1]
                dec_inputs['vel'][:, step + 1] = dec_inputs_gt['vel'][:, step + 1]
                dec_inputs['dur'][:, step + 1] = dec_inputs_gt['dur'][:, step + 1]
                xe_loss += F.cross_entropy(token_out[:, -1], dec_inputs_gt['token'][:, step + 1],
                                           ignore_index=0, reduction='sum').item()
                xe_loss_cnt += bsz
            dec_inputs['bar'][:, step + 1] = dec_inputs['bar'][:, step]
            dec_inputs['pos'][:, step + 1] = dec_inputs['pos'][:, step]
            token_next_s = [self.id2token[x.item()] for x in dec_inputs['token'][:, step + 1]]
            for k in dec_inputs.keys():
                dec_inputs[k][:, step + 1] = dec_inputs[k][:, step + 1] * not_end_flag.long()

            for b in range(bsz):
                bar = dec_inputs['bar'][b, step + 1].item()
                t = token_next_s[b]
                if t == 'Bar_0' or t == '<s>':
                    if bar == hparams['n_target_bar'] - 1 or t == '<s>' or bar == n_bars - 1:
                        not_end_flag[b] = 0
                    else:
                        dec_output_bars[b] = dec_inputs['bar'][b, step + 1] = bar + 1
                        dec_inputs['pos'][b, step + 1] = 0
                elif 'Position' in t:
                    value = int(t.split("_")[1])
                    dec_inputs['pos'][b, step + 1] = value
                if t.split("_")[0] not in ['On', 'Drums']:
                    dec_inputs['vel'][b, step + 1] = 0
                    dec_inputs['dur'][b, step + 1] = 0

            if not_end_flag.sum() == 0:
                break

            if dec_inputs_gt is not None:
                # token_next_s_pred = [self.id2token[x.item()] for x in dec_output[:, -1, 0].argmax(-1)]
                # print(self.id2token[dec_inputs_gt['token'][0, step + 1].item()])
                # print(token_next_s_pred[0], dec_inputs['bar'][0, step + 1].item())
                assert dec_inputs['bar'][:, step + 1] == dec_inputs_gt['bar'][:, step + 1], \
                    (dec_inputs['bar'][:, step + 1], dec_inputs_gt['bar'][:, step + 1])
                assert dec_inputs['pos'][:, step + 1] == dec_inputs_gt['pos'][:, step + 1], \
                    (dec_inputs['pos'][:, step + 1], dec_inputs_gt['pos'][:, step + 1])

        outputs = {k: v[:, :step] for k, v in dec_inputs.items()}
        return outputs, dec_output_bars, xe_loss, xe_loss_cnt
