import torch
import torch.nn as nn

from modules.music_ae.vqvae import VectorQuantizedVAE
from modules.music_transformer.music_transformer import MusicTransformer
from utils.hparams import hparams


class MusicAutoEncoder(MusicTransformer):
    def __init__(self, id2token, token2id):
        super().__init__(id2token, token2id)
        self.feat_hidden_lv1 = hparams['feat_hidden_lv1']
        if hparams.get("lambda_emo", 0) > 0:
            self.vqvae = VectorQuantizedVAE(hparams['feat_hidden_lv1'], hparams['vqvae_dim'], hparams['emo_class'])

    def forward(self, enc_inputs, dec_inputs):
        cond_embeds = self.mumidi_embed(**enc_inputs, cond=True)  # [B, T, H]
        enc_outputs = self.encoder(cond_embeds)
        # print("enc output", enc_outputs.shape)
        # to bar level code
        if hparams.get("group", True):
            codes = self.group_hidden_by_bars(enc_inputs['token'], enc_outputs, enc_inputs['bar'])  # [B, T_bar, H_code]
            # print("group by bars", enc_outputs.shape[1], codes.shape[1])
            feats = codes[:, :, :self.feat_hidden_lv1].mean(1)  # [B, H_feat]
            if hparams.get("lambda_emo", 0) > 0:
                feats_rec, z_e_x, z_q_x, emo_out = self.vqvae(feats)
            contents = codes[:, :, self.feat_hidden_lv1:]
            codes = torch.cat([feats_rec.unsqueeze(1).repeat([1, contents.shape[1], 1]), contents], -1)
            # print("feats", feats.shape)
            # expand to the same length
            codes = self.expand_hidden_by_bars(codes, enc_inputs['bar'])  # [B, T, H_code]
            # print("codes2", codes.shape)
            # split part to calculate the low level feature loss
        else:
            codes = enc_outputs
            feats = codes[:, :, :self.feat_hidden_lv1].mean(1)  # [B, 1. H_feat]
        dec_outputs, _ = self.decoder(codes)
        token_out, vel_out, dur_out = self.split_dec_outputs(dec_outputs)  # [B, T, 842]
        # print("token out", token_out.shape, enc_outputs.shape)
        if hparams.get("lambda_vqvae_rec", 0) > 0:
            return token_out, vel_out, dur_out, feats, feats_rec, z_e_x, z_q_x, emo_out
        if hparams.get("lambda_emo", 0) > 0:
            return token_out, vel_out, dur_out, feats, z_e_x, z_q_x, emo_out
        return token_out, vel_out, dur_out, feats

    def group_hidden_by_bars(self, tokens, h, bar_ids):
        """
        :param tokens: input tokens, [B, T]
        :param h: [B, T, H]
        :param bar_ids: [B, T]
        :return: [B, T_bar, H]
        """
        h = h.permute(1, 0, 2)
        tokens = tokens.T
        bar_ids = bar_ids.T
        T, B, H = h.shape
        num_bars = bar_ids.max() + 1

        nonpadding = (tokens != 0).float()
        h = h * nonpadding[:, :, None]
        h_ref_by_bars = h.new_zeros([num_bars, B, H]).scatter_add_(
            0, bar_ids[:, :, None].repeat([1, 1, H]), h)
        all_ones = h.new_ones(h.shape[:2])
        h_ref_segs_cnts = h.new_zeros([num_bars, B]).scatter_add_(0, bar_ids, all_ones).contiguous()
        h_ref_segs_cnts[0] -= (tokens == 0).float().sum(0)
        h_ref_by_bars = h_ref_by_bars / torch.clamp(h_ref_segs_cnts[:, :, None], min=1)
        h_ref_by_bars = h_ref_by_bars.permute(1, 0, 2)
        return h_ref_by_bars

    def expand_hidden_by_bars(self, h, bar_ids):
        """
        :param h: [B, T_bar, H]
        :param bar_ids: [B, T]
        :return: [B, T, H]
        """
        _, _, H = h.shape
        bar_ids = bar_ids[..., None].repeat([1, 1, H])
        return torch.gather(h, 1, bar_ids)
