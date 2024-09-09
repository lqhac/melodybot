from modules.commons.common_layers import *
from modules.commons.common_layers import Embedding
from modules.fastspeech.tts_modules import ConvEmbedding, ConvEmbedding2, ConvEmbedding3
from modules.voice_conversion.common_layers import ConvStacks, ConvLSTMStacks, ConvGLUStacks
from utils.hparams import hparams


class SpecAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_hidden_size = hparams['spec_hidden_size']
        self.spec_out = hparams['spec_out']
        self.pitch_hidden_size = hparams['pitch_hidden_size']
        self.pitch_out = hparams['pitch_out']
        self.spk_hidden_size = hparams['spk_hidden_size']
        self.spk_out = hparams['spk_out']
        self.dec_hidden_size = hparams['dec_hidden_size']
        self.audio_num_mel_bins = hparams['audio_num_mel_bins']
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        if hparams['arch'] == 'conv':
            self.layer_cls = ConvStacks
        elif hparams['arch'] == 'conv_lstm':
            self.layer_cls = ConvLSTMStacks
        elif hparams['arch'] == 'conv_glu':
            self.layer_cls = ConvGLUStacks

        # content encoder
        self.content_encoder = self.layer_cls(idim=self.audio_num_mel_bins, n_chans=self.spec_hidden_size,
                                              n_layers=self.enc_layers, odim=self.spec_out,
                                              dropout=hparams['dropout'], norm=hparams['norm_type'])
        self.build_spk_encoder(self.spk_hidden_size, self.spk_out)
        self.build_pitch_encoder(self.pitch_hidden_size, self.pitch_out)
        # decoder
        self.decoder_inp_proj = Linear(
            self.spk_out + self.spec_out + self.pitch_out, self.dec_hidden_size)
        self.decoder = self.layer_cls(idim=self.dec_hidden_size, n_chans=self.dec_hidden_size,
                                      n_layers=self.dec_layers, odim=self.audio_num_mel_bins,
                                      dropout=hparams['dropout'], norm=hparams['norm_type'])

    def build_pitch_encoder(self, pitch_hidden_size, pitch_out):
        if hparams['pitch_embed_type'] == 1:
            self.pitch_embed = ConvEmbedding(300, pitch_hidden_size)
        elif hparams['pitch_embed_type'] == 2:
            self.pitch_embed = ConvEmbedding2(300, pitch_hidden_size)
        elif hparams['pitch_embed_type'] == 3:
            self.pitch_embed = ConvEmbedding3(300, pitch_hidden_size)
        elif hparams['pitch_embed_type'] == 4:
            self.pitch_embed = nn.Conv1d(1, pitch_hidden_size, 9, padding=4)
        else:
            self.pitch_embed = Embedding(300, pitch_hidden_size, 0)
        self.pitch_encoder = self.layer_cls(idim=pitch_hidden_size, n_chans=pitch_hidden_size,
                                            n_layers=self.enc_layers, odim=pitch_out,
                                            dropout=hparams['dropout'], norm=hparams['norm_type'])

    def build_spk_encoder(self, spk_hidden_size, spk_out):
        self.spk_encoder = self.layer_cls(idim=self.audio_num_mel_bins, n_chans=spk_hidden_size,
                                          n_layers=self.enc_layers, odim=spk_out,
                                          dropout=hparams['dropout'], norm=hparams['norm_type'])

    def forward_spk_encoder(self, mels):
        T = mels.shape[1]
        h_spk = self.spk_encoder(mels).mean(1)[:, None, :].repeat(1, T, 1)
        return h_spk

    def forward_pitch_encoder(self, pitch, f0):
        if hparams['pitch_embed_type'] == 4:
            f0 = f0.clone()
            pitch_embed = self.pitch_embed(f0[:, None, :]).transpose(1, 2)
        else:
            pitch_embed = self.pitch_embed(pitch)
        h_pitch = self.pitch_encoder(pitch_embed)
        return h_pitch

    def forward(self, mels, pitch, f0=None, uv=None):
        """

        :param mels: [B, T, 80]
        :return: {'mel_out': [B, T, 80], 'pitch': [B, T]}
        """
        ret = {}
        nonpadding = (mels.abs().sum(-1) > 0).float()[:, :, None]
        h_spk = self.forward_spk_encoder(mels)
        h_pitch = self.forward_pitch_encoder(pitch, f0)
        h_mels = self.content_encoder(mels)  # [B, T, H]
        h_mels = self.decoder_inp_proj(torch.cat([h_mels, h_pitch, h_spk], -1))
        h_mels = h_mels * nonpadding
        ret['mel_out'] = self.decoder(h_mels) * nonpadding
        return ret

