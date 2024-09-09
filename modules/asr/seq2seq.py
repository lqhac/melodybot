import torch
from torch import nn
from utils_structMIDI import fill_with_neg_inf
from modules.asr.base import Prenet
from modules.commons.common_layers import SinusoidalPositionalEmbedding, Linear
from modules.commons.common_layers import Embedding
from modules.fastspeech.tts_modules import FFTBlocks, TransformerDecoderLayer, DEFAULT_MAX_TARGET_POSITIONS, LayerNorm
import torch.nn.functional as F

from utils_structMIDI.mumidi.hparams import hparams, set_hparams


class TransformerASRDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, out_dim, use_pos_embed=True, num_heads=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.padding_idx = 0
        self.dropout = dropout
        self.out_dim = out_dim
        self.use_pos_embed = use_pos_embed
        if self.use_pos_embed:
            self.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.embed_positions = SinusoidalPositionalEmbedding(
                self.hidden_size, self.padding_idx,
                init_size=self.max_target_positions + self.padding_idx + 1,
            )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(self.hidden_size, self.dropout, num_heads=num_heads)
            for i in range(self.num_layers)
        ])
        self.layer_norm = LayerNorm(self.hidden_size)
        self.project_out_dim = Linear(self.hidden_size, self.out_dim, bias=False)

    def forward(self, dec_inputs, encoder_out=None, incremental_state=None):
        """

        :param dec_inputs:  [B, T, H]
        :param encoder_out: [B, T, H]
        :return: [B, T, W]
        """
        self_attn_padding_mask = dec_inputs.abs().sum(-1).eq(0).data
        if encoder_out is not None:
            encoder_padding_mask = encoder_out.abs().sum(-1).eq(0)
        else:
            encoder_padding_mask = None
        # embed positions
        x = dec_inputs
        if self.use_pos_embed:
            positions = self.embed_positions(dec_inputs.abs().sum(-1))
            x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        if encoder_out is not None:
            encoder_out = encoder_out.transpose(0, 1)
        all_attn_logits = []
        for layer in self.layers:
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, attn_logits = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            all_attn_logits.append(attn_logits)

        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # B x T x C -> B x T x W
        x = self.project_out_dim(x)
        return x, all_attn_logits

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class TransformerASR(nn.Module):
    def __init__(self, dictionary, enc_layers=None, dec_layers=None):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.enc_layers = hparams['enc_layers'] if enc_layers is None else enc_layers
        self.dec_layers = hparams['dec_layers'] if dec_layers is None else dec_layers
        self.num_heads = 2
        self.hidden_size = hparams['hidden_size']
        self.mel = hparams['audio_num_mel_bins']
        self.token_embed = self.build_embedding(self.dictionary, self.hidden_size)
        self.mel_prenet = Prenet(self.mel, self.hidden_size)
        self.encoder = FFTBlocks(num_layers=self.enc_layers, hidden_size=self.hidden_size)
        self.decoder = TransformerASRDecoder(
            self.hidden_size, self.dec_layers, hparams['drop_out'], len(self.dictionary))

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward_encoder(self, spec):
        """

        :param spec:
        :return: encoder_out: [B, T, H], encoder_padding_mask: [B, T]
        """
        x = self.mel_prenet(spec)[1]
        encoder_out = self.encoder(x)
        encoder_padding_mask = spec.abs().sum(-1).eq(0).data
        encoder_out = encoder_out * encoder_padding_mask[:, :, None]
        return {
            'encoder_out': encoder_out,
            'encoder_padding_mask': encoder_padding_mask
        }

    def forward(self, spec, prev_tokens, *args, **kwargs):
        encoder_out = self.forward_encoder(spec)['encoder_out']
        decoder_output, attn_logits = self.decoder(self.token_embed(prev_tokens), encoder_out)
        return decoder_output, attn_logits

    def infer(self, spec):
        input = spec
        bsz = input.size(0)
        max_input_len = input.size(1)
        decode_length = self.estimate_decode_length(max_input_len)
        encoder_out = self.forward_encoder(input)['encoder_out']

        hit_eos = input.new(bsz, 1).fill_(0).bool()
        decoder_input = input.new(bsz, decode_length).fill_(0).long()
        decoded_tokens = input.new(bsz, 0).fill_(0).long()
        encdec_attn_logits = []

        for i in range(self.dec_layers):
            encdec_attn_logits.append(input.new(bsz, self.num_heads, 0, max_input_len).fill_(0).float())
        incremental_state = {}
        step = 0

        def is_finished(step, decode_length, hit_eos):
            finished = step >= decode_length
            finished |= (hit_eos[:, -1].sum() == hit_eos.size(0)).cpu().numpy()
            return finished

        while not is_finished(step, decode_length, hit_eos):
            decoder_output, attn_logits = self.decoder(
                decoder_input[:, step:step + 1], encoder_out, incremental_state=incremental_state)
            next_token = decoder_output[:, -1:].argmax(-1)
            decoded_tokens = torch.cat((decoded_tokens, next_token), dim=1)
            for i in range(self.dec_layers):
                encdec_attn_logits[i] = torch.cat((encdec_attn_logits[i], attn_logits[i]), dim=2)
            step += 1

            this_hit_eos = hit_eos[:, -1:]
            this_hit_eos |= next_token == self.dictionary.eos()
            hit_eos = torch.cat((hit_eos, this_hit_eos), dim=1)
            decoder_input[:, step] = next_token[:, -1]
        return decoded_tokens, encdec_attn_logits

    def estimate_decode_length(self, input_length):
        return input_length // 2
