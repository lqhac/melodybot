import random
from math import sqrt

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from modules.commons.common_layers import LinearNorm
from modules.fastspeech.tts_modules import ConvNorm
from utils.tts_utils import get_mask_from_lengths


###################
# tacotron module
###################


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size, hparams):
        super(Attention, self).__init__()
        self.hparams = hparams
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -1e8

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, force_attn=None):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask.bool(), self.score_mask_value)
        if force_attn is not None:
            alignment.data.masked_fill_(~force_attn.bool(), self.score_mask_value)
        attention_weights = F.softmax(alignment, dim=1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, hparams):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.hparams = hparams
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=self.hparams['prenet_dropout'], training=True)
        return x


class TacotronDecoder(nn.Module):
    def __init__(self, hparams):
        super(TacotronDecoder, self).__init__()
        self.n_mel_channels = hparams['n_mel_channels']
        self.n_frames_per_step = hparams['n_frames_per_step']
        self.encoder_embedding_dim = hparams['encoder_embedding_dim']
        self.attention_rnn_dim = hparams['attention_rnn_dim']
        self.decoder_rnn_dim = hparams['decoder_rnn_dim']
        self.prenet_dim = hparams['prenet_dim']
        self.max_decoder_steps = hparams['max_decoder_steps']
        self.gate_threshold = hparams['gate_threshold']
        self.p_attention_dropout = hparams['p_attention_dropout']
        self.p_decoder_dropout = hparams['p_decoder_dropout']
        self.hparams = hparams
        self.prenet = Prenet(
            hparams['n_mel_channels'] * hparams['n_frames_per_step'],
            [hparams['prenet_dim'], hparams['prenet_dim']], hparams)

        self.attention_rnn = nn.LSTMCell(
            hparams['prenet_dim'] + hparams['encoder_embedding_dim'],
            hparams['attention_rnn_dim'])

        self.attention_layer = Attention(
            hparams['attention_rnn_dim'], hparams['encoder_embedding_dim'],
            hparams['attention_dim'], hparams['attention_location_n_filters'],
            hparams['attention_location_kernel_size'], hparams)

        self.decoder_rnn = nn.LSTMCell(
            hparams['attention_rnn_dim'] + hparams['encoder_embedding_dim'],
            hparams['decoder_rnn_dim'], True)

        self.linear_projection = LinearNorm(
            hparams['decoder_rnn_dim'] + hparams['encoder_embedding_dim'],
            hparams['n_mel_channels'] * hparams['n_frames_per_step'])

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = memory.data.new(B, self.n_mel_channels * self.n_frames_per_step).zero_()
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = memory.data.new(B, self.attention_rnn_dim).zero_()
        self.attention_cell = memory.data.new(B, self.attention_rnn_dim).zero_()

        self.decoder_hidden = memory.data.new(B, self.decoder_rnn_dim).zero_()
        self.decoder_cell = memory.data.new(B, self.decoder_rnn_dim).zero_()

        self.attention_weights = memory.data.new(B, MAX_TIME).zero_()
        self.attention_weights_cum = memory.data.new(B, MAX_TIME).zero_()
        self.attention_context = memory.data.new(B, self.encoder_embedding_dim).zero_()

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def decode(self, decoder_input, force_attn=None):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask, force_attn=force_attn)

        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = None
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, encoder_outputs, decoder_inputs,
                encoder_outputs_lengths, force_attn=None):
        """

        :param encoder_outputs: [B, T, H]
        :param decoder_inputs: [T, B, 80]
        :param encoder_outputs_lengths: [B]
        :param force_attn:
        :return: [B, T, 80], [B, T_sp, T_txt]
        """
        decoder_input = self.get_go_frame(encoder_outputs).unsqueeze(0)  # [T, B, 80]
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)  # (T, B, H)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            encoder_outputs, mask=~get_mask_from_lengths(encoder_outputs_lengths))

        outputs, alignments = [], []
        while len(outputs) < decoder_inputs.size(0) - 1:
            if len(outputs) >= 1:
                decoder_input = decoder_inputs[len(outputs)]
            else:
                decoder_input = decoder_inputs[0]
            output, gate_output, attention_weights = self.decode(decoder_input)
            outputs += [output.squeeze(1)]
            alignments += [attention_weights]
        return torch.stack(outputs, 1), torch.stack(alignments, 1)


def get_diagonal_mask(alignments_shape, attn_ks, width=5, slope=1.3):
    y = torch.arange(alignments_shape[0]).float().cuda()[None, :, None].repeat([1, 1, alignments_shape[1]])
    x = (torch.arange(alignments_shape[1]).float()).cuda()[None, None, :].repeat([1, alignments_shape[0], 1])
    mask_k = (y > attn_ks[:, None, None] / slope * (x - width)) & (y < attn_ks[:, None, None] * slope * (x + width))
    return mask_k.float()


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def calc_mean_invstddev(feature):
    if len(feature.size()) != 3:
        raise ValueError("We expect the input feature to be 2-D tensor")
    mean = feature.mean(1, keepdim=True)
    var = feature.var(1, keepdim=True)
    # avoid division by ~zero
    eps = 1e-8
    if (var < eps).any():
        return mean, 1.0 / (torch.sqrt(var) + eps)
    return mean, 1.0 / torch.sqrt(var)


def apply_mv_norm(features):
    mean, invstddev = calc_mean_invstddev(features)
    res = (features - mean) * invstddev
    return res


delta_filter_stack = None


def add_delta_deltas(mel):
    """

    :param mel: [B, num_bins, T]
    :return: [B, num_bins*3, T]
    """
    global delta_filter_stack
    if delta_filter_stack is None:
        delta_filter = np.array([2, 1, 0, -1, -2])
        delta_delta_filter = scipy.signal.convolve(delta_filter, delta_filter, "full")
        delta_filter_stack = np.array(
            [[0] * 4 + [1] + [0] * 4, [0] * 2 + list(delta_filter) + [0] * 2,
             list(delta_delta_filter)],
            dtype=np.float32).T[:, None, None, :]
        delta_filter_stack /= np.sqrt(
            np.sum(delta_filter_stack ** 2, axis=0, keepdims=True))

        delta_filter_stack = delta_filter_stack.transpose(3, 2, 1, 0)

    ret = F.conv2d(mel[:, None, ...], torch.FloatTensor(delta_filter_stack).cuda(),
                   padding=(0, 4))
    assert ret.shape[0] == mel.shape[0]
    assert ret.shape[2] == mel.shape[1]
    assert ret.shape[3] == mel.shape[2]
    ret = ret.reshape([ret.shape[0], -1, ret.shape[-1]])
    return ret


def get_best_splits(a_prob):
    a_prob_ori = a_prob
    a_prob = a_prob.copy()
    a_prob[a_prob < 0.01] = 0
    f = np.zeros_like(a_prob)
    best_splits = np.zeros_like(a_prob, dtype=np.int)
    f[0] = np.cumsum(a_prob[0])
    for t in range(1, a_prob.shape[0]):
        prob_cumsum = np.cumsum(a_prob[t]) + 0.000001
        for s in range(t, a_prob.shape[1]):
            new_prob = f[t - 1, :s] + (prob_cumsum[s] - prob_cumsum[:s])
            new_prob[prob_cumsum[:s] / prob_cumsum[s] < 0.05] = 0
            best_f = new_prob.max()
            if best_f > 0:
                best_idx = np.where(new_prob == best_f)[0][-1]
                if new_prob[best_idx] >= f[t, s]:
                    f[t, s] = new_prob[best_idx]
                    best_splits[t, s] = best_idx

    route = [a_prob.shape[1] - 1]
    for i in range(a_prob.shape[0] - 1, 0, -1):
        route.append(best_splits[i, route[-1]])
    route.reverse()

    last_pos = 0
    total_scores = []
    for i in range(a_prob.shape[0]):
        total_scores.append(a_prob_ori[i, last_pos: route[i] + 1].sum())
        last_pos = route[i] + 1
    return np.array(route), np.array(total_scores)


#####################
# models
#####################


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.hparams = hparams

        convolutions = []
        for _ in range(hparams['encoder_n_convolutions']):
            conv_layer = nn.Sequential(
                ConvNorm(hparams['encoder_embedding_dim'],
                         hparams['encoder_embedding_dim'],
                         kernel_size=hparams['encoder_kernel_size'], stride=1,
                         padding=int((hparams['encoder_kernel_size'] - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams['encoder_embedding_dim']))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams['encoder_embedding_dim'],
                            int(hparams['encoder_embedding_dim'] / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams['n_mel_channels']
        self.n_frames_per_step = hparams['n_frames_per_step']
        self.encoder_embedding_dim = hparams['encoder_embedding_dim']
        self.attention_rnn_dim = hparams['attention_rnn_dim']
        self.decoder_rnn_dim = hparams['decoder_rnn_dim']
        self.prenet_dim = hparams['prenet_dim']
        self.max_decoder_steps = hparams['max_decoder_steps']
        self.gate_threshold = hparams['gate_threshold']
        self.p_attention_dropout = hparams['p_attention_dropout']
        self.p_decoder_dropout = hparams['p_decoder_dropout']
        self.hparams = hparams

        if hparams['asr']:
            self.prenet = nn.Embedding(hparams['n_symbols'], hparams['prenet_dim'])
            std = sqrt(2.0 / (hparams['n_symbols'] + hparams['prenet_dim']))
            val = sqrt(3.0) * std  # uniform bounds for std
            self.prenet.weight.data.uniform_(-val, val)
            self.n_mel_channels = hparams['encoder_embedding_dim']
            self.text_top_fc = nn.Linear(hparams['encoder_embedding_dim'], hparams['n_symbols'])
        else:
            self.prenet = Prenet(
                hparams['n_mel_channels'] * hparams['n_frames_per_step'],
                [hparams['prenet_dim'], hparams['prenet_dim']], hparams)

        self.attention_rnn = nn.LSTMCell(
            hparams['prenet_dim'] + hparams['encoder_embedding_dim'],
            hparams['attention_rnn_dim'])

        self.attention_layer = Attention(
            hparams['attention_rnn_dim'], hparams['encoder_embedding_dim'],
            hparams['attention_dim'], hparams['attention_location_n_filters'],
            hparams['attention_location_kernel_size'], hparams)

        self.decoder_rnn = nn.LSTMCell(
            hparams['attention_rnn_dim'] + (
                hparams['encoder_embedding_dim'] * 2 if not hparams['asr'] else hparams['encoder_embedding_dim']),
            hparams['decoder_rnn_dim'], True)

        if hparams['asr']:
            self.linear_projection = LinearNorm(
                hparams['decoder_rnn_dim'] + hparams['encoder_embedding_dim'], hparams['encoder_embedding_dim'])
        else:
            self.linear_projection = LinearNorm(
                hparams['decoder_rnn_dim'] + hparams['encoder_embedding_dim'],
                hparams['n_mel_channels'] * hparams['n_frames_per_step'])

        # self.gate_layer = LinearNorm(
        #     hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
        #     bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        if self.hparams['asr']:
            decoder_input = Variable(memory.data.new(B).zero_()).long()
        else:
            decoder_input = Variable(memory.data.new(
                B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        if not self.hparams['asr']:
            # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
            decoder_inputs = decoder_inputs.transpose(1, 2)
            decoder_inputs = decoder_inputs.view(
                decoder_inputs.size(0),
                int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
            # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        if not self.hparams['asr']:
            mel_outputs = mel_outputs.view(
                mel_outputs.size(0), -1, self.n_mel_channels)
            # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
            mel_outputs = mel_outputs.transpose(1, 2)

        if gate_outputs and gate_outputs[0] is not None:
            gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, pitch=None, force_attn=None):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask, force_attn=force_attn)

        self.attention_weights_cum += self.attention_weights

        if pitch is not None:
            decoder_input = torch.cat(
                (self.attention_hidden, self.attention_context, pitch), -1)
        else:
            decoder_input = torch.cat(
                (self.attention_hidden, self.attention_context), -1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = None
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths, pitch, force_attn=None, steps=None):
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)  # (T, B, H)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        outputs, gate_outputs, alignments = [], [], []
        while len(outputs) < decoder_inputs.size(0) - 1:
            if len(outputs) >= 1:
                if self.hparams['always_teacher_forcing']:
                    tf_ratio = 1
                else:
                    tf_ratio = (1 - np.clip((steps - 5000) / 25000, 0, 0.9)) if steps is not None else 1
                decoder_input = decoder_inputs[len(outputs)] \
                    if random.random() < tf_ratio else self.prenet(outputs[-1][None, ...])[0]
            else:
                decoder_input = decoder_inputs[0]
            output, gate_output, attention_weights = self.decode(
                decoder_input, pitch.transpose(0, 1)[len(outputs)] if pitch is not None else None,
                force_attn=force_attn[:, len(outputs)] if force_attn is not None else None)
            outputs += [output.squeeze(1)]
            alignments += [attention_weights]
            gate_outputs += [gate_output]

        outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            outputs, gate_outputs, alignments)

        if self.hparams['asr']:
            outputs = self.text_top_fc(outputs)
        return outputs, gate_outputs, alignments

    def inference(self, memory, memory_lengths, pitch, force_attn=None, max_decoder_steps=None):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)
        if max_decoder_steps is None:
            max_decoder_steps = pitch.shape[1]

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            output, gate_output, alignment = self.decode(
                decoder_input, pitch.transpose(0, 1)[len(outputs)] if pitch is not None else None,
                force_attn=force_attn[:, len(outputs)] if force_attn is not None else None)
            if self.hparams['asr']:
                output = torch.argmax(self.text_top_fc(output), -1)
                outputs += [output]

            alignments += [alignment]
            outputs += [output.squeeze(1)]
            if len(outputs) == max_decoder_steps:
                break
            decoder_input = output

        outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            outputs, gate_outputs, alignments)

        return outputs, gate_outputs, alignments


class TacotronAsrLoss(nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams
        super(TacotronAsrLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(reduction='sum', zero_infinity=True)

    def forward(self, model_output, targets, inputs):
        losses = {}
        text, text_lengths, mels, mel_lengths, attn_mask = \
            inputs['text_padded'], inputs['text_lengths'], inputs['mel_padded'], inputs['mel_lengths'], inputs['mask']

        text_targets = text
        text_targets.requires_grad = False

        text_outputs, alignments = model_output['text_outputs'], model_output['alignments']

        xe_loss = F.cross_entropy(text_outputs.transpose(1, 2), text_targets, reduce=False)
        out_mask = get_mask_from_lengths(text_lengths).float()
        xe_loss = (xe_loss * out_mask).sum() / out_mask.sum()
        losses['xe_loss'] = xe_loss
        return losses


class TacotronAsrAlign(nn.Module):
    def __init__(self, hparams):
        super(TacotronAsrAlign, self).__init__()
        self.hparams = hparams
        self.mask_padding = hparams['mask_padding']
        self.n_mel_channels = hparams['n_mel_channels']
        self.n_frames_per_step = hparams['n_frames_per_step']
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        convolutions = []
        for i in range(hparams['encoder_n_convolutions']):
            conv_layer = nn.Sequential(
                ConvNorm(hparams['encoder_embedding_dim'] if i > 0 else hparams['n_mel_channels'],
                         hparams['encoder_embedding_dim'],
                         kernel_size=hparams['encoder_kernel_size'], stride=1,
                         padding=int((hparams['encoder_kernel_size'] - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams['encoder_embedding_dim']))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, inputs, steps=None):
        text, text_lengths, mels, mel_lengths = \
            inputs['text_padded'], inputs['text_lengths'], inputs['mel_padded'], inputs['mel_lengths']

        text_lengths, mel_lengths = text_lengths.data, mel_lengths.data

        x = mels
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.1, self.training)
        encoder_outputs = self.encoder(x, mel_lengths)

        assert mel_lengths.max() == x.shape[-1], (mel_lengths.max(), x.shape[-1])
        assert mel_lengths.max() == encoder_outputs.shape[1], (mel_lengths.max(), encoder_outputs.shape[1])
        if self.hparams['encoder_res']:
            encoder_outputs += x.transpose(1, 2)

        text_outputs, _, alignments = self.decoder(
            encoder_outputs, text, memory_lengths=mel_lengths, pitch=None)
        if self.mask_padding and text_lengths is not None:
            mask = ~get_mask_from_lengths(text_lengths)[:, :, None]
            text_outputs.data.masked_fill_(mask, 0.0)
        return {
            'text_outputs': text_outputs,
            'alignments': alignments
        }

    def inference(self, inputs):
        return self(inputs)
