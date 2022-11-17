# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from typing import Optional, Tuple
from layers.lstmcell import StackedLSTMCell
import math

import torch.nn as nn
from slp.attention import MultiheadAttention
from slp.embed import Embed, PositionalEncoding
from slp.feedforward import PositionwiseFF
from slp.norm import LayerNorm, ScaleNorm
from slp.pytorch1 import repeat_layer

def reset_parameters(named_parameters, gain=1.0):
    """Initialize parameters in the transformer model."""

    for name, p in named_parameters:
        if p.dim() > 1:
            if "weight" in name:
                nn.init.xavier_normal_(p, gain=gain)

            if "bias" in name:
                nn.init.constant_(p, 0.0)


class Sublayer1(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        super(Sublayer1, self).__init__()
        self.sublayer = MultiheadAttention(
            attention_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            nystrom=nystrom,
            kernel_size=kernel_size,
            num_landmarks=num_landmarks,
        )
        self.prenorm = prenorm
        self.lnorm = LayerNorm(hidden_size) if not scalenorm else ScaleNorm(hidden_size)

    def _prenorm(self, x, attention_mask=None):
        out, _ = self.sublayer(self.lnorm(x), attention_mask=attention_mask)

        return out + x

    def _postnorm(self, x, attention_mask=None):
        out, _ = self.sublayer(x, attention_mask=attention_mask)

        return self.lnorm(x + out)

    def forward(self, x, attention_mask=None):
        return (
            self._prenorm(x, attention_mask=attention_mask)
            if self.prenorm
            else self._postnorm(x, attention_mask=attention_mask)
        )


class Sublayer2(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        inner_size=2048,
        dropout=0.1,
        prenorm=True,
        scalenorm=True,
    ):
        super(Sublayer2, self).__init__()
        self.sublayer = PositionwiseFF(hidden_size, inner_size, dropout=dropout)
        self.prenorm = prenorm
        self.lnorm = LayerNorm(hidden_size) if not scalenorm else ScaleNorm(hidden_size)

    def _prenorm(self, x):
        out = self.sublayer(self.lnorm(x))

        return out + x

    def _postnorm(self, x):
        out = self.sublayer(x)

        return self.lnorm(x + out)

    def forward(self, x):
        return self._prenorm(x) if self.prenorm else self._postnorm(x)


class Sublayer3(nn.Module):
    def __init__(
        self, hidden_size=512, num_heads=8, dropout=0.1, prenorm=True, scalenorm=True
    ):
        super(Sublayer3, self).__init__()
        self.sublayer = MultiheadAttention(
            attention_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            nystrom=False,  # Nystrom used only for self-attention
            kernel_size=None,  # convolutional residual not used when subsequent mask
        )
        self.prenorm = prenorm
        self.lnorm = LayerNorm(hidden_size) if not scalenorm else ScaleNorm(hidden_size)

        if self.prenorm:
            self.lnormy = (
                LayerNorm(hidden_size) if not scalenorm else ScaleNorm(hidden_size)
            )

    def _prenorm(self, x, y, attention_mask=None):
        out, _ = self.sublayer(
            self.lnorm(x), queries=self.lnormy(y), attention_mask=attention_mask
        )

        return out + y

    def _postnorm(self, x, y, attention_mask=None):
        out, _ = self.sublayer(x, queries=y, attention_mask=attention_mask)

        return self.lnorm(y + out)

    def forward(self, x, y, attention_mask=None):
        return (
            self._prenorm(x, y, attention_mask=attention_mask)
            if self.prenorm
            else self._postnorm(x, y, attention_mask=attention_mask)
        )


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        super(EncoderLayer, self).__init__()
        self.l1 = Sublayer1(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.l2 = Sublayer2(
            hidden_size=hidden_size,
            inner_size=inner_size,
            dropout=dropout,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )

    def forward(self, x, attention_mask=None):
        out = self.l1(x, attention_mask=attention_mask)
        out = self.l2(out)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList(
            repeat_layer(
                EncoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                    nystrom=nystrom,
                    num_landmarks=num_landmarks,
                    kernel_size=kernel_size,
                    prenorm=prenorm,
                    scalenorm=scalenorm,
                ),
                num_layers,
            )
        )

    def forward(self, x, attention_mask=None):
        for layer in self.encoder:
            x = layer(x, attention_mask=attention_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        prenorm=True,
        scalenorm=True,
    ):
        super(DecoderLayer, self).__init__()
        self.in_layer = Sublayer1(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            nystrom=False,
            kernel_size=None,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.fuse_layer = Sublayer3(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.out_layer = Sublayer2(
            hidden_size=hidden_size,
            inner_size=inner_size,
            dropout=dropout,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )

    def forward(self, targets, encoded, source_mask=None, target_mask=None):
        targets = self.in_layer(targets, attention_mask=target_mask)
        out = self.fuse_layer(encoded, targets, attention_mask=source_mask)
        out = self.out_layer(out)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        prenorm=True,
        scalenorm=True,
    ):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList(
            repeat_layer(
                DecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                    prenorm=prenorm,
                    scalenorm=scalenorm,
                ),
                num_layers,
            )
        )

    def forward(self, target, encoded, source_mask=None, target_mask=None):

        for l in self.decoder:
            target = l(
                target, encoded, source_mask=source_mask, target_mask=target_mask
            )

        return target


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )

    def forward(self, source, target, source_mask=None, target_mask=None):
        encoded = self.encoder(source, attention_mask=source_mask)
        decoded = self.decoder(
            target, encoded, source_mask=source_mask, target_mask=target_mask
        )

        return decoded


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size=30000,
        max_length=256,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        super(Transformer, self).__init__()
        self.embed = Embed(
            vocab_size,
            hidden_size,
            scale=math.sqrt(hidden_size),
            dropout=dropout,
            trainable=True,
        )
        self.pe = PositionalEncoding(embedding_dim=hidden_size, max_len=max_length)
        self.transformer_block = EncoderDecoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.drop = nn.Dropout(dropout)
        self.predict = nn.Linear(hidden_size, vocab_size)
        reset_parameters(self.named_parameters(), gain=(2.5 * hidden_size) ** -0.5)
        # nn.init.normal_(self.embed.embedding.weight, mean=0, std=hidden_size**-0.5)

    def forward(self, source, target, source_mask=None, target_mask=None):
        source = self.embed(source)
        target = self.embed(target)
        # Adding embeddings + pos embeddings
        # is done in PositionalEncoding class
        source = self.pe(source)
        target = self.pe(target)
        out = self.transformer_block(
            source, target, source_mask=source_mask, target_mask=target_mask
        )
        out = self.drop(out)
        out = self.predict(out)

        return out


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers=6,
        hidden_size=500,
        num_heads=5,
        max_length=512,
        inner_size=500,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
        feature_normalization=False,
    ):
        super(TransformerSequenceEncoder, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.pe = PositionalEncoding(embedding_dim=hidden_size, max_len=max_length)
        self.feature_norm = None

        if feature_normalization:
            self.feature_norm = ScaleNorm(hidden_size)
        self.transformer_block = Encoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.out_size = hidden_size
        reset_parameters(self.named_parameters(), gain=(2.5 * hidden_size) ** -0.5)

    def forward(self, x, attention_mask=None):
        if self.feature_norm:
            x = self.feature_norm(x)

        x = self.embed(x)
        x = self.pe(x)
        out = self.transformer_block(x, attention_mask=attention_mask).mean(dim=1)
        print('AFTER LINEAR', out.shape)
        return out



        

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Scoring LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, 1),  # bidirection => scalar
            nn.Sigmoid())

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1] (compressed pool5 features)
        Return:
            scores [seq_len, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)
        #print('lstm features', features.shape, h_n.shape, c_n.shape)

        # [seq_len, 1]
        scores = self.out(features.squeeze(1))
      
        return scores
    



class dLSTM(nn.Module):
    def __init__(self, input_size=500, hidden_size=500, num_layers=2):
        """Decoder LSTM"""
        super().__init__()

        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):
        """
        Args:
            seq_len (int)
            init_hidden
                h [num_layers=2, 1, hidden_size]
                c [num_layers=2, 1, hidden_size]
        Return:
            out_features: [seq_len, 1, hidden_size]
        """

        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)

        x = Variable(torch.zeros(batch_size, hidden_size)).cuda()
        h, c = init_hidden  # (h_0, c_0): last state of eLSTM

        out_features = []
        for i in range(seq_len):
            # last_h: [1, hidden_size] (h from last layer)
            # last_c: [1, hidden_size] (c from last layer)
            # h: [2=num_layers, 1, hidden_size] (h from all layers)
            # c: [2=num_layers, 1, hidden_size] (c from all layers)
            (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))
            x = self.out(last_h)
            out_features.append(last_h)
        # list of seq_len '[1, hidden_size]-sized Variables'
        return out_features


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.tr = TransformerSequenceEncoder(input_size = input_size, hidden_size = hidden_size)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)

        self.softplus = nn.Softplus()
        self.linear_mu = nn.Linear(hidden_size, hidden_size)
        self.linear_var = nn.Linear(hidden_size, hidden_size)

    def reparameterize(self, mu, log_variance):
        """Sample z via reparameterization trick
        Args:
            mu: [num_layers, hidden_size]
            log_var: [num_layers, hidden_size]
        Return:
            h: [num_layers, 1, hidden_size]
        """
        std = torch.exp(0.5 * log_variance)

        # e ~ N(0,1)
        epsilon = Variable(torch.randn(std.size())).cuda()

        # [num_layers, 1, hidden_size]
        return (mu + epsilon * std).unsqueeze(1)

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, input_size]
        Return:
            h: [2=num_layers, 1, input_size]
            decoded_features: [seq_len, 1, 1024]
        """
        
        seq_len = features.size(0)
        

        # [num_layers, 1, hidden_size]
        trans = self.tr(features)
        print('TRANS', trans.shape)

        # [num_layers, hidden_size]
        h = trans
        #print('VAE', h.shape)

        # [num_layers, hidden_size]
        h_mu = self.linear_mu(h)
        h_log_variance = torch.log(self.softplus(self.linear_var(h)))

        # [num_layers, 1, hidden_size]
        h = self.reparameterize(h, h)

        # [seq_len, 1, hidden_size]
        decoded_features = self.d_lstm(seq_len, init_hidden=(h, h))

        # [seq_len, 1, hidden_size]
        # reverse
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return h_mu, h_log_variance, decoded_features




class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.vae = VAE(input_size, hidden_size, num_layers)

    def forward(self, image_features, uniform=False):
        # Apply weights
        if not uniform:
            #[300, 1, 1024] for example
            print('IMAGE FEATURES', image_features.shape)
            scores = self.s_lstm(image_features)
            #print('SCORES', scores.shape)

            # [seq_len, 1, input_size]
            weighted_features = image_features * scores.view(-1, 1, 1)
            print('WEIGHTED FEATURES', weighted_features.shape)
        else:
            scores = None
            weighted_features = image_features

        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)

        return scores, h_mu, h_log_variance, decoded_features


if __name__ == '__main__':

    pass
