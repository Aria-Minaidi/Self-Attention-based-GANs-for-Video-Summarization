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
        #source = torch.tensor(source).to(torch.int64)
        source = self.embed(source)
        #target = torch.tensor(target).to(torch.int64)
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



def reset_parameters1(named_parameters):
    """Initialize parameters in the transformer model."""

    for name, p in named_parameters:
        if "weight" in name:
            nn.init.xavier_normal_(p)

        if "bias" in name:
            nn.init.constant_(p, 0.0)

def attention_scores(
    k: torch.Tensor,
    q: torch.Tensor,
    dk: int,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.2,
    training: bool = True,
) -> torch.Tensor:
    r"""Calculate attention scores for scaled dot product attention
    $$s = softmax(\frac{Q \cdot K^T}{\sqrt{d}})$$
    * B: Batch size
    * L: Keys Sequence length
    * M: Queries Sequence length
    * H: Number of heads
    * A: Feature dimension
    Args:
        k (torch.Tensor): Single head [B, L, A] or multi-head [B, H, L, A/H] Keys tensor
        q (torch.Tensor): Single head [B, M, A] or multi-head [B, H, M, A/H] Keys tensor
        dk (int): Model dimension
        attention_mask (Optional[torch.Tensor]): Optional [B, [H], 1, L] pad mask or [B, [H], M, L] pad mask + subsequent mask
            tensor with zeros in sequence indices that should be masked and ones in sequence indices that should be
            preserved. Defaults to None.
        dropout (float): Drop probability. Defaults to 0.2.
        training (bool): Is module in training phase? Defaults to True.
    Returns:
        torch.Tensor: [B, M, L] or [B, H, M, L] attention scores
    """
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)

    if attention_mask is not None:
        scores = scores + ((1 - attention_mask) * -1e5)
    scores = F.softmax(scores, dim=-1)
    scores = F.dropout(scores, p=dropout, training=training)

    return scores

def attention(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    dk: int,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.2,
    training: bool = True,
):
    r"""Reweight values using scaled dot product attention
    $$s = softmax(\frac{Q \cdot K^T}{\sqrt{d}}) V$$
    * B: Batch size
    * L: Keys Sequence length
    * M: Queries Sequence length
    * H: Number of heads
    * A: Feature dimension
    Args:
        k (torch.Tensor): Single head [B, L, A] or multi-head [B, H, L, A/H] Keys tensor
        q (torch.Tensor): Single head [B, M, A] or multi-head [B, H, M, A/H] Keys tensor
        v (torch.Tensor): Single head [B, M, A] or multi-head [B, H, M, A/H] Values tensor
        dk (int): Model dimension
        attention_mask (Optional[torch.Tensor]): Optional [B, [H], 1, L] pad mask or [B, [H], M, L] pad mask + subsequent mask
            tensor with zeros in sequence indices that should be masked and ones in sequence indices that should be
            preserved. Defaults to None.
        dropout (float): Drop probability. Defaults to 0.2.
        training (bool): Is module in training phase? Defaults to True.
    Returns:
        torch.Tensor: [B, M, L] or [B, H, M, L] attention scores
    """

    scores = attention_scores(
        k, q, dk, attention_mask=attention_mask, dropout=dropout, training=training
    )
    out = torch.matmul(scores, v)

    return out, scores

class SelfAttention(nn.Module):
    def __init__(
        self,
        attention_size: int = 512,
        input_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Single-Headed Dot-product self attention module
        Args:
            attention_size (int): Number of hidden features. Defaults to 512.
            input_size (Optional[int]): Input features. Defaults to None.
                If None input_size is set to attention_size.
            dropout (float): Drop probability. Defaults to 0.1.
        """
        super(SelfAttention, self).__init__()

        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.kqv = nn.Linear(input_size, 3 * attention_size, bias=False)
        self.dropout = dropout
        self.outt = nn.Sequential(nn.Linear(500, 1), nn.Sigmoid()) #to scalar
        reset_parameters1(self.named_parameters())

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Single-head scaled dot-product attention forward pass
        Outputs the values, where features for each sequence element are weighted by their respective attention scores
        $$a = softmax(\frac{Q}{K^T}){\sqrt{d}}) \dot V$$
        * B: Batch size
        * L: Keys Sequence length
        * M: Queries Sequence length
        * H: Number of heads
        * A: Feature dimension
        Args:
            x (torch.Tensor): [B, L, D] Input tensor
            attention_mask (Optional[torch.Tensor]): Optional [B, L] or [B, M, L] zero-one mask for sequence elements. Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Reweighted values [B, L, D], attention scores [B, M, L])
        """
        if attention_mask is not None:
            if len(list(attention_mask.size())) == 2:
                attention_mask = attention_mask.unsqueeze(1)

        k, q, v = self.kqv(x).chunk(3, dim=-1)  # (B, L, A)

        # weights => (B, L, L)
        out, scores = attention(
            k,
            q,
            v,
            self.dk,
            attention_mask=attention_mask,
            dropout=self.dropout,
            training=self.training,
        )
        out = self.outt(out)
        return out, scores





 


class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        #self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.attn = SelfAttention(hidden_size, input_size)
        self.trans = TransformerSequenceEncoder(input_size = input_size, hidden_size = hidden_size)#Transformer(hidden_size = 500, num_heads = 5)

    def forward(self, image_features, uniform=False):
        # Apply weights
        if not uniform:
            #[300, 1, 1024] for example
            #print('IMAGE FEATURES', image_features.shape)
            scores = self.attn(image_features)  # [seq_len, 1]
            print('SCORES',  scores[0].shape)
            scores = scores[0]
            #print('SCORES',  scores)
            
            print('SCORES', scores.shape, image_features.shape, scores.view(-1, 1, 1).shape)

            # [seq_len, 1, input_size]
            weighted_features = image_features * scores.view(-1, 1, 1)
            #print('WEIGHTED FEATURES', weighted_features.shape, weighted_features)
        else:
            scores = None
            weighted_features = image_features

        shape = torch.Size((image_features.size(0), 1, 500))
        tgt= torch.cuda.FloatTensor(shape)
        torch.randn(shape, out=tgt)
        decoded_features = self.trans(weighted_features)
        #print('DECODED FEATURES', decoded_features)
        return scores, decoded_features


if __name__ == '__main__':

    pass
