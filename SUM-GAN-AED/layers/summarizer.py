# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from typing import Optional, Tuple
from layers.lstmcell import StackedLSTMCell

def reset_parameters(named_parameters):
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
        reset_parameters(self.named_parameters())

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



class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.linear_mu = nn.Linear(hidden_size, hidden_size)
        self.linear_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, frame_features):
        """
        Args:
            frame_features: [seq_len, 1, input_size]
        Return:
            last hidden
                h_last [num_layers=2, 1, hidden_size]
                c_last [num_layers=2, 1, hidden_size]
        """
        self.lstm.flatten_parameters()
        _, (h_last, c_last) = self.lstm(frame_features)
        
        return (h_last, c_last)


class dLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
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
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)

        self.softplus = nn.Softplus()

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
        h, c = self.e_lstm(features)
        #print('VAE', h.shape, c.shape)

        # [num_layers, hidden_size]
        h = h.squeeze(1)
        #print('VAE', h.shape)

        # [num_layers, hidden_size]
        h_mu = self.e_lstm.linear_mu(h)
        h_log_variance = torch.log(self.softplus(self.e_lstm.linear_var(h)))

        # [num_layers, 1, hidden_size]
        h = self.reparameterize(h_mu, h_log_variance)

        # [seq_len, 1, hidden_size]
        decoded_features = self.d_lstm(seq_len, init_hidden=(h, c))

        # [seq_len, 1, hidden_size]
        # reverse
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return h_mu, h_log_variance, decoded_features




class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        #self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.attn = SelfAttention(hidden_size, input_size)
        self.vae = VAE(input_size, hidden_size, num_layers)

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

        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)
        #print('DECODED FEATURES', decoded_features)
        return scores, h_mu, h_log_variance, decoded_features


if __name__ == '__main__':

    pass
