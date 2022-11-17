# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from layers.lstmcell import StackedLSTMCell

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Scoring LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),  # bidirection => scalar
            nn.Sigmoid())

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, 500] (compressed pool5 features)
        Return:
            scores: [seq_len, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)

        # [seq_len, 1]
        scores = self.out(features.squeeze(1))

        return scores




class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.trans = nn.Transformer(nhead=2, num_encoder_layers=2, d_model = 500)

    def forward(self, image_features):
        """
        Args:
            image_features: [seq_len, 1, hidden_size]
        Return:
            scores: [seq_len, 1]
            decoded_features: [seq_len, 1, hidden_size]
        """

        # Apply weights
        # [seq_len, 1]
        scores = self.s_lstm(image_features)

        # [seq_len, 1, hidden_size]
        weighted_features = image_features * scores.view(-1, 1, 1)

        shape = torch.Size((weighted_features.size(0), 1, 500))
        tgt= torch.cuda.FloatTensor(shape)
        torch.randn(shape, out=tgt)

        decoded_features = self.trans(weighted_features, tgt)

        return scores, decoded_features


if __name__ == '__main__':

    pass