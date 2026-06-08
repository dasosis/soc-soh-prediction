"""torch nn.Module definitions for the SOC roster.

Every module maps an input window x of shape (N, L, F) — (batch, window_len,
n_features=3 for V/I/T) — to a 1-D SOC prediction of shape (N,). Output is a
plain linear head (no sigmoid): SOC labels include exactly 1.0 at the start of
every profile, which a sigmoid could not reach without saturating; MSE on a
linear head is more stable and predictions are evaluated, not clipped.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(self, n_features, window_len, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers, d = [], n_features * window_len
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(-1)


class RNNNet(nn.Module):
    """LSTM / GRU, optionally bidirectional; last timestep -> FC."""

    def __init__(self, n_features, window_len, kind="lstm", hidden=96,
                 num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn_cls(
            n_features, hidden, num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x):
        out, _ = self.rnn(x)          # (N, L, H*dir)
        return self.fc(out[:, -1, :]).squeeze(-1)


class CNNBiLSTMAttn(nn.Module):
    """Conv1d front-end -> BiLSTM -> additive (softmax) attention over time -> FC."""

    def __init__(self, n_features, window_len, filters=16, hidden=64, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(n_features, filters, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.bilstm = nn.LSTM(filters, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        z = self.act(self.conv(x.transpose(1, 2)))   # (N, filters, L)
        out, _ = self.bilstm(z.transpose(1, 2))      # (N, L, 2H)
        w = torch.softmax(self.attn(out).squeeze(-1), dim=1)   # (N, L)
        ctx = (out * w.unsqueeze(-1)).sum(dim=1)      # (N, 2H)
        return self.fc(self.drop(ctx)).squeeze(-1)


class _Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation                # causal padding (left), chomp right
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)
        self.chomp = _Chomp1d(pad)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        y = self.drop(self.relu(self.chomp(self.conv1(x))))
        y = self.drop(self.relu(self.chomp(self.conv2(y))))
        res = x if self.down is None else self.down(x)
        return self.relu(y + res)


class TCNNet(nn.Module):
    """Causal dilated TCN, residual blocks with base-2 dilation; last step -> FC."""

    def __init__(self, n_features, window_len, channels=32, levels=3, kernel=3, dropout=0.1):
        super().__init__()
        blocks, c_in = [], n_features
        for i in range(levels):
            blocks.append(_TCNBlock(c_in, channels, kernel, 2 ** i, dropout))
            c_in = channels
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        z = self.tcn(x.transpose(1, 2))              # (N, C, L)
        return self.fc(z[:, :, -1]).squeeze(-1)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerNet(nn.Module):
    """Encoder-only transformer, sinusoidal pos-enc, mean-pool over time -> FC."""

    def __init__(self, n_features, window_len, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.encoder(self.pos(self.proj(x)))     # (N, L, d_model)
        return self.fc(z.mean(dim=1)).squeeze(-1)


class PatchTSTNet(nn.Module):
    """Channel-independent patch tokenization + shared transformer encoder.

    Each channel's sequence is split into patches, embedded, and run through the
    SAME encoder independently (channel independence); per-channel summaries are
    concatenated for the regression head.
    """

    def __init__(self, n_features, window_len, patch_len=10, patch_stride=10,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.n_features = n_features
        n_patches = 1 + (window_len - patch_len) // patch_stride
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(n_features * d_model, 1)

    def forward(self, x):
        n = x.size(0)
        xc = x.transpose(1, 2)                                   # (N, F, L)
        patches = xc.unfold(-1, self.patch_len, self.patch_stride)  # (N, F, P, patch_len)
        p = patches.size(2)
        z = self.embed(patches) + self.pos[:, :p, :].unsqueeze(1)   # (N, F, P, d_model)
        z = z.reshape(n * self.n_features, p, -1)                # channel-independent
        z = self.encoder(z).mean(dim=1)                          # (N*F, d_model)
        z = z.reshape(n, -1)                                     # (N, F*d_model)
        return self.head(z).squeeze(-1)
