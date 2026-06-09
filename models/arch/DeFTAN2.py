import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from packaging.version import parse as V
from torch.nn import init
from torch.nn.parameter import Parameter

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class DeFTAN2(nn.Module):
    def __init__(self, n_srcs=1, win=512, n_mics=4, n_layers=6, att_dim=64, hidden_dim=256, n_head=4, emb_dim=64, emb_ks=4, emb_hs=1, dropout=0.1, eps=1.0e-5):
        super().__init__()
        self.n_srcs = n_srcs
        self.win = win
        self.hop = win // 2
        self.n_layers = n_layers
        self.n_mics = n_mics
        self.emb_dim = emb_dim
        assert win % 2 == 0

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.up_conv = nn.Sequential(
            nn.Conv2d(2 * n_mics, emb_dim * n_head, ks, padding=padding),
            nn.GroupNorm(1, emb_dim * n_head, eps=eps),
            SDB2d(emb_dim * n_head, emb_dim, n_head)
        )
        self.blocks = nn.ModuleList([])
        for idx in range(n_layers):
            self.blocks.append(DeFTAN2block(idx, emb_dim, emb_ks, emb_hs, att_dim, hidden_dim, n_head, dropout, eps))
        self.down_conv = nn.Sequential(
            nn.Conv2d(emb_dim, 2 * n_srcs * n_head, ks, padding=padding),
            SDB2d(2 * n_srcs * n_head, 2 * n_srcs, n_head))

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)
        rest = self.win - (self.hop + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.hop)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, input: Union[torch.Tensor]) -> Tuple[List[Union[torch.Tensor]], torch.Tensor, OrderedDict]:
        input, rest = self.pad_signal(input)
        B, M, N = input.size()                                                      # batch B, mic M, time samples N
        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization
        # Encoding
        stft_input = torch.stft(input.view([-1, N]), n_fft=self.win, hop_length=self.hop, window=torch.hann_window(self.win).type(input.type()), return_complex=False)
        _, F, T, _ = stft_input.size()                                              # B*M , F= num freqs, T= num frame, 2= real imag
        xi = stft_input.view([B, M, F, T, 2])                                       # B*M, F, T, 2 -> B, M, F, T, 2
        xi = xi.permute(0, 1, 4, 3, 2).contiguous()                                 # [B, M, 2, T, F]
        xi = xi.view([B, M * 2, T, F])                                              # [B, 2*M, T, F]
        # Separation
        feature = self.up_conv(xi)                                                    # [B, C, T, F]
        for ii in range(self.n_layers):
            feature = self.blocks[ii](feature)                                          # [B, C, T, F]
        xo = self.down_conv(feature).view([B, self.n_srcs, 2, T, F]).view([B * self.n_srcs, 2, T, F])
        # Decoding
        xo = xo.permute(0, 3, 2, 1).type(input.type())                        # [B*n_srcs, 2, T, F] -> [B*n_srcs, F, T, 2]
        istft_input = torch.complex(xo[:, :, :, 0], xo[:, :, :, 1])
        istft_output = torch.istft(istft_input, n_fft=self.win, hop_length=self.hop, window=torch.hann_window(self.win).type(input.type()), return_complex=False)

        output = istft_output[:, self.hop:-(rest + self.hop)].unsqueeze(1)          # [B*n_srcs, 1, N]
        output = output.view([B, self.n_srcs, -1])                                  # [B, n_srcs, N]
        output = output * mix_std_  # reverse the RMS normalization

        return output


class SDB1d(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        assert in_channels // out_channels == groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.blocks = nn.ModuleList([])
        for idx in range(groups):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(out_channels * ((idx > 0) + 1), out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, out_channels, 1e-5),
                nn.PReLU(out_channels)
            ))

    def forward(self, x):
        B, C, L = x.size()
        g = self.groups
        # x = x.view(B, g, C//g, L).transpose(1, 2).reshape(B, C, L)
        skip = x[:, ::g, :]
        for idx in range(g):
            output = self.blocks[idx](skip)
            skip = torch.cat([output, x[:, idx+1::g, :]], dim=1)
        return output


class SDB2d(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        assert in_channels // out_channels == groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.blocks = nn.ModuleList([])
        for idx in range(groups):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(out_channels * ((idx > 0) + 1), out_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.GroupNorm(1, out_channels, 1e-5),
                nn.PReLU(out_channels)
            ))

    def forward(self, x):
        B, C, T, Q = x.size()
        g = self.groups
        # x = x.view(B, g, C//g, T, Q).transpose(1, 2).reshape(B, C, T, Q)
        skip = x[:, ::g, :, :]
        for idx in range(g):
            output = self.blocks[idx](skip)
            skip = torch.cat([output, x[:, idx+1::g, :, :]], dim=1)
        return output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class CEA(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.cv_qk = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, bias = False),
            nn.GLU(dim=1))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.att_drop = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qk = self.cv_qk(x.transpose(1, 2)).transpose(1, 2)
        q = rearrange(self.to_q(qk), 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(self.to_k(qk), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h = self.heads)

        weight = torch.matmul(F.softmax(k, dim=2).transpose(-1, -2), v) * self.scale
        out = torch.matmul(F.softmax(q, dim=3), self.att_drop(weight))
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class DPFN(nn.Module):
    def __init__(self, dim, hidden_dim, idx, dropout):
        super().__init__()
        self.PW1 = nn.Sequential(
            nn.Linear(dim, hidden_dim//2),
            nn.GELU(),
	        nn.Dropout(dropout)
        )
        self.PW2 = nn.Sequential(
            nn.Linear(dim, hidden_dim//2),
            nn.GELU(),
	        nn.Dropout(dropout)
        )
        self.DW_Conv = nn.Sequential(
            nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=5, dilation=2**idx, padding='same'),
            nn.GroupNorm(1, hidden_dim//2, 1e-5),
            nn.PReLU(hidden_dim//2)
        )
        self.PW3 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        ffw_out = self.PW1(x)
        dw_out = self.DW_Conv(self.PW2(x).transpose(1, 2)).transpose(1, 2)
        out = self.PW3(torch.cat((ffw_out, dw_out), dim=2))
        return out


class DeFTAN2block(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, idx, emb_dim, emb_ks, emb_hs, att_dim, hidden_dim, n_head, dropout, eps):
        super().__init__()
        in_channels = emb_dim * emb_ks
        self.F_norm = LayerNormalization4D(emb_dim, eps)
        self.F_inv = SDB1d(in_channels, emb_dim, emb_ks)
        self.F_att = PreNorm(emb_dim, CEA(emb_dim, n_head, att_dim, dropout))
        self.F_ffw = PreNorm(emb_dim, DPFN(emb_dim, hidden_dim, idx, dropout))
        self.F_linear = nn.ConvTranspose1d(emb_dim, emb_dim, emb_ks, stride=emb_hs)

        self.T_norm = LayerNormalization4D(emb_dim, eps)
        self.T_inv = SDB1d(in_channels, emb_dim, emb_ks)
        self.T_att = PreNorm(emb_dim, CEA(emb_dim, n_head, att_dim, dropout))
        self.T_ffw = PreNorm(emb_dim, DPFN(emb_dim, hidden_dim, idx, dropout))
        self.T_linear = nn.ConvTranspose1d(emb_dim, emb_dim, emb_ks, stride=emb_hs)

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # F-transformer
        input_ = x
        F_feat = self.F_norm(input_)  # [B, C, T, Q]
        F_feat = F_feat.transpose(1, 2).contiguous().view(B * T, C, Q)  # [BT, C, Q]
        F_feat = F.unfold(F_feat[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # [BT, C*emb_ks, -1]
        F_feat = self.F_inv(F_feat)   # [BT, C, -1]

        F_feat = F_feat.transpose(1, 2)  # [BT, -1, C]
        F_feat = self.F_att(F_feat) + F_feat
        F_feat = self.F_ffw(F_feat) + F_feat
        F_feat = F_feat.transpose(1, 2)  # [BT, H, -1]

        F_feat = self.F_linear(F_feat)  # [BT, C, Q]
        F_feat = F_feat.view([B, T, C, Q])
        F_feat = F_feat.transpose(1, 2).contiguous()  # [B, C, T, Q]
        F_feat = F_feat + input_  # [B, C, T, Q]

        # T-transformer
        input_ = F_feat
        T_feat = self.T_norm(input_)  # [B, C, T, F]
        T_feat = T_feat.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)  # [BF, C, T]
        T_feat = F.unfold(T_feat[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # [BF, C*emb_ks, -1]
        T_feat = self.T_inv(T_feat)   # [BF, C, -1]

        T_feat = T_feat.transpose(1, 2)  # [BF, -1, C]
        T_feat = self.T_att(T_feat) + T_feat
        T_feat = self.T_ffw(T_feat) + T_feat
        T_feat = T_feat.transpose(1, 2)  # [BF, H, -1]

        T_feat = self.T_linear(T_feat)  # [BF, C, T]
        T_feat = T_feat.view([B, Q, C, T])
        T_feat = T_feat.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        T_feat = T_feat + input_  # [B, C, T, Q]

        return T_feat


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat