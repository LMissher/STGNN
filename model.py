import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


class Transform(nn.Module):
    def __init__(self, outfea, d):
        super(Transform, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )

        self.d = d

    def forward(self, x):
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0,2,1,3)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,2,3,1)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0,2,1,3)

        A = torch.matmul(query, key)
        A /= (self.d ** 0.5)
        A = torch.softmax(A, -1)

        value = torch.matmul(A ,value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0,2,1,3)
        value += x

        value = self.ln(value)
        x = self.ff(value) + value
        return self.lnff(x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, outfea, max_len=12):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, outfea).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, outfea, 2) *
                             -(math.log(10000.0) / outfea))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2) #[1,T,1,F]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe, 
                         requires_grad=False)
        return x


class SGNN(nn.Module):
    def __init__(self, outfea):
        super(SGNN, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.Linear(outfea, outfea)
        )
        self.ff1 = nn.Linear(outfea, outfea)

    def forward(self, x):
        p = self.ff(x)
        a = torch.matmul(p, p.transpose(-1,-2))
        R = torch.relu(torch.softmax(a, -1)) + torch.eye(x.shape[1]).to(device)
        
        D = (R.sum(-1) ** -0.5)
        D[torch.isinf(D)] = 0.
        D = torch.diag_embed(D)

        A = torch.matmul(torch.matmul(D, R), D)
        x = torch.relu(self.ff1(torch.matmul(A, x)))
        return x
    
class GRU(nn.Module):
    def __init__(self, outfea):
        super(GRU, self).__init__()
        self.ff = nn.Linear(2*outfea, 2*outfea)
        self.zff = nn.Linear(2*outfea, outfea)
        self.outfea = outfea

    def forward(self, x, xh):
        r, u = torch.split(torch.sigmoid(self.ff(torch.cat([x, xh], -1))), self.outfea, -1)
        z = torch.tanh(self.zff(torch.cat([x, r*xh], -1)))
        x = u * z + (1-u) * xh
        return x


class STGNNwithGRU(nn.Module):
    def __init__(self, outfea):
        super(STGNNwithGRU, self).__init__()
        self.sgnnh = nn.ModuleList([SGNN(outfea) for i in range(12)])
        self.sgnnx = nn.ModuleList([SGNN(outfea) for i in range(12)])
        self.gru = nn.ModuleList([GRU(outfea) for i in range(12)])

    def forward(self, x):
        B,T,N,F = x.shape
        hidden_state = torch.zeros([B,N,F]).to(device)
        output = []

        for i in range(T):
            gx = self.sgnnx[i](x[:,i,:,:])
            gh = hidden_state
            if i != 0:
                gh = self.sgnnh[i](hidden_state)
            hidden_state = self.gru[i](gx, gh)
            output.append(hidden_state)

        output = torch.stack(output, 1)

        return output

class STGNN(nn.Module):
    def __init__(self, infea, outfea, L, d):
        super(STGNN, self).__init__()
        self.start_emb = nn.Linear(infea, outfea)
        self.end_emb = nn.Linear(outfea, infea)

        self.stgnnwithgru = nn.ModuleList([STGNNwithGRU(outfea) for i in range(L)])
        self.positional_encoding = PositionalEncoding(outfea)
        self.transform = nn.ModuleList([Transform(outfea, d) for i in range(L)])

        self.L = L

    def forward(self, x):
        '''
        x:[B,T,N]
        '''
        x = x.unsqueeze(-1)
        x = self.start_emb(x)
        for i in range(self.L):
            x = self.stgnnwithgru[i](x)
        x = self.positional_encoding(x)
        for i in range(self.L):
            x = self.transform[i](x)
        x = self.end_emb(x)

        return x.squeeze(-1)
