import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn



class Prompt_Generation_Model(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.filternum=10

        self.attention = ScaledDotProductAttention(temperature=1)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_v)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, _, dimen_k = k.size()
        sz_b, _, dimen_v = v.size()

        attn = torch.bmm(q, k.transpose(1, 2))

        f_k=torch.zeros(sz_b,self.filternum,dimen_k).to(q.device)
        f_v=torch.zeros(sz_b,self.filternum,dimen_v).to(q.device)

        for i in range(sz_b):
            _, s_index=torch.sort(attn[i,0,:],descending=True)
            f_k[i,:,:]=k[i,s_index[:self.filternum],:]
            f_v[i,:,:]=v[i,s_index[:self.filternum],:]

        len_k=self.filternum
        len_v=self.filternum

        q=q.view(sz_b, len_q, n_head, d_k)
        k=f_k.view(sz_b, len_k, n_head, d_k)
        v=f_v.view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        return output

