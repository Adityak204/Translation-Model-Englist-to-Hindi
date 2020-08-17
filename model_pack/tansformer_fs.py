import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        """ Dividing word's embedding into 'H' different heads
            For ex: embed_size = 512 & heads = 8
            Then 8 heads of 64 size are created
        """
        assert (self.embed_size % self.heads == 0), "Embed size should be in multiple of heads"

        self.values = nn.Linear(in_features=self.head_dim, out_features=self.head_dim, bias=False)
        self.keys = nn.Linear(in_features=self.head_dim, out_features=self.head_dim, bias=False)
        self.queries = nn.Linear(in_features=self.head_dim, out_features=self.head_dim, bias=False)
        self.fc_out = nn.Linear(in_features=self.head_dim*self.heads, out_features=self.embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # Number of training examples
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Splitting embeddings into 'H' heads for creating multi-head attention
        # V, K, Q reshape = num_samp, seq_len, heads, heads_dim
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Attention calculation
        # attention score = softmax(Q*t(K))/sqrt(Q.shape[-1])
        score = torch.einsum("nqhd,nkhd -> nhqk", queries, keys)
        """
        einsum explained: "nqhd,nkhd -> nhqk"
        1. nqhd -> nhqd : queries.transpose(-2,-3) & nkhd -> nhkd : keys.transpose(-2,-3)
        2. nhqk : (torch.bmm(nhqd.view(n*h,q,d), nhkd.view(n*h,k,d).transpose(-1,-2)).view(n,h,q,k)  
        """

        if mask is not None:
            """
            Masking is very critical for implementing decoder side self attention
            Since in decoding side we want to have attention scores with previous time steps elements only
            So for this we use upper triangular masked matrix 
            """
            score = score.masked_fill(mask == 0, float('-1e20'))
        attention_score = torch.softmax(score / math.sqrt(self.head_dim), dim=-1)  # N, heads, query_len, key_len
        out = torch.einsum("nhql,nlhd->nqhd", attention_score, values).view(N, query_len, self.heads*self.head_dim)
        # out.shape >> N, query_len, embed_size

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)  # Normalization for each example for each embed dim across seq_len
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_fwd = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))  # Layernorm1 + Skip connection
        forward = self.feed_fwd(x)
        out = self.dropout(self.norm2(forward + x))  # Layernorm2 + Skip connection
        return out







