import torch
import torch.nn as nn
from model_pack.encoder_decoder import Encoder, Decoder

"""
Input shape for Transformer block = (num_sample, seq_len)  
Out shape of Transformer block =  (num_sample, seq_len, target_vocab_size)
"""

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size,
                 num_layers,
                 forward_expansion,
                 heads,
                 dropout,
                 device="cuda",
                 max_len=500):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size=src_vocab_size,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               heads=heads,
                               device=device,
                               forward_expansion=forward_expansion,
                               dropout=dropout,
                               max_len=max_len)

        self.decoder = Decoder(trg_vocab_size=trg_vocab_size,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               heads=heads,
                               device=device,
                               forward_expansion=forward_expansion,
                               dropout=dropout,
                               max_len=max_len)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len)))
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src=src)
        trg_mask = self.make_trg_mask(trg=trg)
        enc_src = self.encoder(x=src, mask=src_mask)
        out = self.decoder(x=trg, enc_out=enc_src, trg_mask=trg_mask, src_mask=src_mask)
        return out
