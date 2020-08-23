import torch
import torch.nn as nn
from model_pack.buildingblocks import SelfAttention, TransformerBlock, PositionalEncoding, DecoderBlock

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_len):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(max_len, embed_size, dropout, device=self.device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size=embed_size,
                                 heads=heads,
                                 dropout=dropout,
                                 forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.word_embedding(x)
        out = self.position_embedding(x)

        for layer in self.layers:
            out = layer(value=out, key=out, query=out, mask=mask)
        return out


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_len):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(max_len, embed_size, dropout, device=self.device)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size=embed_size,
                             heads=heads,
                             dropout=dropout,
                             forward_expansion=forward_expansion,
                             device=self.device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

    def forward(self, x, enc_out, trg_mask, src_mask=None):
        x = self.word_embedding(x)
        x = self.position_embedding(x)

        """
        In decoder part key & value comes from the encoder output 
        while query comes from the self attention layer's output of the decoder         
        """
        for layer in self.layers:
            x = layer(x=x, value=enc_out, key=enc_out, trg_mask=trg_mask, src_mask=src_mask)

        dec_out = self.fc_out(x)
        return dec_out
