__package__ = "__model__"

import torch
import torch.nn as nn
import math

""" Positional Encoding """
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
""" Transformer model """
class MachineTranslation(nn.Module):
    def __init__(
            self, vocab_size,
            d_model=768,
            num_heads=8, 
            num_encoder_layers=6, 
            num_decoder_layers=6, 
            dim_feedforward=1024, 
            dropout=0.1, 
            device=torch.device('cpu')
        ):

        super(MachineTranslation, self).__init__()

        # Embedding and positional encoding
        self.position_encoding = PositionalEncoding(d_model, max_len=100, dropout=0.1)    
        self.embedding = nn.Embedding(vocab_size, d_model)
        
         # Attributes
        self.device = device
        self.d_model = d_model
        
        # Transformer 
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output linear layers
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_key_padding_mask = (src == 1).to(self.device)
        tgt_key_padding_mask = (tgt == 1).to(self.device)

        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device
        )
        
        # Embed and add positional encoding (with scaling)
        src_embedded = self.position_encoding(
            self.embedding(src) * math.sqrt(self.d_model)
        )
        tgt_embedded = self.position_encoding(
            self.embedding(tgt) * math.sqrt(self.d_model)
        )

        """
        Forward part
        src = torch.rand(32, 10, d_model)   # (batch_size, src_seq_len, d_model)
        tgt = torch.rand(32, 20, d_model)   # (batch_size, tgt_seq_len, d_model)
        
        # Optional: masks (if needed for causal attention)
        src_mask = Nones
        tgt_mask = transformer.generate_square_subsequent_mask(tgt.size(1))
        
        out = transformer(src, tgt, tgt_mask=tgt_mask) # (b, tgt_len, d_model)
        """

        output = self.transformer(
            src_embedded, tgt_embedded, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.fc_out(output)