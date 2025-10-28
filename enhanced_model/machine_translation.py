import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

""" Positional Encoding """
class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:x.size(1), :].detach()

""" Machine Translation Model using pretrained Transformers """
class MachineTranslationModel(nn.Module):
    def __init__(
            self, src_vocab_size: int, tgt_vocab_size: int, 
            d_model: int=512, ff_dim: int=1024, 
            num_heads: int=8, num_decoder_layers: int=6, 
            dropout: float=0.1, device: str='cuda'
    ):
        super(MachineTranslationModel, self).__init__()
        self.d_model = d_model
        self.device = device

        self.encoder = AutoModelForCausalLM.from_pretrained("bert-base-cased")
        # Freeze 2/3 layers of the encoder
        for i, param in enumerate(self.encoder.parameters()):
            if i < len(self.encoder.parameters()) * 2 // 3:
                param.requires_grad = False

        # Layer dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Src and tgt embeddings
        self.src_positional_encoding = PositionEncoding(d_model, max_len=512)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_positional_encoding = PositionEncoding(d_model, max_len=512)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Decoder for generation
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, ff_dim, dropout), num_decoder_layers
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = self.src_embedding(src) * (self.d_model ** 0.5)
        src = self.src_positional_encoding(src)
        src = self.dropout(src)

        tgt = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        tgt = self.tgt_positional_encoding(tgt)
        tgt = self.dropout(tgt)

        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output
