from time import time
import math
import torch
import torch.nn as nn
    
class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, max_len: int = 1000):
        ''' traditional vaswani PEs '''
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0 / embed_dim)))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe = pe.permute(1,0,2)[:, :512, :]
        # print(pe.shape)
        # self.temp = pe[:, :512, :]
        # print(self.temp.shape)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe
        return x

class qt(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layers: int,
            n_heads: int,
            n_heads_kv: int,
            seq_len: int,
            num_embeddings: int,
            device,
        ):
        super(qt, self).__init__()

        self.d_model = d_model
        self.ffw_size = 4*d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len

        # embeddings, tied!
        self.embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=d_model,
        )
        self.output_linear = nn.Linear(d_model, num_embeddings, bias=False)
        self.output_linear.weight = self.embeddings.weight

        # position encoding
        self.pe = PositionalEncoding(embed_dim=d_model).to(device)

        # causal mask
        self.mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

        # model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=self.ffw_size,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )
    
    def forward(self, x):

        x = self.embeddings(x)
        # print(f'x after embedding: {x.shape}')
        x = self.pe(x)
        # print(f'x after pe: {x.shape}')

        decoder_start = time()
        x = self.decoder(x, mask=self.mask)
        # print(f'x after decode: {x.shape}')
        print(f'decoder_body: {time()-decoder_start}')

        x = self.output_linear(x)
        # print(f'x after output layer: {x.shape}')
        x = x.transpose(1,2)
        return x





