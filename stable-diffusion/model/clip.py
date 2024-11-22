import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embeddings: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embeddings)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embeddings))

    def forward(self, tokens):
        # bs, seq_len -> bs, seq_len, dim
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, n_embeddings: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embeddings)
        self.attention = SelfAttention(n_heads, n_embeddings)
        self.layernorm_2 = nn.LayerNorm(n_embeddings)
        self.linear_1 = nn.Linear(n_embeddings, n_embeddings * 4)
        self.linear_2 = nn.Linear(n_embeddings * 4, n_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bs, seq_len, dim
        residue = x

        # Self-Attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        # Feedforward Layer
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) # QuickGELU activation function
        x = self.linear_2(x)
        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self):
        self.embeddings = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        # bs, seq_len -> bs, seq_len, dim
        state = self.embeddings(tokens)
        for layer in self.layers:
            state = layer(state)
        # bs, seq_len,dim
        output = self.layernorm(state)

        return output