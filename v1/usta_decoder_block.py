import torch
import torch.nn as nn

from .usta_multi_head_attention import UstaMultiHeadAttention
from .usta_layer_norm import UstaLayerNorm
from .usta_mlp import UstaMLP


class UstaDecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()

        self.self_attention = UstaMultiHeadAttention(
            embedding_dim,
            embedding_dim,
            context_length,
            num_heads,
            dropout_rate=0.5
        )
        self.norm1 = UstaLayerNorm(embedding_dim)
        self.mlp = UstaMLP(embedding_dim, embedding_dim)
        self.norm2 = UstaLayerNorm(embedding_dim)

    def forward(self, x):
        res = self.norm1(x)

       
        x = self.self_attention(x)

        x = self.norm1(x)

        x = x + res

        res = self.norm2(x)
        x = self.mlp(x)
        x = self.norm2(x)

        x = x + res

        return x


