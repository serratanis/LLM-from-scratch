import torch
import torch.nn as nn

from usta_layer_norm import UstaLayerNorm
from usta_decoder_block import UstaDecoderBlock
from usta_embedding import UstaEmbedding


class UstaModel(nn.Module):
    def __init__(self ,vocab_size , embedding_dim, num_heads,context_length,num_layers):
        super().__init__()

        self.embedding = UstaEmbedding(vocab_size,embedding_dim ,context_length)
        self.layers = nn.Sequential(*[UstaDecoderBlock(embedding_dim,num_heads,context_length) for _ in range(num_layers)])
        self.lm_head = nn.Linear(embedding_dim,vocab_size)
    def forward(self,x):

        x = self.embedding(x) #dictionary meaning of the tokens (words)
        x = self.get_pos(x) # meaning of the tokens in the sentence according to their positions 
        x = self.layers(x)
        x = self.lm_head(x)


        return x 

