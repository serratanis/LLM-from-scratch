import torch
import torch.nn as nn


from .usta_decoder_block import UstaDecoderBlock
from .usta_embedding import UstaEmbedding


class UstaModel(nn.Module):
    def __init__(self ,vocab_size , embedding_dim, num_heads,context_length,num_layers):
        super().__init__()

        self.embedding = UstaEmbedding(vocab_size,embedding_dim ,context_length)
        self.layers = nn.Sequential(*[UstaDecoderBlock(embedding_dim,num_heads,context_length) for _ in range(num_layers)])
        self.lm_head = nn.Linear(embedding_dim,vocab_size)

    def forward(self,x : torch.Tensor):

        x = self.embedding(x) #dictionary meaning of the tokens (words)
        x = self.layers(x)
        x = self.lm_head(x)


        return x 

    def generate(self, x: torch.Tensor, max_new_tokens: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        tokens = x.squeeze(0).tolist()

        for _ in range(max_new_tokens):
            out = self.forward(x)
            logits = out[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            max_prob, max_index = torch.max(probs, dim=-1)
            next_token = max_index.unsqueeze(1)

            x = torch.cat((x, next_token), dim=1)

            tokens.append(max_index.item())

            if max_index.item() == 61 or len(tokens) > 32:
                break

        return tokens