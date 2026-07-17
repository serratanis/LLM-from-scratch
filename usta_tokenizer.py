import json
import torch 

from usta_self_attention import UstaSelfAttention
class UstaTokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)["vocab"]
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        text = text.lower()
        tokens = []

        for word in text.split():
            i = 0

            while i < len(word):
                found_match = False

                for j in range(len(word), i, -1):
                    sub_word = word[i:j]

                    if sub_word in self.vocab:
                        tokens.append(self.vocab[sub_word])
                        i = j
                        found_match = True
                        break

                if not found_match:
                    tokens.append(self.vocab["<unk>"])
                    i += 1

            tokens.append(self.vocab[" "])

        if tokens:
            tokens.pop()

        return torch.tensor(tokens)

    def tokenize(self, text):
        token_ids = self.encode(text)
        token_ids = token_ids.detach().numpy().tolist()
        return [self.reverse_vocab[id] for id in token_ids]

    def decode(self, ids):
        text = ""

        for id in ids:
            text += self.reverse_vocab.get(id.item(), "<unk>")

        return text