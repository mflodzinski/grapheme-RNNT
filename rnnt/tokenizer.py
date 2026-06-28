import pandas as pd


class CharTokenizer:
    def __init__(self, transcript_path, batch_size):
        self.special_tokens = {
            "pad": "<pad>",
            "blank": "<blank>",
        }
        self.batch_size = batch_size

        self.vocab = self.get_vocab(transcript_path)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def get_vocab(self, transcript_path):
        df = pd.read_csv(transcript_path)
        all_txt = df["transcript"].str.cat(sep="")
        vocab = sorted(list(set(all_txt)))
        vocab = [self.special_tokens["pad"]] + vocab + [self.special_tokens["blank"]]
        return vocab

    def ids2tokens(self, ids):
        if not ids:
            return []

        if isinstance(ids[0], list): 
            return [[self.itos[i] for i in sublist] for sublist in ids]
        else:
            return [self.itos[i] for i in ids]

    def tokens2ids(self, tokens):
        if not tokens:
            return []

        if isinstance(tokens[0], list): 
            return [[self.stoi[s] for s in sublist] for sublist in tokens]
        else:
            return [self.stoi[s] for s in tokens]
