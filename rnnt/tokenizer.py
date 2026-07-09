import pandas as pd


class GraphemeTokenizer:
    def __init__(self, transcript_path, batch_size):
        self.batch_size = batch_size
        self.special_tokens = {}
        if self.batch_size > 1:
            self.special_tokens["pad"] = "<pad>"
        self.special_tokens["blank"] = "<blank>"

        self.vocab = self.get_vocab(transcript_path)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def get_vocab(self, transcript_path):
        df = pd.read_csv(transcript_path)
        if "transcript" not in df:
            raise ValueError(f"{transcript_path} must contain a `transcript` column.")
        vocab = sorted(
            {
                token
                for value in df["transcript"].dropna()
                for token in self.tokenize(str(value))
            }
        )
        if "pad" in self.special_tokens:
            vocab = [self.special_tokens["pad"]] + vocab
        vocab = vocab + [self.special_tokens["blank"]]
        return vocab

    def tokenize(self, value):
        if not value:
            return []
        return list(value)

    def decode(self, ids, special_tokens=None):
        special_tokens = special_tokens or set()
        return "".join(
            self.itos[int(token)]
            for token in ids
            if int(token) not in special_tokens
        )

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


class CharTokenizer(GraphemeTokenizer):
    def __init__(self, transcript_path, batch_size):
        super().__init__(
            transcript_path=transcript_path,
            batch_size=batch_size,
        )
