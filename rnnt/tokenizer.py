import pandas as pd


class SequenceTokenizer:
    def __init__(
        self,
        transcript_path,
        batch_size,
        target_column="transcript",
        mode="char",
    ):
        self.special_tokens = {
            "pad": "<pad>",
            "blank": "<blank>",
        }
        self.batch_size = batch_size
        self.target_column = target_column
        self.mode = mode

        self.vocab = self.get_vocab(transcript_path)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        if mode == "char":
            self.decode_separator = ""
        elif mode == "token":
            self.decode_separator = " "
        else:
            raise ValueError(f"Unsupported tokenizer mode: {mode}")

    def get_vocab(self, transcript_path):
        df = pd.read_csv(transcript_path)
        if self.target_column not in df:
            raise ValueError(
                f"{transcript_path} must contain a `{self.target_column}` column."
            )
        vocab = sorted(
            {
                token
                for value in df[self.target_column].dropna()
                for token in self.tokenize(str(value))
            }
        )
        vocab = [self.special_tokens["pad"]] + vocab + [self.special_tokens["blank"]]
        return vocab

    def tokenize(self, value):
        if not value:
            return []
        if self.mode == "char":
            return list(value)
        if self.mode == "token":
            return value.split()
        raise ValueError(f"Unsupported tokenizer mode: {self.mode}")

    def decode(self, ids, special_tokens=None):
        special_tokens = special_tokens or set()
        return self.decode_separator.join(
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


class CharTokenizer(SequenceTokenizer):
    def __init__(self, transcript_path, batch_size):
        super().__init__(
            transcript_path=transcript_path,
            batch_size=batch_size,
            target_column="transcript",
            mode="char",
        )
