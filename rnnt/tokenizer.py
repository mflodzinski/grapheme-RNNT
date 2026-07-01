import pandas as pd


TIMIT_61_TO_39 = {
    "aa": "aa",
    "ae": "ae",
    "ah": "ah",
    "ao": "aa",
    "aw": "aw",
    "ax": "ah",
    "ax-h": "ah",
    "axr": "er",
    "ay": "ay",
    "b": "b",
    "bcl": "sil",
    "ch": "ch",
    "d": "d",
    "dcl": "sil",
    "dh": "dh",
    "dx": "dx",
    "eh": "eh",
    "el": "l",
    "em": "m",
    "en": "n",
    "eng": "ng",
    "epi": "sil",
    "er": "er",
    "ey": "ey",
    "f": "f",
    "g": "g",
    "gcl": "sil",
    "h#": "sil",
    "hh": "hh",
    "hv": "hh",
    "ih": "ih",
    "ix": "ih",
    "iy": "iy",
    "jh": "jh",
    "k": "k",
    "kcl": "sil",
    "l": "l",
    "m": "m",
    "n": "n",
    "ng": "ng",
    "nx": "n",
    "ow": "ow",
    "oy": "oy",
    "p": "p",
    "pau": "sil",
    "pcl": "sil",
    "r": "r",
    "s": "s",
    "sh": "sh",
    "t": "t",
    "tcl": "sil",
    "th": "th",
    "uh": "uh",
    "uw": "uw",
    "ux": "uw",
    "v": "v",
    "w": "w",
    "y": "y",
    "z": "z",
    "zh": "sh",
}

TIMIT_IGNORED_PHONES = {"q"}


def normalize_timit_39(tokens):
    normalized = []
    for token in tokens:
        if token in TIMIT_IGNORED_PHONES:
            continue
        if token not in TIMIT_61_TO_39:
            raise ValueError(f"Unsupported TIMIT phone for 39-phone mapping: {token}")
        normalized.append(TIMIT_61_TO_39[token])
    return normalized


class SequenceTokenizer:
    def __init__(
        self,
        transcript_path,
        batch_size,
        target_column="transcript",
        mode="char",
        target_normalization=None,
    ):
        self.special_tokens = {
            "pad": "<pad>",
            "blank": "<blank>",
        }
        self.batch_size = batch_size
        self.target_column = target_column
        self.mode = mode
        self.target_normalization = target_normalization

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
            if self.target_normalization:
                raise ValueError("Target normalization is only supported for token mode.")
            return list(value)
        if self.mode == "token":
            tokens = value.split()
            if self.target_normalization is None:
                return tokens
            if self.target_normalization == "timit_39":
                return normalize_timit_39(tokens)
            raise ValueError(
                f"Unsupported target normalization: {self.target_normalization}"
            )
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
