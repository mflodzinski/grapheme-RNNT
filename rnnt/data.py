import os

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TimitDataset(Dataset):
    def __init__(self, transcript_path, tokenizer):
        self.df = pd.read_csv(transcript_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        features = self.load_features(row)
        targets = torch.tensor(
            self.tokenizer.tokens2ids(str(row["transcript"])),
            dtype=torch.int32,
        )

        return (
            features,
            torch.tensor(features.size(0), dtype=torch.int32),
            targets,
            torch.tensor(targets.numel(), dtype=torch.int32),
        )

    @staticmethod
    def load_features(row):
        feature_path = row.get("feature_path")
        if feature_path is None or not isinstance(feature_path, str):
            feature_path = f"{os.path.splitext(row['audio_path'])[0]}.npy"
        features = torch.as_tensor(np.load(feature_path), dtype=torch.float32)

        if features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.dim() != 2:
            raise ValueError(
                f"Expected feature shape [T, F] or [1, T, F], got {tuple(features.shape)}"
            )

        return features


class TransducerCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        features, feature_lengths, targets, target_lengths = zip(*batch)
        return (
            pad_sequence(features, batch_first=True),
            torch.stack(feature_lengths),
            pad_sequence(targets, batch_first=True, padding_value=self.pad_idx),
            torch.stack(target_lengths),
        )


def build_data_loader(transcript_path, tokenizer, batch_size, shuffle=False):
    dataset = TimitDataset(transcript_path, tokenizer)
    pad_idx = tokenizer.stoi[tokenizer.special_tokens["pad"]]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=TransducerCollate(pad_idx),
    )
