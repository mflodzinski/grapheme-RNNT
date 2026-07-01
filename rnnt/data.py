import os
import random

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset


class TimitDataset(Dataset):
    def __init__(self, transcript_path, tokenizer, target_column="transcript"):
        self.df = pd.read_csv(transcript_path)
        self.tokenizer = tokenizer
        self.target_column = target_column

        if self.target_column not in self.df:
            raise ValueError(
                f"{transcript_path} must contain a `{self.target_column}` column."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        features = self.load_features(row)
        targets = torch.tensor(
            self.tokenizer.tokens2ids(
                self.tokenizer.tokenize(str(row[self.target_column]))
            ),
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
    def __init__(self, target_fill_idx):
        self.target_fill_idx = target_fill_idx

    def __call__(self, batch):
        features, feature_lengths, targets, target_lengths = zip(*batch)
        return (
            pad_sequence(features, batch_first=True),
            torch.stack(feature_lengths),
            pad_sequence(targets, batch_first=True, padding_value=self.target_fill_idx),
            torch.stack(target_lengths),
        )


class DurationBatchSampler(BatchSampler):
    def __init__(
        self,
        durations,
        batch_size,
        bucket_size_multiplier=100,
        shuffle=False,
        drop_last=False,
    ):
        self.durations = list(durations)
        self.batch_size = batch_size
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.durations)))
        indices.sort(key=self.durations.__getitem__)

        if self.shuffle:
            buckets = [
                indices[start : start + self.bucket_size]
                for start in range(0, len(indices), self.bucket_size)
            ]
            for bucket in buckets:
                random.shuffle(bucket)
            random.shuffle(buckets)
            indices = [index for bucket in buckets for index in bucket]

        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        if self.drop_last:
            return len(self.durations) // self.batch_size
        return (len(self.durations) + self.batch_size - 1) // self.batch_size


def build_data_loader(
    transcript_path,
    tokenizer,
    batch_size,
    target_column="transcript",
    shuffle=False,
    bucket_by_duration=False,
    sort_by_duration=False,
    bucket_size_multiplier=100,
):
    dataset = TimitDataset(transcript_path, tokenizer, target_column=target_column)
    target_fill_token = tokenizer.special_tokens.get("pad", tokenizer.special_tokens["blank"])
    target_fill_idx = tokenizer.stoi[target_fill_token]

    if bucket_by_duration or sort_by_duration:
        if "duration" not in dataset.df:
            raise ValueError(
                f"{transcript_path} must contain a duration column for length-aware batching."
            )
        batch_sampler = DurationBatchSampler(
            dataset.df["duration"],
            batch_size=batch_size,
            bucket_size_multiplier=bucket_size_multiplier,
            shuffle=bucket_by_duration,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=TransducerCollate(target_fill_idx),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=TransducerCollate(target_fill_idx),
    )
