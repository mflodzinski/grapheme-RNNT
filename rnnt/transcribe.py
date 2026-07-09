import csv
import os
from typing import Iterable

import torch

from data import build_data_loader

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def ids_to_sentence(ids: Iterable[int], tokenizer, special_tokens: set[int]):
    return tokenizer.decode(ids, special_tokens=special_tokens)


def build_transcription_loader(config, split_path, tokenizer):
    return build_data_loader(
        os.path.join(config.data.name, split_path),
        tokenizer,
        batch_size=config.training.batch_size,
        shuffle=False,
        bucket_by_duration=False,
        sort_by_duration=False,
    )


def export_split_transcriptions(
    model,
    data_loader,
    tokenizer,
    output_file,
    device,
    decode_method=None,
    split_name=None,
):
    special_tokens = {
        tokenizer.stoi[token]
        for token in tokenizer.special_tokens.values()
    }
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    model.eval()
    sample_id = 0

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["sample_id", "prediction", "target"])

        with torch.no_grad():
            progress = tqdm(
                data_loader,
                total=len(data_loader),
                desc=f"{split_name or 'split'} {decode_method or 'decode'}",
                unit="batch",
                dynamic_ncols=True,
            )
            for inputs, inputs_length, targets, targets_length in progress:
                inputs = inputs.to(device)
                inputs_length = inputs_length.to(device)

                predictions = model.recognize(
                    inputs,
                    inputs_length,
                    decode_method=decode_method,
                )

                for prediction, target, target_length in zip(
                    predictions, targets, targets_length
                ):
                    target = target[: target_length.item()].tolist()
                    writer.writerow(
                        [
                            sample_id,
                            ids_to_sentence(prediction, tokenizer, special_tokens),
                            ids_to_sentence(target, tokenizer, special_tokens),
                        ]
                    )
                    sample_id += 1

    return sample_id


def configured_export_methods(config):
    methods = config.training.export_decode_methods
    if methods is None:
        return ["greedy", "beam"]
    return [str(method).lower() for method in methods]


def configured_export_splits(config):
    split_configs = {
        "train": config.data.core_train,
        "val": config.data.core_val,
        "test": config.data.core_test,
    }
    splits = config.training.export_splits
    if splits is None:
        return list(split_configs.items())
    requested_splits = [str(split).lower() for split in splits]
    unknown_splits = [
        split for split in requested_splits if split not in split_configs
    ]
    if unknown_splits:
        raise ValueError(
            "Unsupported export split(s): "
            f"{', '.join(unknown_splits)}. Expected train, val, or test."
        )
    return [(split, split_configs[split]) for split in requested_splits]


def transcription_output_file(config, split_name, decode_method):
    output_dir = config.data.transcriptions_dir
    if output_dir is None:
        transcriptions_val = config.data.transcriptions_val
        output_dir = (
            os.path.dirname(transcriptions_val)
            if transcriptions_val is not None
            else "outputs"
        )
    output_path = os.path.join(
        output_dir,
        f"transcriptions_{split_name}_{decode_method}.tsv",
    )
    return os.path.join(config.data.name, output_path)


def export_all_transcriptions(model, config, tokenizer, device, logger=None):
    split_configs = configured_export_splits(config)
    decode_methods = configured_export_methods(config)

    for split_name, split_path in split_configs:
        data_loader = build_transcription_loader(config, split_path, tokenizer)
        for decode_method in decode_methods:
            output_file = transcription_output_file(
                config,
                split_name,
                decode_method,
            )
            num_samples = export_split_transcriptions(
                model=model,
                data_loader=data_loader,
                tokenizer=tokenizer,
                output_file=output_file,
                device=device,
                decode_method=decode_method,
                split_name=split_name,
            )
            if logger is not None:
                logger.info(
                    f"Wrote {num_samples} {split_name} {decode_method} "
                    f"transcriptions to {output_file}."
                )
