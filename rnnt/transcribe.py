import csv
import os
from typing import Iterable

import torch

from data import build_data_loader


def ids_to_sentence(ids: Iterable[int], tokenizer, special_tokens: set[int]):
    return "".join(
        tokenizer.itos[int(token)]
        for token in ids
        if int(token) not in special_tokens
    )


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
):
    special_tokens = {
        tokenizer.stoi[tokenizer.special_tokens["pad"]],
        tokenizer.stoi[tokenizer.special_tokens["blank"]],
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
            for inputs, inputs_length, targets, targets_length in data_loader:
                inputs = inputs.to(device)
                inputs_length = inputs_length.to(device)

                predictions = model.recognize(inputs, inputs_length)

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


def export_all_transcriptions(model, config, tokenizer, device, logger=None):
    split_configs = [
        ("train", config.data.core_train, config.data.transcriptions_train),
        ("val", config.data.core_val, config.data.transcriptions_val),
        ("test", config.data.core_test, config.data.transcriptions_test),
    ]

    for split_name, split_path, output_path in split_configs:
        output_file = os.path.join(config.data.name, output_path)
        data_loader = build_transcription_loader(config, split_path, tokenizer)
        num_samples = export_split_transcriptions(
            model=model,
            data_loader=data_loader,
            tokenizer=tokenizer,
            output_file=output_file,
            device=device,
        )
        if logger is not None:
            logger.info(
                f"Wrote {num_samples} {split_name} transcriptions to {output_file}."
            )
