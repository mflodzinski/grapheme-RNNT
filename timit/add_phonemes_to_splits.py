from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


DEFAULT_SPLITS = (
    "splits/core_train.csv",
    "splits/core_val.csv",
    "splits/core_test.csv",
)


def phoneme_path_for_audio(audio_path: str) -> Path:
    path = Path(audio_path)
    if path.suffix.upper() != ".WAV":
        raise ValueError(f"Expected a .WAV audio path, got {audio_path}")
    return path.with_suffix(".PHN")


def read_phonemes(path: Path) -> str:
    phones = []
    with path.open() as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 3:
                raise ValueError(f"Malformed PHN line in {path}: {line.rstrip()}")
            phones.append(parts[2])
    if not phones:
        raise ValueError(f"No phones found in {path}")
    return " ".join(phones)


def add_phoneme_column(timit_root: Path, split_path: str, column: str):
    csv_path = timit_root / split_path
    df = pd.read_csv(csv_path)
    if "audio_path" not in df:
        raise ValueError(f"{csv_path} must contain an `audio_path` column.")

    df[column] = [
        read_phonemes(phoneme_path_for_audio(audio_path))
        for audio_path in df["audio_path"]
    ]
    df.to_csv(csv_path, index=False)
    return csv_path, len(df)


def main():
    parser = ArgumentParser(
        description="Add raw TIMIT phoneme sequences from .PHN files to split CSVs."
    )
    parser.add_argument("--timit-root", default="timit", type=Path)
    parser.add_argument("--column", default="phonemes")
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    args = parser.parse_args()

    for split_path in args.splits:
        csv_path, rows = add_phoneme_column(
            timit_root=args.timit_root,
            split_path=split_path,
            column=args.column,
        )
        print(f"Wrote {rows} rows with `{args.column}` to {csv_path}")


if __name__ == "__main__":
    main()
