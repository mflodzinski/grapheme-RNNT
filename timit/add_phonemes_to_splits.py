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


def add_phoneme_column(timit_root: Path, split_path: str, column: str, force: bool):
    csv_path = timit_root / split_path
    df = pd.read_csv(csv_path)
    if "audio_path" not in df:
        raise ValueError(f"{csv_path} must contain an `audio_path` column.")

    if not force and column in df and not df[column].isna().any():
        return csv_path, len(df), False

    df[column] = [
        read_phonemes(phoneme_path_for_audio(audio_path))
        for audio_path in df["audio_path"]
    ]
    df.to_csv(csv_path, index=False)
    return csv_path, len(df), True


def main():
    parser = ArgumentParser(
        description="Add raw TIMIT phoneme sequences from .PHN files to split CSVs."
    )
    parser.add_argument("--timit-root", default="timit", type=Path)
    parser.add_argument("--column", default="phonemes")
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the phoneme column even if it is already complete.",
    )
    args = parser.parse_args()

    for split_path in args.splits:
        csv_path, rows, changed = add_phoneme_column(
            timit_root=args.timit_root,
            split_path=split_path,
            column=args.column,
            force=args.force,
        )
        if changed:
            print(f"Wrote {rows} rows with `{args.column}` to {csv_path}")
        else:
            print(f"{csv_path} already has complete `{args.column}` targets")


if __name__ == "__main__":
    main()
