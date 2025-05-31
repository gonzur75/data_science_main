from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def convert_json_to_parquet(filename: Path, output_name: str | None = None) -> None:
    """Load line delimited JSON file and save as parquet file."""
    df = pd.read_json(filename, lines=True)

    if output_name is None:
        output_name = filename.name.replace(".json", ".parquet")
    df.to_parquet(PROCESSED_DATA_DIR / output_name)


def convert_csv_to_parquet(filename: Path, output_name: str | None = None) -> None:
    """Load line delimited JSON file and save as parquet file."""
    df = pd.read_csv(filename)

    if output_name is None:
        output_name = filename.name.replace(".csv", ".parquet")
    df.to_parquet(PROCESSED_DATA_DIR / output_name)

if __name__ == "__main__":
    convert_json_to_parquet(RAW_DATA_DIR / "transactions.json")
    convert_csv_to_parquet(RAW_DATA_DIR / "users.csv")
    convert_csv_to_parquet(RAW_DATA_DIR / "merchants.csv")