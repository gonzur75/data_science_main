from pathlib import Path
import pandas as pd
import pandera as pa

def load_validated_parquet(path: Path, schema: pa.DataFrameSchema, **kwargs) -> pd.DataFrame:
    """Load a Parquet file into a pandas DataFrame and validate it against a Pandera schema.

    Args:
        path (Path): Path to the Parquet file to load.
        schema (pa.DataFrameSchema): Pandera schema used to validate the loaded DataFrame.
        **kwargs: Additional keyword arguments passed to ``pandas.read_parquet``.

    Returns:
        pd.DataFrame: A validated pandas DataFrame.

    Raises:
        pa.errors.SchemaError: If the loaded DataFrame does not conform to the schema.
        OSError: If the Parquet file cannot be found or read.
    """
    df = pd.read_parquet(path, **kwargs)
    return schema.validate(df)