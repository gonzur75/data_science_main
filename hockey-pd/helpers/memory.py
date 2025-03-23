def memory_mib(df):
    """Return the memory usage of a DataFrame in MiB."""
    return f'{df.memory_usage().sum() / (1024 ** 2):.2f} MiB'
