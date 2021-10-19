import numpy as np
import pandas as pd

from verbio import utils

def repair_df(df, policy: str):
    if not df.isna().values.any():
        return df

    if isinstance(df, pd.DataFrame):
        df_keys = utils.get_df_keys(df)
        for key in df_keys:
            if df[key].isna().all():
                df[key].fillna(0, inplace=True)  # In the future, raise a warning
    elif isinstance(df, pd.Series):
        if df.isna().all():
            df.fillna(0, inplace=True)
    else:
        raise ValueError(f'df is of type {type(df)}, expecting pd.DataFrame or pd.Series.')

    if policy == 'zero':
        return df.fillna(0)
    elif policy == 'mean':
        return df.fillna(df.mean())
    elif policy == 'inter':
        return df.interpolate(method='linear').ffill().bfill()  # Kinda sketch
    else:
        raise ValueError(f'Policy {policy} not recognized.')

def combine_dfs(dfs):
    shortest_len = np.inf
    for df in dfs:
        shortest_len = min(shortest_len, len(df.index))
    dfs_truncated = [df.reset_index(drop=True).truncate(after=shortest_len-1) for df in dfs]
    return pd.concat(dfs_truncated, axis=1)

def shuffle(x):
    np.random.shuffle(x)  # Shuffles the parameter, I know its weird...

def shuffle_df(df):
    return df.sample(frac=1).reset_index(drop=True)
