import pandas as pd

def open_cvs(paths):
    dfs = []
    for path in paths:
        dfs.append(pd.read_csv(path))
    return dfs

