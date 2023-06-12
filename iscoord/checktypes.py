import pandas as pd


def check_pd_df(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The variable passed is not a pd DataFrame")

