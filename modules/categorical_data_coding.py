import pandas as pd


def code_data(df, category_col):
    """
    This function codes categorical data to binary form
    :param df:
    """
    categoriesList = list(df[category_col].unique())
    for category in categoriesList:
        df.loc[df[category_col] == category, [category_col]] = categoriesList.index(category)
    df[category_col] = df[category_col].astype(int)
