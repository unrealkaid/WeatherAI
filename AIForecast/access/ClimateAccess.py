from AIForecast import utils
import pandas as pd
import os


def get_source_data():
    return pd.read_csv(utils.PathUtils.get_data_path() + '\\mlo_full.csv')
