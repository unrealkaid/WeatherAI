from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_RADIAN_CONVERSION = 180
_SECONDS_DAY = 24 * 60 ** 2
_SECONDS_YEAR = 365.2425 * _SECONDS_DAY


def vector_2d(mag: float, deg: float):
    """
    Takes a magnitude and a direction in degrees and returns the resulting vector's x and y components.
    """
    radians = float(mag) * np.pi / _RADIAN_CONVERSION
    return float(mag) * np.cos(radians), float(deg) * np.sin(radians)


def periodicity(time_stamp: int):
    day_conversion = time_stamp * (2 * np.pi / _SECONDS_DAY)
    year_conversion = time_stamp * (2 * np.pi / _SECONDS_YEAR)
    return np.sin(day_conversion), np.cos(day_conversion), np.sin(year_conversion), np.cos(year_conversion)


def split_data(data: pd.DataFrame, train_split=.7, validate_split=.2, test_split=.1):
    """
    Returns the train, validate, and test subsets of the data passed.
    return train, validate, and split
    """
    if train_split + test_split + validate_split != 1:
        raise ValueError

    total = len(data)
    train_split, train_validate_split = int(total * train_split), int(total * (train_split + validate_split))
    return data[:train_split], data[train_split:train_validate_split], data[train_validate_split:]


def kelvin_to_fahrenheit(k: float):
    """
    Converts Kelvin to Fahrenheit
    """
    return (k - 273.15) * 9 / 5 + 32
