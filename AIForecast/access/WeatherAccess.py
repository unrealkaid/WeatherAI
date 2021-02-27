from datetime import datetime
from typing import List, Set, Dict

import pandas as pd
from pyowm.weatherapi25.weather import Weather

from AIForecast import utils
from AIForecast.utils import owm_access as owm, PathUtils, DataUtils

DEG_F = 'fahrenheit'
DEG_C = 'celsius'
DEG_K = 'kelvin'

csv_columns = [
    'timestamp', 'city_name', 'city_id', 'temperature',
    'temperature_min', 'temperature_max', 'pressure',
    'humidity', 'wind_velocity_x', 'wind_velocity_y',
    'day_x', 'day_y', 'year_x', 'year_y'
]
historic_data: pd.DataFrame = None
cities: Dict = None
years: Set[int] = None


def get_years():
    return list(years)


def get_cities():
    return cities


def load_historical_data() -> None:
    """
    Loads in data/Data.json
    """
    global historic_data, cities, years
    utils.log(__name__).debug("Loading historic data!")
    csv_path = PathUtils.get_file(PathUtils.get_data_path(), "Data.csv")
    if PathUtils.file_exists(csv_path):
        utils.log(__name__).debug("Loading historic data from CSV.")
        historic_data = pd.read_csv(csv_path)
    else:
        utils.log(__name__).debug("No Data.csv found! Loading and converting Data.json to Data.csv.")
        import ijson
        json_matrix = []
        with open(PathUtils.get_file(PathUtils.get_data_path(), "Data.json")) as f:
            for item in ijson.items(f, 'item'):
                wind_x, wind_y = DataUtils.vector_2d(item['wind']['speed'], item['wind']['deg'])
                day_x, day_y, year_x, year_y = DataUtils.periodicity(item['dt'])
                json_matrix.append([
                    datetime.fromtimestamp(item['dt']).isoformat(),
                    item['city_name'],
                    item['city_id'],
                    item['main']['temp'],
                    item['main']['temp_min'],
                    item['main']['temp_max'],
                    item['main']['pressure'],
                    item['main']['humidity'],
                    wind_x, wind_y,
                    day_x, day_y,
                    year_x, year_y
                ])
        historic_data = pd.DataFrame(json_matrix, columns=csv_columns)
        utils.log(__name__).debug("Saving data to CSV.")
        historic_data.to_csv(csv_path)
    cities = {city: city_id for city, city_id in zip(historic_data['city_name'], historic_data['city_id'])}
    years = {datetime.fromisoformat(timestamp).year for timestamp in historic_data['timestamp']}
    utils.log(__name__).debug("Finished loading historic weather data.")


def query_historical_data(training_cities: List[str], start_year, end_year) -> pd.DataFrame:
    if historic_data is None:
        raise ValueError

    if int(start_year) > int(end_year):
        raise IndexError

    utils.log(__name__).debug("Processing data!")
    # Filter the historical matching the passed parameter criteria.
    return historic_data.loc[((historic_data['timestamp'] >= datetime(int(start_year), 1, 1).isoformat())
                             & (historic_data['timestamp'] < datetime(int(end_year), 1, 1).isoformat()))
                             & historic_data['city_name'].isin(training_cities)][csv_columns[3:]]


def get_current_weather_at(city_id) -> pd.DataFrame:
    """
    The city parameters refers to the city name.
    The city_state parameter refers to the state or country that city is located in.
    Returns the weather data for a city in a country.
    Raises ValueError if city could not be found.
    Raises IndexError if more than one city has been returned.
    """
    weather: Weather = owm.weather_manager().weather_at_id(city_id).weather
    temp_dict = weather.temperature()
    wind_dict = weather.wind()
    wind_x, wind_y = DataUtils.vector_2d(wind_dict['speed'], wind_dict['deg'])
    day_x, day_y, year_x, year_y = DataUtils.periodicity(int(datetime.now().timestamp()))
    current_weather = [[
        temp_dict['temp'],
        temp_dict['temp_min'],
        temp_dict['temp_max'],
        weather.pressure['press'],
        weather.humidity,
        wind_x, wind_y,
        day_x, day_y,
        year_x, year_y
    ]]
    return pd.DataFrame(current_weather, columns=csv_columns[3:])
