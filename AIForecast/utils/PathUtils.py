import os
from definitions import ROOT_DIR
from os import path as osp


# Add some comments Here to tell what this class does.
class PathUtils:
    # Private:
    # Optional save path for software files.
    _save_path = None

    # Private:
    # Set to true if a custom save path has been set.
    # Todo: check for custom path on initialization
    _is_custom_path = False

    # Public:
    # Names of the data directory.
    # The data directory stores all relevant system files.
    # This variable should remain constant.
    DATA_DIR, MODEL_DIR, PKL_DIR = 'data', 'data\\models', 'data\\pickles'

    # Private:
    # Name of the file that stores the API key/
    _API_KEY_FILE = 'owm_apikey.txt'

    # Public:
    # Pickle file names
    TESTING_FILE, TRAINING_FILE, VALIDATION_FILE = '\\testing.pkl', '\\training.pkl', '\\validation.pkl'

    # Private:
    @staticmethod
    def get_root_directory() -> str:
        return PathUtils._save_path \
            if PathUtils._is_custom_path \
            else ROOT_DIR

    # Public:
    @staticmethod
    def set_save_path(path: str) -> None:
        PathUtils._save_path = path
        PathUtils._is_custom_path = True

    # Public:
    @staticmethod
    def get_data_path() -> str:
        return osp.join(PathUtils.get_root_directory(), PathUtils.DATA_DIR)

    # Public:
    @staticmethod
    def get_model_path() -> str:
        return osp.join(PathUtils.get_data_path(), PathUtils.MODEL_DIR)

    @staticmethod
    def get_pkl_path() -> str:
        return osp.join(PathUtils.get_data_path(), PathUtils.PKL_DIR)

    @staticmethod
    def get_file(base_dir: str, file_name: str):
        return osp.join(base_dir, file_name)

    @staticmethod
    def file_exists(file_path: str) -> bool:
        return os.path.isfile(file_path)

    @staticmethod
    def get_owm_apikey() -> str:
        """
        Returns the Open Weather Map key used to access openweathermap.orc
        Though this function is public, it should be treated as a private function since
        an OWM object can be accessed by importing utils owm_access.
        'from utils import owm_access'
        """
        key_path = osp.join(ROOT_DIR, PathUtils._API_KEY_FILE).replace('\\', os.path.sep)
        key_file = open(key_path, 'r')
        api_key = key_file.readline()
        key_file.close()
        return api_key
