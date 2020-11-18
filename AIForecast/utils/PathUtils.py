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
    def _get_base_path(self) -> str:
        return self._save_path \
            if self._is_custom_path \
            else ROOT_DIR

    # Public:
    def set_save_path(self, path: str) -> None:
        self._save_path = path
        self._is_custom_path = True

    # Public:
    def get_data_path(self) -> str:
        return osp.join(self._get_base_path(), self.DATA_DIR)

    # Public:
    def get_model_path(self) -> str:
        return osp.join(self._get_base_path(), self.MODEL_DIR)

    # Public:
    def get_pkl_path(self) -> str:
        return osp.join(self._get_base_path(), self.PKL_DIR)

    # Public:
    # Returns the Open Weather Map key used to access openweathermap.orc
    # Though this function is public, it should be treated as a private function since
    # an OWM object can be accessed by importing utils owm_access.
    # 'from utils import owm_access'
    def get_owm_apikey(self) -> str:
        key_path = osp.join(ROOT_DIR, self._API_KEY_FILE).replace('\\', os.path.sep)
        key_file = open(key_path, 'r')
        api_key = key_file.readline()
        key_file.close()
        return api_key
