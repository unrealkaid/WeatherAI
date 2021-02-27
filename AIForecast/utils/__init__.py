from AIForecast.utils.PathUtils import PathUtils
from pyowm.owm import OWM
import logging as logger

owm_access = OWM(PathUtils.get_owm_apikey())
logger.basicConfig(format=logger.BASIC_FORMAT, level=logger.DEBUG)


# Utility function for logging basic messages.
# 'from AIForecast import utils as logger' to call log.
# Todo: Potentially move this to its own class.
def log(name):
    return logger.getLogger(name)
