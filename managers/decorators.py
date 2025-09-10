# Debug
import logging

# Data
import json
import pandas as pd

# Define your date format without milliseconds
date_format = "%Y-%m-%d %H:%M:%S"
# Set up basic configuration with custom date format
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    datefmt=date_format)
# Create a logger
logger = logging.getLogger(__name__)

# Set Matplotlib's logging level to WARNING to suppress debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# If the data is not preprocessed log an error
def is_preprocessed(func):
    def wrapper(self, *args, **kwargs):
        # Check if the data is initialized
        if not len(self.preprocessed_df):
            logger.error('Data are not preprocessed')
            return
        # Call the function here
        result = func(self, *args, **kwargs)
        return result
    return wrapper

# Check if cointegration file exists
def is_coint_existing(func):
    def wrapper(self, *args, **kwargs):
        # Check if coint.csv exist and it has a len
        try:
            coint = pd.read_csv("./data/processed/coint.csv")
        except:
            logger.error('No coint file found')
            return

        if not len(coint):
            logger.error('No rows in coint file')
            return

        # Call the function here
        result = func(self, *args, **kwargs)
        return result
    return wrapper

# If the model is not loaded log an error
def is_model_loaded(func):
    def wrapper(self, *args, **kwargs):
        # Check if the data is initialized
        if not self.model_loaded:
            logger.error('Model not loaded, load it first with load_network()')
            return
        # Call the function here
        result = func(self, *args, **kwargs)
        return result
    return wrapper