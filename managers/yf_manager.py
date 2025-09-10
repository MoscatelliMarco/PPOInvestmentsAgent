# Data
import yfinance as yf
import json

# Debug
import logging

# Error handling
from decorators import *

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
logging.getLogger('yfinance').setLevel(logging.WARNING)

class YfManager():
    def __init__(self):

        with open('./assets_list.json', "r") as f:
            self.assets = json.load(f)

        with open("./rl_parameters.json") as f:
            self.rl_parameters = json.load(f)
        with open("./rl_default_parameters.json") as f:
            rl_default_parameters = json.load(f)
        for item in rl_default_parameters.keys():
            if item not in self.rl_parameters.keys():
                self.rl_parameters[item] = rl_default_parameters[item]

    def download_asset(self, asset, start_date='01-01-2010', timeframe='1d'):
        asset = yf.download(asset, start=start_date, interval=timeframe, progress=False)
        if not len(asset):
            logger.error(f"{asset} is invalid")
            return
        asset.to_csv(f"./data/{asset}_{str(asset.index[0]).replace(' ', 'T').replace(':', '-')}.csv")
        logger.info(f"{asset} downloaded successfully")

    def download_from_list(self):

        start_date = self.rl_parameters["start_date"]
        end_date = self.rl_parameters["end_date"]
        for asset_name in self.assets["assets"]:
            asset = yf.download(asset_name, start=start_date, end=end_date, interval=self.rl_parameters['timeframe'], progress=False)
            if not len(asset):
                logger.error(f"{asset_name} is invalid")
                continue
            asset.to_csv(f"./data/{asset_name}_{self.rl_parameters['timeframe']}_{str(asset.index[0]).replace(' ', 'T').replace(':', '-')}.csv")
        logger.info("Assets downloaded successfully")