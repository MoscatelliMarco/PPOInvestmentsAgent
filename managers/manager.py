# NOTE: this paths are here in case I need to work with this file using different folders
import sys
sys.path.append("./managers")
sys.path.append("./ppo")
sys.path.append("../ppo")

# Stats
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Data
from yf_manager import YfManager
import time
import json
import pandas as pd
import io

# Error handling
from decorators import *

# Utils
import numpy as np
import os
import ta
import ast
import random
import torch
from ppo import PPO
from utils.separate_log import separate_log

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Debug
import logging

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

# Suppress SettingWithCopyWarning
# Without this the code doesn't work during the removal of stationary in some methods
pd.options.mode.chained_assignment = None  # default='warn'

class Manager():
    def __init__(self, download_data=True):

        if download_data:
            yf_manager = YfManager()
            yf_manager.download_from_list()

        with open("./assets_list.json", "r") as f:
            self.assets = json.load(f)
        with open("./rl_parameters.json") as f:
            self.rl_parameters = json.load(f)
        with open("./rl_default_parameters.json") as f:
            rl_default_parameters = json.load(f)
        for item in rl_default_parameters.keys():
            if item not in self.rl_parameters.keys():
                self.rl_parameters[item] = rl_default_parameters[item]

        if self.rl_parameters['split_train'] + self.rl_parameters['split_validation'] > 1 or self.rl_parameters['split_train'] + self.rl_parameters['split_validation'] <= 0:
            logger.error("Invalid set split")
            return

        if self.rl_parameters['architecture_type'] == "FFN" and self.rl_parameters['seqlength'] != 1:
            logger.warning("If the architecture type if FFN the seqlength must be 1, the value of seqlength is automatically changed to 1")
            self.rl_parameters['seqlength'] = 1

        if self.rl_parameters['eval_train_after_n_runs'] > self.rl_parameters['n_runs']:
            logger.warning("eval_train_after_n_runs is bigger than n_runs, manager will never eval the model on the train set during the training process")

        if self.rl_parameters['eval_val_after_n_runs'] > self.rl_parameters['n_runs']:
            logger.warning("eval_val_after_n_runs is bigger than n_runs, manager will never eval the model on the train set during the training process")
        
        # If seed is null generate a random one
        if not self.rl_parameters['seed']:
            self.rl_parameters['seed'] = random.randint(1, 1000)

    def cointegration(self):

        # Transform the asset names into filenames
        all_items = os.listdir('./data')
        files = [item for item in all_items if os.path.isfile(os.path.join('./data', item))]
        assets_files = []
        for inst in self.assets["assets"]:
            for filename in files:
                if inst in filename:
                    assets_files.append(filename)

        # ALGO LOGIC: run trough all possible combinations and find the cointegrated ones
        coint_combinations = []
        done_combinations = []
        for inst_1 in assets_files:
            for inst_2 in assets_files:
                if inst_1 != inst_2:
                    inst_1_name = inst_1.split('-')[0][:-5]
                    inst_2_name = inst_2.split('-')[0][:-5]
                    if f'{inst_1_name}_{inst_2_name}' not in done_combinations or f'{inst_2_name}_{inst_1_name}' not in done_combinations:

                        sys.stdout.write(f'\rDoing {inst_1_name}_{inst_2_name}')
                        sys.stdout.flush()

                        df_1 = pd.read_csv(f"./data/{inst_1}", index_col="Date", parse_dates=["Date"])
                        df_2 = pd.read_csv(f"./data/{inst_2}", index_col="Date", parse_dates=["Date"])

                        # Find the cut because of missing "starting data" and the one from problems inside the rows (like missing days at random)
                        # Getting the starting dates
                        start_date_df1 = df_1.index[0]
                        start_date_df2 = df_2.index[0]
                        missing_starting_candles_asset = None
                        if df_2.index[0] > df_1.index[0]:
                            missing_starting_candles_asset = inst_2
                            difference_in_candles = df_1.loc[:start_date_df2].shape[0] - 1
                        elif df_1.index[0] > df_2.index[0]:
                            missing_starting_candles_asset = inst_1
                            difference_in_candles = df_2.loc[:start_date_df1].shape[0] - 1
                        else:
                            difference_in_candles = 0

                        # Find common indices
                        common_indices = df_1.index.intersection(df_2.index)

                        # Find the real number of "inside" missing candles
                        missing_inside_candles = 0
                        if df_2.index[0] > df_1.index[0]:
                            df_1_temp = df_1.copy().loc[df_2.index[0]:]
                            missing_candles_inside = len(df_1_temp.index.difference(df_2.index))
                        elif df_1.index[0] > df_2.index[0]:
                            df_2_temp = df_2.copy().loc[df_1.index[0]:]
                            missing_candles_inside = len(df_1.index.difference(df_2_temp.index))

                        # Keep only rows with common indices in both dataframes
                        df_1 = df_1.loc[df_1.index.isin(common_indices)]
                        df_2 = df_2.loc[df_2.index.isin(common_indices)]

                        # NOTE: this code is really important because it prevents the model to have forward bias on the validation and test set
                        # Calculate the number of rows for each set
                        n = len(df_1)
                        n_train = int(self.rl_parameters['split_train'] * n)
                        # Slicing the DataFrame
                        df_1 = df_1.iloc[:n_train]
                        df_2 = df_2.iloc[:n_train]

                        # Use log values to calculate cointegration
                        coint_flag, hedge_ratio, p_value = calculate_cointegration(np.log(df_1['Close']), np.log(df_2['Close']))

                        done_combinations.append(f'{inst_1_name}_{inst_2_name}')
                        done_combinations.append(f'{inst_2_name}_{inst_1_name}')

                        if coint_flag == 1:
                            coint_combinations.append({
                                'asset_1': inst_1_name, 
                                'asset_2': inst_2_name, 
                                'hedge_ratio': hedge_ratio, 
                                'p_value': p_value, 
                                'start': str(df_1.index[0]), 
                                'end': str(df_1.index[-1]),
                                'start_difference_candles': difference_in_candles,
                                'start_late_asset': missing_starting_candles_asset,
                                'missing_candles_inside': missing_inside_candles
                                })

        coint_df = pd.DataFrame(coint_combinations)

        # Sort the dataframe with the lowest p-value at the top
        coint_df.sort_values(by='p_value', ascending=True, inplace=True)

        coint_df.to_csv('./data/processed/coint.csv')

    @is_coint_existing
    def visualize_cointegrated(self, pairs=4, save_img=True, asset_1_name=None, asset_2_name=None):
        coint = pd.read_csv('./data/processed/coint.csv')

        # Transform the asset names into filenames
        all_items = os.listdir('./data')
        files = [item for item in all_items if os.path.isfile(os.path.join('./data', item))]

        if not asset_1_name and not asset_2_name:
            permutation = np.random.permutation(range(len(coint)))
            coint = coint.iloc[permutation[:pairs]]

            fig, axs = plt.subplots(len(coint), 1, figsize=[12, 4*len(coint)])
            for i in range(len(coint)):
                asset_1_name = coint.iloc[i]['asset_1']
                asset_2_name = coint.iloc[i]['asset_2']

                for filename in files:
                    if asset_1_name in filename:
                        asset_1 = pd.read_csv(f'./data/{filename}', index_col="Date", parse_dates=["Date"])
                    if asset_2_name in filename:
                        asset_2 = pd.read_csv(f'./data/{filename}', index_col="Date", parse_dates=["Date"])

                ax = axs.flatten()[i] if len(coint) != 1 else axs
                ax.set_title(f"{asset_1_name} - {asset_2_name}, hedge_ratio: {coint.iloc[i]['hedge_ratio']}, p_value: {coint.iloc[i]['p_value']}")
                ax.plot(np.log(asset_1['Close']))
                ax.plot(np.log(asset_2['Close']) * coint['hedge_ratio'].iloc[i])
                ax.legend([asset_1_name, asset_2_name])
        else:
            fig, ax = plt.subplots(1, 1, figsize=[12, 4])
            asset_1_name = asset_1_name
            asset_2_name = asset_2_name

            for i in range(len(coint)):
                if asset_1_name in coint.iloc[i]['asset_1'] and asset_2_name in coint.iloc[i]['asset_2']:
                    coint_index = i

            for filename in files:
                if asset_1_name in filename:
                    asset_1 = pd.read_csv(f'./data/{filename}', index_col="Date", parse_dates=["Date"])
                if asset_2_name in filename:
                    asset_2 = pd.read_csv(f'./data/{filename}', index_col="Date", parse_dates=["Date"])

            ax.set_title(f"{asset_1_name} - {asset_2_name}, hedge_ratio: {coint.iloc[coint_index]['hedge_ratio']}, p_value: {coint.iloc[coint_index]['p_value']}")
            ax.plot(asset_1['Close'] - coint['hedge_ratio'].iloc[coint_index])
            ax.plot(asset_2['Close'])
            ax.legend([asset_1_name, asset_2_name])

        plt.tight_layout()
        if save_img:
            plt.savefig(f"./data/visualize/cointegrated/cointegrated_{int(time.time())}.jpg", dpi=300)
        plt.show()

    @is_coint_existing
    def preprocess_data(self, asset_1_name, asset_2_name, non_stationary_method='auto', check_non_stationary_after=False, indicators=['MACD', 'RSI', 'CMF', 'MA_diff']):
        '''
        non_stationary_method: 'auto' or 'manual'
        '''
        # Transform the asset names into filenames
        all_items = os.listdir('./data')
        files = [item for item in all_items if os.path.isfile(os.path.join('./data', item))]
        asset_1_filename = None
        asset_2_filename = None

        for filename in files:
            if asset_1_name in filename:
                asset_1_filename = filename
            if asset_2_name in filename:
                asset_2_filename = filename

        if not asset_1_filename or not asset_2_filename:
            logger.error("The asset/s is not downloaded")
            return

        df_1 = pd.read_csv(f'./data/{asset_1_filename}', index_col="Date", parse_dates=["Date"])
        df_2 = pd.read_csv(f'./data/{asset_2_filename}', index_col="Date", parse_dates=["Date"])

        # Get just the rows in common
        # Find common indices
        common_indices = df_1.index.intersection(df_2.index)

        # Keep only rows with common indices in both dataframes
        df_1 = df_1.loc[df_1.index.isin(common_indices)]
        df_2 = df_2.loc[df_2.index.isin(common_indices)]

        # Find the row in the coint dataframe that has the same assets
        coint = pd.read_csv("./data/processed/coint.csv")
        hedge_ratio = None
        for i in range(len(coint)):
            if asset_1_name in coint.iloc[i]['asset_1'] and asset_2_name in coint.iloc[i]['asset_2']:
                hedge_ratio = coint.iloc[i]['hedge_ratio']
            elif asset_2_name in coint.iloc[i]['asset_1'] and asset_1_name in coint.iloc[i]['asset_2']:
                hedge_ratio = 1 / coint.iloc[i]['hedge_ratio']
        if not hedge_ratio:
            logger.error("Cannot find assets in coint file")
            return

        # Calculate the statistical arbitrage spread
        spread = np.log(np.array(df_1['Close'])) - hedge_ratio * np.log(np.array(df_2['Close']))

        # Calculate indicators
        if 'MACD' in indicators:
            df_1['MACD'] = ta.trend.MACD(df_1['Close']).macd()
            df_2['MACD'] = ta.trend.MACD(df_2['Close']).macd()
        if 'RSI' in indicators:
            df_1['RSI'] = ta.momentum.RSIIndicator(df_1['Close']).rsi()
            df_2['RSI'] = ta.momentum.RSIIndicator(df_2['Close']).rsi()
        if 'CMF' in indicators:
            df_1['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df_1['High'], df_1['Low'], df_1['Close'], df_1['Volume']).chaikin_money_flow()
            df_2['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df_2['High'], df_2['Low'], df_2['Close'], df_2['Volume']).chaikin_money_flow()
        if 'MA_diff' in indicators:
            df_1['MA_diff'] = np.log(df_1['Close']).rolling(35).mean() - np.log(df_1['Close']).rolling(135).mean()
            df_2['MA_diff'] = np.log(df_2['Close']).rolling(35).mean() - np.log(df_2['Close']).rolling(135).mean()

        # Merge all the dataframes together
        columns_to_pick = list(df_1.columns)
        columns_to_pick.remove('Adj Close')
        merged = pd.concat([df_1[columns_to_pick], 
                            df_2[columns_to_pick],
                            ], 
                        axis=1, 
                        keys=['asset_1', 'asset_2'])
        merged[('spread', 'spread')] = spread
        merged.dropna(inplace=True)

        # PARAM: Add day of the week
        # Extract day names
        merged[('all', 'Day_of_Week')] = merged.index.day_name()
        # One-hot encode the days
        merged_one_hot = pd.get_dummies(merged[('all', 'Day_of_Week')])
        # Merge the one-hot encoded DataFrame with your original DataFrame
        merged = pd.concat([merged, merged_one_hot], axis=1)
        # Optionally, you can drop the 'Day_of_Week' column if it's no longer needed
        del merged[('all', 'Day_of_Week')]

        # PARAM: Add range of candles
        merged[('asset_1', 'range_shadow')] = np.log(merged[('asset_1', 'High')]) - np.log(merged[('asset_1', 'Low')])
        merged[('asset_1', 'range_body')] = abs(np.log(merged[('asset_1', 'Open')]) - np.log(merged[('asset_1', 'Close')]))
        merged[('asset_2', 'range_shadow')] = np.log(merged[('asset_2', 'High')]) - np.log(merged[('asset_2', 'Low')])
        merged[('asset_2', 'range_body')] = abs(np.log(merged[('asset_2', 'Open')]) - np.log(merged[('asset_2', 'Close')]))

        # Create targets
        # NOTE: not shifted because this is done automatically by the environment
        merged[('asset_1', 'TARGET')] = df_1['Close'] / df_1['Close'].shift(1) - 1
        merged[('asset_2', 'TARGET')] = df_2['Close'] / df_2['Close'].shift(1) - 1

        # Remove non stationary
        non_stationary = []
        for col in merged.columns:
            if 'TARGET' not in col[1]:
                if not adf_test(merged[col]) and col[1] not in self.rl_parameters['exclude_stationary']:
                    non_stationary.append(col) 
        
        if non_stationary_method == 'manual':
            logger.info("Which features do you want to be considered as non stationary? (input a list)")
            logger.info(f"All features: {list(merged.columns)}")
            logger.info(f"Non stationary according to adfuller test: {non_stationary}")
            non_stationary_string = input()
            if non_stationary_string:
                non_stationary = ast.literal_eval(non_stationary_string)


        # Removing non stationary methods
        if "sma_detrending_std" in self.rl_parameters['non_stationary_function']: # TESTED!
            trend = merged[non_stationary].rolling(int(self.rl_parameters['non_stationary_function'].split('_')[3])).mean()
            merged[non_stationary] = merged[non_stationary] - trend
            merged[non_stationary] = merged[non_stationary] / merged[non_stationary].rolling(200).std() # in this way there is not risk that the closest values are too wide
            merged.dropna(inplace=True)
        elif "sma_detrending_log" in self.rl_parameters['non_stationary_function']:
            for col in non_stationary:
                if merged[col].min() <= 0: # Log can't accept negative values
                    merged[col] = merged[col] - merged[col].shift(1)
                else:
                    trend = np.log(merged[col]).rolling(int(self.rl_parameters['non_stationary_function'].split('_')[3])).mean()
                    merged[col] = np.log(merged[col]) - trend
            merged.dropna(inplace=True)
        elif "log_returns" == self.rl_parameters['non_stationary_function']:
            for col in non_stationary:
                if merged[col].min() <= 0: # Log can't accept negative values
                    merged[col] = merged[col] - merged[col].shift(1)
                else:
                    merged[col] = np.log(merged[col] / merged[col].shift(1))
            merged.dropna(inplace=True)
        elif "normal_returns" == self.rl_parameters['non_stationary_function']:
            merged[non_stationary] = merged[non_stationary] / merged[non_stationary].shift(1)
            merged.dropna(inplace=True)
        else:
            raise ValueError("Invalid value provided for non_stationary_function in rl_parameters")

        if check_non_stationary_after:
            non_stationary = {}
            for col in merged.columns:
                if 'TARGET' not in col[1]:
                    p_value = adf_test(merged[col], p_value_return=True)
                    if p_value > 0.05 and col[1]:
                        non_stationary[col] = p_value
            logger.info(f"non_stationary after removal: {non_stationary}")
            
        # Get the dates searched in rl_parameters
        merged = merged.loc[self.rl_parameters['start_date']:self.rl_parameters['end_date']]

        # Calculate the number of rows for each set
        n = len(merged)
        n_train = int(self.rl_parameters['split_train'] * n)
        n_validation = int(self.rl_parameters['split_validation'] * n)

        # Slicing the DataFrame
        train_set = merged.iloc[:n_train]
        validation_set = merged.iloc[n_train - self.rl_parameters['seqlength']:n_train + n_validation] # Substracting seqlength to have the right number of data during the test and train evaluation
        test_set = merged.iloc[n_train + n_validation - self.rl_parameters['seqlength']:]

        # Get normalization params
        # NOTE: the parameters are only calculated on the train set to prevent any bias
        normalization_params = {}
        if self.rl_parameters['normalization'] == 'zscore':
            normalization_params['mean'] = train_set.mean()
            normalization_params['std'] = train_set.std()
        elif self.rl_parameters['normalization'] == 'minmax':
            normalization_params['max'] = train_set.max()
            normalization_params['min'] = train_set.min()
        else:
            logger.error("Invalid normalization provided")
            return

        # Apply normalization
        if self.rl_parameters['normalization'] == 'zscore':
            for col in train_set.columns:
                if 'TARGET' != col[1]:
                    train_set[col] = (train_set[col] - normalization_params['mean'][col]) / normalization_params['std'][col]
                    validation_set[col] = (validation_set[col] - normalization_params['mean'][col]) / normalization_params['std'][col]
                    test_set[col] = (test_set[col] - normalization_params['mean'][col]) / normalization_params['std'][col]
        elif self.rl_parameters['normalization'] == 'minmax':
            for col in train_set.columns:
                if 'TARGET' != col[1]:
                    train_set.loc[:, col] = (train_set[col] - normalization_params['min'][col]) / (normalization_params['max'][col] - normalization_params['min'][col])
                    validation_set.loc[:, col] = (validation_set[col] - normalization_params['min'][col]) / (normalization_params['max'][col] - normalization_params['min'][col])
                    test_set.loc[:, col] = (test_set[col] - normalization_params['min'][col]) / (normalization_params['max'][col] - normalization_params['min'][col])
        else:
            logger.error("Invalid normalization provided")
            return

        self.normalization_params = normalization_params
        self.preprocessed_df =  pd.concat([train_set, validation_set, test_set])
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.non_stationary = non_stationary

    @is_preprocessed
    def visualize_preprocessed(self, save_imgs=True):
        df = self.preprocessed_df.copy()

        # Extracting specific columns
        asset1_close = df[('asset_1', 'Close')]
        asset2_close = df[('asset_2', 'Close')]
        spread = df[('spread', 'spread')]

        # Creating histograms for each feature
        num_features = len(df.columns)
        num_rows = num_features // 2 if num_features % 2 == 0 else num_features // 2 + 1
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, num_rows * 3))
        axes = axes.flatten()
        for i, col in enumerate(df.columns):
            sys.stdout.write(f"\rDoing {col}")
            sns.histplot(df[col], ax=axes[i], kde=True)
            axes[i].set_title(col)
        for i in range(num_features, len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        if save_imgs:
            plt.savefig(f'./data/visualize/preprocessed/distribution_features_{int(time.time())}.jpg')
        plt.show()

        # Generating heatmap of correlations
        corr_matrix = df.corr()
        plt.figure(figsize=(17, 17))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis')
        plt.title("Correlation Heatmap")
        if save_imgs:
            plt.savefig(f'./data/visualize/preprocessed/corr_heatmap_{int(time.time())}.jpg')
        plt.show()

        # Plotting specific columns
        plt.figure(figsize=(17, 10))
        plt.plot(df.index, asset1_close, label="Asset 1 Close")
        plt.plot(df.index, asset2_close, label="Asset 2 Close")
        plt.plot(df.index, spread, label="Spread")
        plt.title("Asset 1 & 2 Close Prices and Spread")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        if save_imgs:
            plt.savefig(f'./data/visualize/preprocessed/close_spread_{int(time.time())}.jpg', dpi=300)
        plt.show()

    def load_ppo(self, timestep=None, time_id=None):
        self.model_loaded = False
        self.timestep = timestep
        self.loaded_time_id = time_id
        if timestep and time_id:
            self.ppo = PPO(self.train_set, self.normalization_params, loaded_timestep=timestep)
            self.ppo.agent.load_agent(timestep, time_id)
            self.model_loaded = True
            pass
        else:
            if timestep and not time_id or time_id and not timestep:
                logger.warning("The PPO is loaded at timestep zero because timestep or time_id is null")
            self.ppo = PPO(self.train_set, self.normalization_params, loaded_timestep=0)
            self.model_loaded = True
    
    @is_model_loaded
    def train(self):
        timestep_run = 0 if not self.timestep else self.timestep
        start_timestep = 0 if not self.timestep else self.timestep
        for i in range(self.rl_parameters['n_runs']):
            logging.info(f"Doing run {i + 1} out of {self.rl_parameters['n_runs']}")
            if i > 0:
                self.ppo.run_load = True
            if timestep_run:
                self.ppo.loaded_timestep = timestep_run
            self.ppo.learn()
            self.ppo.agent.save_agent(timestep_run, non_stationary = self.non_stationary, loaded_time_id = self.loaded_time_id)

            real_timestep_length = (self.rl_parameters['total_timesteps'] // (int(self.rl_parameters['num_envs'] * self.rl_parameters['num_steps'])) * int(self.rl_parameters['num_envs'] * self.rl_parameters['num_steps']))
            timestep_run = start_timestep + (i + 1) * real_timestep_length

            # Eval model on val set
            if (i + 1) % self.rl_parameters['eval_val_after_n_runs'] == 0:
                self._save_eval(timestep_run, set_type='validation')

            # Eval model on train set
            if (i + 1) % self.rl_parameters['eval_train_after_n_runs'] == 0:
                self._save_eval(timestep_run, set_type='train')

    def _save_eval(self, timestep_run, set_type='validation'):
        if set_type == 'train':
            df_an, earnings, accuracies = self.train_eval(timestep=timestep_run, time_id=self.ppo.time_id, show_plot=False)
        elif set_type == 'validation':
            df_an, earnings, accuracies = self.validation_eval(timestep=timestep_run, time_id=self.ppo.time_id, show_plot=False)
        else:
            logger.error("Invalid set_type provided, only valid options are validation and train")
            return

        earnings_1 = df_an[('strategy', 'returns_1')].iloc[-1]
        earnings_2 = df_an[('strategy', 'returns_2')].iloc[-1]
        self.ppo.writer.add_scalar(f"_perfomance_{set_type}/earnings", earnings, timestep_run)
        self.ppo.writer.add_scalar(f"_perfomance_{set_type}/earnings_1", earnings_1, timestep_run)
        self.ppo.writer.add_scalar(f"_perfomance_{set_type}/earnings_2", earnings_2, timestep_run)

        trades_1 = df_an[('strategy', 'trades_1')].astype(int).value_counts().to_dict()
        trades_2 = df_an[('strategy', 'trades_2')].astype(int).value_counts().to_dict()
        for key in [0, 1, 2]:
            if key in trades_1.keys():
                self.ppo.writer.add_scalar(f"_logic_{set_type}/trades_1_{key_to_trade(key)}", trades_1[key], timestep_run)
            else:
                self.ppo.writer.add_scalar(f"_logic_{set_type}/trades_1_{key_to_trade(key)}", 0, timestep_run)
        for key in [0, 1, 2]:
            if key in trades_2.keys():
                self.ppo.writer.add_scalar(f"_logic_{set_type}/trades_2_{key_to_trade(key)}", trades_2[key], timestep_run)
            else:
                self.ppo.writer.add_scalar(f"_logic_{set_type}/trades_2_{key_to_trade(key)}", 0, timestep_run)

        positions_1 = df_an[('strategy', 'position_1')].astype(int).value_counts().to_dict()
        positions_2 = df_an[('strategy', 'position_2')].astype(int).value_counts().to_dict()
        for key in [-1, 0, 1]:
            if key in positions_1.keys():
                self.ppo.writer.add_scalar(f"_logic_{set_type}/positions_1_{key_to_position(key)}", positions_1[key], timestep_run)
            else:
                # If the key in not in the unique values just use 0
                self.ppo.writer.add_scalar(f"_logic_{set_type}/positions_1_{key_to_position(key)}", 0, timestep_run)
        for key in [-1, 0, 1]:
            if key in positions_2.keys():
                self.ppo.writer.add_scalar(f"_logic_{set_type}/positions_2_{key_to_position(key)}", positions_2[key], timestep_run)
            else:
                # If the key in not in the unique values just use 0
                self.ppo.writer.add_scalar(f"_logic_{set_type}/positions_2_{key_to_position(key)}", 0, timestep_run)
        
        if accuracies[0]:
            self.ppo.writer.add_scalar(f"_perfomance_{set_type}/accuracy_1", accuracies[0], timestep_run)
        if accuracies[1]:
            self.ppo.writer.add_scalar(f"_perfomance_{set_type}/accuracy_2", accuracies[1], timestep_run)
        if accuracies[2]:
            self.ppo.writer.add_scalar(f"_perfomance_{set_type}/accuracy", accuracies[2], timestep_run)


    def train_eval(self, timestep=None, time_id=None, show_plot=True):
        df_an = self.train_set.copy()
        return self.eval_set(df_an, set_name='train', timestep=timestep, time_id=time_id, show_plot=show_plot)

    def validation_eval(self, timestep=None, time_id=None, show_plot=True):
        df_an = self.validation_set.copy()
        return self.eval_set(df_an, set_name='validation', timestep=timestep, time_id=time_id, show_plot=show_plot)

    def test_eval(self, timestep=None, time_id=None, show_plot=True):
        df_an = self.test_set.copy()
        return self.eval_set(df_an, set_name='test', timestep=timestep, time_id=time_id, show_plot=show_plot)

    def eval_models(self, eval_range, time_id):
        if not isinstance(eval_range, range) and not isinstance(eval_range, list):
            raise ValueError("eval_range must be a list or a range")

        eval_range = list(eval_range) # Converting for more flexibility

        cmap = plt.cm.get_cmap('YlOrBr', len(eval_range) + int(len(eval_range) * .25)) # Add also a little bit more points to it doesn't start almost completely invisible
        plt.figure(figsize=[15, 7])
        for i, timestep in enumerate(eval_range):
            sys.stdout.write(f"\rDoing {timestep}/{eval_range[-1]}")

            df_an = self.validation_set.copy()

            df_results, _, _ = self.eval_set(df_an, timestep=timestep, time_id=time_id, show_plot=False)

            plt.plot(df_results[('strategy', 'returns')].iloc[self.rl_parameters['seqlength']:], color=cmap(i + int(len(eval_range) * .25)))

        # Add also the returns as a benchmark
        plt.plot(df_results[('asset_1', 'TARGET')].cumsum().apply(np.exp).iloc[self.rl_parameters['seqlength']:], color='blue')
        plt.plot(df_results[('asset_2', 'TARGET')].cumsum().apply(np.exp).iloc[self.rl_parameters['seqlength']:], color='green')
        eval_range.append("Benchmark_1")
        eval_range.append("Benchmark_2")  

        plt.legend(eval_range)
        plt.savefig(f"./data/visualize/evaluation/all_validation/all_val_{int(time.time())}")
        plt.show()

    def eval_training_process(self, eval_range, time_id):
        if not isinstance(eval_range, range) and not isinstance(eval_range, list):
            raise ValueError("eval_range must be a list or a range")
        
        eval_range = list(eval_range) # Converting for more flexibility

        cmap = plt.cm.get_cmap('YlOrBr', len(eval_range) + int(len(eval_range) * .25)) # Add also a little bit more points to it doesn't start almost completely invisible
        plt.figure(figsize=[15, 7])
        for i, timestep in enumerate(eval_range):
            sys.stdout.write(f"\rDoing {timestep}/{eval_range[-1]}")

            df_an = self.train_set.copy()

            df_results, _, _ = self.eval_set(df_an, timestep=timestep, time_id=time_id, show_plot=False)

            plt.plot(df_results[('strategy', 'returns')].iloc[self.rl_parameters['seqlength']:], color=cmap(i + int(len(eval_range) * .25)))

        # Add also the returns as a benchmark
        plt.plot(df_results[('asset_1', 'TARGET')].cumsum().apply(np.exp).iloc[self.rl_parameters['seqlength']:], color='blue')
        plt.plot(df_results[('asset_2', 'TARGET')].cumsum().apply(np.exp).iloc[self.rl_parameters['seqlength']:], color='green')
        eval_range.append("Benchmark_1")
        eval_range.append("Benchmark_2")  

        plt.legend(eval_range)
        plt.savefig(f"./data/visualize/evaluation/all_train/all_val_{int(time.time())}")
        plt.show()

    def eval_set(self, df_an, set_name='train', timestep=None, time_id=None, show_plot=True):
        if timestep and time_id:
            eval_ppo = PPO(self.train_set, self.normalization_params, loaded_timestep=timestep)
            eval_ppo.agent.load_agent(timestep, time_id)
        else:
            if timestep and not time_id or time_id and not timestep:
                logger.warning("The PPO is loaded at timestep zero because timestep or time_id is null")
            eval_ppo = PPO(self.train_set, self.normalization_params, loaded_timestep=0)
            timestep = 0

        eval_ppo.agent.eval()

        positions = np.zeros((2, len(df_an)))
        for i in range(len(df_an) - self.rl_parameters['seqlength']):
            if self.rl_parameters['architecture_type'] == 'LSTM':
                with torch.no_grad():
                    X = torch.Tensor(np.array(df_an[df_an.columns[:-2]].iloc[i : i + self.rl_parameters['seqlength']])).unsqueeze(0)
            elif self.rl_parameters['architecture_type'] == 'FFN':
                with torch.no_grad():
                    X = torch.Tensor(np.array(df_an[df_an.columns[:-2]].iloc[i + self.rl_parameters['seqlength'] - 1])).unsqueeze(0)
            else:
                raise ValueError("Invalid architecture type in rl_parameters")

            actions, _, _, _ = eval_ppo.agent.get_action_and_value(X, deterministic = True) # the model should have no stochasticity during the testing period
            action_1 = (actions[:, 0] - 1) * -1 # -1 so the values become -1, 0 and 1, and -1 to change the sign or else the buy action would be -1
            action_2 = (actions[:, 1] - 1) * -1
            positions[0, i + self.rl_parameters['seqlength']] = action_1
            positions[1, i + self.rl_parameters['seqlength']] = action_2

        # Transform return in log returns
        df_an[('asset_1', 'TARGET')] = np.log(df_an[('asset_1', 'TARGET')] + 1)
        df_an[('asset_2', 'TARGET')] = np.log(df_an[('asset_2', 'TARGET')] + 1)
        df_an[('asset_1', 'cum_returns')] = df_an[('asset_1', 'TARGET')].cumsum().apply(np.exp)
        df_an[('asset_2', 'cum_returns')] = df_an[('asset_2', 'TARGET')].cumsum().apply(np.exp)

        # Create strategy trades count and returns
        df_an[('strategy', 'position_1')] = positions[0, :]
        df_an[('strategy', 'position_2')] = positions[1, :]
        df_an[('strategy', 'trades_1')] = df_an[('strategy', 'position_1')].diff().fillna(0).abs()
        df_an[('strategy', 'trades_2')] = df_an[('strategy', 'position_2')].diff().fillna(0).abs()
        df_an[('strategy', 'trades')] = df_an[('strategy', 'trades_1')] + df_an[('strategy', 'trades_2')]
        df_an[('strategy', 'returns_1')] = (df_an[('asset_1', 'TARGET')] * df_an[('strategy', 'position_1')]).cumsum().apply(np.exp)
        df_an[('strategy', 'returns_2')] = (df_an[('asset_2', 'TARGET')] * df_an[('strategy', 'position_2')]).cumsum().apply(np.exp)
        df_an[('strategy', 'returns')] = (df_an[('strategy', 'returns_1')] + df_an[('strategy', 'returns_2')]) / 2

        # Calculate accuracy
        # NOTE: the accuracy is calculated only if the model decides to take a position long or short, if neutral that candle doesn't count
        right_times_1 = 0
        wrong_times_1 = 0
        right_times_2 = 0
        wrong_times_2 = 0
        for i in range(len(df_an)):
            if df_an[('strategy', 'position_1')].iloc[i] != 0:
                next_candle = 1 if df_an[('asset_1', 'TARGET')].iloc[i] >= 0 else -1
                if df_an[('strategy', 'position_1')].iloc[i] == next_candle:
                    right_times_1 += 1
                else:
                    wrong_times_1 += 1

            if df_an[('strategy', 'position_2')].iloc[i] != 0:
                next_candle = 1 if df_an[('asset_2', 'TARGET')].iloc[i] >= 0 else -1
                if df_an[('strategy', 'position_2')].iloc[i] == next_candle:
                    right_times_2 += 1
                else:
                    wrong_times_2 += 1
        
        # The model need to take trades to have an accuracy or else it will have a division by zero
        accuracy_1 = None
        accuracy_2 = None
        accuracy = None
        if right_times_1 + wrong_times_1:
            accuracy_1 = round(right_times_1 / (right_times_1 + wrong_times_1) * 100, 2)
        if right_times_2 + wrong_times_2:
            accuracy_2 = round(right_times_2 / (right_times_2 + wrong_times_2) * 100, 2)
        if right_times_1 + wrong_times_1 and right_times_2 + wrong_times_2:
            accuracy = round((accuracy_1 + accuracy_2) / 2, 2)

        if show_plot:
            fig, axs = plt.subplots(2, 1, figsize=[15, 10])

            fig.suptitle(f"{set_name.capitalize()} Set Evaluation, num_trades (1 to go neutral, 2 to change direction): {df_an[('strategy', 'trades')].sum()}", fontsize=16)

            axs[0].set_title("Performance Comparison")
            axs[0].plot(df_an[('asset_1', 'TARGET')].cumsum().apply(np.exp))
            axs[0].plot(df_an[('asset_2', 'TARGET')].cumsum().apply(np.exp))

            # Show the num trade heatmap x% of the price window lower (rest of the code during the plotting cycle)
            range_price = df_an[[('strategy', 'returns'), ('asset_1', 'cum_returns'), ('asset_2', 'cum_returns')]].max().max() - df_an[[('strategy', 'returns'), ('asset_1', 'cum_returns'), ('asset_2', 'cum_returns')]].min().min()

            # Create heatmap trades to understand better how the model behaves
            returns_min = df_an[[('strategy', 'returns'), ('strategy', 'returns_1'), ('strategy', 'returns_2'), ('asset_1', 'cum_returns'), ('asset_2', 'cum_returns')]].min().min()
            for value, color in zip([-1, 0, 1], ['lightblue', 'gray', 'blue']):
                condition = df_an[('strategy', 'position_1')] == value
                x_coords = list(df_an.index[condition])
                y_length = len(x_coords)
                y_coords = [returns_min - range_price * .03] * y_length

                axs[0].plot(x_coords, y_coords, marker='|', markersize=8, linestyle='', color=color, label=f'Trades 1 - {value}')
            for value, color in zip([-1, 0, 1], [(1, 0.6, 0.2), 'gray', 'orange']):
                condition = df_an[('strategy', 'position_2')] == value
                x_coords = list(df_an.index[condition])
                y_length = len(x_coords)
                y_coords = [returns_min - range_price * .08] * y_length 

                axs[0].plot(x_coords, y_coords, marker='|', markersize=8, linestyle='', color=color, label=f'Trades 2 - {value}')

            axs[0].plot(df_an[('strategy', 'returns')])
            axs[0].legend(["Asset_1", "Asset_2", "Strategy"])

            axs[1].set_title("Strategy Decomposed")
            axs[1].plot(df_an[('strategy', 'returns_1')])
            axs[1].plot(df_an[('strategy', 'returns_2')])
            axs[1].legend(["Asset_1 Strategy", "Asset_2 Strategy"])

            plt.figtext(0.5, 0.95, 
            f"Trades_1: {df_an[('strategy', 'trades_1')].sum()}, Trades_2: {df_an[('strategy', 'trades_2')].sum()}, Accuracy_1: {accuracy_1}, Accuracy_2: {accuracy_2}, Accuracy: {accuracy}", 
            ha='center', va='center', fontsize=12)

            plt.savefig(f"./data/visualize/evaluation/{set_name}/tstep_{timestep}_{int(time.time())}.png")
            plt.show()

        return df_an, df_an[('strategy', 'returns')].iloc[-1], (accuracy_1, accuracy_2, accuracy)

    def fetch_100_etfs(self):
        import requests
        from bs4 import BeautifulSoup

        symbols = []
        
        # Fetch the HTML content of the page
        response = requests.get("https://etfdb.com/compare/market-cap/")
        
        if response.status_code != 200:
            logger.error("Error fetching the page")
            return
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find the table
        table = soup.find('table')
        
        if table is None:
            logger.info("Table not found")
            return
        
        # Find all rows in the table
        rows = table.find_all('tr')
        
        for row in rows:
            # Find all td elements in the row with attribute data-th="Symbol"
            tds = row.find_all('td', {'data-th': 'Symbol'})
            
            for td in tds:
                # Extract and store the symbol
                a_tag = td.find('a')
                if a_tag:
                    symbols.append(a_tag.text.strip())

        return symbols

# Function to calculate the cointegration
def calculate_cointegration(series_1, series_2, log=False):
    coint_flag = 0
    coint_res = coint(series_1, series_2)
    coint_t = coint_res[0]
    p_value = coint_res[1]
    critical_value = coint_res[2][1]
    model = sm.OLS(series_1, series_2).fit()
    hedge_ratio = model.params.iloc[0]
    coint_flag = 1 if p_value < 0.1 and coint_t < critical_value else 0
    return coint_flag, hedge_ratio, p_value

# Run Augmented Dickey-Fuller test and return if stationary
def adf_test(series, p_value_return=False):
    result = adfuller(series, autolag='AIC')
    if p_value_return:
        return result[1]
    else:
        return result[1] <= 0.05

# Convert key to positon
def key_to_position(key):
    if key == -1:
        return 'sell'
    elif key == 0:
        return 'neutral'
    else:
        return 'long'

# Convert key to trade
def key_to_trade(key):
    if key == 0:
        return 'stay'
    elif key == 1:
        return 'open/close'
    else:
        return 'switch'
