# Reinforcement Learning
import gymnasium as gym

# Utils
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, normalization_params, rl_parameters, initial_balance=100, kill_threshold=0):
        super(TradingEnv, self).__init__()

        # Data and parameters
        self.data = data.copy()
        self.rl_parameters = rl_parameters
        self.initial_balance = initial_balance
        self.kill_threshold = kill_threshold
        self.max_steps = len(data) - rl_parameters['seqlength'] - 1
        self.normalization_params = normalization_params

        # Action space: Buy, Neutral, Sell for both base_pair and compare_pair (6 actions)
        # Needed to use box because i couldn't find a way to have two possible actions in just one space
        self.action_space = gym.spaces.MultiDiscrete([3, 3])

        # Observation space: sequence of data
        if self.rl_parameters['architecture_type'] == 'LSTM':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.rl_parameters['seqlength'], self.data.shape[1] - 2), dtype=np.float32)
        elif self.rl_parameters['architecture_type'] == 'FFN':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.shape[1] - 2,), dtype=np.float32)
        else:
            raise ValueError("Invalid architecture type in rl_parameters")

        if self.rl_parameters['seed']:
            np.random.seed(self.rl_parameters['seed'])

        # Calculate the mean and std of the returns
        if self.rl_parameters['env_log_returns']:
            self.mean_returns = (np.log(self.data[('asset_1', 'TARGET')] + 1).mean() + np.log(self.data[('asset_2', 'TARGET')] + 1).mean()) / 2
            self.std_returns = (np.log(self.data[('asset_1', 'TARGET')] + 1).std() + np.log(self.data[('asset_2', 'TARGET')] + 1).std()) / 2
        else:
            self.mean_returns =  (self.data[('asset_1', 'TARGET')].mean() + self.data[('asset_2', 'TARGET')].mean()) / 2
            self.std_returns =  (self.data[('asset_1', 'TARGET')].std() + self.data[('asset_2', 'TARGET')].std()) / 2

        self.reset()

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.current_step = 0
        self.done = False
        self.positions_long_1 = 0.0001
        self.positions_short_1 = 0.0001
        self.positions_long_2 = 0.0001
        self.positions_short_2 = 0.0001 # prevent divison by 0
        self.positions_returns = []
        self.episode_reward = 0
        self.base_position_before = 0
        self.compare_position_before = 0

        obs = self.next_observation()

        return obs, {}

    def step(self, action):

        self.current_step_returns = {
            "base": 0,
            "compare": 0
        }

        action_base = action[0]
        action_compare = action[1]

        self.action_base = action_base
        self.action_compare = action_compare

        base_pair_weight = .5
        # Calculate returns based on action for base_pair
        action_base_return = self.data.iloc[self.current_step + self.rl_parameters['seqlength']][('asset_1', 'TARGET')] * (base_pair_weight) if not self.rl_parameters['env_log_returns'] else np.log(self.data.iloc[self.current_step + self.rl_parameters['seqlength']][('asset_1', 'TARGET')] + 1) * (base_pair_weight) 
        if action_base == 0:  # Buy base_pair
            self.positions_long_1 += 1
            self.balance = self.balance + self.balance * action_base_return
            self.positions_returns.append(action_base_return)
            self.current_step_returns['base'] = action_base_return
        elif action_base  == 1: # Neutral base_pair
            pass
        elif action_base == 2:  # Sell base_pair
            self.positions_short_1 += 1
            self.balance = self.balance + self.balance * action_base_return * -1
            self.positions_returns.append(action_base_return * -1)
            self.current_step_returns['base'] = action_base_return * -1
        
        # # Calculate returns based on action for compare_pair
        action_compare_return = self.data.iloc[self.current_step + self.rl_parameters['seqlength']][('asset_2', 'TARGET')] * (1 - base_pair_weight) if not self.rl_parameters['env_log_returns'] else np.log(self.data.iloc[self.current_step + self.rl_parameters['seqlength']][('asset_2', 'TARGET')] + 1) * (1 - base_pair_weight) 
        if action_compare == 0:  # Buy compare_pair
            self.positions_long_2 += 1
            self.balance = self.balance + self.balance * action_compare_return
            self.positions_returns.append(action_compare_return)
            self.current_step_returns['compare'] = action_compare_return
        elif action_compare == 1: # Neutral compare_pair
            pass
        elif action_compare == 2:  # Sell compare_pair
            self.positions_short_2 += 1
            self.balance = self.balance + self.balance * action_compare_return * -1
            self.positions_returns.append(action_compare_return * -1)
            self.current_step_returns['compare'] = action_compare_return * -1

        reward = self._calculate_reward()
        self.episode_reward += reward

        # Check if done
        self.truncated = self.balance <= self.kill_threshold * self.initial_balance

        # Move to the next step
        self.terminated = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.terminated = True

        obs = self.next_observation()

        return obs, reward, self.terminated, self.truncated, {}

    def next_observation(self):
        columns_to_pick = []
        for col in list(self.data.columns):
            if col[1] != 'TARGET':
                columns_to_pick.append(col)

        if self.rl_parameters['architecture_type'] == 'LSTM':
            obs = self.data.iloc[self.current_step:self.current_step + self.rl_parameters['seqlength']][columns_to_pick]
        elif self.rl_parameters['architecture_type'] == 'FFN':
           obs = self.data.iloc[self.current_step + self.rl_parameters['seqlength'] - 1][columns_to_pick]
        else:
            raise ValueError("Invalid architecture type in rl_parameters")

        # Needed later to calculate if we are in a downtrend
        if self.rl_parameters['architecture_type'] == 'LSTM':
            self.current_last_row = obs.iloc[-1].copy()
        elif self.rl_parameters['architecture_type'] == 'FFN':
           self.current_last_row = obs.copy()
        obs = obs.values

        # Add noise to the data
        if self.rl_parameters['gaussian_noise']:
            obs = obs + np.random.normal(size=obs.shape) * self.rl_parameters['gaussian_noise']
        if self.rl_parameters['uniform_noise']:
            obs = obs + np.random.uniform(size=obs.shape) * self.rl_parameters['uniform_noise']
        
        return obs

    def _calculate_reward(self):
        returns = sum(self.current_step_returns.values())
        reward = returns

        # Scale the reward to have a mean of one to facilitate the calculations
        reward = reward / self.mean_returns

        # Add a value if the bot was right of the direction, this is to make the bot not rely on few trades but on many, increasing generalization
        if self.current_step_returns['base'] >= 0:
            reward += 1
        else:
            reward -= 1
        if self.current_step_returns['compare'] >= 0:
            reward += 1
        else:
            reward -= 1

        # Punish only long or short actions
        # NOTE EXP: It look likes this is fundamental for the model to learn and not get stuck in a local minima (no noise, no dropout, LSTM)
        margin_long = .2
        margin_short = .2
        step_min = 15
        return_min = 15
        if self.action_base == 0 and (self.positions_long_1 / (self.positions_short_1 + self.positions_long_1) > 1 - margin_long or self.positions_long_1 / (self.positions_short_1 + self.positions_long_1) < margin_long) and self.current_step >= step_min and len(self.positions_returns) >= return_min:
            reward -= 0.5
        if self.action_compare == 0 and (self.positions_long_2 / (self.positions_short_2 + self.positions_long_2) > 1 - margin_long or self.positions_long_2 / (self.positions_short_2 + self.positions_long_2) < margin_long) and self.current_step >= step_min and len(self.positions_returns) >= return_min:
            reward -= 0.5
        if self.action_base == 2 and (self.positions_short_1 / (self.positions_short_1 + self.positions_long_1) > 1 - margin_short or self.positions_short_1 / (self.positions_short_1 + self.positions_long_1) < margin_short) and self.current_step >= step_min and len(self.positions_returns) >= return_min:
            reward -= 0.5
        if self.action_compare == 2 and (self.positions_short_2 / (self.positions_short_2 + self.positions_long_2) > 1 - margin_short or self.positions_short_2 / (self.positions_short_2 + self.positions_long_2) < margin_short) and self.current_step >= step_min and len(self.positions_returns) >= return_min:
            reward -= 0.5

        # Punish enter % less than margin% and higher margin%
        # NOTE EXP: It look likes this is fundamental for the model to learn and not get stuck in a local minima (no noise, no dropout, LSTM)
        margin_no_trades = .2
        margin_many_trades = 0
        step_min = 15
        return_min = 15
        # Punish not taking any trade
        if self.action_base == 1 and (self.current_step + 1) * margin_no_trades > self.positions_short_1 + self.positions_long_1 and self.current_step >= step_min and len(self.positions_returns) >= return_min:
            reward -= 1
        if self.action_compare == 1 and (self.current_step + 1) * margin_no_trades > self.positions_short_2 + self.positions_long_2 and self.current_step >= step_min and len(self.positions_returns) >= return_min:
            reward -= 1
        # Punish taking too many trades
        if (self.action_base == 0 or self.action_base == 2) and (self.current_step + 1) * (1 - margin_many_trades) < self.positions_short_1 + self.positions_long_1 and self.current_step >= step_min and len(self.positions_returns) >= return_min:
            reward -= .3
        if (self.action_compare == 0 or self.action_compare == 2) and (self.current_step + 1) * margin_many_trades < self.positions_short_2 + self.positions_long_2 and self.current_step >= step_min and len(self.positions_returns) >= return_min:
            reward -= .3

        # Check if normalization_params has a key "max" if yes the normalization must be minmax
        # Transform MA_diff into is real value to know if the market is in a downtrend and if the model is profitable in a downtrend give a reward
        if 'max' in self.normalization_params:
            real_madiff_1 = self.current_last_row[('asset_1', 'MA_diff')] * (self.normalization_params['max'][('asset_1', 'MA_diff')] - self.normalization_params['min'][('asset_1', 'MA_diff')]) + self.normalization_params['min'][('asset_1', 'MA_diff')]
            real_madiff_2 = self.current_last_row[('asset_2', 'MA_diff')] * (self.normalization_params['max'][('asset_2', 'MA_diff')] - self.normalization_params['min']('asset_2', 'MA_diff')) + self.normalization_params['min'][('asset_2', 'MA_diff')]
            if real_madiff_1 < 0 and real_madiff_2 < 0:
                reward =+ .5
        else:
            real_madiff_1 = self.current_last_row[('asset_1', 'MA_diff')] * self.normalization_params['std'][('asset_1', 'MA_diff')] + self.normalization_params['mean'][('asset_1', 'MA_diff')]
            real_madiff_2 = self.current_last_row[('asset_2', 'MA_diff')] * self.normalization_params['std'][('asset_2', 'MA_diff')] + self.normalization_params['mean'][('asset_2', 'MA_diff')]

            if real_madiff_1 < 0 and real_madiff_2 < 0:
                reward =+ .5

        # If the position before is different from the position after reduce the reward to prevent spending too much on commissions
        if abs(self.base_position_before - ((self.current_step_returns['base'] - 1) * -1)) == 1:
            reward =- .05
        if abs(self.base_position_before - ((self.current_step_returns['base'] - 1) * -1)) == 2:
            reward =- .05
        if abs(self.compare_position_before - ((self.current_step_returns['compare'] - 1) * -1)) == 1:
            reward =- .05
        if abs(self.compare_position_before - ((self.current_step_returns['compare'] - 1) * -1)) == 2:
            reward =- .05

        # Rewarding a low std
        std_mult_factor = 2
        step_min = 0
        return_min = 16
        if len(self.positions_returns) > 1 and self.current_step > step_min and len(self.positions_returns) >= return_min:
            # reward = reward / (np.std(self.positions_returns / self.mean_returns) * std_mult_factor) # Dividing by the mean returns to make sure the mean of the std is one
            # Take the std of the last N positions to make the recent times more relevant
            reward = reward / (np.std((self.positions_returns / self.mean_returns)[-return_min:]) * std_mult_factor)
        else:
            # reward = reward / (self.std_returns / self.mean_returns * std_mult_factor)
            reward = reward / (np.std((self.data[('asset_1', 'TARGET')] + self.data[('asset_2', 'TARGET')]) / 2 / self.mean_returns) * std_mult_factor)
            
        self.base_position_before = self.current_step_returns['base']
        self.compare_position_before = self.current_step_returns['compare']

        return reward


    def render(self, mode='human'):
        pass

    def close(self):
        pass