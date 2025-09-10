import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from trading_environment import TradingEnv

import io
import json
import zipfile
import inspect # Add .py reward function to zip saved file
import os

class Critic(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.critic = nn.ModuleDict()
        self.critic['Conv1d-1'] = layer_init(nn.Conv1d(envs.single_observation_space.shape[1], 64, kernel_size=3, padding=1))
        self.critic['Dropout-1'] = nn.Dropout(p=0.5)
        self.critic['Maxpool-1'] = nn.MaxPool1d(kernel_size=2, stride=2)
        self.critic['Tanh-1'] = nn.Tanh()
        self.critic['LSTM-1'] = layer_init_lstm(nn.LSTM(64, 64, num_layers=3, bidirectional=True, dropout=0.5, batch_first=True))
        self.critic['Tanh-2'] = nn.Tanh()
        self.critic['Linear-1'] = layer_init(nn.Linear(in_features=128, out_features=64))
        self.critic['Dropout-2'] = nn.Dropout(p=0.5)
        self.critic['Tanh-3'] = nn.Tanh()
        self.critic['Linear-2'] = layer_init(nn.Linear(in_features=64, out_features=1), std=1)

    def forward(self, x):
        for key, layer in self.critic.items():
            if issubclass(type(layer), nn.Conv1d):
                x = x.transpose(1, 2)
                x = layer(x)
            elif issubclass(type(layer), nn.MaxPool1d):
                x = layer(x)
                x = x.transpose(1, 2)
            else:
                x = layer(x)
            # Formatting in the correct way for rnn layers
            if issubclass(type(layer), nn.modules.rnn.RNNBase):
                x = x[0]
                if(len(x.shape) == 3):
                    x = x[:, -1, :]
                else:
                    x = x[-1, :]
        return x

class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.actor = nn.ModuleDict()
        self.actor['Conv1d-1'] = layer_init(nn.Conv1d(envs.single_observation_space.shape[1], 64, kernel_size=3, padding=1))
        self.actor['Dropout-1'] = nn.Dropout(p=0.5)
        self.actor['Maxpool-1'] = nn.MaxPool1d(kernel_size=2, stride=2)
        self.actor['Tanh-1'] = nn.Tanh()
        self.actor['LSTM-1'] = layer_init_lstm(nn.LSTM(64, 64, num_layers=3, bidirectional=True, dropout=0.5, batch_first=True))
        self.actor['Tanh-2'] = nn.Tanh()
        self.actor['Linear-1'] = layer_init(nn.Linear(in_features=128, out_features=64))
        self.actor['Dropout-2'] = nn.Dropout(p=0.5)
        self.actor['Tanh-3'] = nn.Tanh()
        self.actor['Linear-2'] = layer_init(nn.Linear(in_features=64, out_features=envs.single_action_space.nvec.sum()), std=0.1)

    def forward(self, x):
        for key, layer in self.actor.items():
            if issubclass(type(layer), nn.Conv1d):
                x = x.transpose(1, 2)
                x = layer(x)
            elif issubclass(type(layer), nn.MaxPool1d):
                x = layer(x)
                x = x.transpose(1, 2)
            else:
                x = layer(x)
            # Formatting in the correct way for rnn layers
            if issubclass(type(layer), nn.modules.rnn.RNNBase):
                x = x[0]
                if(len(x.shape) == 3):
                    x = x[:, -1, :]
                else:
                    x = x[-1, :]
        return x

class Agent(nn.Module):
    def __init__(self, envs, rl_parameters, time_id, normalization_params, data):
        super().__init__()
        
        self.nvec = envs.single_action_space.nvec
        self.rl_parameters = rl_parameters
        self.time_id = time_id
        self.normalization_params = normalization_params
        self.data = data
        
        self.critic = Critic(envs)

        self.actor = Actor(envs)

    def get_value(self, x):
        return self.critic(x)

    def get_logits(self, x):
        return self.actor(x)

    def get_action_and_value(self, x, action=None, deterministic = False):
        logits = self.get_logits(x)
        split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            if self.rl_parameters['deterministic'] or deterministic:
                action = torch.stack([categorical.mode for categorical in multi_categoricals])
            else:
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action.T, logprob.sum(0), entropy.sum(0), self.get_value(x)

    def save_agent(self, timestep, non_stationary = None, custom_time_id = None, loaded_time_id = None):
        actor_buffer = io.BytesIO()
        critic_buffer = io.BytesIO()
        torch.save(self.actor.state_dict(), actor_buffer)
        torch.save(self.critic.state_dict(), critic_buffer)
        params_buffer = io.BytesIO()
        parameters_to_upload = self.rl_parameters

        # Change start and end date and add normalization params
        parameters_to_upload['start_date'] = str(self.data.index[0])
        parameters_to_upload['end_date'] = str(self.data.index[-1])

        # Convert the index tuples in normalization params into string because not accepted by json format
        str_index_norm_params = {}
        for i, param in enumerate(self.normalization_params.keys()):
            str_index_norm_params[param] = {}
        for i, param in enumerate(self.normalization_params.keys()):
            for key in self.normalization_params[param].keys():
                str_index_norm_params[param][str(key)] = self.normalization_params[param][key]

        parameters_to_upload['normalization_params'] = str_index_norm_params

        # Add timestep
        parameters_to_upload['local_timestep'] = timestep

        # Add network architecture
        parameters_to_upload['architecture'] = str(self)

        # Add the loaded time id
        parameters_to_upload['loaded_time_id'] = loaded_time_id

        params_buffer.write(json.dumps(parameters_to_upload, indent = 4).encode())
        
        # Get the source code of the function
        function_code = inspect.getsource(TradingEnv._calculate_reward)

        # Convert the function code to a byte buffer
        function_buffer = io.BytesIO()
        function_buffer.write(function_code.encode())

        if non_stationary:
            # Save non stationary features
            non_stationary_buffer = io.BytesIO()
            non_stationary_buffer.write(str(non_stationary).encode())
            non_stationary_buffer.seek(0)

        # Reset buffers position to the beginning
        actor_buffer.seek(0)
        critic_buffer.seek(0)
        params_buffer.seek(0)
        function_buffer.seek(0)

        # Ensure the directory exists
        os.makedirs(f"./models/{self.rl_parameters['exp_name']}_{self.time_id}", exist_ok=True)
        
        real_timestep_length = (self.rl_parameters['total_timesteps'] // (int(self.rl_parameters['num_envs'] * self.rl_parameters['num_steps'])) * int(self.rl_parameters['num_envs'] * self.rl_parameters['num_steps']))
        with zipfile.ZipFile(f"./models/{self.rl_parameters['exp_name']}_{self.time_id if not custom_time_id else custom_time_id}/ppo_{timestep + real_timestep_length}.zip", 'w') as zipf:
            zipf.writestr('actor_model.pth', actor_buffer.read())
            zipf.writestr('critic_model.pth', critic_buffer.read())
            zipf.writestr('network_parameters.json', params_buffer.read())
            zipf.writestr('reward_function.py', function_buffer.read())
            if non_stationary:
                zipf.writestr('non_stationary_features.txt', non_stationary_buffer.read())

    def load_agent(self, timestep, custom_time_id = None):
        with zipfile.ZipFile(f"./models/{self.rl_parameters['exp_name']}_{self.time_id if not custom_time_id else custom_time_id}/ppo_{timestep}.zip", 'r') as zipf:
            # Load actor model
            with zipf.open('actor_model.pth') as f:
                self.actor.load_state_dict(torch.load(io.BytesIO(f.read())))

            # Load critic model
            with zipf.open('critic_model.pth') as f:
                self.critic.load_state_dict(torch.load(io.BytesIO(f.read())))


def layer_init(layer, std=np.sqrt(2), bias_const=0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_init_lstm(layer):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, 1.0)
    return layer

# NOTE: this is the lstm base architecture model
# self.critic = nn.ModuleDict()
# self.critic['LSTM-1'] = layer_init_lstm(nn.LSTM(envs.single_observation_space.shape[1], 64, num_layers=2, dropout=0.5, batch_first=True))
# self.critic['Tanh-1'] = nn.Tanh()
# self.critic['Linear-1'] = layer_init(nn.Linear(in_features=64, out_features=64))
# self.critic['Dropout-1'] = nn.Dropout(p=.5)
# self.critic['Tanh-2'] = nn.Tanh()
# self.critic['Linear-2'] = layer_init(nn.Linear(in_features=64, out_features=1), std=1)

# self.actor = nn.ModuleDict()
# self.actor['LSTM-1'] = layer_init_lstm(nn.LSTM(envs.single_observation_space.shape[1], 64, num_layers=2, dropout=0.5, batch_first=True))
# self.actor['Tanh-1'] = nn.Tanh()
# self.actor['Linear-1'] = layer_init(nn.Linear(in_features=64, out_features=64))
# self.critic['Dropout-1'] = nn.Dropout(p=.5)
# self.actor['Tanh-2'] = nn.Tanh()
# self.actor['Linear-2'] = layer_init(nn.Linear(in_features=64, out_features=self.nvec.sum()), std=.01)