# Utils
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Data
import time
import json

# Reinforcement Learning
import gymnasium as gym
from trading_environment import TradingEnv
from agent import Agent

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

class PPO():
    def __init__(self, data, normalization_params, loaded_timestep=0):
        # Loading and creating rl_parameters
        with open("./rl_parameters.json") as f:
            rl_parameters = json.load(f)
        with open("./rl_default_parameters.json") as f:
            rl_default_parameters = json.load(f)
        for item in rl_default_parameters.keys():
            if item not in rl_parameters.keys():
                rl_parameters[item] = rl_default_parameters[item]

        self.rl_parameters = rl_parameters
        self.rl_parameters['batch_size'] = int(self.rl_parameters['num_envs'] * self.rl_parameters['num_steps'])
        if self.rl_parameters['minibatch_size'] > self.rl_parameters['batch_size']:
            raise MiniBatchSizeIncompatible("The size of minibatches can be higher than batch_size (batch_size = num_steps * num_envs)")
        self.data = data
        self.normalization_params = normalization_params
        self.loaded_timestep = loaded_timestep

        random.seed(self.rl_parameters['seed'])
        np.random.seed(self.rl_parameters['seed'])
        torch.manual_seed(self.rl_parameters['seed'])
        torch.backends.cudnn.deterministic = rl_parameters['torch_deterministic']

        # Creating agent
        self.time_id = int(time.time())
        self.envs = gym.vector.SyncVectorEnv([
            self.make_env()
        for i in range(self.rl_parameters['num_envs'])])
        assert isinstance(self.envs.single_action_space, gym.spaces.MultiDiscrete), "only multi discrete action space is supported"

        self.device = torch.device('cuda:0' if torch.cuda.is_available() and self.rl_parameters['cuda'] else 'cpu')
        self.agent = Agent(self.envs, self.rl_parameters, self.time_id, self.normalization_params, self.data).to(self.device)

        self.run_load = False

    def learn(self):
        device = self.device
        envs = self.envs

        # Create writer object
        self.writer = SummaryWriter(f"runs/{self.rl_parameters['exp_name']}_{self.time_id}")

        if not self.run_load:
            # Add hyperparameters to the writer
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.rl_parameters.items()])),
            )

            # Add the model architecture to tensorboard
            dummy_input = torch.randn(1, self.rl_parameters['seqlength'], len(self.data.columns) - 2) # Remove two to remove the two target columns
            self.writer.add_graph(self.agent.actor, dummy_input)
            self.writer.add_graph(self.agent.critic, dummy_input)

        self.agent.actor.train() # here only for understanding
        self.agent.critic.train() # here only for understanding
        optimizer = optim.Adam(self.agent.parameters(), lr=self.rl_parameters['learning_rate'], eps=1e-5, weight_decay=self.rl_parameters['weight_decay'])

        # Save agent weights into tensorboard
        for name, weight in self.agent.actor.named_parameters():
            name = f"actor - {name}"
            self.writer.add_histogram(name, weight, 0)

            # Check if the gradient exists
            if weight.grad is not None:
                self.writer.add_histogram(f'{name}.grad', weight.grad, 0)
        for name, weight in self.agent.critic.named_parameters():
            name = f"critic - {name}"
            self.writer.add_histogram(name, weight, 0)

            # Check if the gradient exists
            if weight.grad is not None:
                self.writer.add_histogram(f'{name}.grad', weight.grad, 0)
        
        # ALGO Logic: Storage setup
        obs = torch.zeros((self.rl_parameters['num_steps'], self.rl_parameters['num_envs']) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((self.rl_parameters['num_steps'], self.rl_parameters['num_envs']) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((self.rl_parameters['num_steps'], self.rl_parameters['num_envs'])).to(device)
        rewards = torch.zeros((self.rl_parameters['num_steps'], self.rl_parameters['num_envs'])).to(device)
        dones = torch.zeros((self.rl_parameters['num_steps'], self.rl_parameters['num_envs'])).to(device)
        values = torch.zeros((self.rl_parameters['num_steps'], self.rl_parameters['num_envs'])).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()

        # Do not reset the environment during the training process to prevent terminating the environment by forcing it and not by making the env say done
        if not self.run_load:
            next_obs = torch.Tensor(envs.reset()[0]).to(device)
        else:
            next_obs_list = [env.unwrapped.next_observation().tolist() for env in envs.envs]

            next_obs = torch.Tensor(next_obs_list).to(device)
        next_done = torch.zeros(self.rl_parameters['num_envs']).to(device)
        num_updates = self.rl_parameters['total_timesteps'] // self.rl_parameters['batch_size']

        for update in range(1, num_updates + 1):
            if self.rl_parameters['anneal_lr']:
                frac = 1 - (update - 1) / num_updates
                lrnow = frac * self.rl_parameters['learning_rate']
                optimizer.param_groups[0]['lr'] = lrnow

            for step in range(self.rl_parameters['num_steps']):
                global_step += self.rl_parameters['num_envs']
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO Logic: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = [term or trunc for term, trunc in zip(terminated, truncated)]
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                if "final_info" in infos:
                    for i, info in enumerate(infos["final_info"]):
                        if info and "episode" in info:
                            logger.info(f"global_step={self.loaded_timestep + global_step}, local_step={global_step}, episodic_return={info['episode']['r']}")
                            self.writer.add_scalar(f"charts/episodic_return_env_{i}", info["episode"]["r"], self.loaded_timestep + global_step)
                            self.writer.add_scalar(f"charts/episodic_length_env_{i}", info["episode"]["l"], self.loaded_timestep + global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                if self.rl_parameters['gae']:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(self.rl_parameters['num_steps'])):
                        if t == self.rl_parameters['num_steps'] - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + self.rl_parameters['gamma'] * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + self.rl_parameters['gamma'] * self.rl_parameters['gae_lambda'] * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(self.rl_parameters['num_steps'])):
                        if t == self.rl_parameters['num_steps'] - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + self.rl_parameters['gamma'] * nextnonterminal * next_return
                    advantages = returns - values

            # NOTE: this is the part where the real learning begins
            # Flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.rl_parameters['batch_size'])
            clipfracs = []
            for epoch in range(self.rl_parameters['update_epochs']):
                np.random.shuffle(b_inds)
                for start in range(0, self.rl_parameters['batch_size'], self.rl_parameters['minibatch_size']):
                    end = start + self.rl_parameters['minibatch_size']
                    mb_inds = b_inds[start:end]

                    # Take probability for the same action and then calculate the logratio
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds].T)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.rl_parameters['clip_coef']).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.rl_parameters['norm_adv']:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.rl_parameters['clip_coef'], 1 + self.rl_parameters['clip_coef'])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.rl_parameters['clip_vloss']:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.rl_parameters['clip_coef'],
                            self.rl_parameters['clip_coef'],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.rl_parameters['ent_coef'] * entropy_loss + v_loss * self.rl_parameters['vf_coef']

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.rl_parameters['max_grad_norm'])
                    optimizer.step()

                # Implementing early stopping if the model is changing too fast
                # Implemented at batch level, can also be implemenet at mini batch level
                if self.rl_parameters['target_kl'] is not None:
                    if approx_kl > self.rl_parameters['target_kl']:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], self.loaded_timestep + global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.loaded_timestep + global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.loaded_timestep + global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.loaded_timestep + global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.loaded_timestep + global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.loaded_timestep + global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.loaded_timestep + global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.loaded_timestep + global_step)
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), self.loaded_timestep + global_step)

            # Save agent weights into tensorboard
            # NOTE: this is after every epochs and is not considering the possible variations between the epochs
            for name, weight in self.agent.actor.named_parameters():
                name = f"actor - {name}"
                self.writer.add_histogram(name, weight, global_step)
                self.writer.add_histogram(f'{name}.grad', weight.grad, global_step)
            for name, weight in self.agent.critic.named_parameters():
                name = f"critic - {name}"
                self.writer.add_histogram(name, weight, global_step)
                self.writer.add_histogram(f'{name}.grad', weight.grad, global_step)

        # Because in manager I am going to evaluate the model on train and validation set, this part is fundamental
        self.agent.actor.eval()
        self.agent.critic.eval()

        envs.close()
        self.writer.close()
    
    def make_env(self):
        def thunk():
            env = TradingEnv(self.data, self.normalization_params, self.rl_parameters)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(self.rl_parameters['seed'])
            env.observation_space.seed(self.rl_parameters['seed'])
            return env
        return thunk
