import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import gym
import pybullet_envs

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []

    def add(self, state, action, logprob, reward, is_terminal, state_value=None):
        self.states.append(torch.tensor(state, dtype=torch.float32))
        self.actions.append(torch.tensor(action, dtype=torch.float32))
        self.logprobs.append(torch.tensor(logprob, dtype=torch.float32))
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        if state_value is not None:
            self.state_values.append(torch.tensor(state_value, dtype=torch.float32))

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []

    def sample(self):
        return (self.states, self.actions, self.logprobs, self.rewards, self.state_values, self.is_terminals)

# ActorCritic Model
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
            self.action_logstd = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            return action_mean, self.action_logstd


    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean, action_logstd = self.act(state)
            action_std = action_logstd.exp()
            dist = Normal(action_mean, action_std)
            action_logprobs = dist.log_prob(action).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            state_values = self.critic(state)
            return action_logprobs, state_values, dist_entropy


    def get_log_prob(self, state, action):
        if self.has_continuous_action_space:
            action_mean, action_logstd = self.act(state)
            action_std = action_logstd.exp()
            dist = Normal(action_mean, action_std)
            action_logprobs = dist.log_prob(torch.FloatTensor(action).to(device)).sum(dim=-1)
            return action_logprobs

# PPO Policy
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if self.has_continuous_action_space:
                action_mean, action_logstd = self.policy_old.act(state)
                action_std = action_logstd.exp()
                action = action_mean + action_std * torch.randn_like(action_mean)
                action = action.cpu().numpy()
                action_logprob = self.policy_old.get_log_prob(state, action).cpu().numpy()
        return action, action_logprob

    def train(self, replay_buffer, batch_size):
        # Convert replay buffer items to tensors
        states = torch.stack(replay_buffer.states).to(device)
        actions = torch.stack(replay_buffer.actions).to(device)
        log_probs_old = torch.stack(replay_buffer.logprobs).to(device)
        log_probs_old = log_probs_old.detach()
        rewards = torch.tensor(replay_buffer.rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(replay_buffer.is_terminals, dtype=torch.float32).to(device)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Compute advantages (if applicable)
        if hasattr(replay_buffer, "state_values"):
            state_values = torch.stack(replay_buffer.state_values).to(device)
            advantages = rewards - state_values.detach()
        else:
            advantages = rewards

        # Perform training iterations
        for _ in range(self.K_epochs):
            # Evaluate policy for the current states and actions
            log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)

            # Compute surrogate loss
            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Policy loss
            loss = -torch.min(surr1, surr2).mean()
            value_loss = self.MseLoss(state_values, rewards)
            entropy_loss = dist_entropy.mean()

            total_loss = loss + 0.5 * value_loss - 0.01 * entropy_loss

            # Optimize policy
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Clear the buffer after training
        replay_buffer.clear()

    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action,_ = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

# Training and Evaluation
if __name__ == "__main__":
    import os
    from datetime import datetime
    import numpy as np

    env_name = "Pendulum-v1"
    seed = 0
    start_timesteps = int(25e3)
    max_timesteps = int(1e5)
    eval_freq = int(5e3)
    save_model = True
    batch_size = 64

    # Directory setup
    os.makedirs("./results", exist_ok=True)
    if save_model:
        os.makedirs("./models", exist_ok=True)

    # Environment setup
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # PPO Initialization
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    has_continuous_action_space = True
    action_std_init = 0.6
    lr_actor = 3e-4
    lr_critic = 1e-3
    gamma = 0.99
    K_epochs = 40
    eps_clip = 0.2

    policy = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init)
    evaluations = []

    # Training Variables
    state, done = env.reset(), False
    replay_buffer = RolloutBuffer()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT):", start_time)

    for t in range(max_timesteps):
        episode_timesteps += 1

        # Select action
        if t < start_timesteps:
            action = env.action_space.sample()
            log_prob = 0
        else:
            action, log_prob = policy.select_action(state)

        # Environment step
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env.spec.max_episode_steps else 0

        # Store in buffer
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            state_value = policy.policy_old.critic(state_tensor)
        replay_buffer.add(state, action, log_prob, reward, done_bool, state_value)

        state = next_state
        episode_reward += reward

        # Train if buffer has enough samples
        if t >= start_timesteps and len(replay_buffer.states) >= batch_size:
            policy.train(replay_buffer, batch_size)
            replay_buffer.clear()  # Clear buffer after training

        # End of episode
        if done:
            print(f"Total T: {t+1}, Episode Num: {episode_num+1}, Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluation
        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_policy(policy, env_name, seed))
            np.save(f"./results/{env_name}_PPO", evaluations)
            if save_model:
                policy.save(f"./models/{env_name}_PPO")

    print("Training finished.")
