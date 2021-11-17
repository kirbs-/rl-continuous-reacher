from dataclasses import dataclass
from typing import Any
import model
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import copy


@dataclass
class Params:
    batch_size: int
    gamma: float
    tau: float
    learning_rate_actor: float
    learning_rate_critic: float
    state_dim: int
    action_cnt: int
    target_update_step: int
    seed: int


@dataclass
class Experience:
    state: Any
    action: Any
    reward: Any
    next_state: Any
    done: Any


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(4)
        self.state = x + dx
        return self.state
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Agent interactions with a Unity environment.
    
    Params:
        params (Params): Agent parameter dataclass. 

    Attributes:
        policy_dqn (Torch.Module):
        target_dqn (Torch.Module):
        optimizer (Optimizer):
        step_cnt (int): 
    
    """

    def __init__(self, params):
        self.params = params

        # setup Actor neural net w/ target NN
        self.policy_actor = model.Actor(self.params.state_dim, self.params.action_cnt, 256, self.params.seed).to(device)
        self.target_actor = model.Actor(self.params.state_dim, self.params.action_cnt, 256, self.params.seed).to(device)
        self.optimizer_actor = optim.Adam(self.policy_actor.parameters(), lr=self.params.learning_rate_actor)

        # setup Critic neural net w/ target NN
        self.value_critic = model.Critic(self.params.state_dim, self.params.action_cnt, 256, self.params.seed).to(device)
        self.target_critic = model.Critic(self.params.state_dim, self.params.action_cnt, 256, self.params.seed).to(device)
        self.optimizer_critic = optim.Adam(self.value_critic.parameters(), lr=self.params.learning_rate_critic, weight_decay=0)

        self.noise = OUNoise(self.params.action_cnt, self.params.seed)

        self.step_cnt = 0
        self.buffer = []

        # store losses
        self.actor_losses = []
        self.critic_losses = []
        self.learn_cnt = 0
        self.step_cnt = 0

    def step(self, state, action, reward, next_state, done):
        # add experience into replay buffer
        self.buffer.append(Experience(state, action, reward, next_state, done))

        # update step cnt
        self.step_cnt += 1

        # update target network after target_update_step_size and if replay buffer has enough data
        if self.step_cnt % self.params.target_update_step == 0 and len(self.buffer) > self.params.batch_size:
            self.learn()

    def act(self, state):
        # update policy network
        state = torch.from_numpy(state).float().to(device)
        self.policy_actor.eval()
        with torch.no_grad():
            action = self.policy_actor(state).cpu().data.numpy()
        self.policy_actor.train()

        # if add_noise:
        action += self.noise.sample()
        return np.clip(action, -1, 1)

        # Epsilon greedy action selection
        # if random.random() > self.params.epsilon:
        #     self.params.epsilon = max(0.01, 0.999 * self.params.epsilon) # decrease epsilon
        #     return np.argmax(action_values.cpu().data.numpy())
        # else:
        #     self.params.epsilon = max(0.01, 0.999 * self.params.epsilon) # decrease epsilon
        #     return random.randrange(self.params.action_cnt)

    def reset(self):
        self.noise.reset()

    def learn(self):
        self.learn_cnt += 1

        # sample batch from replay buffer
        experience_batch = random.sample(self.buffer, k=self.params.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experience_batch])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience_batch])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience_batch])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience_batch])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience_batch]).astype(np.uint8)).float().to(device)

        # update target network
        # Get predicted Q values (for next states) from target critic and next action states from target actor.
        actions_next = self.target_actor(next_states)
        targets_next = self.target_critic(next_states, actions_next)

        # Compute Q targets for current states 
        targets = rewards + (self.params.gamma * targets_next * (1 - dones))

        # Get expected Q values from local model
        expected = self.value_critic(states, actions)

        # Compute mean squared error
        loss_critic = F.mse_loss(expected, targets)
        self.critic_losses.append(loss_critic)

        # Minimize Critic loss
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm(self.value_critic.parameters(), 1)
        self.optimizer_critic.step()

        # update actor DNN
        next_actions = self.policy_actor(states)
        loss_actor = -self.value_critic(states, next_actions).mean()
        self.actor_losses.append(loss_actor)

        # minimize Actor loss
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        # copy policy weights to target_network w/ soft update
        for target_param, local_param in zip(self.target_critic.parameters(), self.value_critic.parameters()):
            target_param.data.copy_(self.params.tau*local_param.data + (1.0-self.params.tau) * target_param.data)

        # copy policy weights to target_network w/ soft update
        for target_param, local_param in zip(self.target_actor.parameters(), self.policy_actor.parameters()):
            target_param.data.copy_(self.params.tau*local_param.data + (1.0-self.params.tau) * target_param.data)