#!/usr/bin/env python

import torch as T
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from json import load   
import os

class DeepQNetwork(nn.Module):
    """
    Deep Q-Network (DQN) class for reinforcement learning.

    Attributes:
        input_dims (tuple): Dimensions of the input state.
        fc1_dims (int): Number of units in the first fully connected layer.
        fc2_dims (int): Number of units in the second fully connected layer.
        n_actions (int): Number of actions in the action space.
    """
    def __init__(self, lr: float, input_dims: tuple, fc1_dims: int, fc2_dims: int, n_actions: int):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """
        Perform forward pass through the network.

        Args:
            state (Tensor): Input state tensor.

        Returns:
            Tensor: Output actions tensor.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.output(x)
        return actions

# Load settings from settings.json.
settings_path = os.path.dirname(os.path.realpath(__file__)) 
with open (settings_path+'/settings.json', 'r') as f:
    settings = load(f)


    
class Agent():
    """
    Agent class for reinforcement learning.

    Attributes:
        gamma (float): Discount factor.
        epsilon (float): Exploration-exploitation trade-off parameter.
        lr (float): Learning rate.
        input_dims (tuple): Dimensions of the input state.
        batch_size (int): Size of the training batch.
        n_actions (int): Number of actions in the action space.
    """
    def __init__(self, gamma: float, epsilon: float, lr: float, input_dims: tuple, batch_size: int,
                 n_actions: int, max_mem_size: int = 100000, eps_end: float = 0.01, eps_dec: float = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0
        self.target_network_update_frequency = 1000

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        checkpoint = T.load(r'./catkin_ws/src/dqn_turtlebot3/models/corridor0.pth')
        self.Q_eval.load_state_dict(checkpoint['model_state_dict'])  
        self.target_network = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)



    def store_transition(self, state: np.ndarray, action: int, reward: float, state_: np.ndarray, done: bool) -> None:
        """
        Store transition in memory.

        Args:
            state (ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            state_ (ndarray): Next state.
            done (bool): Whether the episode has ended.
        """
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def choose_action(self, observation: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            observation (ndarray): Current observation.

        Returns:
            int: Selected action.
        """
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self) -> None:
        """
        Update Q-values using gradient descent.
        """
        if self.mem_counter < self.batch_size:
            return

        if self.mem_counter % self.target_network_update_frequency == 0:
            self.target_network.load_state_dict(self.Q_eval.state_dict())

        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.target_network.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

