#!/home/void/kan/pykan/bin/python3

from kan import KAN
import torch.optim as optim
import torch.nn as nn
import torch as T
import os
from json import load
import numpy as np
 

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
        self.device = T.device('cpu')
        self.Q_eval = KAN(width = [*input_dims, 8, n_actions], grid = 5, k=3, device = self.device)
        self.target_network = KAN(width = [*input_dims, 8, n_actions], grid = 5, k=3, device=self.device)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.optimizer = T.optim.Adam(self.Q_eval.parameters(), lr = self.lr)
      



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

    def choose_action(self, observation: np.array) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            observation (ndarray): Current observation.

        Returns:
            int: Selected action.
        """
        if np.random.random() > self.epsilon:
            state = T.tensor(observation).unsqueeze(dim=0).to(self.device)
            actions = self.Q_eval(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def load_models(self, model_path=None):
        print('... loading models...')
        checkpoint = T.load('/home/void/catkin_ws/src/kanqn/models/continual_learning1.pth')
        self.Q_eval.load_state_dict(checkpoint["model_state_dict"])
        self.target_network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def initialize_models(self):
        print('... initializing models...')
        dummy_input = T.zeros(12).unsqueeze(dim=0)
        with T.no_grad():
            action = self.Q_eval(dummy_input)
            action_target = self.target_network(dummy_input)

    def learn(self, lamb=0.0, lamb_l1=1.0, lamb_entropy=2.0, small_mag_threshold=1e-16, small_reg_factor=1.0, lamb_coef=0.0, lamb_coefdiff=0.0) -> None:
        """
        Update Q-values using gradient descent.
        """
        if self.mem_counter < self.batch_size:
            return

        if self.mem_counter % self.target_network_update_frequency == 0:
            self.target_network.load_state_dict(self.Q_eval.state_dict())

  
        def reg(acts_scale):
            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.0
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(
                    -1,
                )

                p = vec / T.sum(vec)
                l1 = T.sum(nonlinear(vec))
                entropy = -T.sum(p * T.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.Q_eval.act_fun)):
                coeff_l1 = T.sum(T.mean(T.abs(self.Q_eval.act_fun[i].coef), dim=1))
                coeff_diff_l1 = T.sum(
                    T.mean(T.abs(T.diff(self.Q_eval.act_fun[i].coef)), dim=1)
                )
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_

        

        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch], requires_grad=True).to(self.device)
        new_state_batch = T.tensor(self.new_state_memory[batch], requires_grad=True).to(self.device)
        reward_batch = T.tensor(self.reward_memory[batch], requires_grad=True).to(self.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.device)
        action_batch = self.action_memory[batch]

        with T.no_grad():
            q_next = self.target_network(new_state_batch)
            q_next = q_next.clone()  # Make a copy of q_next
            q_next[terminal_batch] = 0.0
            q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        q_eval = self.Q_eval(state_batch)[batch_index, action_batch]



        loss = nn.functional.mse_loss(q_target, q_eval)
        reg_ = reg(self.Q_eval.acts_scale)
        loss = loss + lamb * reg_
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        


        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

