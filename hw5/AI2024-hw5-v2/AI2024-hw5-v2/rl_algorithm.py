import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import YOUR_CODE_HERE
import utils

class PacmanActionCNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PacmanActionCNN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Dueling DQN
        self.fc1_value = nn.Linear(64 * 7 * 7, 512)
        self.fc1_advantage = nn.Linear(64 * 7 * 7, 512)
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten

        value = F.relu(self.fc1_value(x))
        value = self.value(value)

        advantage = F.relu(self.fc1_advantage(x))
        advantage = self.advantage(advantage)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, state, action, reward, next_state, terminated):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.terminated[self.ptr] = terminated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.LongTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.terminated[ind]), 
        )

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-5,
        epsilon=0.9,
        epsilon_min=0.1,
        gamma=0.99,
        batch_size=64,
        warmup_steps=10000,
        buffer_size=int(1e6),
        target_update_interval=1000,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1, ), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6
    
    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            # Random action
            action = np.random.randint(0, self.action_dim)
        else:
            # output actions by following epsilon-greedy policy
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q_values = self.network(x)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def learn(self):
        if self.buffer.size < self.batch_size:
            return {}

        state, action, reward, next_state, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        
        # Get current Q values
        q_values = self.network(state)
        q_value = q_values.gather(1, action)

        # Get target Q values
        next_q_values = self.target_network(next_state)
        max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
        td_target = reward + self.gamma * max_next_q_values * (1 - terminated)
        
        # Compute loss
        loss = F.mse_loss(q_value, td_target)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return {"loss": loss.item()}
    
    def process(self, transition):
        state, action, reward, next_state, terminated = transition
        self.buffer.update(state, action, reward, next_state, terminated)
        result = {}
        self.total_steps += 1

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            
        self.epsilon -= self.epsilon_decay
        return result
