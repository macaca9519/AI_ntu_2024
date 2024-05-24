import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils

class PacmanActionCNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PacmanActionCNN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
            torch.LongTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.terminated[ind]), 
        )

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-5,
        epsilon=0.95,
        epsilon_min=0.1,
        gamma=0.99,
        batch_size=64,
        warmup_steps=10000,
        buffer_size=int(1e5),
        target_update_interval=1000,
        #no use if modify hyperparameter in this section
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)
        print(f'{epsilon = }')
        
        self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            action = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q_value = self.network(x)
            action = q_value.argmax(dim=1).item()
        return action

    def learn(self):
        state, action, reward, next_state, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))

        # Compute the target Q value
        # with torch.no_grad():
        next_q_values = self.target_network(next_state).detach()
        next_q_value = next_q_values.max(dim=1, keepdim=True)[0]
        td_target = reward + self.gamma * next_q_value * (1 - terminated)
        
        # Compute the current Q value
        current_q_values = self.network(state)
        current_q_value = current_q_values.gather(1, action)

        # Compute the loss
        loss = F.smooth_l1_loss(current_q_value, td_target)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"value_loss": loss.item()}

    def process(self, transition):
        state, action, reward, next_state, terminated = transition
        self.buffer.update(state, action, reward, next_state, terminated)
        
        self.total_steps += 1
        result = {"reward": reward}

        if self.total_steps > self.warmup_steps:
            learn_result = self.learn()
            result.update(learn_result)
            
        
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        return result
