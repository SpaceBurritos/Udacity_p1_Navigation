import numpy as np
import random
from collections import namedtuple, deque

from model import Model

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 8
SEED = 7

device = "cuda"if torch.cuda.is_available() else "cpu" 

class Agent:
    
    def __init__(self, state_size, action_size):
        self.seed = random.seed(SEED)
        self.state_size = state_size
        self.action_size = action_size
        self.main_model = Model(state_size, action_size, SEED).to(device)
        self.target_model = Model(state_size, action_size, SEED).to(device)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        self.lr = 5e-4
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=self.lr)
        self.i = 0
        
    
    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.main_model.eval()
        with torch.no_grad():
            action_values = self.main_model(state)
        self.main_model.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        self.i = (self.i + 1) % UPDATE_EVERY
        if self.i == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                #self.lr *= LR_DECAY
                #self.optimizer = optim.Adam(self.main_model.parameters(), lr=self.lr)


    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.main_model(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.main_model, self.target_model, TAU)     
        return
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, main_param in zip(self.target_model.parameters(), self.main_model.parameters()):
            target_param.data.copy_(tau*main_param.data + (1.0-tau)*target_param.data)
    
class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size):
        self.experience = namedtuple("experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = deque(maxlen=buffer_size)
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(SEED)

    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append(self.experience(state, action, reward, next_state, done))

    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """ Return the current size of internal memory. """
        return len(self.memory)