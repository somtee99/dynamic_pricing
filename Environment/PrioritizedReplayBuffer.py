import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class PrioritizedReplayBuffer:
    def __init__(self, max_size=int(1e6), alpha=0.6):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha  # how much prioritization to use (0 - no PER, 1 - full PER)

    def add(self, state, next_state, action, reward, done):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, next_state, action, reward, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("The buffer is empty!")

        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state, next_state, action, reward, done = map(np.stack, zip(*samples))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device),
            torch.FloatTensor(weights).unsqueeze(1).to(device),
            indices
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def size(self):
        return len(self.buffer)
