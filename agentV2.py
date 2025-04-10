import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import matplotlib.pyplot as plt

# from Environment.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from dynamic_pricing_deep_rl_v2 import DynamicPricingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        return self.q1(xu), self.q2(xu)

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        return self.q1(xu)


class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, next_state, action, reward, done):
        self.buffer.append((state, next_state, action, reward, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state, next_state, action, reward, done = map(np.stack, zip(*samples))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )

    def size(self):
        return len(self.buffer)


class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = PrioritizedReplayBuffer()

        self.max_action = max_action
        self.policy_noise = 0.3
        self.noise_clip = 0.3
        self.policy_delay = 2
        self.gamma = 0.995
        self.tau = 0.005
        self.total_it = 0

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.uniform(-1.0, 1.0, size=(1,))
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, batch_size=64, beta=0.4):
        if self.replay_buffer.size() < batch_size:
            return

        self.total_it += 1
        state, next_state, action, reward, done, weights, indices = self.replay_buffer.sample(batch_size, beta)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)

        td_errors = (current_Q1 - target_Q).detach().cpu().numpy() + 1e-6  # small constant to avoid zero
        td_errors = np.abs(td_errors).flatten()

        critic_loss = (weights * (current_Q1 - target_Q).pow(2)).mean() + \
                      (weights * (current_Q2 - target_Q).pow(2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


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

def train_dynamic_pricing_td3(env, episodes=10000, epsilon=0.1):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.0  # because TD3 will output in [-1, 1]
    min_epsilon = 0.1
    epsilon_decay = 0.995
    agent = TD3Agent(state_dim, action_dim, max_action)
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, next_state, action, reward, done)
            state = next_state
            episode_reward += reward
            agent.train()
            # if agent.replay_buffer.size() > 10000:
            #     agent.train()

        episode_rewards.append(episode_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}")

    # Plot results
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("TD3 with PyTorch - Reward per Episode")
    plt.show()


if __name__== "__main__":
    env=DynamicPricingEnv()
    train_dynamic_pricing_td3(env)
