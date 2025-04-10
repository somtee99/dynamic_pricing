from collections import defaultdict
from typing import Optional, Union, List

import matplotlib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random
import numpy as np
from gym.core import RenderFrame
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

# plt.switch_backend("TkAgg")
class DynamicPricingEnv(gym.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        self.price =  1.0
        self.min_price = 5
        self.max_price = 50
        self.action_space = spaces.Box(low=-1.0 , high=1.0, shape=(1,), dtype=np.float32)
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)  # Scale actions to [-1, 1]

        self.observation_space = spaces.Box(low=np.array([self.min_price, 0,0]),
                                           high=np.array([self.max_price, 500,52]),)

        #The number of weeks in a year
        self.max_steps = 52
        self.current_step = 0
        self.revenue=0
        self.demand = self.calculate_seasonal_demand()


    def reset(self):
        self.price = 1.0
        self.current_step = 0
        self.demand = self.calculate_seasonal_demand()
        self.revenue = 0
        return np.array([self.price,self.demand,self.current_step])

    def step(self, action):
        # self.price = np.clip(action, self.action_space.low, self.action_space.high)
        previous_price = self.price

        previousRevenue = self.revenue
        previous_demand=self.demand
        self.price = self.min_price + (action[0] + 1) * (self.max_price - self.min_price) / 2

        # Simulate demand using a simple demand curve
        self.demand = self.calculate_seasonal_demand()
        self.revenue = self.price * self.demand
        reward = self.revenue-previousRevenue

        #
        # price_sensitivity = abs(self.price - previous_price) / previous_price if previous_price != 0 else 0
        # # reward= self.price - previous_price
        # # Demand change factor: how much demand changed relative to previous demand
        # demand_change = abs(self.demand - previous_demand)
        #
        # # Year-on-year growth: Reward if revenue increases compared to the previous step (week)
        # revenue_growth = self.revenue - previousRevenue
        #
        # # Reward based on revenue
        # reward += revenue_growth  # Encourages maximizing revenue growth
        #
        # # Penalty for large price fluctuations to encourage stability
        # reward -= price_sensitivity * 0.5
        #
        # # Reward for maintaining demand stability with smaller changes in demand and price
        # if demand_change < 20 and price_sensitivity < 0.1:
        #     reward += 0.5
        # # self.revenue = self.price * self.demand
        # #
        # # reward =  self.revenue-previousRevenue
        # #
        if self.price != previous_price:
            reward += 1
        else:
            reward -= 1

            # Update step counter
        self.current_step += 1
        done = self.current_step >= self.max_steps  # End of episode (1 year)
        # We are returning the state which consist of the price and demand, then we return the reward value and a boolean
        # indicating if we have completed a 1 year cycle

        self.price = float(self.price)  # Convert to float
        self.demand = float(self.demand)  # Convert to float
        self.current_step = float(self.current_step)  # Convert to float

        obs= np.array([self.price,self.demand,self.current_step])
        return obs,reward, done,{}

    def calculate_seasonal_demand(self):
      """Generate demand based on a seasonal curve."""
      week = self.current_step
      seasonal_factor = np.sin((2 * np.pi * week) / 52)  # Sine wave for seasonality
      base_demand = 250  # Average demand
      fluctuation = 150 * seasonal_factor  # Seasonal variation
      noise = np.random.randint(-10, 10)  # Random variation
      return max(0, round(base_demand + fluctuation + noise))

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass


def train_dynamic_pricing_q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    lr=0.01
    ts=5000
    min_epsilon = 0.1
    epsilon_decay = 0.995
    # learning_rates = [0.001]
    # timesteps =
    # for lr in learning_rates:
    #     for ts in timesteps:
    td3 = TD3(
        "MlpPolicy", env,
        action_noise=action_noise,
        learning_rate=lr,
        verbose=0,
        # gamma=0.9
    )
    # Track rewards during training
    episode_rewards = []  # List to store rewards of each episode
    td3.learn(total_timesteps=1000)
    # # Training loop with reward tracking
    # n_episodes = 2 # Total number of episodes to train
    # for episode in range(n_episodes):
    #     obs = env.reset()
    #     done = False
    #     total_reward = 0  # Reward accumulated in this episode
    #
    #     # Run one episode
    #     while not done:
    #         action= epsilon_greedy(td3,obs,epsilon)  # Stochastic actions for exploration
    #         obs, reward, done, _ = env.step(action)
    #         total_reward += reward  # Accumulate reward
    #
    #     # Log total reward for this episode
    #     episode_rewards.append(total_reward)
    #
    #     # Perform learning step after each episode
    #
    #     epsilon = max(min_epsilon, epsilon * epsilon_decay)
    #     # Optionally print progress
    #     if episode % 10 == 0:
    #         print(f"Episode {episode}, Total Reward: {total_reward}")

    # Plot the agent's learning progress
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Progress: Total Reward per Episode")
    plt.show()
    # # total_revenue_per_episode, episode_length = evaluate_policy(td3, env, n_eval_episodes=ts,return_episode_rewards=True)
    # plt.figure(figsize=(10, 5))
    # plt.plot(total_revenue_per_episode, color='green', label='Total Revenue per Episode')
    #
    # plt.xlabel("Episode")
    # plt.ylabel("Total Revenue")
    # plt.title(f"td3 config lr {lr} timestep {ts}")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


def epsilon_greedy(model, obs, epsilon):
    # if np.random.rand() < epsilon:
    #     action,_ = model.predict(obs, deterministic=False)
    # else:
    action,_ = model.predict(obs, deterministic=True)
    return action


def train_DQN(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    pass

if __name__ == "__main__":
    env = DynamicPricingEnv()
    q_table_monteCarlo = train_dynamic_pricing_q_learning(env,episodes=10000)
