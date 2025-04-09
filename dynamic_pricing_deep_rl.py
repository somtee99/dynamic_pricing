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
        self.price = 5
        self.min_price = 5
        self.max_price = 50
        self.action_space = spaces.Box(low=self.min_price , high=self.max_price, shape=(1,), dtype=np.float32)
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)  # Scale actions to [-1, 1]

        self.observation_space = spaces.Box(low=np.array([self.min_price, 0,0]),
                                           high=np.array([self.max_price, 500,52]),)

        #The number of weeks in a year
        self.max_steps = 52
        self.current_step = 0
        self.revenue=0
        self.demand = self.calculate_seasonal_demand()


    def reset(self):
        self.price = 5
        self.current_step = 0
        self.demand = self.calculate_seasonal_demand()
        self.revenue = 0
        return np.array([self.price,self.demand,self.current_step])

    def step(self, action):
        self.price = np.clip(action, self.action_space.low, self.action_space.high)

        # Simulate demand using a simple demand curve
        self.demand = self.calculate_seasonal_demand()

        previousRevenue = self.revenue
        self.revenue = self.price * self.demand

        reward =  self.revenue-previousRevenue

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
    lr=0.1
    ts=5000
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

    td3.learn(total_timesteps=ts*env.max_steps)
    total_revenue_per_episode, episode_length = evaluate_policy(td3, env, n_eval_episodes=ts,return_episode_rewards=True)
    plt.figure(figsize=(10, 5))
    plt.plot(total_revenue_per_episode, color='green', label='Total Revenue per Episode')

    plt.xlabel("Episode")
    plt.ylabel("Total Revenue")
    plt.title(f"td3 config lr {lr} timestep {ts}")
    plt.legend()
    plt.grid(True)
    plt.show()


def epsilon_greedy(q_table, state, epsilon):
    if np.random.rand() < epsilon:
        action = random.randint(0, 2)  # Explore
    else:
        action = np.argmax(q_table[state])
    return action


def train_DQN(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    pass

if __name__ == "__main__":
    env = DynamicPricingEnv()
    q_table_monteCarlo = train_dynamic_pricing_q_learning(env,episodes=10000)
