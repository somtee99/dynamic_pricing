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

        previousRevenue = self.revenue
        previous_demand=self.demand
        self.price = self.min_price + (action[0] + 1) * (self.max_price - self.min_price) / 2

        # Simulate demand using a simple demand curve
        self.demand = self.calculate_seasonal_demand()
        self.revenue = self.price * self.demand
        reward = self.revenue-previousRevenue

        # if self.price != previous_price:
        #     reward += 1
        # else:
        #     reward -= 1

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
        """Generate demand based on a seasonal curve, considering the price."""
        week = self.current_step

        # Seasonal factor based on a sine wave (seasonality effect)
        seasonal_factor = np.sin((2 * np.pi * week) / 52)  # Sine wave for seasonality (annually)

        # Base demand (average demand)
        base_demand = 250

        # Seasonal fluctuation (up or down based on the time of year)
        fluctuation = 150 * seasonal_factor

        # Random noise to simulate unpredictable demand changes
        noise = np.random.randint(-10, 10)

        # Price sensitivity: Demand decreases as price increases (price elasticity)
        # We can adjust this elasticity factor based on how price-sensitive the product is.
        price_elasticity = -0.5  # Example elasticity: price increase reduces demand

        # Adjust demand based on price (the higher the price, the lower the demand)
        price_effect = price_elasticity * (self.price - self.min_price) / (self.max_price - self.min_price)

        # Total demand considering price, seasonality, and noise
        demand = base_demand + fluctuation + noise + (base_demand * price_effect)

        # Ensure demand is non-negative (no negative demand)
        return max(0, round(demand))

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass


def train_dynamic_pricing_q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.6):
    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    lr=0.01
    min_epsilon = 0.1
    epsilon_decay = 0.995
    td3 = TD3(
        "MlpPolicy", env,
        action_noise=action_noise,
        learning_rate=lr,
        verbose=0,
        # gamma=0.9
    )
    td3.learn(total_timesteps=50000)

    result=evaluate_agent(td3,env,100)

    # Plot them together
    plt.plot(result, label="TD3")
    plt.xlabel("Episode")
    plt.ylabel("Total Revenue")
    plt.legend()
    plt.title("TD3 vs Q-Learning on Dynamic Pricing")
    plt.show()
    # # Track rewards during training
    # episode_rewards = []  # List to store rewards of each episode
    # # Training loop with reward tracking
    # n_episodes = 5000 # Total number of episodes to train
    # for episode in range(n_episodes):
    #     obs = env.reset()
    #     done = False
    #     total_reward = 0  # Reward accumulated in this episode
    #
    #     # Run one episode
    #     while not done:
    #         action= epsilon_greedy(td3,obs,epsilon,env)  # Stochastic actions for exploration
    #         obs, reward, done, _ = env.step(action)
    #         total_reward += reward  # Accumulate reward
    #
    #     # Log total reward for this episode
    #     episode_rewards.append(total_reward)
    #     td3.learn(total_timesteps=1)  # Learning after a batch of experiences
    #     # Perform learning step after each episode
    #
    #     epsilon = max(min_epsilon, epsilon * epsilon_decay)
    #     # Optionally print progress
    #     if episode % 10 == 0:
    #         print(f"Episode {episode}, Total Reward: {total_reward}")
    #
    # # Plot the agent's learning progress
    # plt.plot(episode_rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("Learning Progress: Total Reward per Episode")
    # plt.show()


def epsilon_greedy(model, obs, epsilon,env):
    if np.random.rand() < epsilon:
        action = np.random.uniform(env.action_space.low, env.action_space.high)
    else:
        action,_ = model.predict(obs, deterministic=True)
    return action

def evaluate_agent(model, env, n_eval_episodes=100):
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
    return episode_rewards



if __name__ == "__main__":
    env = DynamicPricingEnv()
    q_table_monteCarlo = train_dynamic_pricing_q_learning(env,episodes=10000)
