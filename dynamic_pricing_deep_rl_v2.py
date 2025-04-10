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
        self.min_price = 5
        self.max_price = 10
        self.price = 5
        self.previous_price=0
        self.avg_previous_revenue = 0
        self.action_space = spaces.Box(low=-1.0 , high=1.0, shape=(1,), dtype=np.float32)

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
        self.avg_previous_revenue = 0
        self.demand = self.calculate_seasonal_demand()
        self.revenue = 0
        self.previous_price=0
        return np.array([self.price,self.demand,self.current_step])

    def step(self, action):
        previousRevenue = self.revenue
        previous_demand=self.demand
        self.price = self.min_price + (action[0] + 1) * (self.max_price - self.min_price) / 2

        # Simulate demand using a simple demand curve
        self.demand = self.calculate_seasonal_demand()
        self.revenue = self.price * self.demand
        reward = self.revenue-previousRevenue
        # Smooth previous revenue a bit (like over a moving average)
        # self.avg_previous_revenue = 0.9 * self.avg_previous_revenue + 0.1 * self.revenue

        # Reward = (Current Revenue - Average Previous Revenue)
        # reward = self.revenue - self.avg_previous_revenue

        # reward = self.revenue - self.avg_previous_revenue
        # if self.price != self.previous_price:
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
        self.previous_price = self.price

        obs= np.array([self.price,self.demand,self.current_step])
        return obs,reward, done,{}

    # def calculate_seasonal_demand(self):
    #     """Generate demand based on a seasonal curve, considering the price."""
    #     week = self.current_step
    #     # Seasonal factor based on a sine wave (seasonality effect)
    #     seasonal_factor = np.sin((2 * np.pi * week) / 52)  # Sine wave for seasonality (annually)
    #     # Base demand (average demand)
    #     base_demand = 250
    #     # Seasonal fluctuation (up or down based on the time of year)
    #     fluctuation = 150 * seasonal_factor
    #     # Random noise to simulate unpredictable demand changes
    #     noise = np.random.randint(-10, 10)
    #     # Price sensitivity: Demand decreases as price increases (price elasticity)
    #     # We can adjust this elasticity factor based on how price-sensitive the product is.
    #     price_elasticity = -0.5  # Example elasticity: price increase reduces demand
    #     # Adjust demand based on price (the higher the price, the lower the demand)
    #     price_effect = price_elasticity * (self.price - self.min_price) / (self.max_price - self.min_price)
    #     # Total demand considering price, seasonality, and noise
    #     demand = base_demand + fluctuation + noise + (base_demand * price_effect)
    #
    #     # Ensure demand is non-negative (no negative demand)
    #     return max(0, round(demand))

    # def calculate_seasonal_demand(self):
    #   """Generate demand based on a seasonal curve."""
    #   week = self.current_step
    #   seasonal_factor = np.sin((2 * np.pi * week) / 52)  # Sine wave for seasonality
    #   base_demand = 250  # Average demand
    #   fluctuation = 150 * seasonal_factor  # Seasonal variation
    #   noise = np.random.randint(-10, 10)  # Random variation
    #   return max(0, round(base_demand + fluctuation + noise))

    def calculate_seasonal_demand(self):
        """Generate demand based on a seasonal curve, considering price *and* price change."""
        week = self.current_step

        # Seasonal factor
        seasonal_factor = np.sin((2 * np.pi * week) / 52)
        base_demand = 250
        fluctuation = 150 * seasonal_factor
        noise = np.random.randint(-10, 10)

        # Price sensitivity
        price_elasticity = -0.5
        price_effect = price_elasticity * (self.price - self.min_price) / (self.max_price - self.min_price)

        # New: Sensitivity to price change
        price_change_sensitivity = -0.5  # How sensitive is demand to recent price changes?
        price_change = (self.price - self.previous_price) / (self.max_price - self.min_price)
        price_change_effect = price_change_sensitivity * price_change

        # Total demand
        demand = base_demand + fluctuation + noise + (base_demand * (price_effect + price_change_effect))

        # Update previous price
        self.previous_price = self.price

        return max(0, demand)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass


def train_dynamic_pricing_TD3_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=1):
    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    lr=0.01
    td3 = TD3(
        "MlpPolicy", env,
        action_noise=action_noise,
        learning_rate=lr,
        verbose=0,
    )
    # Track rewards during training
    episode_rewards = []  # List to store rewards of each episode
    # Training loop with reward tracking
    n_episodes = 100 # Total number of episodes to train
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0  # Reward accumulated in this episode

        # Run one episode
        while not done:
            action= epsilon_greedy(td3,obs,epsilon,env)  # Stochastic actions for exploration
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward  # Accumulate reward
            td3.replay_buffer.add(obs, next_obs, action, reward, done)
        # Log total reward for this episode


        episode_rewards.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")


    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Progress: Total Reward per Episode")
    plt.show()


def epsilon_greedy(model, obs, epsilon,env):
    action,_ = model.predict(obs, deterministic=False)
    return action


# if __name__ == "__main__":
#     env = DynamicPricingEnv()
#     q_table_monteCarlo = train_dynamic_pricing_TD3_learning(env,episodes=10000)
