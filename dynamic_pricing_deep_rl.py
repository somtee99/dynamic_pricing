from typing import Optional, Union, List

import gym
from gym import spaces
import random
import numpy as np
from gym.core import RenderFrame
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import matplotlib.pyplot as plt

import os
log_dir ="logs"
model_dir ="models"

os.makedirs(log_dir,exist_ok=True)
os.makedirs(model_dir,exist_ok=True)

class DynamicPricingEnv(gym.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        self.min_price = 5
        self.max_price = 50
        self.price = random.uniform(self.min_price,self.max_price)
        self.action_space = spaces.Box(low=-1.0 , high=1.0, shape=(1,), dtype=np.float32)
        #observation spaces are current price,current demand, and current week
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.max_steps = 52
        self.current_step = 0
        self.revenue=0
        self.previous_price = 0
        self.total_revenue = 0
        self.base_demand=250
        # The price elasticity of demand, which indicates how sensitive demand is to price changes.
        #The impact that an increase in price has on the demand
        self.price_elasticity = -0.7
        # The sensitivity of demand to recent price changes.
        #The impact the difference between the previous price and current price will have on the demand
        self.price_change_sensitivity = -0.5
        self.demand = self.calculate_seasonal_demand()


    def normalize_obs(self, price, demand, step):
        norm_price = (price - self.min_price) / (self.max_price - self.min_price)
        norm_demand = min(demand, (self.base_demand*2)) / (self.base_demand*2)
        norm_step = step / self.max_steps
        return np.array([norm_price, norm_demand, norm_step], dtype=np.float32)
    
    def convert_action_to_price(self, action):
        return self.min_price + (action + 1) * (self.max_price - self.min_price) / 2

    def reset(self):
        self.price = random.uniform(self.min_price,self.max_price)
        # self.price=20
        self.current_step = 0
        self.avg_previous_revenue = 0
        self.previous_price=0
        self.demand = self.calculate_seasonal_demand()
        self.revenue = 0
        self.total_revenue = 0
        return self.normalize_obs(self.price,self.demand,self.current_step)
    
    def step(self, action):
        previousRevenue = self.revenue
        self.price = self.convert_action_to_price(action[0])

        # Simulate demand using a simple demand curve
        self.demand = self.calculate_seasonal_demand()
        self.revenue = self.price * self.demand
        self.total_revenue +=self.revenue

        reward = self.revenue-previousRevenue

        #scale reward
        reward= reward / 100

        self.current_step += 1
        done = self.current_step >= self.max_steps

        self.price = float(self.price)  # Convert to float
        self.demand = float(self.demand)  # Convert to float
        self.current_step = float(self.current_step)  # Convert to float
        self.previous_price =  self.price
        obs=  self.normalize_obs(self.price,self.demand,self.current_step)
        return obs,reward, done,{}


    def calculate_seasonal_demand(self):
        """Generate demand based on a seasonal curve, considering price *and* price change."""
        week = self.current_step
        # Seasonal factor
        seasonal_factor = np.sin((2 * np.pi * week) / self.max_steps)
        fluctuation = 150 * seasonal_factor
        noise = np.random.randint(-10, 10)

        # Price sensitivity
        price_effect = self.price_elasticity * (self.price - self.min_price) / (self.max_price - self.min_price)

        #Sensitivity to price change
        price_change = (self.price - self.previous_price) / (self.max_price - self.min_price)
        price_change_effect = self.price_change_sensitivity * price_change

        # Total demand
        demand =  self.base_demand + fluctuation + noise + (self.base_demand * (price_effect + price_change_effect))

        return max(0, demand)
    
    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

def train_TD3(env):
    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    td3 = TD3(
        "MlpPolicy", env,
        action_noise=action_noise,
        verbose=1,
        device='cpu',
        tensorboard_log=log_dir,
    )

    timesteps=100000
    iters=0
    while True:
        iters +=1
        td3.learn(total_timesteps=timesteps,reset_num_timesteps=False)
        td3.save(f"{model_dir}/td3 _ {timesteps*iters}")

def rescale_total_reward(total_reward):
    return total_reward*100

def evaluate_agent(env, model_path="models/td3 _ 300000.zip",show_plot=True):
    model = TD3.load(model_path, env=env)
    state = env.reset()

    prices = []
    demands = []
    weeks = []
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(state)
        state, reward, done, _ = env.step(action)
        
        prices.append(env.price)
        demands.append(env.demand)
        weeks.append(env.current_step)
        total_reward += reward

    if show_plot:
        # Plotting prices and demand per week using twin axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(weeks, prices, marker='o', linestyle='-', color=color, label='Price')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Demand', color=color)
        ax2.plot(weeks, demands, marker='s', linestyle='--', color=color, label='Demand')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title("Optimal Prices and Demand per Week (TD3 Agent)")
        fig.tight_layout()
        plt.grid(True)
        plt.show()

    print(f"Final Total Revenue: {env.total_revenue:.2f}")
    rescaled_reward = rescale_total_reward(total_reward)
    print(f"Rescaled Total Reward: {rescaled_reward}")
    
    return rescaled_reward, env.total_revenue


if __name__ == "__main__":
    env = DynamicPricingEnv()

    #For training model
    # train_TD3(env)

    #for Evaluation
    td3_evaluation_result,total_revenue=evaluate_agent(env)
    print(f"TD3 Evaluation {td3_evaluation_result}")
    print(f"Total Revenue {total_revenue}")

