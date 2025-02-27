import gym
import numpy as np
from gym import spaces
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DynamicPricingEnv(gym.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        # self.actions = [0, 1, 2]  # 0: decrease, 1: same, 2: increase
        # self.observation_space = spaces.Box(low=np.array([5, 0]),
        #                                    high=np.array([50, 500]),
        #                                    dtype=np.float32)
        # self.states = self.state_init()
        self.price = 20
        self.demand = 0
        #The number of weeks in a year
        self.max_steps = 52
        self.current_step = 0

    def reset(self):
        self.price = 20
        self.demand = self.calculate_seasonal_demand()
        self.current_step = 0
        return np.array([self.price, self.demand], dtype=np.float32)

    def step(self, action, price_change_rate = 0.1):
        if action == 0:
            self.price = max(5, self.price - price_change_rate)
        elif action == 2:
            self.price = min(50, self.price + price_change_rate)

        # Simulate demand using a simple demand curve
        self.demand = self.calculate_seasonal_demand()

        revenue = self.price * self.demand

        reward = revenue

        # Update step counter
        self.current_step += 1
        done = self.current_step >= self.max_steps  # End of episode (1 year)
        # We are returning the state which consist of the price and demand, then we return the reward value and a boolean
        # indicating if we have completed a 1 year cycle
        return np.array([self.price, self.demand], dtype=np.float32), reward, done, {}
    
    def calculate_seasonal_demand(self):
      """Generate demand based on a seasonal curve."""
      week = self.current_step
      seasonal_factor = np.sin((2 * np.pi * week) / 52)  # Sine wave for seasonality
      base_demand = 250  # Average demand
      fluctuation = 150 * seasonal_factor  # Seasonal variation
      noise = np.random.randint(-10, 10)  # Random variation
      return max(0, round(base_demand + fluctuation + noise))


def train_dynamic_pricing(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    step_rewards = np.zeros(env.max_steps)  # Store cumulative rewards per step

    for episode in range(episodes):
        state = env.reset()
        state = tuple(state)  # Convert state to tuple for Q-table indexing
        done = False
        total_reward = 0
        step = 0

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit best action
            
            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)  # Convert to tuple for indexing
            total_reward += reward
            
            # Update step rewards
            step_rewards[step] += reward  # Aggregate reward for this step
            step += 1

            # Q-learning update
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] += alpha * (
                reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
            )

            state = next_state
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward (Profit): {total_reward}")

    # Average rewards per step
    avg_rewards_per_step = step_rewards / episodes

    # Plot performance
    plt.figure(figsize=(10, 5))
    plt.plot(range(env.max_steps), avg_rewards_per_step, label="Average Reward per Step", color='b')
    plt.xlabel("Step (Week)")
    plt.ylabel("Average Reward")
    plt.title("Dynamic Pricing Performance Over Time")
    plt.legend()
    plt.grid()
    plt.show()

    return q_table

def plot_q_table(q_table):
    # Convert dictionary to a structured array for plotting
    prices = []
    demands = []
    best_actions = []
    
    for (price, demand), actions in q_table.items():
        prices.append(price)
        demands.append(demand)
        best_actions.append(np.argmax(actions))  # Get best action

    # Create a scatter plot for best actions at each (price, demand) state
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(prices, demands, c=best_actions, cmap='viridis', edgecolors='k')
    plt.colorbar(scatter, label="Best Action (0: Decrease, 1: Same, 2: Increase)")
    
    plt.xlabel("Price")
    plt.ylabel("Demand")
    plt.title("Q-Table Visualization: Best Action for Each (Price, Demand) State")
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    env = DynamicPricingEnv()
    q_table = train_dynamic_pricing(env)
    plot_q_table(q_table)

