from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

# plt.switch_backend("TkAgg")
class DynamicPricingEnv():
    def __init__(self):
        self.actions = [0, 1, 2]  # 0: decrease, 1: same, 2: increase
        # self.observation_space = spaces.Box(low=np.array([5, 0]),
        #                                    high=np.array([50, 500]),
        #                                    dtype=np.float32)
        # self.states = self.state_init()
        self.price = 20
        self.min_price = 15
        self.max_price = 25
        #The number of weeks in a year
        self.max_steps = 52
        self.current_step = 0
        self.revenue=0
        self.demand = self.calculate_seasonal_demand()

    def reset(self):
        self.price = 20
        self.current_step = 0
        self.demand = self.calculate_seasonal_demand()
        self.revenue = 0
        return np.array([self.price,self.demand,self.current_step], dtype=np.float32)

    def step(self, action, price_change_rate = 0.5):
        if action == 0:
            self.price = max(self.min_price, self.price - price_change_rate)
        elif action == 2:
            self.price = min(self.max_price, self.price + price_change_rate)

        # Simulate demand using a simple demand curve
        self.demand = self.calculate_seasonal_demand()
        # self.demand = max(0, 500 - 10 * self.price + np.random.randint(-10, 10))

        previousRevenue = self.revenue
        self.revenue = self.price * self.demand

        reward =  self.revenue-previousRevenue
        # reward =  self.revenue

        # Update step counter
        self.current_step += 1
        done = self.current_step >= self.max_steps  # End of episode (1 year)
        # We are returning the state which consist of the price and demand, then we return the reward value and a boolean
        # indicating if we have completed a 1 year cycle
        return np.array([self.price,self.demand,self.current_step], dtype=np.float32), reward, done

    def calculate_seasonal_demand(self):
      """Generate demand based on a seasonal curve."""
      week = self.current_step
      seasonal_factor = np.sin((2 * np.pi * week) / 52)  # Sine wave for seasonality
      base_demand = 250  # Average demand
      fluctuation = 150 * seasonal_factor  # Seasonal variation
      noise = np.random.randint(-10, 10)  # Random variation
      return max(0, round(base_demand + fluctuation + noise))


def train_dynamic_pricing_q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = defaultdict(lambda: np.zeros(len(env.actions)))
    # step_rewards = np.zeros(env.max_steps)  # Store cumulative rewards per step
    total_revenue_per_episode = []  # To track total revenue per episode
    min_epsilon = 0.01
    epsilon_decay = 0.995
    for episode in range(episodes):
        state = env.reset()
        state = tuple(state)  # Convert state to tuple for Q-table indexing
        done = False
        total_reward = 0
        total_revenue = 0  # Track total revenue for each episode
        step = 0

        while not done:
            action = epsilon_greedy(q_table, state, epsilon) 
            next_state, reward, done = env.step(action)
            next_state = tuple(next_state)  # Convert to tuple for indexing
            total_reward += reward
            total_revenue += reward  # Update total revenue for this episode

            # Update step rewards
            # step_rewards[step] += reward  # Aggregate reward for this step
            step += 1

            # Q-learning update
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] += alpha * (
                    reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
            )

            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_revenue_per_episode.append(total_revenue)  # Store total revenue for this episode

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward (Profit): {total_reward}, Total Revenue: {total_revenue}")


    # Plot performance (Total Revenue over Time)
    plt.figure(figsize=(10, 5))
    plt.plot(range(episodes), total_revenue_per_episode, label="Total Revenue per Episode", color='g')
    plt.xlabel("Episode")
    plt.ylabel("Total Revenue")
    plt.title("Total Revenue per Episode Over Training (Q-learning)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return q_table


def epsilon_greedy(q_table, state, epsilon):
    if np.random.rand() < epsilon:
        action = random.randint(0, 2)  # Explore
    else:
        action = np.argmax(q_table[state])
    return action

def train_dynamic_pricing_monte_carlo(env, episodes=1000, gamma=0.9, epsilon=0.1):
    q_table = defaultdict(lambda: np.zeros(len(env.actions)))
    returns= defaultdict(lambda: [])
    total_revenue_per_episode = []
    min_epsilon = 0.01
    epsilon_decay = 0.995

    for i in range(episodes):
        if i % 100 == 0:
            print(f"Monte Carlo Episode: {i}")

        episode = []
        state = env.reset()
        state = tuple(state)
        done = False
        total_reward = 0
        total_revenue = 0

        while not done:
            action = epsilon_greedy(q_table, state, epsilon)
            next_state, reward, done = env.step(action)
            next_state = tuple(next_state)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
            total_revenue += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_revenue_per_episode.append(total_reward)

        # Update Q-values using first-visit Monte Carlo
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                q_table[state][action] = np.mean(returns[(state, action)])

        if i % 100 == 0:
            print(f"Episode {i}, Total Reward (Profit): {total_reward}, Total Revenue: {total_revenue}")

    # Plot performance (Total Revenue over Time)
    plt.figure(figsize=(10, 5))
    plt.plot(range(episodes), total_revenue_per_episode, label="Total Revenue per Episode", color='g')
    plt.xlabel("Episode")
    plt.ylabel("Total Revenue")
    plt.title("Total Revenue per Episode Over Training (Monte carlo)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return q_table, total_revenue_per_episode



if __name__ == "__main__":
    env = DynamicPricingEnv()
    q_table_monteCarlo = train_dynamic_pricing_monte_carlo(env,episodes=10000)
    env = DynamicPricingEnv()
    q_table_Q_learning = train_dynamic_pricing_q_learning(env,episodes=10000)