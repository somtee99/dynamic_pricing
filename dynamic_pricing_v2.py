from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

# plt.switch_backend("TkAgg")
class DynamicPricingEnv():
    def __init__(self):
        self.actions = [0, 1, 2]  # 0: decrease, 1: same, 2: increase
        self.price = 20
        self.min_price = 5
        self.max_price = 50

        #The number of weeks in a year
        self.max_steps = 52
        self.current_step = 0
        self.revenue=0
        self.previous_price = 0
        self.total_revenue = 0
        self.base_demand=250
        self.price_elasticity = -0.7
        #Sensitivity to price change
        self.price_change_sensitivity = -0.5
        self.demand = self.calculate_seasonal_demand()

    def reset(self):
        self.price = 20
        self.current_step = 0
        self.demand = self.calculate_seasonal_demand()
        self.revenue = 0
        self.previous_price = 0
        self.total_revenue = 0
        return np.array([self.price,self.demand,self.current_step], dtype=np.float32)

    def step(self, action, price_change_rate = 1):
        if action == 0:
            self.price = max(self.min_price, self.price - price_change_rate)
        elif action == 2:
            self.price = min(self.max_price, self.price + price_change_rate)

        # Simulate demand using a simple demand curve
        self.demand = self.calculate_seasonal_demand()

        previousRevenue = self.revenue
        self.revenue = self.price * self.demand
        self.total_revenue +=  self.revenue

        reward =  self.revenue-previousRevenue

        # Update step counter
        self.current_step += 1
        self.previous_price = self.price
        done = self.current_step >= self.max_steps  # End of episode (1 year)

        return np.array([self.price,self.demand,self.current_step], dtype=np.float32), reward, done


    def calculate_seasonal_demand(self):
        """Generate demand based on a seasonal curve, considering price *and* price change."""
        week = self.current_step

        # Seasonal factor
        seasonal_factor = np.sin((2 * np.pi * week) / 52)
        fluctuation = 150 * seasonal_factor
        noise = np.random.randint(-10, 10)

        # Price sensitivity
        price_effect = self.price_elasticity * (self.price - self.min_price) / (self.max_price - self.min_price)

        #Sensitivity to price change
        price_change = (self.price - self.previous_price) / (self.max_price - self.min_price)
        price_change_effect = self.price_change_sensitivity * price_change

        # Total demand
        demand = self.base_demand + fluctuation + noise + (self.base_demand * (price_effect + price_change_effect))

        return max(0, demand)


def train_dynamic_pricing_q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = defaultdict(lambda: np.zeros(len(env.actions)))
    # step_rewards = np.zeros(env.max_steps)  # Store cumulative rewards per step
    total_reward_per_episode = []  # To track total reward per episode
    min_epsilon = 0.01
    epsilon_decay = 0.995
    for episode in range(episodes):
        state = env.reset()
        state = tuple(state)  # Convert state to tuple for Q-table indexing
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = epsilon_greedy(q_table, state, epsilon)
            next_state, reward, done = env.step(action)
            next_state = tuple(next_state)  # Convert to tuple for indexing
            total_reward += reward

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
        total_reward_per_episode.append(total_reward)  # Store total reward for this episode

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward (Profit): {total_reward}, Total Revenue: {env.total_revenue}")


    # Plot performance (Total Reward over Time)
    plt.figure(figsize=(10, 5))
    plt.plot(range(episodes), total_reward_per_episode, label="Total reward per Episode", color='g')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode Over Training (Q-learning)")
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
    total_reward_per_episode = []
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

        while not done:
            action = epsilon_greedy(q_table, state, epsilon)
            next_state, reward, done = env.step(action)
            next_state = tuple(next_state)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_reward_per_episode.append(total_reward)

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
            print(f"Episode {i}, Total Reward (Profit): {total_reward}, Total Revenue: {env.total_revenue}")

    # Plot performance (Total Revenue over Time)
    plt.figure(figsize=(10, 5))
    plt.plot(range(episodes), total_reward_per_episode, label="Total Reward per Episode", color='g')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode Over Training (Monte carlo)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return q_table


import matplotlib.pyplot as plt

def evaluate_agent(env, q_table):
    state = env.reset()
    state = tuple(state)
    total_reward = 0
    done = False

    prices = []
    demands = []
    timesteps = []

    week = 0
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        state = tuple(state)
        total_reward += reward

        prices.append(env.price)
        demands.append(env.demand)
        timesteps.append(week)
        week += 1

    # Plotting both Price and Demand
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Week")
    ax1.set_ylabel("Price", color='tab:blue')
    ax1.plot(timesteps, prices, marker='o', linestyle='-', color='tab:blue', label='Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for demand
    ax2 = ax1.twinx()
    ax2.set_ylabel("Demand", color='tab:red')
    ax2.plot(timesteps, demands, marker='x', linestyle='--', color='tab:red', label='Demand')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Price and Demand per Week for Evaluation')
    fig.tight_layout()
    plt.grid(True)
    plt.show()

    return total_reward, env.total_revenue


if __name__ == "__main__":
    env = DynamicPricingEnv()
    q_table_monteCarlo = train_dynamic_pricing_monte_carlo(env,episodes=10000)
    env = DynamicPricingEnv()
    q_table_Q_learning = train_dynamic_pricing_q_learning(env,episodes=10000)

    monte_carlo_Evaluation,monte_total_revenue =evaluate_agent(DynamicPricingEnv(),q_table_monteCarlo)
    Q_learning_Evaluation,q_total_revenue = evaluate_agent(DynamicPricingEnv(),q_table_Q_learning)

    print(f"Monte Carlo Evaluation {monte_carlo_Evaluation}, Total Revenue {monte_total_revenue}")
    print(f"Q-Learning Evaluation {Q_learning_Evaluation}, Total Revenue {q_total_revenue}")