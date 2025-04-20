import dynamic_pricing_v2
import dynamic_pricing_deep_rl
import numpy as np
def evaluate(runs: int, agent: str):
    q_table=None
    env=None
    match agent:
        case "monte_carlo":
            env = dynamic_pricing_v2.DynamicPricingEnv()
            q_table = dynamic_pricing_v2.train_dynamic_pricing_monte_carlo(env, episodes=10000,show_plot=False)
        case "q_learning":
            env = dynamic_pricing_v2.DynamicPricingEnv()
            q_table = dynamic_pricing_v2.train_dynamic_pricing_q_learning(env, episodes=10000,show_plot=False)
        case "td3":
            env = dynamic_pricing_deep_rl.DynamicPricingEnv()
            dynamic_pricing_deep_rl.evaluate_agent(env,show_plot=False)
        case _:
            return None,None
    rewards=[]
    revenues=[]
    for run in range(runs):
        if (agent == "monte_carlo")or(agent =="q_learning"):
            reward,revenue=dynamic_pricing_v2.evaluate_agent(env,q_table,show_plot=False)
            rewards.append(reward)
            revenues.append(revenue)
        else:
            reward,revenue=dynamic_pricing_deep_rl.evaluate_agent(env,show_plot=False)
            rewards.append(reward)
            revenues.append(revenue)

    return np.mean(rewards), np.mean(revenues),np.std(rewards), np.std(revenues)

if __name__ == "__main__":
    runs=30
    ql_reward_mean, ql_revenue_mean,ql_reward_std, ql_revenue_std= evaluate(runs,"q_learning")

    print(f"q_learning mean reward: {ql_reward_mean:.2f}, revenue {ql_revenue_mean:.2f}. std reward {ql_reward_std:.2f}, "
          f"revenue {ql_revenue_std:.2f}, for {runs} runs")

    mc_reward_mean, mc_revenue_mean, mc_reward_std, mc_revenue_std= evaluate(runs,"monte_carlo")
    print(f"monte carlo mean reward: {mc_reward_mean:.2f}, revenue {mc_revenue_mean:.2f}. std reward {mc_reward_std:.2f},"
          f" revenue {mc_revenue_std:.2f},  for {runs} runs")

    td3_reward_mean,td3_revenue_mean,td3_reward_std,td3_revenue_std=evaluate(runs,"td3")
    print(f"td3 mean reward: {td3_reward_mean:.2f}, revenue {td3_revenue_mean:.2f}. std reward {td3_reward_std:.2f},"
          f" revenue {td3_revenue_std:.2f} for {runs} runs ")


