from core.data_processing import load_data
from env.trading_env import TradingEnv

data = load_data("data/NIFTY 50_minute.csv")

env = TradingEnv(data, episode_length=8)

obs = env.reset()
done = False

total_reward = 0

while not done:
    rsi = obs["rsi"]

    if rsi < 30:
        action = "BUY_CALL"
    elif rsi > 70:
        action = "BUY_PUT"
    else:
        action = "HOLD"

    obs, reward, done, _ = env.step(action)
    total_reward += reward

print("Total Reward:", total_reward)