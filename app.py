from core.data_processing import load_data
from env.trading_env import TradingEnv

data = load_data("data/nifty.csv")
env = TradingEnv(data)

obs = env.reset()

print("Initial Observation:", obs)