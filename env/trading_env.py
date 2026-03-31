import random

class TradingEnv:
    def __init__(self, data, episode_length=8):
        self.data = data
        self.episode_length = episode_length

    def reset(self):
        # self.start_index = random.randint(0, len(self.data) - 10)
        self.start_index = random.randint(0, len(self.data) - self.episode_length - 1)
        self.current_index = self.start_index

        self.position = "none"
        self.entry_price = None
        self.trade_count = 0

        return self._get_observation()

    def _get_observation(self):
        row = self.data[self.current_index]

        return {
            "price": row["price"],
            "rsi": row["rsi"],
            "trend": row["trend"],
            # "time_to_expiry": 5 - (self.current_index - self.start_index),
            "time_to_expiry": self.episode_length - (self.current_index - self.start_index),
            "position": self.position,
            "entry_price": self.entry_price
        }

    def step(self, action):
        reward = 0
        done = False

        row = self.data[self.current_index]
        price = row["price"]
        rsi = row["rsi"]
        trend = row["trend"]

        # --- ENTRY ---
        if action == "BUY_CALL" and self.position == "none":
            self.position = "call"
            self.entry_price = price
            self.trade_count += 1

            if rsi < 30 and trend == "up":
                reward += 0.4   # good entry
            else:
                reward -= 0.4   # bad entry

        elif action == "BUY_PUT" and self.position == "none":
            self.position = "put"
            self.entry_price = price
            self.trade_count += 1

            if rsi > 70 and trend == "down":
                reward += 0.4
            else:
                reward -= 0.4

        # --- EXIT ---
        elif action == "EXIT" and self.position != "none":
            profit = 0
            # --- SMART HOLD (NO-TRADE ZONE) ---
            if action == "HOLD" and self.position == "none":
                if 40 <= rsi <= 60:
                    reward += 0.1  # good patience

            if self.position == "call":
                profit = price - self.entry_price
            elif self.position == "put":
                profit = self.entry_price - price

            reward += profit * 0.01

            if profit > 0:
                reward += 0.2  # good exit
            else:
                reward -= 0.2  # bad exit

            self.position = "none"
            self.entry_price = None

        # --- HOLD PENALTY ---
        if self.position != "none" and self.entry_price is not None:
            if self.position == "call":
                if price < self.entry_price:
                    reward -= 0.2
            elif self.position == "put":
                if price > self.entry_price:
                    reward -= 0.2

        # --- OVERTRADING PENALTY ---
        if self.trade_count > 3:
            reward -= 0.2

        # --- TIME PRESSURE ---
        time_left = self.episode_length - (self.current_index - self.start_index)
        if time_left <= 1 and self.position != "none":
            reward -= 0.3

        # Move forward
        self.current_index += 1

        if (self.current_index - self.start_index) >= self.episode_length:
            done = True

        if not done:
            obs = self._get_observation()
        else:
            obs = None

        return obs, reward, done, {}