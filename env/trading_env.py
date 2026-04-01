import random

class TradingEnv:
    def __init__(self, data, episode_length=8):
        self.data = data
        self.episode_length = episode_length

        # Capital system
        self.initial_balance = 100000
        self.brokerage = 20
        self.slippage_percent = 0.001

    def reset(self):
        self.equity_curve = []

        self.start_index = random.randint(0, len(self.data) - self.episode_length - 1)
        self.current_index = self.start_index

        self.position = "none"
        self.entry_price = None
        self.trade_count = 0

        self.balance = self.initial_balance
        self.position_size = 0
        self.equity = self.balance

        # Drawdown tracking
        self.peak_equity = self.equity
        self.max_drawdown = 0

        return self._get_observation()

    def _get_observation(self):
        row = self.data[self.current_index]

        return {
            "price": row["price"],
            "rsi": row["rsi"],
            "trend": row["trend"],
            "time_to_expiry": self.episode_length - (self.current_index - self.start_index),
            "position": self.position,
            "entry_price": self.entry_price,
            "balance": self.balance,
            "equity": self.equity
        }

    def step(self, action):
        reward = 0
        done = False

        row = self.data[self.current_index]
        price = row["price"]
        rsi = row["rsi"]
        trend = row["trend"]

        # ❌ Prevent trading if balance too low
        if self.balance < 1000:
            return self._get_observation(), -1, True, {"reason": "insufficient_balance"}

        # --- ENTRY ---
        if action == "BUY_CALL" and self.position == "none":
            self.position = "call"

            self.position_size = 0.1 * self.balance
            entry_price = price * (1 + self.slippage_percent)

            self.entry_price = entry_price
            self.trade_count += 1

            self.balance -= self.position_size
            self.balance -= self.brokerage

            if rsi < 30 and trend == "up":
                reward += 0.4
            else:
                reward -= 0.4

        elif action == "BUY_PUT" and self.position == "none":
            self.position = "put"

            self.position_size = 0.1 * self.balance
            entry_price = price * (1 - self.slippage_percent)

            self.entry_price = entry_price
            self.trade_count += 1

            self.balance -= self.position_size
            self.balance -= self.brokerage

            if rsi > 70 and trend == "down":
                reward += 0.4
            else:
                reward -= 0.4

        # --- EXIT ---
        elif action == "EXIT" and self.position != "none":

            if self.position == "call":
                exit_price = price * (1 - self.slippage_percent)
                profit = (exit_price - self.entry_price) / self.entry_price
            else:
                exit_price = price * (1 + self.slippage_percent)
                profit = (self.entry_price - exit_price) / self.entry_price

            pnl = self.position_size * profit

            self.balance += self.position_size + pnl
            self.balance -= self.brokerage

            reward += pnl * 0.001

            if pnl > 0:
                reward += 0.2
            else:
                reward -= 0.2

            self.position = "none"
            self.entry_price = None
            self.position_size = 0

        # --- HOLD ---
        elif action == "HOLD":
            if self.position == "none" and 40 <= rsi <= 60:
                reward += 0.1

        # --- HOLD PENALTY ---
        if self.position != "none" and self.entry_price is not None:
            if self.position == "call" and price < self.entry_price:
                reward -= 0.2
            elif self.position == "put" and price > self.entry_price:
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
            new_price = obs["price"]  # ✅ FIXED
        else:
            obs = None
            new_price = price

        # --- EQUITY UPDATE (FIXED) ---
        if self.position == "none":
            self.equity = self.balance
        else:
            if self.position == "call":
                unrealized = (new_price - self.entry_price) / self.entry_price
            else:
                unrealized = (self.entry_price - new_price) / self.entry_price

            self.equity = self.balance + self.position_size + (self.position_size * unrealized)

        self.equity_curve.append(self.equity)

        # --- DRAWDOWN TRACKING ---
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # --- INFO ---
        info = {
            "balance": self.balance,
            "equity": self.equity,
            "trade_count": self.trade_count,
            "max_drawdown": self.max_drawdown
        }

        return obs, reward, done, info