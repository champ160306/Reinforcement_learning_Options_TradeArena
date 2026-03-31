def grade(total_profit, max_drawdown, trade_count):
    profit_score = min(max(total_profit / 100, 0), 1)

    risk_score = max(0, 1 - (abs(max_drawdown) / 50))

    discipline_score = max(0, 1 - (trade_count / 10))

    final_score = (
        0.4 * profit_score +
        0.3 * risk_score +
        0.3 * discipline_score
    )

    return round(final_score, 3)