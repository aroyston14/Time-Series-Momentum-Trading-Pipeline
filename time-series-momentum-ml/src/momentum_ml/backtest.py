# Daily long/cash (or optionally long/short) backtest of the model probabilities.
#
# Timing convention (also described in the README and the report):
#   - Features for date t use data up to and including the close of t.
#   - The model's probability for row t is turned into a signal s_t.
#   - With execution_lag = 0 the position pos_t = s_t is taken at the close of t (market on close).
#     This is idealised, because the closing price is also one of the inputs. With
#     execution_lag = L the position is pos_t = s_(t-L), so the trade happens L days later.
#   - pos_t earns the next day adjusted close return fwd_ret_t = P_(t+1) / P_t - 1, which is only
#     known after the decision is made. Each output row is labelled by its decision date t.
#   - Costs are cost_t = |pos_t - pos_(t-1)| * cost_bps / 10,000 (one way, per unit of notional)
#     and are taken off that row's return. The first entry is charged but a final exit is not.
#   - Cash earns 0. risk_free_rate is only used in the Sharpe ratio.
#   - Portfolio: each ticker is an equal weight sleeve, rebalanced back to equal weights every day
#     (the cost of rebalancing between sleeves is ignored). Buy and hold lets the weights drift.

import numpy as np
import pandas as pd

from momentum_ml.features import max_drawdown


# Turns probabilities into positions. The position is 1 when prob >= threshold, otherwise 0
# (cash), or -1 in long/short mode when prob < short_threshold.
def signals_from_probs(prob, threshold=0.5, long_short=False, short_threshold=0.5):
    probabilities = np.asarray(prob, dtype=float)
    positions = (probabilities >= threshold).astype(float)
    if long_short:
        positions = np.where(
            probabilities >= threshold,
            1.0,
            np.where(probabilities < short_threshold, -1.0, 0.0),
        )
    return positions


# Daily profit and loss for each ticker from its signals.
# frame needs date, ticker and fwd_ret columns plus the signal column, with one row per
# (date, ticker). The signal is the position wanted at the close of that date, cost_bps is the
# one way cost in basis points, and execution_lag is the number of days before a signal is acted on.
# Returns date, ticker, fwd_ret, position, turnover, gross_ret, cost and net_ret columns.
def backtest_positions(frame, signal_col="signal", cost_bps=5.0, execution_lag=0):
    required_columns = {"date", "ticker", "fwd_ret", signal_col}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"backtest frame missing columns {sorted(missing_columns)}")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("backtest frame has duplicate (date, ticker) rows")

    df = frame.sort_values(["ticker", "date"], kind="mergesort")[
        ["date", "ticker", "fwd_ret", signal_col]
    ].copy()

    # Lagging the signal within each ticker, so the first `execution_lag` days start in cash
    signals_by_ticker = df.groupby("ticker", sort=False)[signal_col]
    if execution_lag:
        df["position"] = signals_by_ticker.shift(execution_lag).fillna(0.0)
    else:
        df["position"] = df[signal_col].astype(float)

    # Turnover is the change in position from the previous day, which is what gets charged
    previous_position = df.groupby("ticker", sort=False)["position"].shift(1).fillna(0.0)
    df["turnover"] = (df["position"] - previous_position).abs()
    df["gross_ret"] = df["position"] * df["fwd_ret"]
    df["cost"] = df["turnover"] * cost_bps / 1e4
    df["net_ret"] = df["gross_ret"] - df["cost"]
    return (
        df.drop(columns=[signal_col])
        .sort_values(["date", "ticker"], kind="mergesort")
        .reset_index(drop=True)
    )


# Compounded net return of each trade, where a trade is a run of the same non-zero position.
# The exit cost (charged on the row where the position changes) is counted against the trade
# that is being closed.
def trade_returns(per_ticker):
    returns = []
    for _, ticker_rows in per_ticker.sort_values("date").groupby("ticker", sort=False):
        positions = ticker_rows["position"].to_numpy()
        net_returns = ticker_rows["net_ret"].to_numpy()
        costs = ticker_rows["cost"].to_numpy()
        num_rows = len(positions)

        start = 0
        while start < num_rows:

            # Days in cash are not part of any trade
            if positions[start] == 0:
                start += 1
                continue

            # Finding the last day of this run of identical positions
            end = start
            while end + 1 < num_rows and positions[end + 1] == positions[start]:
                end += 1
            trade_return = np.prod(1 + net_returns[start : end + 1]) - 1

            # The next row's cost covers |new position - old position|. The share of it that
            # belongs to closing this trade is |old| / |new - old| (all of it when going to cash).
            if end + 1 < num_rows:
                exit_cost = (
                    costs[end + 1]
                    * abs(positions[start])
                    / abs(positions[end + 1] - positions[start])
                )
                trade_return = (1 + trade_return) * (1 - exit_cost) - 1
            returns.append(trade_return)
            start = end + 1
    return pd.Series(returns, dtype=float)


# Equal weight portfolio of the ticker sleeves, rebalanced daily, indexed by decision date
def portfolio_returns(per_ticker):
    by_date = per_ticker.groupby("date")
    portfolio = pd.DataFrame(
        {
            "net_ret": by_date["net_ret"].mean(),
            "gross_ret": by_date["gross_ret"].mean(),
            "cost": by_date["cost"].mean(),
            "turnover": by_date["turnover"].mean(),
            "exposure": by_date["position"].apply(lambda positions: positions.abs().mean()),
        }
    )
    return portfolio


# Buy and hold with equal starting weights and no rebalancing, so the weights drift, plus a single
# entry cost. A missing (date, ticker) observation counts as a zero return for that ticker.
def buy_and_hold_returns(frame, cost_bps=5.0):
    wide_returns = (
        frame.pivot(index="date", columns="ticker", values="fwd_ret").sort_index().fillna(0.0)
    )
    growth = (1 + wide_returns).cumprod()

    # Portfolio value after each day's return, starting from 1
    value_end = growth.mean(axis=1)
    value_start = value_end.shift(1).fillna(1.0)
    daily_return = value_end / value_start - 1

    # Only the first day has a cost and any turnover
    cost = pd.Series(0.0, index=daily_return.index)
    cost.iloc[0] = cost_bps / 1e4
    turnover = pd.Series(0.0, index=daily_return.index)
    turnover.iloc[0] = 1.0
    return pd.DataFrame(
        {
            "net_ret": daily_return - cost,
            "gross_ret": daily_return,
            "cost": cost,
            "turnover": turnover,
            "exposure": 1.0,
        }
    )


# Summary statistics for a daily portfolio return frame.
#   - Annualised return is geometric, and volatility is std * sqrt(periods per year).
#   - Sharpe = mean daily excess return / std * sqrt(periods), with the risk-free rate converted
#     to a daily rate geometrically.
#   - sharpe_tstat is roughly Sharpe * sqrt(years), a rough significance check assuming iid days.
#   - Calmar = annualised return / |max drawdown|, only given for at least a year of data and a
#     drawdown below zero.
#   - Hit rate is the share of days with a position open that had a positive net return.
#   - Turnover is annualised (total |change in position| per sleeve per year, averaged over sleeves).
def performance_metrics(
    port, risk_free_rate=0.0, periods_per_year=252, trades=None, n_position_changes=None
):
    net_returns = port["net_ret"].astype(float).fillna(0.0)
    num_days = len(net_returns)
    if num_days == 0:
        raise ValueError("Empty return series")
    years = num_days / periods_per_year

    # Return and volatility
    equity = (1 + net_returns).cumprod()
    cumulative_return = float(equity.iloc[-1] - 1)
    if equity.iloc[-1] > 0:
        annualised_return = float(equity.iloc[-1] ** (1 / years) - 1)
    else:
        annualised_return = -1.0
    if num_days > 1:
        annualised_vol = float(net_returns.std(ddof=1) * np.sqrt(periods_per_year))
    else:
        annualised_vol = float("nan")

    # Sharpe ratio on excess returns
    daily_risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = net_returns - daily_risk_free
    excess_std = excess_returns.std(ddof=1)
    if num_days > 1 and excess_std > 0:
        sharpe = float(excess_returns.mean() / excess_std * np.sqrt(periods_per_year))
    else:
        sharpe = float("nan")

    # Drawdown is measured from a starting value of 1 so a loss on day one counts
    drawdown = max_drawdown(np.r_[1.0, equity.to_numpy()])
    calmar = annualised_return / abs(drawdown) if years >= 1 and drawdown < 0 else float("nan")

    exposure = port["exposure"] if "exposure" in port else pd.Series(1.0, index=port.index)
    in_market = exposure > 0
    hit_rate = float((net_returns[in_market] > 0).mean()) if in_market.any() else float("nan")

    return {
        "days": num_days,
        "years": years,
        "cumulative_return": cumulative_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_vol,
        "sharpe": sharpe,
        "sharpe_tstat": sharpe * np.sqrt(years) if not np.isnan(sharpe) else float("nan"),
        "max_drawdown": drawdown,
        "calmar": float(calmar),
        "hit_rate": hit_rate,
        "avg_trade_return": (
            float(trades.mean()) if trades is not None and len(trades) else float("nan")
        ),
        "n_trades": int(len(trades)) if trades is not None else int((port["turnover"] > 0).sum()),
        "n_position_changes": (
            int(n_position_changes)
            if n_position_changes is not None
            else int((port["turnover"] > 0).sum())
        ),
        "annual_turnover": float(port["turnover"].mean() * periods_per_year),
        "avg_exposure": float(exposure.mean()),
        "total_cost": float(port["cost"].sum()),
    }


# Backtests one signal vector (lined up with the rows of frame) and returns the portfolio frame
# and its performance metrics. execution_lag and cost_bps override the config values if given.
def run_strategy(frame, signal, backtest_config, execution_lag=None, cost_bps=None):
    strategy_frame = frame[["date", "ticker", "fwd_ret"]].copy()
    strategy_frame["signal"] = np.asarray(signal, dtype=float)
    if execution_lag is None:
        lag = int(backtest_config.get("execution_lag", 0))
    else:
        lag = execution_lag
    cost = float(backtest_config["cost_bps"]) if cost_bps is None else cost_bps

    per_ticker = backtest_positions(strategy_frame, "signal", cost, lag)
    portfolio = portfolio_returns(per_ticker)
    metrics = performance_metrics(
        portfolio,
        float(backtest_config.get("risk_free_rate", 0.0)),
        int(backtest_config.get("periods_per_year", 252)),
        trades=trade_returns(per_ticker),
        n_position_changes=int((per_ticker["turnover"] > 0).sum()),
    )
    return portfolio, metrics


# Signals for the rule-based benchmarks, lined up with the rows of frame.
#   always_long - a position of 1 every day (equal weight, rebalanced daily)
#   ma_rule     - long when the adjusted close is above its N-day SMA (dist_ma_N > 0), else cash
def benchmark_signals(frame, backtest_config):
    window = int(backtest_config["ma_rule_window"])
    ma_column = f"dist_ma_{window}"
    if ma_column not in frame.columns:
        raise ValueError(f"MA-rule benchmark needs column {ma_column!r} in the frame")
    return {
        "always_long": np.ones(len(frame)),
        f"ma_rule_{window}d": (frame[ma_column].to_numpy() > 0).astype(float),
    }
