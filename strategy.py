"""Single-stock moving-average crossover strategy + TX 區間放空訊號工具。"""

import numpy as np
import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Generate 5/20-day moving-average crossover signals.

    A golden cross produces 1, a death cross produces -1, and all other
    dates produce 0. Signals are calculated after the T-1 close and executed
    at the T open, so the strategy does not use look-ahead information.

    Args:
        df: OHLCV DataFrame containing a ``close`` column.

    Returns:
        Integer Series with the same index as ``df`` and values in 1/-1/0.
    """
    close = df["close"]
    fast = close.rolling(window=5).mean()
    slow = close.rolling(window=20).mean()

    above = fast > slow
    prev_above = above.shift(1, fill_value=False)

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[(~prev_above) & above] = 1
    signals[prev_above & (~above)] = -1

    return signals.shift(1, fill_value=0).astype(int)


def classify_tx_regions(tx_df: pd.DataFrame) -> pd.Series:
    """依「開盤相對前一日收盤的跳空幅度」將每個 TX 交易日分類。

    令 gap = 當日 open − 前一日 close，門檻為前一日 close 的 1%：
      大跳空上漲：open > prev_close * 1.01     （gap > 1%）
      小跳空上漲：0 < gap <= prev_close * 0.01  （0 < gap <= 1%）
      小跳空下跌：-prev_close * 0.01 <= gap < 0 （-1% <= gap < 0）
      大跳空下跌：open < prev_close * 0.99      （gap < -1%）
    open == prev_close（無跳空）或第一天（無前一日資料）標記為 NaN。

    Args:
        tx_df: TX 近月標準 OHLCV DataFrame，index 為 date，含 open/close。

    Returns:
        pd.Series，index 同 tx_df，值為四種跳空分類字串之一或 NaN。
    """
    open_ = tx_df["open"].astype(float)
    close = tx_df["close"].astype(float)
    prev_close = close.shift(1)

    gap = open_ - prev_close
    threshold = prev_close * 0.01  # 1% 門檻

    regions = pd.Series(np.nan, index=tx_df.index, dtype=object)
    regions[gap > threshold] = "大跳空上漲"
    regions[(gap > 0) & (gap <= threshold)] = "小跳空上漲"
    regions[(gap < 0) & (gap >= -threshold)] = "小跳空下跌"
    regions[gap < -threshold] = "大跳空下跌"
    return regions


def build_region_signals(limit_up_events: pd.DataFrame,
                         tx_regions: pd.Series,
                         region_name: str,
                         all_dates,
                         stock_ids: list) -> pd.DataFrame:
    """依「TX 區間 × 漲停事件」建立隔日放空訊號矩陣。

    對每一筆漲停事件 (date, stock_id)，若該 date 的 TX 區間等於 region_name，
    則在 all_dates 中該 date 的「下一個交易日」對應 stock_id 位置標記 -1（放空）。

    Args:
        limit_up_events: 含 date, stock_id 的漲停（且可當沖）事件表。
        tx_regions: classify_tx_regions 的輸出（index=date，值為區間字串）。
        region_name: 目標區間，例如 "開高走高"。
        all_dates: 交易日序列（通常為 tx_df.index），決定「下一個交易日」。
        stock_ids: 訊號矩陣的欄位（股票代號清單）。

    Returns:
        pd.DataFrame，index=all_dates、columns=stock_ids，值為 0 或 -1。
    """
    all_dates = pd.DatetimeIndex(all_dates)
    matrix = pd.DataFrame(0, index=all_dates, columns=stock_ids, dtype=int)

    if limit_up_events is None or limit_up_events.empty:
        return matrix

    sid_set = set(map(str, stock_ids))

    for _, row in limit_up_events.iterrows():
        event_date = pd.Timestamp(row["date"])
        stock_id = str(row["stock_id"])
        if stock_id not in sid_set:
            continue
        if event_date not in tx_regions.index:
            continue
        if tx_regions.loc[event_date] != region_name:
            continue

        # all_dates 中 event_date 之後的第一個交易日
        pos = all_dates.searchsorted(event_date, side="right")
        if pos >= len(all_dates):
            continue
        next_day = all_dates[pos]
        matrix.at[next_day, stock_id] = -1

    return matrix
