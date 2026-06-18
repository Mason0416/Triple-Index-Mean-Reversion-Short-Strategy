"""S&P 500（^GSPC）市場狀態訊號模組。

提供「全球（S&P500）× 台股（TX）」的雙維度 regime 分類，
以及對應的隔日放空訊號矩陣建立函數。

時區對齊重點：S&P500 是美股日期、TX 是台灣日期，兩者交易日不完全相同。
先用 reindex + ffill 把 S&P500 報酬對齊到 TX 交易日，再取 shift(1)（前一日），
確保使用的是「台股開盤前已經收盤、確定已知」的美股資訊，避免未來函數。
"""

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf


_SP500_CSV = os.path.join("data", "sp500_returns.csv")


def get_sp500_returns(start_date: str, end_date: str) -> pd.Series:
    """抓取 ^GSPC 日線並計算每日漲跌幅，帶本地 CSV 快取。

    第一次用 yfinance 抓 ^GSPC，計算收盤價 pct_change，存成
    data/sp500_returns.csv；之後直接讀快取。

    Args:
        start_date: 起始日期 "YYYY-MM-DD"。
        end_date: 結束日期 "YYYY-MM-DD"。

    Returns:
        pd.Series，index 為 date（datetime，已正規化為當日 00:00、不含實際時間），
        value 為當日收盤漲跌幅（float）。

    Raises:
        SystemExit: 當 yfinance 抓取失敗或回傳空資料時（印警告後 sys.exit(1)）。
    """
    if os.path.exists(_SP500_CSV):
        cached = pd.read_csv(_SP500_CSV, index_col=0, parse_dates=True)
        series = cached.iloc[:, 0].astype(float)
        series.index = pd.to_datetime(series.index).normalize()
        print(f"[get_sp500_returns] 讀取快取 {len(series)} 筆")
        return series

    try:
        raw = yf.download("^GSPC", start=start_date, end=end_date,
                          progress=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] yfinance 抓取 ^GSPC 失敗：{exc}")
        sys.exit(1)

    if raw is None or raw.empty:
        print("[警告] yfinance 回傳空的 ^GSPC 資料")
        sys.exit(1)

    # yfinance 可能回傳 MultiIndex 欄位 ('Close', '^GSPC')，取出單一收盤序列
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    returns = close.pct_change()
    returns.index = pd.to_datetime(returns.index).normalize()  # 去掉時間，只留日期
    returns = returns.astype(float)
    returns.name = "sp500_return"

    os.makedirs("data", exist_ok=True)
    returns.to_csv(_SP500_CSV)
    print(f"[get_sp500_returns] 抓取 {len(returns)} 筆並存快取")
    return returns


def classify_market_regime(tx_df: pd.DataFrame,
                           sp500_returns: pd.Series) -> pd.DataFrame:
    """以「全球(S&P500前一日) × 台股(TX當日)」雙維度分類市場 regime。

    tx_direction：台股強（close>open）/ 台股弱（close<open）/ NaN（相等）。
    global_direction：全球強（前一日 S&P500 漲跌幅>0）/ 全球弱（<0）/ NaN（==0）。
      S&P500 報酬先 reindex+ffill 對齊到 TX 交易日，再 shift(1) 取前一日，
      確保是台股開盤前已知的資訊。

    交叉產生 regime：
      逆勢獨走 = 全球弱 × 台股強；同步走強 = 全球強 × 台股強；
      台股獨弱 = 全球強 × 台股弱；同步走弱 = 全球弱 × 台股弱；
      任一維度為 NaN 則 regime 為 NaN。

    Args:
        tx_df: TX 近月標準 OHLCV DataFrame（index=date，含 open/close）。
        sp500_returns: get_sp500_returns 的輸出（美股日期、每日漲跌幅）。

    Returns:
        DataFrame，index=date，欄位 [tx_direction, global_direction, regime]。
    """
    open_ = tx_df["open"].astype(float)
    close = tx_df["close"].astype(float)

    # 台股當日方向
    tx_direction = pd.Series(np.nan, index=tx_df.index, dtype=object)
    tx_direction[close > open_] = "台股強"
    tx_direction[close < open_] = "台股弱"

    # 全球方向：對齊到 TX 交易日（reindex+ffill）後取前一日（shift(1)）
    aligned = sp500_returns.reindex(tx_df.index).ffill()
    prev_global = aligned.shift(1)
    global_direction = pd.Series(np.nan, index=tx_df.index, dtype=object)
    global_direction[prev_global > 0] = "全球強"
    global_direction[prev_global < 0] = "全球弱"

    # 交叉 regime
    regime = pd.Series(np.nan, index=tx_df.index, dtype=object)
    regime[(global_direction == "全球弱") & (tx_direction == "台股強")] = "逆勢獨走"
    regime[(global_direction == "全球強") & (tx_direction == "台股強")] = "同步走強"
    regime[(global_direction == "全球強") & (tx_direction == "台股弱")] = "台股獨弱"
    regime[(global_direction == "全球弱") & (tx_direction == "台股弱")] = "同步走弱"

    out = pd.DataFrame({
        "tx_direction": tx_direction,
        "global_direction": global_direction,
        "regime": regime,
    })

    print("[classify_market_regime] regime 分布：")
    print(out["regime"].value_counts(dropna=False).to_string())
    return out


def build_regime_signals(limit_up_events: pd.DataFrame,
                         regime_series: pd.Series,
                         regime_name: str,
                         all_dates,
                         stock_ids: list) -> pd.DataFrame:
    """依「regime × 漲停事件」建立隔日放空訊號矩陣。

    邏輯與 strategy.build_region_signals 相同，只是用 regime_series 取代
    tx_regions：對每筆漲停事件 (date, stock_id)，若該 date 的 regime 等於
    regime_name，則在 all_dates 中該 date 的下一個交易日對應 stock_id 標 -1。

    Args:
        limit_up_events: 含 date, stock_id 的漲停（且可當沖）事件表。
        regime_series: classify_market_regime 輸出的 regime 欄（index=date）。
        regime_name: 目標 regime，例如 "逆勢獨走"。
        all_dates: 交易日序列（通常為 tx_df.index）。
        stock_ids: 訊號矩陣欄位（股票代號清單）。

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
        if event_date not in regime_series.index:
            continue
        if regime_series.loc[event_date] != regime_name:
            continue

        pos = all_dates.searchsorted(event_date, side="right")
        if pos >= len(all_dates):
            continue
        matrix.at[all_dates[pos], stock_id] = -1

    return matrix
