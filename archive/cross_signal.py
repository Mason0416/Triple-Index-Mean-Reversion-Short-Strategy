"""TX region × S&P500 regime 交叉訊號模組。

把「台股當日 TX 區間」與「全球 S&P500 regime」兩個維度交叉，
產生更細的隔日放空訊號：必須同時滿足指定的 tx_region 與 sp500_regime
兩個條件，才在隔日對該股票標記放空。
"""

import pandas as pd


def build_cross_signals(limit_up_events: pd.DataFrame,
                        tx_regions: pd.Series,
                        regime_df: pd.DataFrame,
                        cross_name: tuple,
                        all_dates,
                        stock_ids: list) -> pd.DataFrame:
    """依「TX region × S&P500 regime」雙條件建立隔日放空訊號矩陣。

    對每筆漲停事件 (date, stock_id)，同時檢查當日的 TX 區間與 regime；
    兩者都符合 cross_name 指定條件時，才在 all_dates 中該 date 的
    下一個交易日對應 stock_id 標 -1。

    Args:
        limit_up_events: 含 date, stock_id 的漲停（且可當沖）事件表。
        tx_regions: classify_tx_regions 的輸出（index=date，TX 區間字串）。
        regime_df: classify_market_regime 的輸出（含 'regime' 欄，index=date）。
        cross_name: tuple (tx_region, sp500_regime)，兩個條件都要符合，
            例如 ("開低走低", "同步走弱")。
        all_dates: 交易日序列（通常為 tx_df.index），決定「下一個交易日」。
        stock_ids: 訊號矩陣欄位（股票代號清單）。

    Returns:
        pd.DataFrame，index=all_dates、columns=stock_ids，值為 0 或 -1。
    """
    tx_region, sp500_regime = cross_name
    regime_series = regime_df["regime"]

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
        if event_date not in regime_series.index:
            continue
        # 兩個維度都要符合
        if tx_regions.loc[event_date] != tx_region:
            continue
        if regime_series.loc[event_date] != sp500_regime:
            continue

        pos = all_dates.searchsorted(event_date, side="right")
        if pos >= len(all_dates):
            continue
        matrix.at[all_dates[pos], stock_id] = -1

    return matrix
