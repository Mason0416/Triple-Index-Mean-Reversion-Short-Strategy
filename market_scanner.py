"""全市場掃描模組。

從漲跌停價資料與當日 OHLCV 收盤價，找出「當日收盤即漲停」的個股事件，
並排除當日屬於「暫停先賣後買當沖」期間的標的。

注意：TaiwanStockPriceLimit 本身**不含收盤價**（欄位為
date, stock_id, reference_price, limit_up, limit_down），因此判斷是否漲停
必須額外用對應股票的實際 close 與 limit_up 比對，故 find_limit_up_events
需要傳入 OHLCV 的 data 字典。
"""

import numpy as np
import pandas as pd


def find_limit_up_events(price_limit_df: pd.DataFrame, data: dict,
                         use_intraday: bool = False) -> pd.DataFrame:
    """找出漲停事件 (date, stock_id)，支援收盤定義與盤中觸及定義。

    TaiwanStockPriceLimit 不含個股價格，故以各股票 OHLCV 的實際價格與
    limit_up 比對：

    - use_intraday=False（預設）：以 close 判定「漲停作收」，
      條件 close == limit_up（容許 0.005 元浮點誤差）。
    - use_intraday=True：以 high 判定「盤中曾觸及漲停」，
      條件 high >= limit_up * 0.995（容許 0.5% 浮點誤差）。

    Args:
        price_limit_df: get_price_limit_data 回傳的原始 DataFrame，
            需含 date, stock_id, limit_up 欄位。
        data: dict[str, pd.DataFrame]，各股票標準 OHLCV（含 close/high）。
        use_intraday: False 用 close 判漲停作收；True 用 high 判盤中觸及。

    Returns:
        DataFrame，欄位 date（datetime）, stock_id，依日期排序。
    """
    if price_limit_df is None or price_limit_df.empty or not data:
        return pd.DataFrame(columns=["date", "stock_id"])

    price_col = "high" if use_intraday else "close"

    pl = price_limit_df[["date", "stock_id", "limit_up"]].copy()
    pl["date"] = pd.to_datetime(pl["date"])
    pl["stock_id"] = pl["stock_id"].astype(str)
    pl["limit_up"] = pl["limit_up"].astype(float)
    pl = pl[pl["limit_up"] > 0]

    # 把所有股票的目標價格欄（close 或 high）攤平成 long-format，再與漲停價 merge
    price_frames = []
    for stock_id, df in data.items():
        if price_col not in df.columns or df.empty:
            continue
        t = pd.DataFrame({
            "date": df.index,
            "stock_id": str(stock_id),
            price_col: df[price_col].astype(float).values,
        })
        price_frames.append(t)

    if not price_frames:
        return pd.DataFrame(columns=["date", "stock_id"])

    price_long = pd.concat(price_frames, ignore_index=True)

    merged = pl.merge(price_long, on=["date", "stock_id"], how="inner")
    if use_intraday:
        # 盤中最高價觸及漲停價（容許 0.5% 浮點誤差）
        is_limit_up = merged["high"] >= merged["limit_up"] * 0.995
    else:
        # 收盤價等於漲停價（容許微小浮點誤差）即視為漲停作收
        is_limit_up = np.isclose(merged["close"], merged["limit_up"], atol=0.005)
    events = merged.loc[is_limit_up, ["date", "stock_id"]].copy()

    events = events.drop_duplicates().sort_values("date").reset_index(drop=True)
    label = "盤中觸及漲停" if use_intraday else "漲停作收"
    print(f"[find_limit_up_events] {label}事件數：{len(events)}")
    return events


def filter_day_trading_eligible(limit_up_df: pd.DataFrame,
                                suspension_df: pd.DataFrame) -> pd.DataFrame:
    """移除當日屬於「暫停先賣後買當沖」期間的漲停事件。

    TaiwanStockDayTradingSuspension 每列為一檔股票的暫停期間
    [date, end_date]。若某漲停事件的日期落在對應股票的任一暫停期間內，
    該事件即被移除（無法當日先賣後買放空）。

    Args:
        limit_up_df: find_limit_up_events 的輸出（date, stock_id）。
        suspension_df: get_day_trading_suspension 的原始輸出
            （stock_id, date, end_date, reason）。

    Returns:
        過濾後的 DataFrame，格式同 limit_up_df。
    """
    if limit_up_df is None or limit_up_df.empty:
        return pd.DataFrame(columns=["date", "stock_id"])
    if suspension_df is None or suspension_df.empty:
        print("[filter_day_trading_eligible] 無暫停名單，全部保留")
        return limit_up_df.copy().reset_index(drop=True)

    ev = limit_up_df.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev["stock_id"] = ev["stock_id"].astype(str)

    susp = suspension_df[["stock_id", "date", "end_date"]].copy()
    susp["stock_id"] = susp["stock_id"].astype(str)
    susp["date"] = pd.to_datetime(susp["date"])
    susp["end_date"] = pd.to_datetime(susp["end_date"])

    # 以 stock_id 連接事件與暫停期間，標記落在區間內者
    merged = ev.merge(
        susp, on="stock_id", how="left", suffixes=("", "_susp")
    )
    in_window = (
        merged["date"].ge(merged["date_susp"])
        & merged["date"].le(merged["end_date"])
    )
    bad = (
        merged.loc[in_window, ["date", "stock_id"]]
        .drop_duplicates()
    )

    # 從事件中剔除落在暫停期間者
    ev_keys = pd.MultiIndex.from_frame(ev[["date", "stock_id"]])
    bad_keys = pd.MultiIndex.from_frame(bad[["date", "stock_id"]])
    keep = ev[~ev_keys.isin(bad_keys)].reset_index(drop=True)

    print(f"[filter_day_trading_eligible] 移除 {len(ev) - len(keep)} 筆暫停期間事件，"
          f"剩餘 {len(keep)} 筆")
    return keep
