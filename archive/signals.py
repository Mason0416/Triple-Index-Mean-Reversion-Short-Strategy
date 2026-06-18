"""投資組合訊號模組（策略撰寫入口）。

這是你撰寫策略的地方。回測引擎與策略完全分離：
只要你產生的訊號矩陣格式符合下方規範，就能直接餵進 PortfolioBacktester。

──────────────────────────────────────────────
訊號矩陣（signals）格式規範
──────────────────────────────────────────────
- 型態：pd.DataFrame
- index：datetime（所有股票交易日的聯集）
- columns：股票代號（字串），例如 "2330"
- 值：1（買入 / 做多）、-1（賣出 / 做空）、0（不動作）
- 缺值請填 0；某股票某日無資料時該格視為不交易

預設的 generate_signals_matrix 不做任何交易（全部回傳 0）。
請在 TODO 處填入你自己的策略邏輯。
檔案底部附有一個「均線交叉」範例，可解除註解作為起點。
"""

import pandas as pd


def generate_signals_matrix(data: dict) -> pd.DataFrame:
    """產生投資組合訊號矩陣（請在此撰寫你的策略）。

    預設回傳全 0 矩陣（不交易），僅建立正確的 index 與 columns，
    讓回測流程可以跑通。請將下方 TODO 換成你的策略邏輯。

    Args:
        data: dict[str, pd.DataFrame]，key 為股票代號，
            value 為標準 OHLCV DataFrame（index=datetime,
            columns=open/high/low/close/volume）。

    Returns:
        pd.DataFrame，index 為所有股票交易日的聯集，
        columns 為股票代號，值為 1 / -1 / 0。
    """
    # 以所有股票交易日的聯集為共同 index，建立全 0 矩陣骨架
    all_dates = sorted(set().union(*(df.index for df in data.values())))
    matrix = pd.DataFrame(
        0,
        index=pd.DatetimeIndex(all_dates, name="date"),
        columns=list(data.keys()),
        dtype=int,
    )

    # ──────────────────────────────────────────────
    # TODO: 在這裡寫你的策略，對 matrix 填入 1 / -1。
    # 範例：
    #   for stock_id, df in data.items():
    #       my_signal = your_logic(df)          # -> pd.Series (1/-1/0)
    #       matrix.loc[my_signal.index, stock_id] = my_signal.astype(int)
    # ──────────────────────────────────────────────

    return matrix.fillna(0).astype(int)


# ══════════════════════════════════════════════════
# 參考範例（預設停用）：5/20 日均線交叉策略
# 想以此為起點，把下面整段解除註解，並在 generate_signals_matrix
# 內呼叫 _example_ma_cross(df) 取代全 0 邏輯即可。
# ══════════════════════════════════════════════════
#
# def _example_ma_cross(df: pd.DataFrame) -> pd.Series:
#     """單股範例策略：5/20 日均線黃金/死亡交叉。
#
#     5 日均線上穿 20 日均線時做多（1），下穿時做空（-1）。
#     訊號以 T-1 日收盤計算、T 日開盤執行，避免 look-ahead bias。
#
#     Args:
#         df: 標準 OHLCV DataFrame，需含 close 欄位。
#
#     Returns:
#         pd.Series，index 與 df 相同，值為 1 / -1 / 0。
#     """
#     close = df["close"]
#     fast = close.rolling(window=5).mean()
#     slow = close.rolling(window=20).mean()
#
#     above = fast > slow
#     prev_above = above.shift(1, fill_value=False)
#
#     sig = pd.Series(0, index=df.index, dtype=int)
#     sig[(~prev_above) & above] = 1     # 黃金交叉 -> 做多
#     sig[prev_above & (~above)] = -1    # 死亡交叉 -> 做空
#
#     # 收盤後確認的訊號延至下一交易日開盤執行
#     return sig.shift(1, fill_value=0).astype(int)
