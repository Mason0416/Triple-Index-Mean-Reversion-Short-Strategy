"""指數資料可用性確認 + 跳空幅度描述性統計。

確認三個指數／期貨日線資料源，並計算 2023 全年每日跳空幅度的分布：
  1. 台指期日線（TX 近月）— 用 data_loader.get_tx_data
  2. 加權指數（TAIEX）— TaiwanStockPrice, data_id='TAIEX'
  3. 櫃買指數（TPEx） — TaiwanStockPrice, data_id='TPEx'

注意：此版本 FinMind 沒有 'TaiwanStockMarketIndex' 這個 dataset，
加權／櫃買指數都掛在 'TaiwanStockPrice' 底下（data_id 為 'TAIEX' / 'TPEx'，
TPEx 的大小寫需正確）。指數的價格欄位為 open/max/min/close。
"""

import os

import pandas as pd
from dotenv import load_dotenv
from FinMind.data import DataLoader

from data_loader import get_tx_data


START_DATE = "2023-01-01"
END_DATE = "2023-12-31"


def get_loader() -> DataLoader:
    """建立並（若有 token）登入 FinMind DataLoader。

    Returns:
        已盡可能登入的 DataLoader 實例。
    """
    load_dotenv()
    api = DataLoader()
    token = os.getenv("FINMIND_TOKEN")
    if token:
        api.login_by_token(api_token=token)
    return api


def get_index(api: DataLoader, data_id: str,
              start_date: str, end_date: str) -> pd.DataFrame:
    """抓取指數日線（掛在 TaiwanStockPrice 底下）。

    Args:
        api: DataLoader 實例。
        data_id: 指數代號，例如 'TAIEX'（加權）或 'TPEx'（櫃買）。
        start_date: 起始日期 "YYYY-MM-DD"。
        end_date: 結束日期 "YYYY-MM-DD"。

    Returns:
        FinMind 原始 DataFrame（含 date, open, max, min, close 等）。
    """
    return api.get_data(
        dataset="TaiwanStockPrice", data_id=data_id,
        start_date=start_date, end_date=end_date,
    )


def compute_gap(df: pd.DataFrame, open_col: str = "open",
                close_col: str = "close") -> pd.Series:
    """計算每日跳空幅度 (open - prev_close) / prev_close。

    會先確保依日期排序；首日無前一日 close 故為 NaN 並剔除。

    Args:
        df: 含日期與 open/close 的 DataFrame。若 index 不是日期，
            需有 'date' 欄。
        open_col: 開盤價欄名。
        close_col: 收盤價欄名。

    Returns:
        跳空幅度 Series（已去除 NaN）。
    """
    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"])
        work = work.set_index("date")
    work = work.sort_index()

    open_ = work[open_col].astype(float)
    close = work[close_col].astype(float)
    prev_close = close.shift(1)
    gap = (open_ - prev_close) / prev_close
    return gap.dropna()


def confirm_source(name: str, df: pd.DataFrame) -> None:
    """印出某資料源的 columns / 前 3 筆 / 筆數。"""
    print("=" * 60)
    print(name)
    print("=" * 60)
    print("columns:", list(df.columns))
    print("筆數:", len(df))
    print(df.head(3).to_string())


def main() -> None:
    """確認三個資料源並印出跳空幅度描述性統計。"""
    api = get_loader()

    # 1. 台指期日線（已有）
    tx_df = get_tx_data(START_DATE, END_DATE)
    confirm_source("1. 台指期日線（TX 近月，get_tx_data）", tx_df)

    # 2. 加權指數 TAIEX
    taiex_df = None
    try:
        taiex_df = get_index(api, "TAIEX", START_DATE, END_DATE)
        if taiex_df is None or taiex_df.empty:
            print("\n[警告] 加權指數 TAIEX 回傳空資料")
            taiex_df = None
        else:
            confirm_source("2. 加權指數（TaiwanStockPrice, TAIEX）", taiex_df)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[錯誤] 抓取加權指數 TAIEX 失敗：{exc}")

    # 3. 櫃買指數 TPEx
    tpex_df = None
    try:
        tpex_df = get_index(api, "TPEx", START_DATE, END_DATE)
        if tpex_df is None or tpex_df.empty:
            print("\n[警告] 櫃買指數 TPEx 回傳空資料")
            tpex_df = None
        else:
            confirm_source("3. 櫃買指數（TaiwanStockPrice, TPEx）", tpex_df)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[錯誤] 抓取櫃買指數 TPEx 失敗：{exc}")

    # 4. 三者跳空幅度描述性統計（2023 全年）
    print("\n" + "=" * 60)
    print("4. 每日跳空幅度 (open - prev_close)/prev_close 描述性統計（2023）")
    print("=" * 60)
    sources = [("台指期 TX", tx_df), ("加權指數 TAIEX", taiex_df),
               ("櫃買指數 TPEx", tpex_df)]
    for label, df in sources:
        if df is None or df.empty:
            print(f"\n--- {label} ---\n（無資料，略過）")
            continue
        gap = compute_gap(df)
        print(f"\n--- {label} ---（樣本 {len(gap)} 天）")
        print(gap.describe().to_string())
        # 補充百分比表示
        print(f"以 % 表示：mean={gap.mean():.4%}，std={gap.std():.4%}，"
              f"min={gap.min():.4%}，max={gap.max():.4%}")


if __name__ == "__main__":
    main()
