"""分鐘 K 線資料探測腳本（TaiwanStockKBar）。

注意：此版本 FinMind 並沒有 'TaiwanStockMinutePrice' 這個 dataset
（API 會回 enum 錯誤）。分鐘級資料改用 'TaiwanStockKBar'，欄位為：
date, minute, stock_id, open, high, low, close, volume
其中 minute 為當日時間字串（HH:MM:SS），high 為該分鐘最高價。

步驟：
  1. 抓單一股票單一天（2330, 2023-03-22），印 columns / 筆數 / 前 5 筆
  2. 抓同一股票一個月（2330, 2023-03），印筆數
  3. 印出 2023-03-22 當天 2330 的最高價出現在幾點
"""

import os

import pandas as pd
from dotenv import load_dotenv
from FinMind.data import DataLoader


DATASET = "TaiwanStockKBar"  # 取代不存在的 TaiwanStockMinutePrice


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


def fetch_kbar(api: DataLoader, stock_id: str,
               start_date: str, end_date: str) -> pd.DataFrame:
    """抓取分鐘 K 線（TaiwanStockKBar）原始資料，支援跨日。

    注意：TaiwanStockKBar 每次只回傳「單一交易日」的資料
    （API 限制：資料量過大，end_date 需與 start_date 同一天）。
    因此跨日區間以逐日呼叫後合併；非交易日（週末/假日）回空、自動略過。

    Args:
        api: DataLoader 實例。
        stock_id: 股票代號，例如 "2330"。
        start_date: 起始日期 "YYYY-MM-DD"。
        end_date: 結束日期 "YYYY-MM-DD"（含）。

    Returns:
        合併後的 FinMind 原始 DataFrame（含 date, minute, open, high,
        low, close, volume 等欄位）；無資料時回空 DataFrame。
    """
    frames = []
    for day in pd.date_range(start_date, end_date):
        ds = day.strftime("%Y-%m-%d")
        try:
            one = api.get_data(
                dataset=DATASET, data_id=stock_id,
                start_date=ds, end_date=ds,
            )
        except Exception:  # noqa: BLE001  # 非交易日或暫時性錯誤，略過
            continue
        if one is not None and not one.empty:
            frames.append(one)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def find_day_high_time(day_df: pd.DataFrame) -> None:
    """印出單日最高價及其出現時間。

    當日最高價取自 high 欄位的最大值；若多個分鐘並列最高，全部列出，
    並標示首次出現時間。

    Args:
        day_df: 單一交易日的 TaiwanStockKBar DataFrame（含 minute, high）。
    """
    if day_df.empty:
        print("（當日無資料）")
        return
    day_high = float(day_df["high"].max())
    hit = day_df[day_df["high"] == day_high]
    print(f"當日最高價：{day_high}")
    print(f"最高價出現時間（minute）：{list(hit['minute'])}")
    print(f"首次出現於：{hit['minute'].iloc[0]}")


def main() -> None:
    """執行三項分鐘 K 線探測。"""
    api = get_loader()

    # 1. 單一股票單一天
    print("=" * 60)
    print("1. TaiwanStockKBar 2330 2023-03-22（單日）")
    print("=" * 60)
    day_df = fetch_kbar(api, "2330", "2023-03-22", "2023-03-22")
    print("columns:", list(day_df.columns))
    print("筆數:", len(day_df))
    print(day_df.head(5).to_string())

    # 2. 同一股票一個月
    print("\n" + "=" * 60)
    print("2. TaiwanStockKBar 2330 2023-03-01 ~ 2023-03-31（整月）")
    print("=" * 60)
    month_df = fetch_kbar(api, "2330", "2023-03-01", "2023-03-31")
    print("筆數:", len(month_df))

    # 3. 2023-03-22 最高價出現在幾點
    print("\n" + "=" * 60)
    print("3. 2023-03-22 2330 最高價出現時間")
    print("=" * 60)
    find_day_high_time(day_df)


if __name__ == "__main__":
    main()
