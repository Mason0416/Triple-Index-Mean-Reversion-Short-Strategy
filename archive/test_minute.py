"""分鐘 K 線探測腳本（TaiwanStockKBar）+ 漲停觸及檢查 + 資料量估算。

注意：此版本 FinMind 沒有 'TaiwanStockMinutePrice' 這個 dataset，
分鐘級資料用 'TaiwanStockKBar'（欄位 date, minute, stock_id,
open, high, low, close, volume；minute 為當日時間 HH:MM:SS）。
TaiwanStockKBar 每次只回傳單一交易日。

步驟：
  1. 抓 2330 2023-03-22 單日分鐘 K：印 columns / 筆數 / 前 5 筆 / 最高價時間
  2. 確認當天分鐘線最高價是否觸及漲停價（max >= limit_up * 0.995）
  3. 印出單股單日資料量（筆數、KB），估算全市場一天的筆數
"""

import os

import pandas as pd
from dotenv import load_dotenv
from FinMind.data import DataLoader


KBAR_DATASET = "TaiwanStockKBar"
PRICE_LIMIT_DATASET = "TaiwanStockPriceLimit"
STOCK_ID = "2330"
DATE = "2023-03-22"
DATA_DIR = "data"


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


def fetch_kbar_one_day(api: DataLoader, stock_id: str,
                       date: str) -> pd.DataFrame:
    """抓取單一股票單一交易日的分鐘 K 線（TaiwanStockKBar）。

    Args:
        api: DataLoader 實例。
        stock_id: 股票代號。
        date: 交易日 "YYYY-MM-DD"（start 與 end 同一天）。

    Returns:
        FinMind 原始 DataFrame。
    """
    return api.get_data(
        dataset=KBAR_DATASET, data_id=stock_id,
        start_date=date, end_date=date,
    )


def get_limit_up(api: DataLoader, stock_id: str, date: str) -> float:
    """取得某股票某日的漲停價（TaiwanStockPriceLimit 的 limit_up）。

    Args:
        api: DataLoader 實例。
        stock_id: 股票代號。
        date: 交易日 "YYYY-MM-DD"。

    Returns:
        該股當日 limit_up（float）；查無資料回 NaN。
    """
    d = api.get_data(
        dataset=PRICE_LIMIT_DATASET, data_id=stock_id,
        start_date=date, end_date=date,
    )
    if d is None or d.empty or "limit_up" not in d.columns:
        return float("nan")
    return float(d["limit_up"].iloc[0])


def main() -> None:
    """執行三項分鐘 K 線探測。"""
    api = get_loader()

    # 1. 單一股票單一天
    print("=" * 60)
    print(f"1. TaiwanStockKBar {STOCK_ID} {DATE}（單日分鐘 K）")
    print("=" * 60)
    df = fetch_kbar_one_day(api, STOCK_ID, DATE)
    print("columns:", list(df.columns))
    print("筆數:", len(df))
    print(df.head(5).to_string())

    day_high = float(df["high"].max())
    hit = df[df["high"] == day_high]
    print(f"\n當日最高價：{day_high}")
    print(f"最高價出現時間：{list(hit['minute'])}（首次 {hit['minute'].iloc[0]}）")

    # 2. 最高價是否觸及漲停價
    print("\n" + "=" * 60)
    print(f"2. {DATE} {STOCK_ID} 分鐘線最高價是否觸及漲停")
    print("=" * 60)
    limit_up = get_limit_up(api, STOCK_ID, DATE)
    threshold = limit_up * 0.995
    touched = bool((df["high"] >= threshold).any())
    print(f"當日 limit_up（漲停價）：{limit_up}")
    print(f"觸及門檻（limit_up × 0.995）：{threshold:.4f}")
    print(f"分鐘線最高價：{day_high}")
    print(f"是否曾觸及（max high >= limit_up*0.995）：{touched}")
    if touched:
        touch_rows = df[df["high"] >= threshold]
        print(f"觸及時間：{list(touch_rows['minute'])}")

    # 3. 資料量估算
    print("\n" + "=" * 60)
    print("3. 資料量與全市場估算")
    print("=" * 60)
    n_rows = len(df)
    size_kb = df.memory_usage(deep=True).sum() / 1024.0
    print(f"單股單日：{n_rows} 筆，約 {size_kb:.1f} KB（in-memory, deep）")

    # 用 data/ 既有 CSV 檔數當作全市場檔數估計
    n_stocks = len([f for f in os.listdir(DATA_DIR)
                    if f.endswith(".csv")]) if os.path.isdir(DATA_DIR) else 0
    if n_stocks:
        est_rows = n_rows * n_stocks
        est_mb = size_kb * n_stocks / 1024.0
        print(f"全市場估計檔數（data/ 的 CSV 數）：{n_stocks}")
        print(f"全市場一天估計筆數：{n_rows} × {n_stocks} ≈ {est_rows:,} 筆")
        print(f"全市場一天估計大小：約 {est_mb:.1f} MB")
    else:
        print("（data/ 無 CSV，無法估算全市場檔數）")


if __name__ == "__main__":
    main()
