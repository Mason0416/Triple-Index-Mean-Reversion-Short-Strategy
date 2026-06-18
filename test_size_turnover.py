"""股本／流通股數／周轉率／市值 資料源確認。

重要發現（已實測）：
- FinMind 的 TaiwanStockInfo（及 TaiwanStockInfoWithWarrant）**沒有股本欄位**，
  只有 industry_category / stock_id / stock_name / type / date。
- 但 TaiwanStockMarketValue 提供每日市值（market_value），且
  市值 / 收盤價 = 流通股數，實測對 2330 完全吻合（25.93 億股 → 應為 25,930,380,458）。

因此本腳本的「流通股數」改由 market_value / close 推導（不需股本），
周轉率 = 當日成交量(股) / 流通股數。
"""

import os

import pandas as pd
from dotenv import load_dotenv
from FinMind.data import DataLoader


STOCK_ID = "2330"
YEAR_START = "2023-01-01"
YEAR_END = "2023-12-31"


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


def confirm_stock_info(api: DataLoader) -> None:
    """步驟 1：確認 TaiwanStockInfo 欄位（並檢查是否含股本）。

    Args:
        api: DataLoader 實例。
    """
    print("=" * 60)
    print("1. TaiwanStockInfo（確認是否有股本 stock_capital）")
    print("=" * 60)
    info = api.get_data(dataset="TaiwanStockInfo")
    print("columns:", list(info.columns))
    print(info.head(5).to_string())

    has_capital = any("capital" in c.lower() for c in info.columns)
    print(f"\n是否含股本欄位（capital）：{has_capital}")
    if not has_capital:
        print("[注意] TaiwanStockInfo 不提供股本；流通股數改由"
              " TaiwanStockMarketValue / close 推導（見步驟 2）。")


def confirm_turnover(api: DataLoader) -> None:
    """步驟 2：以 2330 的 2023 年資料計算每日周轉率並確認合理性。

    周轉率 = 當日成交量(股) / 流通股數；
    流通股數 = 市值(market_value) / 收盤價(close)。

    Args:
        api: DataLoader 實例。
    """
    print("\n" + "=" * 60)
    print(f"2. 每日周轉率（{STOCK_ID} {YEAR_START[:4]}）")
    print("=" * 60)

    daily = api.taiwan_stock_daily(
        stock_id=STOCK_ID, start_date=YEAR_START, end_date=YEAR_END)
    mv = api.get_data(
        dataset="TaiwanStockMarketValue", data_id=STOCK_ID,
        start_date=YEAR_START, end_date=YEAR_END)

    daily = daily[["date", "Trading_Volume", "close"]].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    mv = mv[["date", "market_value"]].copy()
    mv["date"] = pd.to_datetime(mv["date"])

    df = daily.merge(mv, on="date", how="inner").sort_values("date")
    df["close"] = df["close"].astype(float)
    df["market_value"] = df["market_value"].astype(float)
    df["volume_shares"] = df["Trading_Volume"].astype(float)  # FinMind 成交量為股數

    df["shares_outstanding"] = df["market_value"] / df["close"]
    df["turnover"] = df["volume_shares"] / df["shares_outstanding"]

    show = df[["date", "volume_shares", "close", "market_value",
               "shares_outstanding", "turnover"]].head(5).copy()
    show["turnover_pct"] = (show["turnover"] * 100).round(4).astype(str) + "%"
    print(show.to_string(index=False))
    print(f"\n推導流通股數（首日）：{df['shares_outstanding'].iloc[0]:,.0f} 股"
          f"（2330 實際約 25.93 億股）")
    print(f"全年平均周轉率：{df['turnover'].mean():.4%}")


def confirm_market_value(api: DataLoader) -> None:
    """步驟 3：確認 TaiwanStockMarketValue 資料集是否存在。

    Args:
        api: DataLoader 實例。
    """
    print("\n" + "=" * 60)
    print("3. TaiwanStockMarketValue（市值資料集確認）")
    print("=" * 60)
    try:
        mv = api.get_data(
            dataset="TaiwanStockMarketValue", data_id=STOCK_ID,
            start_date="2023-01-01", end_date="2023-01-31")
        if mv is None or mv.empty:
            print("[錯誤] TaiwanStockMarketValue 回傳空資料")
            return
        print("columns:", list(mv.columns))
        print(mv.head(5).to_string())
    except Exception as exc:  # noqa: BLE001
        print(f"[錯誤] TaiwanStockMarketValue 不存在或抓取失敗：{exc}")


def main() -> None:
    """執行三項確認。"""
    api = get_loader()
    confirm_stock_info(api)
    confirm_turnover(api)
    confirm_market_value(api)


if __name__ == "__main__":
    main()
