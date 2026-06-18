"""分點／法人籌碼資料集可用性探測。

測試三個（可能存在的）籌碼資料集，確認欄位與資料樣態。
測試日期含 2023-03-22（先前確認的漲停事件日），股票 2330。
資料集不存在或抓取失敗時印出錯誤訊息，並繼續測試下一個。
"""

import os

import pandas as pd
from dotenv import load_dotenv
from FinMind.data import DataLoader

import index_region_backtest as irb
from data_loader import get_day_trading_suspension, get_tx_data


STOCK_ID = "2330"
START_DATE = "2023-03-21"
END_DATE = "2023-03-23"


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


def probe(api: DataLoader, label: str, dataset: str, head: int) -> None:
    """探測單一資料集並印出欄位與前數筆；失敗則印錯誤訊息。

    Args:
        api: DataLoader 實例。
        label: 顯示用標題。
        dataset: FinMind dataset 名稱。
        head: 要印出的前幾筆。
    """
    print("\n" + "=" * 64)
    print(f"{label}（dataset={dataset}）")
    print("=" * 64)
    try:
        d = api.get_data(dataset=dataset, data_id=STOCK_ID,
                         start_date=START_DATE, end_date=END_DATE)
    except Exception as exc:  # noqa: BLE001
        print(f"[錯誤] 資料集不存在或抓取失敗：{repr(exc)[:200]}")
        return

    if d is None or d.empty:
        print("[警告] 回傳空資料（資料集存在但此股/區間無資料）")
        if d is not None:
            print("columns:", list(d.columns))
        return

    print("rows:", len(d))
    print("columns:", list(d.columns))
    print(d.head(head).to_string())


def get_triple_and_events(api: DataLoader) -> pd.DataFrame:
    """重建「TX AND TAIEX AND TPEx 開低走高 × 8%」的觸發事件 (date, stock_id)。

    Args:
        api: DataLoader 實例（抓指數用）。

    Returns:
        DataFrame（date, stock_id），依日期排序、去重。
    """
    data = irb.load_stock_data_cache_only()
    gains = irb.build_all_gains(data)
    susp = get_day_trading_suspension(irb.START_DATE, irb.END_DATE)
    ev = irb.filter_day_trading_eligible(
        irb.strong_events(gains, 0.08), susp)

    tx = get_tx_data(irb.START_DATE, irb.END_DATE)
    taiex = irb.get_index_ohlcv(api, "TAIEX", irb.START_DATE, irb.END_DATE)
    tpex = irb.get_index_ohlcv(api, "TPEx", irb.START_DATE, irb.END_DATE)
    kdzg = [set(irb.classify_regions(df)[
                irb.classify_regions(df) == "開低走高"].index)
            for df in (tx, taiex, tpex)]
    sig_dates = set.intersection(*kdzg)

    ev = ev[ev["date"].isin(sig_dates)][["date", "stock_id"]]
    ev = ev.drop_duplicates().sort_values(["date", "stock_id"])
    return ev.reset_index(drop=True)


def lock_limit_broker_count(report: pd.DataFrame, close: float,
                            limit_up: float, tol: float = 0.005) -> dict:
    """計算「鎖漲停分點數」。

    定義：當天收盤價 == 漲停價（鎖漲停）時，於該日「最高成交價位」
    買進（buy>0）的不同券商分點數量。非漲停作收則鎖漲停分點數為 0。

    Args:
        report: TaiwanStockTradingDailyReport（含 price/buy/securities_trader_id）。
        close: 當日收盤價。
        limit_up: 當日漲停價。
        tol: close == limit_up 的浮點容差。

    Returns:
        dict：max_price, is_limit_close, n_brokers_at_max, lock_count。
    """
    if report is None or report.empty:
        return {"max_price": None, "is_limit_close": False,
                "n_brokers_at_max": 0, "lock_count": 0}
    price = report["price"].astype(float)
    buy = report["buy"].astype(float)
    max_price = float(price.max())
    is_limit_close = (limit_up is not None and limit_up > 0
                      and abs(close - limit_up) < tol)
    at_max_buy = report[(price == max_price) & (buy > 0)]
    n_at_max = int(at_max_buy["securities_trader_id"].nunique())
    return {
        "max_price": max_price,
        "is_limit_close": is_limit_close,
        "n_brokers_at_max": n_at_max,
        "lock_count": n_at_max if is_limit_close else 0,
    }


def analyze_lock_limit(api: DataLoader) -> None:
    """步驟 1~3：觸發事件數、首筆事件分點明細、鎖漲停分點數。"""
    # 1. 觸發事件
    print("\n" + "=" * 64)
    print("1. 三指數AND 開低走高×8% 觸發事件")
    print("=" * 64)
    events = get_triple_and_events(api)
    print(f"distinct (date, stock_id) 組合數：{len(events)}")
    print(events.head(5).to_string())

    if events.empty:
        print("（無事件，略過後續）")
        return

    # 2. 首筆事件 → 分點買賣明細
    #    分點資料（TaiwanStockTradingDailyReport）約 2022 起才有；早期事件無分點
    #    資料，故步驟 2~3 取「2022 起的第一筆」事件來實際示範計算。
    broker_events = events[events["date"] >= pd.Timestamp("2022-01-01")]
    if broker_events.empty:
        print("（2022 起無事件，無法示範分點計算）")
        return
    print(f"（分點資料約 2022 起；2022 後事件數 {len(broker_events)}，"
          f"以下取其第一筆示範）")
    first = broker_events.iloc[0]
    ev_date = pd.Timestamp(first["date"]).strftime("%Y-%m-%d")
    ev_stock = str(first["stock_id"])
    print("\n" + "=" * 64)
    print(f"2. 首筆事件 {ev_stock} {ev_date} 的分點買賣（TaiwanStockTradingDailyReport）")
    print("=" * 64)
    report = api.get_data(dataset="TaiwanStockTradingDailyReport",
                          data_id=ev_stock, start_date=ev_date, end_date=ev_date)
    print("總筆數:", len(report))
    print("columns:", list(report.columns))
    print(report.head(10).to_string())

    # 3. 鎖漲停分點數
    print("\n" + "=" * 64)
    print(f"3. {ev_stock} {ev_date} 鎖漲停分點數")
    print("=" * 64)
    daily = api.taiwan_stock_daily(stock_id=ev_stock,
                                   start_date=ev_date, end_date=ev_date)
    close = float(daily["close"].iloc[0]) if not daily.empty else float("nan")
    pl = api.get_data(dataset="TaiwanStockPriceLimit", data_id=ev_stock,
                      start_date=ev_date, end_date=ev_date)
    limit_up = float(pl["limit_up"].iloc[0]) if not pl.empty else float("nan")

    res = lock_limit_broker_count(report, close, limit_up)
    print(f"收盤價 close       : {close}")
    print(f"漲停價 limit_up    : {limit_up}")
    print(f"最高成交價 max_price: {res['max_price']}")
    print(f"是否鎖漲停作收     : {res['is_limit_close']}")
    print(f"最高價位買進分點數 : {res['n_brokers_at_max']}")
    print(f"=> 鎖漲停分點數     : {res['lock_count']}")
    if not res["is_limit_close"]:
        print("（註：此事件為 8~9.5% 強勢股、非漲停作收，依定義鎖漲停分點數=0；"
              "最高價位買進分點數仍可參考）")


def main() -> None:
    """依序探測三個籌碼資料集。

    說明：使用者原本指定的 dataset 名稱在此版 FinMind 不存在
    （TaiwanStockShareholderActivity / TaiwanStockInstitutionalInvestors /
    TaiwanStockBrokerBuySell 皆回 enum 錯誤）。以下改用實際存在的對應資料集，
    並先各測一次「原名」以記錄其不存在。
    """
    api = get_loader()

    # 先記錄原始指定名稱不存在
    for original in ["TaiwanStockShareholderActivity",
                     "TaiwanStockInstitutionalInvestors",
                     "TaiwanStockBrokerBuySell"]:
        probe(api, f"[原指定] {original}", original, head=3)

    # 實際可用的對應資料集
    # 1. 外資持股／股權（最接近「股東活動」）
    probe(api, "1. 外資持股/股權（TaiwanStockShareholding）",
          "TaiwanStockShareholding", head=10)
    # 2. 三大法人
    probe(api, "2. 三大法人（TaiwanStockInstitutionalInvestorsBuySell）",
          "TaiwanStockInstitutionalInvestorsBuySell", head=5)
    # 3. 分點券商買賣
    probe(api, "3. 分點券商買賣（TaiwanStockTradingDailyReport）",
          "TaiwanStockTradingDailyReport", head=10)

    # 觸發事件 + 分點明細 + 鎖漲停分點數
    analyze_lock_limit(api)


if __name__ == "__main__":
    main()
