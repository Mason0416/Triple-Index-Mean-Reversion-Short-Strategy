"""分點籌碼分析：依「主力集中度」分組回測強勢股隔日放空。

事件集 = 「TX AND TAIEX AND TPEx 開低走高 × 8% × 市值≤100億 × 周轉率≥0.5%」
且 date >= 2022-01-01（分點資料 TaiwanStockTradingDailyReport 約 2022 起才有）。

主力集中度 = 前 3 大淨買超分點（net_buy = buy - sell，取 net_buy>0）的合計淨買超
÷ 當日總成交量（OHLCV volume，與分點 buy/sell 同為「股」單位）。值介於 0~1，
越高代表籌碼越集中於少數主力。

假說：集中度高 → 主力控盤、隔日沖出貨壓力大 → 放空勝率更高；
      集中度低 → 散戶分散買 → 隔日走勢不確定 → 放空勝率較低。

回測沿用 index_region_backtest 的設定（停損 8%、滑價 0.15%、手續費+當沖稅、1000 股）。
"""

import os
import time

import pandas as pd
from dotenv import load_dotenv
from FinMind.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import index_region_backtest as irb
from data_loader import (
    calc_turnover,
    get_day_trading_suspension,
    get_tx_data,
    load_market_value_multiple,
)
from market_scanner import filter_day_trading_eligible


plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang TC", "Heiti TC", "STHeiti", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

START_2022 = "2022-01-01"
OUTPUT_DIR = "output"
DATA_DIR = "data"
MV_CAP = 100e8        # 市值上限 100 億
TURN_MIN = 0.005      # 周轉率下限 0.5%
BINS = ["低集中(<10%)", "中低(10-30%)", "中高(30-50%)", "高集中(>=50%)"]


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


def get_filtered_events_2022(api: DataLoader, data: dict) -> pd.DataFrame:
    """重建濾網事件並只保留 2022 起的 (date, stock_id)。

    濾網：三指數AND 開低走高、漲幅>=8%且<9.5%、市值<=100億、周轉率>=0.5%。

    Args:
        api: DataLoader 實例（抓指數用）。
        data: 個股 OHLCV 字典。

    Returns:
        DataFrame（date, stock_id），date >= 2022-01-01。
    """
    gains = irb.build_all_gains(data)
    susp = get_day_trading_suspension(irb.START_DATE, irb.END_DATE)
    ev = filter_day_trading_eligible(irb.strong_events(gains, 0.08), susp)

    tx = get_tx_data(irb.START_DATE, irb.END_DATE)
    taiex = irb.get_index_ohlcv(api, "TAIEX", irb.START_DATE, irb.END_DATE)
    tpex = irb.get_index_ohlcv(api, "TPEx", irb.START_DATE, irb.END_DATE)
    kdzg = [set(irb.classify_regions(df)[
                irb.classify_regions(df) == "開低走高"].index)
            for df in (tx, taiex, tpex)]
    sig_dates = set.intersection(*kdzg)
    ev = ev[ev["date"].isin(sig_dates)].copy()

    sids = sorted(ev["stock_id"].unique())
    mv = load_market_value_multiple(sids, irb.START_DATE, irb.END_DATE)
    turn = {sid: calc_turnover(data[sid], mv[sid])
            for sid in sids if sid in mv and sid in data}
    mv_vals, turn_vals = [], []
    for _, row in ev.iterrows():
        sid, d = row["stock_id"], row["date"]
        m = mv.get(sid)
        mv_vals.append(float(m.loc[d, "market_value"])
                       if (m is not None and d in m.index) else float("nan"))
        ts = turn.get(sid)
        turn_vals.append(float(ts.loc[d])
                         if (ts is not None and d in ts.index) else float("nan"))
    ev["market_value"] = mv_vals
    ev["turnover"] = turn_vals
    ev = ev[(ev["market_value"] <= MV_CAP) & (ev["turnover"] >= TURN_MIN)]
    ev = ev[ev["date"] >= pd.Timestamp(START_2022)]
    return ev[["date", "stock_id"]].sort_values(
        ["date", "stock_id"]).reset_index(drop=True)


def get_broker_report(api: DataLoader, date_str: str,
                      stock_id: str) -> pd.DataFrame:
    """抓取單一 (date, stock_id) 的分點買賣，帶 data/broker/ 快取。

    Args:
        api: DataLoader 實例。
        date_str: 日期 "YYYY-MM-DD"。
        stock_id: 股票代號。

    Returns:
        TaiwanStockTradingDailyReport DataFrame（可能為空）。
    """
    broker_dir = os.path.join(DATA_DIR, "broker")
    path = os.path.join(broker_dir, f"{date_str}_{stock_id}.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    try:
        df = api.get_data(dataset="TaiwanStockTradingDailyReport",
                          data_id=stock_id, start_date=date_str,
                          end_date=date_str)
    except Exception:  # noqa: BLE001
        df = pd.DataFrame()
    if df is None:
        df = pd.DataFrame()
    os.makedirs(broker_dir, exist_ok=True)
    df.to_csv(path, index=False)
    time.sleep(0.3)
    return df


def compute_concentration(report: pd.DataFrame, volume: float) -> float:
    """主力集中度 = 前 3 大淨買超分點合計淨買超 ÷ 當日總成交量。

    每個分點先彙總跨價位的 buy/sell，net_buy = buy - sell；取 net_buy>0 者
    由大到小排序，前三大合計除以 volume。

    Args:
        report: TaiwanStockTradingDailyReport（含 buy/sell/securities_trader_id）。
        volume: 當日總成交量（股）。

    Returns:
        集中度（0~1）；資料不足或 volume<=0 回 -1（代表略過）。
    """
    if (report is None or report.empty or "buy" not in report.columns
            or volume is None or volume <= 0):
        return -1.0
    g = report.groupby("securities_trader_id").agg(
        buy=("buy", "sum"), sell=("sell", "sum"))
    g["net"] = g["buy"].astype(float) - g["sell"].astype(float)
    top3 = g.loc[g["net"] > 0, "net"].sort_values(ascending=False).head(3).sum()
    return float(top3 / volume)


def bin_label(c: float) -> str:
    """把主力集中度對應到四個分組標籤。

    Args:
        c: 主力集中度（0~1）。

    Returns:
        分組標籤字串。
    """
    if c < 0.10:
        return "低集中(<10%)"
    if c < 0.30:
        return "中低(10-30%)"
    if c < 0.50:
        return "中高(30-50%)"
    return "高集中(>=50%)"


def main() -> None:
    """主流程：取事件 → 抓分點 → 算主力集中度 → 分組回測 → 摘要+圖。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api = get_loader()

    # 步驟 1：事件清單（2022+）
    data = irb.load_stock_data_cache_only()
    print(f"個股 cache-only 載入：{len(data)} 檔")
    events = get_filtered_events_2022(api, data)
    print(f"\n步驟 1：2022 起濾網事件數 = {len(events)}")

    # 步驟 2~3：抓分點、算主力集中度
    print("\n步驟 2~3：抓分點並計算主力集中度")
    total = len(events)
    records = []
    skipped = 0
    for i, row in enumerate(events.itertuples(index=False), start=1):
        d = pd.Timestamp(row.date)
        date_str = d.strftime("%Y-%m-%d")
        stock_id = str(row.stock_id)
        report = get_broker_report(api, date_str, stock_id)
        ohlcv = data.get(stock_id)
        volume = (float(ohlcv.loc[d, "volume"])
                  if (ohlcv is not None and d in ohlcv.index) else 0.0)
        conc = compute_concentration(report, volume)
        print(f"已抓 {date_str}_{stock_id} ({i}/{total}) 集中度="
              f"{conc:.4f}" if conc >= 0 else
              f"已抓 {date_str}_{stock_id} ({i}/{total}) 無分點，略過")
        if conc < 0:
            skipped += 1
            continue
        records.append({"date": d, "stock_id": stock_id,
                        "concentration": conc, "group": bin_label(conc)})
    print(f"\n有分點資料事件：{len(records)}，無分點略過：{skipped}")

    ev_df = pd.DataFrame(records)
    if ev_df.empty:
        print("（無可用事件）")
        return

    # 集中度描述性統計
    print("\n主力集中度 描述性統計：")
    print(ev_df["concentration"].describe().to_string())

    # 各組樣本量
    print("\n各組事件筆數：")
    for g in BINS:
        print(f"  {g}: {int((ev_df['group'] == g).sum())}")

    # 步驟 4：分組回測
    print("\n" + "=" * 88)
    print("主力集中度分組 → 隔日放空 摘要（停損8%+滑價0.15%）")
    print("=" * 88)
    print(f"{'分組':<14}| {'事件筆數':>6} | {'平均集中度':>8} | {'總損益':>11} | "
          f"{'勝率':>7} | {'Sharpe':>8} | {'最大回撤':>11}")
    print("-" * 92)

    plt.figure(figsize=(12, 6))
    for g in BINS:
        grp = ev_df[ev_df["group"] == g]
        sub = grp[["date", "stock_id"]]
        trades = irb.build_event_trades(sub, data).sort_values("date")
        s = irb.compute_stats(trades)
        wr = f"{s['win_rate']:.2%}" if s["n_trades"] else "-"
        avg_c = f"{grp['concentration'].mean():.2%}" if len(grp) else "-"
        print(f"{g:<14}| {s['n_trades']:>6} | {avg_c:>8} | "
              f"{s['total_pnl']:>11,.0f} | {wr:>7} | {s['sharpe']:>8.4f} | "
              f"{s['max_drawdown']:>11,.0f}")
        if not trades.empty:
            equity = trades.set_index("date")["pnl"].cumsum()
            plt.plot(equity.index, equity.values, label=g)

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("主力集中度分組 → 強勢股隔日放空 累積損益")
    plt.xlabel("date")
    plt.ylabel("累積損益（元）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "broker_concentration.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[plot] 已存 {out}")


if __name__ == "__main__":
    main()
