"""投資組合當沖回測引擎模組。

提供 PortfolioBacktester 類別，支援多檔股票同時回測：
輸入為多股票 OHLCV 字典與訊號矩陣，逐日逐股票模擬當沖交易，
彙總每日損益、資金曲線、曝險與停損狀況，並產出組合層級與
個股層級的績效指標。

成本、滑價與停損邏輯沿用單股票版本（backtest.py），逐股票獨立計算。
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PortfolioBacktester:
    """台股投資組合當沖回測引擎。

    對訊號矩陣中每一天、每一支非 0 訊號的股票，
    以當日開盤價（含滑價）進場，盤中觸停損則以停損價出場，
    否則以收盤價（含滑價）出場，逐股票獨立計算損益與成本。

    Attributes:
        fee_rate: 券商手續費率（單邊），預設 0.001425。
        fee_discount: 手續費折扣，預設 0.6（六折）。
        tax_rate: 證交稅率，預設 0.003。
        day_trade_tax_ratio: 當沖交易稅優惠倍率，預設 0.2（二折）。
        stop_loss_pct: 停損百分比，預設 0.02（2%）。
        slippage_pct: 滑價百分比，預設 0.001（0.1%）。
        shares: 每筆交易固定股數，預設 1000。
    """

    def __init__(
        self,
        fee_rate: float = 0.001425,
        fee_discount: float = 0.6,
        tax_rate: float = 0.003,
        day_trade_tax_ratio: float = 0.2,
        stop_loss_pct: float = 0.02,
        slippage_pct: float = 0.001,
        shares: float = 1000.0,
        commission_discount: float = None,
    ) -> None:
        """初始化投資組合回測引擎，所有成本與風控參數皆可調整。

        Args:
            fee_rate: 券商手續費率（單邊）。
            fee_discount: 手續費折扣（如 0.6 表示六折）。
            tax_rate: 證交稅率。
            day_trade_tax_ratio: 當沖交易稅優惠倍率（如 0.2 表示二折）。
            stop_loss_pct: 停損百分比（如 0.02 表示 2%）。
            slippage_pct: 滑價百分比（如 0.001 表示 0.1%）。
            shares: 每筆交易固定股數。
            commission_discount: fee_discount 的別名（供 optimize 的
                param_grid 使用）；若提供則覆蓋 fee_discount。
        """
        # commission_discount 為 fee_discount 的別名，方便參數網格命名
        if commission_discount is not None:
            fee_discount = commission_discount

        self.fee_rate = float(fee_rate)
        self.fee_discount = float(fee_discount)
        self.tax_rate = float(tax_rate)
        self.day_trade_tax_ratio = float(day_trade_tax_ratio)
        self.stop_loss_pct = float(stop_loss_pct)
        self.slippage_pct = float(slippage_pct)
        self.shares = float(shares)

        # 實際生效成本率
        self.effective_fee = self.fee_rate * self.fee_discount
        self.effective_tax = self.tax_rate * self.day_trade_tax_ratio

        # run() 後填入
        self.trades: pd.DataFrame = pd.DataFrame()

    def _simulate_long(self, row: pd.Series) -> dict:
        """模擬一筆做多當沖交易（單股單日）。

        進場：開盤價往上加滑價買入。
        停損：開盤價往下算停損百分比，再往下加賣出滑價；
        當日 low 觸及停損價則以停損價出場，否則以收盤價往下加滑價出場。

        Args:
            row: 含 open/high/low/close 的單日資料列。

        Returns:
            包含 entry_price, exit_price, exit_reason 的 dict。
        """
        entry_price = row["open"] * (1 + self.slippage_pct)
        stop_price = row["open"] * (1 - self.stop_loss_pct) * (1 - self.slippage_pct)

        if row["low"] <= stop_price:
            exit_price = stop_price
            exit_reason = "stop_loss"
        else:
            exit_price = row["close"] * (1 - self.slippage_pct)
            exit_reason = "close"

        return {
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
        }

    def _simulate_short(self, row: pd.Series) -> dict:
        """模擬一筆做空當沖交易（單股單日）。

        進場：開盤價往下減滑價賣出。
        停損：開盤價往上算停損百分比，再往上加買入滑價；
        當日 high 觸及停損價則以停損價出場，否則以收盤價往上加滑價回補。

        Args:
            row: 含 open/high/low/close 的單日資料列。

        Returns:
            包含 entry_price, exit_price, exit_reason 的 dict。
        """
        entry_price = row["open"] * (1 - self.slippage_pct)
        stop_price = row["open"] * (1 + self.stop_loss_pct) * (1 + self.slippage_pct)

        if row["high"] >= stop_price:
            exit_price = stop_price
            exit_reason = "stop_loss"
        else:
            exit_price = row["close"] * (1 + self.slippage_pct)
            exit_reason = "close"

        return {
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
        }

    def _compute_costs(self, signal: int, entry_price: float,
                       exit_price: float, shares: float) -> tuple:
        """計算單筆交易的進場與出場成本。

        手續費買賣雙邊皆收；交易稅僅賣出收。
        做多時賣出在出場，做空時賣出在進場。

        Args:
            signal: 1（做多）或 -1（做空）。
            entry_price: 進場價。
            exit_price: 出場價。
            shares: 交易股數。

        Returns:
            (entry_cost, exit_cost) 兩個 float。
        """
        entry_value = entry_price * shares
        exit_value = exit_price * shares

        entry_fee = entry_value * self.effective_fee
        exit_fee = exit_value * self.effective_fee

        if signal == 1:
            entry_cost = entry_fee
            exit_cost = exit_fee + exit_value * self.effective_tax
        else:
            entry_cost = entry_fee + entry_value * self.effective_tax
            exit_cost = exit_fee

        return float(entry_cost), float(exit_cost)

    def run(self, data: dict, signals: pd.DataFrame) -> pd.DataFrame:
        """執行投資組合回測，逐日逐股票模擬當沖交易。

        逐日迴圈 signals 的每一列，對每支非 0 訊號的股票，
        若該股票當日有資料（OHLC 皆非缺值）才進行交易。

        Args:
            data: dict[str, pd.DataFrame]，key 為股票代號，
                value 為標準 OHLCV DataFrame。
            signals: 訊號矩陣，index 為日期、columns 為股票代號，
                值為 1（買入）、-1（賣出）、0（不動作）。

        Returns:
            交易明細 DataFrame，欄位：date, stock_id, signal,
            entry_price, exit_price, exit_reason, shares,
            pnl, entry_cost, exit_cost。
        """
        records = []

        for date, row_signals in signals.iterrows():
            for stock_id, signal in row_signals.items():
                if signal == 0 or pd.isna(signal):
                    continue
                signal = int(signal)
                if signal not in (1, -1):
                    continue

                df = data.get(stock_id)
                if df is None or date not in df.index:
                    # 該股票當日無資料，視為不交易
                    continue

                bar = df.loc[date]
                # 缺值（OHLC 任一為 NaN）視為不交易
                if bar[["open", "high", "low", "close"]].isna().any():
                    continue

                if signal == 1:
                    sim = self._simulate_long(bar)
                else:
                    sim = self._simulate_short(bar)

                entry_price = sim["entry_price"]
                exit_price = sim["exit_price"]
                shares = self.shares

                entry_cost, exit_cost = self._compute_costs(
                    signal, entry_price, exit_price, shares
                )

                if signal == 1:
                    gross_pnl = (exit_price - entry_price) * shares
                else:
                    gross_pnl = (entry_price - exit_price) * shares

                pnl = float(gross_pnl - entry_cost - exit_cost)

                records.append({
                    "date": date,
                    "stock_id": stock_id,
                    "signal": signal,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_reason": sim["exit_reason"],
                    "shares": float(shares),
                    "pnl": pnl,
                    "entry_cost": entry_cost,
                    "exit_cost": exit_cost,
                })

        trades = pd.DataFrame(records)
        if not trades.empty:
            float_cols = ["entry_price", "exit_price", "shares",
                          "pnl", "entry_cost", "exit_cost"]
            trades[float_cols] = trades[float_cols].astype(float)
            trades = trades.sort_values("date").reset_index(drop=True)

        self.trades = trades
        return trades

    def daily_summary(self) -> pd.DataFrame:
        """將交易明細依日期彙總為每日組合指標。

        Returns:
            DataFrame，index 為日期，欄位：
            daily_pnl（當日所有標的損益加總）、
            cumulative_pnl（累積損益，即資金曲線）、
            n_positions（當日持倉數量）、
            n_stop_loss（當日觸發停損數量）。

        Raises:
            ValueError: 當尚未執行 run() 或無任何交易時。
        """
        if self.trades.empty:
            raise ValueError("尚無交易紀錄，請先呼叫 run() 並確認有產生訊號。")

        grouped = self.trades.groupby("date")
        summary = pd.DataFrame({
            "daily_pnl": grouped["pnl"].sum(),
            "n_positions": grouped.size(),
            "n_stop_loss": grouped["exit_reason"].apply(
                lambda s: int((s == "stop_loss").sum())
            ),
        })
        summary = summary.sort_index()
        summary["cumulative_pnl"] = summary["daily_pnl"].cumsum()

        # 數值型態統一為 float
        summary["daily_pnl"] = summary["daily_pnl"].astype(float)
        summary["cumulative_pnl"] = summary["cumulative_pnl"].astype(float)
        summary["n_positions"] = summary["n_positions"].astype(int)
        summary["n_stop_loss"] = summary["n_stop_loss"].astype(int)

        # 欄位順序依規格
        return summary[["daily_pnl", "cumulative_pnl",
                        "n_positions", "n_stop_loss"]]

    def _per_stock_performance(self) -> pd.DataFrame:
        """計算個股層級績效表。

        Returns:
            DataFrame，index 為 stock_id，欄位：
            n_trades（交易筆數）、total_pnl（總損益）、win_rate（勝率）。
        """
        grouped = self.trades.groupby("stock_id")
        perf = pd.DataFrame({
            "n_trades": grouped.size(),
            "total_pnl": grouped["pnl"].sum().astype(float),
            "win_rate": grouped["pnl"].apply(lambda s: float((s > 0).mean())),
        })
        return perf.sort_values("total_pnl", ascending=False)

    def report(self, verbose: bool = True) -> dict:
        """計算組合層級與個股層級績效指標（可選擇是否印出）。

        指標包含：總損益、總交易筆數、整體勝率、
        每日報酬率年化 Sharpe Ratio（基於 daily_pnl）、
        最大回撤（基於 cumulative_pnl）、平均每日持倉數、
        總停損觸發次數，以及個股層級績效表。

        Args:
            verbose: 為 True（預設）時印出完整文字報告；
                為 False 時不印出任何文字，僅回傳 dict
                （供參數最佳化等批次呼叫使用）。

        Returns:
            包含各項指標的 dict（含 key "個股績效" 為個股績效表）。

        Raises:
            ValueError: 當尚未執行 run() 或無任何交易時。
        """
        if self.trades.empty:
            raise ValueError("尚無交易紀錄，請先呼叫 run() 並確認有產生訊號。")

        summary = self.daily_summary()
        pnl = self.trades["pnl"]

        total_pnl = float(pnl.sum())
        n_trades = int(len(self.trades))
        win_rate = float((pnl > 0).mean())
        total_stop_loss = int((self.trades["exit_reason"] == "stop_loss").sum())
        avg_positions = float(summary["n_positions"].mean())

        # 每日報酬率 Sharpe：以每日損益相對「當日投入名目本金」計算報酬率。
        # 名目本金 = 進場價 × 股數 之每日總和（做多做空皆計絕對曝險）。
        daily_notional = (
            self.trades.assign(
                notional=self.trades["entry_price"] * self.trades["shares"]
            )
            .groupby("date")["notional"].sum()
            .reindex(summary.index)
        )
        daily_ret = summary["daily_pnl"] / daily_notional
        std = daily_ret.std()
        if std and std > 0:
            sharpe = float(daily_ret.mean() / std * np.sqrt(252))
        else:
            sharpe = 0.0

        # 最大回撤：基於 cumulative_pnl
        equity = summary["cumulative_pnl"]
        running_max = equity.cummax()
        max_drawdown = float((equity - running_max).min())

        per_stock = self._per_stock_performance()

        metrics = {
            "總損益": total_pnl,
            "總交易筆數": n_trades,
            "整體勝率": win_rate,
            "Sharpe Ratio": sharpe,
            "最大回撤": max_drawdown,
            "平均每日持倉數": avg_positions,
            "總停損觸發次數": total_stop_loss,
            "個股績效": per_stock,
        }

        if verbose:
            print("=" * 48)
            print("投資組合回測績效報告")
            print("=" * 48)
            print(f"總損益          : {total_pnl:,.2f}")
            print(f"總交易筆數      : {n_trades}")
            print(f"整體勝率        : {win_rate:.2%}")
            print(f"Sharpe Ratio    : {sharpe:.4f}")
            print(f"最大回撤        : {max_drawdown:,.2f}")
            print(f"平均每日持倉數  : {avg_positions:.2f}")
            print(f"總停損觸發次數  : {total_stop_loss}")
            print("-" * 48)
            print("個股層級績效：")
            per_stock_display = per_stock.copy()
            per_stock_display["total_pnl"] = per_stock_display["total_pnl"].map(
                lambda x: f"{x:,.2f}"
            )
            per_stock_display["win_rate"] = per_stock_display["win_rate"].map(
                lambda x: f"{x:.2%}"
            )
            print(per_stock_display.to_string())
            print("=" * 48)

        return metrics

    def plot_equity_curve(self) -> None:
        """繪製資金曲線與每日曝險（持倉數）兩張圖。

        上圖為 cumulative_pnl 隨時間變化的資金曲線；
        下圖為 n_positions 隨時間變化，顯示曝險水準。

        Raises:
            ValueError: 當尚未執行 run() 或無任何交易時。
        """
        if self.trades.empty:
            raise ValueError("尚無交易紀錄，請先呼叫 run() 並確認有產生訊號。")

        summary = self.daily_summary()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        ax1.plot(summary.index, summary["cumulative_pnl"],
                 label="Cumulative PnL", color="tab:blue")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax1.set_title("Portfolio Equity Curve")
        ax1.set_ylabel("Cumulative PnL")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(summary.index, summary["n_positions"],
                color="tab:orange", width=1.0)
        ax2.set_title("Daily Exposure (Number of Positions)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("n_positions")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def optimize(
    data: dict,
    signals: pd.DataFrame,
    param_grid: dict,
) -> pd.DataFrame:
    """對參數網格做窮舉回測（grid search），找出最佳參數組合。

    對 param_grid 中所有參數做笛卡爾積，逐一以該組合初始化
    PortfolioBacktester 並回測，蒐集每組參數對應的績效指標，
    最後依 Sharpe Ratio 由高到低排序回傳。

    param_grid 的 key 必須是 PortfolioBacktester.__init__ 接受的參數名
    （含別名 commission_discount）；value 為該參數要嘗試的值清單。

    Args:
        data: dict[str, pd.DataFrame]，多股票 OHLCV 字典。
        signals: 訊號矩陣（所有參數組合共用同一份訊號）。
        param_grid: dict，key 為參數名、value 為候選值清單。例如
            {"stop_loss_pct": [0.01, 0.02], "slippage_pct": [0.0005, 0.001]}。

    Returns:
        pd.DataFrame，每一列為一組參數值加上對應績效指標
        （total_pnl, sharpe, max_drawdown, win_rate），依 sharpe 由高到低排序。
        無交易的組合其績效以 NaN 表示。
    """
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    combos = list(itertools.product(*value_lists))
    total = len(combos)

    records = []
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        print(f"正在測試第 {i}/{total} 組參數... {params}")

        backtester = PortfolioBacktester(**params)
        backtester.run(data, signals)

        record = dict(params)
        if backtester.trades.empty:
            # 此組合未產生任何交易，績效以 NaN 表示
            record.update({
                "total_pnl": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "win_rate": np.nan,
            })
        else:
            backtester.daily_summary()
            metrics = backtester.report(verbose=False)
            record.update({
                "total_pnl": metrics["總損益"],
                "sharpe": metrics["Sharpe Ratio"],
                "max_drawdown": metrics["最大回撤"],
                "win_rate": metrics["整體勝率"],
            })

        records.append(record)

    results = pd.DataFrame(records)
    results = results.sort_values(
        "sharpe", ascending=False, na_position="last"
    ).reset_index(drop=True)
    return results
