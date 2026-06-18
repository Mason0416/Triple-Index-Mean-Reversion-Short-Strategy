"""當沖回測引擎模組。

提供 DayTradingBacktester 類別，依據訊號序列模擬台股當沖交易，
計算交易成本、滑價、停損，並產出績效指標與資金曲線。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DayTradingBacktester:
    """台股當沖回測引擎。

    模擬「當日進場、當日出場」的當沖交易：以當日開盤價（含滑價）進場，
    若盤中觸及停損價則以停損價出場，否則以收盤價（含滑價）出場。
    完整計入台股手續費、當沖交易稅與滑價成本。

    Attributes:
        fee_rate: 券商手續費率（單邊），預設 0.001425。
        fee_discount: 手續費折扣，預設 0.6（六折）。
        tax_rate: 證交稅率，預設 0.003。
        day_trade_tax_ratio: 當沖交易稅優惠倍率，預設 0.2（二折）。
        stop_loss_pct: 停損百分比，預設 0.02（2%）。
        slippage_pct: 滑價百分比，預設 0.001（0.1%）。
        capital: 每筆交易投入資金，用以計算股數。
    """

    def __init__(
        self,
        fee_rate: float = 0.001425,
        fee_discount: float = 0.6,
        tax_rate: float = 0.003,
        day_trade_tax_ratio: float = 0.2,
        stop_loss_pct: float = 0.02,
        slippage_pct: float = 0.001,
        capital: float = 1_000_000.0,
    ) -> None:
        """初始化回測引擎，所有成本與風控參數皆可調整。

        Args:
            fee_rate: 券商手續費率（單邊）。
            fee_discount: 手續費折扣（如 0.6 表示六折）。
            tax_rate: 證交稅率。
            day_trade_tax_ratio: 當沖交易稅優惠倍率（如 0.2 表示二折）。
            stop_loss_pct: 停損百分比（如 0.02 表示 2%）。
            slippage_pct: 滑價百分比（如 0.001 表示 0.1%）。
            capital: 每筆交易投入資金（用以計算股數）。
        """
        self.fee_rate = float(fee_rate)
        self.fee_discount = float(fee_discount)
        self.tax_rate = float(tax_rate)
        self.day_trade_tax_ratio = float(day_trade_tax_ratio)
        self.stop_loss_pct = float(stop_loss_pct)
        self.slippage_pct = float(slippage_pct)
        self.capital = float(capital)

        # 實際生效成本率
        self.effective_fee = self.fee_rate * self.fee_discount
        self.effective_tax = self.tax_rate * self.day_trade_tax_ratio

        # run() 後填入
        self.trades: pd.DataFrame = pd.DataFrame()

    def _simulate_long(self, row: pd.Series) -> dict:
        """模擬一筆做多當沖交易。

        進場：開盤價往上加滑價買入。
        停損：開盤價往下算停損百分比，再往下加賣出滑價；
        若當日 low 觸及停損價則以停損價出場。
        否則以收盤價往下加滑價出場。

        Args:
            row: 含 open/high/low/close 的單日資料列。

        Returns:
            包含進出場價、出場原因的 dict。
        """
        entry_price = row["open"] * (1 + self.slippage_pct)

        # 停損價：先往下算停損百分比，再加賣出滑價（往下）
        stop_price = row["open"] * (1 - self.stop_loss_pct) * (1 - self.slippage_pct)

        if row["low"] <= stop_price:
            exit_price = stop_price
            exit_reason = "stop_loss"
        else:
            # 收盤出場為賣出，滑價往下
            exit_price = row["close"] * (1 - self.slippage_pct)
            exit_reason = "close"

        return {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
        }

    def _simulate_short(self, row: pd.Series) -> dict:
        """模擬一筆做空當沖交易。

        進場：開盤價往下減滑價賣出。
        停損：開盤價往上算停損百分比，再往上加買入滑價；
        若當日 high 觸及停損價則以停損價出場。
        否則以收盤價往上加滑價（回補）出場。

        Args:
            row: 含 open/high/low/close 的單日資料列。

        Returns:
            包含進出場價、出場原因的 dict。
        """
        entry_price = row["open"] * (1 - self.slippage_pct)

        # 停損價：先往上算停損百分比，再加買入滑價（往上）
        stop_price = row["open"] * (1 + self.stop_loss_pct) * (1 + self.slippage_pct)

        if row["high"] >= stop_price:
            exit_price = stop_price
            exit_reason = "stop_loss"
        else:
            # 收盤回補為買入，滑價往上
            exit_price = row["close"] * (1 + self.slippage_pct)
            exit_reason = "close"

        return {
            "entry_price": entry_price,
            "exit_price": exit_price,
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
            # 做多：進場買入（僅手續費），出場賣出（手續費 + 交易稅）
            entry_cost = entry_fee
            exit_cost = exit_fee + exit_value * self.effective_tax
        else:
            # 做空：進場賣出（手續費 + 交易稅），出場買入（僅手續費）
            entry_cost = entry_fee + entry_value * self.effective_tax
            exit_cost = exit_fee

        return float(entry_cost), float(exit_cost)

    def run(self, df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """執行回測，逐日依訊號模擬當沖交易。

        Args:
            df: 標準 OHLCV DataFrame，index 為 DatetimeIndex。
            signals: 與 df 同 index 的訊號序列，值為 1/-1/0。

        Returns:
            交易紀錄 DataFrame，每列含 date, signal, entry_price,
            exit_price, exit_reason, shares, pnl, entry_cost, exit_cost。
        """
        records = []

        for date, signal in signals.items():
            if signal == 0 or date not in df.index:
                continue

            row = df.loc[date]

            if signal == 1:
                sim = self._simulate_long(row)
            elif signal == -1:
                sim = self._simulate_short(row)
            else:
                continue

            entry_price = float(sim["entry_price"])
            exit_price = float(sim["exit_price"])

            # 以投入資金計算股數（無條件捨去至整股）
            shares = float(int(self.capital / entry_price))
            if shares <= 0:
                continue

            entry_cost, exit_cost = self._compute_costs(
                signal, entry_price, exit_price, shares
            )

            # 毛損益：做多賺價差上漲，做空賺價差下跌
            if signal == 1:
                gross_pnl = (exit_price - entry_price) * shares
            else:
                gross_pnl = (entry_price - exit_price) * shares

            pnl = float(gross_pnl - entry_cost - exit_cost)

            records.append({
                "date": date,
                "signal": int(signal),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": sim["exit_reason"],
                "shares": shares,
                "pnl": pnl,
                "entry_cost": entry_cost,
                "exit_cost": exit_cost,
            })

        trades = pd.DataFrame(records)
        if not trades.empty:
            trades = trades.set_index("date")
            # 數值欄位統一為 float，避免 object 型態
            float_cols = ["entry_price", "exit_price", "shares",
                          "pnl", "entry_cost", "exit_cost"]
            trades[float_cols] = trades[float_cols].astype(float)

        self.trades = trades
        return trades

    def report(self) -> dict:
        """計算並印出績效指標。

        指標包含：總損益、勝率、平均單筆損益、最大單筆虧損、
        年化 Sharpe Ratio（假設 252 交易日）、最大回撤、停損觸發次數。

        Returns:
            包含各項指標的 dict。

        Raises:
            ValueError: 當尚未執行 run() 或無任何交易時。
        """
        if self.trades.empty:
            raise ValueError("尚無交易紀錄，請先呼叫 run() 並確認有產生訊號。")

        pnl = self.trades["pnl"]

        total_pnl = float(pnl.sum())
        win_rate = float((pnl > 0).mean())
        avg_pnl = float(pnl.mean())
        max_loss = float(pnl.min())
        stop_loss_count = int((self.trades["exit_reason"] == "stop_loss").sum())

        # 年化 Sharpe：以每筆損益佔投入資金的報酬率計算
        returns = pnl / self.capital
        std = returns.std()
        if std and std > 0:
            sharpe = float(returns.mean() / std * np.sqrt(252))
        else:
            sharpe = 0.0

        # 最大回撤：以累積資金曲線計算
        equity = pnl.cumsum()
        running_max = equity.cummax()
        drawdown = equity - running_max
        max_drawdown = float(drawdown.min())

        metrics = {
            "總損益": total_pnl,
            "勝率": win_rate,
            "平均單筆損益": avg_pnl,
            "最大單筆虧損": max_loss,
            "Sharpe Ratio": sharpe,
            "最大回撤": max_drawdown,
            "停損觸發次數": stop_loss_count,
        }

        print("=" * 40)
        print("回測績效報告")
        print("=" * 40)
        print(f"交易筆數      : {len(self.trades)}")
        print(f"總損益        : {total_pnl:,.2f}")
        print(f"勝率          : {win_rate:.2%}")
        print(f"平均單筆損益  : {avg_pnl:,.2f}")
        print(f"最大單筆虧損  : {max_loss:,.2f}")
        print(f"Sharpe Ratio  : {sharpe:.4f}")
        print(f"最大回撤      : {max_drawdown:,.2f}")
        print(f"停損觸發次數  : {stop_loss_count}")
        print("=" * 40)

        return metrics

    def plot_equity_curve(self) -> None:
        """繪製資金曲線（累積損益）。

        以 matplotlib 畫出每筆交易後的累積損益變化。

        Raises:
            ValueError: 當尚未執行 run() 或無任何交易時。
        """
        if self.trades.empty:
            raise ValueError("尚無交易紀錄，請先呼叫 run() 並確認有產生訊號。")

        equity = self.trades["pnl"].cumsum()

        plt.figure(figsize=(12, 6))
        plt.plot(equity.index, equity.values, label="Equity Curve")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.title("Day Trading Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
