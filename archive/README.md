# archive/ — 已放棄／實驗性檔案

這個資料夾收納開發過程中的實驗、驗證與除錯腳本。它們**不屬於主程式**，
保留下來只為了記錄探索過程與結論。主程式（`main.py` 等）不依賴這裡的任何檔案。

## 主程式（保留在專案根目錄）

- `main.py`：主流程（全市場漲停股放空、TX region 維度回測）
- `index_region_backtest.py`：指數 region → 強勢股隔日放空（多門檻掃描，目前主力研究）
- `portfolio_backtest.py`：`PortfolioBacktester`（投資組合回測引擎）
- `data_loader.py`：資料載入（FinMind + 本地 CSV 快取）
- `market_scanner.py`：漲停／強勢股事件偵測與當沖可放空過濾
- `strategy.py`：region 分類與訊號建立

## 封存檔案與放棄原因

| 檔案 | 是什麼 | 為什麼放棄 |
|------|--------|-----------|
| `test_pipeline.py` | 150 檔抽樣的流程驗證腳本 | 已被全市場版（`main.py`）取代 |
| `test_intraday.py` | 分鐘線（TaiwanStockKBar）可行性測試 | API 一次只回單日單股、呼叫次數過多，改用日線 `high` 替代 |
| `test_minute.py` | 分鐘線資料量／漲停觸及測試 | 同上：分鐘級全市場成本不可行，改用日線 `high` 替代 |
| `test_event_minute.py` | 事件日分鐘線測試 | 同上：分鐘級路線放棄 |
| `diagnose_region.py` | TX region 除錯腳本 | 用來確認「開高走低=0」是市場結構現象、非 bug，任務已完成 |
| `insample_outsample.py` | 開低走低的 IS/OOS 切分 | 樣本只來自 4–5 天，統計意義不足 |
| `cross_signal.py` | TX region × S&P500 regime 交叉 | 事件過度碎片化，Sharpe 皆為單日群聚的假象 |
| `sp500_signal.py` | S&P500 同步／逆勢濾網 | 後續整合進 `market_scanner.py`（此獨立版封存） |
| `yearly_split_kgzd.py` | 開高走低事件三種定義的逐年績效拆解 | 一次性分析腳本，依賴 `diagnose_region.py`，任務完成後封存 |
| `backtest.py` | `DayTradingBacktester`（單股票當沖回測引擎） | 已被 `portfolio_backtest.py`（投資組合版）取代 |
| `signals.py` | 早期單股訊號產生範本（generate_signals_matrix） | 已被 `strategy.py` / `index_region_backtest.py` 的 region 訊號取代 |
| `test_index_data.py` | 指數資料源（TAIEX/TPEx）可用性與跳空統計探測 | 一次性探測腳本，結論已併入 `index_region_backtest.py` |

> 註：`test_pipeline.py` 與 `test_event_minute.py` 在封存時已不存在於專案中，
> 僅在此處留存說明備查。
