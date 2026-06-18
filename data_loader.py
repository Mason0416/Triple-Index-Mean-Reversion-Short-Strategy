"""資料載入模組。

負責從 FinMind 抓取台股日線資料，或從本地 CSV 讀取，
並統一輸出標準 OHLCV DataFrame。
"""

import os
import signal
import time

import numpy as np
import pandas as pd
from FinMind.data import DataLoader


# 單檔 FinMind API 抓取逾時秒數（僅作用於 API，不含 CSV 讀取）
_API_TIMEOUT = 30


class _FetchTimeout(Exception):
    """單檔 API 抓取逾時。"""


def _raise_fetch_timeout(signum, frame):
    """SIGALRM 處理器：拋出 _FetchTimeout 以中斷逾時的 API 抓取。"""
    raise _FetchTimeout()


def _fetch_with_timeout(func, *args, timeout: int = _API_TIMEOUT):
    """以 signal.alarm 對 func 設定逾時保護後執行（僅用於 API 抓取）。

    逾時會拋出 _FetchTimeout；無論成功或逾時，finally 都會以
    signal.alarm(0) 取消鬧鐘並還原原本的 handler，避免影響後續股票。

    Args:
        func: 要執行的函數（通常為 get_ohlcv）。
        *args: 傳給 func 的位置參數。
        timeout: 逾時秒數，預設 _API_TIMEOUT。

    Returns:
        func 的回傳值。

    Raises:
        _FetchTimeout: 當執行超過 timeout 秒。
    """
    old_handler = signal.signal(signal.SIGALRM, _raise_fetch_timeout)
    signal.alarm(timeout)
    try:
        return func(*args)
    finally:
        signal.alarm(0)  # 務必取消鬧鐘，避免影響後續股票
        signal.signal(signal.SIGALRM, old_handler)


# FinMind 原始欄位 -> 標準欄位 對應
_COLUMN_MAP = {
    "open": "open",
    "max": "high",
    "min": "low",
    "close": "close",
    "Trading_Volume": "volume",
}

# 標準 OHLCV 欄位（需確保為 float）
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _to_standard_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """將任意來源的 DataFrame 轉為標準 OHLCV 格式。

    確保 index 為 DatetimeIndex，欄位為 open/high/low/close/volume，
    且數值型態統一為 float，避免 object 型態造成計算錯誤。

    Args:
        df: 含有標準欄位（或可重新命名為標準欄位）的 DataFrame。

    Returns:
        標準 OHLCV DataFrame，index 為 DatetimeIndex，依日期排序。
    """
    df = df.copy()

    # index 統一為 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 數值型態統一為 float
    for col in _OHLCV_COLUMNS:
        df[col] = df[col].astype(float)

    df = df[_OHLCV_COLUMNS].sort_index()
    df.index.name = "date"
    return df


def get_ohlcv(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """透過 FinMind 抓取台股日線資料並轉為標準 OHLCV DataFrame。

    Args:
        stock_id: 股票代號，例如 "2330"。
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 結束日期，格式 "YYYY-MM-DD"。

    Returns:
        標準 OHLCV DataFrame。

    Raises:
        ValueError: 當 FinMind 回傳空資料時。
    """
    loader = DataLoader()

    # 若有 token 則登入，可提升 API 流量上限
    token = os.getenv("FINMIND_TOKEN")
    if token:
        loader.login_by_token(api_token=token)

    raw = loader.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"FinMind 回傳空資料：stock_id={stock_id}, "
            f"{start_date} ~ {end_date}"
        )

    raw = raw.rename(columns=_COLUMN_MAP)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date")

    return _to_standard_ohlcv(raw)


def load_data(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """載入台股日線資料，優先讀取本地快取 CSV，否則從 FinMind 抓取。

    CSV 路徑為 data/{stock_id}.csv。
    若快取完整涵蓋指定期間則直接讀取；若範圍不足，會從 FinMind
    補抓指定期間並合併快取。回傳資料一律裁切至指定日期範圍。

    Args:
        stock_id: 股票代號，例如 "2330"。
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 結束日期，格式 "YYYY-MM-DD"。

    Returns:
        標準 OHLCV DataFrame，index 為 DatetimeIndex，
        欄位為 open/high/low/close/volume（皆為 float）。
    """
    csv_path = os.path.join("data", f"{stock_id}.csv")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    business_days = pd.bdate_range(start, end)
    expected_start = business_days.min() if not business_days.empty else start
    expected_end = business_days.max() if not business_days.empty else end

    if os.path.exists(csv_path):
        cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        cached = _to_standard_ohlcv(cached)

        cache_incomplete = (
            cached.empty
            or expected_start < cached.index.min()
            or expected_end > cached.index.max()
        )
        if not cache_incomplete:
            return cached.loc[start:end]

        # API 抓取套用逾時保護（逾時會拋 _FetchTimeout，下方 to_csv 不會執行）
        fresh = _fetch_with_timeout(get_ohlcv, stock_id, start_date, end_date)
        df = pd.concat([cached, fresh])
        df = df[~df.index.duplicated(keep="last")]
        df = _to_standard_ohlcv(df)
        df.to_csv(csv_path)
        return df.loc[start:end]

    # 檔案不存在：從 FinMind 抓取（套用逾時保護；逾時則不會寫入 CSV）
    df = _fetch_with_timeout(get_ohlcv, stock_id, start_date, end_date)

    os.makedirs("data", exist_ok=True)
    df.to_csv(csv_path)

    return df.loc[start:end]


def load_multiple(stock_ids: list, start_date: str,
                  end_date: str) -> dict:
    """批次載入多檔股票，回傳以股票代號為 key 的 DataFrame 字典。

    逐檔呼叫 load_data，沿用其本地 CSV 快取與 FinMind 抓取邏輯。
    單檔載入失敗時印出警告並略過，不中斷其餘股票。

    Args:
        stock_ids: 股票代號清單，例如 ["2330", "2317"]。
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 結束日期，格式 "YYYY-MM-DD"。

    Returns:
        dict[str, pd.DataFrame]，key 為股票代號，
        value 為標準 OHLCV DataFrame。
    """
    data = {}
    for stock_id in stock_ids:
        try:
            data[stock_id] = load_data(stock_id, start_date, end_date)
        except _FetchTimeout:
            print(f"[警告] 載入 {stock_id} 逾時({_API_TIMEOUT}秒)，已略過")
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 載入 {stock_id} 失敗，已略過：{exc}")
    return data


def _get_loader() -> DataLoader:
    """建立並（若有 token）登入 FinMind DataLoader。

    Returns:
        已（盡可能）登入的 DataLoader 實例。
    """
    loader = DataLoader()
    token = os.getenv("FINMIND_TOKEN")
    if token:
        loader.login_by_token(api_token=token)
    return loader


def _date_chunks(start_date: str, end_date: str, days: int = 90):
    """將日期區間切成數段，供分批抓取 API 使用。

    Args:
        start_date: 起始日期 "YYYY-MM-DD"。
        end_date: 結束日期 "YYYY-MM-DD"。
        days: 每段的天數長度。

    Yields:
        (chunk_start, chunk_end) 字串 tuple，皆為 "YYYY-MM-DD"。
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=days - 1), end)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + pd.Timedelta(days=1)


# 非個股的 industry_category（明確排除：指數、ETN、受益證券）。
# 依步驟 0 印出的 taiwan_stock_info 實際類別判斷，ETF 與一般股票保留。
_NON_EQUITY_CATEGORIES = {
    "Index", "大盤", "所有證券",
    "ETN", "指數投資證券(ETN)",
    "受益證券",
}


def get_all_stock_ids() -> list:
    """取得全市場可交易股票清單（排除明顯非個股類別）。

    用 taiwan_stock_info() 取得全市場清單，印出欄位與前 10 筆供確認，
    排除指數、ETN、受益證券等非個股類別，以及興櫃（emerging，無法當沖），
    保留 ETF 與一般上市櫃股票。

    Returns:
        去重後的 stock_id 字串清單。
    """
    loader = _get_loader()
    info = loader.taiwan_stock_info()

    print(f"[get_all_stock_ids] taiwan_stock_info 共 {len(info)} 筆")
    print(f"[get_all_stock_ids] columns: {list(info.columns)}")
    print(info.head(10).to_string())

    df = info.copy()
    # 排除非個股類別
    df = df[~df["industry_category"].isin(_NON_EQUITY_CATEGORIES)]
    # 排除興櫃（無逐筆撮合、無法當沖）
    if "type" in df.columns:
        df = df[df["type"] != "emerging"]

    stock_ids = sorted(df["stock_id"].astype(str).unique().tolist())
    print(f"[get_all_stock_ids] 過濾後股票數：{len(stock_ids)}")
    return stock_ids


def get_tx_data(start_date: str, end_date: str) -> pd.DataFrame:
    """抓取台指期（TX）近月合約日線，輸出標準 OHLCV。

    taiwan_futures_daily 每日含多個月份合約與價差組合（如 202101/202102），
    且分日盤（position）與盤後（after_market）。本函數只取日盤、排除價差組合，
    並對每個交易日保留「成交量最大」的合約（即近月）。

    Args:
        start_date: 起始日期 "YYYY-MM-DD"。
        end_date: 結束日期 "YYYY-MM-DD"。

    Returns:
        標準 OHLCV DataFrame，index 為 DatetimeIndex（date），
        欄位 open/high/low/close/volume 皆為 float。
    """
    loader = _get_loader()
    raw = loader.taiwan_futures_daily(
        futures_id="TX", start_date=start_date, end_date=end_date
    )
    print(f"[get_tx_data] 原始筆數：{len(raw)}，columns: {list(raw.columns)}")

    if raw is None or raw.empty:
        raise ValueError("FinMind 回傳空的 TX 期貨資料。")

    df = raw.copy()
    # 只取日盤
    if "trading_session" in df.columns:
        df = df[df["trading_session"] == "position"]
    # 排除價差組合（contract_date 含 '/'）與零量
    df = df[~df["contract_date"].astype(str).str.contains("/")]
    df = df[df["volume"] > 0]

    # 每個交易日保留成交量最大的合約（近月）
    idx = df.groupby("date")["volume"].idxmax()
    near = df.loc[idx].copy()

    near = near.rename(columns={"max": "high", "min": "low"})
    near["date"] = pd.to_datetime(near["date"])
    near = near.set_index("date").sort_index()

    out = near[["open", "high", "low", "close", "volume"]].astype(float)
    out.index.name = "date"
    print(f"[get_tx_data] 近月合約交易日數：{len(out)}")
    return out


def get_price_limit_data(start_date: str, end_date: str) -> pd.DataFrame:
    """抓取全市場每日漲跌停價（TaiwanStockPriceLimit），分批合併。

    不指定 data_id，依日期區間分段抓取後合併去重。
    回傳原始欄位（date, stock_id, reference_price, limit_up, limit_down）。

    Args:
        start_date: 起始日期 "YYYY-MM-DD"。
        end_date: 結束日期 "YYYY-MM-DD"。

    Returns:
        原始 DataFrame；若無資料則回傳空 DataFrame。
    """
    loader = _get_loader()
    frames = []
    for chunk_start, chunk_end in _date_chunks(start_date, end_date, days=90):
        d = loader.get_data(
            dataset="TaiwanStockPriceLimit",
            start_date=chunk_start,
            end_date=chunk_end,
        )
        if d is not None and not d.empty:
            frames.append(d)
            print(f"[get_price_limit_data] {chunk_start}~{chunk_end}: {len(d)} 筆")

    if not frames:
        print("[get_price_limit_data] 無資料")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    print(f"[get_price_limit_data] 合併後共 {len(out)} 筆")
    return out


def get_day_trading_suspension(start_date: str, end_date: str) -> pd.DataFrame:
    """抓取暫停先賣後買當沖名單（TaiwanStockDayTradingSuspension）。

    不指定 data_id，依日期區間抓取全市場資料。
    回傳原始欄位（stock_id, date, end_date, reason）；
    其中 date~end_date 為該股票暫停先賣後買當沖的期間。

    Args:
        start_date: 起始日期 "YYYY-MM-DD"。
        end_date: 結束日期 "YYYY-MM-DD"。

    Returns:
        原始 DataFrame；若無資料則回傳空 DataFrame。
    """
    loader = _get_loader()
    d = loader.get_data(
        dataset="TaiwanStockDayTradingSuspension",
        start_date=start_date,
        end_date=end_date,
    )
    if d is None or d.empty:
        print("[get_day_trading_suspension] 無資料")
        return pd.DataFrame()
    print(f"[get_day_trading_suspension] 共 {len(d)} 筆，columns: {list(d.columns)}")
    return d


def get_market_value(stock_id: str, start_date: str,
                     end_date: str) -> pd.DataFrame:
    """抓取單一股票每日市值（TaiwanStockMarketValue），帶本地 CSV 快取。

    快取路徑 data/mv/{stock_id}.csv；存在則直接讀，否則抓取後存檔。

    Args:
        stock_id: 股票代號，例如 "2330"。
        start_date: 起始日期 "YYYY-MM-DD"。
        end_date: 結束日期 "YYYY-MM-DD"。

    Returns:
        DataFrame，index 為 date（DatetimeIndex），欄位 [market_value]（float）。

    Raises:
        ValueError: 當 FinMind 回傳空資料時。
    """
    mv_dir = os.path.join("data", "mv")
    csv_path = os.path.join(mv_dir, f"{stock_id}.csv")

    if os.path.exists(csv_path):
        cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        cached["market_value"] = cached["market_value"].astype(float)
        return cached[["market_value"]].sort_index()

    loader = _get_loader()
    raw = loader.get_data(
        dataset="TaiwanStockMarketValue", data_id=stock_id,
        start_date=start_date, end_date=end_date,
    )
    if raw is None or raw.empty:
        raise ValueError(
            f"TaiwanStockMarketValue 回傳空資料：stock_id={stock_id}, "
            f"{start_date} ~ {end_date}"
        )

    df = raw[["date", "market_value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["market_value"] = df["market_value"].astype(float)
    df.index.name = "date"

    os.makedirs(mv_dir, exist_ok=True)
    df[["market_value"]].to_csv(csv_path)
    return df[["market_value"]]


def load_market_value_multiple(stock_ids: list, start_date: str,
                               end_date: str) -> dict:
    """批次載入多檔股票每日市值，回傳以股票代號為 key 的字典。

    逐檔呼叫 get_market_value（已快取直接讀，否則抓 API）。實際打 API 後
    sleep 0.3 秒以避免速率限制；單檔失敗印警告並略過。

    Args:
        stock_ids: 股票代號清單。
        start_date: 起始日期 "YYYY-MM-DD"。
        end_date: 結束日期 "YYYY-MM-DD"。

    Returns:
        dict[str, pd.DataFrame]，value 為市值 DataFrame（index=date）。
    """
    data = {}
    total = len(stock_ids)
    for i, stock_id in enumerate(stock_ids, start=1):
        csv_path = os.path.join("data", "mv", f"{stock_id}.csv")
        had_cache = os.path.exists(csv_path)
        try:
            data[stock_id] = get_market_value(stock_id, start_date, end_date)
            print(f"已載入 {stock_id} ({i}/{total})")
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 載入 {stock_id} 市值失敗，已略過：{exc}")
        if not had_cache:
            time.sleep(0.3)  # 僅在實際打 API 後 sleep，避免速率限制
    return data


def calc_turnover(ohlcv_df: pd.DataFrame, mv_df: pd.DataFrame) -> pd.Series:
    """計算每日周轉率 = 成交量(股) / 流通股數，流通股數 = 市值 / 收盤價。

    成交量取 ohlcv_df 的 volume 欄（即 FinMind 的 Trading_Volume，單位為股；
    若無 volume 欄則退而取 Trading_Volume 欄）。以日期 index 對齊兩份資料。
    某日 market_value 或 close 為 0 時，該日回傳 NaN。

    Args:
        ohlcv_df: 單一股票的 OHLCV DataFrame（index=date，含 close 與成交量）。
        mv_df: get_market_value 的輸出（index=date，含 market_value）。

    Returns:
        pd.Series，index=date，值為周轉率（float），無法計算者為 NaN。
    """
    vol_col = "volume" if "volume" in ohlcv_df.columns else "Trading_Volume"
    base = pd.DataFrame({
        "volume": ohlcv_df[vol_col].astype(float),
        "close": ohlcv_df["close"].astype(float),
    })
    joined = base.join(mv_df["market_value"].astype(float), how="inner")

    close = joined["close"]
    mv = joined["market_value"]
    shares = mv / close.where(close != 0)
    turnover = joined["volume"] / shares
    # market_value 或 close 為 0 → NaN
    turnover = turnover.where((mv != 0) & (close != 0), other=np.nan)
    turnover.name = "turnover"
    return turnover
