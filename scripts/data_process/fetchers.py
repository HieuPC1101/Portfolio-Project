"""Data fetching utilities for CSV files and market APIs."""

import warnings
# Suppress specific warning about pkg_resources
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import os
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import pytz  # Recommended for timezone handling
from vnstock import Vnstock

# Thiết lập múi giờ Việt Nam
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

def fetch_data_from_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file containing company metadata."""
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        print(f"File {file_path} không tồn tại. Vui lòng kiểm tra lại.")
        return pd.DataFrame()
    except Exception as exc:
        print(f"Lỗi khi đọc dữ liệu từ file CSV: {exc}")
        return pd.DataFrame()

def create_vnstock_instance():
    """Return a default Vnstock instance."""
    return Vnstock().stock(symbol='VNINDEX', source='MSN')

def _normalize_symbols(symbols: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Return uppercase symbols without duplicates and list of discarded ones."""
    seen = set()
    unique: List[str] = []
    duplicates: List[str] = []
    for raw in symbols:
        ticker = (raw or "").strip().upper()
        if not ticker:
            continue
        if ticker in seen:
            duplicates.append(ticker)
            continue
        seen.add(ticker)
        unique.append(ticker)
    return unique, duplicates

@lru_cache(maxsize=128)
def _fetch_single_stock_cached(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch a single ticker history and cache the response.
    Returns a DataFrame with DatetimeIndex.
    """
    # Try MSN first, fallback to VCI if needed
    sources = ['MSN', 'VCI']
    last_error = None
    
    for source in sources:
        try:
            stock = Vnstock().stock(symbol=ticker, source=source)
            stock_data = stock.quote.history(start=str(start_date), end=str(end_date))
            
            if stock_data is not None and not stock_data.empty:
                # Giữ lại time và close
                df = stock_data[['time', 'close']].copy()
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
                return df
        except Exception as exc:
            last_error = exc
            continue
    
    # If all sources failed
    error_msg = str(last_error) if last_error else "Không có dữ liệu"
    if "404" in error_msg or "Not Found" in error_msg:
        error_msg = "Mã không tồn tại hoặc đã hủy niêm yết"
    elif "RetryError" in error_msg:
        error_msg = "Không thể kết nối đến server"
    elif "ValueError" in error_msg:
        error_msg = "Dữ liệu không hợp lệ"
    raise RuntimeError(error_msg) from last_error

def fetch_stock_data2(symbols: List[str], start_date: str, end_date: str,
                      verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Download historical prices for a list of tickers using parallel processing.
    Optimized using pd.concat instead of iterative merge.
    """
    unique_symbols, duplicates = _normalize_symbols(symbols)
    skipped_tickers: List[str] = []
    
    start_str = str(start_date)
    end_str = str(end_date)

    if duplicates and verbose:
        print(f"Bỏ qua mã trùng lặp: {', '.join(sorted(set(duplicates)))}")

    if not unique_symbols:
        if verbose:
            print("Danh sách cổ phiếu rỗng.")
        return pd.DataFrame(), skipped_tickers

    if verbose:
        print(f"Đang tải dữ liệu song song cho {len(unique_symbols)} cổ phiếu...")

    # Hàm worker nội bộ
    def fetch_worker(ticker: str):
        try:
            # Lấy từ cache (trả về tham chiếu)
            cached_df = _fetch_single_stock_cached(ticker, start_str, end_str)
            # BẮT BUỘC .copy() để không làm hỏng cache khi sửa tên cột
            df_copy = cached_df.copy()
            df_copy.columns = [ticker] # Rename 'close' -> 'TICKER'
            return ticker, df_copy, None
        except Exception as exc:
            return ticker, None, str(exc)

    results = []
    max_workers = min(8, max(1, len(unique_symbols)))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_worker, ticker): ticker 
            for ticker in unique_symbols
        }

        for i, future in enumerate(as_completed(future_to_ticker), 1):
            ticker = future_to_ticker[future]
            tk_name, df_res, err = future.result()
            
            if df_res is not None and not df_res.empty:
                results.append(df_res)
                if verbose:
                    print(f"\r[{i}/{len(unique_symbols)}] {ticker}: ✓ Thành công", end="")
            else:
                skipped_tickers.append(tk_name)
                if verbose:
                    print(f"\r[{i}/{len(unique_symbols)}] {ticker}: ✗ Bỏ qua ({err})", end="")

    print("") # Xuống dòng sau khi chạy xong loop

    if not results:
        if verbose:
            print("✗ Không thể tải dữ liệu cho bất kỳ cổ phiếu nào.")
        return pd.DataFrame(), skipped_tickers

    # OPTIMIZATION: Dùng concat axis=1 thay vì merge loop
    if verbose:
        print("Đang tổng hợp dữ liệu...")
    
    try:
        # Concat sẽ tự động align theo Index (Time)
        final_data = pd.concat(results, axis=1)
        # Sort theo thời gian
        final_data = final_data.sort_index()
        # Interpolate để điền khuyết thiếu (nếu cần)
        final_data = final_data.interpolate(method='linear', limit_direction='both')
        
        if verbose:
            print(f"✓ Hoàn thành! Tải thành công {len(final_data.columns)}/{len(unique_symbols)} cổ phiếu")
        
        return final_data, skipped_tickers

    except Exception as e:
        print(f"Lỗi khi gộp dữ liệu: {e}")
        return pd.DataFrame(), skipped_tickers


@lru_cache(maxsize=256)
def _fetch_latest_price_single(ticker: str, start_date: str, end_date: str) -> Tuple[Optional[float], Optional[str]]:
    """Return latest close price (in VND) for ticker."""
    sources = ['MSN', 'VCI']
    
    for source in sources:
        try:
            stock = Vnstock().stock(symbol=ticker, source=source)
            stock_data = stock.quote.history(start=str(start_date), end=str(end_date))
            
            if stock_data is not None and not stock_data.empty:
                # Lấy giá đóng cửa mới nhất
                latest_price = float(stock_data['close'].iloc[-1]) * 1000
                return latest_price, None
        except Exception:
            continue
    
    return None, "Không lấy được giá (404 hoặc mã không tồn tại)"


def get_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch the latest close price for each ticker."""
    latest_prices: Dict[str, float] = {}
    
    # Sử dụng giờ VN để đảm bảo ngày "hôm nay" chính xác
    now = datetime.datetime.now(VN_TZ)
    end_date = now.date()
    start_date = end_date - datetime.timedelta(days=7) # Lấy dư ra 1 tuần đề phòng ngày lễ/cuối tuần

    unique_tickers, _ = _normalize_symbols(tickers)
    
    if not unique_tickers:
        return latest_prices

    print(f"\nĐang lấy giá mới nhất cho {len(unique_tickers)} cổ phiếu...")

    def worker(ticker: str):
        p, e = _fetch_latest_price_single(
            ticker, 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d")
        )
        return ticker, p, e

    max_workers = min(8, max(1, len(unique_tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, t): t for t in unique_tickers}
        for future in as_completed(futures):
            sym, price, err = future.result()
            if price is not None:
                latest_prices[sym] = price
            # Có thể print log lỗi nếu cần thiết nhưng để gọn output ta bỏ qua

    print(f"✓ Hoàn thành! Lấy giá thành công cho {len(latest_prices)}/{len(unique_tickers)} cổ phiếu")
    return latest_prices


def fetch_ohlc_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker."""
    sources = ['MSN', 'VCI']
    
    for source in sources:
        try:
            stock = Vnstock().stock(symbol=ticker, source=source)
            stock_data = stock.quote.history(start=str(start_date), end=str(end_date))
            
            if stock_data is None or stock_data.empty:
                continue

            required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            # Chuẩn hóa tên cột về chữ thường để tránh lỗi case-sensitive
            stock_data.columns = [c.lower() for c in stock_data.columns]
            
            available = [col for col in required_columns if col in stock_data.columns]
            if available:
                ohlc_data = stock_data[available].copy()
                ohlc_data['time'] = pd.to_datetime(ohlc_data['time'])
                return ohlc_data
        except Exception:
            continue
    
    print(f"Không thể lấy dữ liệu OHLC cho {ticker} (404 hoặc mã không tồn tại)")
    return pd.DataFrame()


def get_index_history(symbol: str = "VNINDEX", start_date: Optional[str] = None,
                      end_date: Optional[str] = None, months: int = 6,
                      source: str = "MSN") -> pd.DataFrame:
    """Fetch historical quotes for a market index."""
    # Xử lý ngày tháng
    today = datetime.datetime.now(VN_TZ).date()
    e_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else today
    
    if start_date:
        s_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        s_date = e_date - datetime.timedelta(days=months * 30)
    
    # Đảm bảo start < end
    if s_date > e_date:
         s_date = e_date - datetime.timedelta(days=30)

    sources = [source, 'MSN', 'VCI'] if source else ['MSN', 'VCI']
    
    for src in sources:
        try:
            stock = Vnstock().stock(symbol=symbol, source=src)
            history = stock.quote.history(start=s_date.strftime("%Y-%m-%d"), 
                                          end=e_date.strftime("%Y-%m-%d"))
            
            if history is not None and not history.empty:
                history = history.copy()
                history['time'] = pd.to_datetime(history['time'])
                history['symbol'] = symbol
                
                cols = ['time', 'close', 'volume', 'symbol']
                return history[[c for c in cols if c in history.columns]]
        except Exception:
            continue
    
    print(f"Không thể lấy dữ liệu chỉ số {symbol} (thử tất cả sources)")
    return pd.DataFrame()


@lru_cache(maxsize=4)
def _get_sector_snapshot_cached(exchange: str, size: int, source: str) -> pd.DataFrame:
    """Helper cached function for screener - using listing API as fallback."""
    # Try listing API which is more stable than screener
    sources_to_try = ['VCI', 'TCBS'] if not source else [source, 'VCI', 'TCBS']
    
    for src in sources_to_try:
        # Try listing API first (more reliable)
        try:
            stock = Vnstock().stock(symbol='VNINDEX', source=src)
            # Correct method name for listing
            snapshot = stock.listing.symbols_by_exchange(exchange)
            if snapshot is not None and not snapshot.empty:
                print(f"✓ Lấy listing data thành công từ source: {src}")
                # Ensure required columns exist
                if 'symbol' in snapshot.columns and 'ticker' not in snapshot.columns:
                    snapshot['ticker'] = snapshot['symbol']
                # Add default industry if missing
                if 'industry' not in snapshot.columns:
                    snapshot['industry'] = 'Ngành khác'
                # Limit size
                if len(snapshot) > size:
                    snapshot = snapshot.head(size)
                return snapshot
        except Exception as e:
            print(f"✗ Listing from {src} failed: {str(e)[:50]}")
            
        # Fallback to screener (may not work with current version)
        try:
            stock = Vnstock().stock(symbol='VNINDEX', source=src)
            params = {"exchangeName": exchange, "size": size}
            snapshot = stock.screener.stock(params=params)
            if snapshot is not None and not snapshot.empty:
                print(f"✓ Lấy screener data thành công từ source: {src}")
                # Ensure required columns
                if 'symbol' in snapshot.columns and 'ticker' not in snapshot.columns:
                    snapshot['ticker'] = snapshot['symbol']
                if 'industry' not in snapshot.columns:
                    snapshot['industry'] = 'Ngành khác'
                return snapshot
        except Exception as e:
            print(f"✗ Screener from {src} failed: {str(e)[:50]}")
            continue
    
    print(f"✗ Không thể lấy dữ liệu từ tất cả sources: {sources_to_try}")
    return pd.DataFrame()


GROWTH_CONFIG = {
    'price_growth_1w': {
        'aliases': ['1w', '1wk', '1week', 'week1', 'w1', 'one_week', 'weekly'],
        'exclude': ['52w']
    },
    'price_growth_1m': {
        'aliases': ['1m', '1mo', '1month', 'month1', 'm1', 'one_month', 'monthly', '30d', '30day'],
        'exclude': []
    }
}
PCT_KEYWORDS = ['pct', 'percent', 'percentage', 'ratio', 'rate']
CHANGE_KEYWORDS = ['change', 'chg', 'growth', 'return', 'perf', 'delta']
PRICE_KEYWORDS = ['price', 'close', 'match', 'last']


def _prioritize_price_column(columns: List[str]) -> Optional[str]:
    if not columns:
        return None
    ordered = sorted(columns)
    for keyword in ['close', 'price', 'match', 'last']:
        for col in ordered:
            if keyword in col.lower():
                return col
    return ordered[0]


def _select_column_by_keywords(columns: List[str], aliases: List[str], exclude: List[str], candidates: List[str]) -> Optional[str]:
    for col in columns:
        lowered = col.lower()
        if not any(alias in lowered for alias in aliases):
            continue
        if any(ex in lowered for ex in exclude):
            continue
        if any(keyword in lowered for keyword in candidates):
            return col
    return None


def _compute_growth_from_levels(df: pd.DataFrame, aliases: List[str], exclude: List[str]) -> Optional[pd.Series]:
    price_like = [col for col in df.columns if any(key in col.lower() for key in PRICE_KEYWORDS)]
    if not price_like:
        return None

    past_cols = [col for col in price_like if any(alias in col.lower() for alias in aliases)
                 and not any(ex in col.lower() for ex in exclude)]
    current_cols = [col for col in price_like if col not in past_cols]

    past_column = _prioritize_price_column(past_cols)
    current_column = _prioritize_price_column(current_cols)

    if not past_column or not current_column or past_column == current_column:
        return None

    previous = pd.to_numeric(df[past_column], errors='coerce')
    current = pd.to_numeric(df[current_column], errors='coerce')
    denominator = previous.replace({0: np.nan})
    with np.errstate(divide='ignore', invalid='ignore'):
        growth = (current - denominator) / denominator * 100
    growth = growth.replace([np.inf, -np.inf], np.nan)
    return growth


def _infer_growth_column(df: pd.DataFrame, target: str) -> None:
    if target in df.columns:
        return

    config = GROWTH_CONFIG.get(target)
    if not config:
        df[target] = pd.NA
        return

    aliases = config['aliases']
    exclude = config.get('exclude', [])
    columns = list(df.columns)

    candidate = _select_column_by_keywords(columns, aliases, exclude, PCT_KEYWORDS)
    if candidate is None:
        candidate = _select_column_by_keywords(columns, aliases, exclude, CHANGE_KEYWORDS)

    if candidate is not None:
        df[target] = pd.to_numeric(df[candidate], errors='coerce')
        return

    computed = _compute_growth_from_levels(df, aliases, exclude)
    df[target] = computed if computed is not None else pd.NA

def get_sector_snapshot(exchange: str = "HOSE,HNX,UPCOM", size: int = 400,
                        source: str = "VCI", columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch the latest screener snapshot."""
    snapshot = _get_sector_snapshot_cached(exchange, size, source)
    
    if snapshot.empty:
        return pd.DataFrame()

    df = snapshot.copy() # Copy từ cache
    if 'ticker' not in df.columns and 'symbol' in df.columns:
        df['ticker'] = df['symbol']
    
    # Ensure industry column always exists
    if 'industry' not in df.columns:
        df['industry'] = 'Ngành khác'
    else:
        df['industry'] = df['industry'].fillna('Ngành khác')

    for growth_col in GROWTH_CONFIG.keys():
        _infer_growth_column(df, growth_col)

    numeric_columns = [
        'market_cap', 'price_growth_1w', 'price_growth_1m', 
        'avg_trading_value_20d', 'foreign_buysell_20s'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if columns:
        # Chỉ giữ lại các cột tồn tại
        valid_cols = [c for c in columns if c in df.columns]
        if valid_cols:
            return df[valid_cols]
            
    return df


def get_realtime_index_board(symbols: List[str]) -> pd.DataFrame:
    """Fetch real-time index board data using the price_board API."""
    if not symbols:
        return pd.DataFrame()

    sources = ['MSN', 'VCI']
    board = None
    
    for source in sources:
        try:
            stock = Vnstock().stock(symbol='VNINDEX', source=source) 
            board = stock.trading.price_board(symbols)
            if board is not None and not board.empty:
                break
        except Exception:
            continue
    
    if board is None or board.empty:
        return pd.DataFrame()

    if board is None or board.empty:
        return pd.DataFrame()

    board = board.copy()
    
    # Xử lý làm phẳng MultiIndex Columns (nếu có) một cách an toàn
    if isinstance(board.columns, pd.MultiIndex):
        new_cols = []
        for col in board.columns.values:
            # col là tuple, ví dụ ('match', 'price')
            clean_col = "_".join([str(x) for x in col if x and str(x) != 'nan']).strip().lower()
            new_cols.append(clean_col)
        board.columns = new_cols

    # Map tên cột phổ biến từ API về tên chuẩn
    # API thường trả về: a (symbol), b (ceiling), c (floor), ... hoặc tên đầy đủ tùy version
    # Ở đây giả định API trả về tên có chứa từ khóa
    
    rename_map = {
        'thong_tin_cophieu_dang_ky_mack': 'symbol', # Tên cột cũ của TCBS/VND
        'symbol': 'symbol',
        'khop_lenh_gia': 'gia_khop',
        'match_price': 'gia_khop',
        'gia_tham_chieu': 'gia_tham_chieu',
        'reference_price': 'gia_tham_chieu',
        'ref_price': 'gia_tham_chieu'
    }
    
    # Cố gắng rename
    for col in board.columns:
        for key, val in rename_map.items():
            if key in col:
                board.rename(columns={col: val}, inplace=True)

    required = ['symbol', 'gia_khop', 'gia_tham_chieu']
    if not all(col in board.columns for col in required):
        # Fallback: Nếu không tìm thấy cột, trả về DF rỗng thay vì lỗi
        # print(f"Cấu trúc API thay đổi, các cột hiện có: {board.columns.tolist()}")
        return pd.DataFrame()

    board = board.dropna(subset=['symbol'])
    board['symbol'] = board['symbol'].astype(str).str.upper()
    
    for col in ['gia_khop', 'gia_tham_chieu']:
        board[col] = pd.to_numeric(board[col], errors='coerce')

    # Tính toán
    board['thay_doi'] = board['gia_khop'] - board['gia_tham_chieu']
    
    def calc_pct(row):
        ref = row['gia_tham_chieu']
        if ref is None or ref == 0:
            return 0.0
        return ((row['gia_khop'] - ref) / ref) * 100

    board['ty_le_thay_doi'] = board.apply(calc_pct, axis=1)
    board['last_updated'] = datetime.datetime.now(VN_TZ)

    return board[['symbol', 'gia_khop', 'gia_tham_chieu', 'thay_doi', 'ty_le_thay_doi', 'last_updated']]