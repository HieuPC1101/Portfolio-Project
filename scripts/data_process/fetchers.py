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


@lru_cache(maxsize=1)
def _load_company_info_mapping() -> Dict[str, str]:
    """Load symbol -> industry mapping from local CSV."""
    mapping = {}
    try:
        csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'company_info.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            if 'symbol' in df.columns and 'icb_name' in df.columns:
                df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
                mapping = dict(zip(df['symbol'], df['icb_name']))
    except Exception as e:
        print(f"Error loading company info: {e}")
    return mapping


@lru_cache(maxsize=4)
def _get_sector_snapshot_cached(exchange: str, size: int, source: str) -> pd.DataFrame:
    """
    Robust implementation using Listing API + Price Board.
    Replaces the broken Screener API.
    """
    try:
        # 1. Get List of Symbols for Exchange (HOSE, HNX, UPCOM)
        # Note: listing.symbols_by_exchange does NOT accept 'exchange' kwarg, strict positional
        # But commonly we just want everything or filter by group
        
        # Strategy: Get ALL symbols from listing first
        stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
        listing_df = stock.listing.symbols_by_exchange()
        
        if listing_df is None or listing_df.empty:
            print("✗ Listing API returned empty.")
            return pd.DataFrame()

        # Relaxed logic: Do not strictly filter by 'type'='STO' because API might return varied values.
        # We rely on the intersection with industry_map to pick valid companies.
        
        all_symbols = listing_df['symbol'].tolist()
        
        # 2. Map Industry from Local CSV
        industry_map = _load_company_info_mapping()
        
        candidates = [s for s in all_symbols if s in industry_map]
        if not candidates:
             candidates = all_symbols[:size] # Fallback
        candidates = candidates[:600]
        print(f"Fetching snapshot for {len(candidates)} symbols...")
        
        # Batch fetching
        chunk_size = 100
        frames = []
        for i in range(0, len(candidates), chunk_size):
            chunk = candidates[i : i + chunk_size]
            try:
                board = stock.trading.price_board(chunk)
                if board is not None and not board.empty:
                     frames.append(board)
            except Exception:
                continue
                
        if not frames:
             return pd.DataFrame()
             
        full_board = pd.concat(frames, ignore_index=True)
        
        # Flatten columns (MultiIndex)
        # Columns like ('match', 'match_price'), ('match', 'accumulated_value'), ...
        new_cols = []
        for col in full_board.columns.values:
            if isinstance(col, tuple):
                name = "_".join([str(x) for x in col if x]).strip().lower()
            else:
                name = str(col).lower()
            new_cols.append(name)
        full_board.columns = new_cols

        # 4. Standardize & Compute Columns
        # Target: ticker, industry, market_cap, price_growth_1w (use daily), avg_trading_value_20d (use val)
        
        # Map Symbol
        if 'listing_symbol' in full_board.columns:
            full_board['ticker'] = full_board['listing_symbol']
        elif 'symbol' in full_board.columns:
            full_board['ticker'] = full_board['symbol']
            
        # Map Industry
        full_board['industry'] = full_board['ticker'].map(industry_map).fillna('Ngành khác')
        
        # Map Price / Change
        # match_match_price, match_match_vol
        # listing_ref_price
        
        # Ensure numeric
        numeric_cols = ['match_match_price', 'listing_ref_price', 'match_accumulated_value', 
                        'listing_listed_share', 'match_foreign_buy_value', 'match_foreign_sell_value']
                        
        for col in numeric_cols:
            if col in full_board.columns:
                full_board[col] = pd.to_numeric(full_board[col], errors='coerce').fillna(0)
            else:
                full_board[col] = 0.0

        # Derived Metrics
        full_board['price'] = full_board['match_match_price']
        
        # Daily Change % as proxy for growth
        full_board['daily_change'] = 0.0
        mask = full_board['listing_ref_price'] > 0
        full_board.loc[mask, 'daily_change'] = (
            (full_board.loc[mask, 'match_match_price'] - full_board.loc[mask, 'listing_ref_price']) 
            / full_board.loc[mask, 'listing_ref_price'] * 100
        )
        
        full_board['market_cap'] = full_board['match_match_price'] * full_board['listing_listed_share']
        
        # Liquidity (Value)
        full_board['avg_trading_value_20d'] = full_board['match_accumulated_value'] # Proxy using today's value
        
        # Foreign Flow
        full_board['foreign_buysell_20s'] = full_board['match_foreign_buy_value'] - full_board['match_foreign_sell_value']
        
        # Growth Map (Fallback)
        full_board['price_growth_1w'] = full_board['daily_change']
        full_board['price_growth_1m'] = full_board['daily_change']
        
        return full_board

    except Exception as e:
        print(f"Error in robust sector snapshot: {e}")
        return pd.DataFrame()


GROWTH_CONFIG = {} # Deprecated config but kept for compatibility logic below if needed

def get_sector_snapshot(exchange: str = "HOSE,HNX,UPCOM", size: int = 400,
                        source: str = "VCI", columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch the latest snapshot using robust method (PriceBoard + CSV)."""
    snapshot = _get_sector_snapshot_cached(exchange, size, source)
    if snapshot.empty:
        return pd.DataFrame()
        
    if columns:
        valid_cols = [c for c in columns if c in snapshot.columns]
        return snapshot[valid_cols]
        
    return snapshot


def get_realtime_index_board(symbols: List[str]) -> pd.DataFrame:
    """Fetch real-time index board data using the price_board API."""
    if not symbols:
        return pd.DataFrame()
    try:
        stock = Vnstock().stock(symbol='VNINDEX', source='VCI') 
        board = stock.trading.price_board(symbols)
    except Exception as e:
        print(f"Error fetching VCI board: {e}")
        board = None

    
    if board is None or board.empty:
        return pd.DataFrame()

    board = board.copy()
    
    # Xử lý làm phẳng MultiIndex Columns (nếu có)
    if isinstance(board.columns, pd.MultiIndex):
        new_cols = []
        for col in board.columns.values:
            # col là tuple, ví dụ ('match', 'price')
            clean_col = "_".join([str(x) for x in col if x and str(x) != 'nan']).strip().lower()
            new_cols.append(clean_col)
        board.columns = new_cols

    rename_map = {
        'thong_tin_cophieu_dang_ky_mack': 'symbol', # Tên cột cũ của TCBS/VND
        'listing_symbol': 'symbol',
        'symbol': 'symbol',
        'khop_lenh_gia': 'gia_khop',
        'match_price': 'gia_khop',
        'match_match_price': 'gia_khop',
        'gia_tham_chieu': 'gia_tham_chieu',
        'reference_price': 'gia_tham_chieu',
        'listing_ref_price': 'gia_tham_chieu',
        'ref_price': 'gia_tham_chieu'
    }
    
    # Cố gắng rename
    for col in board.columns:
        if col in rename_map:
             board.rename(columns={col: rename_map[col]}, inplace=True)
             continue
        # Fuzzy match if exact match fail
        for key, val in rename_map.items():
            if key in col and val not in board.columns:
                board.rename(columns={col: val}, inplace=True)

    required = ['symbol', 'gia_khop', 'gia_tham_chieu']
    if not all(col in board.columns for col in required):
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