"""
Module: vnstock_fetcher.py
Chức năng: Lấy dữ liệu cổ phiếu từ VNStock API
"""

import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

import pandas as pd
import time
import logging
from typing import Dict, List, Tuple
from vnstock import Vnstock
import datetime
from db_config import VNSTOCK_CONFIG, DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_stock_data_from_vnstock(
    symbols_list: List[str], 
    start_date: str = None, 
    end_date: str = None,
    delay: float = None
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Lấy dữ liệu lịch sử giá cổ phiếu từ VNStock API
    
    Args:
        symbols_list: Danh sách mã cổ phiếu
        start_date: Ngày bắt đầu (định dạng 'YYYY-MM-DD'), default from .env
        end_date: Ngày kết thúc (định dạng 'YYYY-MM-DD'), default from .env
        delay: Thời gian delay giữa các request (seconds), default from .env
        
    Returns:
        tuple: (stock_data_dict, failed_symbols)
            - stock_data_dict: Dictionary {symbol: dataframe}
            - failed_symbols: Danh sách cổ phiếu không tải được dữ liệu
    """
    # Use config from .env if not provided
    if start_date is None:
        start_date = DATA_CONFIG['start_date']
    if end_date is None:
        end_date = DATA_CONFIG['end_date']
    if delay is None:
        delay = VNSTOCK_CONFIG['delay']
    
    stock_data_dict = {}
    failed_symbols = []
    
    logger.info(f"Đang tải dữ liệu cho {len(symbols_list)} cổ phiếu từ {start_date} đến {end_date}...")
    
    for i, symbol in enumerate(symbols_list, 1):
        logger.info(f"[{i}/{len(symbols_list)}] Đang tải {symbol}...")
        
        try:
            # Khởi tạo VNStock instance
            stock_instance = Vnstock().stock(symbol=symbol, source='VCI')
            
            # Lấy dữ liệu lịch sử
            historical_data = stock_instance.quote.history(
                start=str(start_date), 
                end=str(end_date)
            )
            
            if historical_data is not None and not historical_data.empty:
                # Chuẩn hóa dữ liệu
                processed_data = process_stock_data(historical_data, symbol)
                stock_data_dict[symbol] = processed_data
                
                logger.info(f"✓ Success: {symbol} - {len(processed_data)} rows")
            else:
                failed_symbols.append(symbol)
                logger.warning(f"✗ No data: {symbol}")
                
        except Exception as e:
            error_msg = str(e)
            
            # Lọc các loại lỗi phổ biến
            if "RetryError" in error_msg or "ConnectionError" in error_msg:
                error_msg = "Không thể kết nối đến server"
            elif "ValueError" in error_msg:
                error_msg = "Dữ liệu không hợp lệ"
            
            failed_symbols.append(symbol)
            logger.error(f"✗ Failed: {symbol} - {error_msg}")
        
        # Đợi giữa các request để tránh overload
        if i < len(symbols_list):  # Không cần đợi sau request cuối cùng
            time.sleep(delay)
    
    success_count = len(stock_data_dict)
    failed_count = len(failed_symbols)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"✓ Hoàn thành! Tải thành công {success_count}/{len(symbols_list)} cổ phiếu")
    
    if failed_count > 0:
        logger.warning(f"✗ Không tải được {failed_count} cổ phiếu: {failed_symbols[:10]}{'...' if failed_count > 10 else ''}")
    
    return stock_data_dict, failed_symbols


def process_stock_data(raw_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Xử lý và chuẩn hóa dữ liệu cổ phiếu
    
    Args:
        raw_data: DataFrame dữ liệu thô từ VNStock
        symbol: Mã cổ phiếu
        
    Returns:
        pd.DataFrame: Dữ liệu đã được xử lý và chuẩn hóa
    """
    # Chọn các cột cần thiết
    required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    available_columns = [col for col in required_columns if col in raw_data.columns]
    
    processed_data = raw_data[available_columns].copy()
    
    # Chuyển đổi kiểu dữ liệu
    processed_data['time'] = pd.to_datetime(processed_data['time'])
    processed_data['symbol'] = symbol
    
    # Xử lý giá trị thiếu bằng nội suy
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in processed_data.columns:
            processed_data[col] = processed_data[col].interpolate(method='linear', limit_direction='both')
    
    # Tính toán chỉ số bổ sung
    if 'close' in processed_data.columns:
        # Daily return (%)
        processed_data['daily_return'] = processed_data['close'].pct_change()
        
        # Volatility (rolling standard deviation of returns)
        processed_data['volatility'] = processed_data['daily_return'].rolling(window=20, min_periods=1).std()
        
        # Fill NaN values for first rows
        processed_data['daily_return'] = processed_data['daily_return'].fillna(0)
        processed_data['volatility'] = processed_data['volatility'].fillna(0)
    
    # Sắp xếp theo thời gian
    processed_data = processed_data.sort_values('time').reset_index(drop=True)
    
    return processed_data


def fetch_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Lấy giá cổ phiếu mới nhất từ vnstock3
    
    Args:
        tickers: Danh sách mã cổ phiếu
        
    Returns:
        dict: Dictionary chứa giá cổ phiếu mới nhất {ticker: price}
    """
    latest_prices = {}
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=7)
    
    logger.info(f"\nĐang lấy giá mới nhất cho {len(tickers)} cổ phiếu...")
    
    for i, ticker in enumerate(tickers, 1):
        try:
            stock = Vnstock().stock(symbol=ticker, source='VCI')
            stock_data = stock.quote.history(start=str(start_date), end=str(end_date))
            
            if stock_data is not None and not stock_data.empty:
                # Lấy giá đóng cửa (close) của ngày cuối cùng trong dữ liệu
                latest_price = stock_data['close'].iloc[-1] * 1000
                latest_prices[ticker] = latest_price
                logger.info(f"[{i}/{len(tickers)}] {ticker}: {latest_price:,.0f} VND ✓")
            else:
                logger.warning(f"[{i}/{len(tickers)}] {ticker}: Không có dữ liệu ✗")
                
        except Exception as e:
            error_msg = str(e)
            if "RetryError" in error_msg:
                error_msg = "Không thể kết nối"
            elif "ValueError" in error_msg:
                error_msg = "Dữ liệu không hợp lệ"
            logger.error(f"[{i}/{len(tickers)}] {ticker}: Lỗi - {error_msg} ✗")
        
        # Small delay
        time.sleep(0.3)
    
    logger.info(f"✓ Hoàn thành! Lấy giá thành công cho {len(latest_prices)}/{len(tickers)} cổ phiếu\n")
    
    return latest_prices


def get_stock_data_summary(stock_data_dict: Dict[str, pd.DataFrame]) -> dict:
    """
    Tạo tóm tắt dữ liệu cổ phiếu
    
    Args:
        stock_data_dict: Dictionary chứa dữ liệu cổ phiếu
        
    Returns:
        dict: Thông tin tóm tắt
    """
    if not stock_data_dict:
        return {}
    
    total_records = sum(len(df) for df in stock_data_dict.values())
    
    date_ranges = {}
    for symbol, df in stock_data_dict.items():
        if not df.empty and 'time' in df.columns:
            date_ranges[symbol] = {
                'start': df['time'].min(),
                'end': df['time'].max(),
                'days': len(df)
            }
    
    summary = {
        'total_stocks': len(stock_data_dict),
        'total_records': total_records,
        'avg_records_per_stock': total_records / len(stock_data_dict) if stock_data_dict else 0,
        'date_ranges': date_ranges,
        'symbols': list(stock_data_dict.keys())
    }
    
    return summary


def fetch_financial_metrics(
    symbols_list: List[str],
    start_date: str = None,
    end_date: str = None,
    delay: float = None
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Lấy các chỉ số tài chính (Beta, P/E, EPS, P/B, ROE, ROA) từ VNStock
    
    Args:
        symbols_list: Danh sách mã cổ phiếu
        start_date: Ngày bắt đầu (định dạng 'YYYY-MM-DD')
        end_date: Ngày kết thúc (định dạng 'YYYY-MM-DD')
        delay: Thời gian delay giữa các request (seconds)
        
    Returns:
        tuple: (metrics_dict, failed_symbols)
            - metrics_dict: Dictionary {symbol: dataframe with financial metrics}
            - failed_symbols: Danh sách cổ phiếu không tải được
    """
    # Use config from .env if not provided
    if start_date is None:
        start_date = DATA_CONFIG['start_date']
    if end_date is None:
        end_date = DATA_CONFIG['end_date']
    if delay is None:
        delay = VNSTOCK_CONFIG['delay']
    
    metrics_dict = {}
    failed_symbols = []
    
    logger.info(f"Đang tải financial metrics cho {len(symbols_list)} cổ phiếu...")
    
    for i, symbol in enumerate(symbols_list, 1):
        logger.info(f"[{i}/{len(symbols_list)}] Đang tải metrics {symbol}...")
        
        try:
            # Khởi tạo VNStock instance
            stock_instance = Vnstock().stock(symbol=symbol, source='VCI')
            
            # Lấy financial ratio data (sử dụng tiếng Việt)
            financial_ratios = stock_instance.finance.ratio(period='year', lang='vi')
            
            if financial_ratios is not None and not financial_ratios.empty:
                # Xử lý dữ liệu - lấy dữ liệu năm gần nhất
                processed_metrics = process_financial_metrics(financial_ratios, symbol)
                
                if not processed_metrics.empty:
                    metrics_dict[symbol] = processed_metrics
                    logger.info(f"✓ Success: {symbol} - {len(processed_metrics)} metric rows")
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"✗ No metrics: {symbol}")
            else:
                failed_symbols.append(symbol)
                logger.warning(f"✗ No data: {symbol}")
                
        except Exception as e:
            error_msg = str(e)
            
            if "RetryError" in error_msg or "ConnectionError" in error_msg:
                error_msg = "Không thể kết nối đến server"
            
            failed_symbols.append(symbol)
            logger.error(f"✗ Failed: {symbol} - {error_msg}")
        
        # Đợi giữa các request
        if i < len(symbols_list):
            time.sleep(delay)
    
    success_count = len(metrics_dict)
    failed_count = len(failed_symbols)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"✓ Hoàn thành! Tải metrics thành công {success_count}/{len(symbols_list)} cổ phiếu")
    
    if failed_count > 0:
        logger.warning(f"✗ Không tải được {failed_count} cổ phiếu")
    
    return metrics_dict, failed_symbols


def process_financial_metrics(
    raw_data: pd.DataFrame,
    symbol: str
) -> pd.DataFrame:
    """
    Xử lý và chuẩn hóa financial metrics (lấy dữ liệu năm gần nhất)
    
    Args:
        raw_data: DataFrame dữ liệu thô từ VNStock finance.ratio() với lang='vi'
        symbol: Mã cổ phiếu
        
    Returns:
        pd.DataFrame: Dữ liệu metrics đã xử lý (1 row cho năm gần nhất)
    """
    try:
        if raw_data.empty:
            return pd.DataFrame()
        
        # Lấy dữ liệu hàng mới nhất (năm gần nhất)
        latest_data = raw_data.iloc[0]
        
        # Lấy năm để làm date
        year = latest_data.get('yearReport', latest_data.get('year', datetime.datetime.now().year))
        date = pd.to_datetime(f'{year}-12-31')
        
        # Tạo dictionary với các chỉ số tài chính (sử dụng tên cột tiếng Việt từ VNStock)
        metrics_data = {
            'symbol': symbol,
            'date': date,
            'pe_ratio': latest_data.get('priceToEarning', None),
            'pb_ratio': latest_data.get('priceToBook', None),
            'eps': latest_data.get('earningPerShare', None),
            'roe': latest_data.get('roe', None),
            'roa': latest_data.get('roa', None),
            'beta': latest_data.get('beta', None),
            'market_cap': latest_data.get('marketCap', None)
        }
        
        # Tạo DataFrame với 1 row
        processed_data = pd.DataFrame([metrics_data])
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý metrics cho {symbol}: {e}")
        return pd.DataFrame()


def fetch_market_indices(
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Lấy dữ liệu các chỉ số thị trường (VN-Index, VN30, HNX-Index)
    
    Args:
        start_date: Ngày bắt đầu (định dạng 'YYYY-MM-DD')
        end_date: Ngày kết thúc (định dạng 'YYYY-MM-DD')
        
    Returns:
        pd.DataFrame: DataFrame chứa các chỉ số thị trường theo ngày
    """
    # Use config from .env if not provided
    if start_date is None:
        start_date = DATA_CONFIG['start_date']
    if end_date is None:
        end_date = DATA_CONFIG['end_date']
    
    logger.info(f"Đang tải market indices từ {start_date} đến {end_date}...")
    
    # Các chỉ số cần lấy
    indices = {
        'VNINDEX': 'vnindex',
        'VN30': 'vn30',
        'HNX30': 'hnx30',
        'HNXINDEX': 'hnx_index'
    }
    
    market_data = None
    
    for index_symbol, column_name in indices.items():
        logger.info(f"Đang tải {index_symbol}...")
        
        try:
            # Lấy dữ liệu index
            stock_instance = Vnstock().stock(symbol=index_symbol, source='VCI')
            index_data = stock_instance.quote.history(start=str(start_date), end=str(end_date))
            
            if index_data is not None and not index_data.empty:
                # Chuẩn bị dữ liệu
                temp_df = pd.DataFrame()
                temp_df['date'] = pd.to_datetime(index_data['time'])
                temp_df[column_name] = index_data['close']
                
                # Tính % thay đổi
                temp_df[f'{column_name}_change'] = index_data['close'].pct_change() * 100
                
                # Merge vào market_data
                if market_data is None:
                    market_data = temp_df
                else:
                    market_data = pd.merge(market_data, temp_df, on='date', how='outer')
                
                logger.info(f"✓ {index_symbol} loaded - {len(index_data)} rows")
            else:
                logger.warning(f"✗ No data for {index_symbol}")
                
        except Exception as e:
            logger.error(f"✗ Lỗi khi tải {index_symbol}: {e}")
        
        time.sleep(VNSTOCK_CONFIG['delay'])
    
    # Thêm các cột khác (có thể tính từ stock data hoặc để null)
    if market_data is not None:
        market_data['total_volume'] = None
        market_data['total_value'] = None
        market_data['advancing'] = None
        market_data['declining'] = None
        market_data['unchanged'] = None
        
        market_data = market_data.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✓ Hoàn thành! Market indices: {len(market_data)} rows")
    else:
        logger.warning("✗ Không có dữ liệu market indices")
        market_data = pd.DataFrame()
    
    return market_data


if __name__ == "__main__":
    # Test với một vài mã cổ phiếu
    test_symbols = ['VCB', 'VNM', 'HPG', 'FPT', 'VIC']
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    print("\n" + "="*50)
    print("VNSTOCK FETCHER TEST")
    print("="*50)
    
    stock_data, failed = fetch_stock_data_from_vnstock(
        test_symbols,
        start_date,
        end_date,
        delay=0.5
    )
    
    if stock_data:
        summary = get_stock_data_summary(stock_data)
        
        print(f"\n✓ Tải thành công: {summary['total_stocks']} cổ phiếu")
        print(f"✓ Tổng số records: {summary['total_records']}")
        print(f"✓ Trung bình: {summary['avg_records_per_stock']:.0f} records/cổ phiếu")
        
        # Hiển thị sample data
        first_symbol = list(stock_data.keys())[0]
        print(f"\nSample data cho {first_symbol}:")
        print(stock_data[first_symbol].head())
        print(f"\nColumns: {stock_data[first_symbol].columns.tolist()}")
