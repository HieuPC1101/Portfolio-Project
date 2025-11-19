"""
Module data_loader.py
Chứa các hàm lấy dữ liệu từ PostgreSQL database.
"""

import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

import pandas as pd
import os
import sys
import datetime
from typing import Dict, List, Tuple

# Thêm path để import từ data_pipeline
pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))
if pipeline_path not in sys.path:
    sys.path.insert(0, pipeline_path)

# Import postgres_connector
from postgres_connector import setup_postgres_connection

# Import config từ scripts (tránh xung đột với config.py trong data_pipeline)
scripts_path = os.path.dirname(__file__)
sys.path.insert(0, scripts_path)

try:
    from config import ANALYSIS_START_DATE, ANALYSIS_END_DATE
except ImportError:
    # Fallback nếu import thất bại
    ANALYSIS_START_DATE = '2024-01-01'
    ANALYSIS_END_DATE = '2024-12-31'


def fetch_data_from_csv(file_path):
    """
    Đọc dữ liệu từ file CSV chứa thông tin công ty.
    
    Args:
        file_path (str): Đường dẫn đến file CSV
        
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu công ty
    """
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df
        else:
            print(f"File {file_path} không tồn tại. Vui lòng kiểm tra lại.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu từ file CSV: {e}")
        return pd.DataFrame()


def fetch_stock_data2(symbols, start_date, end_date):
    """
    Lấy dữ liệu giá lịch sử cho danh sách cổ phiếu từ PostgreSQL.
    
    Args:
        symbols (list): Danh sách mã cổ phiếu
        start_date (str): Ngày bắt đầu (định dạng 'YYYY-MM-DD')
        end_date (str): Ngày kết thúc (định dạng 'YYYY-MM-DD')

    Returns:
        tuple: (data, skipped_tickers)
            - data (pd.DataFrame): Dữ liệu giá lịch sử, mỗi cổ phiếu là một cột
            - skipped_tickers (list): Danh sách cổ phiếu không có dữ liệu
    """
    data = pd.DataFrame()
    skipped_tickers = []
    
    try:
        # Kết nối database
        connection = setup_postgres_connection()
        cursor = connection.cursor()
        
        print(f"Đang tải dữ liệu cho {len(symbols)} cổ phiếu từ database...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Đang tải {symbol}...", end=" ")
            
            # Query dữ liệu từ stock_prices_daily
            query = """
            SELECT date, close
            FROM stock_prices_daily
            WHERE symbol = %s
                AND date BETWEEN %s AND %s
            ORDER BY date
            """
            
            cursor.execute(query, (symbol, start_date, end_date))
            results = cursor.fetchall()
            
            if results:
                # Tạo DataFrame cho symbol này
                symbol_df = pd.DataFrame(results, columns=['time', symbol])
                symbol_df['time'] = pd.to_datetime(symbol_df['time'])
                symbol_df[symbol] = pd.to_numeric(symbol_df[symbol], errors='coerce')  # Convert sang numeric
                symbol_df.set_index('time', inplace=True)
                
                # Merge vào data chính
                if data.empty:
                    data = symbol_df
                else:
                    data = pd.merge(data, symbol_df, how='outer', left_index=True, right_index=True)
                
                print(f"✓ Thành công ({len(results)} rows)")
            else:
                print("✗ Không có dữ liệu")
                skipped_tickers.append(symbol)
        
        cursor.close()
        connection.close()
        
        # Xử lý giá trị bị thiếu bằng nội suy
        if not data.empty:
            # Đảm bảo tất cả các cột là numeric
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.interpolate(method='linear', limit_direction='both')
            print(f"\n✓ Hoàn thành! Tải thành công {len(data.columns)}/{len(symbols)} cổ phiếu")
        else:
            print(f"\n✗ Không có dữ liệu trong database")
        
        return data, skipped_tickers
        
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ database: {e}")
        return pd.DataFrame(), symbols


def get_latest_prices(tickers):
    """
    Lấy giá cổ phiếu mới nhất từ PostgreSQL.
    
    Args:
        tickers (list): Danh sách mã cổ phiếu
        
    Returns:
        dict: Dictionary chứa giá cổ phiếu mới nhất {ticker: price}
    """
    latest_prices = {}
    
    try:
        # Kết nối database
        connection = setup_postgres_connection()
        cursor = connection.cursor()
        
        print(f"\nĐang lấy giá mới nhất cho {len(tickers)} cổ phiếu từ database...")
        
        for i, ticker in enumerate(tickers, 1):
            # Query giá mới nhất
            query = """
            SELECT close, date
            FROM stock_prices_daily
            WHERE symbol = %s
            ORDER BY date DESC
            LIMIT 1
            """
            
            cursor.execute(query, (ticker,))
            result = cursor.fetchone()
            
            if result:
                latest_price = float(result[0]) * 1000  # Chuyển sang VND
                latest_prices[ticker] = latest_price
                print(f"[{i}/{len(tickers)}] {ticker}: {latest_price:,.0f} VND ✓")
            else:
                print(f"[{i}/{len(tickers)}] {ticker}: Không có dữ liệu ✗")
        
        cursor.close()
        connection.close()
        
        print(f"✓ Hoàn thành! Lấy giá thành công cho {len(latest_prices)}/{len(tickers)} cổ phiếu\n")
        
    except Exception as e:
        print(f"Lỗi khi lấy giá từ database: {e}")
    
    return latest_prices


def calculate_metrics(data):
    """
    Tính lợi nhuận kỳ vọng và phương sai (rủi ro).
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        
    Returns:
        tuple: (mean_returns, volatility)
    """
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    volatility = returns.std()
    return mean_returns, volatility


def fetch_fundamental_data(symbol):
    """
    Lấy dữ liệu phân tích cơ bản của một cổ phiếu từ PostgreSQL.
    
    Args:
        symbol (str): Mã cổ phiếu
        
    Returns:
        dict: Dictionary chứa các chỉ số phân tích cơ bản
    """
    try:
        # Kết nối database
        connection = setup_postgres_connection()
        cursor = connection.cursor()
        
        # Query dữ liệu từ stock_metrics (lấy dữ liệu mới nhất)
        query = """
        SELECT 
            symbol,
            pe_ratio,
            pb_ratio,
            eps,
            roe,
            roa,
            market_cap,
            date
        FROM stock_metrics
        WHERE symbol = %s
        ORDER BY date DESC
        LIMIT 1
        """
        
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if result:
            fundamental_data = {
                'symbol': result[0],
                'pe': result[1],
                'pb': result[2],
                'eps': result[3],
                'roe': result[4],
                'roa': result[5],
                'profit_margin': None,  # Không có trong database
                'revenue': None,  # Không có trong database
                'profit': None,  # Không có trong database
                'market_cap': result[6]
            }
            return fundamental_data
        else:
            print(f"Không có dữ liệu phân tích cơ bản cho {symbol}")
            return None
        
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu phân tích cơ bản cho {symbol}: {str(e)}")
        return None


def fetch_fundamental_data_batch(symbols):
    """
    Lấy dữ liệu phân tích cơ bản cho nhiều mã cổ phiếu từ PostgreSQL.
    
    Args:
        symbols (list): Danh sách mã cổ phiếu
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu phân tích cơ bản của các mã cổ phiếu
    """
    fundamental_list = []
    
    try:
        # Kết nối database
        connection = setup_postgres_connection()
        cursor = connection.cursor()
        
        print(f"\nĐang lấy dữ liệu phân tích cơ bản cho {len(symbols)} mã cổ phiếu từ database...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Đang xử lý {symbol}...", end=" ")
            
            # Query dữ liệu từ stock_metrics
            query = """
            SELECT 
                symbol,
                pe_ratio,
                pb_ratio,
                eps,
                roe,
                roa,
                market_cap
            FROM stock_metrics
            WHERE symbol = %s
            ORDER BY date DESC
            LIMIT 1
            """
            
            cursor.execute(query, (symbol,))
            result = cursor.fetchone()
            
            if result:
                data = {
                    'symbol': result[0],
                    'pe': result[1],
                    'pb': result[2],
                    'eps': result[3],
                    'roe': result[4],
                    'roa': result[5],
                    'profit_margin': None,
                    'revenue': None,
                    'profit': None,
                    'market_cap': result[6]
                }
                fundamental_list.append(data)
                print("✓ Thành công")
            else:
                print("✗ Không có dữ liệu")
        
        cursor.close()
        connection.close()
        
        if fundamental_list:
            df = pd.DataFrame(fundamental_list)
            print(f"\n✓ Hoàn thành! Lấy dữ liệu thành công cho {len(df)}/{len(symbols)} mã cổ phiếu")
            return df
        else:
            print(f"\n✗ Không có dữ liệu trong database")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ database: {e}")
        return pd.DataFrame()


def fetch_ohlc_data(ticker, start_date, end_date):
    """
    Lấy dữ liệu OHLC (Open, High, Low, Close) cho một mã cổ phiếu từ PostgreSQL.
    
    Args:
        ticker (str): Mã cổ phiếu
        start_date (str): Ngày bắt đầu (định dạng 'YYYY-MM-DD')
        end_date (str): Ngày kết thúc (định dạng 'YYYY-MM-DD')
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu OHLC với các cột time, open, high, low, close, volume
    """
    try:
        # Kết nối database
        connection = setup_postgres_connection()
        cursor = connection.cursor()
        
        # Query dữ liệu OHLC
        query = """
        SELECT date, open, high, low, close, volume
        FROM stock_prices_daily
        WHERE symbol = %s
            AND date BETWEEN %s AND %s
        ORDER BY date
        """
        
        cursor.execute(query, (ticker, start_date, end_date))
        results = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        if results:
            ohlc_data = pd.DataFrame(
                results,
                columns=['time', 'open', 'high', 'low', 'close', 'volume']
            )
            ohlc_data['time'] = pd.to_datetime(ohlc_data['time'])
            return ohlc_data
        else:
            print(f"Không có dữ liệu OHLC cho {ticker}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu OHLC cho {ticker}: {str(e)}")
        return pd.DataFrame()


def get_market_indices(start_date, end_date):
    """
    Lấy dữ liệu chỉ số thị trường từ PostgreSQL.
    
    Args:
        start_date (str): Ngày bắt đầu
        end_date (str): Ngày kết thúc
        
    Returns:
        pd.DataFrame: DataFrame chứa các chỉ số thị trường
    """
    try:
        connection = setup_postgres_connection()
        cursor = connection.cursor()
        
        query = """
        SELECT 
            date,
            vnindex,
            vnindex_change,
            vn30,
            vn30_change,
            hnx30,
            hnx30_change,
            hnx_index,
            hnx_index_change
            
        FROM market_summary
        WHERE date BETWEEN %s AND %s
        ORDER BY date
        """
        
        cursor.execute(query, (start_date, end_date))
        results = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        if results:
            columns = ['date', 'vnindex', 'vnindex_change', 'vn30', 'vn30_change',
                      'hnx_index', 'hnx_index_change', 'hnx30', 'hnx30_change']
            df = pd.DataFrame(results, columns=columns)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Lỗi khi lấy market indices: {e}")
        return pd.DataFrame()
