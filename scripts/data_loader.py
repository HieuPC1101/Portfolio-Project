"""
Module data_loader.py
Chứa các hàm liên quan đến việc đọc dữ liệu từ file CSV và lấy dữ liệu giá cổ phiếu từ API.
"""

import warnings
# Tắt cảnh báo pkg_resources deprecated từ thư viện vnai
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

import pandas as pd
import os
import datetime
from vnstock import Vnstock
from concurrent.futures import ThreadPoolExecutor
from scripts.config import ANALYSIS_START_DATE, ANALYSIS_END_DATE


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


def create_vnstock_instance():
    """
    Tạo đối tượng Vnstock để lấy dữ liệu giá cổ phiếu.
    
    Returns:
        Vnstock: Đối tượng Vnstock
    """
    return Vnstock().stock(symbol='VN30F1M', source='VCI')


def fetch_stock_data2(symbols, start_date, end_date):
    """
    Lấy dữ liệu giá lịch sử cho danh sách cổ phiếu từ Vnstock.
    
    Args:
        symbols (list): Danh sách mã cổ phiếu
        start_date (str): Ngày bắt đầu (định dạng 'YYYY-MM-DD')
        end_date (str): Ngày kết thúc (định dạng 'YYYY-MM-DD')

    Returns:
        tuple: (data, skipped_tickers)
            - data (pd.DataFrame): Dữ liệu giá lịch sử, mỗi cổ phiếu là một cột
            - skipped_tickers (list): Danh sách cổ phiếu không tải được dữ liệu
    """
    data = pd.DataFrame()
    skipped_tickers = []

    def fetch_single_stock(ticker):
        """
        Lấy dữ liệu cho một cổ phiếu với xử lý lỗi cải tiến.
        """
        try:
            stock = Vnstock().stock(symbol=ticker, source='VCI')
            stock_data = stock.quote.history(start=str(start_date), end=str(end_date))
            if stock_data is not None and not stock_data.empty:
                stock_data = stock_data[['time', 'close']].rename(columns={'close': ticker})
                stock_data['time'] = pd.to_datetime(stock_data['time'])
                return ticker, stock_data.set_index('time'), None
            else:
                return ticker, pd.DataFrame(), f"Không có dữ liệu"
        except Exception as e:
            error_msg = str(e)
            # Lọc các loại lỗi phổ biến
            if "RetryError" in error_msg:
                error_msg = "Không thể kết nối đến server"
            elif "ValueError" in error_msg:
                error_msg = "Dữ liệu không hợp lệ"
            print(f"Lỗi khi lấy dữ liệu {ticker}: {error_msg}")
            return ticker, pd.DataFrame(), error_msg

    # Tải dữ liệu tuần tự thay vì song song để tránh quá tải
    print(f"Đang tải dữ liệu cho {len(symbols)} cổ phiếu...")
    for i, ticker in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Đang tải {ticker}...", end=" ")
        ticker_name, result, error = fetch_single_stock(ticker)
        
        if not result.empty:
            print("✓ Thành công")
            if data.empty:
                data = result
            else:
                data = pd.merge(data, result, how='outer', on='time')
        else:
            print(f"✗ Bỏ qua ({error})")
            skipped_tickers.append(ticker)

    # Xử lý giá trị bị thiếu bằng nội suy
    if not data.empty:
        data = data.interpolate(method='linear', limit_direction='both')
        print(f"\n✓ Hoàn thành! Tải thành công {len(data.columns)}/{len(symbols)} cổ phiếu")
    else:
        print(f"\n✗ Không thể tải dữ liệu cho bất kỳ cổ phiếu nào")

    return data, skipped_tickers


def get_latest_prices(tickers):
    """
    Lấy giá cổ phiếu mới nhất từ vnstock3.
    
    Args:
        tickers (list): Danh sách mã cổ phiếu
        
    Returns:
        dict: Dictionary chứa giá cổ phiếu mới nhất {ticker: price}
    """
    latest_prices = {}
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=7)

    print(f"\nĐang lấy giá mới nhất cho {len(tickers)} cổ phiếu...")
    for i, ticker in enumerate(tickers, 1):
        try:
            stock = Vnstock().stock(symbol=ticker, source='VCI')
            stock_data = stock.quote.history(start=str(start_date), end=str(end_date))
            if stock_data is not None and not stock_data.empty:
                # Lấy giá đóng cửa (close) của ngày cuối cùng trong dữ liệu
                latest_price = stock_data['close'].iloc[-1] * 1000
                latest_prices[ticker] = latest_price
                print(f"[{i}/{len(tickers)}] {ticker}: {latest_price:,.0f} VND ✓")
            else:
                print(f"[{i}/{len(tickers)}] {ticker}: Không có dữ liệu ✗")
        except Exception as e:
            error_msg = str(e)
            if "RetryError" in error_msg:
                error_msg = "Không thể kết nối"
            elif "ValueError" in error_msg:
                error_msg = "Dữ liệu không hợp lệ"
            print(f"[{i}/{len(tickers)}] {ticker}: Lỗi - {error_msg} ✗")
    
    print(f"✓ Hoàn thành! Lấy giá thành công cho {len(latest_prices)}/{len(tickers)} cổ phiếu\n")
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
    Lấy dữ liệu phân tích cơ bản của một cổ phiếu từ vnstock.
    
    Args:
        symbol (str): Mã cổ phiếu
        
    Returns:
        dict: Dictionary chứa các chỉ số phân tích cơ bản
            {
                'symbol': str,
                'pe': float,      # P/E ratio
                'pb': float,      # P/B ratio  
                'eps': float,     # EPS (Thu nhập trên mỗi cổ phiếu)
                'roe': float,     # ROE (%) 
                'roa': float,     # ROA (%)
                'profit_margin': float,  # Biên lợi nhuận (%)
                'revenue': float, # Doanh thu (tỷ VND)
                'profit': float   # Lợi nhuận (tỷ VND)
            }
    """
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        
        # Lấy dữ liệu tài chính
        financial_ratio = stock.finance.ratio(period='year', lang='vi')
        
        if financial_ratio is None or financial_ratio.empty:
            print(f"Không có dữ liệu phân tích cơ bản cho {symbol}")
            return None
        
        # Lấy dữ liệu hàng mới nhất (năm gần nhất)
        latest_data = financial_ratio.iloc[0] if not financial_ratio.empty else None
        
        if latest_data is None:
            return None
        
        # Trích xuất các chỉ số quan trọng
        fundamental_data = {
            'symbol': symbol,
            'pe': latest_data.get('priceToEarning', None),
            'pb': latest_data.get('priceToBook', None),
            'eps': latest_data.get('earningPerShare', None),
            'roe': latest_data.get('roe', None),
            'roa': latest_data.get('roa', None),
            'profit_margin': latest_data.get('netProfitMargin', None),
            'revenue': latest_data.get('revenue', None),
            'profit': latest_data.get('postTaxProfit', None)
        }
        
        return fundamental_data
        
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu phân tích cơ bản cho {symbol}: {str(e)}")
        return None


def fetch_fundamental_data_batch(symbols):
    """
    Lấy dữ liệu phân tích cơ bản cho nhiều cổ phiếu.
    
    Args:
        symbols (list): Danh sách mã cổ phiếu
        
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu phân tích cơ bản của các cổ phiếu
    """
    fundamental_list = []
    
    print(f"\nĐang lấy dữ liệu phân tích cơ bản cho {len(symbols)} cổ phiếu...")
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Đang xử lý {symbol}...", end=" ")
        data = fetch_fundamental_data(symbol)
        if data:
            fundamental_list.append(data)
            print("✓ Thành công")
        else:
            print("✗ Không có dữ liệu")
    
    if fundamental_list:
        df = pd.DataFrame(fundamental_list)
        print(f"\n✓ Hoàn thành! Lấy dữ liệu thành công cho {len(df)}/{len(symbols)} cổ phiếu")
        return df
    else:
        print(f"\n✗ Không thể lấy dữ liệu phân tích cơ bản cho bất kỳ cổ phiếu nào")
        return pd.DataFrame()
