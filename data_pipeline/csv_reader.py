"""
Module: csv_reader.py
Chức năng: Đọc dữ liệu công ty từ file CSV
"""

import pandas as pd
import os
import logging
from db_config import DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_company_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Đọc danh sách công ty từ file CSV
    
    Args:
        file_path (str): Đường dẫn đến file CSV
        
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu công ty với columns [symbol, organ_name, icb_name, exchange]
        
    Raises:
        FileNotFoundError: Nếu file không tồn tại
        ValueError: Nếu thiếu columns bắt buộc
    """
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} không tồn tại. Vui lòng kiểm tra lại.")
        
        # Đọc CSV
        logger.info(f"Đang đọc dữ liệu từ {file_path}...")
        company_df = pd.read_csv(file_path)
        
        # Validate columns
        required_columns = ['symbol', 'organ_name', 'icb_name', 'exchange']
        missing_columns = [col for col in required_columns if col not in company_df.columns]
        
        if missing_columns:
            raise ValueError(f"Thiếu các columns bắt buộc: {missing_columns}")
        
        # Làm sạch dữ liệu
        original_count = len(company_df)
        
        # Xóa duplicates
        company_df = company_df.drop_duplicates(subset=['symbol'], keep='first')
        
        # Xóa null values
        company_df = company_df.dropna(subset=required_columns)
        
        # Trim whitespace
        for col in company_df.columns:
            if company_df[col].dtype == 'object':
                company_df[col] = company_df[col].str.strip()
        
        cleaned_count = len(company_df)
        removed_count = original_count - cleaned_count
        
        if removed_count > 0:
            logger.warning(f"Đã loại bỏ {removed_count} dòng dữ liệu trùng lặp hoặc thiếu")
        
        logger.info(f"✓ Loaded {cleaned_count} công ty từ CSV")
        
        return company_df
        
    except FileNotFoundError as e:
        logger.error(f"Lỗi: {e}")
        return pd.DataFrame()
        
    except ValueError as e:
        logger.error(f"Lỗi validate dữ liệu: {e}")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Lỗi khi đọc dữ liệu từ file CSV: {e}")
        return pd.DataFrame()


def validate_company_data(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Validate dữ liệu công ty
    
    Args:
        df: DataFrame cần validate
        
    Returns:
        tuple: (is_valid, errors_list)
    """
    errors = []
    
    if df.empty:
        errors.append("DataFrame rỗng")
        return False, errors
    
    # Check required columns
    required_columns = ['symbol', 'organ_name', 'icb_name', 'exchange']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        errors.append(f"Thiếu columns: {missing_columns}")
    
    # Check for duplicates
    duplicates = df[df.duplicated(subset=['symbol'], keep=False)]
    if not duplicates.empty:
        errors.append(f"Có {len(duplicates)} symbols trùng lặp")
    
    # Check for null values
    null_counts = df[required_columns].isnull().sum()
    null_columns = null_counts[null_counts > 0].to_dict()
    
    if null_columns:
        errors.append(f"Có giá trị null: {null_columns}")
    
    # Check valid exchanges
    valid_exchanges = ['HOSE', 'HNX', 'UPCOM']
    invalid_exchanges = df[~df['exchange'].isin(valid_exchanges)]['exchange'].unique()
    
    if len(invalid_exchanges) > 0:
        errors.append(f"Sàn giao dịch không hợp lệ: {invalid_exchanges.tolist()}")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors


def get_companies_summary(df: pd.DataFrame) -> dict:
    """
    Lấy tóm tắt thông tin công ty
    
    Args:
        df: DataFrame công ty
        
    Returns:
        dict: Thông tin tóm tắt
    """
    if df.empty:
        return {}
    
    summary = {
        'total_companies': len(df),
        'by_exchange': df['exchange'].value_counts().to_dict(),
        'by_industry': df['icb_name'].value_counts().to_dict(),
        'unique_industries': df['icb_name'].nunique(),
        'symbols': df['symbol'].tolist()
    }
    
    return summary


if __name__ == "__main__":
    # Test - Use config from .env
    csv_path = DATA_CONFIG['csv_file_path']
    
    df = read_company_data_from_csv(csv_path)
    
    if not df.empty:
        print("\n" + "="*50)
        print("COMPANY DATA SUMMARY")
        print("="*50)
        
        is_valid, errors = validate_company_data(df)
        
        if is_valid:
            print("✓ Dữ liệu hợp lệ")
        else:
            print("✗ Dữ liệu có lỗi:")
            for error in errors:
                print(f"  - {error}")
        
        summary = get_companies_summary(df)
        print(f"\nTổng số công ty: {summary['total_companies']}")
        print(f"Số ngành: {summary['unique_industries']}")
        print(f"\nPhân bố theo sàn:")
        for exchange, count in summary['by_exchange'].items():
            print(f"  - {exchange}: {count}")
        
        print("\n5 công ty đầu tiên:")
        print(df.head())
