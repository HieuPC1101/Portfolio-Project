"""
Module: config.py
Chức năng: Load configuration từ file .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'portfolio_db'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'ktknn')
}

# VNStock Configuration
VNSTOCK_CONFIG = {
    'source': os.getenv('VNSTOCK_SOURCE', 'VCI'),
    'delay': float(os.getenv('VNSTOCK_DELAY', '5'))
}

# Data Configuration
# Tính toán đường dẫn tuyệt đối đến file CSV
_project_root = Path(__file__).parent.parent
_csv_default_path = str(_project_root / 'data' / 'company_info.csv')

DATA_CONFIG = {
    'start_date': os.getenv('DATA_START_DATE', '2024-01-01'),
    'end_date': os.getenv('DATA_END_DATE', '2024-12-31'),
    'csv_file_path': os.getenv('CSV_FILE_PATH', _csv_default_path)
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'port': int(os.getenv('DASHBOARD_PORT', '8501')),
    'theme': os.getenv('DASHBOARD_THEME', 'light')
}


def get_postgres_connection_string() -> str:
    """
    Tạo connection string cho PostgreSQL
    
    Returns:
        str: Connection string
    """
    return (
        f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}"
        f"@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"
    )


def print_config():
    """In ra cấu hình hiện tại (ẩn password)"""
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    
    print("\nPostgreSQL:")
    print(f"  Host: {POSTGRES_CONFIG['host']}")
    print(f"  Port: {POSTGRES_CONFIG['port']}")
    print(f"  Database: {POSTGRES_CONFIG['database']}")
    print(f"  User: {POSTGRES_CONFIG['user']}")
    print(f"  Password: {'*' * len(POSTGRES_CONFIG['password'])}")
    
    print("\nVNStock:")
    print(f"  Source: {VNSTOCK_CONFIG['source']}")
    print(f"  Delay: {VNSTOCK_CONFIG['delay']}s")
    
    print("\nData:")
    print(f"  Start Date: {DATA_CONFIG['start_date']}")
    print(f"  End Date: {DATA_CONFIG['end_date']}")
    print(f"  CSV File: {DATA_CONFIG['csv_file_path']}")
    
    print("\nDashboard:")
    print(f"  Port: {DASHBOARD_CONFIG['port']}")
    print(f"  Theme: {DASHBOARD_CONFIG['theme']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_config()
