"""
Data Pipeline Package
Quản lý luồng dữ liệu từ CSV -> VNStock -> PostgreSQL -> Dashboard
"""

from .config import (
    POSTGRES_CONFIG, 
    VNSTOCK_CONFIG, 
    DATA_CONFIG, 
    DASHBOARD_CONFIG,
    get_postgres_connection_string,
    print_config
)
from .csv_reader import read_company_data_from_csv
from .vnstock_fetcher import fetch_stock_data_from_vnstock, process_stock_data
from .postgres_connector import (
    setup_postgres_connection,
    create_database_schema,
    load_companies_to_postgres,
    load_stock_prices_to_postgres,
    fetch_companies_from_database,
    fetch_stock_prices_from_database,
    calculate_portfolio_metrics_from_database
)
from .pipeline_orchestrator import main_pipeline, run_pipeline

__all__ = [
    # Config
    'POSTGRES_CONFIG',
    'VNSTOCK_CONFIG',
    'DATA_CONFIG',
    'DASHBOARD_CONFIG',
    'get_postgres_connection_string',
    'print_config',
    # CSV Reader
    'read_company_data_from_csv',
    # VNStock Fetcher
    'fetch_stock_data_from_vnstock',
    'process_stock_data',
    # Postgres Connector
    'setup_postgres_connection',
    'create_database_schema',
    'load_companies_to_postgres',
    'load_stock_prices_to_postgres',
    'fetch_companies_from_database',
    'fetch_stock_prices_from_database',
    'calculate_portfolio_metrics_from_database',
    # Pipeline Orchestrator
    'main_pipeline',
    'run_pipeline'
]

__version__ = '1.0.0'
