"""
Module: pipeline_orchestrator.py
Chức năng: Orchestrate toàn bộ data pipeline
"""

import logging
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from db_config import POSTGRES_CONFIG, DATA_CONFIG, VNSTOCK_CONFIG, print_config
from csv_reader import read_company_data_from_csv
from vnstock_fetcher import (
    fetch_stock_data_from_vnstock, 
    get_stock_data_summary,
    fetch_financial_metrics,
    fetch_market_indices
)
from postgres_connector import (
    setup_postgres_connection,
    create_database_schema,
    load_companies_to_postgres,
    load_stock_prices_to_postgres,
    load_stock_metrics_to_postgres,
    load_market_summary_to_postgres,
    get_database_stats
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main_pipeline(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Orchestrate toàn bộ pipeline: CSV -> VNStock -> PostgreSQL
    
    Args:
        config: Dictionary chứa cấu hình pipeline
                {
                    'csv_file': 'path/to/csv',
                    'start_date': 'YYYY-MM-DD',
                    'end_date': 'YYYY-MM-DD',
                    'postgres': {...},
                    'max_symbols': int (optional, để test với subset)
                }
                Nếu None, sẽ load từ .env
    
    Returns:
        bool: True nếu thành công, False nếu có lỗi
    """
    
    logger.info("="*60)
    logger.info("STARTING DATA PIPELINE")
    logger.info("="*60)
    
    # Default configuration from .env
    if config is None:
        config = {
            'csv_file': DATA_CONFIG['csv_file_path'],
            'start_date': DATA_CONFIG['start_date'],
            'end_date': DATA_CONFIG['end_date'],
            'postgres': POSTGRES_CONFIG
        }
    
    # Print configuration
    print_config()
    
    connection = None
    
    try:
        # ========== STEP 1: READ CSV ==========
        logger.info("\n" + "="*60)
        logger.info("[STEP 1/5] READING COMPANY DATA FROM CSV")
        logger.info("="*60)
        
        companies_df = read_company_data_from_csv(config['csv_file'])
        
        if companies_df.empty:
            logger.error("❌ Failed to read company data from CSV")
            return False
        
        symbols_list = companies_df['symbol'].tolist()
        
        # Limit symbols for testing if specified
        if 'max_symbols' in config and config['max_symbols'] > 0:
            symbols_list = symbols_list[:config['max_symbols']]
            companies_df = companies_df[companies_df['symbol'].isin(symbols_list)]
            logger.info(f"⚠ Testing mode: Limited to {len(symbols_list)} symbols")
        
        logger.info(f"✓ Loaded {len(symbols_list)} companies")
        
        # ========== STEP 2: FETCH FROM VNSTOCK ==========
        logger.info("\n" + "="*60)
        logger.info("[STEP 2/6] FETCHING STOCK PRICES FROM VNSTOCK API")
        logger.info("="*60)
        
        stock_data_dict, failed_symbols = fetch_stock_data_from_vnstock(
            symbols_list,
            config['start_date'],
            config['end_date'],
            delay=0.5
        )
        
        if not stock_data_dict:
            logger.error("❌ Failed to fetch any stock data from VNStock")
            return False
        
        # Display summary
        summary = get_stock_data_summary(stock_data_dict)
        logger.info(f"\n✓ Fetched price data for {summary['total_stocks']} stocks")
        logger.info(f"✓ Total records: {summary['total_records']}")
        
        if failed_symbols:
            logger.warning(f"⚠ Failed to fetch: {len(failed_symbols)} stocks")
        
        # ========== STEP 2B: FETCH FINANCIAL METRICS ==========
        logger.info("\n" + "="*60)
        logger.info("[STEP 2B/6] FETCHING FINANCIAL METRICS (Beta, P/E, EPS)")
        logger.info("="*60)
        
        metrics_dict, failed_metrics = fetch_financial_metrics(
            symbols_list,
            config['start_date'],
            config['end_date'],
            delay=5
        )
        
        logger.info(f"\n✓ Fetched metrics for {len(metrics_dict)} stocks")
        if failed_metrics:
            logger.warning(f"⚠ Failed to fetch metrics: {len(failed_metrics)} stocks")
        
        # ========== STEP 2C: FETCH MARKET INDICES ==========
        logger.info("\n" + "="*60)
        logger.info("[STEP 2C/6] FETCHING MARKET INDICES (VN-Index, VN30, HNX, UPCOM)")
        logger.info("="*60)
        
        market_data = fetch_market_indices(
            config['start_date'],
            config['end_date']
        )
        
        if not market_data.empty:
            logger.info(f"\n✓ Fetched market data: {len(market_data)} days")
        else:
            logger.warning("⚠ No market index data fetched")
        
        # ========== STEP 3: CONNECT TO POSTGRESQL ==========
        logger.info("\n" + "="*60)
        logger.info("[STEP 3/6] CONNECTING TO POSTGRESQL")
        logger.info("="*60)
        
        connection = setup_postgres_connection(config.get('postgres'))
        
        if connection is None:
            logger.error("❌ Failed to connect to PostgreSQL")
            return False
        
        # ========== STEP 4: CREATE SCHEMA ==========
        logger.info("\n" + "="*60)
        logger.info("[STEP 4/6] CREATING DATABASE SCHEMA")
        logger.info("="*60)
        
        create_database_schema(connection)
        
        # ========== STEP 5: LOAD DATA ==========
        logger.info("\n" + "="*60)
        logger.info("[STEP 5/6] LOADING DATA TO POSTGRESQL")
        logger.info("="*60)
        
        # Load companies
        logger.info("\n📊 Loading companies...")
        company_stats = load_companies_to_postgres(connection, companies_df)
        logger.info(f"✓ Companies: {company_stats['inserted']} inserted, {company_stats['updated']} updated")
        
        # Load stock prices
        logger.info("\n📊 Loading stock prices...")
        total_price_rows = load_stock_prices_to_postgres(connection, stock_data_dict)
        logger.info(f"✓ Total price records loaded: {total_price_rows}")
        
        # Load financial metrics
        if metrics_dict:
            logger.info("\n📊 Loading financial metrics...")
            total_metric_rows = load_stock_metrics_to_postgres(connection, metrics_dict)
            logger.info(f"✓ Total metric records loaded: {total_metric_rows}")
        else:
            logger.warning("⚠ No metrics to load")
            total_metric_rows = 0
        
        # Load market summary
        if not market_data.empty:
            logger.info("\n📊 Loading market indices...")
            total_market_rows = load_market_summary_to_postgres(connection, market_data)
            logger.info(f"✓ Total market records loaded: {total_market_rows}")
        else:
            logger.warning("⚠ No market data to load")
            total_market_rows = 0
        
        # ========== VERIFICATION ==========
        logger.info("\n" + "="*60)
        logger.info("VERIFYING DATA IN DATABASE")
        logger.info("="*60)
        
        stats = get_database_stats(connection)
        
        logger.info("\n📊 Database Statistics:")
        logger.info(f"  ✓ Companies: {stats['num_companies']}")
        logger.info(f"  ✓ Stocks with data: {stats['num_stocks_with_data']}")
        logger.info(f"  ✓ Price records: {stats['num_price_records']}")
        logger.info(f"  ✓ Metric records: {stats.get('num_metric_records', 0)}")
        logger.info(f"  ✓ Market records: {stats.get('num_market_records', 0)}")
        logger.info(f"  ✓ Date range: {stats['earliest_date']} to {stats['latest_date']}")
        
        # ========== COMPLETION ==========
        logger.info("\n" + "="*60)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("\n📊 Summary:")
        logger.info(f"  • CSV: {len(companies_df)} companies loaded")
        logger.info(f"  • VNStock Prices: {len(stock_data_dict)}/{len(symbols_list)} stocks fetched")
        logger.info(f"  • VNStock Metrics: {len(metrics_dict)} stocks fetched")
        logger.info(f"  • Market Indices: {len(market_data)} days")
        logger.info(f"  • PostgreSQL: {total_price_rows} prices + {total_metric_rows} metrics + {total_market_rows} market records")
        logger.info("\n🚀 Dashboard ready to use!")
        logger.info("   Run: streamlit run scripts/dashboard.py")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}", exc_info=True)
        return False
        
    finally:
        # Close connection
        if connection:
            try:
                connection.close()
                logger.info("\n✓ Database connection closed")
            except:
                pass


def run_pipeline(
    csv_file: str = './data/company_info.csv',
    start_date: str = '2024-01-01',
    end_date: str = '2024-12-31',
    max_symbols: Optional[int] = None,
    db_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Chạy pipeline với các tham số được chỉ định
    
    Args:
        csv_file: Đường dẫn đến file CSV
        start_date: Ngày bắt đầu lấy dữ liệu
        end_date: Ngày kết thúc lấy dữ liệu
        max_symbols: Số lượng symbols tối đa (None = tất cả)
        db_config: Cấu hình PostgreSQL
        
    Returns:
        bool: True nếu thành công
    """
    config = {
        'csv_file': csv_file,
        'start_date': start_date,
        'end_date': end_date,
        'postgres': db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'portfolio_db',
            'user': 'postgres',
            'password': 'postgres'
        }
    }
    
    if max_symbols:
        config['max_symbols'] = max_symbols
    
    return main_pipeline(config)


def run_test_pipeline(num_stocks: int = 10):
    """
    Chạy pipeline ở chế độ test với số lượng cổ phiếu giới hạn
    
    Args:
        num_stocks: Số lượng cổ phiếu để test
    """
    logger.info(f"\n🧪 Running TEST pipeline với {num_stocks} cổ phiếu...\n")
    
    success = run_pipeline(
        csv_file='./data/company_info.csv',
        start_date='2024-11-01',
        end_date='2024-11-30',
        max_symbols=num_stocks
    )
    
    if success:
        logger.info("\n✓ Test pipeline completed successfully!")
    else:
        logger.error("\n✗ Test pipeline failed!")
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Portfolio Data Pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited stocks')
    parser.add_argument('--num-stocks', type=int, default=10, help='Number of stocks for test mode')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--csv', type=str, default='data/company_info.csv', help='Path to CSV file')
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode
        run_test_pipeline(args.num_stocks)
    else:
        # Full pipeline
        run_pipeline(
            csv_file=args.csv,
            start_date=args.start_date,
            end_date=args.end_date
        )
