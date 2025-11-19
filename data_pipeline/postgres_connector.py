"""
Module: postgres_connector.py
Chức năng: Kết nối và quản lý dữ liệu với PostgreSQL
"""

import psycopg2
from psycopg2 import sql, extras
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from db_config import POSTGRES_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_postgres_connection(config: Optional[Dict[str, Any]] = None) -> psycopg2.extensions.connection:
    """
    Thiết lập kết nối PostgreSQL
    
    Args:
        config: Dictionary chứa thông tin kết nối {host, port, database, user, password}
                Nếu None, sẽ đọc từ config.py (load từ .env)
        
    Returns:
        psycopg2.connection: Connection object
        
    Raises:
        ConnectionError: Nếu không thể kết nối
    """
    try:
        # Sử dụng config hoặc load từ .env
        if config is None:
            config = POSTGRES_CONFIG
        
        logger.info(f"Đang kết nối đến PostgreSQL database '{config['database']}' tại {config['host']}:{config['port']}...")
        
        connection = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        
        # Test connection
        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        cursor.close()
        
        logger.info(f"✓ Đã kết nối đến PostgreSQL database")
        logger.debug(f"PostgreSQL version: {version[0]}")
        
        return connection
        
    except psycopg2.OperationalError as e:
        logger.error(f"❌ Lỗi kết nối database: {e}")
        raise ConnectionError(f"Không thể kết nối đến PostgreSQL: {e}")
        
    except Exception as e:
        logger.error(f"❌ Lỗi không xác định: {e}")
        raise


def create_database_schema(connection: psycopg2.extensions.connection) -> None:
    """
    Tạo schema và bảng trong PostgreSQL với thiết kế normalized
    
    Schema Design:
    1. exchanges - Sàn giao dịch
    2. industries - Ngành nghề (ICB)
    3. companies - Công ty niêm yết
    4. stock_prices_daily - Giá hàng ngày (OHLCV)
    5. stock_metrics - Chỉ số tính toán (returns, volatility)
    6. market_summary - Tóm tắt thị trường theo ngày
    
    Args:
        connection: PostgreSQL connection object
    """
    cursor = connection.cursor()
    
    try:
        logger.info("Đang tạo database schema (normalized)...")
        
        # 1. Bảng exchanges (sàn giao dịch)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS exchanges (
            id SERIAL PRIMARY KEY,
            code VARCHAR(20) UNIQUE NOT NULL,
            name VARCHAR(100),
            country VARCHAR(50) DEFAULT 'Vietnam',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        INSERT INTO exchanges (code, name) VALUES 
            ('HOSE', 'Ho Chi Minh Stock Exchange'),
            ('HNX', 'Hanoi Stock Exchange'),
            ('UPCOM', 'Unlisted Public Company Market')
        ON CONFLICT (code) DO NOTHING;
        
        CREATE INDEX IF NOT EXISTS idx_exchange_code ON exchanges(code);
        """)
        logger.info("✓ Bảng 'exchanges' đã được tạo")
        
        # 2. Bảng industries (ngành nghề theo ICB)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS industries (
            id SERIAL PRIMARY KEY,
            icb_name VARCHAR(100) UNIQUE NOT NULL,
            icb_code VARCHAR(20),
            sector VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_icb_name ON industries(icb_name);
        CREATE INDEX IF NOT EXISTS idx_sector ON industries(sector);
        """)
        logger.info("✓ Bảng 'industries' đã được tạo")
        
        # 3. Bảng companies (công ty niêm yết)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) UNIQUE NOT NULL,
            organ_name VARCHAR(255) NOT NULL,
            short_name VARCHAR(100),
            exchange_id INTEGER REFERENCES exchanges(id),
            industry_id INTEGER REFERENCES industries(id),
            listing_date DATE,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_symbol ON companies(symbol);
        CREATE INDEX IF NOT EXISTS idx_exchange_id ON companies(exchange_id);
        CREATE INDEX IF NOT EXISTS idx_industry_id ON companies(industry_id);
        CREATE INDEX IF NOT EXISTS idx_is_active ON companies(is_active);
        """)
        logger.info("✓ Bảng 'companies' đã được tạo")
        
        # 4. Bảng stock_prices_daily (giá OHLCV hàng ngày)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices_daily (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL REFERENCES companies(symbol) ON DELETE CASCADE,
            date DATE NOT NULL,
            open DECIMAL(15, 2),
            high DECIMAL(15, 2),
            low DECIMAL(15, 2),
            close DECIMAL(15, 2),
            volume BIGINT,
            value DECIMAL(20, 2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON stock_prices_daily(symbol, date DESC);
        CREATE INDEX IF NOT EXISTS idx_prices_date ON stock_prices_daily(date DESC);
        CREATE INDEX IF NOT EXISTS idx_prices_symbol ON stock_prices_daily(symbol);
        """)
        logger.info("✓ Bảng 'stock_prices_daily' đã được tạo")
        
        # 5. Bảng stock_metrics (chỉ số tài chính)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_metrics (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL REFERENCES companies(symbol) ON DELETE CASCADE,
            date DATE NOT NULL,
            beta DECIMAL(10, 6),              -- Hệ số beta (rủi ro hệ thống)
            pe_ratio DECIMAL(15, 4),          -- Tỷ số P/E (Price-to-Earnings)
            eps DECIMAL(15, 4),               -- Thu nhập trên mỗi cổ phiếu (Earnings Per Share)
            pb_ratio DECIMAL(15, 4),          -- Tỷ số P/B (Price-to-Book)
            roe DECIMAL(10, 6),               -- ROE (Return on Equity)
            roa DECIMAL(10, 6),               -- ROA (Return on Assets)
            market_cap DECIMAL(20, 2),        -- Vốn hóa thị trường
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_metrics_symbol_date ON stock_metrics(symbol, date DESC);
        CREATE INDEX IF NOT EXISTS idx_metrics_date ON stock_metrics(date DESC);
        CREATE INDEX IF NOT EXISTS idx_metrics_pe ON stock_metrics(pe_ratio);
        """)
        logger.info("✓ Bảng 'stock_metrics' đã được tạo")
        
        # 6. Bảng market_summary (chỉ số thị trường)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_summary (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL UNIQUE,
            -- Chỉ số thị trường chính
            vnindex DECIMAL(15, 2),           -- Chỉ số VN-Index (HOSE)
            vnindex_change DECIMAL(10, 4),    -- % thay đổi VN-Index
            vn30 DECIMAL(15, 2),              -- Chỉ số VN30
            vn30_change DECIMAL(10, 4),       -- % thay đổi VN30
            hnx30 DECIMAL(15, 2),             -- Chỉ số HNX30
            hnx30_change DECIMAL(10, 4),      -- % thay đổi HNX30
            hnx_index DECIMAL(15, 2),         -- Chỉ số HNX-Index
            hnx_index_change DECIMAL(10, 4),  -- % thay đổi HNX-Index
            -- Thống kê giao dịch
            total_volume BIGINT,              -- Tổng khối lượng giao dịch
            total_value DECIMAL(20, 2),       -- Tổng giá trị giao dịch
            advancing INTEGER,                -- Số mã tăng
            declining INTEGER,                -- Số mã giảm
            unchanged INTEGER,                -- Số mã đứng giá
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_market_date ON market_summary(date DESC);
        CREATE INDEX IF NOT EXISTS idx_vnindex ON market_summary(vnindex);
        CREATE INDEX IF NOT EXISTS idx_hnx30 ON market_summary(hnx30);
        """)
        logger.info("✓ Bảng 'market_summary' đã được tạo")
        
        # 7. Tạo view để query dễ dàng
        cursor.execute("""
        CREATE OR REPLACE VIEW v_stock_full_info AS
        SELECT 
            c.symbol,
            c.organ_name,
            c.short_name,
            e.code as exchange,
            e.name as exchange_name,
            i.icb_name as industry,
            i.sector,
            c.is_active,
            c.listing_date
        FROM companies c
        LEFT JOIN exchanges e ON c.exchange_id = e.id
        LEFT JOIN industries i ON c.industry_id = i.id;
        
        CREATE OR REPLACE VIEW v_latest_prices AS
        SELECT DISTINCT ON (symbol)
            symbol,
            date,
            close as latest_price,
            volume as latest_volume,
            (close - open) / NULLIF(open, 0) * 100 as daily_change_pct
        FROM stock_prices_daily
        ORDER BY symbol, date DESC;
        
        CREATE OR REPLACE VIEW v_stock_dashboard AS
        SELECT 
            f.symbol,
            f.organ_name,
            f.exchange,
            f.industry,
            l.latest_price,
            l.daily_change_pct,
            l.date as last_update,
            m.beta,
            m.pe_ratio,
            m.eps,
            m.pb_ratio,
            m.roe,
            m.market_cap
        FROM v_stock_full_info f
        LEFT JOIN v_latest_prices l ON f.symbol = l.symbol
        LEFT JOIN stock_metrics m ON f.symbol = m.symbol AND l.date = m.date
        WHERE f.is_active = TRUE;
        """)
        logger.info("✓ Views đã được tạo")
        
        # Commit changes
        connection.commit()
        logger.info("✓ Database schema (normalized) đã được tạo thành công")
        
    except Exception as e:
        connection.rollback()
        logger.error(f"❌ Lỗi khi tạo schema: {e}")
        raise
        
    finally:
        cursor.close()


def load_companies_to_postgres(
    connection: psycopg2.extensions.connection, 
    company_df: pd.DataFrame
) -> Dict[str, int]:
    """
    Load thông tin công ty vào PostgreSQL với schema normalized
    
    Args:
        connection: PostgreSQL connection object
        company_df: DataFrame chứa thông tin công ty [symbol, organ_name, icb_name, exchange]
        
    Returns:
        dict: {'inserted': count, 'updated': count}
    """
    cursor = connection.cursor()
    inserted_count = 0
    updated_count = 0
    
    try:
        logger.info(f"Đang load {len(company_df)} công ty vào database...")
        
        for _, row in company_df.iterrows():
            # 1. Insert/Get industry
            cursor.execute("""
                INSERT INTO industries (icb_name)
                VALUES (%s)
                ON CONFLICT (icb_name) DO NOTHING
                RETURNING id;
            """, (row['icb_name'],))
            
            result = cursor.fetchone()
            if result:
                industry_id = result[0]
            else:
                cursor.execute("SELECT id FROM industries WHERE icb_name = %s", (row['icb_name'],))
                industry_id = cursor.fetchone()[0]
            
            # 2. Get exchange_id
            cursor.execute("SELECT id FROM exchanges WHERE code = %s", (row['exchange'],))
            exchange_result = cursor.fetchone()
            exchange_id = exchange_result[0] if exchange_result else None
            
            # 3. Insert/Update company
            insert_sql = """
            INSERT INTO companies (symbol, organ_name, exchange_id, industry_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (symbol) 
            DO UPDATE SET 
                organ_name = EXCLUDED.organ_name,
                exchange_id = EXCLUDED.exchange_id,
                industry_id = EXCLUDED.industry_id,
                updated_at = CURRENT_TIMESTAMP
            RETURNING (xmax = 0) AS inserted;
            """
            
            try:
                cursor.execute(insert_sql, (
                    row['symbol'],
                    row['organ_name'],
                    exchange_id,
                    industry_id
                ))
                
                was_inserted = cursor.fetchone()[0]
                if was_inserted:
                    inserted_count += 1
                else:
                    updated_count += 1
                    
            except Exception as e:
                logger.error(f"Lỗi khi insert {row['symbol']}: {e}")
                continue
        
        connection.commit()
        logger.info(f"✓ Companies đã được load: {inserted_count} inserted, {updated_count} updated")
        
        return {'inserted': inserted_count, 'updated': updated_count}
        
    except Exception as e:
        connection.rollback()
        logger.error(f"❌ Lỗi khi load companies: {e}")
        raise
        
    finally:
        cursor.close()


def load_stock_prices_to_postgres(
    connection: psycopg2.extensions.connection, 
    stock_data_dict: Dict[str, pd.DataFrame]
) -> int:
    """
    Load dữ liệu giá cổ phiếu vào PostgreSQL (Batch Insert) - Schema mới
    Load vào bảng: stock_prices_daily
    
    Args:
        connection: PostgreSQL connection object
        stock_data_dict: Dictionary {symbol: dataframe with OHLCV data}
        
    Returns:
        int: Tổng số records đã load
    """
    cursor = connection.cursor()
    total_rows = 0
    
    try:
        logger.info(f"Đang load dữ liệu giá cho {len(stock_data_dict)} cổ phiếu...")
        
        for symbol, dataframe in stock_data_dict.items():
            logger.info(f"Loading {symbol} với {len(dataframe)} rows...")
            
            # Chuẩn bị batch data cho stock_prices_daily
            batch_prices = []
            
            for _, row in dataframe.iterrows():
                # Prices data (OHLCV)
                batch_prices.append((
                    symbol,
                    row.get('time'),
                    row.get('open'),
                    row.get('high'),
                    row.get('low'),
                    row.get('close'),
                    row.get('volume'),
                    None  # value - có thể tính = close * volume
                ))
            
            # Bulk insert prices
            insert_prices_sql = """
            INSERT INTO stock_prices_daily 
                (symbol, date, open, high, low, close, volume, value)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) 
            DO UPDATE SET 
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                value = EXCLUDED.value;
            """
            
            try:
                extras.execute_batch(cursor, insert_prices_sql, batch_prices, page_size=100)
                logger.info(f"✓ {symbol}: {len(batch_prices)} price records loaded")
                
                connection.commit()
                total_rows += len(batch_prices)
                
            except Exception as e:
                connection.rollback()
                logger.error(f"❌ Lỗi khi load prices cho {symbol}: {e}")
                continue
        
        logger.info(f"✓ Tổng cộng {total_rows} price records đã được load")
        
        return total_rows
        
    except Exception as e:
        connection.rollback()
        logger.error(f"❌ Lỗi khi load stock prices: {e}")
        raise
        
    finally:
        cursor.close()


def load_stock_metrics_to_postgres(
    connection: psycopg2.extensions.connection, 
    metrics_dict: Dict[str, pd.DataFrame]
) -> int:
    """
    Load financial metrics (Beta, P/E, EPS) vào PostgreSQL
    
    Args:
        connection: PostgreSQL connection object
        metrics_dict: Dictionary {symbol: dataframe with financial metrics}
        
    Returns:
        int: Tổng số records đã load
    """
    cursor = connection.cursor()
    total_rows = 0
    
    try:
        logger.info(f"Đang load financial metrics cho {len(metrics_dict)} cổ phiếu...")
        
        for symbol, dataframe in metrics_dict.items():
            logger.info(f"Loading metrics for {symbol} với {len(dataframe)} rows...")
            
            # Chuẩn bị batch data cho stock_metrics
            batch_metrics = []
            
            for _, row in dataframe.iterrows():
                # Financial metrics
                batch_metrics.append((
                    symbol,
                    row.get('date'),
                    row.get('beta'),
                    row.get('pe_ratio'),
                    row.get('eps'),
                    row.get('pb_ratio'),
                    row.get('roe'),
                    row.get('roa'),
                    row.get('market_cap')
                ))
            
            # Bulk insert metrics
            insert_metrics_sql = """
            INSERT INTO stock_metrics 
                (symbol, date, beta, pe_ratio, eps, pb_ratio, roe, roa, market_cap)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) 
            DO UPDATE SET 
                beta = EXCLUDED.beta,
                pe_ratio = EXCLUDED.pe_ratio,
                eps = EXCLUDED.eps,
                pb_ratio = EXCLUDED.pb_ratio,
                roe = EXCLUDED.roe,
                roa = EXCLUDED.roa,
                market_cap = EXCLUDED.market_cap;
            """
            
            try:
                extras.execute_batch(cursor, insert_metrics_sql, batch_metrics, page_size=100)
                logger.info(f"✓ {symbol}: {len(batch_metrics)} metric records loaded")
                
                connection.commit()
                total_rows += len(batch_metrics)
                
            except Exception as e:
                connection.rollback()
                logger.error(f"❌ Lỗi khi load metrics cho {symbol}: {e}")
                continue
        
        logger.info(f"✓ Tổng cộng {total_rows} metric records đã được load")
        
        return total_rows
        
    except Exception as e:
        connection.rollback()
        logger.error(f"❌ Lỗi khi load stock metrics: {e}")
        raise
        
    finally:
        cursor.close()


def load_market_summary_to_postgres(
    connection: psycopg2.extensions.connection,
    market_data: pd.DataFrame
) -> int:
    """
    Load market indices (VN-Index, VN30, HNX-Index, UPCOM-Index) vào PostgreSQL
    
    Args:
        connection: PostgreSQL connection object
        market_data: DataFrame with columns: date, vnindex, vn30, hnx_index, upcom_index, etc.
        
    Returns:
        int: Số records đã load
    """
    cursor = connection.cursor()
    total_rows = 0
    
    try:
        logger.info(f"Đang load market summary với {len(market_data)} rows...")
        
        batch_data = []
        for _, row in market_data.iterrows():
            batch_data.append((
                row.get('date'),
                row.get('vnindex'),
                row.get('vnindex_change'),
                row.get('vn30'),
                row.get('vn30_change'),
                row.get('hnx30'),
                row.get('hnx30_change'),
                row.get('hnx_index'),
                row.get('hnx_index_change'),
                row.get('total_volume'),
                row.get('total_value'),
                row.get('advancing'),
                row.get('declining'),
                row.get('unchanged')
            ))
        
        insert_sql = """
        INSERT INTO market_summary 
            (date, vnindex, vnindex_change, vn30, vn30_change, 
             hnx30, hnx30_change, hnx_index, hnx_index_change,
             total_volume, total_value, advancing, declining, unchanged)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date) 
        DO UPDATE SET 
            vnindex = EXCLUDED.vnindex,
            vnindex_change = EXCLUDED.vnindex_change,
            vn30 = EXCLUDED.vn30,
            vn30_change = EXCLUDED.vn30_change,
            hnx30 = EXCLUDED.hnx30,
            hnx30_change = EXCLUDED.hnx30_change,
            hnx_index = EXCLUDED.hnx_index,
            hnx_index_change = EXCLUDED.hnx_index_change,
            total_volume = EXCLUDED.total_volume,
            total_value = EXCLUDED.total_value,
            advancing = EXCLUDED.advancing,
            declining = EXCLUDED.declining,
            unchanged = EXCLUDED.unchanged;
        """
        
        extras.execute_batch(cursor, insert_sql, batch_data, page_size=100)
        connection.commit()
        total_rows = len(batch_data)
        
        logger.info(f"✓ {total_rows} market summary records đã được load")
        
        return total_rows
        
    except Exception as e:
        connection.rollback()
        logger.error(f"❌ Lỗi khi load market summary: {e}")
        raise
        
    finally:
        cursor.close()


def fetch_companies_from_database(
    connection: psycopg2.extensions.connection,
    filters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Lấy danh sách công ty từ database
    
    Args:
        connection: PostgreSQL connection object
        filters: Dictionary chứa điều kiện lọc {'exchange': 'HOSE', 'icb_name': 'Ngân hàng'}
        
    Returns:
        pd.DataFrame: DataFrame chứa thông tin công ty
    """
    cursor = connection.cursor()
    
    try:
        # Build dynamic query
        query = "SELECT * FROM companies WHERE 1=1"
        params = []
        
        if filters:
            if 'exchange' in filters:
                if isinstance(filters['exchange'], list):
                    placeholders = ','.join(['%s'] * len(filters['exchange']))
                    query += f" AND exchange IN ({placeholders})"
                    params.extend(filters['exchange'])
                else:
                    query += " AND exchange = %s"
                    params.append(filters['exchange'])
            
            if 'icb_name' in filters:
                if isinstance(filters['icb_name'], list):
                    placeholders = ','.join(['%s'] * len(filters['icb_name']))
                    query += f" AND icb_name IN ({placeholders})"
                    params.extend(filters['icb_name'])
                else:
                    query += " AND icb_name = %s"
                    params.append(filters['icb_name'])
        
        query += " ORDER BY symbol"
        
        # Execute query
        cursor.execute(query, params)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        companies_df = pd.DataFrame(results, columns=columns)
        
        logger.info(f"✓ Lấy được {len(companies_df)} công ty từ database")
        
        return companies_df
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy dữ liệu companies: {e}")
        return pd.DataFrame()
        
    finally:
        cursor.close()


def fetch_stock_prices_from_database(
    connection: psycopg2.extensions.connection,
    symbols: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Lấy dữ liệu giá cổ phiếu từ database
    
    Args:
        connection: PostgreSQL connection object
        symbols: Danh sách mã cổ phiếu
        start_date: Ngày bắt đầu 'YYYY-MM-DD'
        end_date: Ngày kết thúc 'YYYY-MM-DD'
        
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu giá
    """
    cursor = connection.cursor()
    
    try:
        query = """
        SELECT 
            sp.symbol,
            sp.date,
            sp.open,
            sp.high,
            sp.low,
            sp.close,
            sp.volume,
            sp.daily_return,
            sp.volatility,
            c.organ_name,
            c.icb_name,
            c.exchange
        FROM stock_prices sp
        JOIN companies c ON sp.symbol = c.symbol
        WHERE sp.symbol = ANY(%s)
            AND sp.date BETWEEN %s AND %s
        ORDER BY sp.symbol, sp.date
        """
        
        cursor.execute(query, (symbols, start_date, end_date))
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        stock_prices_df = pd.DataFrame(results, columns=columns)
        
        logger.info(f"✓ Lấy được {len(stock_prices_df)} price records từ database")
        
        return stock_prices_df
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy dữ liệu stock prices: {e}")
        return pd.DataFrame()
        
    finally:
        cursor.close()


def calculate_portfolio_metrics_from_database(
    connection: psycopg2.extensions.connection,
    symbols: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Tính toán metrics cho portfolio từ database
    
    Args:
        connection: PostgreSQL connection object
        symbols: Danh sách mã cổ phiếu
        start_date: Ngày bắt đầu
        end_date: Ngày kết thúc
        
    Returns:
        pd.DataFrame: DataFrame chứa metrics
    """
    cursor = connection.cursor()
    
    try:
        query = """
        WITH price_stats AS (
            SELECT 
                symbol,
                COUNT(*) as trading_days,
                AVG(daily_return) as avg_return,
                STDDEV(daily_return) as std_return,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(volume) as avg_volume,
                (MAX(close) - MIN(close)) / NULLIF(MIN(close), 0) * 100 as price_range_pct
            FROM stock_prices
            WHERE symbol = ANY(%s)
                AND date BETWEEN %s AND %s
            GROUP BY symbol
        )
        SELECT 
            ps.*,
            c.organ_name,
            c.icb_name,
            c.exchange
        FROM price_stats ps
        JOIN companies c ON ps.symbol = c.symbol
        ORDER BY avg_return DESC
        """
        
        cursor.execute(query, (symbols, start_date, end_date))
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        metrics_df = pd.DataFrame(results, columns=columns)
        
        logger.info(f"✓ Tính toán metrics cho {len(metrics_df)} cổ phiếu")
        
        return metrics_df
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi tính toán metrics: {e}")
        return pd.DataFrame()
        
    finally:
        cursor.close()


def get_database_stats(connection: psycopg2.extensions.connection) -> Dict[str, Any]:
    """
    Lấy thống kê database
    
    Args:
        connection: PostgreSQL connection object
        
    Returns:
        dict: Thống kê database
    """
    cursor = connection.cursor()
    
    try:
        # Stats cho companies
        cursor.execute("SELECT COUNT(*) FROM companies")
        num_companies = cursor.fetchone()[0]
        
        # Stats cho stock_prices_daily
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT symbol) as num_stocks,
                COUNT(*) as num_records,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM stock_prices_daily
        """)
        
        stock_stats = cursor.fetchone()
        
        # Stats cho stock_metrics
        cursor.execute("SELECT COUNT(*) FROM stock_metrics")
        num_metrics = cursor.fetchone()[0]
        
        # Stats cho market_summary
        cursor.execute("SELECT COUNT(*) FROM market_summary")
        num_market = cursor.fetchone()[0]
        
        stats = {
            'num_companies': num_companies,
            'num_stocks_with_data': stock_stats[0],
            'num_price_records': stock_stats[1],
            'num_metric_records': num_metrics,
            'num_market_records': num_market,
            'earliest_date': stock_stats[2],
            'latest_date': stock_stats[3]
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy database stats: {e}")
        return {}
        
    finally:
        cursor.close()


if __name__ == "__main__":
    # Test connection
    print("\n" + "="*50)
    print("POSTGRES CONNECTOR TEST")
    print("="*50)
    
    try:
        conn = setup_postgres_connection()
        
        # Create schema
        create_database_schema(conn)
        
        # Get stats
        stats = get_database_stats(conn)
        
        print("\n📊 Database Statistics:")
        print(f"  - Companies: {stats.get('num_companies', 0)}")
        print(f"  - Stocks with data: {stats.get('num_stocks_with_data', 0)}")
        print(f"  - Price records: {stats.get('num_price_records', 0)}")
        print(f"  - Date range: {stats.get('earliest_date')} to {stats.get('latest_date')}")
        
        conn.close()
        print("\n✓ Test completed successfully")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
