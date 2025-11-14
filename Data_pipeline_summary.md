### 1. Core Pipeline Modules (`scripts/data_pipeline/`)

#### 📄 `__init__.py`
- Package initialization
- Export các functions chính
- Export config modules
- Version management

**Exports:**
- Config: `POSTGRES_CONFIG`, `VNSTOCK_CONFIG`, `DATA_CONFIG`, `DASHBOARD_CONFIG`
- CSV: `read_company_data_from_csv`
- VNStock: `fetch_stock_data_from_vnstock`, `process_stock_data`
- Postgres: All CRUD operations
- Pipeline: `main_pipeline`, `run_pipeline`

#### 📄 `csv_reader.py`
**Chức năng:**
- Đọc dữ liệu công ty từ CSV
- Validate columns bắt buộc
- Làm sạch dữ liệu (duplicates, nulls)
- Tạo summary statistics
- Auto-load CSV path từ `.env`

**Key Functions:**
- `read_company_data_from_csv(file_path)` - Đọc CSV (default từ .env)
- `validate_company_data(df)` - Validate dữ liệu
- `get_companies_summary(df)` - Lấy thống kê

#### 📄 `vnstock_fetcher.py`
**Chức năng:**
- Lấy dữ liệu giá lịch sử từ VNStock API
- Lấy financial metrics (Beta, P/E, EPS, P/B, ROE, ROA)
- Lấy chỉ số thị trường (VN-Index, VN30, HNX-Index, UPCOM-Index)
- Retry logic cho failed requests
- Rate limiting với configurable delay (từ .env)
- Data processing cho giá và metrics
- Auto-load dates và delay từ `.env`

**Key Functions:**
- `fetch_stock_data_from_vnstock(symbols, start_date, end_date, delay)` - Fetch OHLCV data
  - Params optional, defaults từ .env
- `process_stock_data(raw_data, symbol)` - Xử lý dữ liệu giá
- `fetch_financial_metrics(symbols, start_date, end_date, delay)` - Fetch Beta, P/E, EPS, etc.
- `process_financial_metrics(raw_data, symbol, start_date, end_date)` - Xử lý metrics
- `fetch_market_indices(start_date, end_date)` - Fetch VN-Index, VN30, HNX, UPCOM
- `fetch_latest_prices(tickers)` - Lấy giá mới nhất
- `get_stock_data_summary(stock_data_dict)` - Summary

#### 📄 `config.py` (NEW)
**Chức năng:**
- Load configuration từ `.env` file
- Export config dictionaries
- Connection string builder
- Config validation

**Key Exports:**
- `POSTGRES_CONFIG` - Database credentials
- `VNSTOCK_CONFIG` - API settings
- `DATA_CONFIG` - Data paths and dates
- `DASHBOARD_CONFIG` - Dashboard settings
- `print_config()` - Debug config (hide password)
- `get_postgres_connection_string()` - Build connection URL

#### 📄 `postgres_connector.py`
**Chức năng:**
- Kết nối PostgreSQL
- Tạo normalized schema với 6 tables + 3 views
- Batch insert với ON CONFLICT handling
- Query helpers cho dashboard

**Key Functions:**
- `setup_postgres_connection(config)` - Kết nối DB (auto-load từ .env)
- `create_database_schema(connection)` - Tạo normalized schema
- `load_companies_to_postgres(connection, df)` - Load companies + industries
- `load_stock_prices_to_postgres(connection, stock_data)` - Load OHLCV prices
- `load_stock_metrics_to_postgres(connection, metrics_data)` - Load financial metrics (Beta, P/E, EPS, etc.)
- `load_market_summary_to_postgres(connection, market_data)` - Load market indices (VN-Index, VN30, HNX, UPCOM)
- `fetch_companies_from_database(connection, filters)` - Query companies
- `fetch_stock_prices_from_database(connection, symbols, dates)` - Query prices
- `calculate_portfolio_metrics_from_database(connection, symbols, dates)` - Metrics
- `get_database_stats(connection)` - Thống kê DB

**Database Schema (Normalized):**
```sql
-- 1. exchanges (Sàn giao dịch)
CREATE TABLE exchanges (
    id SERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,  -- HOSE, HNX, UPCOM
    name VARCHAR(100),
    country VARCHAR(50) DEFAULT 'Vietnam',
    created_at TIMESTAMP
);

-- 2. industries (Ngành nghề ICB)
CREATE TABLE industries (
    id SERIAL PRIMARY KEY,
    icb_name VARCHAR(100) UNIQUE NOT NULL,
    icb_code VARCHAR(20),
    sector VARCHAR(100),
    created_at TIMESTAMP
);

-- 3. companies (Công ty niêm yết)
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    organ_name VARCHAR(255) NOT NULL,
    short_name VARCHAR(100),
    exchange_id INTEGER REFERENCES exchanges(id),
    industry_id INTEGER REFERENCES industries(id),
    listing_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- 4. stock_prices_daily (Giá OHLCV hàng ngày)
CREATE TABLE stock_prices_daily (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES companies(symbol),
    date DATE NOT NULL,
    open DECIMAL(15, 2),
    high DECIMAL(15, 2),
    low DECIMAL(15, 2),
    close DECIMAL(15, 2),
    volume BIGINT,
    value DECIMAL(20, 2),
    created_at TIMESTAMP,
    UNIQUE(symbol, date)
);

-- 5. stock_metrics (Chệ số tài chính)
CREATE TABLE stock_metrics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES companies(symbol),
    date DATE NOT NULL,
    beta DECIMAL(10, 6),              -- Hệ số beta (rủi ro hệ thống)
    pe_ratio DECIMAL(15, 4),          -- Tỷ số P/E (Price-to-Earnings)
    eps DECIMAL(15, 4),               -- Thu nhập trên mỗi cổ phiếu (Earnings Per Share)
    pb_ratio DECIMAL(15, 4),          -- Tỷ số P/B (Price-to-Book)
    roe DECIMAL(10, 6),               -- ROE (Return on Equity)
    roa DECIMAL(10, 6),               -- ROA (Return on Assets)
    market_cap DECIMAL(20, 2),        -- Vốn hóa thị trường
    created_at TIMESTAMP,
    UNIQUE(symbol, date)
);

-- 6. market_summary (Chỉ số thị trường)
CREATE TABLE market_summary (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    -- Chỉ số thị trường chính
    vnindex DECIMAL(15, 2),           -- Chỉ số VN-Index (HOSE)
    vnindex_change DECIMAL(10, 4),    -- % thay đổi VN-Index
    vn30 DECIMAL(15, 2),              -- Chỉ số VN30
    vn30_change DECIMAL(10, 4),       -- % thay đổi VN30
    hnx_index DECIMAL(15, 2),         -- Chỉ số HNX-Index
    hnx_index_change DECIMAL(10, 4),  -- % thay đổi HNX-Index
    upcom_index DECIMAL(15, 2),       -- Chỉ số UPCOM-Index
    upcom_index_change DECIMAL(10, 4),-- % thay đổi UPCOM-Index
    -- Thống kê giao dịch
    total_volume BIGINT,              -- Tổng khối lượng giao dịch
    total_value DECIMAL(20, 2),       -- Tổng giá trị giao dịch
    advancing INTEGER,                -- Số mã tăng
    declining INTEGER,                -- Số mã giảm
    unchanged INTEGER,                -- Số mã đứng giá
    created_at TIMESTAMP
);

-- Views for Easy Querying
-- v_stock_full_info: Join companies + exchanges + industries
-- v_latest_prices: Latest price for each stock
-- v_stock_dashboard: Ready-to-use dashboard view
```

**Schema Benefits:**
- ✅ Normalized design (3NF)
- ✅ Separated concerns (prices vs metrics)
- ✅ 10+ indexes for performance
- ✅ 3 pre-built views for common queries
- ✅ Foreign key constraints
- ✅ Scalable for future features

#### 📄 `pipeline_orchestrator.py`
**Chức năng:**
- Điều phối toàn bộ pipeline flow
- Step-by-step execution với logging
- Error handling và recovery
- Verification và stats
- Auto-load config từ `.env`
- Print config before execution

**Key Functions:**
- `main_pipeline(config)` - Main orchestration (auto-load config từ .env)
- `run_pipeline(csv_file, dates, db_config)` - Simplified run
- `run_test_pipeline(num_stocks)` - Test mode

**Pipeline Steps:**
1. ✅ Print Configuration (từ .env)
2. ✅ Read CSV
3. ✅ Fetch from VNStock
4. ✅ Connect to PostgreSQL
5. ✅ Create normalized schema (6 tables + 3 views)
6. ✅ Load data (companies → industries → prices → metrics)
7. ✅ Verify data

### 2. Supporting Files

#### 📄 `.env` (User Created)
Environment variables file (load bởi `python-dotenv`):
```env
# PostgreSQL Configuration
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=portfolio_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# VNStock Configuration
VNSTOCK_SOURCE=VCI
VNSTOCK_DELAY=0.5

# Data Configuration
DATA_START_DATE=2024-01-01
DATA_END_DATE=2024-12-31
CSV_FILE_PATH=../data/company_info.csv

# Dashboard Configuration
DASHBOARD_PORT=8501
DASHBOARD_THEME=light
```

#### 📄 `.env.example`
Template cho environment variables:
- PostgreSQL configuration
- VNStock settings
- Data configuration
- Dashboard settings

**Note:** Copy `.env.example` → `.env` và update với credentials thực tế

#### 📄 `requirements.txt`
Updated với dependencies:
- `psycopg2-binary` - PostgreSQL adapter
- `python-dotenv` - Environment variables
- `tqdm` - Progress bars

#### 📄 `run.py`
Interactive script với 4 modes:
1. **TEST MODE** - 10 stocks, 1 month
2. **SMALL MODE** - 50 stocks, 3 months
3. **FULL MODE** - All stocks, config từ .env
4. **SHOW CONFIG** - Hiển thị cấu hình từ .env (NEW)
5. **EXIT** - Thoát

**Features:**
- Interactive menu
- Config validation
- Auto-load settings từ `.env`

#### 📄 `test_pipeline.py`
Unit tests cho:
- CSV reader
- VNStock fetcher
- Postgres connector
- Pipeline orchestrator

## 📊 Data Flow

```
1. CSV File (company_info.csv)
   │
   ├─→ csv_reader.py (loads CSV_FILE_PATH from .env)
   │   ├─ Validate columns
   │   ├─ Remove duplicates
   │   └─ Clean nulls
   │
2. VNStock API
   │
   ├─→ vnstock_fetcher.py (loads dates & delay from .env)
   │   ├─ Fetch historical OHLCV data
   │   ├─ Fetch financial metrics (Beta, P/E, EPS, P/B, ROE, ROA)
   │   ├─ Fetch market indices (VN-Index, VN30, HNX-Index, UPCOM-Index)
   │   ├─ Retry on failures (3 attempts)
   │   ├─ Rate limiting (configurable delay)
   │   └─ Process data
   │
3. PostgreSQL Database (credentials from .env)
   │
   ├─→ postgres_connector.py
   │   ├─ Create normalized schema
   │   │  ├─ exchanges (3 rows pre-populated)
   │   │  ├─ industries (auto-populated from CSV)
   │   │  ├─ companies (with FK references)
   │   │  ├─ stock_prices_daily (OHLCV data)
   │   │  ├─ stock_metrics (Beta, P/E, EPS, P/B, ROE, ROA, Market Cap)
   │   │  └─ market_summary (VN-Index, VN30, HNX-Index, UPCOM-Index)
   │   │
   │   ├─ Create 3 views
   │   │  ├─ v_stock_full_info (companies joined)
   │   │  ├─ v_latest_prices (latest price per stock)
   │   │  └─ v_stock_dashboard (dashboard ready)
   │   │
   │   ├─ Batch insert companies → industries
   │   ├─ Batch insert prices → stock_prices_daily
   │   ├─ Batch insert metrics → stock_metrics
   │   ├─ Batch insert indices → market_summary
   │   └─ Handle conflicts (ON CONFLICT UPDATE)
   │
4. Dashboard
   │
   └─→ Query from views
       ├─ v_stock_dashboard (Main view)
       ├─ v_latest_prices (Quick access)
       └─ Custom queries with joins
```

## 🔧 Features Implemented

### Configuration Management (NEW)
- ✅ Centralized config module (`config.py`)
- ✅ `.env` file support với `python-dotenv`
- ✅ Auto-load config across all modules
- ✅ Config validation và debug (`print_config()`)
- ✅ No hardcoded credentials

### Data Processing
- ✅ CSV validation và cleaning
- ✅ API retry logic với exponential backoff
- ✅ Rate limiting (configurable từ .env)
- ✅ Data interpolation cho missing values
- ✅ Calculated metrics (daily_return, volatility)

### Database (Normalized Schema)
- ✅ 6-table normalized design (3NF)
- ✅ Auto-create schema với views
- ✅ Indexed tables (10+ indexes)
- ✅ Batch insert với ON CONFLICT
- ✅ Foreign key constraints
- ✅ 3 pre-built views for common queries
- ✅ Query helpers
- ✅ Separated concerns (prices vs metrics)

### Error Handling
- ✅ Connection retry logic
- ✅ Failed symbols tracking
- ✅ Rollback on errors
- ✅ Comprehensive logging

### Performance
- ✅ Batch inserts (100 rows/batch)
- ✅ Database indexes on frequently queried columns
- ✅ Rate limiting để tránh ban
- ✅ Configurable delays từ .env
- ✅ Views for complex queries

### Testing
- ✅ Test mode với limited stocks
- ✅ Unit tests
- ✅ Verification queries
- ✅ Config validation


## 📈 Performance Metrics

### Test Mode (10 stocks, 1 month)
- Time: ~2-3 minutes
- Records: ~200-250
- API calls: ~10

### Small Mode (50 stocks, 3 months)
- Time: ~10-15 minutes
- Records: ~3,000-4,000
- API calls: ~50

### Full Mode (500+ stocks, 1 year)
- Time: ~30-60 minutes
- Records: ~100,000+
- API calls: ~500+

