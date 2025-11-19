# Configuration Guide

## File `.env` Configuration

Tất cả cấu hình của pipeline được quản lý trong file `.env` ở root directory.

### 📝 Cấu trúc file `.env`

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

### 🔧 Các Module Sử Dụng Config

#### 1. **config.py** - Module quản lý config chính
```python
from config import POSTGRES_CONFIG, DATA_CONFIG, VNSTOCK_CONFIG, print_config

# In ra config hiện tại
print_config()

# Sử dụng config
db_host = POSTGRES_CONFIG['host']
csv_path = DATA_CONFIG['csv_file_path']
delay = VNSTOCK_CONFIG['delay']
```

#### 2. **csv_reader.py** - Đọc CSV
```python
from config import DATA_CONFIG
from csv_reader import read_company_data_from_csv

# Tự động load CSV_FILE_PATH từ .env
df = read_company_data_from_csv(DATA_CONFIG['csv_file_path'])
```

#### 3. **vnstock_fetcher.py** - Fetch dữ liệu stock
```python
from vnstock_fetcher import fetch_stock_data_from_vnstock

# Tự động dùng DATA_START_DATE, DATA_END_DATE, VNSTOCK_DELAY từ .env
stock_data, failed = fetch_stock_data_from_vnstock(symbols_list)
```

#### 4. **postgres_connector.py** - Kết nối database
```python
from postgres_connector import setup_postgres_connection

# Tự động dùng POSTGRES_* từ .env
connection = setup_postgres_connection()
```

#### 5. **pipeline_orchestrator.py** - Chạy pipeline
```python
from pipeline_orchestrator import run_pipeline

# Tự động load tất cả config từ .env
success = run_pipeline()
```

### ⚙️ Kiểm Tra Config

**Xem config hiện tại:**
```bash
python config.py
```

**Hoặc trong run_quick.py, chọn option 4:**
```bash
python run_quick.py
# Chọn 4: SHOW CONFIG
```

### 🔄 Thay Đổi Config

1. **Chỉnh sửa file `.env`**
2. **Restart các script** (config được load khi import module)

### 📌 Lưu Ý

✅ **DO:**
- Luôn kiểm tra `.env` trước khi chạy pipeline
- Dùng `print_config()` để verify config
- Backup `.env` trước khi thay đổi

❌ **DON'T:**
- Commit `.env` lên Git (đã có trong `.gitignore`)
- Hardcode config trong code
- Share `.env` file (có password)

### 🔐 Bảo Mật

- File `.env` chứa mật khẩu PostgreSQL
- Đã được thêm vào `.gitignore`
- Dùng `.env.example` để share template

### 🎯 Ví Dụ Sử Dụng

**Chạy với config mặc định từ .env:**
```bash
python pipeline_orchestrator.py
```

**Chạy test mode:**
```bash
python pipeline_orchestrator.py --test --num-stocks 10
```

**Chạy interactive với config từ .env:**
```bash
python run_quick.py
# Chọn mode 1, 2, hoặc 3
```

---

📚 **Xem thêm:** `README.md` trong thư mục `data_pipeline/`
