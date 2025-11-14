# 🚀 Quick Start Guide

## Bắt đầu nhanh trong 5 phút!

### Bước 1: Setup (chỉ chạy 1 lần)

```powershell
# Chạy setup script
.\setup.ps1
```

Script này sẽ:
- ✅ Kiểm tra Python
- ✅ Cài đặt dependencies
- ✅ Setup PostgreSQL (Docker hoặc local)
- ✅ Tạo file .env
- ✅ Kiểm tra CSV file

### Bước 2: Chạy Pipeline

**Option 1: Quick Start (Khuyến nghị cho lần đầu)**

```powershell
python scripts/data_pipeline/run_quick.py
```

Chọn mode:
- `1` - TEST MODE: 10 cổ phiếu, 1 tháng (~2-3 phút)
- `2` - SMALL MODE: 50 cổ phiếu, 3 tháng (~10-15 phút)
- `3` - FULL MODE: Tất cả cổ phiếu, 1 năm (~30-60 phút)

**Option 2: Command Line**

```powershell
# Test với 10 cổ phiếu
python scripts/data_pipeline/pipeline_orchestrator.py --test --num-stocks 10

# Custom
python scripts/data_pipeline/pipeline_orchestrator.py --start-date 2024-01-01 --end-date 2024-12-31
```

### Bước 3: Chạy Dashboard

```powershell
streamlit run scripts/dashboard.py
```

Mở browser tại: http://localhost:8501

## 📊 Pipeline Flow

```
CSV File (company_info.csv)
    ↓
📥 Read & Validate
    ↓
🌐 Fetch from VNStock API
    ↓
💾 Save to PostgreSQL
    ↓
📊 Dashboard Display
```

## 🔧 Troubleshooting

### Lỗi: "Connection refused" - PostgreSQL

```powershell
# Khởi động PostgreSQL Docker
docker start portfolio-postgres

# Hoặc tạo mới
docker run --name portfolio-postgres \
  -e POSTGRES_DB=portfolio_db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -d postgres:14
```

### Lỗi: "Module not found"

```powershell
# Cài đặt lại dependencies
pip install -r requirements.txt
```

### Lỗi: "Rate limit exceeded" - VNStock

Chờ vài phút và chạy lại, hoặc tăng delay:

```python
# Trong vnstock_fetcher.py, tăng delay từ 0.5 lên 1.0
delay=1.0
```

## 📁 Cấu trúc Files

```
Portfolio-v1/
├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── csv_reader.py              # Đọc CSV
│   │   ├── vnstock_fetcher.py         # API VNStock
│   │   ├── postgres_connector.py      # PostgreSQL
│   │   ├── pipeline_orchestrator.py   # Main pipeline
│   │   ├── run_quick.py              # Quick start
│   │   └── README.md  
├── scripts/
│   └── dashboard.py                   # Streamlit app
├── data/
│   └── company_info.csv              # Danh sách công ty
├── .env                               # Environment vars
└── requirements.txt                   # Dependencies
```

## ⚡ Commands Cheat Sheet
# Test import
python data_pipeline/test_pipeline.py
# Test quick (10 stocks)
python data_pipeline/run_pipeline.py

# Test command line
python data_pipeline/pipeline_orchestrator.py --test --num-stocks 10

# Full pipeline
python data_pipeline/pipeline_orchestrator.py

# Run tests
python data_pipeline/test_pipeline.py

# Dashboard
streamlit run scripts/dashboard.py


## 💡 Tips

1. **Lần đầu chạy**: Dùng TEST MODE để kiểm tra
2. **Production**: Dùng FULL MODE để có đủ dữ liệu
3. **Update data**: Chạy lại pipeline với date range mới
4. **Performance**: Chạy vào thời gian ít người dùng API

## 🎯 Next Steps

1. ✅ Chạy test pipeline thành công
2. ✅ Kiểm tra dashboard hoạt động
3. 📊 Explore data và tạo insights
4. 🚀 Deploy lên server (optional)

Happy Analyzing! 📈
