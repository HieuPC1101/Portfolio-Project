# Portfolio-v1

Ứng dụng tối ưu hóa danh mục đầu tư chứng khoán Việt Nam với Data Pipeline tự động và Dashboard phân tích.

## 🎯 Giới thiệu
Portfolio-v1 là một hệ thống hoàn chỉnh hỗ trợ nhà đầu tư phân tích, tối ưu hóa và quản lý danh mục đầu tư chứng khoán. Ứng dụng tích hợp:
- **Data Pipeline tự động**: CSV → VNStock API → PostgreSQL → Dashboard
- **Mô hình toán học**: Tối ưu hóa danh mục theo các tiêu chí khác nhau
- **Phân tích kỹ thuật**: Các chỉ báo và patterns
- **AI Chatbot**: Tư vấn đầu tư thông minh
- **Giao diện trực quan**: Dashboard tương tác với Streamlit

## ✨ Tính năng chính

### 📊 Data Pipeline (MỚI!)
- **Tự động hóa hoàn toàn**: Thu thập và xử lý dữ liệu từ CSV → API → Database
- **VNStock API Integration**: Lấy dữ liệu giá lịch sử cho 500+ cổ phiếu
- **PostgreSQL Database**: Lưu trữ dữ liệu có cấu trúc, tối ưu cho queries
- **Error Handling**: Retry logic, rate limiting, validation
- **Performance**: Batch processing, indexing, caching

### 📈 Phân tích & Tối ưu hóa
- **Thu thập dữ liệu**: Tự động từ VNStock API
- **Phân tích thị trường & ngành**: Tổng quan, heatmap, drill-down
- **Tối ưu hóa danh mục**: Markowitz, Max Sharpe, Min Volatility, Min CVaR, Min CDaR, HRP
- **Phân tích kỹ thuật**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Backtesting**: Kiểm tra hiệu quả chiến lược

### 🤖 AI & Automation
- **Chatbot AI**: Tư vấn đầu tư, giải thích chỉ số
- **Session Management**: Lưu trữ trạng thái làm việc
- **Scheduled Updates**: Cập nhật dữ liệu tự động (coming soon)

## 🚀 Quick Start

### Cách 1: Setup tự động (Khuyến nghị)

```powershell
# 1. Clone project
git clone https://github.com/HieuPC1101/Portfolio-v1.git
cd Portfolio-v1

# 2. Chạy setup (cài packages + PostgreSQL)
.\setup.ps1

# 3. Chạy pipeline (interactive)
python scripts/data_pipeline/run_quick.py
# Chọn mode 1 (TEST) cho lần đầu

# 4. Chạy dashboard
streamlit run scripts/dashboard.py
```

### Cách 2: Manual setup

```powershell
# 1. Clone và install
git clone https://github.com/HieuPC1101/Portfolio-v1.git
cd Portfolio-v1
pip install -r requirements.txt

# 2. Setup PostgreSQL (Docker)
docker run --name portfolio-postgres `
  -e POSTGRES_DB=portfolio_db `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=postgres `
  -p 5432:5432 `
  -d postgres:14

# 3. Copy và config .env
cp .env.example .env
# Edit .env với thông tin database

# 4. Run pipeline test
python data_pipeline/pipeline_orchestrator.py --test --num-stocks 10

# 5. Run dashboard
streamlit run scripts/dashboard.py
```

### 📖 Chi tiết hơn
- **Quick Guide**: [QUICKSTART.md](QUICKSTART.md)
- **Pipeline Docs**: [scripts/data_pipeline/README.md](scripts/data_pipeline/README.md)
- **Implementation**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
4. Chạy ứng dụng:
	```powershell
	streamlit run scripts/dashboard.py

## Yêu cầu hệ thống
- Python >= 3.8
- Kết nối Internet để lấy dữ liệu từ API

## Tài liệu & Hướng dẫn sử dụng
1. Chạy ứng dụng và truy cập giao diện web Streamlit.
2. Chọn các tham số phân tích, danh mục cổ phiếu, mô hình tối ưu hóa.
3. Xem kết quả phân tích, biểu đồ, backtest và xuất danh mục đầu tư.


