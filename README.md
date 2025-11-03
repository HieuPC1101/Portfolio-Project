# DanhMucDauTu

Ứng dụng hỗ trợ tối ưu hóa danh mục đầu tư chứng khoán Việt Nam, xây dựng trên nền tảng Streamlit.

## Tính năng chính

- **Tối ưu hóa danh mục đầu tư**: Sử dụng các mô hình Markowitz, Max Sharpe, Min Volatility, Min CVaR, Min CDaR, HRP.
- **Phân tích thị trường & ngành**: Tổng quan thị trường, hiệu suất ngành, lọc cổ phiếu theo nhiều tiêu chí.
- **Dự báo giá cổ phiếu**: Áp dụng các mô hình chuỗi thời gian (ARIMA, SARIMAX, Holt-Winters).
- **Phân tích kỹ thuật**: Tính toán và hiển thị các chỉ báo như SMA, EMA, RSI, MACD, Bollinger Bands.
- **Giao diện trực quan**: Streamlit dashboard, bộ lọc, hiển thị danh sách cổ phiếu, thao tác thêm/xóa linh hoạt.
- **Quản lý dữ liệu**: Đọc dữ liệu từ file CSV, lấy dữ liệu giá cổ phiếu từ API, cache thông tin.

## Công nghệ sử dụng

- Python, Streamlit, Pandas, Plotly, Numpy, vnstock, pypfopt, scipy, pandas-ta, statsmodels

## Cấu trúc thư mục

- `scripts/`: Chứa các module chức năng (dashboard, data_loader, portfolio_models, forecasting_models, market_overview, visualization, ui_components, config)
- `data/`: Dữ liệu đầu vào (company_info.csv)
- `config/`: Cấu hình bổ sung

## Cách sử dụng

1. Cài đặt các package cần thiết:
   ```
   pip install -r requirements.txt
   ```
2. Chạy ứng dụng:
   ```
   streamlit run scripts/dashboard.py
   ```
3. Tùy chỉnh các tham số phân tích trong `scripts/config.py` nếu cần.

