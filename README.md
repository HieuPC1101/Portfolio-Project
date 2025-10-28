# Dashboard Hỗ Trợ Tối Ưu Hóa Danh Mục Đầu Tư Chứng Khoán

## Giới thiệu

Đây là một dự án Python sử dụng Streamlit để xây dựng dashboard hỗ trợ tối ưu hóa danh mục đầu tư chứng khoán. Ứng dụng cho phép người dùng lựa chọn cổ phiếu thủ công hoặc sử dụng hệ thống đề xuất tự động, thực hiện các mô hình tối ưu hóa hiện đại (Markowitz, Max Sharpe, Min Volatility, Min CVaR, Min CDaR, HRP), và trực quan hóa kết quả cũng như backtest hiệu suất danh mục.

## Tính năng chính
- **Lọc và chọn cổ phiếu** theo sàn giao dịch, ngành nghề, mã chứng khoán từ file dữ liệu `company_info.csv`.
- **Tối ưu hóa danh mục** với nhiều mô hình hiện đại:
  - Markowitz (Mean-Variance)
  - Max Sharpe Ratio
  - Min Volatility
  - Min CVaR (Conditional Value at Risk)
  - Min CDaR (Conditional Drawdown at Risk)
  - HRP (Hierarchical Risk Parity)
- **Đề xuất cổ phiếu tự động** theo tiêu chí lợi nhuận/rủi ro và số lượng mong muốn cho từng ngành.
- **Backtesting** hiệu suất danh mục và so sánh với các chỉ số benchmark (VNINDEX, VN30, HNX30, HNXINDEX).
- **Biểu đồ tương tác** với Plotly và giao diện thân thiện với Streamlit.

## Cấu trúc thư mục
```
├── config/
│   ├── __init__.py
│   └── db_config.py
├── data/
│   └── company_info.csv
├── scripts/
│   ├── __init__.py
│   ├── dashboard.py  # File chính chạy dashboard
│   └── data_collection.py
├── Chạy dash.txt
```

## Hướng dẫn cài đặt
1. **Cài đặt Python 3.8+**

2. **Cài đặt các thư viện cần thiết:**
   
   Bạn có thể cài đặt nhanh tất cả các thư viện bằng file `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Hoặc cài đặt thủ công từng thư viện:

   ```bash
   pip install streamlit pandas plotly numpy vnstock pypfopt scipy
   ```

3. **Chuẩn bị dữ liệu:**
   - Đảm bảo file `data/company_info.csv` chứa thông tin các mã cổ phiếu, ngành, sàn giao dịch.

4. **Chạy ứng dụng:**
   
   ```bash
   streamlit run scripts/dashboard.py
   ```

## Sử dụng
- **Chủ động:**
  - Lọc và chọn cổ phiếu theo sàn/ngành/mã.
  - Thêm vào danh mục, chọn thời gian, chạy tối ưu hóa và xem kết quả.
- **Tự động:**
  - Chọn sàn/ngành, số lượng cổ phiếu mỗi ngành, tiêu chí lọc.
  - Đề xuất cổ phiếu, thêm vào danh mục, tối ưu hóa và backtest.

