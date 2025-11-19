# Portfolio-v1

Ứng dụng tối ưu hóa danh mục đầu tư chứng khoán Việt Nam với Data Pipeline tự động và Dashboard phân tích.

## 🎯 Giới thiệu
Portfolio-v1 là một hệ thống hoàn chỉnh hỗ trợ nhà đầu tư phân tích, tối ưu hóa và quản lý danh mục đầu tư chứng khoán. Ứng dụng tích hợp:
- **Data Pipeline tự động**: CSV → VNStock API → PostgreSQL → Dashboard
- **Mô hình toán học**: Tối ưu hóa danh mục theo các tiêu chí khác nhau
- **Phân tích kỹ thuật**: Các chỉ báo và patterns
- **AI Chatbot**: Tư vấn đầu tư thông minh
- **Giao diện trực quan**: Dashboard tương tác với Streamlit

## Tính năng chính
- **Thu thập dữ liệu**: Tự động lấy thông tin công ty, ngành, giá cổ phiếu từ các nguồn dữ liệu Việt Nam.
- **Phân tích thị trường & ngành**: Hiển thị tổng quan thị trường, heatmap, drill-down theo ngành/sàn.
- **Tối ưu hóa danh mục**: Hỗ trợ các mô hình Markowitz, Max Sharpe, Min Volatility, Min CVaR, Min CDaR, HRP.
- **Phân tích kỹ thuật**: Tính toán các chỉ báo như SMA, EMA, RSI, MACD, Bollinger Bands.
- **Backtesting**: Kiểm tra hiệu quả danh mục đầu tư qua các giai đoạn lịch sử.
- **Quản lý phiên làm việc**: Lưu trữ trạng thái, bộ lọc, danh sách cổ phiếu đã chọn.
- **Trợ lý AI (Chatbot)**: Hỗ trợ tư vấn đầu tư, giải thích các chỉ số tài chính và chiến lược thông qua AI.
- **Giao diện trực quan**: Streamlit + Plotly, dễ sử dụng, thao tác nhanh.

## Cài đặt
1. Clone dự án:
	```powershell
	git clone https://github.com/HieuPC1101/Portfolio-v1.git
	```
2. Cài đặt các thư viện Python:
	```powershell
	pip install -r requirements.txt
	```
3. Cấu hình OpenAI API Key (cho chatbot):
	- Mở file `scripts/config.py`
	- Thay thế `your-openai-api-key-here` bằng API key của bạn
	- Lấy API key tại: https://platform.openai.com/api-keys
4. Chạy ứng dụng:
	```powershell
	streamlit run scripts/dashboard.py
	```

## Yêu cầu hệ thống
- Python >= 3.8
- Kết nối Internet để lấy dữ liệu từ API

## Tài liệu & Hướng dẫn sử dụng
1. Chạy ứng dụng và truy cập giao diện web Streamlit.
2. Chọn các tham số phân tích, danh mục cổ phiếu, mô hình tối ưu hóa.
3. Xem kết quả phân tích, biểu đồ, backtest và xuất danh mục đầu tư.
4. **Sử dụng Chatbot AI**: 
   - Mở mục "Trợ lý AI" trong sidebar
   - Đặt câu hỏi về đầu tư, chiến lược, hoặc các chỉ số tài chính
   - Xem hướng dẫn chi tiết tại: [CHATBOT_GUIDE.md](CHATBOT_GUIDE.md)


