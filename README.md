## Portfolio – Dashboard tối ưu hóa danh mục cổ phiếu

Ứng dụng Streamlit hỗ trợ nhà đầu tư chứng khoán Việt Nam: phân tích thị trường, lọc cổ phiếu, tối ưu hóa danh mục (Markowitz, Sharpe, HRP...), backtest và tích hợp trợ lý AI.

---

## 1. Yêu cầu hệ thống

- **Python 3.11+**
- **Quản lý gói**: `pip` hoặc `uv` (khuyến nghị)
- **Kết nối Internet**

---

## 2. Cài đặt

### Clone dự án

```bash
git clone https://github.com/HieuPC1101/Portfolio-v1.git
cd Portfolio-v1
```

### Cài đặt thư viện

**Cách 1: Dùng UV (Nhanh, khuyến nghị)**

```bash
pip install uv
uv sync
```

**Cách 2: Dùng Pip**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Unix: source .venv/bin/activate
pip install -r requirements.txt
```

### Cấu hình API (cho Chatbot)

Tạo file `scripts/secret_config.py`:

```python
GEMINI_API_KEY = "your-api-key"
```

---

## 3. Chạy ứng dụng

**Với UV:**

```bash
uv run streamlit run scripts/dashboard.py
```

**Với Pip:**

```bash
streamlit run scripts/dashboard.py
```

Truy cập: `http://localhost:8501`

---

## 4. Các tính năng chính

1. **Tổng quan Thị trường**: Theo dõi VNINDEX, VN30, HNX, dòng tiền khối ngoại và hiệu suất ngành.
2. **Tự chọn mã cổ phiếu**: Lọc theo ngành/sàn, chọn mã thủ công và chạy tối ưu hóa.
3. **Đề xuất tự động**: Hệ thống gợi ý mã cổ phiếu dựa trên tiêu chí lợi nhuận hoặc rủi ro.
4. **Tổng hợp & So sánh**: So sánh hiệu quả giữa các mô hình (Sharpe, Risk, Return) và backtest.
5. **Tin tức**: Cập nhật tin tức tài chính mới nhất.
6. **Trợ lý AI**: Chatbot tư vấn phân tích thị trường sử dụng Gemini API.

---

## 5. Các mô hình tối ưu hóa

Ứng dụng hỗ trợ các mô hình từ thư viện `PyPortfolioOpt`:

- **Markowitz (Mean-Variance)**: Cân bằng lợi nhuận/rủi ro.
- **Max Sharpe**: Tối đa hóa tỷ lệ Sharpe.
- **Min Volatility**: Tối thiểu hóa rủi ro (độ lệch chuẩn).
- **Min CVaR / CDaR**: Tối thiểu hóa rủi ro đuôi và drawdown.
- **HRP**: Phân bổ rủi ro theo cấu trúc phân cấp (Hierarchical Risk Parity).

---

## 6. Công nghệ sử dụng

- **Core**: Python 3.11, Streamlit
- **Data & Math**: Pandas, Numpy, Scipy, Statsmodels
- **Finance**: Vnstock, Vnai, PyPortfolioOpt
- **Viz**: Plotly
- **AI**: Google Generative AI (Gemini)

---

## 7. Luồng hoạt động

**Quy trình cơ bản:**

1. **Khởi tạo**: Load data và session state.
2. **Input**: Người dùng chọn cổ phiếu (Thủ công hoặc Tự động).
3. **Xử lý**: Lấy dữ liệu giá -> Tính toán Metrics -> Chạy Video mô hình tối ưu hóa.
4. **Kết quả**: Hiển thị bảng phân bổ vốn, biểu đồ hiệu quả và kết quả Backtest.

**Kiến trúc:**
`Dashboard (UI)` -> `Data Layer` (vnstock) -> `Model Layer` (PyPortfolioOpt) -> `Result`

---
