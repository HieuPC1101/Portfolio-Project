## Portfolio – Dashboard tối ưu hóa danh mục cổ phiếu

Ứng dụng Streamlit hỗ trợ nhà đầu tư chứng khoán Việt Nam:

- Phân tích tổng quan thị trường & ngành (VNINDEX, VN30, HNX, UPCoM…)
- Lọc & chọn cổ phiếu theo sàn/ngành, tự chọn hoặc đề xuất tự động
- Chạy nhiều mô hình tối ưu hóa danh mục (Markowitz, Max Sharpe, Min Vol, Min CVaR, Min CDaR, HRP)
- Backtest hiệu quả danh mục trong khoảng thời gian phân tích
- So sánh kết quả giữa các mô hình và gợi ý phân bổ số lượng cổ phiếu thực tế
- Tích hợp tab tin tức & trợ lý AI (chatbot) hỗ trợ phân tích.

---

## 1. Yêu cầu hệ thống

- Python 
- Hệ điều hành: Windows
- Kết nối Internet (để lấy dữ liệu thị trường & tin tức)

---

## 2. Cài đặt

1. Clone hoặc tải mã nguồn về máy:

```bash
git clone https://github.com/HieuPC1101/Portfolio-v1.git
cd Portfolio-v1
```

2. Tạo môi trường ảo (khuyến nghị):

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux
```

3. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

4. Cấu hình khóa API (nếu dùng chatbot AI):

- Tạo file `scripts/utils/secret_config.py` (hoặc chỉnh sửa nếu đã có) dựa trên `scripts/secret_config_example.py`.
- Điền API key tương ứng (ví dụ: Google Generative AI…).

---

## 3. Chạy ứng dụng

Trong thư mục gốc dự án (`Portfolio-v1`), chạy:

```bash
streamlit run scripts/dashboard.py
```


## 4. Các tab chính trong ứng dụng

Thanh sidebar bên trái cho phép chuyển giữa các tab:

1. **Tổng quan Thị trường & Ngành**  
	- Bảng điều hành thị trường realtime: VN-Index, VN30, HNX, UPCoM…  
	- So sánh hiệu suất các chỉ số, hiệu suất ngành (1W, 1M)  
	- Treemap vốn hóa, dòng tiền khối ngoại, ma trận tương quan.

2. **Tự chọn mã cổ phiếu**  
	- Lọc theo **sàn giao dịch** và **ngành** từ file `data/company_info.csv`.  
	- Chọn thủ công các mã muốn đầu tư, chọn khoảng thời gian phân tích.  
	- Xem biểu đồ giá (line / candlestick) và chạy từng mô hình tối ưu hóa.

3. **Hệ thống đề xuất mã cổ phiếu tự động**  
	- Chọn **nhiều sàn** và **nhiều ngành** cùng lúc.  
	- Chọn số lượng cổ phiếu mỗi ngành, tiêu chí lọc: *Lợi nhuận lớn nhất* hoặc *Rủi ro bé nhất*.  
	- Hệ thống đề xuất danh sách mã, sau đó bạn có thể thêm vào danh mục để tối ưu.

4. **Tổng hợp Kết quả Tối ưu hóa**  
	- Hiển thị lại các kết quả đã chạy (theo 2 mode: *Tự chọn* / *Đề xuất tự động*).  
	- So sánh lợi nhuận kỳ vọng, rủi ro, Sharpe, phân bổ cổ phiếu giữa các mô hình.

5. **Tin tức Thị trường & Phân tích**  
	- Tổng hợp tin tức tài chính/chứng khoán, phục vụ đọc nhanh và tham khảo.

6. **Trợ lý AI**  
	- Chatbot hỗ trợ trả lời câu hỏi về thị trường, cổ phiếu, hoặc giải thích kết quả phân tích.  
	- Cần cấu hình API key trong `utils/secret_config.py`.

---

## 5. Các mô hình tối ưu hóa danh mục

Ứng dụng sử dụng thư viện `PyPortfolioOpt` và một số tối ưu hóa bổ sung:

- **Markowitz (Mean-Variance)** – Tối ưu hóa giữa lợi nhuận và rủi ro, tạo đường biên hiệu quả.  
- **Max Sharpe Ratio** – Tối đa hóa tỷ lệ Sharpe, so sánh với danh mục ngẫu nhiên.  
- **Min Volatility** – Giảm thiểu độ lệch chuẩn danh mục, so sánh với Max Sharpe.  
- **Min CVaR** – Tối thiểu hóa rủi ro tổn thất cực đoan (Conditional Value at Risk).  
- **Min CDaR** – Tối thiểu hóa rủi ro drawdown kéo dài (Conditional Drawdown at Risk).  
- **HRP (Hierarchical Risk Parity)** – Phân bổ rủi ro theo cấu trúc phân cấp, phù hợp danh mục nhiều mã.

Sau khi tối ưu, hệ thống còn:

- Chuyển trọng số lý thuyết thành **số lượng cổ phiếu nguyên** gần nhất phù hợp số tiền đầu tư.  
- Thực hiện **backtest** hiệu quả danh mục trong khoảng thời gian cấu hình (`ANALYSIS_START_DATE`, `ANALYSIS_END_DATE` trong `utils/config.py`).

---

## 6. Cấu trúc thư mục (rút gọn)

```text
Portfolio-v1/
│  README.md
│  requirements.txt
│
├─data/
│   └─ company_info.csv        # Thông tin mã, ngành, sàn giao dịch
├─scripts/
│   ├─ dashboard.py            # Entry chính của ứng dụng Streamlit
│   ├─ portfolio_models.py     # Các mô hình tối ưu hóa danh mục
│   ├─ auto_optimization.py    # Chạy tất cả mô hình & so sánh
│   ├─ news_tab.py             # Tab tin tức
│   ├─ optimization_comparison.py
│   ├─ chatbot/                # Chatbot & tích hợp AI
│   ├─ data_process/           # Lấy & xử lý dữ liệu giá
│   ├─ ui/                     # Biểu đồ và component giao diện
│   └─ utils/                  # Cấu hình, session state, secret config…
```

---


