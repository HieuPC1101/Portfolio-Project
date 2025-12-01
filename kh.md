# Kế hoạch Chi tiết: Sửa lỗi Chấm điểm & Cải tiến Trực quan hóa

## Mô tả Mục tiêu
Cải tổ module so sánh danh mục đầu tư (`scripts/optimization_comparison.py`) để đảm bảo tính chính xác về mặt toán học trong việc chấm điểm và cung cấp các biểu đồ trực quan chuyên nghiệp.

## Yêu cầu Người dùng Xem xét
> [!IMPORTANT]
> **Thay đổi Logic Chấm điểm**: Công thức tính tổng thô hiện tại sẽ được thay thế bằng **Tổng Trọng số Chuẩn hóa** (thang 0-100).
>
> **Chỉ số Mới**: Thêm cột "Maximum Drawdown" (MDD) vào bảng so sánh.

## Các Thay đổi Đề xuất Chi tiết

### 1. Hệ thống Chấm điểm (`scripts/optimization_comparison.py`)

#### [SỬA ĐỔI] Hàm `provide_investment_recommendation`
Thay thế toàn bộ logic tính điểm hiện tại bằng quy trình chuẩn hóa sau:

1.  **Thu thập Dữ liệu Thô**: Tạo một list các dictionary chứa các chỉ số thô (Return, Sharpe, Volatility, Diversification, Capital Utilization) từ tất cả các mô hình hợp lệ.
2.  **Xác định Min/Max**: Tìm giá trị nhỏ nhất và lớn nhất cho từng chỉ số trong tập dữ liệu.
3.  **Chuẩn hóa (Normalization)**:
    *   **Công thức chung (Càng cao càng tốt)**:
        ```python
        score = ((value - min_val) / (max_val - min_val)) * 100
        ```
    *   **Công thức đảo ngược (Càng thấp càng tốt - Rủi ro)**:
        ```python
        score = ((max_val - value) / (max_val - min_val)) * 100
        ```
    *   *Xử lý ngoại lệ*: Nếu `max_val == min_val`, gán `score = 50` cho tất cả.
4.  **Tính Điểm Tổng hợp (Weighted Score)**:
    ```python
    TotalScore = (Norm_Sharpe * 0.4) + (Norm_Return * 0.3) + (Norm_Diversification * 0.2) + (Norm_Capital * 0.1)
    ```
5.  **Hiển thị**:
    *   Thêm `st.expander("Chi tiết bảng điểm")` hiển thị DataFrame chứa cả giá trị thô và điểm số đã chuẩn hóa của từng thành phần để minh bạch hóa cách tính.

### 2. Trực quan hóa (`scripts/optimization_comparison.py`)

#### [SỬA ĐỔI] Hàm `plot_radar_comparison`
Cải tiến thuật toán scaling để tránh biểu đồ bị méo mó:
*   **Baseline Scaling**: Thay vì scale 0-100 từ Min đến Max tuyệt đối, hãy thêm "padding" (khoảng đệm).
    ```python
    range_val = max_val - min_val
    padding = range_val * 0.1  # 10% đệm
    baseline_min = min_val - padding
    baseline_max = max_val + padding
    # Scale dựa trên baseline_min và baseline_max
    ```
*   Điều này giúp các mô hình có chỉ số gần nhau (ví dụ 10% và 10.1%) sẽ nằm gần nhau trên radar, thay vì một cái ở tâm (0) và một cái ở đỉnh (100).

#### [SỬA ĐỔI] Hàm `plot_allocation_comparison`
Chuyển đổi từ Pie Chart sang **Stacked Bar Chart** dùng `plotly.graph_objects`:
*   **Dữ liệu chuẩn bị**: Tạo DataFrame với index là Tên Mô hình, columns là Mã Cổ phiếu, values là Tỷ trọng (%).
*   **Vẽ biểu đồ**:
    ```python
    fig = go.Figure()
    for ticker in all_tickers:
        fig.add_trace(go.Bar(
            name=ticker,
            x=model_names,
            y=weights_of_ticker_across_models
        ))
    fig.update_layout(barmode='stack')
    ```
*   Giúp so sánh trực quan tỷ trọng của cùng một mã cổ phiếu giữa các mô hình khác nhau.

### 3. Bảng So sánh & Chỉ số (`scripts/optimization_comparison.py`)

#### [SỬA ĐỔI] Hàm `calculate_portfolio_metrics`
*   **Thêm Maximum Drawdown (MDD)**:
    *   Nếu có chuỗi lợi nhuận (`ret_arr` hoặc tương tự), tính MDD:
        ```python
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        mdd = drawdown.min()
        ```
    *   Nếu không có dữ liệu lịch sử chi tiết, trả về `NaN` hoặc dùng CDaR làm tham chiếu.

#### [SỬA ĐỔI] Hàm `create_comparison_table`
*   **Cấu trúc bảng mới**:
    *   Thêm cột: `Điểm số (Score)`, `Max Drawdown`.
*   **Định dạng (Styling)**:
    *   Sử dụng `df.style`:
        *   `.format("{:.2f}")`: Cho Sharpe, Diversification.
        *   `.format("{:.2f}%")`: Cho Return, Risk, MDD.
        *   `.background_gradient(cmap='RdYlGn')`: Cho cột Return, Sharpe, Score (Xanh tốt, Đỏ xấu).
        *   `.background_gradient(cmap='RdYlGn_r')`: Cho cột Risk, MDD (Đỏ cao xấu, Xanh thấp tốt).

## Kế hoạch Xác minh

1.  **Kiểm tra Logic Chấm điểm**: Tạo một test case giả định với 2 mô hình (1 rủi ro cao/lợi nhuận cao, 1 an toàn). Xác nhận rằng mô hình an toàn có điểm Rủi ro cao (tốt) và mô hình rủi ro có điểm Lợi nhuận cao.
2.  **Kiểm tra Biểu đồ**:
    *   Radar: Đảm bảo không có đỉnh nhọn bất thường khi dữ liệu đầu vào chênh lệch ít.
    *   Allocation: Đảm bảo tổng tỷ trọng mỗi cột là 100% (hoặc xấp xỉ).
3.  **Kiểm tra Giao diện**: Bảng so sánh hiển thị đẹp, dễ đọc, màu sắc heatmap hợp lý (Xanh = Tốt).
