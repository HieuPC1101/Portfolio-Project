"""
Dashboard chính - Ứng dụng Streamlit hỗ trợ tối ưu hóa danh mục đầu tư chứng khoán.
File này import các module đã được tách riêng để dễ quản lý và bảo trì.
"""

import warnings
# Tắt cảnh báo pkg_resources deprecated từ thư viện vnai
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

import pandas as pd
import streamlit as st
import datetime
import sys
import os

# Thêm đường dẫn để import các module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import cấu hình
from scripts.config import ANALYSIS_START_DATE, ANALYSIS_END_DATE, DEFAULT_MARKET, DEFAULT_INVESTMENT_AMOUNT

# Import các module đã tách
from scripts.data_loader import (
    fetch_data_from_csv,
    fetch_stock_data2,
    get_latest_prices,
    calculate_metrics,
    fetch_fundamental_data_batch
)
from scripts.portfolio_models import (
    markowitz_optimization,
    max_sharpe,
    min_volatility,
    min_cvar,
    min_cdar,
    hrp_model
)
from scripts.visualization import (
    plot_interactive_stock_chart,
    plot_interactive_stock_chart_with_indicators,
    plot_stock_chart_with_forecast,
    plot_efficient_frontier,
    display_results,
    backtest_portfolio
)
from scripts.forecasting_models import get_forecast
from scripts.ui_components import (
    display_selected_stocks,
    display_selected_stocks_2
)
from scripts.market_overview import (
    show_sector_overview_page
)
import scripts.data_loader as data_loader_module

# Đường dẫn đến file CSV
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
file_path = os.path.join(data_dir, "company_info.csv")

# Lấy dữ liệu từ file CSV
df = fetch_data_from_csv(file_path)

# Tạo session state để lưu mã cổ phiếu đã chọn
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'selected_stocks_2' not in st.session_state:
    st.session_state.selected_stocks_2 = []
if 'final_selected_stocks' not in st.session_state:
    st.session_state.final_selected_stocks = {}


def run_models(data):
    """
    Hàm xử lý các chiến lược tối ưu hóa danh mục và tích hợp backtesting tự động.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
    """
    if data.empty:
        st.error("Dữ liệu cổ phiếu bị thiếu hoặc không hợp lệ.")
        return
    
    st.sidebar.title("Chọn chiến lược đầu tư")
    total_investment = st.sidebar.number_input(
        "Nhập số tiền đầu tư (VND)", 
        min_value=1000, 
        value=1000000, 
        step=100000,
        key="number_input_2"
    )

    models = {
        "Tối ưu hóa giữa lợi nhuận và rủi ro": {
            "function": lambda d, ti: markowitz_optimization(d, ti, get_latest_prices),
            "original_name": "Mô hình Markowitz"
        },
        "Hiệu suất tối đa": {
            "function": lambda d, ti: max_sharpe(d, ti, get_latest_prices),
            "original_name": "Mô hình Max Sharpe Ratio"
        },
        "Đầu tư an toàn": {
            "function": lambda d, ti: min_volatility(d, ti, get_latest_prices),
            "original_name": "Mô hình Min Volatility"
        },
        "Đa dạng hóa thông minh": {
            "function": lambda d, ti: hrp_model(d, ti, get_latest_prices),
            "original_name": "Mô hình HRP"
        },
        "Phòng ngừa tổn thất cực đại": {
            "function": lambda d, ti: min_cvar(d, ti, get_latest_prices),
            "original_name": "Mô hình Min CVaR"
        },
        "Kiểm soát tổn thất kéo dài": {
            "function": lambda d, ti: min_cdar(d, ti, get_latest_prices),
            "original_name": "Mô hình Min CDaR"
        },
    }

    for strategy_name, model_details in models.items():
        if st.sidebar.button(f"Chiến lược {strategy_name}"):
            try:
                # Chạy mô hình tối ưu hóa
                result = model_details["function"](data, total_investment)
                if result:
                    # Hiển thị kết quả tối ưu hóa
                    display_results(model_details["original_name"], result)

                    # Vẽ đường biên hiệu quả cho mô hình Markowitz
                    if strategy_name == "Tối ưu hóa giữa lợi nhuận và rủi ro":
                        tickers = list(result["Trọng số danh mục"].keys())
                        plot_efficient_frontier(
                            result["ret_arr"],
                            result["vol_arr"],
                            result["sharpe_arr"],
                            result["all_weights"],
                            tickers,
                            result["max_sharpe_idx"],
                            list(result["Trọng số danh mục"].values())
                        )

                    # Lấy thông tin cổ phiếu và trọng số từ kết quả
                    symbols = list(result["Trọng số danh mục"].keys())
                    weights = list(result["Trọng số danh mục"].values())

                    # Chạy backtesting ngay sau tối ưu hóa
                    st.subheader("Kết quả Backtesting")
                    with st.spinner("Đang chạy Backtesting..."):
                        # Sử dụng cấu hình từ config
                        start_date = pd.to_datetime(ANALYSIS_START_DATE).date()
                        end_date = pd.to_datetime(ANALYSIS_END_DATE).date()
                        backtest_result = backtest_portfolio(
                            symbols, 
                            weights, 
                            start_date, 
                            end_date,
                            fetch_stock_data2
                        )

                        # Hiển thị kết quả backtesting
                        if backtest_result:
                            st.write(f"Mean Sharpe Ratio: {backtest_result['Sharpe Ratio']:.2f}")
                            st.write(f"Maximum Drawdown: {backtest_result['Maximum Drawdown']:.2%}")
                        else:
                            st.error("Không thể thực hiện Backtesting. Vui lòng kiểm tra dữ liệu đầu vào.")
                else:
                    st.error(f"Không thể chạy {strategy_name}.")
            except Exception as e:
                st.error(f"Lỗi khi chạy {strategy_name}: {e}")


def main_manual_selection():
    """
    Hàm chính cho chế độ tự chọn cổ phiếu.
    """
    st.title("Tối ưu hóa danh mục đầu tư")
    
    # Kiểm tra session state và lấy danh sách cổ phiếu đã chọn
    if "selected_stocks" in st.session_state and st.session_state.selected_stocks:
        selected_stocks = st.session_state.selected_stocks
        # Lấy dữ liệu giá cổ phiếu
        data, skipped_tickers = fetch_stock_data2(selected_stocks, start_date, end_date)

        if not data.empty:
            st.subheader("Giá cổ phiếu")
            
            # === THÊM UI CHỌN CHỈ BÁO KỸ THUẬT ===
            with st.expander("Chỉ báo kỹ thuật", expanded=False):
                st.markdown("*Chọn các chỉ báo kỹ thuật để hiển thị trên biểu đồ*")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    show_sma_20 = st.checkbox("SMA(20) - Đường trung bình động đơn giản 20 ngày", value=False)
                    show_sma_50 = st.checkbox("SMA(50) - Đường trung bình động đơn giản 50 ngày", value=False)
                    show_ema_20 = st.checkbox("EMA(20) - Đường trung bình động mũ 20 ngày", value=False)
                    show_ema_50 = st.checkbox("EMA(50) - Đường trung bình động mũ 50 ngày", value=False)
                
                with col2:
                    show_rsi = st.checkbox("RSI - Chỉ số sức mạnh tương đối", value=False)
                    show_macd = st.checkbox("MACD - Phân kỳ hội tụ trung bình động", value=False)
                    show_bb = st.checkbox("Bollinger Bands - Dải Bollinger", value=False)
            
            # Tạo danh sách chỉ báo được chọn
            selected_indicators = []
            if show_sma_20:
                selected_indicators.append('SMA_20')
            if show_sma_50:
                selected_indicators.append('SMA_50')
            if show_ema_20:
                selected_indicators.append('EMA_20')
            if show_ema_50:
                selected_indicators.append('EMA_50')
            if show_rsi:
                selected_indicators.append('RSI')
            if show_macd:
                selected_indicators.append('MACD')
            if show_bb:
                selected_indicators.append('BB')
            
            # === THÊM UI DỰ BÁO ===
            with st.expander("Dự báo giá cổ phiếu", expanded=False):
                st.markdown("*Chọn cổ phiếu và phương pháp dự báo*")
                
                # Chỉ hiển thị dự báo nếu có 1 cổ phiếu được chọn
                if len(selected_stocks) == 1:
                    enable_forecast = st.checkbox("Hiển thị dự báo", value=False)
                    
                    if enable_forecast:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            forecast_method = st.selectbox(
                                "Phương pháp dự báo",
                                ["auto", "arima", "exp_smoothing", "moving_average"],
                                format_func=lambda x: {
                                    "auto": "Tự động (ARIMA ưu tiên)",
                                    "arima": "ARIMA - Phân tích chuỗi thời gian",
                                    "exp_smoothing": "Exponential Smoothing",
                                    "moving_average": "Moving Average - Trung bình động"
                                }[x]
                            )
                        
                        with col2:
                            forecast_days = st.slider(
                                "Số ngày dự báo",
                                min_value=7,
                                max_value=90,
                                value=30,
                                step=7
                            )
                        
                        st.info("💡 **Lưu ý:** Dự báo chỉ là ước tính dựa trên dữ liệu lịch sử và không đảm bảo chính xác 100%.")
                else:
                    enable_forecast = False
                    if len(selected_stocks) > 1:
                        st.warning("⚠️ Dự báo chỉ khả dụng khi chọn 1 cổ phiếu. Hiện đang chọn nhiều cổ phiếu.")
                    else:
                        st.info("Vui lòng chọn ít nhất 1 cổ phiếu để sử dụng chức năng dự báo.")
            
            # Vẽ biểu đồ giá cổ phiếu
            if enable_forecast and len(selected_stocks) == 1:
                # Lấy dự báo
                ticker = selected_stocks[0]
                with st.spinner(f"Đang dự báo giá {ticker}..."):
                    forecast_result = get_forecast(
                        data, 
                        ticker, 
                        method=forecast_method, 
                        forecast_periods=forecast_days
                    )
                
                if forecast_result:
                    # Vẽ biểu đồ với dự báo
                    plot_stock_chart_with_forecast(data, ticker, forecast_result, selected_indicators)
                else:
                    st.error("Không thể tạo dự báo. Vui lòng thử phương pháp khác hoặc tăng khoảng thời gian dữ liệu.")
                    # Vẽ biểu đồ bình thường
                    if selected_indicators:
                        plot_interactive_stock_chart_with_indicators(data, selected_stocks, selected_indicators)
                    else:
                        plot_interactive_stock_chart(data, selected_stocks)
            else:
                # Vẽ biểu đồ bình thường
                if selected_indicators:
                    plot_interactive_stock_chart_with_indicators(data, selected_stocks, selected_indicators)
                else:
                    plot_interactive_stock_chart(data, selected_stocks)
            
            # Chạy các mô hình
            run_models(data)
        else:
            st.error("Dữ liệu cổ phiếu bị thiếu hoặc không có.")
    else:
        st.warning("Chưa có mã cổ phiếu nào trong danh mục. Vui lòng chọn mã cổ phiếu trước.")


def main_auto_selection():
    """
    Hàm chính cho chế độ đề xuất cổ phiếu tự động.
    """
    st.title("Tối ưu hóa danh mục đầu tư")
    
    # Kiểm tra session state và lấy danh sách cổ phiếu đã chọn
    if "selected_stocks_2" in st.session_state and st.session_state.selected_stocks_2:
        selected_stocks_2 = st.session_state.selected_stocks_2
        st.sidebar.title("Chọn thời gian tính toán")
        today = datetime.date.today()
        start_date_2 = st.sidebar.date_input(
            "Ngày bắt đầu", 
            value=pd.to_datetime(ANALYSIS_START_DATE).date(), 
            min_value=pd.to_datetime(ANALYSIS_START_DATE).date(),
            max_value=pd.to_datetime(ANALYSIS_END_DATE).date(),
            key="start_date_2"
        )
        end_date_2 = st.sidebar.date_input(
            "Ngày kết thúc", 
            value=pd.to_datetime(ANALYSIS_END_DATE).date(), 
            min_value=pd.to_datetime(ANALYSIS_START_DATE).date(),
            max_value=pd.to_datetime(ANALYSIS_END_DATE).date(),
            key="end_date_2"
        )
        
        # Kiểm tra ngày bắt đầu và ngày kết thúc
        if start_date_2 > today or end_date_2 > today:
            st.sidebar.error("Ngày bắt đầu và ngày kết thúc không được vượt quá ngày hiện tại.")
        elif start_date_2 > end_date_2:
            st.sidebar.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc.")
        else:
            st.sidebar.success("Ngày tháng hợp lệ.")
            
        # Lấy dữ liệu giá cổ phiếu
        data, skipped_tickers = fetch_stock_data2(selected_stocks_2, start_date_2, end_date_2)

        if not data.empty:
            st.subheader("Giá cổ phiếu")
            
            # === THÊM UI CHỌN CHỈ BÁO KỸ THUẬT ===
            with st.expander("Chỉ báo kỹ thuật", expanded=False):
                st.markdown("*Chọn các chỉ báo kỹ thuật để hiển thị trên biểu đồ*")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    show_sma_20_2 = st.checkbox("SMA(20) - Đường trung bình động đơn giản 20 ngày", value=False, key="sma20_2")
                    show_sma_50_2 = st.checkbox("SMA(50) - Đường trung bình động đơn giản 50 ngày", value=False, key="sma50_2")
                    show_ema_20_2 = st.checkbox("EMA(20) - Đường trung bình động mũ 20 ngày", value=False, key="ema20_2")
                    show_ema_50_2 = st.checkbox("EMA(50) - Đường trung bình động mũ 50 ngày", value=False, key="ema50_2")
                
                with col2:
                    show_rsi_2 = st.checkbox("RSI - Chỉ số sức mạnh tương đối", value=False, key="rsi_2")
                    show_macd_2 = st.checkbox("MACD - Phân kỳ hội tụ trung bình động", value=False, key="macd_2")
                    show_bb_2 = st.checkbox("Bollinger Bands - Dải Bollinger", value=False, key="bb_2")
            
            # Tạo danh sách chỉ báo được chọn
            selected_indicators_2 = []
            if show_sma_20_2:
                selected_indicators_2.append('SMA_20')
            if show_sma_50_2:
                selected_indicators_2.append('SMA_50')
            if show_ema_20_2:
                selected_indicators_2.append('EMA_20')
            if show_ema_50_2:
                selected_indicators_2.append('EMA_50')
            if show_rsi_2:
                selected_indicators_2.append('RSI')
            if show_macd_2:
                selected_indicators_2.append('MACD')
            if show_bb_2:
                selected_indicators_2.append('BB')
            
            # === THÊM UI DỰ BÁO ===
            with st.expander("🔮 Dự báo giá cổ phiếu", expanded=False):
                st.markdown("*Chọn cổ phiếu và phương pháp dự báo*")
                
                # Chỉ hiển thị dự báo nếu có 1 cổ phiếu được chọn
                if len(selected_stocks_2) == 1:
                    enable_forecast_2 = st.checkbox("Hiển thị dự báo", value=False, key="enable_forecast_2")
                    
                    if enable_forecast_2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            forecast_method_2 = st.selectbox(
                                "Phương pháp dự báo",
                                ["auto", "arima", "exp_smoothing", "moving_average"],
                                format_func=lambda x: {
                                    "auto": "Tự động (ARIMA ưu tiên)",
                                    "arima": "ARIMA - Phân tích chuỗi thời gian",
                                    "exp_smoothing": "Exponential Smoothing",
                                    "moving_average": "Moving Average - Trung bình động"
                                }[x],
                                key="forecast_method_2"
                            )
                        
                        with col2:
                            forecast_days_2 = st.slider(
                                "Số ngày dự báo",
                                min_value=7,
                                max_value=90,
                                value=30,
                                step=7,
                                key="forecast_days_2"
                            )
                        
                        st.info("💡 **Lưu ý:** Dự báo chỉ là ước tính dựa trên dữ liệu lịch sử và không đảm bảo chính xác 100%.")
                else:
                    enable_forecast_2 = False
                    if len(selected_stocks_2) > 1:
                        st.warning("⚠️ Dự báo chỉ khả dụng khi chọn 1 cổ phiếu. Hiện đang chọn nhiều cổ phiếu.")
                    else:
                        st.info("Vui lòng chọn ít nhất 1 cổ phiếu để sử dụng chức năng dự báo.")
            
            # Vẽ biểu đồ giá cổ phiếu
            if enable_forecast_2 and len(selected_stocks_2) == 1:
                # Lấy dự báo
                ticker = selected_stocks_2[0]
                with st.spinner(f"Đang dự báo giá {ticker}..."):
                    forecast_result_2 = get_forecast(
                        data, 
                        ticker, 
                        method=forecast_method_2, 
                        forecast_periods=forecast_days_2
                    )
                
                if forecast_result_2:
                    # Vẽ biểu đồ với dự báo
                    plot_stock_chart_with_forecast(data, ticker, forecast_result_2, selected_indicators_2)
                else:
                    st.error("Không thể tạo dự báo. Vui lòng thử phương pháp khác hoặc tăng khoảng thời gian dữ liệu.")
                    # Vẽ biểu đồ bình thường
                    if selected_indicators_2:
                        plot_interactive_stock_chart_with_indicators(data, selected_stocks_2, selected_indicators_2)
                    else:
                        plot_interactive_stock_chart(data, selected_stocks_2)
            else:
                # Vẽ biểu đồ bình thường
                if selected_indicators_2:
                    plot_interactive_stock_chart_with_indicators(data, selected_stocks_2, selected_indicators_2)
                else:
                    plot_interactive_stock_chart(data, selected_stocks_2)
            
            # Chạy các mô hình
            run_models(data)
        else:
            st.error("Dữ liệu cổ phiếu bị thiếu hoặc không có.")
    else:
        st.warning("Chưa có mã cổ phiếu nào trong danh mục. Vui lòng chọn mã cổ phiếu trước.")


# ========== GIAO DIỆN CHÍNH ==========

# Sidebar
st.sidebar.title("Lựa chọn phương thức")

# Tùy chọn giữa các chế độ
option = st.sidebar.radio(
    "Chọn phương thức", 
    ["Tổng quan Thị trường & Ngành", "Tự chọn cổ phiếu", "Hệ thống đề xuất cổ phiếu tự động"]
)

if option == "Tổng quan Thị trường & Ngành":
    # Hiển thị trang tổng quan ngành
    show_sector_overview_page(df, data_loader_module)

elif option == "Tự chọn cổ phiếu":
    # Giao diện người dùng để lọc từ file CSV
    st.title("Dashboard hỗ trợ tối ưu hóa danh mục đầu tư chứng khoán")
    
    # Sidebar
    st.sidebar.title("Bộ lọc và Cấu hình")
    
    # Bộ lọc theo sàn giao dịch (exchange)
    exchanges = df['exchange'].unique()
    default_index = list(exchanges).index(DEFAULT_MARKET) if DEFAULT_MARKET in exchanges else 0
    selected_exchange = st.sidebar.selectbox('Chọn sàn giao dịch', exchanges, index=default_index)

    # Lọc dữ liệu dựa trên sàn giao dịch đã chọn
    filtered_df = df[df['exchange'] == selected_exchange]

    # Bộ lọc theo loại ngành (icb_name)
    selected_icb_name = st.sidebar.selectbox('Chọn ngành', filtered_df['icb_name'].unique())

    # Lọc dữ liệu dựa trên ngành đã chọn
    filtered_df = filtered_df[filtered_df['icb_name'] == selected_icb_name]

    # === BỘ LỌC PHÂN TÍCH CƠ BẢN ===
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Bộ lọc phân tích cơ bản")
    
    # Checkbox để bật/tắt bộ lọc phân tích cơ bản
    enable_fundamental_filter = st.sidebar.checkbox("Bật bộ lọc cổ phiếu giá trị", value=False)
    
    if enable_fundamental_filter:
        st.sidebar.markdown("*Lọc cổ phiếu theo tiêu chí phân tích cơ bản*")
        
        # Bộ lọc P/E (Price to Earnings)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            pe_min = st.number_input("P/E tối thiểu", min_value=0.0, value=0.0, step=0.5, key="pe_min")
        with col2:
            pe_max = st.number_input("P/E tối đa", min_value=0.0, value=30.0, step=0.5, key="pe_max")
        
        # Bộ lọc P/B (Price to Book)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            pb_min = st.number_input("P/B tối thiểu", min_value=0.0, value=0.0, step=0.1, key="pb_min")
        with col2:
            pb_max = st.number_input("P/B tối đa", min_value=0.0, value=3.0, step=0.1, key="pb_max")
        
        # Bộ lọc ROE (Return on Equity)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            roe_min = st.number_input("ROE tối thiểu (%)", min_value=0.0, value=10.0, step=1.0, key="roe_min")
        with col2:
            roe_max = st.number_input("ROE tối đa (%)", min_value=0.0, value=100.0, step=1.0, key="roe_max")
        
        # Bộ lọc ROA (Return on Assets)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            roa_min = st.number_input("ROA tối thiểu (%)", min_value=0.0, value=5.0, step=1.0, key="roa_min")
        with col2:
            roa_max = st.number_input("ROA tối đa (%)", min_value=0.0, value=100.0, step=1.0, key="roa_max")
        
        # Bộ lọc biên lợi nhuận (Profit Margin)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            margin_min = st.number_input("Biên lợi nhuận tối thiểu (%)", min_value=0.0, value=5.0, step=1.0, key="margin_min")
        with col2:
            margin_max = st.number_input("Biên lợi nhuận tối đa (%)", min_value=0.0, value=100.0, step=1.0, key="margin_max")
        
        # Bộ lọc EPS (Earnings per Share)
        eps_min = st.sidebar.number_input("EPS tối thiểu (nghìn VND)", min_value=0.0, value=1000.0, step=100.0, key="eps_min")
        
        # Nút áp dụng bộ lọc
        if st.sidebar.button("🔍 Áp dụng bộ lọc phân tích cơ bản"):
            with st.spinner("Đang lấy dữ liệu phân tích cơ bản..."):
                symbols_to_filter = filtered_df['symbol'].tolist()
                fundamental_df = fetch_fundamental_data_batch(symbols_to_filter)
                
                if not fundamental_df.empty:
                    # Áp dụng các bộ lọc
                    filtered_fundamental = fundamental_df.copy()
                    
                    # Lọc P/E
                    if 'pe' in filtered_fundamental.columns:
                        filtered_fundamental = filtered_fundamental[
                            (filtered_fundamental['pe'].notna()) &
                            (filtered_fundamental['pe'] >= pe_min) & 
                            (filtered_fundamental['pe'] <= pe_max)
                        ]
                    
                    # Lọc P/B
                    if 'pb' in filtered_fundamental.columns:
                        filtered_fundamental = filtered_fundamental[
                            (filtered_fundamental['pb'].notna()) &
                            (filtered_fundamental['pb'] >= pb_min) & 
                            (filtered_fundamental['pb'] <= pb_max)
                        ]
                    
                    # Lọc ROE
                    if 'roe' in filtered_fundamental.columns:
                        filtered_fundamental = filtered_fundamental[
                            (filtered_fundamental['roe'].notna()) &
                            (filtered_fundamental['roe'] >= roe_min) & 
                            (filtered_fundamental['roe'] <= roe_max)
                        ]
                    
                    # Lọc ROA
                    if 'roa' in filtered_fundamental.columns:
                        filtered_fundamental = filtered_fundamental[
                            (filtered_fundamental['roa'].notna()) &
                            (filtered_fundamental['roa'] >= roa_min) & 
                            (filtered_fundamental['roa'] <= roa_max)
                        ]
                    
                    # Lọc biên lợi nhuận
                    if 'profit_margin' in filtered_fundamental.columns:
                        filtered_fundamental = filtered_fundamental[
                            (filtered_fundamental['profit_margin'].notna()) &
                            (filtered_fundamental['profit_margin'] >= margin_min) & 
                            (filtered_fundamental['profit_margin'] <= margin_max)
                        ]
                    
                    # Lọc EPS
                    if 'eps' in filtered_fundamental.columns:
                        filtered_fundamental = filtered_fundamental[
                            (filtered_fundamental['eps'].notna()) &
                            (filtered_fundamental['eps'] >= eps_min)
                        ]
                    
                    # Lưu vào session state
                    st.session_state.filtered_fundamental = filtered_fundamental
                    st.sidebar.success(f"✓ Đã lọc được {len(filtered_fundamental)} cổ phiếu đáp ứng tiêu chí")
                else:
                    st.sidebar.error("Không thể lấy dữ liệu phân tích cơ bản")
        
        # Hiển thị kết quả lọc
        if 'filtered_fundamental' in st.session_state and not st.session_state.filtered_fundamental.empty:
            st.subheader(" Kết quả lọc cổ phiếu giá trị")
            display_df = st.session_state.filtered_fundamental.copy()
            
            # Format các cột để dễ đọc
            if 'pe' in display_df.columns:
                display_df['P/E'] = display_df['pe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            if 'pb' in display_df.columns:
                display_df['P/B'] = display_df['pb'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            if 'eps' in display_df.columns:
                display_df['EPS'] = display_df['eps'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
            if 'roe' in display_df.columns:
                display_df['ROE (%)'] = display_df['roe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            if 'roa' in display_df.columns:
                display_df['ROA (%)'] = display_df['roa'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            if 'profit_margin' in display_df.columns:
                display_df['Biên LN (%)'] = display_df['profit_margin'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            
            # Chọn các cột để hiển thị
            cols_to_display = ['symbol', 'P/E', 'P/B', 'EPS', 'ROE (%)', 'ROA (%)', 'Biên LN (%)']
            cols_to_display = [col for col in cols_to_display if col in display_df.columns]
            
            st.dataframe(display_df[cols_to_display], use_container_width=True)
            
            # Cho phép thêm các cổ phiếu đã lọc vào danh mục
            if st.button(" Thêm tất cả cổ phiếu đã lọc vào danh mục"):
                added_count = 0
                for symbol in st.session_state.filtered_fundamental['symbol'].tolist():
                    if symbol not in st.session_state.selected_stocks:
                        st.session_state.selected_stocks.append(symbol)
                        added_count += 1
                st.success(f"✓ Đã thêm {added_count} cổ phiếu vào danh mục!")
            
            # Cập nhật filtered_df để hiển thị trong multiselect
            filtered_df = filtered_df[filtered_df['symbol'].isin(st.session_state.filtered_fundamental['symbol'].tolist())]
    
    st.sidebar.markdown("---")
    
    # Bộ lọc theo mã chứng khoán (symbol)
    selected_symbols = st.sidebar.multiselect('Chọn mã chứng khoán', filtered_df['symbol'])

    # Lưu các mã chứng khoán đã chọn vào session state khi nhấn nút "Thêm mã"
    if st.sidebar.button("Thêm mã vào danh sách"):
        for symbol in selected_symbols:
            if symbol not in st.session_state.selected_stocks:
                st.session_state.selected_stocks.append(symbol)
        st.sidebar.success(f"Đã thêm {len(selected_symbols)} mã cổ phiếu vào danh mục!")

    # Hiển thị danh sách mã cổ phiếu đã chọn và xử lý thao tác xóa
    display_selected_stocks(df)

    # Lựa chọn thời gian lấy dữ liệu (sử dụng config mặc định)
    today = datetime.date.today()
    start_date = st.sidebar.date_input(
        "Ngày bắt đầu", 
        value=pd.to_datetime(ANALYSIS_START_DATE).date(), 
        max_value=today
    )
    end_date = st.sidebar.date_input(
        "Ngày kết thúc", 
        value=pd.to_datetime(ANALYSIS_END_DATE).date(), 
        max_value=today
    )
    
    # Kiểm tra ngày bắt đầu và ngày kết thúc
    if start_date > today or end_date > today:
        st.sidebar.error("Ngày bắt đầu và ngày kết thúc không được vượt quá ngày hiện tại.")
    elif start_date > end_date:
        st.sidebar.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc.")
    else:
        st.sidebar.success("Ngày tháng hợp lệ.")

    # Gọi hàm chính
    if __name__ == "__main__":
        main_manual_selection()

elif option == "Hệ thống đề xuất cổ phiếu tự động":
    # Giao diện Streamlit
    st.title("Hệ thống đề xuất cổ phiếu")
    st.sidebar.title("Cấu hình đề xuất cổ phiếu")

    # Bước 1: Chọn sàn giao dịch
    if not df.empty:
        selected_exchanges = st.sidebar.multiselect(
            "Chọn sàn giao dịch", 
            df['exchange'].unique(), 
            default=[DEFAULT_MARKET] if DEFAULT_MARKET in df['exchange'].unique() else None
        )

        # Lọc dữ liệu theo nhiều sàn giao dịch
        filtered_df = df[df['exchange'].isin(selected_exchanges)]

        # Bước 2: Chọn nhiều ngành
        selected_sectors = st.sidebar.multiselect("Chọn ngành", filtered_df['icb_name'].unique())

        if selected_sectors:
            # Lọc theo các ngành đã chọn
            sector_df = filtered_df[filtered_df['icb_name'].isin(selected_sectors)]

            # Bước 3: Chọn số lượng cổ phiếu cho từng ngành
            stocks_per_sector = {}
            for sector in selected_sectors:
                num_stocks = st.sidebar.number_input(
                    f"Số cổ phiếu muốn đầu tư trong ngành '{sector}'", 
                    min_value=1, 
                    max_value=10, 
                    value=3
                )
                stocks_per_sector[sector] = num_stocks

            # Bước 4: Chọn cách lọc
            filter_method = st.sidebar.radio("Cách lọc cổ phiếu", ["Lợi nhuận lớn nhất", "Rủi ro bé nhất", "Phân tích cơ bản (Cổ phiếu giá trị)"])

            # === BỘ LỌC PHÂN TÍCH CƠ BẢN CHO ĐỀ XUẤT TỰ ĐỘNG ===
            fundamental_filters = {}
            if filter_method == "Phân tích cơ bản (Cổ phiếu giá trị)":
                st.sidebar.markdown("---")
                st.sidebar.subheader("📊 Tiêu chí phân tích cơ bản")
                
                # Bộ lọc P/E
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    pe_min_auto = st.number_input("P/E tối thiểu", min_value=0.0, value=0.0, step=0.5, key="pe_min_auto")
                with col2:
                    pe_max_auto = st.number_input("P/E tối đa", min_value=0.0, value=20.0, step=0.5, key="pe_max_auto")
                
                # Bộ lọc P/B
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    pb_min_auto = st.number_input("P/B tối thiểu", min_value=0.0, value=0.0, step=0.1, key="pb_min_auto")
                with col2:
                    pb_max_auto = st.number_input("P/B tối đa", min_value=0.0, value=2.0, step=0.1, key="pb_max_auto")
                
                # Bộ lọc ROE
                roe_min_auto = st.sidebar.number_input("ROE tối thiểu (%)", min_value=0.0, value=15.0, step=1.0, key="roe_min_auto")
                
                # Bộ lọc ROA
                roa_min_auto = st.sidebar.number_input("ROA tối thiểu (%)", min_value=0.0, value=8.0, step=1.0, key="roa_min_auto")
                
                # Bộ lọc biên lợi nhuận
                margin_min_auto = st.sidebar.number_input("Biên lợi nhuận tối thiểu (%)", min_value=0.0, value=10.0, step=1.0, key="margin_min_auto")
                
                # Bộ lọc EPS
                eps_min_auto = st.sidebar.number_input("EPS tối thiểu (nghìn VND)", min_value=0.0, value=1000.0, step=100.0, key="eps_min_auto")
                
                fundamental_filters = {
                    'pe_min': pe_min_auto,
                    'pe_max': pe_max_auto,
                    'pb_min': pb_min_auto,
                    'pb_max': pb_max_auto,
                    'roe_min': roe_min_auto,
                    'roa_min': roa_min_auto,
                    'margin_min': margin_min_auto,
                    'eps_min': eps_min_auto
                }
                st.sidebar.markdown("---")

            # Lựa chọn thời gian lấy dữ liệu
            today = datetime.date.today()
            start_date = st.sidebar.date_input(
                "Ngày bắt đầu", 
                value=pd.to_datetime(ANALYSIS_START_DATE).date(),
                min_value=pd.to_datetime(ANALYSIS_START_DATE).date(),
                max_value=pd.to_datetime(ANALYSIS_END_DATE).date(),
                key="start_date_1"
            )
            end_date = st.sidebar.date_input(
                "Ngày kết thúc", 
                value=pd.to_datetime(ANALYSIS_END_DATE).date(),
                min_value=pd.to_datetime(ANALYSIS_START_DATE).date(),
                max_value=pd.to_datetime(ANALYSIS_END_DATE).date(),
                key="end_date_1"
            )
            
            # Kiểm tra ngày bắt đầu và ngày kết thúc
            if start_date > today or end_date > today:
                st.sidebar.error("Ngày bắt đầu và ngày kết thúc không được vượt quá ngày hiện tại.")
            elif start_date > end_date:
                st.sidebar.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc.")
            else:
                st.sidebar.success("Ngày tháng hợp lệ.")

            # Bộ lọc và xử lý nhiều sàn, nhiều ngành, và đề xuất cổ phiếu
            if st.sidebar.button("Đề xuất cổ phiếu"):
                final_selected_stocks = {}

                for exchange in selected_exchanges:
                    st.subheader(f"Sàn giao dịch: {exchange}")
                    exchange_df = df[df['exchange'] == exchange]

                    for sector, num_stocks in stocks_per_sector.items():
                        # Lọc cổ phiếu theo ngành trong từng sàn
                        sector_df = exchange_df[exchange_df['icb_name'] == sector]

                        if sector_df.empty:
                            st.warning(f"Không có cổ phiếu nào trong ngành '{sector}' của sàn '{exchange}' để phân tích.")
                            continue

                        symbols = sector_df['symbol'].tolist()

                        # Kéo dữ liệu giá cổ phiếu
                        data, skipped_tickers = fetch_stock_data2(symbols, start_date, end_date)

                        if data.empty:
                            st.warning(f"Không có dữ liệu giá cổ phiếu cho ngành '{sector}' của sàn '{exchange}'.")
                            continue

                        # Lọc cổ phiếu theo cách lọc
                        if filter_method == "Phân tích cơ bản (Cổ phiếu giá trị)":
                            # Lấy dữ liệu phân tích cơ bản
                            fundamental_df = fetch_fundamental_data_batch(symbols)
                            
                            if not fundamental_df.empty:
                                # Áp dụng các bộ lọc phân tích cơ bản
                                filtered_fundamental = fundamental_df.copy()
                                
                                # Lọc P/E
                                if 'pe' in filtered_fundamental.columns and fundamental_filters:
                                    filtered_fundamental = filtered_fundamental[
                                        (filtered_fundamental['pe'].notna()) &
                                        (filtered_fundamental['pe'] >= fundamental_filters['pe_min']) & 
                                        (filtered_fundamental['pe'] <= fundamental_filters['pe_max'])
                                    ]
                                
                                # Lọc P/B
                                if 'pb' in filtered_fundamental.columns and fundamental_filters:
                                    filtered_fundamental = filtered_fundamental[
                                        (filtered_fundamental['pb'].notna()) &
                                        (filtered_fundamental['pb'] >= fundamental_filters['pb_min']) & 
                                        (filtered_fundamental['pb'] <= fundamental_filters['pb_max'])
                                    ]
                                
                                # Lọc ROE
                                if 'roe' in filtered_fundamental.columns and fundamental_filters:
                                    filtered_fundamental = filtered_fundamental[
                                        (filtered_fundamental['roe'].notna()) &
                                        (filtered_fundamental['roe'] >= fundamental_filters['roe_min'])
                                    ]
                                
                                # Lọc ROA
                                if 'roa' in filtered_fundamental.columns and fundamental_filters:
                                    filtered_fundamental = filtered_fundamental[
                                        (filtered_fundamental['roa'].notna()) &
                                        (filtered_fundamental['roa'] >= fundamental_filters['roa_min'])
                                    ]
                                
                                # Lọc biên lợi nhuận
                                if 'profit_margin' in filtered_fundamental.columns and fundamental_filters:
                                    filtered_fundamental = filtered_fundamental[
                                        (filtered_fundamental['profit_margin'].notna()) &
                                        (filtered_fundamental['profit_margin'] >= fundamental_filters['margin_min'])
                                    ]
                                
                                # Lọc EPS
                                if 'eps' in filtered_fundamental.columns and fundamental_filters:
                                    filtered_fundamental = filtered_fundamental[
                                        (filtered_fundamental['eps'].notna()) &
                                        (filtered_fundamental['eps'] >= fundamental_filters['eps_min'])
                                    ]
                                
                                # Tính điểm tổng hợp cho từng cổ phiếu (Value Score)
                                # Điểm càng cao càng tốt (ROE cao, ROA cao, P/E thấp, P/B thấp, biên lợi nhuận cao)
                                if not filtered_fundamental.empty:
                                    filtered_fundamental['value_score'] = 0
                                    
                                    # ROE cao = tốt
                                    if 'roe' in filtered_fundamental.columns:
                                        filtered_fundamental['value_score'] += filtered_fundamental['roe'].fillna(0) / 10
                                    
                                    # ROA cao = tốt
                                    if 'roa' in filtered_fundamental.columns:
                                        filtered_fundamental['value_score'] += filtered_fundamental['roa'].fillna(0) / 10
                                    
                                    # P/E thấp = tốt (điểm càng cao khi P/E càng thấp)
                                    if 'pe' in filtered_fundamental.columns:
                                        max_pe = filtered_fundamental['pe'].max()
                                        if max_pe > 0:
                                            filtered_fundamental['value_score'] += (max_pe - filtered_fundamental['pe'].fillna(max_pe)) / max_pe * 10
                                    
                                    # P/B thấp = tốt
                                    if 'pb' in filtered_fundamental.columns:
                                        max_pb = filtered_fundamental['pb'].max()
                                        if max_pb > 0:
                                            filtered_fundamental['value_score'] += (max_pb - filtered_fundamental['pb'].fillna(max_pb)) / max_pb * 10
                                    
                                    # Biên lợi nhuận cao = tốt
                                    if 'profit_margin' in filtered_fundamental.columns:
                                        filtered_fundamental['value_score'] += filtered_fundamental['profit_margin'].fillna(0) / 10
                                    
                                    # Chọn top cổ phiếu theo điểm
                                    filtered_fundamental = filtered_fundamental.nlargest(num_stocks, 'value_score')
                                    selected_stocks = filtered_fundamental['symbol'].tolist()
                                    
                                    # Hiển thị thông tin chi tiết
                                    st.write(f"**Top {len(selected_stocks)} cổ phiếu giá trị trong ngành '{sector}':**")
                                    display_cols = ['symbol', 'pe', 'pb', 'roe', 'roa', 'profit_margin', 'value_score']
                                    display_cols = [col for col in display_cols if col in filtered_fundamental.columns]
                                    st.dataframe(filtered_fundamental[display_cols].round(2), use_container_width=True)
                                else:
                                    st.warning(f"Không có cổ phiếu nào trong ngành '{sector}' đáp ứng tiêu chí phân tích cơ bản.")
                                    selected_stocks = []
                            else:
                                st.warning(f"Không thể lấy dữ liệu phân tích cơ bản cho ngành '{sector}'.")
                                selected_stocks = []
                        else:
                            # Tính toán lợi nhuận kỳ vọng và phương sai
                            mean_returns, volatility = calculate_metrics(data)

                            # Tạo DataFrame kết quả
                            stock_analysis = pd.DataFrame({
                                "Mã cổ phiếu": mean_returns.index,
                                "Lợi nhuận kỳ vọng (%)": mean_returns.values * 100,
                                "Rủi ro (Phương sai)": volatility.values * 100
                            })

                            # Lọc cổ phiếu theo cách lọc và số lượng
                            if filter_method == "Lợi nhuận lớn nhất":
                                selected_stocks = stock_analysis.nlargest(num_stocks, "Lợi nhuận kỳ vọng (%)")["Mã cổ phiếu"].tolist()
                            elif filter_method == "Rủi ro bé nhất":
                                selected_stocks = stock_analysis.nsmallest(num_stocks, "Rủi ro (Phương sai)")["Mã cổ phiếu"].tolist()

                        # Lưu cổ phiếu được chọn theo sàn và ngành vào session_state
                        if exchange not in st.session_state.final_selected_stocks:
                            st.session_state.final_selected_stocks[exchange] = {}
                        st.session_state.final_selected_stocks[exchange][sector] = selected_stocks

    # Hiển thị danh mục cổ phiếu được lọc
    if st.session_state.final_selected_stocks:
        st.subheader("Danh mục cổ phiếu được lọc theo sàn và ngành")
        if st.button("Xóa hết các cổ phiếu đã được đề xuất"):
            st.session_state.final_selected_stocks = {}
            st.success("Đã xóa hết tất cả cổ phiếu khỏi danh sách!")
        
        for exchange, sectors in st.session_state.final_selected_stocks.items():
            st.write(f"### Sàn: {exchange}")
            for sector, stocks in sectors.items():
                st.write(f"#### Ngành: {sector}")
                for stock in stocks:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"- {stock}")
                    with col2:
                        if st.button("➕ Thêm", key=f"add_{exchange}_{sector}_{stock}"):
                            if stock not in st.session_state.selected_stocks_2:
                                st.session_state.selected_stocks_2.append(stock)
                                st.success(f"Đã thêm mã cổ phiếu '{stock}' vào danh sách.")
                            else:
                                st.warning(f"Mã cổ phiếu '{stock}' đã tồn tại trong danh sách.")

    # Hiển thị danh sách mã cổ phiếu đã chọn
    display_selected_stocks_2(df)

    # Gọi hàm chính
    if __name__ == "__main__":
        main_auto_selection()
