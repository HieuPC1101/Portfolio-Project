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
    fetch_fundamental_data_batch,
    fetch_ohlc_data
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
    plot_efficient_frontier,
    plot_max_sharpe_with_cal,
    plot_min_volatility_scatter,
    display_results,
    backtest_portfolio,
    plot_candlestick_chart,
    plot_min_cvar_analysis,
    plot_min_cdar_analysis,
    visualize_hrp_model
)
from scripts.ui_components import (
    display_selected_stocks,
    display_selected_stocks_2
)
from scripts.market_overview import (
    show_sector_overview_page
)
from scripts.session_manager import (
    initialize_session_state,
    save_manual_filter_state,
    save_manual_fundamental_filters,
    save_auto_filter_state,
    save_auto_fundamental_filters,
    get_manual_filter_state,
    get_manual_fundamental_filters,
    get_auto_filter_state,
    get_auto_fundamental_filters,
    update_current_tab,
    get_current_tab
)
import scripts.data_loader as data_loader_module

# Đường dẫn đến file CSV
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
file_path = os.path.join(data_dir, "company_info.csv")

# Lấy dữ liệu từ file CSV
df = fetch_data_from_csv(file_path)

# Khởi tạo session state khi ứng dụng khởi động
initialize_session_state()


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
    
    # Lấy số tiền đầu tư từ session state dựa trên tab hiện tại
    current_tab = get_current_tab()
    if current_tab == "Tự chọn mã cổ phiếu":
        default_investment = st.session_state.manual_investment_amount
        investment_key = "manual_investment_amount"
    else:
        default_investment = st.session_state.auto_investment_amount
        investment_key = "auto_investment_amount"
    
    total_investment = st.sidebar.number_input(
        "Nhập số tiền đầu tư (VND)", 
        min_value=1000, 
        value=default_investment, 
        step=100000,
        key=f"number_input_{investment_key}"
    )
    
    # Lưu số tiền đầu tư vào session state
    if current_tab == "Tự chọn mã cổ phiếu":
        st.session_state.manual_investment_amount = total_investment
    else:
        st.session_state.auto_investment_amount = total_investment

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
                    
                    # Vẽ biểu đồ Max Sharpe với đường CAL
                    elif strategy_name == "Hiệu suất tối đa":
                        tickers = list(result["Trọng số danh mục"].keys())
                        plot_max_sharpe_with_cal(
                            result["ret_arr"],
                            result["vol_arr"],
                            result["sharpe_arr"],
                            result["all_weights"],
                            tickers,
                            result["Lợi nhuận kỳ vọng"],
                            result["Rủi ro (Độ lệch chuẩn)"],
                            result.get("risk_free_rate", 0.04)
                        )
                    
                    # Vẽ biểu đồ Min Volatility với scatter plot
                    elif strategy_name == "Đầu tư an toàn":
                        tickers = list(result["Trọng số danh mục"].keys())
                        plot_min_volatility_scatter(
                            result["ret_arr"],
                            result["vol_arr"],
                            result["sharpe_arr"],
                            result["all_weights"],
                            tickers,
                            result["Lợi nhuận kỳ vọng"],
                            result["Rủi ro (Độ lệch chuẩn)"],
                            result.get("max_sharpe_return"),
                            result.get("max_sharpe_volatility"),
                            result.get("min_vol_weights"),
                            result.get("max_sharpe_weights"),
                            result.get("risk_free_rate", 0.02)
                        )
                    
                    # Vẽ biểu đồ phân tích Min CVaR
                    elif strategy_name == "Phòng ngừa tổn thất cực đại":
                        plot_min_cvar_analysis(result)
                    
                    # Vẽ biểu đồ phân tích Min CDaR
                    elif strategy_name == "Kiểm soát tổn thất kéo dài":
                        # Tính Max Sharpe để so sánh
                        max_sharpe_result = max_sharpe(data, total_investment, get_latest_prices)
                        # Tính returns data từ price data
                        returns_data = data.pct_change().dropna()
                        plot_min_cdar_analysis(result, max_sharpe_result, returns_data)
                    
                    # Vẽ biểu đồ phân tích HRP với Dendrogram
                    elif strategy_name == "Đa dạng hóa thông minh":
                        visualize_hrp_model(data, result)

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
                            pass  
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
        
        # Lấy trạng thái ngày đã lưu
        filter_state = get_manual_filter_state()
        default_start = filter_state.get('start_date') or pd.to_datetime(ANALYSIS_START_DATE).date()
        default_end = filter_state.get('end_date') or pd.to_datetime(ANALYSIS_END_DATE).date()
        
        # Lấy dữ liệu giá cổ phiếu (sử dụng start_date và end_date từ sidebar)
        data, skipped_tickers = fetch_stock_data2(selected_stocks, start_date, end_date)

        if not data.empty:
            st.subheader("Giá cổ phiếu")
            
            # === THÊM OPTION BIỂU ĐỒ NẾN ===
            show_candlestick = False
            if len(selected_stocks) == 1:
                # Lấy trạng thái đã lưu
                default_candlestick = st.session_state.manual_show_candlestick
                show_candlestick = st.checkbox(
                    "Hiển thị biểu đồ nến (Candlestick)", 
                    value=default_candlestick, 
                    key="candlestick_1"
                )
                # Lưu trạng thái
                st.session_state.manual_show_candlestick = show_candlestick
            
            # Vẽ biểu đồ giá cổ phiếu
            if show_candlestick and len(selected_stocks) == 1:
                # Hiển thị biểu đồ nến
                ticker = selected_stocks[0]
                with st.spinner(f"Đang tải dữ liệu OHLC cho {ticker}..."):
                    ohlc_data = fetch_ohlc_data(ticker, data_loader_module.ANALYSIS_START_DATE, data_loader_module.ANALYSIS_END_DATE)
                    if not ohlc_data.empty:
                        plot_candlestick_chart(ohlc_data, ticker)
                    else:
                        st.error("Không thể tải dữ liệu OHLC. Hiển thị biểu đồ đường thay thế.")
                        plot_interactive_stock_chart(data, selected_stocks)
            else:
                # Vẽ biểu đồ bình thường
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
        
        # Lấy trạng thái ngày đã lưu
        filter_state = get_auto_filter_state()
        default_start_2 = filter_state.get('start_date') or pd.to_datetime(ANALYSIS_START_DATE).date()
        default_end_2 = filter_state.get('end_date') or pd.to_datetime(ANALYSIS_END_DATE).date()
        
        start_date_2 = st.sidebar.date_input(
            "Ngày bắt đầu", 
            value=default_start_2, 
            min_value=pd.to_datetime(ANALYSIS_START_DATE).date(),
            max_value=pd.to_datetime(ANALYSIS_END_DATE).date(),
            key="start_date_2"
        )
        end_date_2 = st.sidebar.date_input(
            "Ngày kết thúc", 
            value=default_end_2, 
            min_value=pd.to_datetime(ANALYSIS_START_DATE).date(),
            max_value=pd.to_datetime(ANALYSIS_END_DATE).date(),
            key="end_date_2"
        )
        
        # Lưu trạng thái ngày
        if 'auto_filter_state' in st.session_state:
            st.session_state.auto_filter_state['start_date'] = start_date_2
            st.session_state.auto_filter_state['end_date'] = end_date_2
        
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
            
            # === THÊM OPTION BIỂU ĐỒ NẾN ===
            show_candlestick_2 = False
            if len(selected_stocks_2) == 1:
                # Lấy trạng thái đã lưu
                default_candlestick_2 = st.session_state.auto_show_candlestick
                show_candlestick_2 = st.checkbox(
                    "Hiển thị biểu đồ nến (Candlestick)", 
                    value=default_candlestick_2, 
                    key="candlestick_2"
                )
                # Lưu trạng thái
                st.session_state.auto_show_candlestick = show_candlestick_2
            
            # Vẽ biểu đồ giá cổ phiếu
            if show_candlestick_2 and len(selected_stocks_2) == 1:
                # Hiển thị biểu đồ nến
                ticker = selected_stocks_2[0]
                with st.spinner(f"Đang tải dữ liệu OHLC cho {ticker}..."):
                    ohlc_data = fetch_ohlc_data(ticker, data_loader_module.ANALYSIS_START_DATE, data_loader_module.ANALYSIS_END_DATE)
                    if not ohlc_data.empty:
                        plot_candlestick_chart(ohlc_data, ticker)
                    else:
                        st.error("Không thể tải dữ liệu OHLC. Hiển thị biểu đồ đường thay thế.")
                        plot_interactive_stock_chart(data, selected_stocks_2)
            else:
                # Vẽ biểu đồ bình thường
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

# Tùy chọn giữa các chế độ - Lấy giá trị mặc định từ session state
default_option = get_current_tab()
option = st.sidebar.radio(
    "Chọn phương thức", 
    ["Tổng quan Thị trường & Ngành", "Tự chọn mã cổ phiếu", "Hệ thống đề xuất mã cổ phiếu tự động"],
    index=["Tổng quan Thị trường & Ngành", "Tự chọn mã cổ phiếu", "Hệ thống đề xuất mã cổ phiếu tự động"].index(default_option) if default_option in ["Tổng quan Thị trường & Ngành", "Tự chọn mã cổ phiếu", "Hệ thống đề xuất mã cổ phiếu tự động"] else 0
)

# Cập nhật tab hiện tại vào session state
update_current_tab(option)

if option == "Tổng quan Thị trường & Ngành":
    # Hiển thị trang tổng quan ngành
    show_sector_overview_page(df, data_loader_module)

elif option == "Tự chọn mã cổ phiếu":
    # Giao diện người dùng để lọc từ file CSV
    st.title("Dashboard hỗ trợ tối ưu hóa danh mục đầu tư chứng khoán")
    
    # Sidebar
    st.sidebar.title("Bộ lọc và Cấu hình")
    
    # Lấy trạng thái đã lưu
    filter_state = get_manual_filter_state()
    
    # Bộ lọc theo sàn giao dịch (exchange)
    exchanges = df['exchange'].unique()
    # Sử dụng giá trị đã lưu hoặc mặc định
    saved_exchange = filter_state.get('exchange')
    if saved_exchange and saved_exchange in exchanges:
        default_index = list(exchanges).index(saved_exchange)
    else:
        default_index = list(exchanges).index(DEFAULT_MARKET) if DEFAULT_MARKET in exchanges else 0
    
    selected_exchange = st.sidebar.selectbox('Chọn sàn giao dịch', exchanges, index=default_index)

    # Lọc dữ liệu dựa trên sàn giao dịch đã chọn
    filtered_df = df[df['exchange'] == selected_exchange]

    # Bộ lọc theo loại ngành (icb_name)
    icb_names = filtered_df['icb_name'].unique()
    saved_icb = filter_state.get('icb_name')
    if saved_icb and saved_icb in icb_names:
        default_icb_index = list(icb_names).index(saved_icb)
    else:
        default_icb_index = 0
    
    selected_icb_name = st.sidebar.selectbox('Chọn ngành', icb_names, index=default_icb_index)

    # Lọc dữ liệu dựa trên ngành đã chọn
    filtered_df = filtered_df[filtered_df['icb_name'] == selected_icb_name]

    # === BỘ LỌC PHÂN TÍCH CƠ BẢN ===
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Bộ lọc phân tích cơ bản")
    
    # Checkbox để bật/tắt bộ lọc phân tích cơ bản - lấy từ session state
    enable_fundamental_filter = st.sidebar.checkbox(
        "Bật bộ lọc mã cổ phiếu giá trị", 
        value=filter_state.get('enable_fundamental_filter', False)
    )
    
    if enable_fundamental_filter:
        st.sidebar.markdown("*Lọc mã cổ phiếu theo tiêu chí phân tích cơ bản*")
        
        # Lấy giá trị đã lưu
        saved_filters = get_manual_fundamental_filters()
        
        # Bộ lọc P/E (Price to Earnings)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            pe_min = st.number_input("P/E tối thiểu", min_value=0.0, value=saved_filters['pe_min'], step=0.5, key="pe_min")
        with col2:
            pe_max = st.number_input("P/E tối đa", min_value=0.0, value=saved_filters['pe_max'], step=0.5, key="pe_max")
        
        # Bộ lọc P/B (Price to Book)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            pb_min = st.number_input("P/B tối thiểu", min_value=0.0, value=saved_filters['pb_min'], step=0.1, key="pb_min")
        with col2:
            pb_max = st.number_input("P/B tối đa", min_value=0.0, value=saved_filters['pb_max'], step=0.1, key="pb_max")
        
        # Bộ lọc ROE (Return on Equity)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            roe_min = st.number_input("ROE tối thiểu (%)", min_value=0.0, value=saved_filters['roe_min'], step=1.0, key="roe_min")
        with col2:
            roe_max = st.number_input("ROE tối đa (%)", min_value=0.0, value=saved_filters['roe_max'], step=1.0, key="roe_max")
        
        # Bộ lọc ROA (Return on Assets)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            roa_min = st.number_input("ROA tối thiểu (%)", min_value=0.0, value=saved_filters['roa_min'], step=1.0, key="roa_min")
        with col2:
            roa_max = st.number_input("ROA tối đa (%)", min_value=0.0, value=saved_filters['roa_max'], step=1.0, key="roa_max")
        
        # Bộ lọc biên lợi nhuận (Profit Margin)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            margin_min = st.number_input("Biên lợi nhuận tối thiểu (%)", min_value=0.0, value=saved_filters['margin_min'], step=1.0, key="margin_min")
        with col2:
            margin_max = st.number_input("Biên lợi nhuận tối đa (%)", min_value=0.0, value=saved_filters['margin_max'], step=1.0, key="margin_max")
        
        # Bộ lọc EPS (Earnings per Share)
        eps_min = st.sidebar.number_input("EPS tối thiểu (nghìn VND)", min_value=0.0, value=saved_filters['eps_min'], step=100.0, key="eps_min")
        
        # Lưu trạng thái bộ lọc
        save_manual_fundamental_filters(pe_min, pe_max, pb_min, pb_max, roe_min, roe_max, 
                                       roa_min, roa_max, margin_min, margin_max, eps_min)
        
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
                    st.sidebar.success(f"✓ Đã lọc được {len(filtered_fundamental)} mã cổ phiếu đáp ứng tiêu chí")
                else:
                    st.sidebar.error("Không thể lấy dữ liệu phân tích cơ bản")
        
        # Hiển thị kết quả lọc
        if 'filtered_fundamental' in st.session_state and not st.session_state.filtered_fundamental.empty:
            st.subheader(" Kết quả lọc mã cổ phiếu giá trị")
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
            
            # Cho phép thêm các mã cổ phiếu đã lọc vào danh mục
            if st.button(" Thêm tất cả mã cổ phiếu đã lọc vào danh mục"):
                added_count = 0
                for symbol in st.session_state.filtered_fundamental['symbol'].tolist():
                    if symbol not in st.session_state.selected_stocks:
                        st.session_state.selected_stocks.append(symbol)
                        added_count += 1
                st.success(f"✓ Đã thêm {added_count} mã cổ phiếu vào danh mục!")
            
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
    
    # Lấy giá trị ngày đã lưu
    default_start = filter_state.get('start_date') or pd.to_datetime(ANALYSIS_START_DATE).date()
    default_end = filter_state.get('end_date') or pd.to_datetime(ANALYSIS_END_DATE).date()
    
    start_date = st.sidebar.date_input(
        "Ngày bắt đầu", 
        value=default_start, 
        max_value=today
    )
    end_date = st.sidebar.date_input(
        "Ngày kết thúc", 
        value=default_end, 
        max_value=today
    )
    
    # Lưu trạng thái bộ lọc
    save_manual_filter_state(selected_exchange, selected_icb_name, start_date, end_date, enable_fundamental_filter)
    
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

elif option == "Hệ thống đề xuất mã cổ phiếu tự động":
    # Giao diện Streamlit
    st.title("Hệ thống đề xuất mã cổ phiếu tự động")
    st.sidebar.title("Cấu hình đề xuất cổ phiếu")

    # Lấy trạng thái đã lưu
    auto_state = get_auto_filter_state()
    
    # Bước 1: Chọn sàn giao dịch
    if not df.empty:
        # Sử dụng giá trị đã lưu hoặc mặc định
        saved_exchanges = auto_state.get('exchanges', [])
        if not saved_exchanges:
            saved_exchanges = [DEFAULT_MARKET] if DEFAULT_MARKET in df['exchange'].unique() else []
        
        selected_exchanges = st.sidebar.multiselect(
            "Chọn sàn giao dịch", 
            df['exchange'].unique(), 
            default=saved_exchanges
        )

        # Lọc dữ liệu theo nhiều sàn giao dịch
        filtered_df = df[df['exchange'].isin(selected_exchanges)]

        # Bước 2: Chọn nhiều ngành
        saved_sectors = auto_state.get('sectors', [])
        selected_sectors = st.sidebar.multiselect("Chọn ngành", filtered_df['icb_name'].unique(), default=saved_sectors)

        if selected_sectors:
            # Lọc theo các ngành đã chọn
            sector_df = filtered_df[filtered_df['icb_name'].isin(selected_sectors)]

            # Bước 3: Chọn số lượng cổ phiếu cho từng ngành
            stocks_per_sector = {}
            saved_stocks_per_sector = auto_state.get('stocks_per_sector', {})
            
            for sector in selected_sectors:
                # Sử dụng giá trị đã lưu hoặc mặc định
                default_num = saved_stocks_per_sector.get(sector, 3)
                num_stocks = st.sidebar.number_input(
                    f"Số cổ phiếu muốn đầu tư trong ngành '{sector}'", 
                    min_value=1, 
                    max_value=10, 
                    value=default_num,
                    key=f"num_stocks_{sector}"
                )
                stocks_per_sector[sector] = num_stocks

            # Bước 4: Chọn cách lọc
            saved_filter_method = auto_state.get('filter_method', 'Lợi nhuận lớn nhất')
            filter_method_options = ["Lợi nhuận lớn nhất", "Rủi ro bé nhất", "Phân tích cơ bản (Cổ phiếu giá trị)"]
            default_method_index = filter_method_options.index(saved_filter_method) if saved_filter_method in filter_method_options else 0
            
            filter_method = st.sidebar.radio(
                "Cách lọc cổ phiếu", 
                filter_method_options,
                index=default_method_index
            )

            # === BỘ LỌC PHÂN TÍCH CƠ BẢN CHO ĐỀ XUẤT TỰ ĐỘNG ===
            fundamental_filters = {}
            if filter_method == "Phân tích cơ bản (Cổ phiếu giá trị)":
                st.sidebar.markdown("---")
                st.sidebar.subheader("Tiêu chí phân tích cơ bản")
                
                # Lấy giá trị đã lưu
                saved_auto_filters = get_auto_fundamental_filters()
                
                # Bộ lọc P/E
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    pe_min_auto = st.number_input("P/E tối thiểu", min_value=0.0, value=saved_auto_filters['pe_min'], step=0.5, key="pe_min_auto")
                with col2:
                    pe_max_auto = st.number_input("P/E tối đa", min_value=0.0, value=saved_auto_filters['pe_max'], step=0.5, key="pe_max_auto")
                
                # Bộ lọc P/B
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    pb_min_auto = st.number_input("P/B tối thiểu", min_value=0.0, value=saved_auto_filters['pb_min'], step=0.1, key="pb_min_auto")
                with col2:
                    pb_max_auto = st.number_input("P/B tối đa", min_value=0.0, value=saved_auto_filters['pb_max'], step=0.1, key="pb_max_auto")
                
                # Bộ lọc ROE
                roe_min_auto = st.sidebar.number_input("ROE tối thiểu (%)", min_value=0.0, value=saved_auto_filters['roe_min'], step=1.0, key="roe_min_auto")
                
                # Bộ lọc ROA
                roa_min_auto = st.sidebar.number_input("ROA tối thiểu (%)", min_value=0.0, value=saved_auto_filters['roa_min'], step=1.0, key="roa_min_auto")
                
                # Bộ lọc biên lợi nhuận
                margin_min_auto = st.sidebar.number_input("Biên lợi nhuận tối thiểu (%)", min_value=0.0, value=saved_auto_filters['margin_min'], step=1.0, key="margin_min_auto")
                
                # Bộ lọc EPS
                eps_min_auto = st.sidebar.number_input("EPS tối thiểu (nghìn VND)", min_value=0.0, value=saved_auto_filters['eps_min'], step=100.0, key="eps_min_auto")
                
                # Lưu trạng thái
                save_auto_fundamental_filters(pe_min_auto, pe_max_auto, pb_min_auto, pb_max_auto, 
                                            roe_min_auto, roa_min_auto, margin_min_auto, eps_min_auto)
                
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
            
            # Lấy giá trị ngày đã lưu
            default_start_1 = auto_state.get('start_date') or pd.to_datetime(ANALYSIS_START_DATE).date()
            default_end_1 = auto_state.get('end_date') or pd.to_datetime(ANALYSIS_END_DATE).date()
            
            start_date = st.sidebar.date_input(
                "Ngày bắt đầu", 
                value=default_start_1,
                min_value=pd.to_datetime(ANALYSIS_START_DATE).date(),
                max_value=pd.to_datetime(ANALYSIS_END_DATE).date(),
                key="start_date_1"
            )
            end_date = st.sidebar.date_input(
                "Ngày kết thúc", 
                value=default_end_1,
                min_value=pd.to_datetime(ANALYSIS_START_DATE).date(),
                max_value=pd.to_datetime(ANALYSIS_END_DATE).date(),
                key="end_date_1"
            )
            
            # Lưu trạng thái bộ lọc
            save_auto_filter_state(selected_exchanges, selected_sectors, stocks_per_sector, 
                                  filter_method, start_date, end_date)
            
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
