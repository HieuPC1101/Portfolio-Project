"""
Dashboard chính - Ứng dụng Streamlit hỗ trợ tối ưu hóa danh mục đầu tư chứng khoán.
Sử dụng dữ liệu từ PostgreSQL Database.
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

import pandas as pd
import numpy as np
import os
import sys
import datetime

# Thêm đường dẫn để import các module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import cấu hình
from scripts.config import ANALYSIS_START_DATE, ANALYSIS_END_DATE, DEFAULT_MARKET, DEFAULT_INVESTMENT_AMOUNT

# Import data_loader_db thay vì data_loader
from scripts.data_loader import (
    fetch_data_from_csv,
    fetch_stock_data2,
    get_latest_prices,
    calculate_metrics,
    fetch_ohlc_data,
    fetch_fundamental_data_batch,
    get_market_indices
)

# Import các module khác
from scripts.portfolio_models import (
    markowitz_optimization,
    max_sharpe,
    min_volatility,
    min_cvar,
    min_cdar,
    hrp_model
)
# Sử dụng visualization_db thay vì visualization để hỗ trợ database
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
from scripts.session_manager import (
    initialize_session_state,
    save_manual_filter_state,
    save_auto_filter_state,
    get_manual_filter_state,
    get_auto_filter_state,
    update_current_tab,
    get_current_tab
)
from scripts.chatbot_ui import (
    render_chatbot_page,
    render_chat_controls
)

# Import market_overview_db
import scripts.data_loader as data_loader_module

# Đường dẫn đến file CSV
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
file_path = os.path.join(data_dir, "company_info.csv")

# Lấy dữ liệu từ file CSV
df = fetch_data_from_csv(file_path)

# Khởi tạo session state khi ứng dụng khởi động
initialize_session_state()

# Thêm thông báo rằng đang sử dụng database
st.sidebar.info("📊 Sử dụng dữ liệu từ PostgreSQL Database")


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
                        start_date = pd.to_datetime(ANALYSIS_START_DATE).strftime('%Y-%m-%d')
                        end_date = pd.to_datetime(ANALYSIS_END_DATE).strftime('%Y-%m-%d')
                        backtest_result = backtest_portfolio(
                            symbols, 
                            weights, 
                            start_date, 
                            end_date,
                            fetch_stock_data2,
                            get_market_indices_func=get_market_indices
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
        
        # Lấy dữ liệu giá cổ phiếu từ database
        start_date = filter_state.get('start_date') or default_start
        end_date = filter_state.get('end_date') or default_end
        
        data, skipped_tickers = fetch_stock_data2(selected_stocks, start_date, end_date)

        if not data.empty:
            st.subheader("Giá cổ phiếu")
            
            # Option biểu đồ nến
            show_candlestick = False
            if len(selected_stocks) == 1:
                default_candlestick = st.session_state.manual_show_candlestick
                show_candlestick = st.checkbox(
                    "Hiển thị biểu đồ nến (Candlestick)", 
                    value=default_candlestick, 
                    key="candlestick_1"
                )
                st.session_state.manual_show_candlestick = show_candlestick
            
            # Vẽ biểu đồ giá cổ phiếu
            if show_candlestick and len(selected_stocks) == 1:
                ticker = selected_stocks[0]
                with st.spinner(f"Đang tải dữ liệu OHLC cho {ticker}..."):
                    ohlc_data = fetch_ohlc_data(ticker, start_date, end_date)
                    if not ohlc_data.empty:
                        plot_candlestick_chart(ohlc_data, ticker)
                    else:
                        st.error(f"Không có dữ liệu OHLC cho {ticker}")
            else:
                plot_interactive_stock_chart(data, selected_stocks)
            
            # Chạy các mô hình
            run_models(data)
        else:
            st.error("Dữ liệu cổ phiếu bị thiếu hoặc không có trong database.")
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
        max_date_2 = pd.to_datetime(ANALYSIS_END_DATE).date()
        min_date_2 = pd.to_datetime(ANALYSIS_START_DATE).date()
        
        default_start_2 = filter_state.get('start_date') or min_date_2
        default_end_2 = filter_state.get('end_date') or max_date_2
        
        # Kiểm tra và thông báo nếu giá trị vượt quá giới hạn
        adjusted_2 = False
        if default_start_2 < min_date_2 or default_start_2 > max_date_2:
            adjusted_2 = True
            default_start_2 = max(min_date_2, min(default_start_2, max_date_2))
        if default_end_2 < min_date_2 or default_end_2 > max_date_2:
            adjusted_2 = True
            default_end_2 = max(min_date_2, min(default_end_2, max_date_2))
        
        if adjusted_2:
            st.sidebar.warning(f"⚠️ Ngày đã lưu không hợp lệ, đã tự động điều chỉnh về khoảng {min_date_2.strftime('%d/%m/%Y')} - {max_date_2.strftime('%d/%m/%Y')}")
        
        start_date_2 = st.sidebar.date_input(
            "Ngày bắt đầu", 
            value=default_start_2, 
            min_value=min_date_2,
            max_value=max_date_2,
            key="start_date_2"
        )
        end_date_2 = st.sidebar.date_input(
            "Ngày kết thúc", 
            value=default_end_2, 
            min_value=min_date_2,
            max_value=max_date_2,
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
            
        # Lấy dữ liệu giá cổ phiếu từ database
        data, skipped_tickers = fetch_stock_data2(selected_stocks_2, start_date_2, end_date_2)

        if not data.empty:
            st.subheader("Giá cổ phiếu")
            
            # Option biểu đồ nến
            show_candlestick_2 = False
            if len(selected_stocks_2) == 1:
                default_candlestick_2 = st.session_state.auto_show_candlestick
                show_candlestick_2 = st.checkbox(
                    "Hiển thị biểu đồ nến (Candlestick)", 
                    value=default_candlestick_2, 
                    key="candlestick_2"
                )
                st.session_state.auto_show_candlestick = show_candlestick_2
            
            # Vẽ biểu đồ
            if show_candlestick_2 and len(selected_stocks_2) == 1:
                ticker = selected_stocks_2[0]
                with st.spinner(f"Đang tải dữ liệu OHLC cho {ticker}..."):
                    ohlc_data = fetch_ohlc_data(ticker, start_date_2, end_date_2)
                    if not ohlc_data.empty:
                        plot_candlestick_chart(ohlc_data, ticker)
                    else:
                        st.error(f"Không có dữ liệu OHLC cho {ticker}")
            else:
                plot_interactive_stock_chart(data, selected_stocks_2)
            
            # Chạy các mô hình
            run_models(data)
        else:
            st.error("Dữ liệu cổ phiếu bị thiếu hoặc không có trong database.")
    else:
        st.warning("Chưa có mã cổ phiếu nào trong danh mục. Vui lòng chọn mã cổ phiếu trước.")


# ========== GIAO DIỆN CHÍNH ==========

# Sidebar
st.sidebar.title("Lựa chọn phương thức")

# Tùy chọn giữa các chế độ - Lấy giá trị mặc định từ session state
default_option = get_current_tab()
option = st.sidebar.radio(
    "Chọn phương thức", 
    ["Tổng quan Thị trường & Ngành", "Tự chọn mã cổ phiếu", "Hệ thống đề xuất mã cổ phiếu tự động", "Trợ lý AI"],
    index=["Tổng quan Thị trường & Ngành", "Tự chọn mã cổ phiếu", "Hệ thống đề xuất mã cổ phiếu tự động", "Trợ lý AI"].index(default_option) if default_option in ["Tổng quan Thị trường & Ngành", "Tự chọn mã cổ phiếu", "Hệ thống đề xuất mã cổ phiếu tự động", "Trợ lý AI"] else 0
)

# Cập nhật tab hiện tại vào session state
update_current_tab(option)

if option == "Trợ lý AI":
    # Hiển thị trang chatbot
    render_chatbot_page()

    # Thêm 2 nút vào sidebar dưới chức năng Trợ lý AI khi chatbot đã sẵn sàng
    if st.session_state.get("chatbot") is not None:
        st.sidebar.markdown("#### Tiện ích Trợ lý AI")
        controls_container = st.sidebar.container()
        render_chat_controls(controls_container, key_prefix="main_sidebar")

elif option == "Tổng quan Thị trường & Ngành":
    # Import market_overview
    from scripts.market_overview import show_sector_overview_page
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
    max_date = today
    min_date = today - datetime.timedelta(days=365*3)  # 3 năm trước
    
    default_start = filter_state.get('start_date') or pd.to_datetime(ANALYSIS_START_DATE).date()
    default_end = filter_state.get('end_date') or pd.to_datetime(ANALYSIS_END_DATE).date()
    
    # Kiểm tra và thông báo nếu giá trị vượt quá giới hạn
    adjusted = False
    if default_start < min_date or default_start > max_date:
        adjusted = True
        default_start = max(min_date, min(default_start, max_date))
    if default_end < min_date or default_end > max_date:
        adjusted = True
        default_end = max(min_date, min(default_end, max_date))
    
    if adjusted:
        st.sidebar.warning(f"⚠️ Ngày đã lưu không hợp lệ, đã tự động điều chỉnh về khoảng {min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}")
    
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
    save_manual_filter_state(selected_exchange, selected_icb_name, start_date, end_date, False)
    
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
                sector_stock_count = len(sector_df[sector_df['icb_name'] == sector])
                default_count = saved_stocks_per_sector.get(sector, min(5, sector_stock_count))
                stocks_per_sector[sector] = st.sidebar.number_input(
                    f"Số cổ phiếu cho {sector}",
                    min_value=1,
                    max_value=sector_stock_count,
                    value=default_count,
                    key=f"stocks_{sector}"
                )

            # Bước 4: Chọn cách lọc
            saved_filter_method = auto_state.get('filter_method', 'Lợi nhuận lớn nhất')
            filter_method_options = ["Lợi nhuận lớn nhất", "Rủi ro bé nhất"]
            default_method_index = filter_method_options.index(saved_filter_method) if saved_filter_method in filter_method_options else 0
            
            filter_method = st.sidebar.radio(
                "Cách lọc cổ phiếu", 
                filter_method_options,
                index=default_method_index
            )

            # Lựa chọn thời gian lấy dữ liệu
            today = datetime.date.today()
            max_date = pd.to_datetime(ANALYSIS_END_DATE).date()
            min_date = pd.to_datetime(ANALYSIS_START_DATE).date()
            
            # Lấy giá trị ngày đã lưu
            default_start_1 = auto_state.get('start_date') or min_date
            default_end_1 = auto_state.get('end_date') or max_date
            
            # Đảm bảo giá trị mặc định nằm trong khoảng hợp lệ
            default_start_1 = max(min_date, min(default_start_1, max_date))
            default_end_1 = max(min_date, min(default_end_1, max_date))
            
            start_date = st.sidebar.date_input(
                "Ngày bắt đầu", 
                value=default_start_1,
                min_value=min_date,
                max_value=max_date,
                key="start_date_1"
            )
            end_date = st.sidebar.date_input(
                "Ngày kết thúc", 
                value=default_end_1,
                min_value=min_date,
                max_value=max_date,
                key="end_date_1"
            )
            
            # Lưu trạng thái bộ lọc
            save_auto_filter_state(selected_exchanges, selected_sectors, stocks_per_sector, 
                                  filter_method, start_date, end_date)
            
            if start_date > max_date or end_date > today:
                st.sidebar.error("Ngày không hợp lệ.")
            elif start_date > end_date:
                st.sidebar.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc.")
            else:
                st.sidebar.success("Ngày tháng hợp lệ.")

            if st.sidebar.button("Đề xuất cổ phiếu"):
                st.info("Đang tải dữ liệu từ database và đề xuất cổ phiếu...")
                
                final_selected = {}
                for exchange in selected_exchanges:
                    final_selected[exchange] = {}
                    for sector in selected_sectors:
                        sector_symbols = sector_df[
                            (sector_df['exchange'] == exchange) & 
                            (sector_df['icb_name'] == sector)
                        ]['symbol'].tolist()
                        
                        if sector_symbols:
                            data, _ = fetch_stock_data2(sector_symbols, start_date, end_date)
                            if not data.empty:
                                mean_returns, volatility = calculate_metrics(data)
                                
                                if filter_method == "Lợi nhuận lớn nhất":
                                    top_stocks = mean_returns.nlargest(stocks_per_sector[sector])
                                else:
                                    top_stocks = volatility.nsmallest(stocks_per_sector[sector])
                                
                                final_selected[exchange][sector] = top_stocks.index.tolist()
                
                st.session_state.final_selected_stocks = final_selected
                st.session_state.selected_stocks_2 = [
                    stock for sectors in final_selected.values() 
                    for stocks in sectors.values() 
                    for stock in stocks
                ]
                st.success("✓ Đã đề xuất cổ phiếu thành công!")

    if st.session_state.final_selected_stocks:
        st.subheader("Danh mục cổ phiếu được lọc theo sàn và ngành")
        if st.button("Xóa hết các cổ phiếu đã được đề xuất"):
            st.session_state.final_selected_stocks = {}
            st.success("Đã xóa hết tất cả cổ phiếu khỏi danh sách!")
        
        for exchange, sectors in st.session_state.final_selected_stocks.items():
            st.write(f"### Sàn: {exchange}")
            for sector, stocks in sectors.items():
                st.write(f"**{sector}**: {', '.join(stocks)}")

    display_selected_stocks_2(df)

    if __name__ == "__main__":
        main_auto_selection()
