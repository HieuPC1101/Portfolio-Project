"""
Module session_manager.py
Quản lý trạng thái phiên làm việc (session state) cho ứng dụng Streamlit.
Lưu trữ và khôi phục trạng thái khi người dùng chuyển tab hoặc thao tác.
"""

import streamlit as st
import datetime


def initialize_session_state():
    """
    Khởi tạo tất cả các biến session state cần thiết cho ứng dụng.
    Hàm này được gọi khi ứng dụng khởi động.
    """
    
    # ========== TRẠ THÁI CHUNG ==========
    # Tab/trang hiện tại
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Tổng quan Thị trường & Ngành"
    
    # ========== CHẾ ĐỘ TỰ CHỌN (Manual Selection) ==========
    # Danh sách cổ phiếu đã chọn
    if 'selected_stocks' not in st.session_state:
        st.session_state.selected_stocks = []
    
    # Trạng thái bộ lọc chế độ tự chọn
    if 'manual_filter_state' not in st.session_state:
        st.session_state.manual_filter_state = {
            'exchange': None,
            'icb_name': None,
            'start_date': None,
            'end_date': None,
            'enable_fundamental_filter': False,
        }
    
    # Trạng thái biểu đồ
    if 'manual_show_candlestick' not in st.session_state:
        st.session_state.manual_show_candlestick = False
    
    # Trạng thái chiến lược đã chọn
    if 'manual_selected_strategy' not in st.session_state:
        st.session_state.manual_selected_strategy = None
    
    # Số tiền đầu tư
    if 'manual_investment_amount' not in st.session_state:
        st.session_state.manual_investment_amount = 1000000
    
    # ========== CHẾ ĐỘ ĐỀ XUẤT TỰ ĐỘNG (Auto Selection) ==========
    # Danh sách cổ phiếu đã chọn
    if 'selected_stocks_2' not in st.session_state:
        st.session_state.selected_stocks_2 = []
    
    # Danh sách cổ phiếu cuối cùng (theo sàn và ngành)
    if 'final_selected_stocks' not in st.session_state:
        st.session_state.final_selected_stocks = {}
    
    # Trạng thái bộ lọc chế độ đề xuất
    if 'auto_filter_state' not in st.session_state:
        st.session_state.auto_filter_state = {
            'exchanges': [],
            'sectors': [],
            'stocks_per_sector': {},
            'filter_method': 'Lợi nhuận lớn nhất',
            'start_date': None,
            'end_date': None,
        }
    
    # Trạng thái biểu đồ
    if 'auto_show_candlestick' not in st.session_state:
        st.session_state.auto_show_candlestick = False
    
    # Trạng thái chiến lược đã chọn
    if 'auto_selected_strategy' not in st.session_state:
        st.session_state.auto_selected_strategy = None
    
    # Số tiền đầu tư
    if 'auto_investment_amount' not in st.session_state:
        st.session_state.auto_investment_amount = 1000000
    
    # ========== TỔNG QUAN THỊ TRƯỜNG & NGÀNH ==========
    # Trạng thái trang tổng quan
    if 'market_overview_state' not in st.session_state:
        st.session_state.market_overview_state = {
            'selected_exchange': None,
            'view_mode': 'market',  # 'market' hoặc 'sector'
        }


def save_manual_filter_state(exchange, icb_name, start_date, end_date, enable_fundamental_filter):
    """
    Lưu trạng thái bộ lọc cho chế độ tự chọn.
    
    Args:
        exchange (str): Sàn giao dịch đã chọn
        icb_name (str): Ngành đã chọn
        start_date (date): Ngày bắt đầu
        end_date (date): Ngày kết thúc
        enable_fundamental_filter (bool): Có bật bộ lọc phân tích cơ bản không
    """
    st.session_state.manual_filter_state = {
        'exchange': exchange,
        'icb_name': icb_name,
        'start_date': start_date,
        'end_date': end_date,
        'enable_fundamental_filter': enable_fundamental_filter,
    }


def save_auto_filter_state(exchanges, sectors, stocks_per_sector, filter_method, start_date, end_date):
    """
    Lưu trạng thái bộ lọc cho chế độ đề xuất tự động.
    
    Args:
        exchanges (list): Danh sách sàn giao dịch đã chọn
        sectors (list): Danh sách ngành đã chọn
        stocks_per_sector (dict): Số lượng cổ phiếu cho từng ngành
        filter_method (str): Phương pháp lọc
        start_date (date): Ngày bắt đầu
        end_date (date): Ngày kết thúc
    """
    st.session_state.auto_filter_state = {
        'exchanges': exchanges,
        'sectors': sectors,
        'stocks_per_sector': stocks_per_sector,
        'filter_method': filter_method,
        'start_date': start_date,
        'end_date': end_date,
    }


def get_manual_filter_state():
    """
    Lấy trạng thái bộ lọc đã lưu cho chế độ tự chọn.
    
    Returns:
        dict: Trạng thái bộ lọc
    """
    return st.session_state.manual_filter_state


def get_auto_filter_state():
    """
    Lấy trạng thái bộ lọc đã lưu cho chế độ đề xuất tự động.
    
    Returns:
        dict: Trạng thái bộ lọc
    """
    return st.session_state.auto_filter_state


def save_market_overview_state(selected_exchange, view_mode):
    """
    Lưu trạng thái trang tổng quan thị trường.
    
    Args:
        selected_exchange (str): Sàn giao dịch đã chọn
        view_mode (str): Chế độ xem ('market' hoặc 'sector')
    """
    st.session_state.market_overview_state = {
        'selected_exchange': selected_exchange,
        'view_mode': view_mode,
    }


def get_market_overview_state():
    """
    Lấy trạng thái trang tổng quan thị trường.
    
    Returns:
        dict: Trạng thái trang tổng quan
    """
    return st.session_state.market_overview_state


def clear_manual_selection():
    """
    Xóa toàn bộ trạng thái chế độ tự chọn.
    """
    st.session_state.selected_stocks = []


def clear_auto_selection():
    """
    Xóa toàn bộ trạng thái chế độ đề xuất tự động.
    """
    st.session_state.selected_stocks_2 = []
    st.session_state.final_selected_stocks = {}


def update_current_tab(tab_name):
    """
    Cập nhật tab hiện tại.
    
    Args:
        tab_name (str): Tên tab/trang hiện tại
    """
    st.session_state.current_tab = tab_name


def get_current_tab():
    """
    Lấy tab hiện tại.
    
    Returns:
        str: Tên tab hiện tại
    """
    return st.session_state.current_tab
