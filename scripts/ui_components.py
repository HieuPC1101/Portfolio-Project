"""
Module ui_components.py
Chứa các hàm giao diện Streamlit: hiển thị danh sách cổ phiếu, bộ lọc, thêm/xóa cổ phiếu.
"""

import streamlit as st


def display_selected_stocks(df):
    """
    Hiển thị danh sách mã cổ phiếu đã chọn và xử lý thao tác xóa (chế độ tự chọn).
    
    Args:
        df (pd.DataFrame): DataFrame chứa thông tin công ty
    """
    if st.button("Xóa hết các cổ phiếu"):
        st.session_state.selected_stocks = []
        st.success("Đã xóa hết tất cả cổ phiếu khỏi danh sách!")
    
    if st.session_state.selected_stocks:
        st.markdown("### Danh sách mã cổ phiếu đã chọn:")
        for stock in st.session_state.selected_stocks:
            # Tìm thông tin chi tiết từ DataFrame dựa trên symbol
            stock_info = df[df['symbol'] == stock]
            if not stock_info.empty:
                organ_name = stock_info.iloc[0]['organ_name']
                icb_name = stock_info.iloc[0]['icb_name']
                exchange = stock_info.iloc[0]['exchange']

                # Hiển thị thông tin: Mã cổ phiếu, tên công ty, và ngành
                col1, col2, col3, col4, col5 = st.columns([2, 4, 3, 2, 1])
                col1.write(stock)  # Mã cổ phiếu
                col2.write(organ_name)  # Tên công ty
                col3.write(icb_name)  # Tên ngành
                col4.write(exchange)
                if col5.button(f"❌", key=f"remove_{stock}"):  # Nút xóa
                    st.session_state.selected_stocks.remove(stock)
                    st.rerun()  # Làm mới lại giao diện sau khi xóa
    else:
        st.write("Chưa có mã cổ phiếu nào được chọn.")


def display_selected_stocks_2(df):
    """
    Hiển thị danh sách mã cổ phiếu đã được chọn và cung cấp tùy chọn xóa từng mã hoặc toàn bộ (chế độ đề xuất).
    
    Args:
        df (pd.DataFrame): DataFrame chứa thông tin công ty
    """
    # Nút xóa toàn bộ danh sách cổ phiếu
    if st.button("Xóa hết các cổ phiếu trong danh mục"):
        if "selected_stocks_2" in st.session_state:
            st.session_state.selected_stocks_2 = []
            st.success("Đã xóa hết tất cả cổ phiếu khỏi danh sách!")
        else:
            st.warning("Không có mã cổ phiếu nào để xóa.")

    # Hiển thị danh sách mã cổ phiếu đã chọn
    if "selected_stocks_2" in st.session_state and st.session_state.selected_stocks_2:
        st.markdown("### Danh sách cổ phiếu trong danh mục đầu tư:")

        # Duyệt qua từng mã cổ phiếu trong danh sách
        for stock in st.session_state.selected_stocks_2:
            # Lấy thông tin chi tiết từ DataFrame dựa trên mã cổ phiếu
            stock_info = df[df['symbol'] == stock]
            if not stock_info.empty:
                organ_name = stock_info.iloc[0]['organ_name']
                icb_name = stock_info.iloc[0]['icb_name']
                exchange = stock_info.iloc[0]['exchange']

                # Hiển thị thông tin cổ phiếu
                col1, col2, col3, col4, col5 = st.columns([2, 4, 3, 2, 1])
                col1.write(stock)  # Mã cổ phiếu
                col2.write(organ_name)  # Tên công ty
                col3.write(icb_name)  # Tên ngành
                col4.write(exchange)  # Sàn giao dịch
                with col5:
                    if st.button(f"❌", key=f"remove_{stock}"):  # Nút xóa từng mã cổ phiếu
                        st.session_state.selected_stocks_2.remove(stock)
                        st.success(f"Đã xóa mã cổ phiếu '{stock}' khỏi danh sách!")
                        st.rerun()  # Làm mới giao diện sau khi xóa
    else:
        st.info("Chưa có mã cổ phiếu nào được chọn.")
