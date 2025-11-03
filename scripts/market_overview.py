"""
Module market_overview.py
Chứa các hàm liên quan đến tổng quan thị trường và phân tích ngành.
"""

import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scripts.config import ANALYSIS_START_DATE, ANALYSIS_END_DATE, DEFAULT_MARKET, DEFAULT_INVESTMENT_AMOUNT


def analyze_sector_performance(df, data_loader_module):
    """
    Phân tích hiệu suất các ngành trong thị trường.
    
    Args:
        df (pd.DataFrame): DataFrame chứa thông tin công ty
        data_loader_module: Module data_loader để lấy dữ liệu giá
        
    Returns:
        pd.DataFrame: DataFrame chứa phân tích hiệu suất ngành
    """
    st.subheader("Phân tích Hiệu suất Ngành")
    
    # Lấy danh sách các ngành
    sectors = df['icb_name'].unique()
    
    # Chọn sàn giao dịch
    exchanges = df['exchange'].unique()
    default_index = list(exchanges).index(DEFAULT_MARKET) if DEFAULT_MARKET in exchanges else 0
    exchange = st.selectbox(
        "Chọn sàn giao dịch để phân tích",
        exchanges,
        index=default_index,
        key="sector_exchange"
    )
    
    # Lọc theo sàn
    filtered_df = df[df['exchange'] == exchange]
    sectors = filtered_df['icb_name'].unique()
    
    # Chọn thời gian phân tích
    col1, col2 = st.columns(2)
    with col1:
        analysis_period = st.selectbox(
            "Chọn khoảng thời gian phân tích",
            ["1 Tháng", "3 Tháng", "6 Tháng", "1 Năm"],
            key="sector_period"
        )
    
    # Tính toán ngày bắt đầu và kết thúc
    end_date = datetime.now().date()
    period_map = {
        "1 Tháng": 30,
        "3 Tháng": 90,
        "6 Tháng": 180,
        "1 Năm": 365
    }
    start_date = end_date - timedelta(days=period_map[analysis_period])
    
    if st.button("Phân tích Ngành", key="analyze_sectors"):
        sector_stats = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, sector in enumerate(sectors):
            status_text.text(f"Đang phân tích ngành {sector}... ({idx+1}/{len(sectors)})")
            
            # Lấy các cổ phiếu trong ngành
            sector_stocks = filtered_df[filtered_df['icb_name'] == sector]['symbol'].tolist()
            
            if len(sector_stocks) == 0:
                continue
            
            # Lấy dữ liệu giá (chỉ lấy tối đa 10 cổ phiếu đại diện)
            sample_stocks = sector_stocks[:10]
            data, skipped = data_loader_module.fetch_stock_data2(
                sample_stocks, 
                start_date, 
                end_date
            )
            
            if data.empty:
                continue
            
            # Tính toán các chỉ số
            returns = data.pct_change().dropna()
            
            # Lợi nhuận trung bình của ngành
            avg_return = returns.mean().mean() * 100
            
            # Biến động (volatility)
            avg_volatility = returns.std().mean() * 100
            
            # Lợi nhuận tích lũy
            cumulative_return = ((1 + returns).prod().mean() - 1) * 100
            
            # Số lượng cổ phiếu trong ngành
            num_stocks = len(sector_stocks)
            
            sector_stats.append({
                'Ngành': sector,
                'Số CP': num_stocks,
                'LN TB (%)': round(avg_return, 2),
                'Biến động (%)': round(avg_volatility, 2),
                'LN Tích lũy (%)': round(cumulative_return, 2),
                'Số CP phân tích': len(sample_stocks)
            })
            
            progress_bar.progress((idx + 1) / len(sectors))
        
        progress_bar.empty()
        status_text.empty()
        
        if sector_stats:
            # Tạo DataFrame kết quả
            result_df = pd.DataFrame(sector_stats)
            result_df = result_df.sort_values('LN Tích lũy (%)', ascending=False)
            
            # Hiển thị bảng kết quả
            st.success(f"Hoàn thành phân tích {len(result_df)} ngành!")
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Vẽ biểu đồ so sánh
            plot_sector_comparison(result_df)
            
            # Phân tích chi tiết
            show_sector_insights(result_df)
            
            return result_df
        else:
            st.warning("Không có dữ liệu để phân tích.")
            return pd.DataFrame()
    
    return pd.DataFrame()


def plot_sector_comparison(sector_df):
    """
    Vẽ biểu đồ so sánh hiệu suất các ngành.
    
    Args:
        sector_df (pd.DataFrame): DataFrame chứa dữ liệu ngành
    """
    st.subheader("Biểu đồ So sánh Ngành")
    
    # Tab cho các loại biểu đồ
    tab1, tab2, tab3 = st.tabs([
        "Lợi nhuận Tích lũy", 
        "Rủi ro - Lợi nhuận", 
        "Số lượng Cổ phiếu"
    ])
    
    with tab1:
        # Biểu đồ cột - Lợi nhuận tích lũy
        fig1 = px.bar(
            sector_df.sort_values('LN Tích lũy (%)', ascending=True),
            x='LN Tích lũy (%)',
            y='Ngành',
            orientation='h',
            title='Lợi nhuận Tích lũy theo Ngành',
            color='LN Tích lũy (%)',
            color_continuous_scale=['red', 'yellow', 'green'],
            text='LN Tích lũy (%)'
        )
        fig1.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig1.update_layout(height=max(400, len(sector_df) * 30))
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        # Biểu đồ scatter - Rủi ro vs Lợi nhuận
        fig2 = px.scatter(
            sector_df,
            x='Biến động (%)',
            y='LN Tích lũy (%)',
            size='Số CP',
            color='LN Tích lũy (%)',
            hover_data=['Ngành', 'Số CP'],
            title='Ma trận Rủi ro - Lợi nhuận',
            color_continuous_scale=['red', 'yellow', 'green'],
            text='Ngành'
        )
        fig2.update_traces(textposition='top center')
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.add_vline(x=sector_df['Biến động (%)'].median(), line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Biểu đồ tròn - Phân bổ số lượng cổ phiếu
        fig3 = px.pie(
            sector_df,
            values='Số CP',
            names='Ngành',
            title='Phân bổ Số lượng Cổ phiếu theo Ngành'
        )
        st.plotly_chart(fig3, use_container_width=True)


def show_sector_insights(sector_df):
    """
    Hiển thị các insights và phân tích chuyên sâu về ngành.
    
    Args:
        sector_df (pd.DataFrame): DataFrame chứa dữ liệu ngành
    """
    st.subheader("Phân tích Chuyên sâu")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Ngành tốt nhất",
            sector_df.iloc[0]['Ngành'],
            f"+{sector_df.iloc[0]['LN Tích lũy (%)']}%"
        )
    
    with col2:
        st.metric(
            "Ngành ổn định nhất",
            sector_df.nsmallest(1, 'Biến động (%)').iloc[0]['Ngành'],
            f"{sector_df.nsmallest(1, 'Biến động (%)').iloc[0]['Biến động (%)']}%"
        )
    
    with col3:
        st.metric(
            "Tổng số CP",
            f"{sector_df['Số CP'].sum()}",
            f"{len(sector_df)} ngành"
        )
    
    # Phân loại ngành
    st.markdown("### Phân loại Ngành")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Ngành Tiềm năng (Lợi nhuận cao, Rủi ro thấp):**")
        potential = sector_df[
            (sector_df['LN Tích lũy (%)'] > sector_df['LN Tích lũy (%)'].median()) &
            (sector_df['Biến động (%)'] < sector_df['Biến động (%)'].median())
        ]
        if not potential.empty:
            for _, row in potential.iterrows():
                st.write(f"- {row['Ngành']}: +{row['LN Tích lũy (%)']}%")
        else:
            st.write("Không có ngành phù hợp")
    
    with col2:
        st.markdown("**Ngành Rủi ro cao (Biến động lớn):**")
        risky = sector_df.nlargest(5, 'Biến động (%)')
        for _, row in risky.iterrows():
            st.write(f"- {row['Ngành']}: {row['Biến động (%)']}%")


def show_market_heatmap(df, data_loader_module):
    """
    Hiển thị bản đồ nhiệt (heatmap) của thị trường.
    
    Args:
        df (pd.DataFrame): DataFrame chứa thông tin công ty
        data_loader_module: Module data_loader để lấy dữ liệu giá
    """
    st.subheader("Bản đồ Nhiệt Thị trường")
    
    # Chọn sàn và ngành
    col1, col2 = st.columns(2)
    with col1:
        exchanges = df['exchange'].unique()
        default_index = list(exchanges).index(DEFAULT_MARKET) if DEFAULT_MARKET in exchanges else 0
        exchange = st.selectbox(
            "Chọn sàn giao dịch",
            exchanges,
            index=default_index,
            key="heatmap_exchange"
        )
    
    filtered_df = df[df['exchange'] == exchange]
    
    with col2:
        sector = st.selectbox(
            "Chọn ngành",
            ["Tất cả"] + list(filtered_df['icb_name'].unique()),
            key="heatmap_sector"
        )
    
    # Lọc theo ngành
    if sector != "Tất cả":
        filtered_df = filtered_df[filtered_df['icb_name'] == sector]
    
    # Giới hạn số lượng cổ phiếu
    max_stocks = st.slider("Số lượng cổ phiếu hiển thị", 10, 50, 20, key="heatmap_stocks")
    stocks = filtered_df['symbol'].tolist()[:max_stocks]
    
    if st.button("Tạo Bản đồ Nhiệt", key="create_heatmap"):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        
        st.info(f"Đang tải dữ liệu cho {len(stocks)} cổ phiếu...")
        data, skipped = data_loader_module.fetch_stock_data2(stocks, start_date, end_date)
        
        if data.empty:
            st.error("Không có dữ liệu để hiển thị.")
            return
        
        # Tính toán % thay đổi
        pct_change = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100).sort_values(ascending=False)
        
        # Tạo màu sắc
        colors = ['green' if x > 0 else 'red' for x in pct_change.values]
        
        # Vẽ biểu đồ treemap
        fig = go.Figure(go.Treemap(
            labels=pct_change.index,
            parents=[""] * len(pct_change),
            values=abs(pct_change.values),
            text=[f"{x:.2f}%" for x in pct_change.values],
            textposition="middle center",
            marker=dict(
                colors=pct_change.values,
                colorscale='RdYlGn',
                cmid=0,
                line=dict(width=2)
            )
        ))
        
        fig.update_layout(
            title=f"Bản đồ Nhiệt - {sector if sector != 'Tất cả' else 'Tất cả Ngành'} ({exchange})",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Thống kê
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tăng giá", f"{(pct_change > 0).sum()} CP", f"{(pct_change > 0).sum()/len(pct_change)*100:.1f}%")
        with col2:
            st.metric("Giảm giá", f"{(pct_change < 0).sum()} CP", f"{(pct_change < 0).sum()/len(pct_change)*100:.1f}%")
        with col3:
            st.metric("Trung bình", f"{pct_change.mean():.2f}%")


def show_sector_overview_page(df, data_loader_module):
    """
    Hiển thị trang tổng quan ngành đầy đủ.
    
    Args:
        df (pd.DataFrame): DataFrame chứa thông tin công ty
        data_loader_module: Module data_loader để lấy dữ liệu giá
    """
    st.title("Tổng quan Thị trường & Ngành")
    
    # Tạo tabs
    tab1, tab2 = st.tabs(["Phân tích Ngành", "Bản đồ Nhiệt"])
    
    with tab1:
        analyze_sector_performance(df, data_loader_module)
    
    with tab2:
        show_market_heatmap(df, data_loader_module)
