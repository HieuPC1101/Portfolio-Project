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


def show_market_heatmap(df, data_loader_module):
    """
    Hiển thị Biểu đồ nhiệt (heatmap) của thị trường.
    
    Args:
        df (pd.DataFrame): DataFrame chứa thông tin công ty
        data_loader_module: Module data_loader để lấy dữ liệu giá
    """
    st.subheader("Biểu đồ Nhiệt Thị trường")
    
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
    
    if st.button("Tạo Biểu đồ Nhiệt", key="create_heatmap"):
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
            title=f"Biểu đồ Nhiệt - {sector if sector != 'Tất cả' else 'Tất cả Ngành'} ({exchange})",
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


def show_sector_treemap(df, data_loader_module):
    """
    Hiển thị biểu đồ cây phân cấp theo ngành và công ty.
    Cấp 1: Ngành
    Cấp 2: Công ty trong ngành
    Màu sắc: Tăng trưởng (xanh = tăng, đỏ = giảm)
    Kích thước: Doanh thu hoặc giá trị giao dịch
    
    Args:
        df (pd.DataFrame): DataFrame chứa thông tin công ty
        data_loader_module: Module data_loader để lấy dữ liệu giá
    """
    st.subheader("Biểu đồ Cây Phân tích Ngành")
    st.markdown("*Biểu đồ phân cấp: Ngành > Công ty. Màu sắc thể hiện tăng trưởng, kích thước thể hiện doanh thu.*")
    
    # Chọn sàn giao dịch
    col1, col2 = st.columns(2)
    with col1:
        exchanges = df['exchange'].unique()
        default_index = list(exchanges).index(DEFAULT_MARKET) if DEFAULT_MARKET in exchanges else 0
        exchange = st.selectbox(
            "Chọn sàn giao dịch",
            exchanges,
            index=default_index,
            key="treemap_exchange"
        )
    
    with col2:
        analysis_period = st.selectbox(
            "Khoảng thời gian tính tăng trưởng",
            ["1 Tuần", "1 Tháng", "3 Tháng", "6 Tháng"],
            key="treemap_period"
        )
    
    # Lọc theo sàn
    filtered_df = df[df['exchange'] == exchange]
    
    # Giới hạn số lượng cổ phiếu mỗi ngành
    max_stocks_per_sector = st.slider(
        "Số lượng công ty tối đa mỗi ngành",
        5, 20, 10,
        key="treemap_stocks_per_sector"
    )
    
    # Chọn ngành để phân tích (tối đa 5 ngành)
    all_sectors = list(filtered_df['icb_name'].unique())
    selected_sectors = st.multiselect(
        "Chọn ngành để phân tích (tối đa 5 ngành)",
        all_sectors,
        default=all_sectors[:5] if len(all_sectors) >= 5 else all_sectors,
        max_selections=5,
        key="treemap_sectors"
    )
    
    if not selected_sectors:
        st.warning("Vui lòng chọn ít nhất một ngành để phân tích.")
        return
    
    if st.button("Tạo Biểu đồ Cây", key="create_treemap"):
        # Tính toán ngày
        end_date = datetime.now().date()
        period_map = {
            "1 Tuần": 7,
            "1 Tháng": 30,
            "3 Tháng": 90,
            "6 Tháng": 180
        }
        start_date = end_date - timedelta(days=period_map[analysis_period])
        
        treemap_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, sector in enumerate(selected_sectors):
            status_text.text(f"Đang xử lý ngành {sector}... ({idx+1}/{len(selected_sectors)})")
            
            # Lấy danh sách công ty trong ngành
            sector_companies = filtered_df[filtered_df['icb_name'] == sector]['symbol'].tolist()[:max_stocks_per_sector]
            
            if not sector_companies:
                continue
            
            # Lấy dữ liệu giá
            price_data, skipped = data_loader_module.fetch_stock_data2(
                sector_companies,
                start_date,
                end_date
            )
            
            if price_data.empty:
                continue
            
            # Lấy dữ liệu phân tích cơ bản cho doanh thu
            fundamental_data = data_loader_module.fetch_fundamental_data_batch(sector_companies)
            
            for company in sector_companies:
                if company not in price_data.columns:
                    continue
                
                # Tính % thay đổi giá
                company_prices = price_data[company].dropna()
                if len(company_prices) < 2:
                    continue
                
                price_change = ((company_prices.iloc[-1] - company_prices.iloc[0]) / company_prices.iloc[0]) * 100
                
                # Lấy doanh thu (nếu có)
                revenue = 1  # Giá trị mặc định
                if fundamental_data is not None and not fundamental_data.empty:
                    company_fund = fundamental_data[fundamental_data['symbol'] == company]
                    if not company_fund.empty and pd.notna(company_fund.iloc[0].get('revenue')):
                        revenue = abs(float(company_fund.iloc[0]['revenue']))
                
                # Nếu không có doanh thu, dùng giá trị trung bình giao dịch
                if revenue <= 1:
                    revenue = float(company_prices.mean())
                
                # Lấy tên công ty
                company_name = filtered_df[filtered_df['symbol'] == company]['organ_name'].values
                company_display = company_name[0] if len(company_name) > 0 else company
                
                treemap_data.append({
                    'labels': f"{company}<br>{company_display[:30]}",
                    'parents': sector,
                    'values': revenue,
                    'growth': price_change,
                    'text': f"{company}<br>{price_change:.2f}%"
                })
            
            # Thêm node ngành (cha)
            sector_total_revenue = sum([item['values'] for item in treemap_data if item['parents'] == sector])
            sector_avg_growth = sum([item['growth'] for item in treemap_data if item['parents'] == sector]) / max(len([item for item in treemap_data if item['parents'] == sector]), 1)
            
            treemap_data.append({
                'labels': sector,
                'parents': '',
                'values': sector_total_revenue,
                'growth': sector_avg_growth,
                'text': f"{sector}<br>{sector_avg_growth:.2f}%"
            })
            
            progress_bar.progress((idx + 1) / len(selected_sectors))
        
        progress_bar.empty()
        status_text.empty()
        
        if not treemap_data:
            st.error("Không có dữ liệu để hiển thị biểu đồ cây.")
            return
        
        # Tạo DataFrame
        treemap_df = pd.DataFrame(treemap_data)
        
        # Tạo biểu đồ Treemap
        fig = go.Figure(go.Treemap(
            labels=treemap_df['labels'],
            parents=treemap_df['parents'],
            values=treemap_df['values'],
            text=treemap_df['text'],
            textposition="middle center",
            marker=dict(
                colors=treemap_df['growth'],
                colorscale=[
                    [0, '#d32f2f'],      # Đỏ đậm (giảm mạnh)
                    [0.25, '#ef5350'],   # Đỏ nhạt
                    [0.45, '#ffeb3b'],   # Vàng
                    [0.5, '#ffffff'],    # Trắng (không thay đổi)
                    [0.55, '#c8e6c9'],   # Xanh nhạt
                    [0.75, '#66bb6a'],   # Xanh lá
                    [1, '#2e7d32']       # Xanh đậm (tăng mạnh)
                ],
                cmid=0,
                colorbar=dict(
                    title="Tăng trưởng (%)",
                    thickness=20,
                    len=0.7
                ),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{label}</b><br>Tăng trưởng: %{color:.2f}%<br>Giá trị: %{value:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Biểu đồ Cây Phân tích Ngành - {exchange} ({analysis_period})",
            height=700,
            font=dict(size=11)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Thống kê tổng quan
        st.markdown("### Thống kê Tổng quan")
        col1, col2, col3, col4 = st.columns(4)
        
        companies_data = treemap_df[treemap_df['parents'] != '']
        
        with col1:
            st.metric(
                "Tổng số công ty",
                len(companies_data)
            )
        
        with col2:
            positive = len(companies_data[companies_data['growth'] > 0])
            st.metric(
                "Số CP tăng giá",
                positive,
                f"{positive/len(companies_data)*100:.1f}%"
            )
        
        with col3:
            st.metric(
                "Tăng trưởng TB",
                f"{companies_data['growth'].mean():.2f}%"
            )
        
        with col4:
            best_company = companies_data.loc[companies_data['growth'].idxmax()]
            st.metric(
                "Tăng mạnh nhất",
                best_company['labels'].split('<br>')[0],
                f"+{best_company['growth']:.2f}%"
            )
        
        # Top 5 công ty
        st.markdown("### Top 5 Công ty Tốt nhất")
        top_companies = companies_data.nlargest(5, 'growth')[['labels', 'growth', 'parents']]
        top_companies['Mã CP'] = top_companies['labels'].apply(lambda x: x.split('<br>')[0])
        top_companies['Tên'] = top_companies['labels'].apply(lambda x: x.split('<br>')[1] if '<br>' in x else '')
        top_companies['Tăng trưởng (%)'] = top_companies['growth'].round(2)
        top_companies['Ngành'] = top_companies['parents']
        
        st.dataframe(
            top_companies[['Mã CP', 'Tên', 'Ngành', 'Tăng trưởng (%)']],
            use_container_width=True,
            hide_index=True
        )


def show_sector_overview_page(df, data_loader_module):
    """
    Hiển thị trang tổng quan ngành đầy đủ.
    
    Args:
        df (pd.DataFrame): DataFrame chứa thông tin công ty
        data_loader_module: Module data_loader để lấy dữ liệu giá
    """
    st.title("Tổng quan Thị trường & Ngành")
    
    # Tạo tabs
    tab1, tab2 = st.tabs(["Biểu đồ Nhiệt", "Biểu đồ Cây"])
    
    with tab1:
        show_market_heatmap(df, data_loader_module)
    
    with tab2:
        show_sector_treemap(df, data_loader_module)
