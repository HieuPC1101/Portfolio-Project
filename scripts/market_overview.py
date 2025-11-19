"""
Module market_overview.py
Tổng quan thị trường sử dụng dữ liệu từ PostgreSQL.
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scripts.config import ANALYSIS_START_DATE, ANALYSIS_END_DATE, DEFAULT_MARKET, DEFAULT_INVESTMENT_AMOUNT
from scripts.session_manager import save_market_overview_state, get_market_overview_state


def show_market_heatmap(df, data_loader_module, preselected_sector=None, preselected_exchange=None):
    """
    Hiển thị Biểu đồ nhiệt (heatmap) của thị trường từ database.
    """
    st.subheader("Biểu đồ Nhiệt Thị trường (Database)")
    
    if preselected_sector:
        st.info(f"Đang hiển thị chi tiết ngành: **{preselected_sector}**")
    
    # Chọn sàn và ngành
    col1, col2 = st.columns(2)
    with col1:
        exchanges = df['exchange'].unique()
        # Ưu tiên sàn được chọn trước, nếu không thì dùng mặc định
        if preselected_exchange and preselected_exchange in exchanges:
            default_index = list(exchanges).index(preselected_exchange)
        else:
            default_index = list(exchanges).index(DEFAULT_MARKET) if DEFAULT_MARKET in exchanges else 0
        exchange = st.selectbox(
            "Chọn sàn giao dịch",
            exchanges,
            index=default_index,
            key="heatmap_exchange"
        )
    
    filtered_df = df[df['exchange'] == exchange]
    
    with col2:
        sectors_list = ["Tất cả"] + list(filtered_df['icb_name'].unique())
        # Ưu tiên ngành được chọn trước
        if preselected_sector and preselected_sector in sectors_list:
            default_sector_index = sectors_list.index(preselected_sector)
        else:
            default_sector_index = 0
        sector = st.selectbox(
            "Chọn ngành",
            sectors_list,
            index=default_sector_index,
            key="heatmap_sector"
        )
    
    # Lọc theo ngành
    if sector != "Tất cả":
        filtered_df = filtered_df[filtered_df['icb_name'] == sector]
    
    # Giới hạn số lượng mã cổ phiếu
    max_stocks = st.slider("Số lượng mã cổ phiếu hiển thị", 10, 50, 20, key="heatmap_stocks")
    stocks = filtered_df['symbol'].tolist()[:max_stocks]
    
    if st.button("Tạo Biểu đồ Nhiệt", key="create_heatmap") or preselected_sector:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
    
        st.info(f"Đang tải dữ liệu cho {len(stocks)} mã cổ phiếu từ database...")
        data, skipped = data_loader_module.fetch_stock_data2(stocks, start_date, end_date)

        if data.empty:
            st.error("Không có dữ liệu trong database.")
            return

        # Tính % thay đổi
        pct_change = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100).sort_values(ascending=False)

        # Lấy thông tin vốn hóa từ database
        st.info("Đang lấy thông tin vốn hóa thị trường từ database...")
        tooltip_data = {}
        market_caps = {}
        
        fundamental_data = data_loader_module.fetch_fundamental_data_batch(list(pct_change.index))
        
        for symbol in pct_change.index:
            company_info = filtered_df[filtered_df['symbol'] == symbol]
            if company_info.empty:
                continue
            
            closing_price = data[symbol].iloc[-1] if symbol in data.columns else None
            market_cap = None
            
            if fundamental_data is not None and not fundamental_data.empty:
                fund_row = fundamental_data[fundamental_data['symbol'] == symbol]
                if not fund_row.empty and fund_row.iloc[0].get('market_cap'):
                    market_cap = float(fund_row.iloc[0]['market_cap'])
            
            # Nếu không có, ước tính
            if not market_cap or market_cap <= 0:
                if closing_price:
                    market_cap = closing_price * 1000000000
                else:
                    market_cap = 1000000000
            
            market_caps[symbol] = market_cap
            
            company_name = company_info.iloc[0]['organ_name']
            tooltip_data[symbol] = {
                'company_name': company_name,
                'market_cap': market_cap,
                'change_pct': pct_change[symbol]
            }
        
        # Tạo hover text và treemap values
        hover_texts = []
        treemap_values = []
        
        for symbol in pct_change.index:
            if symbol in tooltip_data:
                info = tooltip_data[symbol]
                hover_text = (
                    f"<b>{symbol}</b><br>"
                    f"{info['company_name']}<br>"
                    f"Vốn hóa: {info['market_cap']/1e12:.2f} nghìn tỷ VND<br>"
                    f"Thay đổi: {info['change_pct']:.2f}%"
                )
                treemap_values.append(info['market_cap'])
            else:
                hover_text = f"<b>{symbol}</b><br>Không có dữ liệu"
                treemap_values.append(1000000000)
            hover_texts.append(hover_text)
        
        # Colorscale
        custom_colorscale = [
            [0.0, '#8B0000'],
            [0.15, '#DC143C'],
            [0.25, '#FF6347'],
            [0.35, '#FFA07A'],
            [0.45, '#F5DEB3'],
            [0.48, '#E8E8E8'],
            [0.50, '#D3D3D3'],
            [0.52, '#E8E8E8'],
            [0.55, '#E0F2E9'],
            [0.65, '#B2DFDB'],
            [0.75, '#66BB6A'],
            [0.85, '#43A047'],
            [1.0, '#2E7D32']
        ]
        
        fig = go.Figure(go.Treemap(
            labels=pct_change.index,
            parents=[""] * len(pct_change),
            values=treemap_values,
            text=[f"{x:.2f}%" for x in pct_change.values],
            textposition="middle center",
            hovertext=hover_texts,
            hovertemplate='%{hovertext}<extra></extra>',
            marker=dict(
                colors=pct_change.values,
                colorscale=custom_colorscale,
                cmid=0,
                cmin=max(pct_change.min(), -10),
                cmax=min(pct_change.max(), 10),
                colorbar=dict(
                    title=dict(text="% Thay đổi", side="right"),
                    tickmode="linear",
                    tick0=-10,
                    dtick=2,
                    thickness=15,
                    len=0.7,
                    x=1.02
                ),
                line=dict(width=2, color='white')
            )
        ))
        
        fig.update_layout(
            title=f"Biểu đồ Nhiệt - {sector if sector != 'Tất cả' else 'Tất cả Ngành'} ({exchange}) [Database]<br>"
                  f"<sub>Kích thước ô = Vốn hóa thị trường | Màu sắc: Đỏ (giảm) → Xám (0) → Xanh (tăng)</sub>",
            height=600,
            font=dict(size=11)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Thống kê
        st.markdown("### Thống kê Thị trường")
        num_increase = (pct_change > 0.5).sum()
        num_decrease = (pct_change < -0.5).sum()
        num_light_fluctuation = ((pct_change >= -0.5) & (pct_change <= 0.5)).sum()
        total = len(pct_change)

        total_market_cap = sum(market_caps.values())
        weighted_avg_change = sum(pct_change[symbol] * market_caps.get(symbol, 0) / total_market_cap 
                                  for symbol in pct_change.index if symbol in market_caps)
        simple_avg_change = pct_change.mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tăng giá", f"{num_increase} mã", f"{num_increase/total*100:.1f}%")
        with col2:
            st.metric("Giảm giá", f"{num_decrease} mã", f"{num_decrease/total*100:.1f}%")
        with col3:
            st.metric("Biến động nhẹ", f"{num_light_fluctuation} mã", f"{num_light_fluctuation/total*100:.1f}%")
        with col4:
            st.metric("Tổng số", f"{total} mã")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Thay đổi TB (Đơn giản)", f"{simple_avg_change:.2f}%")
        with col2:
            st.metric("Thay đổi TB (Theo Vốn hóa)", f"{weighted_avg_change:.2f}%")


def show_sector_treemap(df, data_loader_module):
    """
    Hiển thị biểu đồ cây phân cấp từ database.
    """
    st.subheader("Biểu đồ Cây Phân tích Hiệu Suất Ngành (Database)")
    st.markdown("*Dữ liệu từ PostgreSQL*")
    
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
            
            fundamental_data = data_loader_module.fetch_fundamental_data_batch(sector_companies)
            
            sector_total_market_cap = 0
            sector_weighted_growth = 0
            sector_company_count = 0
            
            for company in sector_companies:
                if company not in price_data.columns:
                    continue
                
                company_price_data = price_data[company].dropna()
                if len(company_price_data) < 2:
                    continue
                
                growth = ((company_price_data.iloc[-1] - company_price_data.iloc[0]) / company_price_data.iloc[0]) * 100
                
                market_cap = None
                if fundamental_data is not None and not fundamental_data.empty:
                    fund_row = fundamental_data[fundamental_data['symbol'] == company]
                    if not fund_row.empty and fund_row.iloc[0].get('market_cap'):
                        market_cap = float(fund_row.iloc[0]['market_cap'])
                
                if not market_cap or market_cap <= 0:
                    market_cap = company_price_data.iloc[-1] * 1000000000
                
                company_name = filtered_df[filtered_df['symbol'] == company]['organ_name'].iloc[0] if not filtered_df[filtered_df['symbol'] == company].empty else company
                
                treemap_data.append({
                    'labels': f"{company}<br>{growth:.2f}%",
                    'parents': sector,
                    'values': market_cap,
                    'growth': growth,
                    'text': f"{growth:.2f}%",
                    'hover': f"<b>{company}</b><br>{company_name}<br>Vốn hóa: {market_cap/1e12:.2f} nghìn tỷ<br>Tăng trưởng: {growth:.2f}%",
                    'sector': sector,
                    'symbol': company,
                    'company_name': company_name
                })
                
                sector_total_market_cap += market_cap
                sector_weighted_growth += growth * market_cap
                sector_company_count += 1
            
            if sector_company_count > 0:
                avg_growth = sector_weighted_growth / sector_total_market_cap
                treemap_data.append({
                    'labels': f"{sector}<br>{avg_growth:.2f}%",
                    'parents': '',
                    'values': sector_total_market_cap,
                    'growth': avg_growth,
                    'text': f"{avg_growth:.2f}%",
                    'hover': f"<b>{sector}</b><br>Vốn hóa: {sector_total_market_cap/1e12:.2f} nghìn tỷ<br>Tăng trưởng TB: {avg_growth:.2f}%<br>Số công ty: {sector_company_count}",
                    'sector': sector,
                    'symbol': '',
                    'company_name': ''
                })
            
            progress_bar.progress((idx + 1) / len(selected_sectors))
        
        progress_bar.empty()
        status_text.empty()
        
        treemap_df = pd.DataFrame(treemap_data)
        
        fig = go.Figure(go.Treemap(
            labels=treemap_df['labels'],
            parents=treemap_df['parents'],
            values=treemap_df['values'],
            text=treemap_df['text'],
            textposition="middle center",
            textfont=dict(size=11, family="Arial"),
            hovertext=treemap_df['hover'],
            hovertemplate='%{hovertext}<extra></extra>',
            marker=dict(
                colors=treemap_df['growth'],
                colorscale=[
                    [0, '#d32f2f'],
                    [0.25, '#ef5350'],
                    [0.45, '#ffcdd2'],
                    [0.5, '#f5f5f5'],
                    [0.55, '#c8e6c9'],
                    [0.75, '#66bb6a'],
                    [1, '#2e7d32']
                ],
                cmid=0,
                colorbar=dict(
                    title=dict(text="% Tăng trưởng", side="right", font=dict(size=12)),
                    thickness=20,
                    len=0.7
                ),
                line=dict(width=3, color='white')
            )
        ))
        
        fig.update_layout(
            title=f"Biểu đồ Cây - {exchange} ({analysis_period}) [Database]",
            height=700,
            font=dict(size=11)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Thống kê
        st.markdown("### Thống kê Tổng quan")
        companies_data = treemap_df[treemap_df['parents'] != '']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng số công ty", len(companies_data))
        with col2:
            st.metric("Tăng trưởng TB", f"{companies_data['growth'].mean():.2f}%")
        with col3:
            st.metric("Cao nhất", f"{companies_data['growth'].max():.2f}%")
        with col4:
            st.metric("Thấp nhất", f"{companies_data['growth'].min():.2f}%")


def show_sector_overview_page(df, data_loader_module):
    """
    Hiển thị trang tổng quan ngành từ database.
    """
    st.title("Tổng quan Thị trường & Ngành (Database)")
    
    market_state = get_market_overview_state()
    drilldown_sector = st.session_state.get('drilldown_sector', None)
    drilldown_exchange = st.session_state.get('drilldown_exchange', None)
    
    if drilldown_sector:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("⬅ Quay lại"):
                st.session_state.drilldown_sector = None
                st.session_state.drilldown_exchange = None
                st.rerun()
        with col2:
            st.info(f"Chi tiết ngành: **{drilldown_sector}**")
        
        save_market_overview_state(drilldown_exchange, 'sector')
    
    tab1, tab2 = st.tabs(["Biểu đồ Nhiệt", "Biểu đồ Cây"])
    
    if drilldown_sector:
        with tab1:
            show_market_heatmap(df, data_loader_module, 
                              preselected_sector=drilldown_sector,
                              preselected_exchange=drilldown_exchange)
        with tab2:
            show_sector_treemap(df, data_loader_module)
    else:
        with tab1:
            show_market_heatmap(df, data_loader_module)
        with tab2:
            show_sector_treemap(df, data_loader_module)
