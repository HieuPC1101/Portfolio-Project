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
import data_loader as dl
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

# Import postgres connector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline')))
from postgres_connector import setup_postgres_connection

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

# Đường dẫn đến file CSV
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
file_path = os.path.join(data_dir, "company_info.csv")

# Lấy dữ liệu từ file CSV
df = fetch_data_from_csv(file_path)

# Khởi tạo session state khi ứng dụng khởi động
initialize_session_state()

# Import plotly cho executive dashboard
import plotly.graph_objects as go
import plotly.express as px

# Ẩn sidebar mặc định
st.set_page_config(page_title="Portfolio Dashboard", layout="wide", initial_sidebar_state="collapsed")


# ==================== EXECUTIVE DASHBOARD CSS ====================
EXECUTIVE_DASHBOARD_CSS = """
<style>
    /* Header styling - adapts to theme */
    .dashboard-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .dashboard-subtitle {
        font-size: 0.9rem;
        opacity: 0.6;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* KPI Card styling - clear borders */
    .kpi-card {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
        border: 2px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2);
        border: 2px solid rgba(102, 126, 234, 0.5);
    }
    
    .kpi-title {
        font-size: 0.85rem;
        opacity: 0.7;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .kpi-change {
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    .kpi-change.positive {
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .kpi-change.negative {
        color: #ef4444;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
</style>
"""


# ==================== EXECUTIVE DASHBOARD FUNCTIONS ====================

def get_market_kpi_data():
    try:
        # Lấy dữ liệu 30 ngày gần nhất
        start_date = pd.to_datetime(ANALYSIS_START_DATE).date()
        end_date = pd.to_datetime(ANALYSIS_END_DATE).date()
        
        market_data = get_market_indices(start_date, end_date)
        if market_data is not None and len(market_data) > 0:
            latest = market_data.iloc[-1]
            return {
                'VN-Index': {
                    'value': float(latest.get('vnindex', 0)),
                    'change': float(latest.get('vnindex_change', 0))
                },
                'VN30': {
                    'value': float(latest.get('vn30', 0)),
                    'change': float(latest.get('vn30_change', 0))
                },
                'HNX': {
                    'value': float(latest.get('hnx_index', 0)),
                    'change': float(latest.get('hnx_index_change', 0))
                },
                'HNX30': {
                    'value': float(latest.get('hnx30', 0)),
                    'change': float(latest.get('hnx30_change', 0))
                }
            }
    except Exception as e:
        st.warning(f"Không thể tải dữ liệu KPI: {e}")
    
    # Fallback data
    return {
        'VN-Index': {'value': 0, 'change': 0},
        'VN30': {'value': 0, 'change': 0},
        'HNX': {'value': 0, 'change': 0},
        'HNX30': {'value': 0, 'change': 0}
    }


def generate_market_indices_kpi():
    """Module hiển thị các chỉ số thị trường chính"""
    indices_data = get_market_kpi_data()
    
    cols = st.columns(4)
    
    for idx, (index_name, data) in enumerate(indices_data.items()):
        with cols[idx]:
            change_class = "positive" if data['change'] >= 0 else "negative"
            change_sign = "+" if data['change'] >= 0 else ""
            
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{index_name}</div>
                <div class="kpi-value">{data['value']:,.2f}</div>
                <div class="kpi-change {change_class}">{change_sign}{data['change']}%</div>
            </div>
            """, unsafe_allow_html=True)


def generate_index_comparison_chart():
    """Module so sánh các chỉ số thị trường từ database"""
    try:
        start_date = pd.to_datetime(ANALYSIS_START_DATE).date()
        end_date = pd.to_datetime(ANALYSIS_END_DATE).date()
        market_data = get_market_indices(start_date, end_date)
        if market_data is not None and not market_data.empty:
            fig = go.Figure()
            
            colors = {
                'vnindex': '#667eea',
                'hnx_index': '#f093fb',
                'hnx30': '#4facfe'
            }
            
            labels = {
                'vnindex': 'VN-INDEX',
                'hnx_index': 'HNX',
                'hnx30': 'HNX30'
            }
            
            for column, color in colors.items():
                if column in market_data.columns:
                    fig.add_trace(go.Scatter(
                        x=market_data['date'],
                        y=market_data[column],
                        mode='lines',
                        name=labels[column],
                        line=dict(color=color, width=2.5),
                        hovertemplate='%{y:.2f}<extra></extra>'
                    ))
            
            fig.update_layout(
                title=dict(
                    text='<b>INDEX COMPARISON</b>',
                    font=dict(size=15, family='Arial, sans-serif'),
                    x=0.02
                ),
                template='plotly_white',
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=11),
                    bordercolor='rgba(102, 126, 234, 0.3)',
                    borderwidth=2
                ),
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.1)',
                    zeroline=False,
                    showline=True,
                    linewidth=2,
                    linecolor='rgba(102, 126, 234, 0.3)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.1)',
                    zeroline=False,
                    showline=True,
                    linewidth=2,
                    linecolor='rgba(102, 126, 234, 0.3)'
                ),
                height=350,
                margin=dict(l=50, r=40, t=60, b=40),
                plot_bgcolor='rgba(102, 126, 234, 0.02)',
                paper_bgcolor='rgba(0, 0, 0, 0)'
            )
            
            return fig
    except Exception as e:
        st.warning(f"Không thể tải dữ liệu so sánh chỉ số: {e}")
    
    # Fallback to sample data
    dates = pd.date_range(start='2024-01-01', end='2024-10-31', freq='D')
    np.random.seed(42)
    base = 30
    
    vn_index = base + np.cumsum(np.random.randn(len(dates)) * 0.3) + np.linspace(0, 15, len(dates))
    hnx = base + np.cumsum(np.random.randn(len(dates)) * 0.25)
    hnx[60:180] = hnx[60:180] + np.linspace(0, 25, 120)
    hnx30 = base - 5 + np.cumsum(np.random.randn(len(dates)) * 0.2) + np.linspace(0, 10, len(dates))
    
    df_sample = pd.DataFrame({
        'Date': dates,
        'VN-INDEX': vn_index,
        'HNX': hnx,
        'HNX30': hnx30
    })
    
    fig = go.Figure()
    
    colors = {
        'VN-INDEX': '#667eea',
        'HNX': '#f093fb',
        'HNX30': '#4facfe'
    }
    
    for column in ['VN-INDEX', 'HNX', 'HNX30']:
        fig.add_trace(go.Scatter(
            x=df_sample['Date'],
            y=df_sample[column],
            mode='lines',
            name=column,
            line=dict(color=colors[column], width=2.5),
            hovertemplate='%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>INDEX COMPARISON</b>',
            font=dict(size=15, family='Arial, sans-serif'),
            x=0.02
        ),
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
            bordercolor='rgba(102, 126, 234, 0.3)',
            borderwidth=2
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.1)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.1)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        height=350,
        margin=dict(l=50, r=40, t=60, b=40),
        plot_bgcolor='rgba(102, 126, 234, 0.02)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    return fig


def generate_vn_index_trend():
    """Module hiển thị xu hướng VN-Index"""
    try:
        start_date = pd.to_datetime(ANALYSIS_START_DATE).date()
        end_date = pd.to_datetime(ANALYSIS_END_DATE).date()
        market_data = get_market_indices(start_date, end_date)
        if market_data is not None and not market_data.empty and 'vnindex' in market_data.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=market_data['date'],
                y=market_data['vnindex'],
                mode='lines',
                name='VN-INDEX',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)',
                hovertemplate='%{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text='<b>VN-INDEX TREND</b>',
                    font=dict(size=15, family='Arial, sans-serif'),
                    x=0.02
                ),
                template='plotly_white',
                showlegend=False,
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.1)',
                    zeroline=False,
                    showline=True,
                    linewidth=2,
                    linecolor='rgba(102, 126, 234, 0.3)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.1)',
                    zeroline=False,
                    showline=True,
                    linewidth=2,
                    linecolor='rgba(102, 126, 234, 0.3)'
                ),
                height=350,
                margin=dict(l=50, r=40, t=60, b=40),
                plot_bgcolor='rgba(102, 126, 234, 0.02)',
                paper_bgcolor='rgba(0, 0, 0, 0)'
            )
            
            return fig
    except Exception as e:
        st.warning(f"Không thể tải dữ liệu VN-Index: {e}")
    
    # Fallback
    dates = pd.date_range(start='2024-01-01', end='2024-10-31', freq='D')
    np.random.seed(42)
    
    base = 20
    trend = np.cumsum(np.random.randn(len(dates)) * 1.5)
    seasonal = 15 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    vn_index_tr = base + trend + seasonal + np.linspace(0, 25, len(dates))
    
    df_sample = pd.DataFrame({
        'Date': dates,
        'VN_INDEX_TR': vn_index_tr
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_sample['Date'],
        y=df_sample['VN_INDEX_TR'],
        mode='lines',
        name='VN-INDEX TR',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>VN-INDEX TREND</b>',
            font=dict(size=15, family='Arial, sans-serif'),
            x=0.02
        ),
        template='plotly_white',
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.1)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.1)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        height=350,
        margin=dict(l=50, r=40, t=60, b=40),
        plot_bgcolor='rgba(102, 126, 234, 0.02)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    return fig


def generate_sector_performance():
    """Module hiển thị hiệu suất các ngành"""
    sectors_data = {
        'Technology': 4.5,
        'Healthcare': 3.7,
        'Consumer': 2.9,
        'Energy': -1.8,
        'Utilities': -2.6,
        'Real estate': -3.1
    }
    
    df_sector = pd.DataFrame(list(sectors_data.items()), columns=['Sector', 'Performance'])
    df_sector = df_sector.sort_values('Performance', ascending=True)
    
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sector['Performance']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sector['Sector'],
        x=df_sector['Performance'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        text=[f"{val}%" for val in df_sector['Performance']],
        textposition='outside',
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>SECTOR PERFORMANCE</b>',
            font=dict(size=15, family='Arial, sans-serif'),
            x=0.02
        ),
        template='plotly_white',
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.1)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(102, 126, 234, 0.3)',
            title='Performance (%)',
            range=[-4, 6],
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        height=350,
        margin=dict(l=120, r=40, t=60, b=40),
        plot_bgcolor='rgba(102, 126, 234, 0.02)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    return fig


def generate_market_cap_treemap():
    try:
        start_date = pd.to_datetime(ANALYSIS_START_DATE).date()
        end_date = pd.to_datetime(ANALYSIS_END_DATE).date()
        companies = dl.get_companies()
        
        if companies.empty or 'symbol' not in companies.columns:
            raise Exception("Không lấy được danh sách công ty từ database")
        
        # Lấy danh sách mã cổ phiếu
        symbols_list = companies['symbol'].tolist()[:20]
        stock_data, _ = dl.fetch_stock_data2(symbols_list, start_date, end_date)
        
        if stock_data.empty:
            raise Exception("Không có dữ liệu giá cổ phiếu")
        
        # Lấy fundamental data để tính market cap
        fundamental_data = fetch_fundamental_data_batch(symbols_list)
        
        # Tính % thay đổi giữa ngày đầu và ngày cuối
        pct_change = ((stock_data.iloc[-1] - stock_data.iloc[0]) / stock_data.iloc[0] * 100)
        
        # Tính market cap cho từng cổ phiếu
        market_caps = {}
        
        for symbol in symbols_list:
            # Lấy giá đóng cửa mới nhất
            closing_price = stock_data[symbol].iloc[-1] if symbol in stock_data.columns else None
            
            # Lấy thông tin từ fundamental data
            market_cap = None
            
            if fundamental_data is not None and not fundamental_data.empty:
                fund_info = fundamental_data[fundamental_data['symbol'] == symbol]
                if not fund_info.empty:
                    # Tính vốn hóa thị trường = EPS * P/E * số lượng cổ phiếu
                    eps = fund_info.iloc[0].get('eps', None)
                    pe = fund_info.iloc[0].get('pe', None)
                    
                    if eps and pe and pd.notna(eps) and pd.notna(pe) and closing_price:
                        # Ước tính từ lợi nhuận và P/E
                        net_profit = fund_info.iloc[0].get('net_profit', None)
                        if net_profit and pd.notna(net_profit):
                            market_cap = abs(float(net_profit)) * float(pe) if float(pe) > 0 else None
            
            # Nếu không có market cap, ước tính từ giá trị giao dịch
            if not market_cap or market_cap <= 0:
                # Ước tính từ giá trung bình
                avg_price = stock_data[symbol].mean() if symbol in stock_data.columns else closing_price
                # Sử dụng giá trung bình * hệ số ước lượng làm proxy
                market_cap = float(avg_price) * 1000000 if avg_price and pd.notna(avg_price) else 1000000
            
            # Lưu market cap
            market_caps[symbol] = market_cap
        
        # Sắp xếp theo % tăng trưởng
        pct_change = pct_change.sort_values(ascending=False)
        
        if pct_change is not None and not pct_change.empty:
            colorscale = [
                [0.0, '#DC3545'],   # Đỏ - giảm
                [0.5, '#FFC107'],   # Vàng - trung tính (0%)
                [1.0, '#28A745']    # Xanh - tăng
            ]  
            
            # Chuẩn bị dữ liệu cho treemap
            df_tree = pd.DataFrame({
                'symbol': pct_change.index,
                'growth_pct': pct_change.values,
                'current_price': stock_data.iloc[-1].values,
                'market_cap': [market_caps.get(sym, 1000000) for sym in pct_change.index]
            })

            def map_color_value(growth):
                if growth < 0:
                    # Giảm: map từ giá trị âm nhỏ nhất về 0 -> [0.0, 0.5)
                    min_negative = df_tree[df_tree['growth_pct'] < 0]['growth_pct'].min() if (df_tree['growth_pct'] < 0).any() else -1
                    if min_negative < 0:
                        return 0.5 * (growth / min_negative)  # 0.0 đến 0.5
                    return 0.25
                elif growth > 0:
                    # Tăng: map từ 0 về giá trị dương lớn nhất -> (0.5, 1.0]
                    max_positive = df_tree[df_tree['growth_pct'] > 0]['growth_pct'].max() if (df_tree['growth_pct'] > 0).any() else 1
                    if max_positive > 0:
                        return 0.5 + 0.5 * (growth / max_positive)  # 0.5 đến 1.0
                    return 0.75
                else:
                    return 0.5  # Đúng 0%
            
            df_tree['color_val'] = df_tree['growth_pct'].apply(map_color_value)
            
            # Tạo treemap
            fig = go.Figure(go.Treemap(
                labels=df_tree['symbol'],
                parents=[''] * len(df_tree),
                values=df_tree['market_cap'],  # Kích thước theo market cap
                text=[f"{row['symbol']}<br>{row['growth_pct']:.1f}%" 
                      for _, row in df_tree.iterrows()],
                textposition='middle center',
                textfont=dict(size=13, color='white', family='Arial, sans-serif', weight='bold'),
                marker=dict(
                    colorscale=colorscale,
                    cmid=0.5,
                    cmin=0,
                    cmax=1,
                    colors=df_tree['color_val'],
                    line=dict(width=3, color='rgba(255, 255, 255, 0.8)')
                ),
                hovertemplate='<b>%{label}</b><br>%{text}<br>Market Cap: %{value:.0f}M VND<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text='<b>STOCK OVERVIEW</b>',
                    font=dict(size=15, family='Arial, sans-serif'),
                    x=0.02
                ),
                template='plotly_white',
                height=350,
                margin=dict(l=10, r=50, t=60, b=10),
                paper_bgcolor='rgba(0, 0, 0, 0)'
            )
            
            return fig
            
    except Exception as e:
        st.warning(f"Không thể tải dữ liệu treemap: {e}")
        
    # Fallback data nếu có lỗi
    fallback_data = {
        'symbol': ['VCB', 'VHM', 'VIC', 'HPG', 'TCB', 'FPT', 'CTG', 'MBB', 'VNM', 'MSN',
                   'PLX', 'VPB', 'GVR', 'POW', 'SAB', 'SSI', 'BID', 'VRE', 'HDB', 'TPB'],
        'growth_pct': [15.2, -8.5, 12.3, 20.1, -5.2, 18.7, 10.5, -3.8, 8.9, 25.4,
                       -12.3, 7.6, 14.2, -6.1, 9.8, 22.5, 5.4, -9.7, 13.1, 16.8],
        'market_cap': [450000, 320000, 280000, 180000, 150000, 140000, 130000, 120000, 110000, 95000,
                       85000, 80000, 75000, 70000, 65000, 60000, 550000, 90000, 125000, 72000]
    }
    
    df_tree = pd.DataFrame(fallback_data)
    
    def map_color_value(growth):
        if growth < 0:
            min_negative = df_tree[df_tree['growth_pct'] < 0]['growth_pct'].min()
            return 0.5 * (growth / min_negative)  # 0.0 đến 0.5
        elif growth > 0:
            max_positive = df_tree[df_tree['growth_pct'] > 0]['growth_pct'].max()
            return 0.5 + 0.5 * (growth / max_positive)  # 0.5 đến 1.0
        else:
            return 0.5  # Đúng 0%
    
    df_tree['color_val'] = df_tree['growth_pct'].apply(map_color_value)
    
    colorscale = [
        [0.0, '#DC3545'],   # Đỏ - giảm
        [0.5, '#FFC107'],   # Vàng - trung tính (0%)
        [1.0, '#28A745']    # Xanh - tăng
    ]
    
    fig = go.Figure(go.Treemap(
        labels=df_tree['symbol'],
        parents=[''] * len(df_tree),
        values=df_tree['market_cap'],
        text=[f"{row['symbol']}<br>{row['growth_pct']:.1f}%" for _, row in df_tree.iterrows()],
        textposition='middle center',
        textfont=dict(size=13, color='white', family='Arial, sans-serif', weight='bold'),
        marker=dict(
            colorscale=colorscale,
            cmid=0.5,
            cmin=0,
            cmax=1,
            colors=df_tree['color_val'],
            line=dict(width=3, color='rgba(255, 255, 255, 0.8)')
        ),
        hovertemplate='<b>%{label}</b><br>%{text}<br>Market Cap: %{value:.0f}M VND<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>TOP STOCK GROWTH BY MARKET CAP</b>',
            font=dict(size=15, family='Arial, sans-serif'),
            x=0.02
        ),
        template='plotly_white',
        height=350,
        margin=dict(l=10, r=50, t=60, b=10),
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    return fig

def generate_net_foreign_buying():
    """Module hiển thị dòng tiền nước ngoài"""
    dates = pd.date_range(start='2024-01-01', end='2024-10-31', freq='D')
    np.random.seed(43)
    
    net_foreign = 150 + np.cumsum(np.random.randn(len(dates)) * 30) + np.linspace(0, 100, len(dates))
    
    df_foreign = pd.DataFrame({
        'Date': dates,
        'Net Foreign': net_foreign
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_foreign['Date'],
        y=df_foreign['Net Foreign'],
        mode='lines',
        name='Net Foreign Buying',
        line=dict(color='#667eea', width=0),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)',
        hovertemplate='%{y:.0f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(102, 126, 234, 0.5)", line_width=2)
    
    fig.update_layout(
        title=dict(
            text='<b>NET FOREIGN BUYING</b>',
            font=dict(size=15, family='Arial, sans-serif'),
            x=0.02
        ),
        template='plotly_white',
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.1)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.1)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(102, 126, 234, 0.3)',
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        height=350,
        margin=dict(l=50, r=40, t=60, b=40),
        plot_bgcolor='rgba(102, 126, 234, 0.02)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    return fig


def generate_inflation_correlation():
    """Module hiển thị tương quan lạm phát"""
    dates = pd.date_range(start='2024-04-01', end='2024-10-31', freq='M')
    
    cpi = [0.25, 0.28, 0.32, 0.38, 0.42, 0.45, 0.50]
    vn_index_corr = [0.15, 0.18, 0.22, 0.28, 0.35, 0.42, 0.48]
    
    df_inflation = pd.DataFrame({
        'Date': dates,
        'CPI': cpi,
        'VN-INDEX': vn_index_corr
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_inflation['Date'],
        y=df_inflation['CPI'],
        mode='lines+markers',
        name='CPI',
        line=dict(color='#f093fb', width=3),
        marker=dict(size=8, color='#f093fb'),
        hovertemplate='CPI: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_inflation['Date'],
        y=df_inflation['VN-INDEX'],
        mode='lines+markers',
        name='VN-INDEX',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#667eea'),
        hovertemplate='VN-INDEX: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>INFLATION & MARKET CORR.</b>',
            font=dict(size=15, family='Arial, sans-serif'),
            x=0.02
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
            bordercolor='rgba(102, 126, 234, 0.3)',
            borderwidth=2
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.1)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.1)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        height=350,
        margin=dict(l=50, r=40, t=60, b=40),
        plot_bgcolor='rgba(102, 126, 234, 0.02)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    return fig


def generate_top_stocks_table():
    """Module hiển thị bảng top cổ phiếu theo tiêu chí"""
    
    try:
            start_date = pd.to_datetime(ANALYSIS_START_DATE).date()
            end_date = pd.to_datetime(ANALYSIS_END_DATE).date()
            companies = dl.get_companies()
            
            if companies.empty or 'symbol' not in companies.columns:
                raise Exception("Không lấy được danh sách công ty")
            
            # Lấy top 50 cổ phiếu
            symbols_list = companies['symbol'].tolist()[:50]
            stock_data, _ = dl.fetch_stock_data2(symbols_list, start_date, end_date)
            
            if not stock_data.empty:
                # Tính % thay đổi 1 ngày
                pct_change_1d = ((stock_data.iloc[-1] - stock_data.iloc[-2]) / stock_data.iloc[-2] * 100)
                
                # Lấy top 10 tăng mạnh nhất
                top_gainers = pct_change_1d.nlargest(10)
                
               
                for idx, (symbol, pct) in enumerate(top_gainers.items(), 1):
                    price = stock_data[symbol].iloc[-1] if symbol in stock_data.columns else 0
                    color = '#10b981' if pct > 0 else '#ef4444' if pct < 0 else '#666'
                    
                 
    except Exception as e:
        st.warning(f"Không thể tải dữ liệu tăng giá: {e}")
    
    # Fallback data
    title = "TOP CỔ PHIẾU TĂNG GIÁ MẠNH"
    data = {
            'STT': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Mã CK': ['AAH', 'ABB', 'ABI', 'ABT', 'ACG', 'AGX', 'ALV', 'AMP', 'ANV', 'APG'],
            'Giá đóng cửa': [3700, 14800, 20100, 73000, 35750, 161800, 6600, 13500, 30400, 11000],
            '% Thay đổi 1D': [2.78, 1.37, 0.50, 1.11, 0.56, 0.37, 1.54, 1.50, 0.66, 0.46]
        }
    df = pd.DataFrame(data)
    # Tạo HTML table với styling
    html_table = f"""
    <div style="background: rgba(102, 126, 234, 0.05); border-radius: 12px; padding: 1rem; border: 2px solid rgba(102, 126, 234, 0.3);">
        <h4 style="margin-bottom: 1rem; color: #667eea; font-size: 14px; font-weight: 700;">{title}</h4>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                <thead>
                    <tr style="background: rgba(102, 126, 234, 0.1); border-bottom: 2px solid rgba(102, 126, 234, 0.3);">
    """
    
    # Add headers
    for col in df.columns:
        html_table += f'<th style="padding: 8px; text-align: left; font-weight: 600;">{col}</th>'
    
    html_table += "</tr></thead><tbody>"
    
    # Add rows
    for idx, row in df.iterrows():
        html_table += '<tr style="border-bottom: 1px solid rgba(128, 128, 128, 0.1);">'
        for col_idx, (col, value) in enumerate(row.items()):
            if col == '% Thay đổi 1D' or col == 'Thay đổi':
                color = '#10b981' if value > 0 else '#ef4444' if value < 0 else '#666'
                html_table += f'<td style="padding: 8px; color: {color}; font-weight: 600;">{value:+.2f}%</td>' if col == '% Thay đổi 1D' else f'<td style="padding: 8px; color: {color}; font-weight: 600;">{value:+,}</td>'
            elif col == 'Mã CK':
                html_table += f'<td style="padding: 8px; font-weight: 700; color: #667eea;">{value}</td>'
            elif isinstance(value, float):
                html_table += f'<td style="padding: 8px;">{value:,.2f}</td>'
            elif isinstance(value, int) and col not in ['STT']:
                html_table += f'<td style="padding: 8px;">{value:,}</td>'
            else:
                html_table += f'<td style="padding: 8px;">{value}</td>'
        html_table += '</tr>'
    
    html_table += """
                </tbody>
            </table>
        </div>
    </div>
    """
    
    return html_table


def generate_correlation_matrix():
    """Module hiển thị ma trận tương quan"""
    sectors = ['Technology', 'Financials', 'Consumer', 'Energy']
    
    correlation_data = [
        [1.0, 0.3, 0.6, 0.8],
        [0.3, 1.0, 0.4, 0.5],
        [0.6, 0.4, 1.0, 0.7],
        [0.8, 0.5, 0.7, 1.0]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data,
        x=sectors,
        y=sectors,
        colorscale=[
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
        ],
        text=correlation_data,
        texttemplate='%{text:.1f}',
        textfont=dict(size=14, weight='bold'),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>',
        colorbar=dict(
            thickness=15,
            len=0.7,
            tickfont=dict(size=11),
            title=dict(text='Corr.', side='right', font=dict(size=11)),
            outlinewidth=2,
            outlinecolor='rgba(102, 126, 234, 0.3)'
        )
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>CORRELATION MATRIX</b>',
            font=dict(size=15, family='Arial, sans-serif'),
            x=0.02
        ),
        template='plotly_white',
        xaxis=dict(
            side='bottom',
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        yaxis=dict(
            showgrid=False,
            autorange='reversed',
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        height=350,
        margin=dict(l=120, r=40, t=60, b=80),
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    return fig


def generate_inflation_heatmap():
    """Module hiển thị heatmap lạm phát"""
    categories = ['TC', 'FC', 'Mat']
    metrics = ['CPI', 'VN-INDEX', '0.1']
    
    heatmap_data = [
        [0.6, 0.5, 0.7],
        [0.8, 0.7, 0.8],
        [0.6, 0.8, 0.8]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=metrics,
        y=categories,
        colorscale=[
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
        ],
        text=heatmap_data,
        texttemplate='%{text:.1f}',
        textfont=dict(size=14, weight='bold'),
        hovertemplate='%{y} - %{x}<br>Value: %{z:.1f}<extra></extra>',
        showscale=False
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>INFLATION & MARKET HEATMAP</b>',
            font=dict(size=15, family='Arial, sans-serif'),
            x=0.02
        ),
        template='plotly_white',
        xaxis=dict(
            side='top',
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(102, 126, 234, 0.3)'
        ),
        height=250,
        margin=dict(l=60, r=40, t=70, b=40),
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    return fig


def show_executive_dashboard():
    """Dashboard chính với layout theo bố cục yêu cầu"""
    
    # Apply CSS
    st.markdown(EXECUTIVE_DASHBOARD_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="dashboard-header">MARKET & SECTOR ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Updated Daily from PostgreSQL Database</div>', unsafe_allow_html=True)
    
    # Row 1: Market Indices KPI (4 cards)
    generate_market_indices_kpi()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2: Main Charts (3 columns)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.plotly_chart(generate_index_comparison_chart(), width='stretch')
        st.plotly_chart(generate_market_cap_treemap(), width='stretch')
    
    with col2:
        st.plotly_chart(generate_vn_index_trend(), width='stretch')
        st.plotly_chart(generate_net_foreign_buying(), width='stretch')
    
    with col3:
        st.plotly_chart(generate_sector_performance(), width='stretch')
        st.markdown(generate_top_stocks_table(), unsafe_allow_html=True)
        
    
    st.markdown("<br>", unsafe_allow_html=True)
    

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

# Header
st.title("📊 Portfolio Investment Dashboard")
st.markdown("---")

# Khởi tạo active tab trong session state
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Tạo tabs chính với on_change callback
def on_tab_change():
    # Hàm này sẽ được gọi khi user click vào tab khác
    pass

# Tạo tabs
tab_names = ["📈 Market Overview", "🎯 Tối Ưu Danh Mục", "🤖 Trợ Lý AI"]

# Sử dụng radio button ẩn để tracking tab
selected_tab = st.radio("Navigation", tab_names, key="tab_selector", 
                        label_visibility="collapsed",
                        horizontal=True,
                        index=st.session_state.active_tab)

# Cập nhật active tab
if selected_tab == tab_names[0]:
    st.session_state.active_tab = 0
elif selected_tab == tab_names[1]:
    st.session_state.active_tab = 1
elif selected_tab == tab_names[2]:
    st.session_state.active_tab = 2

# ==================== TAB 1: MARKET OVERVIEW ====================
if st.session_state.active_tab == 0:
    st.markdown("### Tổng Quan Thị Trường")
    
    # Executive Dashboard
    show_executive_dashboard()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")


# ==================== TAB 2: TỐI ỨU DANH MỤC ====================
elif st.session_state.active_tab == 1:
    st.markdown("### Tối Ưu Hóa Danh Mục Đầu Tư")
    
    # Hiển thị sidebar cho tab này
    st.sidebar.info("📊 Sử dụng dữ liệu từ PostgreSQL Database")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Cấu Hình Danh Mục")
        
    portfolio_mode = st.sidebar.radio(
        "Chế độ lựa chọn",
        ["Tự chọn mã cổ phiếu", "Hệ thống đề xuất tự động"],
        key="portfolio_mode"
    )
    
    if portfolio_mode == "Tự chọn mã cổ phiếu":
        # Giao diện người dùng để lọc từ file CSV
        st.subheader("📝 Tự Chọn Mã Cổ Phiếu")
        
        # Sidebar filters
        st.sidebar.markdown("#### Bộ Lọc")
        
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
        
        selected_exchange = st.sidebar.selectbox('Chọn sàn giao dịch', exchanges, index=default_index, key="manual_exchange")

        # Lọc dữ liệu dựa trên sàn giao dịch đã chọn
        filtered_df = df[df['exchange'] == selected_exchange]

        # Bộ lọc theo loại ngành (icb_name)
        icb_names = filtered_df['icb_name'].unique()
        saved_icb = filter_state.get('icb_name')
        if saved_icb and saved_icb in icb_names:
            default_icb_index = list(icb_names).index(saved_icb)
        else:
            default_icb_index = 0
        
        selected_icb_name = st.sidebar.selectbox('Chọn ngành', icb_names, index=default_icb_index, key="manual_icb")

        # Lọc dữ liệu dựa trên ngành đã chọn
        filtered_df = filtered_df[filtered_df['icb_name'] == selected_icb_name]
        
        st.sidebar.markdown("---")
        
        # Bộ lọc theo mã chứng khoán (symbol)
        selected_symbols = st.sidebar.multiselect('Chọn mã chứng khoán', filtered_df['symbol'], key="manual_symbols")

        # Lưu các mã chứng khoán đã chọn vào session state khi nhấn nút "Thêm mã"
        if st.sidebar.button("Thêm mã vào danh sách", key="manual_add"):
            for symbol in selected_symbols:
                if symbol not in st.session_state.selected_stocks:
                    st.session_state.selected_stocks.append(symbol)
            st.success(f"Đã thêm {len(selected_symbols)} mã cổ phiếu vào danh mục!")

        st.sidebar.markdown("---")
        
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
            max_value=today,
            key="manual_start"
        )
        end_date = st.sidebar.date_input(
            "Ngày kết thúc", 
            value=default_end, 
            max_value=today,
            key="manual_end"
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

        # Hiển thị danh sách mã cổ phiếu đã chọn và xử lý thao tác xóa
        display_selected_stocks(df)
        
        # Gọi hàm chính
        main_manual_selection()
    
    else:  # Hệ thống đề xuất tự động
        # Giao diện Streamlit
        st.subheader("🤖 Hệ Thống Đề Xuất Tự Động")
        
        st.sidebar.markdown("#### Cấu Hình Đề Xuất")

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
                default=saved_exchanges,
                key="auto_exchanges"
            )

            # Lọc dữ liệu theo nhiều sàn giao dịch
            filtered_df = df[df['exchange'].isin(selected_exchanges)]

            # Bước 2: Chọn nhiều ngành
            saved_sectors = auto_state.get('sectors', [])
            selected_sectors = st.sidebar.multiselect("Chọn ngành", filtered_df['icb_name'].unique(), default=saved_sectors, key="auto_sectors")

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
                        key=f"auto_stocks_{sector}"
                    )

                # Bước 4: Chọn cách lọc
                saved_filter_method = auto_state.get('filter_method', 'Lợi nhuận lớn nhất')
                filter_method_options = ["Lợi nhuận lớn nhất", "Rủi ro bé nhất"]
                default_method_index = filter_method_options.index(saved_filter_method) if saved_filter_method in filter_method_options else 0
                
                filter_method = st.sidebar.radio(
                    "Cách lọc cổ phiếu", 
                    filter_method_options,
                    index=default_method_index,
                    key="auto_filter_method"
                )

                st.sidebar.markdown("---")

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
                    key="auto_start_date"
                )
                end_date = st.sidebar.date_input(
                    "Ngày kết thúc", 
                    value=default_end_1,
                    min_value=min_date,
                    max_value=max_date,
                    key="auto_end_date"
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

                if st.sidebar.button("Đề xuất cổ phiếu", key="auto_recommend"):
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
            if st.button("Xóa hết các cổ phiếu đã được đề xuất", key="auto_clear"):
                st.session_state.final_selected_stocks = {}
                st.success("Đã xóa hết tất cả cổ phiếu khỏi danh sách!")
            
            for exchange, sectors in st.session_state.final_selected_stocks.items():
                st.write(f"### Sàn: {exchange}")
                for sector, stocks in sectors.items():
                    st.write(f"**{sector}**: {', '.join(stocks)}")

        display_selected_stocks_2(df)
        
        # Gọi hàm chính
        main_auto_selection()


# ==================== TAB 3: TRỢ LÝ AI ====================
elif st.session_state.active_tab == 2:
    st.markdown("### 🤖 Trợ Lý AI")
    
    # Sidebar riêng cho Tab 3
    if st.session_state.get("chatbot") is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### ⚙️ Tiện Ích AI")
        controls_container = st.sidebar.container()
        render_chat_controls(controls_container, key_prefix="tab3_sidebar")
    
    # Hiển thị trang chatbot
    render_chatbot_page()
