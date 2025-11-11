"""
Executive Dashboard - Market & Sector Analysis
Dashboard hiện đại với giao diện trực quan, chia theo module
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CSS STYLING ====================
st.markdown("""
<style>
    /* Background và font chính */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Header styling */
    .dashboard-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    
    .dashboard-subtitle {
        font-size: 0.9rem;
        color: #8b92a8;
        margin-bottom: 2rem;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #2d3748;
        transition: transform 0.2s;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        border: 1px solid #4a5568;
    }
    
    .kpi-title {
        font-size: 0.85rem;
        color: #a0aec0;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    
    .kpi-change {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .kpi-change.positive {
        color: #48bb78;
    }
    
    .kpi-change.negative {
        color: #f56565;
    }
    
    /* Chart container */
    .chart-container {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #2d3748;
    }
    
    .chart-title {
        font-size: 1rem;
        color: #e2e8f0;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Sector performance bars */
    .sector-bar {
        background: #2d3748;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1f2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #718096;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #1a1f2e;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #a0aec0;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff;
        border-bottom: 3px solid #4299e1;
    }
</style>
""", unsafe_allow_html=True)


# ==================== MODULE 1: MARKET INDICES KPI ====================
def generate_market_indices_kpi():
    """Module hiển thị các chỉ số thị trường chính"""
    
    # Dữ liệu mẫu - thay thế bằng dữ liệu thực tế
    indices_data = {
        'VN-Index': {'value': 1120.45, 'change': 1.3},
        'VN30': {'value': 1120.58, 'change': 1.3},
        'HNX': {'value': 228.60, 'change': 2.1},
        'UPCOM': {'value': 85.32, 'change': 0.4}
    }
    
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


# ==================== MODULE 2: INDEX COMPARISON CHART ====================
def generate_index_comparison_chart():
    """Module so sánh các chỉ số thị trường"""
    
    # Tạo dữ liệu mẫu
    dates = pd.date_range(start='2024-01-01', end='2024-10-31', freq='D')
    
    # Tạo dữ liệu với xu hướng khác nhau
    np.random.seed(42)
    base = 30
    
    # VN-Index: tăng trưởng ổn định
    vn_index = base + np.cumsum(np.random.randn(len(dates)) * 0.3) + np.linspace(0, 15, len(dates))
    
    # HNX: tăng mạnh từ tháng 3-6
    hnx = base + np.cumsum(np.random.randn(len(dates)) * 0.25)
    hnx[60:180] = hnx[60:180] + np.linspace(0, 25, 120)
    
    # UPCOM: ổn định, ít biến động
    upcom = base - 10 + np.cumsum(np.random.randn(len(dates)) * 0.15)
    
    df = pd.DataFrame({
        'Date': dates,
        'VN-INDEX': vn_index,
        'HNX': hnx,
        'UPCOM': upcom
    })
    
    fig = go.Figure()
    
    # Màu sắc hiện đại cho từng chỉ số
    colors = {
        'VN-INDEX': '#4299e1',  # Blue
        'HNX': '#f6ad55',        # Orange
        'UPCOM': '#fc8181'       # Red
    }
    
    for column in ['VN-INDEX', 'HNX', 'UPCOM']:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df[column],
            mode='lines',
            name=column,
            line=dict(color=colors[column], width=2.5),
            hovertemplate='%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='INDEX COMPARISON',
            font=dict(size=14, color='#e2e8f0', family='Arial, sans-serif'),
            x=0
        ),
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#1a1f2e',
        font=dict(color='#a0aec0', size=11),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
            bgcolor='rgba(26, 31, 46, 0.8)'
        ),
        xaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=False,
            range=[10, 60]
        ),
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    # Thêm annotation cho insight
    fig.add_annotation(
        x=dates[120],
        y=52,
        text="Tech outperforming<br>from Mar-Jun",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#f6ad55',
        font=dict(size=10, color='#f6ad55'),
        bgcolor='#1a1f2e',
        bordercolor='#f6ad55',
        borderwidth=1
    )
    
    return fig


# ==================== MODULE 3: VN-INDEX TREND ====================
def generate_vn_index_trend():
    """Module hiển thị xu hướng VN-Index"""
    
    dates = pd.date_range(start='2024-01-01', end='2024-10-31', freq='D')
    np.random.seed(42)
    
    # Tạo xu hướng với biến động
    base = 20
    trend = np.cumsum(np.random.randn(len(dates)) * 1.5)
    seasonal = 15 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    vn_index_tr = base + trend + seasonal + np.linspace(0, 25, len(dates))
    
    df = pd.DataFrame({
        'Date': dates,
        'VN_INDEX_TR': vn_index_tr
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['VN_INDEX_TR'],
        mode='lines',
        name='VN-INDEX TR',
        line=dict(color='#4299e1', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(66, 153, 225, 0.2)',
        hovertemplate='%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='VN-INDEX TR',
            font=dict(size=14, color='#e2e8f0', family='Arial, sans-serif'),
            x=0
        ),
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#1a1f2e',
        font=dict(color='#a0aec0', size=11),
        showlegend=False,
        xaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=False,
            range=[0, 85]
        ),
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


# ==================== MODULE 4: SECTOR PERFORMANCE ====================
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
    
    df = pd.DataFrame(list(sectors_data.items()), columns=['Sector', 'Performance'])
    df = df.sort_values('Performance', ascending=True)
    
    colors = ['#48bb78' if x > 0 else '#f56565' for x in df['Performance']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['Sector'],
        x=df['Performance'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        text=[f"{val}%" for val in df['Performance']],
        textposition='outside',
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='SECTOR PERFORMANCE',
            font=dict(size=14, color='#e2e8f0', family='Arial, sans-serif'),
            x=0
        ),
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#1a1f2e',
        font=dict(color='#a0aec0', size=11),
        showlegend=False,
        xaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=True,
            zerolinecolor='#4a5568',
            title='Performance (%)',
            range=[-4, 6]
        ),
        yaxis=dict(
            showgrid=False
        ),
        height=350,
        margin=dict(l=120, r=40, t=50, b=40)
    )
    
    return fig


# ==================== MODULE 5: MARKET CAP BY SECTOR ====================
def generate_market_cap_treemap():
    """Module hiển thị vốn hóa thị trường theo ngành"""
    
    sectors_data = {
        'Technology': 35,
        'Financials': 30,
        'Industrials': 15,
        'Materials': 8,
        'Real Estate': 7,
        'Consumer': 5
    }
    
    df = pd.DataFrame(list(sectors_data.items()), columns=['Sector', 'Market Cap'])
    
    colors_map = {
        'Technology': '#48bb78',
        'Financials': '#f56565',
        'Industrials': '#fc8181',
        'Materials': '#f6ad55',
        'Real Estate': '#ed8936',
        'Consumer': '#dd6b20'
    }
    
    df['Color'] = df['Sector'].map(colors_map)
    
    fig = go.Figure(go.Treemap(
        labels=df['Sector'],
        parents=[''] * len(df),
        values=df['Market Cap'],
        marker=dict(
            colors=df['Color'],
            line=dict(color='#0e1117', width=2)
        ),
        text=[f"{sector}<br>{cap}%" for sector, cap in zip(df['Sector'], df['Market Cap'])],
        textposition='middle center',
        textfont=dict(size=13, color='#ffffff', family='Arial, sans-serif', weight='bold'),
        hovertemplate='<b>%{label}</b><br>Market Cap: %{value}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='MARKET CAP BY SECTOR',
            font=dict(size=14, color='#e2e8f0', family='Arial, sans-serif'),
            x=0
        ),
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#1a1f2e',
        font=dict(color='#a0aec0', size=11),
        height=350,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig


# ==================== MODULE 6: NET FOREIGN BUYING ====================
def generate_net_foreign_buying():
    """Module hiển thị dòng tiền nước ngoài"""
    
    dates = pd.date_range(start='2024-01-01', end='2024-10-31', freq='D')
    np.random.seed(43)
    
    # Tạo dữ liệu dòng tiền với xu hướng tăng
    net_foreign = 150 + np.cumsum(np.random.randn(len(dates)) * 30) + np.linspace(0, 100, len(dates))
    
    df = pd.DataFrame({
        'Date': dates,
        'Net Foreign': net_foreign
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Net Foreign'],
        mode='lines',
        name='Net Foreign Buying',
        line=dict(color='#4299e1', width=0),
        fill='tozeroy',
        fillcolor='rgba(66, 153, 225, 0.6)',
        hovertemplate='%{y:.0f}<extra></extra>'
    ))
    
    # Thêm đường zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#4a5568", line_width=1)
    
    fig.update_layout(
        title=dict(
            text='NET FOREIGN BUYING',
            font=dict(size=14, color='#e2e8f0', family='Arial, sans-serif'),
            x=0
        ),
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#1a1f2e',
        font=dict(color='#a0aec0', size=11),
        showlegend=False,
        xaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=True,
            zerolinecolor='#4a5568',
            range=[-500, 450]
        ),
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


# ==================== MODULE 7: INFLATION CORRELATION ====================
def generate_inflation_correlation():
    """Module hiển thị tương quan lạm phát"""
    
    dates = pd.date_range(start='2024-04-01', end='2024-10-31', freq='M')
    
    # Dữ liệu mẫu cho CPI và VN-INDEX correlation
    cpi = [0.25, 0.28, 0.32, 0.38, 0.42, 0.45, 0.50]
    vn_index_corr = [0.15, 0.18, 0.22, 0.28, 0.35, 0.42, 0.48]
    
    df = pd.DataFrame({
        'Date': dates,
        'CPI': cpi,
        'VN-INDEX': vn_index_corr
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['CPI'],
        mode='lines+markers',
        name='CPI',
        line=dict(color='#f6ad55', width=3),
        marker=dict(size=8, color='#f6ad55'),
        hovertemplate='CPI: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['VN-INDEX'],
        mode='lines+markers',
        name='VN-INDEX',
        line=dict(color='#4299e1', width=3),
        marker=dict(size=8, color='#4299e1'),
        hovertemplate='VN-INDEX: %{y:.2f}<extra></extra>'
    ))
    
    # Highlight points
    fig.add_trace(go.Scatter(
        x=[dates[4]],
        y=[cpi[4]],
        mode='markers',
        marker=dict(size=15, color='#f6ad55', symbol='circle-open', line=dict(width=2)),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=[dates[5]],
        y=[vn_index_corr[5]],
        mode='markers',
        marker=dict(size=15, color='#4299e1', symbol='circle-open', line=dict(width=2)),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(
            text='INFLATION AND MARKET CORRELATION',
            font=dict(size=14, color='#e2e8f0', family='Arial, sans-serif'),
            x=0
        ),
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#1a1f2e',
        font=dict(color='#a0aec0', size=11),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
            bgcolor='rgba(26, 31, 46, 0.8)'
        ),
        xaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=False,
            range=[0, 0.6]
        ),
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


# ==================== MODULE 8: CORRELATION MATRIX ====================
def generate_correlation_matrix():
    """Module hiển thị ma trận tương quan"""
    
    sectors = ['Technology', 'Financials', 'Consumer', 'Energy']
    
    # Ma trận tương quan mẫu
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
            [0, '#1a365d'],
            [0.5, '#2d3748'],
            [1, '#4299e1']
        ],
        text=correlation_data,
        texttemplate='%{text:.1f}',
        textfont=dict(size=13, color='#ffffff'),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>',
        colorbar=dict(
            thickness=15,
            len=0.7,
            bgcolor='#1a1f2e',
            tickfont=dict(color='#a0aec0'),
            title=dict(text='Corr.', side='right', font=dict(color='#a0aec0'))
        )
    ))
    
    fig.update_layout(
        title=dict(
            text='CORRELATION MATRIX',
            font=dict(size=14, color='#e2e8f0', family='Arial, sans-serif'),
            x=0
        ),
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#1a1f2e',
        font=dict(color='#a0aec0', size=11),
        xaxis=dict(
            side='bottom',
            showgrid=False
        ),
        yaxis=dict(
            showgrid=False,
            autorange='reversed'
        ),
        height=350,
        margin=dict(l=120, r=40, t=50, b=80)
    )
    
    return fig


# ==================== MODULE 9: INFLATION HEATMAP ====================
def generate_inflation_heatmap():
    """Module hiển thị heatmap lạm phát và tương quan"""
    
    categories = ['TC', 'FC', 'Mat']
    metrics = ['CPI', 'VN-INDEX', '0.1']
    
    # Dữ liệu heatmap
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
            [0, '#2d5016'],
            [0.5, '#c05621'],
            [1, '#c05621']
        ],
        text=heatmap_data,
        texttemplate='%{text:.1f}',
        textfont=dict(size=14, color='#ffffff', weight='bold'),
        hovertemplate='%{y} - %{x}<br>Value: %{z:.1f}<extra></extra>',
        showscale=False
    ))
    
    fig.update_layout(
        title=dict(
            text='INFLATION AND MARKET CORR.',
            font=dict(size=14, color='#e2e8f0', family='Arial, sans-serif'),
            x=0
        ),
        paper_bgcolor='#1a1f2e',
        plot_bgcolor='#1a1f2e',
        font=dict(color='#a0aec0', size=11),
        xaxis=dict(
            side='top',
            showgrid=False
        ),
        yaxis=dict(
            showgrid=False
        ),
        height=250,
        margin=dict(l=60, r=40, t=70, b=40)
    )
    
    return fig


# ==================== MAIN DASHBOARD ====================
def main():
    """Hàm chính để render dashboard"""
    
    # Header
    st.markdown('<div class="dashboard-header">MARKET & SECTOR ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Updated Monthly</div>', unsafe_allow_html=True)
    
    # Thêm tab để chuyển đổi giữa các dashboard
    tab1, tab2 = st.tabs(["📊 Executive Dashboard", "💼 Finance Dashboard"])
    
    with tab1:
        # Row 1: Market Indices KPI
        generate_market_indices_kpi()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2: Charts - 3 columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.plotly_chart(generate_index_comparison_chart(), use_container_width=True)
            st.plotly_chart(generate_market_cap_treemap(), use_container_width=True)
        
        with col2:
            st.plotly_chart(generate_vn_index_trend(), use_container_width=True)
            st.plotly_chart(generate_net_foreign_buying(), use_container_width=True)
        
        with col3:
            st.plotly_chart(generate_sector_performance(), use_container_width=True)
            st.plotly_chart(generate_inflation_correlation(), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 3: Bottom charts
        col4, col5 = st.columns(2)
        
        with col4:
            st.plotly_chart(generate_correlation_matrix(), use_container_width=True)
        
        with col5:
            st.plotly_chart(generate_inflation_heatmap(), use_container_width=True)
    
    with tab2:
        st.markdown("### 💼 Portfolio Dashboard")
        st.info("Tích hợp dashboard quản lý danh mục đầu tư hiện tại tại đây...")
        
        # Có thể import và gọi các hàm từ dashboard.py gốc
        # from scripts.dashboard import main_manual_selection, main_auto_selection
        # main_manual_selection()


if __name__ == "__main__":
    main()
