"""
Bảng điều hành - Phân tích thị trường & ngành
Hệ thống hiển thị hiện đại với giao diện trực quan, chia rõ từng mô-đun.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

from data_process.data_loader import (
    get_market_indices_metrics,
    get_indices_history,
    get_index_history,
    get_realtime_index_board,
    fetch_stock_data2
)
from ui.visualization import plot_interactive_stock_chart
from utils.config import ANALYSIS_START_DATE, ANALYSIS_END_DATE

warnings.filterwarnings('ignore')

PAPER_BG = '#ffffff'
PLOT_BG = '#ffffff'
FONT_COLOR = '#2d3748'
GRID_COLOR = '#e2e8f0'
ZERO_LINE_COLOR = '#cbd5f5'
POSITIVE_COLOR = '#2f855a'
NEGATIVE_COLOR = '#c53030'
REFERENCE_COLOR = '#d69e2e'
POSITIVE_COLOR_DARK = '#1f6b46'
POSITIVE_COLOR_LIGHT = 'rgba(47, 133, 90, 0.45)'
NEGATIVE_COLOR_DARK = '#9b2c2c'
NEGATIVE_COLOR_LIGHT = 'rgba(197, 48, 48, 0.45)'
PERIOD_COLOR_STRONG = '#2d3748'
PERIOD_COLOR_LIGHT = '#a0aec0'
BASE_FONT_FAMILY = 'Inter, "Be VietNam Pro", "Segoe UI", sans-serif'
BOLD_FONT_FAMILY = 'Inter SemiBold, "Be VietNam Pro SemiBold", "Segoe UI Semibold", "Segoe UI", sans-serif'
LEGEND_GRAY_DARK = '#4a5568'
LEGEND_GRAY_LIGHT = '#cbd5d5'
TITLE_FONT = dict(size=15, color=FONT_COLOR, family=BOLD_FONT_FAMILY)
TITLE_PAD = dict(b=12)
REALTIME_INDEX_SYMBOLS = ["VNINDEX", "VN30", "HNXIndex", "HNX30", "UpcomIndex"]
REALTIME_LABELS = {
    "VNINDEX": "VN-Index",
    "VN30": "VN30",
    "HNXINDEX": "HNX-Index",
    "HNX30": "HNX30",
    "UPCOMINDEX": "UPCoM",
    "UPCOM": "UPCoM",
}
REALTIME_SYMBOL_KEYS = sorted({symbol.upper() for symbol in REALTIME_INDEX_SYMBOLS} | {symbol.upper() for symbol in REALTIME_LABELS.keys()})

SNAPSHOT_COLUMNS = [
    'ticker'
]

COMPANY_INFO_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'company_info.csv')


@st.cache_data(ttl=1800, show_spinner=False)
def load_overview_data():
    """Fetch lightweight data powering the headline KPI cards and charts."""

    analysis_start = pd.to_datetime(ANALYSIS_START_DATE).strftime("%Y-%m-%d")
    analysis_end = pd.to_datetime(ANALYSIS_END_DATE).strftime("%Y-%m-%d")
    months_span = max(1, int((pd.to_datetime(analysis_end) - pd.to_datetime(analysis_start)).days / 30))

    indices_metrics = get_market_indices_metrics()
    index_history = get_indices_history(start_date=analysis_start, end_date=analysis_end, months=months_span)
    
    if not indices_metrics:
        st.warning("⚠️ Không thể tải dữ liệu chỉ số thị trường.")
    
    if index_history.empty:
        st.warning("⚠️ Không thể tải lịch sử chỉ số.")
    
    return {
        'indices_metrics': indices_metrics,
        'index_history': index_history,
    }





@st.cache_data(ttl=1800, show_spinner=False)
def get_top_movers(top_n: int = 10):
    """Lấy top cổ phiếu tăng/giảm mạnh nhất từ VN30."""
    from data_process.fetchers import fetch_stock_data2
    import datetime
    
    # Danh sách VN30
    vn30_symbols = ['ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
                    'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
                    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE']
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=7)
    
    try:
        data, _ = fetch_stock_data2(vn30_symbols, start_date.strftime("%Y-%m-%d"), 
                                     end_date.strftime("%Y-%m-%d"), verbose=False)
        if data.empty:
            return pd.DataFrame()
        
        # Tính % thay đổi
        pct_change = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100).sort_values(ascending=False)
        
        gainers = pct_change.head(top_n)
        losers = pct_change.tail(top_n)
        
        return {
            'gainers': gainers,
            'losers': losers
        }
    except Exception as e:
        print(f"Lỗi khi lấy top movers: {e}")
        return pd.DataFrame()

# ==================== TÙY CHỈNH CSS ====================
DASHBOARD_STYLE = """
<style>
    /* Nền trang và phông chữ tổng thể */
    html, body, [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: #f5f7fb !important;
        color: #1a202c;
        font-family: "Inter", "Be VietNam Pro", "Segoe UI", sans-serif;
    }

    /* Tiêu đề chính và mô tả */
    .dashboard-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .dashboard-subtitle {
        font-size: 0.95rem;
        color: #4a5568;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }

    /* Thẻ KPI */
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #edf2f7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px rgba(15, 23, 42, 0.12);
    }

    .kpi-title {
        font-size: 0.85rem;
        color: #718096;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 0.3rem;
    }

    .kpi-change {
        font-size: 0.9rem;
        font-weight: 600;
    }

    .kpi-change.positive {
        color: #2f855a;
    }

    .kpi-change.negative {
        color: #c53030;
    }

    .kpi-change.neutral {
        color: #d69e2e;
    }

    /* Hộp chứa biểu đồ */
    .chart-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
    }

    .chart-title {
        font-size: 1rem;
        color: #2d3748;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Thanh hiệu suất ngành */
    .sector-bar {
        background: #edf2f7;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }

    /* Thanh cuộn */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #e2e8f0;
    }

    ::-webkit-scrollbar-thumb {
        background: #cbd5f5;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #a0aec0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #ffffff;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        color: #4a5568;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.25rem 0.5rem;
    }

    .stTabs [aria-selected="true"] {
        color: #1a202c;
        border-bottom: 3px solid #3182ce;
    }

    .stTabs [data-baseweb="tab-content"] {
        background-color: #ffffff;
        border-radius: 0 0 12px 12px;
        border: 1px solid #e2e8f0;
        margin-top: -0.5rem;
        padding-top: 1.5rem;
    }

    .chart-gap {
        height: 1rem;
        width: 100%;
    }

    /* Light Stock List Styles */
    .portfolio-container {
        background: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .portfolio-header {
        display: flex;
        align-items: center;
        padding: 1rem 1.2rem;
        background: #ffffff;
        border-bottom: 1px solid #e2e8f0;
    }
    
    
    
    .portfolio-title {
        color: #2d3748;
        font-size: 1rem;
        font-weight: 600;
        flex-grow: 1;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .portfolio-time {
        color: #718096;
        font-size: 0.8rem;
    }
    
    .stock-list-header {
        display: grid;
        grid-template-columns: 1fr 100px 100px;
        padding: 0.6rem 1.2rem;
        background: #f7fafc;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stock-list-header span {
        color: #718096;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    .stock-list-header span:nth-child(2),
    .stock-list-header span:nth-child(3) {
        text-align: right;
    }
    
    .stock-list {
        background: white;
    }
    
    .stock-item {
        display: grid;
        grid-template-columns: 1fr 100px 100px;
        align-items: center;
        padding: 0.9rem 1.2rem;
        border-bottom: 1px solid #edf2f7;
        transition: background 0.2s;
    }
    
    .stock-item:hover {
        background: #f7fafc;
    }
    
    .stock-item:last-child {
        border-bottom: none;
    }
    
    .stock-symbol {
        font-weight: 700;
        font-size: 1rem;
        color: #2d3748;
    }
    
    .stock-price {
        text-align: right;
    }
    
    .stock-price-value {
        font-weight: 600;
        font-size: 0.95rem;
        color: #2d3748;
    }
    
    .stock-price-label {
        font-size: 0.7rem;
        color: #a0aec0;
        margin-top: 2px;
    }
    
    .stock-change {
        text-align: right;
    }
    
    .stock-change-positive {
        font-weight: 700;
        font-size: 0.95rem;
        color: #22c55e;
    }
    
    .stock-change-negative {
        font-weight: 700;
        font-size: 0.95rem;
        color: #ef4444;
    }
    
    .stock-change-label {
        font-size: 0.7rem;
        color: #a0aec0;
        margin-top: 2px;
    }
    
    /* Top Movers Compact Styles */
    .movers-container {
        background: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .movers-header {
        display: flex;
        align-items: center;
        padding: 0.8rem 1rem;
        background: #ffffff;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .movers-header-icon {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.6rem;
    }
    
    .movers-header-icon svg {
        width: 16px;
        height: 16px;
        fill: white;
    }
    
    .movers-title {
        color: #2d3748;
        font-size: 0.9rem;
        font-weight: 600;
        flex-grow: 1;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .movers-list {
        background: white;
    }
    
    .mover-item {
        display: flex;
        align-items: center;
        padding: 0.6rem 1rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .mover-item:last-child {
        border-bottom: none;
    }
    
    .mover-rank {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #f0f0f0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        color: #718096;
        margin-right: 0.8rem;
    }
    
    .mover-symbol {
        font-weight: 700;
        font-size: 0.9rem;
        color: #2d3748;
        width: 60px;
    }
    
    .mover-bar-container {
        flex-grow: 1;
        height: 20px;
        background: #f0f0f0;
        border-radius: 10px;
        margin: 0 0.8rem;
        overflow: hidden;
    }
    
    .mover-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .mover-bar.positive {
        background: linear-gradient(90deg, #22c55e, #4ade80);
    }
    
    .mover-bar.negative {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }
    
    .mover-value {
        font-weight: 700;
        font-size: 0.85rem;
        min-width: 60px;
        text-align: right;
    }
    
    .mover-value.positive {
        color: #22c55e;
    }
    
    .mover-value.negative {
        color: #ef4444;
    }
        color: #22c55e;
    }
    
    .stock-change-negative {
        font-weight: 700;
        font-size: 0.95rem;
        color: #ef4444;
    }
    
    .stock-change-label {
        font-size: 0.7rem;
        color: #a0aec0;
        margin-top: 2px;
    }
</style>
"""

CHART_GAP_DIV = "<div class='chart-gap'></div>"


# ==================== MÔ-ĐUN 1: KPI CHỈ SỐ THỊ TRƯỜNG ====================
def generate_market_indices_kpi(metrics):
    """Hiển thị các chỉ số chính dạng thẻ KPI dựa trên dữ liệu thực."""

    if not metrics:
        st.info("Không có dữ liệu chỉ số để hiển thị.")
        return

    cols = st.columns(len(metrics))

    for col, metric in zip(cols, metrics):
        value = metric.get('value')
        change_pct = metric.get('pct_change')
        note = metric.get('note', '')
        timestamp = metric.get('timestamp')

        value_display = f"{value:,.2f}" if value is not None else "—"
        if change_pct is None:
            trend_class = ""
            trend_value = "Chưa có dữ liệu"
        else:
            if change_pct > 0:
                trend_class = "positive"
            elif change_pct < 0:
                trend_class = "negative"
            else:
                trend_class = "neutral"
            trend_value = f"{change_pct:+.2f}%"

        time_suffix = f" · {timestamp.strftime('%d/%m %H:%M')}" if timestamp is not None else ""

        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">{metric.get('label')}</div>
                <div class="kpi-value">{value_display}</div>
                <div class="kpi-change {trend_class}">{trend_value} · {note}{time_suffix}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(ttl=300, show_spinner=False)
def _build_realtime_metrics():
    board = get_realtime_index_board(REALTIME_INDEX_SYMBOLS)
    if board is None or board.empty:
        return []

    history_cache = {}

    def _safe_float(value):
        try:
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    def _get_sorted_history(symbol_key):
        if symbol_key not in history_cache:
            history = get_index_history(symbol_key, months=1)
            history_cache[symbol_key] = history.sort_values('time') if not history.empty else pd.DataFrame()
        return history_cache[symbol_key]

    metrics = []
    for _, row in board.iterrows():
        symbol_key = str(row['symbol']).upper()
        price = _safe_float(row.get('gia_khop'))
        reference = _safe_float(row.get('gia_tham_chieu'))
        change = _safe_float(row.get('thay_doi'))
        pct_change = _safe_float(row.get('ty_le_thay_doi'))
        note_ts = row.get('last_updated')
        note_text = None

        if reference in (None, 0) and price is not None and change is not None:
            reference = price - change

        if price in (None, 0) and reference not in (None, 0):
            price = reference
            change = 0.0
            pct_change = 0.0
            note_text = 'Chưa có khớp · Hiển thị tham chiếu'
        elif price in (None, 0):
            history = _get_sorted_history(symbol_key)
            if history.empty:
                continue
            last_row = history.iloc[-1]
            prev_row = history.iloc[-2] if len(history) > 1 else last_row
            price = float(last_row['close'])
            reference = float(prev_row['close']) if pd.notna(prev_row['close']) else price
            change = price - reference
            pct_change = (change / reference * 100) if reference not in (0, None) else 0.0
            note_text = 'Dữ liệu cuối phiên'
            note_ts = pd.to_datetime(last_row['time']).to_pydatetime()
        else:
            base_reference = reference if reference not in (None, 0) else None
            if base_reference is None and change is not None:
                base_reference = price - change
            if change is None and base_reference is not None:
                change = price - base_reference
            if pct_change is None and base_reference not in (None, 0):
                pct_change = (change / base_reference * 100) if change is not None else 0.0

        if change is None:
            change = 0.0
        if pct_change is None:
            pct_change = 0.0
        if note_text is None:
            note_text = f"Thay đổi {change:+.2f} điểm"

        metrics.append({
            'symbol': symbol_key,
            'label': REALTIME_LABELS.get(symbol_key, symbol_key),
            'value': price,
            'change': change,
            'pct_change': pct_change,
            'note': note_text,
            'timestamp': note_ts
        })

    available_symbols = {metric['symbol'] for metric in metrics}
    for symbol_key in REALTIME_SYMBOL_KEYS:
        if symbol_key in available_symbols:
            continue
        history = _get_sorted_history(symbol_key)
        if history.empty:
            continue
        last_row = history.iloc[-1]
        prev_row = history.iloc[-2] if len(history) > 1 else last_row
        last_close = float(last_row['close']) if pd.notna(last_row['close']) else None
        prev_close = float(prev_row['close']) if pd.notna(prev_row['close']) else None
        if last_close is None:
            continue
        change = last_close - (prev_close if prev_close is not None else last_close)
        pct_change = (change / prev_close * 100) if prev_close not in (0, None) else 0.0
        metrics.append({
            'symbol': symbol_key,
            'label': REALTIME_LABELS.get(symbol_key, symbol_key),
            'value': last_close,
            'change': change,
            'pct_change': pct_change,
            'note': 'Dữ liệu cuối phiên',
            'timestamp': pd.to_datetime(last_row['time']).to_pydatetime()
        })
    return metrics

def render_realtime_market_overview():
    metrics = _build_realtime_metrics()
    if not metrics:
        st.info("Không thể tải dữ liệu realtime cho các chỉ số.")
        return

    generate_market_indices_kpi(metrics)

    latest_ts = max((metric.get('timestamp') for metric in metrics if metric.get('timestamp')), default=None)
    if latest_ts:
        st.caption(f"Cập nhật: {latest_ts.strftime('%d/%m/%Y %H:%M:%S')}")


# ==================== MÔ-ĐUN 2: SO SÁNH CHỈ SỐ CHÍNH ====================
@st.cache_data(ttl=1800, show_spinner=False)
def generate_index_comparison_chart(index_history: pd.DataFrame):
    """Biểu đồ so sánh VN-Index, HNX và UPCoM dựa trên dữ liệu lịch sử."""

    if index_history is None or index_history.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        fig.add_annotation(text='Không có dữ liệu chỉ số', xref='paper', yref='paper', x=0.5, y=0.5)
        return fig

    pivot_df = index_history.pivot(index='time', columns='symbol', values='close').dropna(how='all')
    pivot_df = pivot_df.fillna(method='ffill')
    pivot_df = pivot_df.dropna(how='all')

    def normalize_series(series: pd.Series) -> pd.Series:
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            return series
        base_value = series.loc[first_valid_idx]
        if base_value in (0, None):
            return series
        return (series / base_value - 1) * 100

    pct_change_df = pivot_df.apply(normalize_series)

    fig = go.Figure()

    palette = {
        'VN-Index': '#1D4ED8',
        'HNX-Index': '#F97316',
        'UPCoM': '#7C3AED'
    }

    for column in pct_change_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pct_change_df.index,
                y=pct_change_df[column],
                mode='lines',
                name=column,
                line=dict(color=palette.get(column, '#2d3748'), width=2.6),
                hovertemplate='%{y:+.2f}%<extra></extra>'
            )
        )

    fig.update_layout(
        title=dict(
            text='SO SÁNH CÁC CHỈ SỐ CHÍNH (TỶ LỆ % SO ĐẦU KỲ)',
            font=TITLE_FONT,
            x=0,
            pad=TITLE_PAD
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, size=11),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color=FONT_COLOR),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#e2e8f0',
            borderwidth=1,
            itemclick='toggleothers',
            itemsizing='constant'
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            zeroline=True,
            zerolinecolor=ZERO_LINE_COLOR,
            title='Thay đổi so với đầu kỳ (%)'
        ),
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )


    if len(pct_change_df) > 30:
        anchor_idx = pct_change_df.index[int(len(pct_change_df) * 0.7)]
        anchor_symbol = 'HNX-Index' if 'HNX-Index' in pct_change_df.columns else pct_change_df.columns[0]
        anchor_value = pct_change_df[anchor_symbol].loc[anchor_idx]
        fig.add_annotation(
            x=anchor_idx,
            y=anchor_value,
            text="Xu hướng ngắn hạn",
            showarrow=True,
            arrowhead=2,
            arrowcolor=REFERENCE_COLOR,
            arrowwidth=1.2,
            ax=0,
            ay=-70,
            font=dict(size=10, color='#1a202c'),
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor=REFERENCE_COLOR,
            borderwidth=1
        )

    return fig


# ==================== MÔ-ĐUN 3: TOP MOVERS====================
def render_top_gainers_list(gainers_series):
    if gainers_series is None or gainers_series.empty:
        st.info("Không có dữ liệu top tăng giá.")
        return
    
    # Sắp xếp từ cao xuống thấp
    sorted_series = gainers_series.sort_values(ascending=False)
    max_val = sorted_series.max() if sorted_series.max() > 0 else 1
    
    stock_items = []
    for rank, (symbol, pct) in enumerate(sorted_series.items(), 1):
        bar_width = (pct / max_val * 100) if max_val > 0 else 0
        
        stock_items.append(f'''<div class="mover-item">
            <div class="mover-rank">{rank}</div>
            <div class="mover-symbol">{symbol}</div>
            <div class="mover-bar-container">
                <div class="mover-bar positive" style="width: {bar_width:.1f}%;"></div>
            </div>
            <div class="mover-value positive">+{pct:.2f}%</div>
        </div>''')
    
    full_html = f'''
    <div class="movers-container">
        <div class="movers-header">
            <div class="movers-header-icon" style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);">
                <svg viewBox="0 0 24 24"><path d="M7 14l5-5 5 5z" fill="white"/></svg>
            </div>
            <span class="movers-title">Top Tăng Giá Tuần (VN30)</span>
        </div>
        <div class="movers-list">
            {''.join(stock_items)}
        </div>
    </div>
    '''
    st.markdown(full_html, unsafe_allow_html=True)


def render_top_losers_list(losers_series):
    """Hiển thị top cổ phiếu giảm dạng list với progress bar."""
    if losers_series is None or losers_series.empty:
        st.info("Không có dữ liệu top giảm giá.")
        return
    
    # Sắp xếp từ âm nhiều nhất đến ít âm
    sorted_series = losers_series.sort_values(ascending=True)
    abs_values = sorted_series.abs()
    max_val = abs_values.max() if abs_values.max() > 0 else 1
    
    stock_items = []
    for rank, (symbol, pct) in enumerate(sorted_series.items(), 1):
        abs_pct = abs(pct)
        # Bar width tỷ lệ trực tiếp với giá trị (max = 100%)
        bar_width = (abs_pct / max_val * 100) if max_val > 0 else 0
            
        stock_items.append(f'''<div class="mover-item">
            <div class="mover-rank">{rank}</div>
            <div class="mover-symbol">{symbol}</div>
            <div class="mover-bar-container">
                <div class="mover-bar negative" style="width: {bar_width:.1f}%;"></div>
            </div>
            <div class="mover-value negative">{pct:.2f}%</div>
        </div>''')
    
    full_html = f'''
    <div class="movers-container">
        <div class="movers-header">
            <div class="movers-header-icon" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                <svg viewBox="0 0 24 24"><path d="M7 10l5 5 5-5z" fill="white"/></svg>
            </div>
            <span class="movers-title">Top Giảm Giá Tuần</span>
        </div>
        <div class="movers-list">
            {''.join(stock_items)}
        </div>
    </div>
    '''
    st.markdown(full_html, unsafe_allow_html=True)


# ==================== MÔ-ĐUN 4: DANH MỤC ĐẦU TƯ HIỆN TẠI ====================
def render_current_portfolio():
    """Hiển thị tổng quan danh mục đầu tư."""
    from utils.session_manager import update_current_tab
    
    # Lấy danh sách cổ phiếu đã chọn từ session state
    selected_stocks = st.session_state.get('selected_stocks', [])
    
    if not selected_stocks:
        # Nếu chưa có danh mục, hiển thị thông báo và nút điều hướng
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 12px; padding: 2rem; margin-bottom: 1.5rem; 
                    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);">
            <h3 style="color: white; margin: 0 0 1rem 0; font-size: 1.5rem;">
                 Chưa có Danh Mục Đầu Tư
            </h3>
            <p style="color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 1rem; line-height: 1.6;">
                Bạn chưa chọn cổ phiếu nào cho danh mục đầu tư. 
                Hãy bắt đầu bằng cách chọn các mã cổ phiếu để phân tích và tối ưu hóa!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(" Đi đến Chọn Cổ Phiếu", use_container_width=True, type="primary"):
                update_current_tab("Tự chọn mã cổ phiếu")
                st.rerun()
        return
    
    
    # Lấy dữ liệu giá
    left_col, right_col = st.columns([1, 1])
    with left_col:
        try:
            with st.spinner("Đang tải dữ liệu..."):
                stock_data, skipped_ticker = fetch_stock_data2(
                    selected_stocks, 
                    start_date=ANALYSIS_START_DATE, 
                    end_date=ANALYSIS_END_DATE, 
                    verbose=False
                )
                
                if not stock_data.empty:
                    # Tính % thay đổi 7 ngày
                    if len(stock_data) > 1:
                        pct_change_7d = ((stock_data.iloc[-1] - stock_data.iloc[0]) / stock_data.iloc[0] * 100)
                    else:
                        pct_change_7d = pd.Series(0, index=stock_data.columns)
                    
                    # Lấy giá mới nhất
                    latest_prices = stock_data.iloc[-1]
                    
                   
                    stock_items = []
                    for symbol in selected_stocks:
                        price = latest_prices.get(symbol, 0)
                        pct = pct_change_7d.get(symbol, 0)
                        
                        price_display = f"{price*1000:,.0f}" if price > 0 else "—"
                        
                        if pct > 0:
                            change_class = "stock-change-positive"
                            change_text = f"{pct:.2f}%"
                        elif pct < 0:
                            change_class = "stock-change-negative"
                            change_text = f"{pct:.2f}%"
                        else:
                            change_class = "stock-change-positive"
                            change_text = "0.00%"
                        
                        stock_items.append(f'''<div class="stock-item">
                            <div class="stock-symbol">{symbol}</div>
                            <div class="stock-price">
                                <div class="stock-price-value">{price_display}</div>
                            </div>
                            <div class="stock-change">
                                <span class="{change_class}">{change_text}</span>
                            </div>
                        </div>''')
                    
                    # Full HTML với container và header
                    full_html = f'''
                    <div class="portfolio-container">
                        <div class="portfolio-header">
                            <div class="portfolio-header-icon">
                                <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
                            </div>
                            <span class="portfolio-title">Danh Mục Của Tôi</span>
                        </div>
                        <div class="stock-list-header">
                            <span>Mã CP</span>
                            <span>Giá (VNĐ)</span>
                            <span>% Thay đổi</span>
                        </div>
                        <div class="stock-list">
                            {''.join(stock_items)}
                        </div>
                    </div>
                    '''
                    st.markdown(full_html, unsafe_allow_html=True)
                else:
                    st.info("Không có dữ liệu để hiển thị.")
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu: {e}")
    with right_col:
        plot_interactive_stock_chart(stock_data, selected_stocks)


# ==================== KHU VỰC HIỂN THỊ CHÍNH ====================
def render_bang_dieu_hanh():
    """Hiển thị bảng điều hành chính cho tab Tổng quan Thị trường & Ngành."""
    st.markdown(DASHBOARD_STYLE, unsafe_allow_html=True)

    st.markdown('<div class="dashboard-header">PHÂN TÍCH THỊ TRƯỜNG & NGÀNH</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Dữ liệu tổng hợp & cập nhật theo thời gian thực</div>', unsafe_allow_html=True)

    with st.spinner("Đang tải dữ liệu tổng quan..."):
        overview_data = load_overview_data()

    with st.spinner("Đang tải dữ liệu thị trường..."):
        render_realtime_market_overview()

    st.markdown(CHART_GAP_DIV, unsafe_allow_html=True)

    # Chỉ hiển thị biểu đồ lịch sử chỉ số và top movers
    st.plotly_chart(
        generate_index_comparison_chart(overview_data.get('index_history')), width='stretch'
    )
    
    st.markdown(CHART_GAP_DIV, unsafe_allow_html=True)
    
    # Hiển thị Top Movers dạng list
    left_col, right_col = st.columns(2)
    
    with st.spinner("Đang tải dữ liệu VN30..."):
        top_movers = get_top_movers(top_n=10)
    
    if top_movers and isinstance(top_movers, dict):
        with left_col:
            if 'gainers' in top_movers and not top_movers['gainers'].empty:
                render_top_gainers_list(top_movers['gainers'])
            else:
                st.info("Không thể tải dữ liệu top tăng giá.")
        
        with right_col:
            if 'losers' in top_movers and not top_movers['losers'].empty:
                render_top_losers_list(top_movers['losers'])
            else:
                st.info("Không thể tải dữ liệu top giảm giá.")
    else:
        left_col.info("Không thể tải dữ liệu VN30.")
        right_col.info("Không thể tải dữ liệu VN30.")
    
    st.markdown(CHART_GAP_DIV, unsafe_allow_html=True)
    
    # Hiển thị danh mục đầu tư hiện tại ở cuối
    render_current_portfolio()


def main():
    """Giữ hàm main để có thể chạy file độc lập."""
    render_bang_dieu_hanh()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Bảng điều hành tài chính",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    main()
