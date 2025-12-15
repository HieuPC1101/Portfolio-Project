"""
Bảng điều hành - Phân tích thị trường & ngành
Hệ thống hiển thị hiện đại với giao diện trực quan, chia rõ từng mô-đun.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings

from data_process.data_loader import (
    get_market_indices_metrics,
    get_sector_snapshot,
    summarize_sector_performance,
    summarize_market_cap_distribution,
    get_foreign_flow_leaderboard,
    get_liquidity_leaders,
    get_indices_history,
    get_index_history,
    get_realtime_index_board,
    fetch_stock_data2,
    get_return_correlation_matrix,
    fetch_data_from_csv
)
from plotly.subplots import make_subplots
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

SNAPSHOT_COLUMNS = [
    'ticker',
    'industry',
    'market_cap',
    'price_growth_1w',
    'price_growth_1m',
    'avg_trading_value_20d',
    'foreign_buysell_20s'
]

COMPANY_INFO_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'company_info.csv')


def _get_scale_and_suffix(values, base_unit='VND'):
    """Return a compact scaling factor and Vietnamese suffix for large numbers."""
    series = pd.to_numeric(pd.Series(values), errors='coerce').dropna().abs()
    if series.empty:
        return 1, base_unit

    max_val = series.max()
    if max_val >= 1e12:
        return 1e12, f"nghìn tỷ {base_unit}"
    if max_val >= 1e9:
        return 1e9, f"tỷ {base_unit}"
    if max_val >= 1e6:
        return 1e6, f"triệu {base_unit}"
    if max_val >= 1e3:
        return 1e3, f"nghìn {base_unit}"
    return 1, base_unit


@st.cache_data(ttl=3600, show_spinner=False)
def load_company_industries():
    """Load level-1 industry classification from local CSV once."""
    company_df = fetch_data_from_csv(COMPANY_INFO_PATH)
    if company_df.empty or 'symbol' not in company_df.columns:
        return pd.DataFrame(columns=['symbol', 'industry_level_1'])

    mapping = company_df.copy()
    mapping['symbol'] = mapping['symbol'].astype(str).str.upper()

    if 'icb_name' in mapping.columns:
        mapping = mapping.rename(columns={'icb_name': 'industry_level_1'})
    elif 'industry' in mapping.columns:
        mapping = mapping.rename(columns={'industry': 'industry_level_1'})
    else:
        mapping['industry_level_1'] = 'Ngành khác'

    mapping['industry_level_1'] = mapping['industry_level_1'].fillna('Ngành khác')

    return mapping[['symbol', 'industry_level_1']]


def get_industry_order():
    """Return the canonical ordering of industries defined in company_info.csv."""
    companies = load_company_industries()
    if companies.empty or 'industry_level_1' not in companies.columns:
        return []
    order_series = companies['industry_level_1'].dropna().astype(str).str.strip()
    return order_series.drop_duplicates().tolist()

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
</style>
"""

CHART_GAP_DIV = "<div class='chart-gap'></div>"


# ==================== MÔ-ĐUN 4: DANH MỤC ĐẦU TƯ HIỆN TẠI ====================
def render_current_portfolio():
    """Hiển thị tổng quan danh mục đầu tư."""
    # Lấy danh sách cổ phiếu đã chọn từ session state
    selected_stocks = st.session_state.get('selected_stocks', [])
    
    if not selected_stocks:
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


@st.cache_data(ttl=1800, show_spinner=False)
def load_overview_data():
    """Fetch lightweight data powering the headline KPI cards and charts."""

    analysis_start = pd.to_datetime(ANALYSIS_START_DATE).strftime("%Y-%m-%d")
    analysis_end = pd.to_datetime(ANALYSIS_END_DATE).strftime("%Y-%m-%d")
    months_span = max(1, int((pd.to_datetime(analysis_end) - pd.to_datetime(analysis_start)).days / 30))

    return {
        'indices_metrics': get_market_indices_metrics(),
        'index_history': get_indices_history(start_date=analysis_start, end_date=analysis_end, months=months_span),
    }


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
                <div class="kpi-change {trend_class}">{trend_value}{note}{time_suffix}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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

    fig.update_yaxes(tickformat='+.0f')

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


@st.cache_data(ttl=1800, show_spinner=False)
def load_sector_snapshot_cached():
    """Cache-reuse the sector snapshot with only essential columns."""
    snapshot = get_sector_snapshot(exchange='HOSE', size=600)
    
    if snapshot.empty:
        return snapshot

    # Filter columns if they exist
    cols_to_keep = [c for c in SNAPSHOT_COLUMNS if c in snapshot.columns]
    if cols_to_keep:
        snapshot = snapshot[cols_to_keep]

    companies = load_company_industries()
    if not companies.empty and 'ticker' in snapshot.columns:
        working = snapshot.copy()
        working['ticker'] = working['ticker'].astype(str).str.upper()
        merged = working.merge(companies, left_on='ticker', right_on='symbol', how='left')
        merged['industry_level_1'] = merged['industry_level_1'].fillna(merged.get('industry', 'Ngành khác'))
        merged['industry'] = merged['industry_level_1']
        merged = merged.drop(columns=['symbol'], errors='ignore')
        return merged

    return snapshot


@st.cache_data(ttl=1800, show_spinner=False)
def load_detail_data():
    """Load heavier, sector-dependent datasets for secondary visuals."""

    sector_snapshot = load_sector_snapshot_cached()
    return {
        'sector_snapshot': sector_snapshot,
        'sector_perf': summarize_sector_performance(sector_snapshot, top_n=None),
        'market_cap': summarize_market_cap_distribution(sector_snapshot, top_n=8),
        'foreign_flow': get_foreign_flow_leaderboard(sector_snapshot, top_n=6),
        'liquidity': get_liquidity_leaders(sector_snapshot, top_n=40),
        'correlation': get_return_correlation_matrix(),
    }


# ==================== MÔ-ĐUN 4: HIỆU SUẤT NGÀNH ====================
def generate_sector_performance(sector_perf: pd.DataFrame):
    """Biểu đồ hiệu suất ngành dựa trên dữ liệu vnstock."""

    if sector_perf is None or sector_perf.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        fig.add_annotation(text='Chưa có dữ liệu ngành', xref='paper', yref='paper', x=0.5, y=0.5)
        return fig

    ordered = sector_perf.copy()
    ordered['industry'] = ordered.get('industry', pd.Series(index=ordered.index, dtype=str))
    ordered['industry'] = ordered['industry'].astype(str).str.strip()

    industry_order = get_industry_order()
    trimmed_order = []
    if industry_order:
        trimmed_order = [name.strip() for name in industry_order if isinstance(name, str) and name.strip()]
        ordered = ordered[ordered['industry'].isin(trimmed_order)].copy()
        if ordered.empty:
            # Fallback if strict mapping fails
            ordered = sector_perf.copy()
        else:
            ordered = ordered.set_index('industry')
            ordered = ordered.reindex(trimmed_order)
            ordered = ordered.dropna(subset=['avg_growth_1m', 'avg_growth_1w'], how='all').reset_index()
            ordered = ordered.rename(columns={'index': 'industry'})

    ordered['industry'] = ordered['industry'].astype(str)

    avg_growth_1m_series = pd.to_numeric(ordered.get('avg_growth_1m'), errors='coerce')
    avg_growth_1w_series = pd.to_numeric(ordered.get('avg_growth_1w'), errors='coerce')
    ordered['avg_growth_1m'] = avg_growth_1m_series
    ordered['avg_growth_1w'] = avg_growth_1w_series

    market_avg_1m = avg_growth_1m_series.mean(skipna=True)
    market_avg_1w = avg_growth_1w_series.mean(skipna=True)
    market_avg_1m = float(market_avg_1m) if pd.notna(market_avg_1m) else 0.0
    market_avg_1w = float(market_avg_1w) if pd.notna(market_avg_1w) else 0.0

    delta_growth_1m_source = ordered['delta_growth_1m'] if 'delta_growth_1m' in ordered.columns else (avg_growth_1m_series - market_avg_1m)
    delta_growth_1w_source = ordered['delta_growth_1w'] if 'delta_growth_1w' in ordered.columns else (avg_growth_1w_series - market_avg_1w)
    ordered['delta_growth_1m'] = pd.to_numeric(delta_growth_1m_source, errors='coerce')
    ordered['delta_growth_1w'] = pd.to_numeric(delta_growth_1w_source, errors='coerce')
    
    # Fill NA delta
    ordered['delta_growth_1m'] = ordered['delta_growth_1m'].fillna(0)
    ordered['delta_growth_1w'] = ordered['delta_growth_1w'].fillna(0)

    # Sort by 1W since 1M might be 0
    ordered = ordered.sort_values('delta_growth_1w', ascending=False)

    delta_1m = ordered['delta_growth_1m']
    delta_1w = ordered['delta_growth_1w']

    def _dual_palette(values):
        strong = []
        light = []
        for value in values:
            if value >= 0:
                strong.append(POSITIVE_COLOR_DARK)
                light.append(POSITIVE_COLOR_LIGHT)
            else:
                strong.append(NEGATIVE_COLOR_DARK)
                light.append(NEGATIVE_COLOR_LIGHT)
        return strong, light

    colors_1m_strong, colors_1m_light = _dual_palette(delta_1m)
    colors_1w_strong, colors_1w_light = _dual_palette(delta_1w)
    colors_1m = colors_1m_strong
    colors_1w = colors_1w_light

    labels_1m = [f"{value:+.1f}%" if pd.notna(value) and value != 0 else '' for value in delta_1m]
    labels_1w = [f"{value:+.1f}%" if pd.notna(value) else '' for value in delta_1w]

    combined_delta = pd.concat([delta_1m, delta_1w], axis=0)
    combined_abs_max = combined_delta.abs().max() if not combined_delta.empty else None
    max_abs_delta = float(combined_abs_max) if combined_abs_max is not None and not pd.isna(combined_abs_max) else 0.0
    padding = max(1.0, max_abs_delta * 0.12)
    axis_min = -max_abs_delta - padding
    axis_max = max_abs_delta + padding

    custom_1m = np.column_stack((ordered['avg_growth_1m'], np.full(len(ordered), market_avg_1m)))
    custom_1w = np.column_stack((ordered['avg_growth_1w'], np.full(len(ordered), market_avg_1w)))

    fig = go.Figure()
    # 1M Bar (might be empty/zero for now)
    fig.add_trace(
        go.Bar(
            y=ordered['industry'],
            x=delta_1m,
            name='1M so với thị trường',
            orientation='h',
            marker=dict(
                color=colors_1m,
                line=dict(color='rgba(255, 255, 255, 0.4)', width=0.8)
            ),
            text=labels_1m,
            textposition='outside',
            textfont=dict(size=14, color=FONT_COLOR, family=BOLD_FONT_FAMILY),
            cliponaxis=False,
            customdata=custom_1m,
            hovertemplate='<b>%{y}</b><br>1M: %{customdata[0]:+.2f}% | TT: %{customdata[1]:+.2f}%<br>Chênh lệch: %{x:+.2f} điểm<extra></extra>',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Bar(
            y=ordered['industry'],
            x=delta_1w,
            name='Daily (1W placeholder) so với thị trường',
            orientation='h',
            marker=dict(
                color=colors_1w,
                line=dict(color='rgba(255, 255, 255, 0.6)', width=0.8)
            ),
            text=labels_1w,
            textposition='outside',
            textfont=dict(size=14, color=FONT_COLOR, family=BOLD_FONT_FAMILY),
            cliponaxis=False,
            customdata=custom_1w,
            hovertemplate='<b>%{y}</b><br>Daily/1W: %{customdata[0]:+.2f}% | TT: %{customdata[1]:+.2f}%<br>Chênh lệch: %{x:+.2f} điểm<extra></extra>',
            showlegend=False
        )
    )

    fig.add_vline(x=0, line_width=2.8, line_dash='dash', line_color=ZERO_LINE_COLOR)

    fig.update_layout(
        title=dict(
            text='HIỆU SUẤT NGÀNH: DAILY SO VỚI THỊ TRƯỜNG',
            font=TITLE_FONT,
            x=0,
            pad=TITLE_PAD
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, size=13, family=BOLD_FONT_FAMILY),
        barmode='group',
        bargap=0.35,
        xaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            zeroline=False,
            title=dict(text='<b>Chênh lệch so với trung bình thị trường (điểm %)</b>'),
            ticksuffix='%',
            range=[axis_min, axis_max],
            tickfont=dict(size=13, color=FONT_COLOR, family=BOLD_FONT_FAMILY)
        ),
        yaxis=dict(
            showgrid=False,
            title=dict(text='<b>Ngành</b>'),
            autorange='reversed',
            tickfont=dict(size=13, color=FONT_COLOR, family=BOLD_FONT_FAMILY)
        ),
        height=430,
        margin=dict(l=140, r=40, t=70, b=100)
    )

    return fig


# ==================== MÔ-ĐUN 5: VỐN HÓA THEO NGÀNH ====================
def generate_market_cap_treemap(market_cap_df: pd.DataFrame):
    """Treemap tỷ trọng vốn hóa theo ngành từ dữ liệu thực."""

    if market_cap_df is None or market_cap_df.empty:
        return go.Figure()

    top_sectors = market_cap_df
    scale, unit_label = _get_scale_and_suffix(top_sectors['market_cap'], base_unit='VND')
    scaled_values = top_sectors['market_cap'] / scale
    palette = ['#48bb78', '#f6ad55', '#63b3ed', '#fc8181', '#dd6b20', '#9f7aea', '#38b2ac', '#ed8936']
    colors = [palette[i % len(palette)] for i in range(len(top_sectors))]

    fig = go.Figure(
        go.Treemap(
            labels=top_sectors['industry'],
            parents=[''] * len(top_sectors),
            values=scaled_values,
            marker=dict(colors=colors, line=dict(color='#f5f7fb', width=2)),
            texttemplate=f'<b>%{{label}}</b><br>%{{value:,.1f}} {unit_label}',
            textfont=dict(size=13, color='#1a202c', family='Inter, "Be VietNam Pro", "Segoe UI", sans-serif'),
            hovertemplate=f'<b>%{{label}}</b><br>Vốn hóa: %{{value:,.2f}} {unit_label}<extra></extra>'
        )
    )

    fig.update_layout(
        title=dict(
            text='TỶ TRỌNG DOANH NGHIỆP THEO NGÀNH',
            font=TITLE_FONT,
            x=0,
            pad=TITLE_PAD
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, size=11),
        height=350,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


# ==================== MÔ-ĐUN 6: DÒNG TIỀN KHỐI NGOẠI ====================
def generate_net_foreign_buying(foreign_flow_df: pd.DataFrame):
    """Hiển thị Top mua/bán ròng khối ngoại dạng 2 cột đối xứng."""

    if foreign_flow_df is None or foreign_flow_df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        fig.add_annotation(text='Không có dữ liệu giao dịch khối ngoại', xref='paper', yref='paper', x=0.5, y=0.5)
        return fig

    df = foreign_flow_df.copy()
    buys = df[df['foreign_buysell_20s'] > 0].nlargest(5, 'foreign_buysell_20s')
    sells = df[df['foreign_buysell_20s'] < 0].nsmallest(5, 'foreign_buysell_20s')
    all_values = pd.concat([buys['foreign_buysell_20s'], sells['foreign_buysell_20s']]).abs()
    scale, unit_label = _get_scale_and_suffix(all_values, base_unit='VND')

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Top Mua", "Top Bán"),
        horizontal_spacing=0.12
    )

    if not buys.empty:
        buy_scaled = buys['foreign_buysell_20s'] / scale
        buy_custom = np.column_stack((buys['industry'], buy_scaled)).tolist()
        fig.add_trace(
            go.Bar(
                x=buy_scaled,
                y=buys['ticker'],
                orientation='h',
                marker_color=POSITIVE_COLOR,
                text=[f"{val:,.1f}" for val in buy_scaled],
                textposition='outside',
                hovertemplate=f'%{{y}} · %{{customdata[0]}}<br>Mua ròng: %{{x:,.2f}} {unit_label}<extra></extra>',
                customdata=buy_custom,
                name='Top Mua'
            ),
            row=1,
            col=1
        )

    if not sells.empty:
        sell_values = np.abs(sells['foreign_buysell_20s']) / scale
        sell_custom = np.column_stack((sells['industry'], sell_values)).tolist()
        fig.add_trace(
            go.Bar(
                x=sell_values,
                y=sells['ticker'],
                orientation='h',
                marker_color=NEGATIVE_COLOR,
                text=[f"{val:,.1f}" for val in sell_values],
                textposition='outside',
                hovertemplate=f'%{{y}} · %{{customdata[0]}}<br>Bán ròng: %{{x:,.2f}} {unit_label}<extra></extra>',
                customdata=sell_custom,
                name='Top Bán'
            ),
            row=1,
            col=2
        )

    fig.update_layout(
        title=dict(
            text='GIAO DỊCH KHỐI NGOẠI (REAL-TIME)',
            font=TITLE_FONT,
            x=0,
            pad=TITLE_PAD
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, size=11),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=70, b=40)
    )
    
    # Update axes titles
    fig.update_xaxes(row=1, col=1, title_text=f'Giá trị mua ròng ({unit_label})')
    fig.update_xaxes(row=1, col=2, title_text=f'Giá trị bán ròng ({unit_label})')
    fig.update_xaxes(tickformat=',.0f')

    return fig


# ==================== MÔ-ĐUN 7: TƯƠNG QUAN LẠM PHÁT (LIQUIDITY/GROWTH) ====================
def generate_inflation_correlation(liquidity_df: pd.DataFrame):
    """Biểu đồ scatter thể hiện mối tương quan giữa tăng trưởng giá và thanh khoản."""

    if liquidity_df is None or liquidity_df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        fig.add_annotation(text='Không có dữ liệu thanh khoản', xref='paper', yref='paper', x=0.5, y=0.5)
        return fig

    df = liquidity_df.copy()
    numeric_cols = ['avg_trading_value_20d', 'price_growth_1m', 'market_cap']
    # Map price_growth_1w to 1m for visualization if 1m is missing/zero
    if 'price_growth_1w' in df.columns:
         df['price_growth_1m'] = df['price_growth_1w']

    for column in numeric_cols:
        df[column] = pd.to_numeric(df.get(column), errors='coerce')

    df = df.dropna(subset=['avg_trading_value_20d', 'price_growth_1m'])
    if df.empty:
        return go.Figure()

    df['market_cap'] = df['market_cap'].fillna(0)
    size_base = df['market_cap'].replace({0: np.nan}).max()
    if pd.isna(size_base) or size_base == 0:
        size_base = 1
    scale_x, unit_label = _get_scale_and_suffix(df['avg_trading_value_20d'], base_unit='VND')
    df['avg_trading_value_scaled'] = df['avg_trading_value_20d'] / scale_x
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['avg_trading_value_scaled'],
        y=df['price_growth_1m'],
        mode='markers',
        marker=dict(
            size=np.clip((df['market_cap'] / size_base) * 30, 8, 30),
            color=[POSITIVE_COLOR if val >= 0 else NEGATIVE_COLOR for val in df['price_growth_1m']],
            line=dict(color='#ffffff', width=0.5)
        ),
        text=df['ticker'],
        hovertemplate=f'%{{text}} · %{{customdata}}<br>Tăng trưởng: %{{y:.2f}}%<br>Thanh khoản: %{{x:,.2f}} {unit_label}<extra></extra>',
        customdata=df['industry']
    ))

    fig.update_layout(
        title=dict(
            text='TĂNG TRƯỞNG (DAILY) VS THANH KHOẢN',
            font=TITLE_FONT,
            x=0,
            pad=TITLE_PAD
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, size=11),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            zeroline=False,
            title=f'Thanh khoản ({unit_label})'
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            zeroline=False,
            title='Tăng trưởng (%)'
        ),
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig.update_xaxes(tickformat=',.0f')
    
    return fig


# ==================== MÔ-ĐUN 8: MA TRẬN TƯƠNG QUAN ====================
def generate_correlation_matrix(correlation_df: pd.DataFrame):
    """Hiển thị ma trận tương quan lợi suất thực tế."""

    if correlation_df is None or correlation_df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        fig.add_annotation(text='Không đủ dữ liệu để tính tương quan', xref='paper', yref='paper', x=0.5, y=0.5)
        return fig

    numeric_corr = correlation_df.apply(pd.to_numeric, errors='coerce')
    corr_values = numeric_corr.to_numpy(dtype=float)

    fig = go.Figure(data=go.Heatmap(
        z=corr_values,
        x=numeric_corr.columns,
        y=numeric_corr.index,
        colorscale=[
            [0, '#ebf4ff'],
            [0.5, '#90cdf4'],
            [1, '#2b6cb0']
        ],
        text=np.round(corr_values, 2),
        texttemplate='%{text:.1f}',
        textfont=dict(size=13, color='#1a202c'),
        hovertemplate='%{x} so với %{y}<br>Hệ số: %{z:.2f}<extra></extra>',
        colorbar=dict(
            thickness=15,
            len=0.7,
            bgcolor=PAPER_BG,
            tickfont=dict(color=FONT_COLOR),
            title=dict(text='Hệ số', side='right', font=dict(color=FONT_COLOR))
        )
    ))
    
    fig.update_layout(
        title=dict(
            text='MA TRẬN TƯƠNG QUAN',
            font=TITLE_FONT,
            x=0,
            pad=TITLE_PAD
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, size=11),
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

    left_col, right_col = st.columns((1.6, 1))

    with left_col:
        if overview_data.get('index_history') is not None:
             st.plotly_chart(
                generate_index_comparison_chart(overview_data.get('index_history')), use_container_width=True
            )
        else:
             st.info("Chưa có dữ liệu lịch sử chỉ số.")
             
        st.markdown(CHART_GAP_DIV, unsafe_allow_html=True)
        sector_perf_placeholder = st.empty()
        st.markdown(CHART_GAP_DIV, unsafe_allow_html=True)
        correlation_placeholder = st.empty()

    with right_col:
        market_cap_placeholder = st.empty()
        st.markdown(CHART_GAP_DIV, unsafe_allow_html=True)
        foreign_flow_placeholder = st.empty()
        st.markdown(CHART_GAP_DIV, unsafe_allow_html=True)
        liquidity_placeholder = st.empty()

    # Load detail data
    with st.spinner("Đang tải dữ liệu chi tiết..."):
        detail_data = load_detail_data()

    if detail_data:
        if detail_data.get('sector_perf') is not None and not detail_data['sector_perf'].empty:
             sector_perf_placeholder.plotly_chart(generate_sector_performance(detail_data['sector_perf']), use_container_width=True)
        else:
             sector_perf_placeholder.info("Chưa có dữ liệu ngành.")

        if detail_data.get('market_cap') is not None and not detail_data['market_cap'].empty:
             market_cap_placeholder.plotly_chart(generate_market_cap_treemap(detail_data['market_cap']), use_container_width=True)
        else:
             market_cap_placeholder.info("Chưa có dữ liệu vốn hóa.")
        
        if detail_data.get('foreign_flow') is not None and not detail_data['foreign_flow'].empty:
             foreign_flow_placeholder.plotly_chart(generate_net_foreign_buying(detail_data['foreign_flow']), use_container_width=True)
        else:
             foreign_flow_placeholder.info("Chưa có dữ liệu khối ngoại.")

        if detail_data.get('liquidity') is not None and not detail_data['liquidity'].empty:
             liquidity_placeholder.plotly_chart(generate_inflation_correlation(detail_data['liquidity']), use_container_width=True)
        else:
             liquidity_placeholder.info("Chưa có dữ liệu thanh khoản.")

    st.markdown(CHART_GAP_DIV, unsafe_allow_html=True)
    render_current_portfolio()


def main():
    """Giữ hàm main để có thể chạy file độc lập."""
    render_bang_dieu_hanh()


if __name__ == "__main__":

    main()
