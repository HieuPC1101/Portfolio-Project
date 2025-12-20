"""
Module optimization_comparison.py
Tạo tab tổng hợp kết quả tối ưu hóa của các mô hình để so sánh và hỗ trợ quyết định đầu tư.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Import utility functions
from utils.portfolio_utils import (
    normalize_metric,
    validate_result,
    calculate_max_drawdown_safe
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import UI styling
from utils.ui_styling import get_custom_css, create_metric_card, create_section_header


def calculate_portfolio_metrics(result):
    """
    Tính toán các chỉ số đánh giá danh mục đầu tư.
    
    Args:
        result (dict): Kết quả tối ưu hóa từ một mô hình
        
    Returns:
        dict: Các chỉ số đánh giá
    """
    metrics = {}
    
    # Lợi nhuận kỳ vọng (%)
    metrics['expected_return'] = result.get('Lợi nhuận kỳ vọng', 0) * 100
    
    # Rủi ro (độ lệch chuẩn) (%)
    metrics['volatility'] = result.get('Rủi ro (Độ lệch chuẩn)', 0) * 100
    
    # Tỷ lệ Sharpe
    metrics['sharpe_ratio'] = result.get('Tỷ lệ Sharpe', 0)
    
    # Số mã cổ phiếu trong danh mục
    allocation = result.get('Số mã cổ phiếu cần mua', {})
    metrics['num_stocks'] = len([k for k, v in allocation.items() if v > 0])
    
    # Tổng số lượng cổ phiếu
    metrics['total_shares'] = sum(allocation.values())
    
    # Số tiền đã đầu tư
    prices = result.get('Giá mã cổ phiếu', {})
    total_invested = sum(allocation.get(ticker, 0) * prices.get(ticker, 0) 
                        for ticker in allocation.keys())
    metrics['total_invested'] = total_invested
    
    # Số tiền còn lại
    metrics['leftover'] = result.get('Số tiền còn lại', 0)
    
    # Tỷ lệ sử dụng vốn (%)
    total_capital = total_invested + metrics['leftover']
    metrics['capital_utilization'] = (total_invested / total_capital * 100) if total_capital > 0 else 0
    
    # Tỷ lệ Return/Risk
    metrics['return_risk_ratio'] = (metrics['expected_return'] / metrics['volatility']) if metrics['volatility'] > 0 else 0
    
    # CVaR và CDaR nếu có
    metrics['cvar'] = result.get('Rủi ro CVaR', None)
    metrics['cdar'] = result.get('Rủi ro CDaR', None)
    
    # Maximum Drawdown (MDD) - Sử dụng utility function với fallback an toàn
    returns_data = result.get('ret_arr')
    metrics['max_drawdown'] = calculate_max_drawdown_safe(
        returns_data=returns_data,
        cdar=metrics['cdar'],
        volatility=metrics['volatility']
    )
    
    # Mức độ đa dạng hóa (Herfindahl Index)
    weights = result.get('Trọng số danh mục', {})
    if weights:
        weight_values = np.array(list(weights.values()))
        herfindahl = np.sum(weight_values ** 2)
        # Chuyển đổi thành chỉ số đa dạng hóa (1 = đa dạng tối đa, 0 = tập trung)
        metrics['diversification_index'] = (1 - herfindahl) / (1 - 1/len(weights)) if len(weights) > 1 else 0
    else:
        metrics['diversification_index'] = 0
    
    return metrics


def precompute_all_metrics(results_dict):
    """
    Pre-compute metrics cho tất cả mô hình một lần duy nhất.
    Tránh tính toán lặp lại khi render nhiều biểu đồ.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
                           {'Tên mô hình': result_dict}
    
    Returns:
        dict: {model_name: metrics_dict}
    
    Example:
        >>> results = {'Model A': {...}, 'Model B': {...}}
        >>> metrics_cache = precompute_all_metrics(results)
        >>> # Sử dụng cache thay vì tính lại
        >>> model_a_metrics = metrics_cache['Model A']
    """
    metrics_cache = {}
    
    for model_name, result in results_dict.items():
        if result is None:
            logger.warning(f"Result for {model_name} is None, skipping")
            continue
        
        # Validate result trước khi tính metrics
        if not validate_result(result):
            logger.warning(f"Result for {model_name} failed validation, skipping")
            continue
        
        try:
            metrics = calculate_portfolio_metrics(result)
            metrics_cache[model_name] = metrics
            logger.info(f"Pre-computed metrics for {model_name}")
        except Exception as e:
            logger.error(f"Failed to compute metrics for {model_name}: {e}")
            continue
    
    logger.info(f"Pre-computed metrics for {len(metrics_cache)}/{len(results_dict)} models")
    return metrics_cache


def create_comparison_table(results_dict, metrics_cache=None):
    """
    Tạo bảng so sánh các mô hình tối ưu hóa.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
                           {'Tên mô hình': result_dict}
        metrics_cache (dict, optional): Pre-computed metrics cache
    
    Returns:
        pd.DataFrame: Bảng so sánh
    """
    # Nếu không có cache, tính mới
    if metrics_cache is None:
        metrics_cache = precompute_all_metrics(results_dict)
    
    comparison_data = []
    
    for model_name, result in results_dict.items():
        if result is None or model_name not in metrics_cache:
            continue
        
        # Sử dụng metrics từ cache
        metrics = metrics_cache[model_name]
        
        comparison_data.append({
            'Mô hình': model_name,
            'Lợi nhuận KV (%)': metrics['expected_return'],
            'Rủi ro - Std (%)': metrics['volatility'],
            'Tỷ lệ Sharpe': metrics['sharpe_ratio'],
            'Return/Risk': metrics['return_risk_ratio'],
            'Chỉ số đa dạng hóa': metrics['diversification_index'],
            'Tỷ lệ sử dụng vốn (%)': metrics['capital_utilization'],
            'Số mã CP': int(metrics['num_stocks']) if metrics['num_stocks'] else 0,
            'Tổng số cổ phiếu đầu tư': int(metrics['total_shares']) if metrics['total_shares'] else 0,
            'Vốn sử dụng (VND)': float(metrics['total_invested']) if metrics['total_invested'] else 0.0,
            'Vốn còn lại (VND)': float(metrics['leftover']) if metrics['leftover'] else 0.0
        })
    
    return pd.DataFrame(comparison_data)


def highlight_best_values(df):
    """
    Tô màu chỉ số tốt nhất trong bảng so sánh.
    
    Args:
        df (pd.DataFrame): Bảng so sánh
    
    Returns:
        Styled DataFrame
    """
    styled = df.style
    
    # Format các cột số
    format_dict = {
        'Lợi nhuận KV (%)': '{:.2f}',
        'Rủi ro - Std (%)': '{:.2f}',
        'Tỷ lệ Sharpe': '{:.4f}',
        'Return/Risk': '{:.4f}',
        'Chỉ số đa dạng hóa': '{:.4f}',
        'Tỷ lệ sử dụng vốn (%)': '{:.2f}',
        'Vốn sử dụng (VND)': '{:,.0f}',
        'Vốn còn lại (VND)': '{:,.0f}'
    }
    styled = styled.format(format_dict)
    
    # Hàm highlight MAX (giá trị cao = tốt)
    def highlight_max(col):
        is_max = col == col.max()
        return ['background-color: #90EE90; font-weight: bold' if v else '' for v in is_max]
    
    # Hàm highlight MIN (giá trị thấp = tốt)
    def highlight_min(col):
        is_min = col == col.min()
        return ['background-color: #90EE90; font-weight: bold' if v else '' for v in is_min]
    
    # Highlight MAX cho các chỉ số cao = tốt
    max_cols = ['Lợi nhuận KV (%)', 'Tỷ lệ Sharpe', 'Return/Risk', 
                'Chỉ số đa dạng hóa', 'Tỷ lệ sử dụng vốn (%)']
    
    for col in max_cols:
        if col in df.columns:
            styled = styled.apply(highlight_max, subset=[col])
    
    # Highlight MIN cho rủi ro (thấp = tốt)
    min_cols = ['Rủi ro - Std (%)']
    for col in min_cols:
        if col in df.columns:
            styled = styled.apply(highlight_min, subset=[col])
    
    return styled


def plot_risk_return_comparison(results_dict, metrics_cache=None):
    """
    Vẽ biểu đồ so sánh rủi ro - lợi nhuận của các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
        metrics_cache (dict, optional): Pre-computed metrics cache
    """
    # Nếu không có cache, tính mới
    if metrics_cache is None:
        metrics_cache = precompute_all_metrics(results_dict)
    
    fig = go.Figure()
    
    for model_name in results_dict.keys():
        if model_name not in metrics_cache:
            continue
        
        metrics = metrics_cache[model_name]
        
        fig.add_trace(go.Scatter(
            x=[metrics['volatility']],
            y=[metrics['expected_return']],
            mode='markers+text',
            name=model_name,
            text=[model_name],
            textposition="top center",
            marker=dict(size=15, line=dict(width=2)),
            hovertemplate=f"<b>{model_name}</b><br>" +
                         f"Lợi nhuận: {metrics['expected_return']:.2f}%<br>" +
                         f"Rủi ro: {metrics['volatility']:.2f}%<br>" +
                         f"Sharpe: {metrics['sharpe_ratio']:.4f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="So sánh Rủi ro - Lợi nhuận các Mô hình",
        xaxis_title="Rủi ro (Độ lệch chuẩn) %",
        yaxis_title="Lợi nhuận kỳ vọng %",
        hovermode='closest',
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_sharpe_comparison(results_dict, metrics_cache=None):
    """
    Vẽ biểu đồ cột so sánh tỷ lệ Sharpe của các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
        metrics_cache (dict, optional): Pre-computed metrics cache
    """
    # Nếu không có cache, tính mới
    if metrics_cache is None:
        metrics_cache = precompute_all_metrics(results_dict)
    
    model_names = []
    sharpe_ratios = []
    
    for model_name in results_dict.keys():
        if model_name not in metrics_cache:
            continue
        
        metrics = metrics_cache[model_name]
        model_names.append(model_name)
        sharpe_ratios.append(metrics['sharpe_ratio'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=sharpe_ratios,
            text=[f"{sr:.4f}" for sr in sharpe_ratios],
            textposition='auto',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title="So sánh Tỷ lệ Sharpe",
        xaxis_title="Mô hình",
        yaxis_title="Tỷ lệ Sharpe",
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_allocation_comparison(results_dict):
    """
    Vẽ biểu đồ Stacked Bar Chart so sánh phân bổ tài sản của các mô hình.
    Dễ so sánh tỷ trọng của cùng mã cổ phiếu giữa các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
    """
    if not results_dict:
        st.warning("Không có dữ liệu để so sánh phân bổ.")
        return
    
    # Tập hợp tất cả các tickers
    all_tickers = set()
    for result in results_dict.values():
        if result:
            weights = result.get('Trọng số danh mục', {})
            all_tickers.update(weights.keys())
    
    all_tickers = sorted(list(all_tickers))
    model_names = [name for name, result in results_dict.items() if result]
    
    if not all_tickers or not model_names:
        st.warning("Không có dữ liệu phân bổ để hiển thị.")
        return
    
    # Tạo Stacked Bar Chart
    fig = go.Figure()
    
    # Thêm bar cho mỗi ticker
    for ticker in all_tickers:
        weights_across_models = []
        for model_name in model_names:
            result = results_dict[model_name]
            weights = result.get('Trọng số danh mục', {})
            weight_pct = weights.get(ticker, 0) * 100
            weights_across_models.append(weight_pct)
        
        fig.add_trace(go.Bar(
            name=ticker,
            x=model_names,
            y=weights_across_models,
            text=[f"{w:.1f}%" if w > 0 else "" for w in weights_across_models],
            textposition='inside',
            hovertemplate=f"<b>{ticker}</b><br>" +
                         "Mô hình: %{x}<br>" +
                         "Tỷ trọng: %{y:.2f}%<extra></extra>"
        ))
    
    fig.update_layout(
        title="So sánh Phân bổ Trọng số Danh mục (Stacked Bar)",
        xaxis_title="Mô hình",
        yaxis_title="Tỷ trọng (%)",
        barmode='stack',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_diversification_comparison(results_dict, metrics_cache=None):
    """
    Vẽ biểu đồ so sánh mức độ đa dạng hóa của các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
        metrics_cache (dict, optional): Pre-computed metrics cache
    """
    # Nếu không có cache, tính mới
    if metrics_cache is None:
        metrics_cache = precompute_all_metrics(results_dict)
    
    model_names = []
    diversification_scores = []
    num_stocks = []
    
    for model_name in results_dict.keys():
        if model_name not in metrics_cache:
            continue
        
        metrics = metrics_cache[model_name]
        model_names.append(model_name)
        diversification_scores.append(metrics['diversification_index'])
        num_stocks.append(metrics['num_stocks'])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Chỉ số Đa dạng hóa", "Số lượng Mã cổ phiếu"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Chỉ số đa dạng hóa
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=diversification_scores,
            text=[f"{ds:.4f}" for ds in diversification_scores],
            textposition='auto',
            marker_color='lightcoral',
            name='Đa dạng hóa'
        ),
        row=1, col=1
    )
    
    # Số lượng mã cổ phiếu
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=num_stocks,
            text=num_stocks,
            textposition='auto',
            marker_color='lightyellow',
            name='Số mã CP'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_radar_comparison(results_dict, metrics_cache=None):
    """
    Vẽ biểu đồ radar so sánh toàn diện các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
        metrics_cache (dict, optional): Pre-computed metrics cache
    """
    if len(results_dict) == 0:
        return
    
    # Nếu không có cache, tính mới
    if metrics_cache is None:
        metrics_cache = precompute_all_metrics(results_dict)
    
    if not metrics_cache:
        return
    
    fig = go.Figure()
    
    # Lấy tất cả metrics từ cache
    all_metrics = list(metrics_cache.values())
    
    
    # Tìm min/max để chuẩn hóa
    max_return = max(m['expected_return'] for m in all_metrics)
    min_return = min(m['expected_return'] for m in all_metrics)
    max_volatility = max(m['volatility'] for m in all_metrics)
    min_volatility = min(m['volatility'] for m in all_metrics)
    max_sharpe = max(m['sharpe_ratio'] for m in all_metrics)
    min_sharpe = min(m['sharpe_ratio'] for m in all_metrics)
    max_div = max(m['diversification_index'] for m in all_metrics)
    min_div = min(m['diversification_index'] for m in all_metrics)
    max_capital = max(m['capital_utilization'] for m in all_metrics)
    min_capital = min(m['capital_utilization'] for m in all_metrics)
    
    # Sử dụng utility function normalize_metric với padding
    for model_name in metrics_cache.keys():
        metrics = metrics_cache[model_name]
        
        # Chuẩn hóa với padding (volatility reverse vì thấp = tốt)
        norm_return = normalize_metric(metrics['expected_return'], min_return, max_return, reverse=False, padding=0.1)
        norm_volatility = normalize_metric(metrics['volatility'], min_volatility, max_volatility, reverse=True, padding=0.1)
        norm_sharpe = normalize_metric(metrics['sharpe_ratio'], min_sharpe, max_sharpe, reverse=False, padding=0.1)
        norm_div = normalize_metric(metrics['diversification_index'], min_div, max_div, reverse=False, padding=0.1)
        norm_capital = normalize_metric(metrics['capital_utilization'], min_capital, max_capital, reverse=False, padding=0.1)
        
        fig.add_trace(go.Scatterpolar(
            r=[norm_return, norm_volatility, norm_sharpe, norm_div, norm_capital],
            theta=['Lợi nhuận', 'An toàn<br>(Low Risk)', 'Sharpe Ratio', 'Đa dạng hóa', 'Hiệu quả vốn'],
            fill='toself',
            name=model_name,
            hovertemplate=f"<b>{model_name}</b><br>" +
                         "%{theta}: %{r:.1f}/100<extra></extra>"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Biểu đồ Radar",
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')


def display_detailed_allocation(results_dict):
    """
    Hiển thị bảng chi tiết phân bổ số lượng cổ phiếu của từng mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
    """
    st.subheader("📊 Chi tiết Phân bổ Số lượng Cổ phiếu")
    
    # Tạo DataFrame tổng hợp
    all_tickers = set()
    for result in results_dict.values():
        if result:
            all_tickers.update(result.get('Số mã cổ phiếu cần mua', {}).keys())
    
    all_tickers = sorted(list(all_tickers))
    
    allocation_data = {'Mã CP': all_tickers}
    
    for model_name, result in results_dict.items():
        if result is None:
            allocation_data[model_name] = ['-'] * len(all_tickers)
        else:
            allocation = result.get('Số mã cổ phiếu cần mua', {})
            # Convert tất cả về string để tránh lỗi Arrow serialization
            allocation_data[model_name] = [str(allocation.get(ticker, '-')) for ticker in all_tickers]
    
    df_allocation = pd.DataFrame(allocation_data)
    
    st.dataframe(df_allocation, width='stretch', height=400)


def display_weight_comparison(results_dict):
    """
    Hiển thị bảng so sánh trọng số của từng mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
    """
    st.subheader("📈 So sánh Trọng số Danh mục (%)")
    
    # Tạo DataFrame tổng hợp
    all_tickers = set()
    for result in results_dict.values():
        if result:
            all_tickers.update(result.get('Trọng số danh mục', {}).keys())
    
    all_tickers = sorted(list(all_tickers))
    
    weight_data = {'Mã CP': all_tickers}
    
    for model_name, result in results_dict.items():
        if result is None:
            weight_data[model_name] = ['-'] * len(all_tickers)
        else:
            weights = result.get('Trọng số danh mục', {})
            weight_data[model_name] = [f"{weights.get(ticker, 0)*100:.2f}%" if ticker in weights else '-' 
                                       for ticker in all_tickers]
    
    df_weights = pd.DataFrame(weight_data)
    
    st.dataframe(df_weights, width='stretch', height=400)


def provide_investment_recommendation(results_dict, metrics_cache=None):
    """
    Đưa ra khuyến nghị đầu tư với hệ thống chấm điểm chuẩn hóa (0-100).
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
        metrics_cache (dict, optional): Pre-computed metrics cache
    """
    st.markdown(create_section_header("Hệ thống Khuyến nghị Đầu tư", "💡"), unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 15px; border-radius: 10px; margin: 15px 0;">
        <p style="margin: 0; color: #1a73e8; font-weight: 600;">
            🎯 Phân tích dựa trên hệ thống chấm điểm ai chuẩn hóa, kết hợp nhiều tiêu chí quan trọng để đưa ra khuyến nghị tốt nhất.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not results_dict or all(r is None for r in results_dict.values()):
        st.warning("Chưa có kết quả tối ưu hóa để đưa ra khuyến nghị.")
        return
    
    # Nếu không có cache, tính mới
    if metrics_cache is None:
        metrics_cache = precompute_all_metrics(results_dict)
    
    if not metrics_cache:
        st.warning("Không có mô hình hợp lệ để đánh giá.")
        return
    
    # 1. Lấy tất cả metrics từ cache
    all_metrics = list(metrics_cache.values())
    
    # 2. Xác định Min/Max cho chuẩn hóa
    min_return = min(m['expected_return'] for m in all_metrics)
    max_return = max(m['expected_return'] for m in all_metrics)
    min_volatility = min(m['volatility'] for m in all_metrics)
    max_volatility = max(m['volatility'] for m in all_metrics)
    min_sharpe = min(m['sharpe_ratio'] for m in all_metrics)
    max_sharpe = max(m['sharpe_ratio'] for m in all_metrics)
    min_div = min(m['diversification_index'] for m in all_metrics)
    max_div = max(m['diversification_index'] for m in all_metrics)
    min_capital = min(m['capital_utilization'] for m in all_metrics)
    max_capital = max(m['capital_utilization'] for m in all_metrics)
    
    # 2. Tính điểm cho từng mô hình
    scores = {}
    score_details = []  # Để hiển thị bảng chi tiết
    
    for model_name in metrics_cache.keys():
        metrics = metrics_cache[model_name]
        
        # Chuẩn hóa từng thành phần (0-100) - Sử dụng utility function
        norm_sharpe = normalize_metric(metrics['sharpe_ratio'], min_sharpe, max_sharpe, reverse=False, padding=0)
        norm_return = normalize_metric(metrics['expected_return'], min_return, max_return, reverse=False, padding=0)
        norm_volatility = normalize_metric(metrics['volatility'], min_volatility, max_volatility, reverse=True, padding=0)
        norm_div = normalize_metric(metrics['diversification_index'], min_div, max_div, reverse=False, padding=0)
        norm_capital = normalize_metric(metrics['capital_utilization'], min_capital, max_capital, reverse=False, padding=0)
        
        # 5. Tính Điểm Tổng hợp (Weighted Score)
        total_score = (
            norm_sharpe * 0.4 +      # 40% Sharpe
            norm_return * 0.3 +      # 30% Return
            norm_div * 0.2 +         # 20% Diversification
            norm_capital * 0.1       # 10% Capital Efficiency
        )
        
        scores[model_name] = {
            'total_score': total_score,
            'sharpe': metrics['sharpe_ratio'],
            'return': metrics['expected_return'],
            'risk': metrics['volatility'],
            'diversification': metrics['diversification_index'],
            'capital_util': metrics['capital_utilization']
        }
        
        score_details.append({
            'Mô hình': model_name,
            'Return (raw)': f"{metrics['expected_return']:.2f}%",
            'Sharpe (raw)': f"{metrics['sharpe_ratio']:.4f}",
            'Risk (raw)': f"{metrics['volatility']:.2f}%",
            'Div (raw)': f"{metrics['diversification_index']:.4f}",
            'Capital (raw)': f"{metrics['capital_utilization']:.2f}%",
            'Score Return': f"{norm_return:.1f}",
            'Score Sharpe': f"{norm_sharpe:.1f}",
            'Score Risk': f"{norm_volatility:.1f}",
            'Score Div': f"{norm_div:.1f}",
            'Score Capital': f"{norm_capital:.1f}",
            'Tổng điểm': f"{total_score:.2f}"
        })
    
    # 6. Hiển thị bảng chi tiết điểm số (minh bạch hóa)
    with st.expander("📊 Chi tiết Bảng Điểm - Cách tính Điểm số", expanded=False):
        st.markdown("""
        **Phương pháp chấm điểm chuẩn hóa (Normalized Scoring)**
        
        1. **Thu thập dữ liệu thô**: Lấy các chỉ số từ tất cả mô hình
        2. **Chuẩn hóa về thang 0-100**: 
           - Công thức: `Score = ((Value - Min) / (Max - Min)) × 100`
           - Đảo ngược cho Rủi ro: `Score = ((Max - Value) / (Max - Min)) × 100`
        3. **Tính Tổng điểm**: `Sharpe×40% + Return×30% + Div×20% + Capital×10%`
        
        **Bảng chi tiết các thành phần điểm:**
        """)
        
        df_scores = pd.DataFrame(score_details)
        st.dataframe(df_scores, width='stretch', height=300)
        
        st.caption("💡 Cột 'Score' là điểm chuẩn hóa (0-100), cột 'raw' là giá trị gốc")
    
    # Sắp xếp theo điểm tổng hợp
    sorted_models = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    
    # Hiển thị top 3 khuyến nghị với styled header
    st.markdown("### 🏆 Top 3 Phương án Được Khuyến nghị")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%); 
                padding: 12px 20px; 
                border-radius: 8px; 
                border-left: 4px solid #00acc1;
                margin-bottom: 20px;">
        <p style="margin: 0; color: #006064; font-size: 14px;">
            💡 <strong>Công thức tính điểm (Thang 0-100):</strong> Sharpe (40%) + Lợi nhuận (30%) + Đa dạng hóa (20%) + Hiệu quả vốn (10%)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    for rank, (model_name, score_data) in enumerate(sorted_models[:3], 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        
        # Tự động mở rộng top 1
        is_expanded = (rank == 1)
        with st.expander(f"{medal} #{rank}: **{model_name}** (Điểm: {score_data['total_score']:.2f})", expanded=is_expanded):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tỷ lệ Sharpe", f"{score_data['sharpe']:.4f}")
                st.metric("Lợi nhuận KV", f"{score_data['return']:.2f}%")
            
            with col2:
                st.metric("Rủi ro (Std)", f"{score_data['risk']:.2f}%")
                st.metric("Return/Risk", f"{score_data['return']/score_data['risk']:.4f}")
            
            with col3:
                st.metric("Đa dạng hóa", f"{score_data['diversification']:.4f}")
                st.metric("Sử dụng vốn", f"{score_data['capital_util']:.1f}%")
            
            # Đưa ra nhận xét
            if rank == 1:
                st.success(f"✅ **{model_name}** là lựa chọn tốt nhất với hiệu suất tổng hợp cao nhất.")
            
            # Phân tích điểm mạnh
            strengths = []
            if score_data['sharpe'] == max(s['sharpe'] for s in scores.values()):
                strengths.append("Tỷ lệ Sharpe cao nhất")
            if score_data['return'] == max(s['return'] for s in scores.values()):
                strengths.append("Lợi nhuận kỳ vọng cao nhất")
            if score_data['risk'] == min(s['risk'] for s in scores.values()):
                strengths.append("Rủi ro thấp nhất")
            if score_data['diversification'] == max(s['diversification'] for s in scores.values()):
                strengths.append("Đa dạng hóa tốt nhất")
            
            if strengths:
                st.info(f"**Điểm mạnh:** {', '.join(strengths)}")
    
    # Hướng dẫn lựa chọn
    st.markdown("---")
    st.markdown("### 📝 Hướng dẫn Lựa chọn")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Chọn mô hình phù hợp với mục tiêu:**
        - **Max Sharpe / Markowitz**: Cân bằng lợi nhuận và rủi ro
        - **Min Volatility**: Ưu tiên an toàn, ít biến động
        - **Min CVaR / Min CDaR**: Phòng ngừa tổn thất cực đoan
        - **HRP**: Đa dạng hóa thông minh, phân tán rủi ro
        """)
    
    with col2:
        st.markdown("""
        **🔍 Các tiêu chí quan trọng:**
        - **Tỷ lệ Sharpe**: Hiệu suất điều chỉnh theo rủi ro
        - **Return/Risk**: Lợi nhuận trên mỗi đơn vị rủi ro
        - **Đa dạng hóa**: Mức độ phân tán đầu tư
        - **Sử dụng vốn**: Hiệu quả tận dụng nguồn vốn
        """)


def render_optimization_comparison_tab(results_dict):
    """
    Render tab tổng hợp kết quả tối ưu hóa.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
                           {'Tên mô hình': result_dict}
    """
    st.title("📊 Tổng hợp & So sánh Kết quả Tối ưu hóa")
    
    if not results_dict or all(r is None for r in results_dict.values()):
        st.info("""
        👋 Chào mừng đến với tab **Tổng hợp Kết quả**!
        
        📌 **Hướng dẫn sử dụng:**
        1. Chọn tab **"Tự chọn mã cổ phiếu"** hoặc **"Hệ thống đề xuất mã cổ phiếu tự động"**
        2. Chạy các mô hình tối ưu hóa (Markowitz, Max Sharpe, Min Volatility, v.v.)
        3. Kết quả sẽ được tự động lưu và hiển thị ở đây để so sánh
        
        💡 Tab này giúp bạn:
        - So sánh hiệu suất các mô hình
        - Phân tích rủi ro - lợi nhuận
        - Đưa ra quyết định đầu tư tối ưu
        """)
        return
    
    # Lọc các kết quả hợp lệ
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    
    if not valid_results:
        st.warning("Không có kết quả tối ưu hóa nào để hiển thị.")
        return
    
    
    # Inject custom CSS for better styling
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Pre-compute tất cả metrics một lần duy nhất (Performance Optimization)
    logger.info("Pre-computing metrics for all models...")
    metrics_cache = precompute_all_metrics(valid_results)
    logger.info(f"Successfully cached metrics for {len(metrics_cache)} models")
    
    # Summary metrics cards with gradients
    st.markdown("### 📈 Tổng quan Nhanh")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            create_metric_card("Số mô hình", len(valid_results), "purple"),
            unsafe_allow_html=True
        )
    
    with col2:
        all_metrics = list(metrics_cache.values())
        avg_return = sum(m['expected_return'] for m in all_metrics) / len(all_metrics) if all_metrics else 0
        st.markdown(
            create_metric_card("Lợi nhuận TB", f"{avg_return:.1f}%", "pink"),
            unsafe_allow_html=True
        )
    
    with col3:
        avg_sharpe = sum(m['sharpe_ratio'] for m in all_metrics) / len(all_metrics) if all_metrics else 0
        st.markdown(
            create_metric_card("Sharpe TB", f"{avg_sharpe:.3f}", "blue"),
            unsafe_allow_html=True
        )
    
    with col4:
        avg_stocks = sum(m['num_stocks'] for m in all_metrics) / len(all_metrics) if all_metrics else 0
        st.markdown(
            create_metric_card("Số mã TB", f"{avg_stocks:.1f}", "green"),
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tab con cho các phần khác nhau
    tab1, tab2, tab3 = st.tabs([
        "📋 Bảng So sánh Tổng quan",
        "📊 Biểu đồ Phân tích",
        "💡 Khuyến nghị Đầu tư"
    ])
    
    with tab1:
        st.markdown("### 📋 Bảng So sánh Các Chỉ số Chính")
        comparison_df = create_comparison_table(valid_results, metrics_cache)
        
        # Hiển thị bảng với highlight
        styled_df = highlight_best_values(comparison_df)
        st.dataframe(styled_df, width='stretch', height=400)
        
        st.markdown("""
        **📌 Chú thích:**
        - <span style="background-color: #90EE90; font-weight: bold; padding: 2px 6px;">Màu xanh đậm</span>: Giá trị tốt nhất trong cột
        - **Lợi nhuận KV**: Lợi nhuận kỳ vọng hàng năm (càng cao càng tốt)
        - **Rủi ro - Std**: Độ lệch chuẩn - biến động giá (càng thấp càng an toàn)
        - **Tỷ lệ Sharpe**: Hiệu suất điều chỉnh rủi ro (càng cao càng tốt)
        - **Return/Risk**: Tỷ lệ lợi nhuận/rủi ro trực tiếp (càng cao càng tốt)
        - **Chỉ số đa dạng hóa**: 0-1, với 1 là đa dạng hoàn hảo (càng cao càng phân tán)
        """, unsafe_allow_html=True)
        
        # Nút download
        csv = comparison_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Tải xuống bảng so sánh (CSV)",
            data=csv,
            file_name="so_sanh_toi_uu.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.markdown("### 📊 Biểu đồ Phân tích So sánh")
        
        # Biểu đồ Radar tổng quan
        st.markdown("#### Biểu đồ Radar")
        plot_radar_comparison(valid_results, metrics_cache)
        
        st.markdown("---")
        
        # Rủi ro - Lợi nhuận
        st.markdown("#### 📈 Rủi ro - Lợi nhuận")
        plot_risk_return_comparison(valid_results, metrics_cache)
        
        st.markdown("---")
        st.markdown("◼️### 📊 So sánh Chi tiết theo Chỉ số")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⚡ Tỷ lệ Sharpe**")
            st.caption("Hiệu suất điều chỉnh rủi ro (càng cao càng tốt)")
            plot_sharpe_comparison(valid_results, metrics_cache)
        
        with col2:
            st.markdown("**🎯 Mức độ Đa dạng hóa**")
            st.caption("Chỉ số đa dạng hóa & số lượng mã cổ phiếu")
            plot_diversification_comparison(valid_results, metrics_cache)
        
        # Phân bổ trọng số
        st.markdown("---")
        st.markdown("####  Phân bổ Trọng số Danh mục")
        plot_allocation_comparison(valid_results)
        
        # Chi tiết phân bổ
        st.markdown("---")
        with st.expander("🔍 Xem Chi tiết Trọng số & Số lượng Cổ phiếu", expanded=False):
            st.info("📌 Bảng chi tiết phân bổ cổ phiếu cho từng mô hình")
            col_a, col_b = st.columns(2)
            with col_a:
                display_weight_comparison(valid_results)
            with col_b:
                display_detailed_allocation(valid_results)
    
    with tab3:
        # Khuyến nghị
        provide_investment_recommendation(valid_results, metrics_cache)
