"""
Module optimization_comparison.py
Tạo tab tổng hợp kết quả tối ưu hóa của các mô hình để so sánh và hỗ trợ quyết định đầu tư.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def create_comparison_table(results_dict):
    """
    Tạo bảng so sánh các mô hình tối ưu hóa.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
                           {'Tên mô hình': result_dict}
    
    Returns:
        pd.DataFrame: Bảng so sánh
    """
    comparison_data = []
    
    for model_name, result in results_dict.items():
        if result is None:
            continue
            
        metrics = calculate_portfolio_metrics(result)
        
        comparison_data.append({
            'Mô hình': model_name,
            'Lợi nhuận KV (%)': f"{metrics['expected_return']:.2f}",
            'Rủi ro - Std (%)': f"{metrics['volatility']:.2f}",
            'Tỷ lệ Sharpe': f"{metrics['sharpe_ratio']:.4f}",
            'Return/Risk': f"{metrics['return_risk_ratio']:.4f}",
            'Số mã CP': metrics['num_stocks'],
            'Tổng số cổ phiếu đầu tư': int(metrics['total_shares']),
            'Vốn sử dụng (VND)': f"{metrics['total_invested']:,.0f}",
            'Vốn còn lại (VND)': f"{metrics['leftover']:,.0f}",
            'Tỷ lệ sử dụng vốn (%)': f"{metrics['capital_utilization']:.2f}",
            'Chỉ số đa dạng hóa': f"{metrics['diversification_index']:.4f}"
        })
    
    return pd.DataFrame(comparison_data)


def highlight_best_values(df):
    """
    Tô màu các giá trị tốt nhất trong bảng so sánh.
    
    Args:
        df (pd.DataFrame): Bảng so sánh
    
    Returns:
        Styled DataFrame
    """
    def highlight_max(s):
        """Tô màu xanh đậm cho giá trị lớn nhất"""
        is_max = s == s.max()
        return ['background-color: #90EE90; font-weight: bold' if v else '' for v in is_max]
    
    def highlight_min(s):
        """Tô màu xanh đậm cho giá trị nhỏ nhất"""
        is_min = s == s.min()
        return ['background-color: #90EE90; font-weight: bold' if v else '' for v in is_min]
    
    # Chuyển đổi các cột về số để so sánh
    numeric_cols = ['Lợi nhuận KV (%)', 'Rủi ro - Std (%)', 'Tỷ lệ Sharpe', 
                   'Return/Risk', 'Tỷ lệ sử dụng vốn (%)', 'Chỉ số đa dạng hóa']
    
    styled_df = df.copy()
    
    for col in numeric_cols:
        if col in styled_df.columns:
            # Chuyển về float để so sánh
            styled_df[col] = styled_df[col].apply(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x)
    
    # Apply highlighting - Sử dụng cách tiếp cận đơn giản hơn
    styled = df.style
    
    # Các cột cần highlight MAX (giá trị cao = tốt)
    cols_max = ['Lợi nhuận KV (%)', 'Tỷ lệ Sharpe', 'Return/Risk', 
                'Tỷ lệ sử dụng vốn (%)', 'Chỉ số đa dạng hóa']
    
    for col in cols_max:
        if col in styled_df.columns:
            max_val = styled_df[col].max()
            def style_max(val, max_value=max_val, column=col):
                if column in styled_df.columns:
                    try:
                        val_float = float(str(val).replace(',', '')) if isinstance(val, str) else val
                        if abs(val_float - max_value) < 0.0001:
                            return 'background-color: #90EE90; font-weight: bold'
                    except:
                        pass
                return ''
            styled = styled.applymap(style_max, subset=[col])
    
    # Highlight MIN cho rủi ro (giá trị thấp = tốt)
    if 'Rủi ro - Std (%)' in styled_df.columns:
        min_val = styled_df['Rủi ro - Std (%)'].min()
        def style_min(val, min_value=min_val):
            try:
                val_float = float(str(val).replace(',', '')) if isinstance(val, str) else val
                if abs(val_float - min_value) < 0.0001:
                    return 'background-color: #90EE90; font-weight: bold'
            except:
                pass
            return ''
        styled = styled.applymap(style_min, subset=['Rủi ro - Std (%)'])
    
    return styled


def plot_risk_return_comparison(results_dict):
    """
    Vẽ biểu đồ so sánh rủi ro - lợi nhuận của các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
    """
    fig = go.Figure()
    
    for model_name, result in results_dict.items():
        if result is None:
            continue
        
        metrics = calculate_portfolio_metrics(result)
        
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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_sharpe_comparison(results_dict):
    """
    Vẽ biểu đồ cột so sánh tỷ lệ Sharpe của các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
    """
    model_names = []
    sharpe_ratios = []
    
    for model_name, result in results_dict.items():
        if result is None:
            continue
        
        metrics = calculate_portfolio_metrics(result)
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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_allocation_comparison(results_dict):
    """
    Vẽ biểu đồ so sánh phân bổ tài sản của các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
    """
    # Tạo subplot cho từng mô hình
    num_models = sum(1 for r in results_dict.values() if r is not None)
    
    if num_models == 0:
        st.warning("Không có dữ liệu để so sánh phân bổ.")
        return
    
    fig = make_subplots(
        rows=1, 
        cols=num_models,
        subplot_titles=list(results_dict.keys()),
        specs=[[{'type': 'pie'}] * num_models]
    )
    
    col_idx = 1
    for model_name, result in results_dict.items():
        if result is None:
            continue
        
        weights = result.get('Trọng số danh mục', {})
        
        # Lọc các trọng số > 0
        filtered_weights = {k: v for k, v in weights.items() if v > 0.001}
        
        if filtered_weights:
            fig.add_trace(
                go.Pie(
                    labels=list(filtered_weights.keys()),
                    values=list(filtered_weights.values()),
                    name=model_name,
                    textinfo='label+percent',
                    hovertemplate="<b>%{label}</b><br>Tỷ trọng: %{percent}<extra></extra>"
                ),
                row=1, col=col_idx
            )
            col_idx += 1
    
    fig.update_layout(
        title_text="So sánh Phân bổ Trọng số Danh mục",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_diversification_comparison(results_dict):
    """
    Vẽ biểu đồ so sánh mức độ đa dạng hóa của các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
    """
    model_names = []
    diversification_scores = []
    num_stocks = []
    
    for model_name, result in results_dict.items():
        if result is None:
            continue
        
        metrics = calculate_portfolio_metrics(result)
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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_radar_comparison(results_dict):
    """
    Vẽ biểu đồ radar so sánh toàn diện các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
    """
    if len(results_dict) == 0:
        return
    
    fig = go.Figure()
    
    # Chuẩn hóa các chỉ số về thang 0-100
    all_metrics = []
    for model_name, result in results_dict.items():
        if result is None:
            continue
        metrics = calculate_portfolio_metrics(result)
        all_metrics.append(metrics)
    
    if not all_metrics:
        return
    
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
    
    def normalize(value, min_val, max_val):
        if max_val == min_val:
            return 50
        return ((value - min_val) / (max_val - min_val)) * 100
    
    for model_name, result in results_dict.items():
        if result is None:
            continue
        
        metrics = calculate_portfolio_metrics(result)
        
        # Chuẩn hóa (volatility đảo ngược vì thấp = tốt)
        norm_return = normalize(metrics['expected_return'], min_return, max_return)
        norm_volatility = 100 - normalize(metrics['volatility'], min_volatility, max_volatility)
        norm_sharpe = normalize(metrics['sharpe_ratio'], min_sharpe, max_sharpe)
        norm_div = normalize(metrics['diversification_index'], min_div, max_div)
        norm_capital = normalize(metrics['capital_utilization'], min_capital, max_capital)
        
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
        title="Biểu đồ Radar - So sánh Toàn diện",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


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
            allocation_data[model_name] = [allocation.get(ticker, '-') for ticker in all_tickers]
    
    df_allocation = pd.DataFrame(allocation_data)
    
    st.dataframe(df_allocation, use_container_width=True, height=400)


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
    
    st.dataframe(df_weights, use_container_width=True, height=400)


def provide_investment_recommendation(results_dict):
    """
    Đưa ra khuyến nghị đầu tư dựa trên kết quả so sánh các mô hình.
    
    Args:
        results_dict (dict): Dictionary chứa kết quả của các mô hình
    """
    st.subheader("💡 Khuyến nghị Lựa chọn Phương án Đầu tư")
    
    if not results_dict or all(r is None for r in results_dict.values()):
        st.warning("Chưa có kết quả tối ưu hóa để đưa ra khuyến nghị.")
        return
    
    # Tính điểm cho từng mô hình dựa trên nhiều tiêu chí
    scores = {}
    
    for model_name, result in results_dict.items():
        if result is None:
            continue
        
        metrics = calculate_portfolio_metrics(result)
        
        # Điểm tổng hợp (weighted score)
        score = 0
        
        # Tỷ lệ Sharpe (40% trọng số)
        score += metrics['sharpe_ratio'] * 40
        
        # Lợi nhuận kỳ vọng (30% trọng số)
        score += metrics['expected_return'] * 30
        
        # Đa dạng hóa (20% trọng số)
        score += metrics['diversification_index'] * 20
        
        # Hiệu quả sử dụng vốn (10% trọng số)
        score += (metrics['capital_utilization'] / 100) * 10
        
        scores[model_name] = {
            'total_score': score,
            'sharpe': metrics['sharpe_ratio'],
            'return': metrics['expected_return'],
            'risk': metrics['volatility'],
            'diversification': metrics['diversification_index']
        }
    
    # Sắp xếp theo điểm tổng hợp
    sorted_models = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    
    # Hiển thị top 3 khuyến nghị
    st.markdown("### 🏆 Top 3 Phương án Được Khuyến nghị")
    st.info("💡 **Công thức tính điểm**: Sharpe (40%) + Lợi nhuận (30%) + Đa dạng hóa (20%) + Hiệu quả vốn (10%)")
    
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
    
    st.success(f"✅ Đã tải {len(valid_results)} kết quả tối ưu hóa")
    
    # Tab con cho các phần khác nhau
    tab1, tab2, tab3 = st.tabs([
        "📋 Bảng So sánh Tổng quan",
        "📊 Biểu đồ Phân tích",
        "💡 Khuyến nghị Đầu tư"
    ])
    
    with tab1:
        st.markdown("### 📋 Bảng So sánh Các Chỉ số Chính")
        comparison_df = create_comparison_table(valid_results)
        
        # Hiển thị bảng với highlight
        styled_df = highlight_best_values(comparison_df)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
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
        st.markdown("#### 🎯 So sánh Toàn diện - Biểu đồ Radar")
        st.caption("Tất cả chỉ số được chuẩn hóa về thang 0-100 để so sánh dễ dàng")
        plot_radar_comparison(valid_results)
        
        st.markdown("---")
        
        # Rủi ro - Lợi nhuận
        st.markdown("#### 📈 Rủi ro - Lợi nhuận")
        plot_risk_return_comparison(valid_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tỷ lệ Sharpe
            plot_sharpe_comparison(valid_results)
        
        with col2:
            # Đa dạng hóa
            plot_diversification_comparison(valid_results)
        
        # Phân bổ trọng số
        st.markdown("---")
        st.markdown("#### 🥧 Phân bổ Trọng số Danh mục")
        plot_allocation_comparison(valid_results)
        
        # Chi tiết phân bổ
        st.markdown("---")
        with st.expander("📋 Xem Chi tiết Trọng số & Số lượng Cổ phiếu"):
            col_a, col_b = st.columns(2)
            with col_a:
                display_weight_comparison(valid_results)
            with col_b:
                display_detailed_allocation(valid_results)
    
    with tab3:
        # Khuyến nghị
        provide_investment_recommendation(valid_results)
