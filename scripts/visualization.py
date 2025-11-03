"""
Module visualization.py
Chứa các hàm vẽ biểu đồ: giá cổ phiếu, đường biên hiệu quả, backtesting.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

try:
    import pandas_ta as ta
except ImportError:
    ta = None
    st.warning("Thư viện pandas_ta chưa được cài đặt. Vui lòng chạy: pip install pandas-ta")


def calculate_technical_indicators(data, ticker, indicators=None):
    """
    Tính toán các chỉ báo kỹ thuật cho một cổ phiếu.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá với cột 'time' và ticker
        ticker (str): Mã cổ phiếu
        indicators (list): Danh sách chỉ báo cần tính ['SMA', 'EMA', 'RSI', 'MACD', 'BB']
        
    Returns:
        pd.DataFrame: Dữ liệu với các chỉ báo kỹ thuật
    """
    if indicators is None or not indicators:
        return data
    
    if ta is None:
        st.warning("Không thể tính chỉ báo kỹ thuật. Vui lòng cài đặt pandas-ta")
        return data
    
    # Tạo bản sao để không thay đổi dữ liệu gốc
    df = data.copy()
    
    # Đảm bảo có cột giá
    if ticker not in df.columns:
        return df
    
    # Tạo DataFrame tạm với tên cột chuẩn
    temp_df = pd.DataFrame()
    temp_df['close'] = df[ticker]
    
    try:
        # Tính SMA (Simple Moving Average)
        if 'SMA_20' in indicators:
            df[f'{ticker}_SMA_20'] = ta.sma(temp_df['close'], length=20)
        if 'SMA_50' in indicators:
            df[f'{ticker}_SMA_50'] = ta.sma(temp_df['close'], length=50)
        
        # Tính EMA (Exponential Moving Average)
        if 'EMA_20' in indicators:
            df[f'{ticker}_EMA_20'] = ta.ema(temp_df['close'], length=20)
        if 'EMA_50' in indicators:
            df[f'{ticker}_EMA_50'] = ta.ema(temp_df['close'], length=50)
        
        # Tính RSI (Relative Strength Index)
        if 'RSI' in indicators:
            df[f'{ticker}_RSI'] = ta.rsi(temp_df['close'], length=14)
        
        # Tính MACD (Moving Average Convergence Divergence)
        if 'MACD' in indicators:
            macd = ta.macd(temp_df['close'], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                df[f'{ticker}_MACD'] = macd[f'MACD_12_26_9']
                df[f'{ticker}_MACD_signal'] = macd[f'MACDs_12_26_9']
                df[f'{ticker}_MACD_hist'] = macd[f'MACDh_12_26_9']
        
        # Tính Bollinger Bands
        if 'BB' in indicators:
            bb = ta.bbands(temp_df['close'], length=20, std=2)
            if bb is not None and not bb.empty:
                df[f'{ticker}_BB_upper'] = bb[f'BBU_20_2.0']
                df[f'{ticker}_BB_middle'] = bb[f'BBM_20_2.0']
                df[f'{ticker}_BB_lower'] = bb[f'BBL_20_2.0']
        
    except Exception as e:
        st.warning(f"Lỗi khi tính chỉ báo cho {ticker}: {str(e)}")
    
    return df


def plot_interactive_stock_chart_with_indicators(data, tickers, indicators=None):
    """
    Vẽ biểu đồ giá cổ phiếu tương tác với các chỉ báo kỹ thuật.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        tickers (list): Danh sách mã cổ phiếu
        indicators (list): Danh sách chỉ báo cần hiển thị
    """
    if data.empty:
        st.warning("Không có dữ liệu để hiển thị biểu đồ.")
        return
    
    # Nếu không chọn chỉ báo, dùng biểu đồ cũ
    if not indicators or not indicators:
        plot_interactive_stock_chart(data, tickers)
        return
    
    # Reset index
    df = data.reset_index()
    
    # Xác định số lượng subplot cần thiết
    num_subplots = 1  # Luôn có biểu đồ giá chính
    has_rsi = 'RSI' in indicators
    has_macd = 'MACD' in indicators
    
    if has_rsi:
        num_subplots += 1
    if has_macd:
        num_subplots += 1
    
    # Tạo subplot
    row_heights = [0.6]  # Biểu đồ giá chiếm 60%
    if has_rsi:
        row_heights.append(0.2)  # RSI 20%
    if has_macd:
        row_heights.append(0.2)  # MACD 20%
    
    subplot_titles = ['Giá cổ phiếu']
    if has_rsi:
        subplot_titles.append('RSI')
    if has_macd:
        subplot_titles.append('MACD')
    
    fig = make_subplots(
        rows=num_subplots, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )
    
    # Vẽ giá cho từng cổ phiếu
    for ticker in tickers:
        if ticker not in df.columns:
            continue
        
        # Tính chỉ báo cho ticker này
        df_with_indicators = calculate_technical_indicators(df, ticker, indicators)
        
        # Vẽ đường giá
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators['time'],
                y=df_with_indicators[ticker],
                name=ticker,
                mode='lines',
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # Vẽ SMA
        if 'SMA_20' in indicators and f'{ticker}_SMA_20' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_SMA_20'],
                    name=f'{ticker} SMA(20)',
                    mode='lines',
                    line=dict(dash='dash', width=1),
                    visible='legendonly'
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in indicators and f'{ticker}_SMA_50' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_SMA_50'],
                    name=f'{ticker} SMA(50)',
                    mode='lines',
                    line=dict(dash='dot', width=1),
                    visible='legendonly'
                ),
                row=1, col=1
            )
        
        # Vẽ EMA
        if 'EMA_20' in indicators and f'{ticker}_EMA_20' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_EMA_20'],
                    name=f'{ticker} EMA(20)',
                    mode='lines',
                    line=dict(dash='dash', width=1),
                    visible='legendonly'
                ),
                row=1, col=1
            )
        
        if 'EMA_50' in indicators and f'{ticker}_EMA_50' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_EMA_50'],
                    name=f'{ticker} EMA(50)',
                    mode='lines',
                    line=dict(dash='dot', width=1),
                    visible='legendonly'
                ),
                row=1, col=1
            )
        
        # Vẽ Bollinger Bands
        if 'BB' in indicators:
            if f'{ticker}_BB_upper' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_indicators['time'],
                        y=df_with_indicators[f'{ticker}_BB_upper'],
                        name=f'{ticker} BB Upper',
                        mode='lines',
                        line=dict(dash='dot', width=1, color='gray'),
                        visible='legendonly'
                    ),
                    row=1, col=1
                )
            
            if f'{ticker}_BB_middle' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_indicators['time'],
                        y=df_with_indicators[f'{ticker}_BB_middle'],
                        name=f'{ticker} BB Middle',
                        mode='lines',
                        line=dict(dash='dash', width=1, color='gray'),
                        visible='legendonly'
                    ),
                    row=1, col=1
                )
            
            if f'{ticker}_BB_lower' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_indicators['time'],
                        y=df_with_indicators[f'{ticker}_BB_lower'],
                        name=f'{ticker} BB Lower',
                        mode='lines',
                        line=dict(dash='dot', width=1, color='gray'),
                        visible='legendonly'
                    ),
                    row=1, col=1
                )
        
        # Vẽ RSI
        current_row = 2 if has_rsi else None
        if has_rsi and f'{ticker}_RSI' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_RSI'],
                    name=f'{ticker} RSI',
                    mode='lines',
                    line=dict(width=2)
                ),
                row=current_row, col=1
            )
            
            # Thêm vùng quá mua/quá bán
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Quá mua", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Quá bán", row=current_row, col=1)
        
        # Vẽ MACD
        macd_row = num_subplots if has_macd else None
        if has_macd and f'{ticker}_MACD' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_MACD'],
                    name=f'{ticker} MACD',
                    mode='lines',
                    line=dict(width=2, color='blue')
                ),
                row=macd_row, col=1
            )
            
            if f'{ticker}_MACD_signal' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_indicators['time'],
                        y=df_with_indicators[f'{ticker}_MACD_signal'],
                        name=f'{ticker} Signal',
                        mode='lines',
                        line=dict(width=2, color='orange')
                    ),
                    row=macd_row, col=1
                )
            
            if f'{ticker}_MACD_hist' in df_with_indicators.columns:
                fig.add_trace(
                    go.Bar(
                        x=df_with_indicators['time'],
                        y=df_with_indicators[f'{ticker}_MACD_hist'],
                        name=f'{ticker} Histogram',
                        marker_color='gray',
                        opacity=0.5
                    ),
                    row=macd_row, col=1
                )
    
    # Cập nhật layout
    fig.update_layout(
        title="Biểu đồ giá cổ phiếu với chỉ báo kỹ thuật",
        xaxis_title="Thời gian",
        hovermode="x unified",
        height=600 if num_subplots == 1 else 800,
        showlegend=True,
        template="plotly_white"
    )
    
    # Cập nhật trục y
    fig.update_yaxes(title_text="Giá (VND)", row=1, col=1)
    if has_rsi:
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    if has_macd:
        fig.update_yaxes(title_text="MACD", row=macd_row, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_interactive_stock_chart(data, tickers):
    """
    Vẽ biểu đồ giá cổ phiếu tương tác sử dụng Plotly Express.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        tickers (list): Danh sách mã cổ phiếu
    """
    if data.empty:
        st.warning("Không có dữ liệu để hiển thị biểu đồ.")
        return

    # Reset index để hiển thị cột 'time' dưới dạng trục X
    data_reset = data.reset_index()
    
    # Định dạng dữ liệu cho biểu đồ dạng dài
    data_long = pd.melt(
        data_reset,
        id_vars=['time'],
        value_vars=tickers,
        var_name='Mã cổ phiếu',
        value_name='Giá đóng cửa'
    )

    # Sử dụng Plotly Express để vẽ biểu đồ
    fig = px.line(
        data_long,
        x='time',
        y='Giá đóng cửa',
        color='Mã cổ phiếu',
        title="Biểu đồ giá cổ phiếu",
        labels={"time": "Thời gian", "Giá đóng cửa": "Giá cổ phiếu (VND)"},
        template="plotly_white"
    )

    # Tuỳ chỉnh giao diện
    fig.update_layout(
        xaxis_title="Thời gian",
        yaxis_title="Giá cổ phiếu (VND)",
        legend_title="Mã cổ phiếu",
        hovermode="x unified"
    )

    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_efficient_frontier(ret_arr, vol_arr, sharpe_arr, all_weights, tickers, max_sharpe_idx, optimal_weights):
    """
    Vẽ biểu đồ đường biên hiệu quả.
    
    Args:
        ret_arr (np.array): Mảng lợi nhuận kỳ vọng
        vol_arr (np.array): Mảng độ lệch chuẩn (rủi ro)
        sharpe_arr (np.array): Mảng tỷ lệ Sharpe
        all_weights (np.array): Ma trận trọng số tất cả các danh mục
        tickers (list): Danh sách mã cổ phiếu
        max_sharpe_idx (int): Index của danh mục tối ưu
        optimal_weights (np.array): Trọng số tối ưu
    """
    # Chuẩn bị thông tin hover
    hover_texts = [
        ", ".join([f"{tickers[j]}: {weight * 100:.2f}%" for j, weight in enumerate(weights)])
        for weights in all_weights
    ]

    fig = px.scatter(
        x=vol_arr,
        y=ret_arr,
        color=sharpe_arr,
        hover_data={
            'Tỷ lệ Sharpe': sharpe_arr,
            'Thông tin danh mục': hover_texts
        },
        labels={'x': 'Rủi ro (Độ lệch chuẩn)', 'y': 'Lợi nhuận kỳ vọng', 'color': 'Tỷ lệ Sharpe'},
        title='Đường biên hiệu quả Markowitz'
    )

    # Đánh dấu danh mục tối ưu
    fig.add_scatter(
        x=[vol_arr[max_sharpe_idx]],
        y=[ret_arr[max_sharpe_idx]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Danh mục tối ưu',
        hovertext=[", ".join([f"{tickers[j]}: {optimal_weights[j] * 100:.2f}%" for j in range(len(tickers))])]
    )
    st.plotly_chart(fig)


def display_results(original_name, result):
    """
    Hiển thị kết quả tối ưu hóa với giao diện đẹp.
    
    Args:
        original_name (str): Tên mô hình
        result (dict): Kết quả tối ưu hóa
    """
    if result:
        st.markdown(f"## {original_name}")
        st.markdown("### Hiệu suất danh mục:")

        # Lợi nhuận kỳ vọng
        st.write(f"- **Lợi nhuận kỳ vọng:** {result.get('Lợi nhuận kỳ vọng', 0):.2%}")

        # Rủi ro (Độ lệch chuẩn)
        risk_std = result.get('Rủi ro (Độ lệch chuẩn)', 0)
        if risk_std == 0:
            st.write("- **Rủi ro (Độ lệch chuẩn):** Chỉ số không áp dụng cho mô hình này")
        else:
            st.write(f"- **Rủi ro (Độ lệch chuẩn):** {risk_std:.2%}")

        # Rủi ro CVaR
        if "Rủi ro CVaR" in result:
            st.write(f"- **Mức tổn thất trung bình trong tình huống xấu nhất:** {result['Rủi ro CVaR']:.2%}")

        # Rủi ro CDaR
        if "Rủi ro CDaR" in result:
            st.write(f"- **Mức giảm giá trị trung bình trong giai đoạn có sự giảm giá trị sâu:** {result['Rủi ro CDaR']:.2%}")

        # Tỷ lệ Sharpe
        sharpe_ratio = result.get('Tỷ lệ Sharpe', 0)
        if sharpe_ratio == 0:
            st.write("- **Tỷ lệ Sharpe:** Chỉ số không áp dụng cho mô hình này")
        else:
            st.write(f"- **Tỷ lệ Sharpe:** {sharpe_ratio:.2f}")

        # Trọng số danh mục
        weights = result["Trọng số danh mục"]
        tickers = list(weights.keys())

        # Tạo bảng trọng số
        weights_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Trọng số (%)"])
        weights_df["Trọng số (%)"] = weights_df["Trọng số (%)"] * 100

        # Giá cổ phiếu và phân bổ cổ phiếu
        latest_prices = result.get("Giá cổ phiếu", {})
        allocation = result.get("Số cổ phiếu cần mua", {})

        # Nếu không có phân bổ, mặc định là 0
        allocation = {ticker: allocation.get(ticker, 0) for ticker in tickers}
        latest_prices = {ticker: latest_prices.get(ticker, 0) for ticker in tickers}

        # Tạo DataFrame kết hợp các thông tin
        combined_data = {
            "Cổ phiếu": tickers,
            "Giá cổ phiếu": [f"{latest_prices.get(ticker, 0):.2f}" for ticker in tickers],
            "Trọng số (%)": [f"{weights_df.loc[ticker, 'Trọng số (%)']:.2f}" for ticker in tickers],
            "Số cổ phiếu cần mua": [allocation.get(ticker, 0) for ticker in tickers]
        }
        
        # Chuyển đổi thành DataFrame và hiển thị
        combined_df = pd.DataFrame(combined_data)

        # Hiển thị bảng kết hợp
        st.markdown("### Bảng phân bổ danh mục đầu tư:")
        st.table(combined_df)
        st.write(f"- **Số tiền còn lại:** {round(result.get('Số tiền còn lại', 0))}")


def backtest_portfolio(symbols, weights, start_date, end_date, fetch_stock_data_func, benchmark_symbols=["VNINDEX", "VN30", "HNX30", "HNXINDEX"]):
    """
    Hàm backtesting danh mục đầu tư, hỗ trợ nhiều chỉ số benchmark và hiển thị biểu đồ tương tác.

    Args:
        symbols (list): Danh sách mã cổ phiếu trong danh mục
        weights (list): Trọng số của mỗi mã cổ phiếu
        start_date (str): Ngày bắt đầu (định dạng 'YYYY-MM-DD')
        end_date (str): Ngày kết thúc (định dạng 'YYYY-MM-DD')
        fetch_stock_data_func (function): Hàm lấy dữ liệu giá cổ phiếu
        benchmark_symbols (list): Danh sách các chỉ số benchmark

    Returns:
        dict: Kết quả backtesting bao gồm Sharpe Ratio, Maximum Drawdown, và lợi suất tích lũy
    """
    # Lấy dữ liệu giá cổ phiếu trong danh mục
    stock_data, skipped_tickers = fetch_stock_data_func(symbols, start_date, end_date)
    if skipped_tickers:
        st.warning(f"Các mã không tải được dữ liệu: {', '.join(skipped_tickers)}")

    if stock_data.empty:
        st.error("Không có dữ liệu để backtesting.")
        return

    # Tính lợi suất hàng ngày của danh mục
    returns = stock_data.pct_change().dropna()
    portfolio_returns = returns.dot(weights)  # Lợi suất danh mục đầu tư
    cumulative_returns = (1 + portfolio_returns).cumprod()  # Lợi suất tích lũy

    # Lấy dữ liệu benchmark
    benchmark_data = {}
    for benchmark in benchmark_symbols:
        benchmark_df, _ = fetch_stock_data_func([benchmark], start_date, end_date)
        if not benchmark_df.empty:
            benchmark_returns = benchmark_df.pct_change().dropna()
            benchmark_cumulative = (1 + benchmark_returns[benchmark]).cumprod()
            benchmark_data[benchmark] = benchmark_cumulative
        else:
            st.warning(f"Không có dữ liệu benchmark cho {benchmark}.")

    # Gộp dữ liệu lợi suất tích lũy của danh mục và các benchmark
    results_df = pd.DataFrame({
        "time": cumulative_returns.index,
        "Danh mục đầu tư": cumulative_returns.values
    }).set_index("time")

    for benchmark, benchmark_cumulative in benchmark_data.items():
        results_df[benchmark] = benchmark_cumulative

    # Chuyển đổi dữ liệu sang dạng dài (long format) để vẽ biểu đồ
    results_df = results_df.reset_index().melt(id_vars=["time"], var_name="Danh mục", value_name="Lợi suất tích lũy")

    # Vẽ biểu đồ lợi suất tích lũy
    fig = px.line(
        results_df,
        x="time",
        y="Lợi suất tích lũy",
        color="Danh mục",
        title="Biểu đồ So sánh Lợi suất Tích lũy",
        labels={"time": "Thời gian", "Lợi suất tích lũy": "Lợi suất"},
        template="plotly_white"
    )
    fig.update_layout(
        xaxis_title="Thời gian",
        yaxis_title="Lợi suất tích lũy",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tính toán chỉ số hiệu suất
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

    return {
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Cumulative Returns": cumulative_returns,
        "Skipped Tickers": skipped_tickers,
    }


def plot_stock_chart_with_forecast(data, ticker, forecast_result=None, indicators=None):
    """
    Vẽ biểu đồ giá cổ phiếu với dự báo và chỉ báo kỹ thuật.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá lịch sử
        ticker (str): Mã cổ phiếu
        forecast_result (dict): Kết quả dự báo từ forecasting_models
        indicators (list): Danh sách chỉ báo kỹ thuật
    """
    if data.empty or ticker not in data.columns:
        st.warning("Không có dữ liệu để hiển thị biểu đồ.")
        return
    
    # Reset index
    df = data.reset_index()
    
    # Xác định số subplot
    num_subplots = 1
    has_rsi = indicators and 'RSI' in indicators
    has_macd = indicators and 'MACD' in indicators
    
    if has_rsi:
        num_subplots += 1
    if has_macd:
        num_subplots += 1
    
    # Tạo subplot
    row_heights = [0.6]
    if has_rsi:
        row_heights.append(0.2)
    if has_macd:
        row_heights.append(0.2)
    
    subplot_titles = ['Giá cổ phiếu với Dự báo']
    if has_rsi:
        subplot_titles.append('RSI')
    if has_macd:
        subplot_titles.append('MACD')
    
    fig = make_subplots(
        rows=num_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )
    
    # Tính chỉ báo nếu có
    if indicators:
        df_with_indicators = calculate_technical_indicators(df, ticker, indicators)
    else:
        df_with_indicators = df
    
    # Vẽ giá lịch sử
    fig.add_trace(
        go.Scatter(
            x=df_with_indicators['time'],
            y=df_with_indicators[ticker],
            name=f'{ticker} (Lịch sử)',
            mode='lines',
            line=dict(width=2, color='blue')
        ),
        row=1, col=1
    )
    
    # Vẽ dự báo nếu có
    if forecast_result:
        forecast_series = forecast_result['forecast']
        lower_bound = forecast_result['lower_bound']
        upper_bound = forecast_result['upper_bound']
        model_name = forecast_result.get('model_name', 'Dự báo')
        
        # Vẽ đường dự báo
        fig.add_trace(
            go.Scatter(
                x=forecast_series.index,
                y=forecast_series.values,
                name=f'{ticker} - {model_name}',
                mode='lines',
                line=dict(width=2, color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Vẽ khoảng tin cậy
        fig.add_trace(
            go.Scatter(
                x=forecast_series.index.tolist() + forecast_series.index.tolist()[::-1],
                y=upper_bound.values.tolist() + lower_bound.values.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='rgba(255, 0, 0, 0)'),
                name='Khoảng tin cậy 95%',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    # Vẽ các chỉ báo kỹ thuật
    if indicators:
        # SMA
        if 'SMA_20' in indicators and f'{ticker}_SMA_20' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_SMA_20'],
                    name=f'{ticker} SMA(20)',
                    mode='lines',
                    line=dict(dash='dash', width=1),
                    visible='legendonly'
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in indicators and f'{ticker}_SMA_50' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_SMA_50'],
                    name=f'{ticker} SMA(50)',
                    mode='lines',
                    line=dict(dash='dot', width=1),
                    visible='legendonly'
                ),
                row=1, col=1
            )
        
        # EMA
        if 'EMA_20' in indicators and f'{ticker}_EMA_20' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_EMA_20'],
                    name=f'{ticker} EMA(20)',
                    mode='lines',
                    line=dict(dash='dash', width=1),
                    visible='legendonly'
                ),
                row=1, col=1
            )
        
        if 'EMA_50' in indicators and f'{ticker}_EMA_50' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_EMA_50'],
                    name=f'{ticker} EMA(50)',
                    mode='lines',
                    line=dict(dash='dot', width=1),
                    visible='legendonly'
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'BB' in indicators:
            if f'{ticker}_BB_upper' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_indicators['time'],
                        y=df_with_indicators[f'{ticker}_BB_upper'],
                        name=f'{ticker} BB Upper',
                        mode='lines',
                        line=dict(dash='dot', width=1, color='gray'),
                        visible='legendonly'
                    ),
                    row=1, col=1
                )
            
            if f'{ticker}_BB_middle' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_indicators['time'],
                        y=df_with_indicators[f'{ticker}_BB_middle'],
                        name=f'{ticker} BB Middle',
                        mode='lines',
                        line=dict(dash='dash', width=1, color='gray'),
                        visible='legendonly'
                    ),
                    row=1, col=1
                )
            
            if f'{ticker}_BB_lower' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_indicators['time'],
                        y=df_with_indicators[f'{ticker}_BB_lower'],
                        name=f'{ticker} BB Lower',
                        mode='lines',
                        line=dict(dash='dot', width=1, color='gray'),
                        visible='legendonly'
                    ),
                    row=1, col=1
                )
        
        # RSI
        current_row = 2 if has_rsi else None
        if has_rsi and f'{ticker}_RSI' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_RSI'],
                    name=f'{ticker} RSI',
                    mode='lines',
                    line=dict(width=2)
                ),
                row=current_row, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                         annotation_text="Quá mua", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                         annotation_text="Quá bán", row=current_row, col=1)
        
        # MACD
        macd_row = num_subplots if has_macd else None
        if has_macd and f'{ticker}_MACD' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators['time'],
                    y=df_with_indicators[f'{ticker}_MACD'],
                    name=f'{ticker} MACD',
                    mode='lines',
                    line=dict(width=2, color='blue')
                ),
                row=macd_row, col=1
            )
            
            if f'{ticker}_MACD_signal' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_indicators['time'],
                        y=df_with_indicators[f'{ticker}_MACD_signal'],
                        name=f'{ticker} Signal',
                        mode='lines',
                        line=dict(width=2, color='orange')
                    ),
                    row=macd_row, col=1
                )
            
            if f'{ticker}_MACD_hist' in df_with_indicators.columns:
                fig.add_trace(
                    go.Bar(
                        x=df_with_indicators['time'],
                        y=df_with_indicators[f'{ticker}_MACD_hist'],
                        name=f'{ticker} Histogram',
                        marker_color='gray',
                        opacity=0.5
                    ),
                    row=macd_row, col=1
                )
    
    # Cập nhật layout
    title = f"Biểu đồ giá {ticker}"
    if forecast_result:
        title += f" với Dự báo ({forecast_result.get('model_name', '')})"
    
    fig.update_layout(
        title=title,
        xaxis_title="Thời gian",
        hovermode="x unified",
        height=600 if num_subplots == 1 else 800,
        showlegend=True,
        template="plotly_white"
    )
    
    # Cập nhật trục y
    fig.update_yaxes(title_text="Giá (VND)", row=1, col=1)
    if has_rsi:
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    if has_macd:
        fig.update_yaxes(title_text="MACD", row=macd_row, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị thông tin dự báo
    if forecast_result:
        with st.expander("ℹ️ Thông tin Dự báo", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                    st.markdown(f"**Mô hình:** {forecast_result.get('model_name', 'N/A')}")
                    st.markdown(f"**Số ngày dự báo:** {len(forecast_result['forecast'])}")
                    # Giá dự báo đầu và cuối
                    first_forecast = forecast_result['forecast'].iloc[0]
                    last_forecast = forecast_result['forecast'].iloc[-1]
                    # Hiển thị đúng đơn vị 1,000 VND
                    first_forecast_display = 0 if pd.isna(first_forecast) or first_forecast is None else first_forecast * 1000
                    last_forecast_display = 0 if pd.isna(last_forecast) or last_forecast is None else last_forecast * 1000
                    st.markdown(f"**Giá dự báo ngày đầu:** {first_forecast_display:,.0f} VND")
                    st.markdown(f"**Giá dự báo ngày cuối:** {last_forecast_display:,.0f} VND")
            
            with col2:
                    # Thay đổi dự kiến
                    current_price = data[ticker].iloc[-1]
                    current_price_display = 0 if pd.isna(current_price) or current_price is None else current_price * 1000
                    change = ((last_forecast_display - current_price_display) / current_price_display) * 100 if current_price_display != 0 else 0
                    st.markdown(f"**Giá hiện tại:** {current_price_display:,.0f} VND")
                    if change > 0:
                        st.markdown(f"**Thay đổi dự kiến:** <span style='color:green'>+{change:.2f}%</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Thay đổi dự kiến:** <span style='color:red'>{change:.2f}%</span>", unsafe_allow_html=True)
                    # Thông tin mô hình
                    if 'params' in forecast_result:
                        params = forecast_result['params']
                        if 'order' in params:
                            st.markdown(f"**Tham số ARIMA:** {params['order']}")
                        if 'aic' in params:
                            st.markdown(f"**AIC:** {params['aic']:.2f}")
