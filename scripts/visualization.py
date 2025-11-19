"""
Module visualization.py
Chứa các hàm vẽ biểu đồ: giá mã cổ phiếu, đường biên hiệu quả, backtesting.
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
    Tính toán các chỉ báo kỹ thuật cho một mã cổ phiếu.
    
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
    Vẽ biểu đồ giá mã cổ phiếu tương tác với các chỉ báo kỹ thuật.
    
    Args:
    data (pd.DataFrame): Dữ liệu giá mã cổ phiếu
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
    
    subplot_titles = ['Giá mã cổ phiếu']
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
    
    # Vẽ giá cho từng mã cổ phiếu
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
    title="Biểu đồ giá mã cổ phiếu với chỉ báo kỹ thuật",
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
    Vẽ biểu đồ giá mã cổ phiếu tương tác sử dụng Plotly Express.
    
    Args:
    data (pd.DataFrame): Dữ liệu giá mã cổ phiếu
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
    title="Biểu đồ giá mã cổ phiếu",
    labels={"time": "Thời gian", "Giá đóng cửa": "Giá mã cổ phiếu (VND)"},
        template="plotly_white"
    )

    # Tuỳ chỉnh giao diện
    fig.update_layout(
        xaxis_title="Thời gian",
    yaxis_title="Giá mã cổ phiếu (VND)",
    legend_title="Mã cổ phiếu",
        hovermode="x unified"
    )

    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_candlestick_chart(ohlc_data, ticker):
    """
    Vẽ biểu đồ nến (Candlestick Chart) cho một mã cổ phiếu với các chỉ báo kỹ thuật.
    
    Args:
        ohlc_data (pd.DataFrame): Dữ liệu OHLC với các cột time, open, high, low, close, volume
    ticker (str): Mã cổ phiếu
    """
    if ohlc_data.empty:
        st.warning("Không có dữ liệu OHLC để hiển thị biểu đồ nến.")
        return
    
    # Tạo copy để tính toán chỉ báo
    df = ohlc_data.copy()
    
    # Tính toán các chỉ báo kỹ thuật
    # MA - Moving Average (20, 50)
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # EMA - Exponential Moving Average (12, 26)
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Bollinger Bands (20, 2)
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # RSI - Relative Strength Index (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD - Moving Average Convergence Divergence
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Tạo biểu đồ với 4 subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=[
            f'Biểu đồ nến - {ticker}',
            'RSI (14)',
            'MACD',
            'Khối lượng giao dịch'
        ]
    )
    
    # Row 1: Biểu đồ nến với MA, EMA, Bollinger Bands
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=ticker,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Thêm MA20
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['MA20'],
            name='MA(20)',
            line=dict(color='#2962FF', width=1.5),
            visible='legendonly'
        ),
        row=1, col=1
    )
    
    # Thêm MA50
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['MA50'],
            name='MA(50)',
            line=dict(color='#FF6D00', width=1.5),
            visible='legendonly'
        ),
        row=1, col=1
    )
    
    # Thêm EMA12
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['EMA12'],
            name='EMA(12)',
            line=dict(color='#00897B', width=1.5, dash='dash'),
            visible='legendonly'
        ),
        row=1, col=1
    )
    
    # Thêm EMA26
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['EMA26'],
            name='EMA(26)',
            line=dict(color='#E91E63', width=1.5, dash='dash'),
            visible='legendonly'
        ),
        row=1, col=1
    )
    
    # Thêm Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['BB_upper'],
            name='BB Upper',
            line=dict(color='rgba(250, 128, 114, 0.5)', width=1),
            visible='legendonly'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['BB_middle'],
            name='BB Middle',
            line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
            fill=None,
            visible='legendonly'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['BB_lower'],
            name='BB Lower',
            line=dict(color='rgba(250, 128, 114, 0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(250, 128, 114, 0.1)',
            visible='legendonly'
        ),
        row=1, col=1
    )
    
    # Row 2: RSI
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['RSI'],
            name='RSI',
            line=dict(color='#9C27B0', width=2)
        ),
        row=2, col=1
    )
    
    # Thêm ngưỡng RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    # Row 3: MACD
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['MACD'],
            name='MACD',
            line=dict(color='#2962FF', width=2)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['MACD_signal'],
            name='Signal',
            line=dict(color='#FF6D00', width=2)
        ),
        row=3, col=1
    )
    
    # MACD Histogram
    colors_macd = ['#26a69a' if val >= 0 else '#ef5350' for val in df['MACD_hist']]
    fig.add_trace(
        go.Bar(
            x=df['time'],
            y=df['MACD_hist'],
            name='MACD Hist',
            marker_color=colors_macd,
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Row 4: Khối lượng
    if 'volume' in df.columns:
        colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] 
                  else '#ef5350' for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df['time'],
                y=df['volume'],
                name='Khối lượng',
                marker_color=colors,
                showlegend=False
            ),
            row=4, col=1
        )
    
    # Cập nhật layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=1000,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Thời gian", row=4, col=1)
    fig.update_yaxes(title_text="Giá (VND)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Khối lượng", row=4, col=1)
    
    # Hiển thị biểu đồ
    st.plotly_chart(fig, use_container_width=True)
    
    # Thêm hướng dẫn sử dụng
    with st.expander("ℹ️ Hướng dẫn đọc chỉ báo kỹ thuật"):
        st.markdown("""
        **Các chỉ báo được hiển thị:**
        
        1. **MA (Moving Average)** - Đường trung bình động:
           - MA(20): Xu hướng ngắn hạn (màu xanh dương)
           - MA(50): Xu hướng trung hạn (màu cam)
           - Khi giá > MA: Xu hướng tăng | Khi giá < MA: Xu hướng giảm
        
        2. **EMA (Exponential Moving Average)** - Đường trung bình động mũ:
           - EMA(12): Phản ứng nhanh với giá (màu xanh lá, nét đứt)
           - EMA(26): Phản ứng chậm hơn (màu hồng, nét đứt)
           - Nhạy cảm hơn MA với thay đổi giá gần đây
        
        3. **Bollinger Bands** - Dải Bollinger:
           - Dải trên & dải dưới: Đo biến động giá
           - Khi giá chạm dải trên: Có thể quá mua
           - Khi giá chạm dải dưới: Có thể quá bán
        
        4. **RSI (Relative Strength Index)** - Chỉ số sức mạnh tương đối:
           - RSI > 70: Vùng quá mua (có thể điều chỉnh)
           - RSI < 30: Vùng quá bán (có thể phục hồi)
           - RSI = 50: Trạng thái trung lập
        
        5. **MACD (Moving Average Convergence Divergence)**:
           - Đường MACD cắt lên Signal: Tín hiệu mua
           - Đường MACD cắt xuống Signal: Tín hiệu bán
           - Histogram dương/âm: Xu hướng tăng/giảm
        
        **Lưu ý:** Click vào tên chỉ báo trong legend để bật/tắt hiển thị.
        """)
    
    # Hiển thị giá trị chỉ báo hiện tại
    st.markdown("### Giá trị chỉ báo hiện tại")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    last_idx = df.index[-1]
    with col1:
        st.metric("MA(20)", f"{df['MA20'].iloc[-1]:,.0f}" if pd.notna(df['MA20'].iloc[-1]) else "N/A")
        st.metric("MA(50)", f"{df['MA50'].iloc[-1]:,.0f}" if pd.notna(df['MA50'].iloc[-1]) else "N/A")
    
    with col2:
        st.metric("EMA(12)", f"{df['EMA12'].iloc[-1]:,.0f}" if pd.notna(df['EMA12'].iloc[-1]) else "N/A")
        st.metric("EMA(26)", f"{df['EMA26'].iloc[-1]:,.0f}" if pd.notna(df['EMA26'].iloc[-1]) else "N/A")
    
    with col3:
        st.metric("BB Upper", f"{df['BB_upper'].iloc[-1]:,.0f}" if pd.notna(df['BB_upper'].iloc[-1]) else "N/A")
        st.metric("BB Lower", f"{df['BB_lower'].iloc[-1]:,.0f}" if pd.notna(df['BB_lower'].iloc[-1]) else "N/A")
    
    with col4:
        rsi_val = df['RSI'].iloc[-1]
        if pd.notna(rsi_val):
            if rsi_val > 70:
                rsi_trend = "Quá mua"
                rsi_color = "red"
            elif rsi_val < 30:
                rsi_trend = "Quá bán"
                rsi_color = "green"
            else:
                rsi_trend = "Trung lập"
                rsi_color = "gray"
            st.markdown(f"<span style='font-size:20px'><b>RSI(14):</b> <span style='color:{rsi_color}'>{rsi_val:.2f} ({rsi_trend})</span></span>", unsafe_allow_html=True)
        else:
            st.metric("RSI(14)", "N/A")
    
    with col5:
        macd_val = df['MACD'].iloc[-1]
        signal_val = df['MACD_signal'].iloc[-1]
        if pd.notna(macd_val) and pd.notna(signal_val):
            if macd_val > signal_val:
                macd_trend = "Tăng"
                macd_color = "green"
            elif macd_val < signal_val:
                macd_trend = "Giảm"
                macd_color = "red"
            else:
                macd_trend = "Trung lập"
                macd_color = "gray"
            st.markdown(f"<span style='font-size:20px'><b>MACD:</b> <span style='color:{macd_color}'>{macd_val:.2f} ({macd_trend})</span></span>", unsafe_allow_html=True)
            st.metric("Signal", f"{signal_val:.2f}")
        else:
            st.metric("MACD", "N/A")
            st.metric("Signal", "N/A")


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
    
    # Hiển thị tooltip giải thích bên dưới biểu đồ (có thể ẩn/hiện)
    with st.expander("Giải thích Đường Biên Hiệu Quả Markowitz"):
        st.markdown("""
        **Khái niệm:**  
        Tập hợp các danh mục đầu tư tối ưu mang lại lợi nhuận kỳ vọng cao nhất với mức rủi ro cho trước, 
        hoặc rủi ro thấp nhất với mức lợi nhuận cho trước.
        
        **Ý nghĩa:**
        - **Các điểm trên đường biên:** Danh mục hiệu quả
        - **Các điểm bên trong:** Danh mục kém hiệu quả  
        - **Điểm đỏ:** Danh mục có Sharpe Ratio tối đa
        
        **Nguyên lý:**  
        Đa dạng hóa đầu tư giúp giảm rủi ro mà vẫn duy trì hoặc tăng lợi nhuận kỳ vọng.
        """)


def plot_max_sharpe_with_cal(ret_arr, vol_arr, sharpe_arr, all_weights, tickers, optimal_return, optimal_volatility, risk_free_rate=0.04):
    """
    Vẽ biểu đồ Max Sharpe Ratio với đường CAL (Capital Allocation Line).
    
    Args:
        ret_arr (np.array): Mảng lợi nhuận kỳ vọng của các danh mục
        vol_arr (np.array): Mảng độ lệch chuẩn (rủi ro) của các danh mục
        sharpe_arr (np.array): Mảng tỷ lệ Sharpe của các danh mục
        all_weights (np.array): Ma trận trọng số tất cả các danh mục
    tickers (list): Danh sách mã cổ phiếu
        optimal_return (float): Lợi nhuận của danh mục Max Sharpe
        optimal_volatility (float): Rủi ro của danh mục Max Sharpe
        risk_free_rate (float): Lãi suất phi rủi ro (mặc định 4%/năm)
    """
    # Chuẩn bị thông tin hover
    hover_texts = [
        ", ".join([f"{tickers[j]}: {weight * 100:.2f}%" for j, weight in enumerate(weights)])
        for weights in all_weights
    ]

    # Tạo biểu đồ scatter
    fig = px.scatter(
        x=vol_arr,
        y=ret_arr,
        color=sharpe_arr,
        hover_data={
            'Tỷ lệ Sharpe': sharpe_arr,
            'Thông tin danh mục': hover_texts
        },
        labels={'x': 'Rủi ro (Độ lệch chuẩn)', 'y': 'Lợi nhuận kỳ vọng', 'color': 'Tỷ lệ Sharpe'},
        title='Tối đa hóa Tỷ lệ Sharpe (Max Sharpe Ratio) với Đường CAL'
    )

    # Đánh dấu danh mục Max Sharpe (màu đỏ)
    optimal_weights = all_weights[np.argmax(sharpe_arr)]
    fig.add_scatter(
        x=[optimal_volatility],
        y=[optimal_return],
        mode='markers',
        marker=dict(color='red', size=12, symbol='star'),
        name='Danh mục Max Sharpe',
        hovertext=[f"<b>Max Sharpe Portfolio</b><br>" + 
                   ", ".join([f"{tickers[j]}: {optimal_weights[j] * 100:.2f}%" for j in range(len(tickers))]) +
                   f"<br>Return: {optimal_return:.2%}<br>Risk: {optimal_volatility:.2%}<br>Sharpe: {(optimal_return - risk_free_rate) / optimal_volatility:.2f}"]
    )

    # Vẽ đường CAL (Capital Allocation Line)
    # CAL là đường thẳng đi qua điểm risk-free rate (0, rf) và điểm Max Sharpe
    # Phương trình: y = rf + (optimal_return - rf) / optimal_volatility * x
    max_vol_for_cal = max(vol_arr) * 1.2  # Kéo dài đường CAL
    cal_x = np.array([0, max_vol_for_cal])
    cal_y = risk_free_rate + (optimal_return - risk_free_rate) / optimal_volatility * cal_x
    
    fig.add_scatter(
        x=cal_x,
        y=cal_y,
        mode='lines',
        line=dict(color='green', width=3, dash='dash'),
        name='Đường CAL',
        hovertext=[
            f"<b>Capital Allocation Line (CAL)</b><br>"
            f"Đường phân bổ vốn tối ưu<br><br>"
            f"<b>Ý nghĩa:</b><br>"
            f"• CAL là đường thẳng kẻ từ lãi suất phi rủi ro ({risk_free_rate:.1%})<br>"
            f"• Tiếp tuyến với đường biên hiệu quả tại điểm Max Sharpe<br>"
            f"• Cho thấy tỷ lệ lợi nhuận/rủi ro tốt nhất có thể đạt được<br>"
            f"• Nhà đầu tư có thể kết hợp tài sản phi rủi ro và<br>"
            f"  danh mục Max Sharpe để đạt điểm bất kỳ trên đường này<br>"
            f"• Độ dốc của CAL chính là tỷ lệ Sharpe tối đa: {(optimal_return - risk_free_rate) / optimal_volatility:.2f}"
        ] * len(cal_x)
    )
    
    # Đánh dấu điểm risk-free rate trên trục y
    fig.add_scatter(
        x=[0],
        y=[risk_free_rate],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='diamond'),
        name='Lãi suất phi rủi ro',
        hovertext=[f"<b>Risk-Free Rate</b><br>Lợi nhuận: {risk_free_rate:.2%}<br>Rủi ro: 0%"]
    )

    # Cập nhật layout - Di chuyển legend sang bên trái
    fig.update_layout(
        xaxis_title='Rủi ro (Độ lệch chuẩn)',
        yaxis_title='Lợi nhuận kỳ vọng',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.01,  # Vị trí bên trái
            y=0.99,  # Vị trí trên cùng
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Thêm chú thích về ý nghĩa của CAL với khả năng đóng/mở
    with st.expander("Giải thích về Đường CAL (Capital Allocation Line)"):
        st.markdown("""
        **Đường CAL** là đường phân bổ vốn tối ưu, thể hiện các điểm đầu tư tối ưu khi kết hợp:
        - **Tài sản phi rủi ro** (ví dụ: trái phiếu chính phủ, tiền gửi ngân hàng) với lãi suất {:.1%}/năm
        - **Danh mục rủi ro tối ưu** (danh mục Max Sharpe - điểm màu đỏ)
        
        **Ý nghĩa thực tế:**
        - Điểm trên CAL gần điểm phi rủi ro → Đầu tư bảo thủ (nhiều tài sản phi rủi ro)
        - Điểm Max Sharpe → Đầu tư 100% vào danh mục mã cổ phiếu tối ưu
        - Điểm xa hơn Max Sharpe trên CAL → Đầu tư margin (vay nợ để đầu tư nhiều hơn)
        
        **Độ dốc của CAL** chính là **Tỷ lệ Sharpe**, cho biết phần bù rủi ro (excess return) trên mỗi đơn vị rủi ro.
        """.format(risk_free_rate))


def plot_min_volatility_scatter(ret_arr, vol_arr, sharpe_arr, all_weights, tickers, 
                                  min_vol_return, min_vol_volatility, 
                                  max_sharpe_return, max_sharpe_volatility,
                                  min_vol_weights_dict, max_sharpe_weights_dict,
                                  risk_free_rate=0.02):
    """
    Vẽ biểu đồ Min Volatility với scatter plot 10,000 danh mục.
    Đánh dấu cả danh mục Max Sharpe và Min Volatility để so sánh.
    
    Args:
        ret_arr (np.array): Mảng lợi nhuận kỳ vọng của các danh mục
        vol_arr (np.array): Mảng độ lệch chuẩn (rủi ro) của các danh mục
        sharpe_arr (np.array): Mảng tỷ lệ Sharpe của các danh mục
        all_weights (np.array): Ma trận trọng số tất cả các danh mục
    tickers (list): Danh sách mã cổ phiếu
        min_vol_return (float): Lợi nhuận của danh mục Min Volatility
        min_vol_volatility (float): Rủi ro của danh mục Min Volatility
        max_sharpe_return (float): Lợi nhuận của danh mục Max Sharpe
        max_sharpe_volatility (float): Rủi ro của danh mục Max Sharpe
        min_vol_weights_dict (dict): Trọng số thực sự của danh mục Min Volatility
        max_sharpe_weights_dict (dict): Trọng số thực sự của danh mục Max Sharpe
        risk_free_rate (float): Lãi suất phi rủi ro (mặc định 2%/năm)
    """
    # Chuẩn bị thông tin hover
    hover_texts = [
        ", ".join([f"{tickers[j]}: {weight * 100:.2f}%" for j, weight in enumerate(weights)])
        for weights in all_weights
    ]

    # Tạo biểu đồ scatter
    fig = px.scatter(
        x=vol_arr,
        y=ret_arr,
        color=sharpe_arr,
        hover_data={
            'Tỷ lệ Sharpe': sharpe_arr,
            'Thông tin danh mục': hover_texts
        },
        labels={'x': 'Rủi ro (Độ lệch chuẩn)', 'y': 'Lợi nhuận kỳ vọng', 'color': 'Tỷ lệ Sharpe'},
        title='Tối thiểu hóa Rủi ro (Min Volatility)'
    )

    # Đánh dấu danh mục Min Volatility (màu xanh lá cây)
    # Sử dụng trọng số thực sự từ kết quả tối ưu hóa
    fig.add_scatter(
        x=[min_vol_volatility],
        y=[min_vol_return],
        mode='markers',
        marker=dict(color='green', size=15, symbol='star', line=dict(color='darkgreen', width=2)),
        name='Danh mục Rủi ro Tối thiểu',
        hovertext=[f"<b>Min Volatility Portfolio</b><br>" + 
                   ", ".join([f"{ticker}: {min_vol_weights_dict.get(ticker, 0) * 100:.2f}%" for ticker in tickers]) +
                   f"<br>Return: {min_vol_return:.2%}<br>Risk: {min_vol_volatility:.2%}<br>Sharpe: {(min_vol_return - risk_free_rate) / min_vol_volatility:.2f}"]
    )

    # Đánh dấu danh mục Max Sharpe để so sánh (màu đỏ)
    # Sử dụng trọng số thực sự từ kết quả tối ưu hóa
    fig.add_scatter(
        x=[max_sharpe_volatility],
        y=[max_sharpe_return],
        mode='markers',
        marker=dict(color='red', size=15, symbol='star', line=dict(color='darkred', width=2)),
        name='Danh mục Sharpe Tối đa',
        hovertext=[f"<b>Max Sharpe Portfolio</b><br>" + 
                   ", ".join([f"{ticker}: {max_sharpe_weights_dict.get(ticker, 0) * 100:.2f}%" for ticker in tickers]) +
                   f"<br>Return: {max_sharpe_return:.2%}<br>Risk: {max_sharpe_volatility:.2%}<br>Sharpe: {(max_sharpe_return - risk_free_rate) / max_sharpe_volatility:.2f}"]
    )

    # Cập nhật layout - Di chuyển legend sang khoảng trống
    fig.update_layout(
        xaxis_title='Rủi ro (Độ lệch chuẩn)',
        yaxis_title='Lợi nhuận kỳ vọng',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.65,  # Vị trí bên phải
            y=0.15,  # Vị trí dưới cùng (khoảng trống)
            xanchor='left',
            yanchor='bottom',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Thêm chú thích về ý nghĩa của Min Volatility
    with st.expander("Giải thích về Mô hình Min Volatility"):
        st.markdown(f"""
        **Mô hình Min Volatility** tập trung vào việc giảm thiểu rủi ro (độ lệch chuẩn) của danh mục đầu tư.
        
        **So sánh hai chiến lược:**
        - **Danh mục Rủi ro Tối thiểu** (ngôi sao xanh lá): 
          - Rủi ro: {min_vol_volatility:.2%}
          - Lợi nhuận: {min_vol_return:.2%}
          - Tỷ lệ Sharpe: {(min_vol_return - risk_free_rate) / min_vol_volatility:.2f}
          - Phù hợp cho nhà đầu tư ưa thích an toàn, chấp nhận lợi nhuận thấp hơn
        
        - **Danh mục Sharpe Tối đa** (ngôi sao đỏ):
          - Rủi ro: {max_sharpe_volatility:.2%}
          - Lợi nhuận: {max_sharpe_return:.2%}
          - Tỷ lệ Sharpe: {(max_sharpe_return - risk_free_rate) / max_sharpe_volatility:.2f}
          - Cân bằng tốt nhất giữa lợi nhuận và rủi ro
        
        **Khi nào nên chọn Min Volatility?**
        - Thị trường biến động mạnh, không ổn định
        - Nhà đầu tư có độ chấp nhận rủi ro thấp
        - Mục tiêu bảo toàn vốn quan trọng hơn tăng trưởng
        - Gần thời điểm cần rút vốn (ví dụ: nghỉ hưu)
        """)


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

    # Giá mã cổ phiếu và phân bổ mã cổ phiếu
    latest_prices = result.get("Giá mã cổ phiếu", {})
    allocation = result.get("Số mã cổ phiếu cần mua", {})

    # Nếu không có phân bổ, mặc định là 0
    allocation = {ticker: allocation.get(ticker, 0) for ticker in tickers}
    latest_prices = {ticker: latest_prices.get(ticker, 0) for ticker in tickers}

    # Tạo DataFrame kết hợp các thông tin
    combined_data = {
        "Mã cổ phiếu": tickers,
        "Giá mã cổ phiếu": [f"{latest_prices.get(ticker, 0):.2f}" for ticker in tickers],
        "Trọng số (%)": [f"{weights_df.loc[ticker, 'Trọng số (%)']:.2f}" for ticker in tickers],
        "Số mã cổ phiếu cần mua": [allocation.get(ticker, 0) for ticker in tickers]
    }
    
    # Chuyển đổi thành DataFrame và hiển thị
    combined_df = pd.DataFrame(combined_data)

    # Hiển thị bảng kết hợp
    st.markdown("### Bảng phân bổ danh mục đầu tư:")
    st.table(combined_df)
    st.write(f"- **Số tiền còn lại:** {round(result.get('Số tiền còn lại', 0))}")


def backtest_portfolio(symbols, weights, start_date, end_date, fetch_stock_data_func, 
                       benchmark_symbols=["VNINDEX", "VN30", "HNX30", "HNXINDEX"],
                       get_market_indices_func=None):
    """
    Hàm backtesting danh mục đầu tư.
    Hỗ trợ lấy benchmark data từ market_summary table thông qua get_market_indices_func.

    Args:
        symbols (list): Danh sách mã cổ phiếu trong danh mục
        weights (list): Trọng số của mỗi mã cổ phiếu
        start_date (str): Ngày bắt đầu (định dạng 'YYYY-MM-DD')
        end_date (str): Ngày kết thúc (định dạng 'YYYY-MM-DD')
        fetch_stock_data_func (function): Hàm lấy dữ liệu giá mã cổ phiếu
        benchmark_symbols (list): Danh sách các chỉ số benchmark
        get_market_indices_func (function): Hàm lấy dữ liệu market indices từ database

    Returns:
        dict: Kết quả backtesting bao gồm Sharpe Ratio, Maximum Drawdown, và lợi suất tích lũy
    """
    # Lấy dữ liệu giá mã cổ phiếu trong danh mục
    stock_data, skipped_tickers = fetch_stock_data_func(symbols, start_date, end_date)
    if skipped_tickers:
        st.warning(f"Các mã không tải được dữ liệu: {', '.join(skipped_tickers)}")

    if stock_data.empty:
        st.error("Không có dữ liệu để backtesting.")
        return

    # Đảm bảo thứ tự trọng số khớp với thứ tự cột trong stock_data
    # Tạo Series từ symbols và weights để map chính xác
    weights_dict = dict(zip(symbols, weights))
    # Sắp xếp lại weights theo thứ tự cột trong stock_data
    aligned_weights = [weights_dict.get(col, 0) for col in stock_data.columns]
    
    # Tính lợi suất hàng ngày của danh mục
    returns = stock_data.pct_change().dropna()
    portfolio_returns = returns.dot(aligned_weights)  # Lợi suất danh mục đầu tư
    cumulative_returns = (1 + portfolio_returns).cumprod()  # Lợi suất tích lũy

    # Lấy dữ liệu benchmark
    benchmark_data = {}
    
    if get_market_indices_func is not None:
        # Sử dụng hàm lấy market indices từ database
        try:
            market_indices_df = get_market_indices_func(start_date, end_date)
            
            if not market_indices_df.empty:
                # Set date làm index
                market_indices_df = market_indices_df.set_index('date')
                
                # Map benchmark symbols to database columns
                benchmark_column_map = {
                    "VNINDEX": "vnindex",
                    "VN30": "vn30",
                    "HNX30": "hnx30",
                    "HNXINDEX": "hnx_index",
                }
                
                for benchmark in benchmark_symbols:
                    db_column = benchmark_column_map.get(benchmark)
                    if db_column and db_column in market_indices_df.columns:
                        # Chuyển đổi giá trị thành numeric và loại bỏ NaN
                        benchmark_series = pd.to_numeric(market_indices_df[db_column], errors='coerce')
                        benchmark_series = benchmark_series.dropna()
                        
                        if len(benchmark_series) > 0:
                            # Tính lợi suất
                            benchmark_returns = benchmark_series.pct_change().dropna()
                            if len(benchmark_returns) > 0:
                                benchmark_cumulative = (1 + benchmark_returns).cumprod()
                                benchmark_data[benchmark] = benchmark_cumulative
                            else:
                                st.warning(f"Không đủ dữ liệu benchmark để tính lợi suất cho {benchmark}.")
                        else:
                            st.warning(f"Không có dữ liệu benchmark hợp lệ cho {benchmark}.")
                    else:
                        st.warning(f"Không tìm thấy cột dữ liệu cho benchmark {benchmark} trong database.")
            else:
                st.warning("Không có dữ liệu market indices từ database.")
        except Exception as e:
            st.error(f"Lỗi khi lấy dữ liệu market indices: {e}")
    else:
        # Fallback: Sử dụng fetch_stock_data_func (phương pháp cũ)
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
    
    # Tính toán các chỉ số bổ sung
    # 1. Total Return (Lợi nhuận tổng)
    total_return = (cumulative_returns.iloc[-1] - 1) * 100  # Phần trăm
    
    # 2. Annualized Return (Lợi nhuận hàng năm)
    num_days = len(portfolio_returns)
    num_years = num_days / 252
    annualized_return = ((cumulative_returns.iloc[-1]) ** (1 / num_years) - 1) * 100 if num_years > 0 else 0
    
    # 3. Volatility (Độ biến động hàng năm)
    volatility = portfolio_returns.std() * np.sqrt(252) * 100  # Phần trăm
    
    # 4. Sortino Ratio (chỉ xét độ lệch chuẩn của lợi nhuận âm)
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    sortino_ratio = (portfolio_returns.mean() * 252) / downside_std if downside_std > 0 else 0
    
    # 5. Alpha và Beta (so với benchmark đầu tiên nếu có)
    alpha = 0
    beta = 0
    if benchmark_data:
        first_benchmark = list(benchmark_data.keys())[0]
        benchmark_cumulative = benchmark_data[first_benchmark]
        benchmark_returns = benchmark_cumulative.pct_change().dropna()
        
        # Đảm bảo cùng index
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) > 1:
            portfolio_aligned = portfolio_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]
            
            # Chuyển Series thành numpy arrays
            portfolio_array = np.array(portfolio_aligned)
            benchmark_array = np.array(benchmark_aligned)
            
            # Validate có đủ dữ liệu không
            if len(portfolio_array) > 1 and len(benchmark_array) > 1:
                # Tính covariance matrix
                cov_matrix = np.cov(portfolio_array, benchmark_array)
                covariance = cov_matrix[0, 1]
                benchmark_variance = np.var(benchmark_array)
                
                # Tính beta
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Tính alpha hàng năm
                portfolio_annual_return = portfolio_aligned.mean() * 252
                benchmark_annual_return = benchmark_aligned.mean() * 252
                alpha = (portfolio_annual_return - benchmark_annual_return) * 100  # Phần trăm
            else:
                st.warning(f"Không đủ dữ liệu để tính Alpha/Beta với benchmark {first_benchmark}")
        else:
            st.warning(f"Không có dữ liệu chung với benchmark {first_benchmark}")

    # Tạo bảng thống kê tổng hợp 
    st.markdown("### Bảng Thống kê Danh mục")
    
    metrics_data = {
        "Chỉ số": [
            "Lợi nhuận tổng (Total Return)",
            "Lợi nhuận hàng năm (Annualized Return)",
            "Độ biến động (Volatility)",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Maximum Drawdown"
        ],
        "Giá trị": [
            f"{total_return:.2f}%",
            f"{annualized_return:.2f}%",
            f"{volatility:.2f}%",
            f"{sharpe_ratio:.4f}",
            f"{sortino_ratio:.4f}",
            f"{max_drawdown * 100:.2f}%"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)

    return {
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": max_drawdown,
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Alpha": alpha,
        "Beta": beta,
        "Cumulative Returns": cumulative_returns,
        "Skipped Tickers": skipped_tickers,
    }


def plot_min_cvar_analysis(result):
    """
    Trực quan hóa kết quả mô hình Min CVaR.
    
    Phân tích cấu trúc logic:
    - Mô hình Min CVaR tối thiểu hóa Conditional Value at Risk (CVaR)
    - CVaR là giá trị kỳ vọng của tổn thất vượt quá VaR (Value at Risk)
    - Mô hình này tập trung vào việc giảm thiểu rủi ro đuôi (tail risk)
    - Phù hợp cho nhà đầu tư ngại rủi ro, quan tâm đến tổn thất cực đoan
    
    Trực quan hóa bao gồm:
    1. Biểu đồ cột phân bổ: Thể hiện trọng số của từng tài sản trong danh mục
    2. Thông tin mô hình: Các chỉ số quan trọng (Lợi nhuận, CVaR, Sharpe)
    3. Giải thích chi tiết về mô hình
    
    Args:
        result (dict): Kết quả từ mô hình min_cvar
    """
    if result is None:
        st.error("Không có kết quả để hiển thị")
        return
    
    # Lấy dữ liệu trọng số
    weights = result.get("Trọng số danh mục", {})
    
    if not weights:
        st.warning("Không có dữ liệu trọng số để hiển thị")
        return
    
    # Chuyển đổi trọng số thành DataFrame
    weights_df = pd.DataFrame({
    'Mã cổ phiếu': list(weights.keys()),
        'Trọng số': [w * 100 for w in weights.values()]  # Chuyển sang phần trăm
    })
    
    # Sắp xếp theo trọng số giảm dần
    weights_df = weights_df.sort_values('Trọng số', ascending=False).reset_index(drop=True)
    
    # Biểu đồ cột phân bổ đơn giản
    st.subheader("Biểu Đồ Phân Bổ Trọng Số")
    
    # Tạo biểu đồ cột với Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
    x=weights_df['Mã cổ phiếu'],
        y=weights_df['Trọng số'],
        text=[f"{w:.2f}%" for w in weights_df['Trọng số']],
        textposition='outside',
        marker=dict(
            color=weights_df['Trọng số'],
            colorscale='Blues',
            showscale=False
        ),
        hovertemplate='<b>%{x}</b><br>Trọng số: %{y:.2f}%<extra></extra>'
    ))
    
    # Cập nhật layout
    fig.update_layout(
        title={
            'text': 'Phân Bổ Trọng Số Danh Mục Min CVaR',
            'x': 0.5,
            'xanchor': 'center'
        },
    xaxis_title='Mã cổ phiếu',
        yaxis_title='Trọng số (%)',
        showlegend=False,
        height=500,
        hovermode='x',
        plot_bgcolor='white',
        yaxis=dict(
            gridcolor='lightgray',
            range=[0, max(weights_df['Trọng số']) * 1.15]  # Thêm khoảng trống cho text
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Thông tin bổ sung về mô hình
    st.subheader("Thông Tin Mô Hình Min CVaR")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Lợi nhuận kỳ vọng",
            value=f"{result.get('Lợi nhuận kỳ vọng', 0) * 100:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Rủi ro CVaR",
            value=f"{result.get('Rủi ro CVaR', 0) * 100:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Tỷ lệ Sharpe",
            value=f"{result.get('Tỷ lệ Sharpe', 0):.4f}"
        )
    
    # Giải thích mô hình
    with st.expander("Giải thích về mô hình Min CVaR"):
        st.markdown("""
        ### Mô hình Min CVaR (Minimum Conditional Value at Risk)
        
        **Cấu trúc logic:**
        
        1. **Mục tiêu:** Tối thiểu hóa CVaR - rủi ro tổn thất trong trường hợp xấu nhất
        
        2. **CVaR là gì?**
           - CVaR (hay ES - Expected Shortfall) đo lường giá trị kỳ vọng của tổn thất 
             trong các trường hợp tồi tệ nhất (vượt quá ngưỡng VaR)
           - Ví dụ: CVaR 95% là tổn thất trung bình trong 5% trường hợp tệ nhất
        
        3. **Ưu điểm:**
           - Phản ánh rủi ro đuôi (tail risk) tốt hơn độ lệch chuẩn
           - Là thước đo rủi ro nhất quán (coherent risk measure)
           - Phù hợp cho nhà đầu tư ngại rủi ro cực đoan
        
        4. **Nhược điểm:**
           - Có thể bỏ qua cơ hội sinh lời cao
           - Yêu cầu nhiều dữ liệu lịch sử để ước lượng chính xác
           - Có thể quá bảo thủ trong một số trường hợp
        
        5. **Khi nào nên sử dụng:**
           - Khi bạn lo ngại về tổn thất lớn trong điều kiện thị trường xấu
           - Khi muốn bảo vệ danh mục khỏi sự kiện đuôi (tail events)
           - Phù hợp với nhà đầu tư có khẩu vị rủi ro thấp
        """)


def plot_min_cdar_analysis(min_cdar_result, max_sharpe_result, returns_data):
    """
    Trực quan hóa mô hình Min CDaR với biểu đồ so sánh mức sụt giảm.
    
    Args:
        min_cdar_result (dict): Kết quả từ mô hình min_cdar
        max_sharpe_result (dict): Kết quả từ mô hình max_sharpe để so sánh
    returns_data (pd.DataFrame): Dữ liệu lợi suất lịch sử của các mã cổ phiếu
    """
    if min_cdar_result is None:
        st.error("Không có kết quả Min CDaR để hiển thị")
        return
    
    # Lấy trọng số từ cả hai mô hình
    min_cdar_weights = min_cdar_result.get("Trọng số danh mục", {})
    max_sharpe_weights = max_sharpe_result.get("Trọng số danh mục", {}) if max_sharpe_result else {}
    
    if not min_cdar_weights:
        st.warning("Không có dữ liệu trọng số Min CDaR")
        return
    
    # Hiển thị thông tin mô hình
    st.subheader("Thông Tin Mô Hình Min CDaR")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Lợi nhuận kỳ vọng",
            value=f"{min_cdar_result.get('Lợi nhuận kỳ vọng', 0) * 100:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Rủi ro CDaR",
            value=f"{min_cdar_result.get('Rủi ro CDaR', 0) * 100:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Tỷ lệ Sharpe",
            value=f"{min_cdar_result.get('Tỷ lệ Sharpe', 0):.4f}"
        )
    
    # Tính toán giá trị danh mục và drawdown
    if returns_data is not None and not returns_data.empty:
        # Đảm bảo returns_data là lợi suất (returns), không phải giá
        if returns_data.max().max() > 10:  # Nếu giá trị lớn, có thể là giá chứ không phải lợi suất
            returns = returns_data.pct_change().dropna()
        else:
            returns = returns_data
        
        # Tính giá trị danh mục tích lũy cho Min CDaR
        min_cdar_weights_array = np.array([min_cdar_weights.get(ticker, 0) for ticker in returns.columns])
        min_cdar_portfolio_returns = returns.dot(min_cdar_weights_array)
        min_cdar_cumulative = (1 + min_cdar_portfolio_returns).cumprod()
        
        # Tính drawdown cho Min CDaR
        min_cdar_running_max = min_cdar_cumulative.cummax()
        min_cdar_drawdown = (min_cdar_cumulative - min_cdar_running_max) / min_cdar_running_max
        
        # Tính cho Max Sharpe nếu có
        if max_sharpe_weights:
            max_sharpe_weights_array = np.array([max_sharpe_weights.get(ticker, 0) for ticker in returns.columns])
            max_sharpe_portfolio_returns = returns.dot(max_sharpe_weights_array)
            max_sharpe_cumulative = (1 + max_sharpe_portfolio_returns).cumprod()
            
            # Tính drawdown cho Max Sharpe
            max_sharpe_running_max = max_sharpe_cumulative.cummax()
            max_sharpe_drawdown = (max_sharpe_cumulative - max_sharpe_running_max) / max_sharpe_running_max
        else:
            max_sharpe_cumulative = None
            max_sharpe_drawdown = None
        
        # Tạo biểu đồ so sánh
        st.subheader("Biểu đồ So sánh Mức Sụt Giảm")
        
        # Tạo subplot với 2 hàng
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Giá trị Danh mục Tích lũy', 'Mức Sụt Giảm (Drawdown)')
        )
        
        # Hàng 1: Giá trị danh mục tích lũy
        fig.add_trace(
            go.Scatter(
                x=min_cdar_cumulative.index,
                y=min_cdar_cumulative.values,
                name='Min CDaR',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Min CDaR</b><br>Ngày: %{x}<br>Giá trị: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        if max_sharpe_cumulative is not None:
            fig.add_trace(
                go.Scatter(
                    x=max_sharpe_cumulative.index,
                    y=max_sharpe_cumulative.values,
                    name='Max Sharpe',
                    line=dict(color='red', width=2),
                    hovertemplate='<b>Max Sharpe</b><br>Ngày: %{x}<br>Giá trị: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Hàng 2: Drawdown
        fig.add_trace(
            go.Scatter(
                x=min_cdar_drawdown.index,
                y=min_cdar_drawdown.values * 100,
                name='Min CDaR Drawdown',
                fill='tozeroy',
                line=dict(color='blue', width=1),
                fillcolor='rgba(0, 0, 255, 0.2)',
                hovertemplate='<b>Min CDaR Drawdown</b><br>Ngày: %{x}<br>Sụt giảm: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        if max_sharpe_drawdown is not None:
            fig.add_trace(
                go.Scatter(
                    x=max_sharpe_drawdown.index,
                    y=max_sharpe_drawdown.values * 100,
                    name='Max Sharpe Drawdown',
                    fill='tozeroy',
                    line=dict(color='red', width=1),
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    hovertemplate='<b>Max Sharpe Drawdown</b><br>Ngày: %{x}<br>Sụt giảm: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Cập nhật layout
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Thời gian", row=2, col=1)
        fig.update_yaxes(title_text="Giá trị tích lũy", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        # Thêm annotation giải thích
        fig.add_annotation(
            text="Biểu đồ này cho thấy danh mục Min CDaR có mức sụt giảm nông hơn và phục hồi nhanh hơn,<br>" +
                 "mặc dù danh mục Max Sharpe có thể tăng trưởng nhanh hơn trong điều kiện thuận lợi.",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=11, color="gray"),
            align="center"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tính toán và hiển thị các chỉ số so sánh
        st.subheader("So sánh Chỉ số Drawdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Min CDaR (Kiểm soát tổn thất kéo dài)**")
            st.metric("Max Drawdown", f"{min_cdar_drawdown.min() * 100:.2f}%")
            st.metric("Số ngày Drawdown trung bình", f"{(min_cdar_drawdown < 0).sum()}")
            
        with col2:
            if max_sharpe_drawdown is not None:
                st.write("**Max Sharpe (Hiệu suất tối đa)**")
                st.metric("Max Drawdown", f"{max_sharpe_drawdown.min() * 100:.2f}%")
                st.metric("Số ngày Drawdown trung bình", f"{(max_sharpe_drawdown < 0).sum()}")
    
    # Giải thích mô hình
    with st.expander("Giải thích về mô hình Min CDaR"):
        st.markdown("""
        ### Mô hình Min CDaR (Minimum Conditional Drawdown at Risk)
        
        **Cấu trúc logic:**
        
        1. **Mục tiêu:** Tối thiểu hóa CDaR - kiểm soát tổn thất kéo dài
        
        2. **CDaR là gì?**
           - CDaR đo lường mức giảm giá trị trung bình trong các giai đoạn có sự sụt giảm sâu nhất
           - Drawdown là mức giảm giá trị từ đỉnh cao nhất trước đó
           - CDaR tập trung vào các drawdown tồi tệ nhất (ví dụ: 5% trường hợp xấu nhất)
        
        3. **Ưu điểm:**
           - Bảo vệ tốt trong các giai đoạn thị trường giảm kéo dài
           - Giúp danh mục phục hồi nhanh hơn sau thua lỗ
           - Phù hợp cho nhà đầu tư lo ngại về tổn thất liên tục
        
        4. **Nhược điểm:**
           - Có thể giới hạn tiềm năng tăng trưởng
           - Lợi nhuận có thể thấp hơn trong thị trường tăng mạnh
           - Yêu cầu dữ liệu lịch sử dài để ước lượng chính xác
        
        5. **Khi nào nên sử dụng:**
           - Khi lo ngại về tổn thất kéo dài trong thị trường biến động
           - Muốn giảm thiểu thời gian phục hồi sau thua lỗ
           - Ưu tiên bảo toàn vốn hơn là tối đa hóa lợi nhuận
        """)


def plot_hrp_dendrogram(data, weights):
    """
    Vẽ biểu đồ Dendrogram cho mô hình HRP (Hierarchical Risk Parity).
    
    Args:
    data (pd.DataFrame): Dữ liệu giá mã cổ phiếu
        weights (dict): Trọng số danh mục từ mô hình HRP
    """
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    
    st.subheader("Biểu đồ Phân Cấp Tài Sản (Dendrogram)")
    
    # Tính ma trận tương quan
    returns = data.pct_change().dropna(how="all")
    corr_matrix = returns.corr()
    
    # Chuyển đổi correlation thành distance matrix
    # Distance = 1 - correlation (tương quan càng cao thì khoảng cách càng nhỏ)
    distance_matrix = np.sqrt((1 - corr_matrix) / 2)
    
    # Chuyển thành dạng condensed distance matrix
    dist_condensed = squareform(distance_matrix, checks=False)
    
    # Thực hiện hierarchical clustering
    linkage = sch.linkage(dist_condensed, method='single')
    
    # Tạo dendrogram
    fig = go.Figure()
    
    # Lấy dữ liệu dendrogram từ scipy
    dendro = sch.dendrogram(linkage, labels=corr_matrix.columns.tolist(), no_plot=True)
    
    # Vẽ các đường nối trong dendrogram
    icoord = np.array(dendro['icoord'])
    dcoord = np.array(dendro['dcoord'])
    colors = ['rgb(99, 110, 250)'] * len(icoord)
    
    for i in range(len(icoord)):
        fig.add_trace(go.Scatter(
            x=icoord[i],
            y=dcoord[i],
            mode='lines',
            line=dict(color=colors[i], width=2),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Lấy labels và positions
    labels = dendro['ivl']
    label_coords = dendro['leaves']
    
    # Tạo dictionary để map từ label sang trọng số
    weight_values = {k: v for k, v in weights.items()}
    
    # Thêm labels ở dưới cùng với trọng số
    label_text = []
    for label in labels:
        weight = weight_values.get(label, 0)
        label_text.append(f"{label}<br>({weight:.1%})")
    
    # Cập nhật layout
    fig.update_layout(
        title={
            'text': "Biểu đồ Phân Cấp Tài Sản - Cấu Trúc Tương Quan",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f77b4'}
        },
        xaxis={
            'showticklabels': True,
            'tickmode': 'array',
            'tickvals': [(label_coords[i] + 1) * 10 for i in range(len(labels))],
            'ticktext': label_text,
            'tickangle': -45,
            'side': 'bottom',
            'title': 'Mã Cổ Phiếu (% Trọng số)'
        },
        yaxis={
            'title': 'Mức Độ Khác Biệt (Distance)',
            'showgrid': True,
            'gridcolor': 'lightgray'
        },
        plot_bgcolor='white',
        height=500,
        hovermode='closest',
        showlegend=False,
        margin=dict(b=150)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Thêm giải thích ý nghĩa
    with st.expander("Ý nghĩa của Biểu đồ Dendrogram"):
        st.markdown("""
        ### Cách đọc biểu đồ Dendrogram:
        **1. Cấu trúc hình cây:**  
        - Mỗi mã cổ phiếu là một nhánh ở dưới cùng  
        - Các nhánh được kết nối với nhau qua các điểm nối  
        - Điểm nối càng thấp → Tương quan càng cao  

        **2. Nhóm tài sản:**  
        - **Các mã cổ phiếu nằm trên cùng một nhánh gần gốc:** Có tương quan cao nhất với nhau  
        - **Các mã cổ phiếu ở các nhánh xa nhau:** Có mối tương quan thấp, đa dạng hóa tốt  

        **3. Chiều cao của điểm nối (Trục Y):**  
        - Giá trị thấp (gần 0): Hai tài sản rất giống nhau về biến động giá  
        - Giá trị cao: Hai nhóm tài sản khác biệt nhau nhiều  

        **4. Ứng dụng trong đầu tư:**  
        - Giúp nhận biết nhóm mã cổ phiếu có xu hướng tăng/giảm cùng nhau  
        - Tránh đầu tư quá nhiều vào các mã cổ phiếu trong cùng một nhóm (cùng nhánh)  
        - Chọn mã cổ phiếu từ các nhánh khác nhau để giảm rủi ro  

        **5. Tư duy của mô hình HRP:**  
        - Mô hình sử dụng cấu trúc này để phân bổ vốn thông minh  
        - Các nhóm tài sản có tương quan cao sẽ được phân bổ ít vốn hơn  
        - Các nhóm độc lập được phân bổ nhiều vốn hơn để đa dạng hóa  

        **💡 Ví dụ:**  
        Nếu 3 mã cổ phiếu ngân hàng nằm trên cùng một nhánh thấp, điều này có nghĩa chúng biến động rất giống nhau.  
        Thay vì đầu tư 30% vào mỗi mã cổ phiếu, mô hình HRP sẽ phân bổ tổng cộng 30% cho cả nhóm và tăng phân bổ cho các mã cổ phiếu ở nhóm khác.
        """)
    
    # Hiển thị ma trận tương quan
    st.subheader("Ma Trận Tương Quan")
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Tương quan")
    ))
    
    fig_corr.update_layout(
        title={
            'text': "Ma Trận Tương Quan giữa các Mã Cổ Phiếu",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis={'side': 'bottom'},
        height=500,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Phân tích các nhóm
    st.subheader("Phân Tích Nhóm Tương Quan")
    
    # Tìm các cặp có tương quan cao
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > 0.5:  # Ngưỡng tương quan cao
                high_corr_pairs.append({
                    'Cổ phiếu 1': corr_matrix.columns[i],
                    'Cổ phiếu 2': corr_matrix.columns[j],
                    'Tương quan': f"{corr_val:.2f}"
                })
    
    if high_corr_pairs:
        st.write("**Các cặp cổ phiếu có tương quan cao (> 0.5):**")
        df_corr = pd.DataFrame(high_corr_pairs)
        st.dataframe(df_corr, use_container_width=True)
        st.info("⚠️ Các cặp cổ phiếu này có xu hướng biến động cùng chiều. "
                "Đầu tư vào nhiều cổ phiếu trong cùng nhóm có thể không mang lại hiệu quả đa dạng hóa.")
    else:
        st.success("✅ Không có cặp cổ phiếu nào có tương quan cao. Danh mục được đa dạng hóa tốt!")
    
    # Giải thích mô hình HRP
    with st.expander("Giải thích về Mô hình HRP (Hierarchical Risk Parity)"):
        st.markdown("""
        ### Mô hình Đa Dạng Hóa Thông Minh - HRP
        
        **1. Ý tưởng cốt lõi:**
        - HRP là mô hình phân bổ vốn dựa trên cấu trúc phân cấp của tương quan
        - Thay vì tối ưu hóa toán học phức tạp, HRP sử dụng "tư duy cây"
        - Phân bổ rủi ro đồng đều giữa các nhóm tài sản độc lập
        
        **2. Cách hoạt động:**
        
        **Bước 1 - Phân cụm (Tree Clustering):**
        - Nhóm các tài sản có tương quan cao vào cùng một cụm
        - Tạo ra cấu trúc phân cấp như bạn thấy trong Dendrogram
        
        **Bước 2 - Quasi-Diagonalization:**
        - Sắp xếp lại các tài sản theo thứ tự trong cây
        - Các tài sản giống nhau sẽ ở gần nhau
        
        **Bước 3 - Recursive Bisection:**
        - Phân chia vốn theo từng cấp trong cây
        - Mỗi nhóm nhận phần vốn tỷ lệ nghịch với rủi ro của nó
        - Tiếp tục chia nhỏ cho đến từng tài sản
        
        **3. Ưu điểm:**
        - ✅ Ổn định hơn các mô hình tối ưu hóa truyền thống
        - ✅ Không cần ước lượng lợi nhuận kỳ vọng (khó dự đoán chính xác)
        - ✅ Xử lý tốt khi các tài sản có tương quan cao
        - ✅ Giảm thiểu "đặt tất cả trứng vào một giỏ"
        - ✅ Dễ hiểu và giải thích cho nhà đầu tư
        
        **4. Nhược điểm:**
        - ⚠️ Không tối đa hóa lợi nhuận như Max Sharpe
        - ⚠️ Không tối thiểu hóa rủi ro tuyệt đối như Min Volatility
        - ⚠️ Phụ thuộc vào phương pháp clustering (single, complete, average)
        - ⚠️ Hiệu quả phụ thuộc vào chất lượng dữ liệu lịch sử
        
        **5. Khi nào nên sử dụng HRP:**
        - 📌 Khi danh mục có nhiều tài sản với tương quan phức tạp
        - 📌 Muốn một giải pháp ổn định, không nhạy cảm với nhiễu dữ liệu
        - 📌 Ưu tiên đa dạng hóa thay vì tối đa hóa lợi nhuận
        - 📌 Không tự tin vào dự báo lợi nhuận tương lai
        
        **6. So sánh với các mô hình khác:**
        
        | Tiêu chí | Markowitz/Max Sharpe | Min Volatility | HRP |
        |----------|---------------------|----------------|-----|
        | Mục tiêu | Tối đa Sharpe | Tối thiểu rủi ro | Cân bằng rủi ro |
        | Độ phức tạp | Cao | Cao | Trung bình |
        | Ổn định | Thấp | Trung bình | Cao |
        | Đa dạng hóa | Có thể tập trung | Tập trung | Tốt nhất |
        
        **Lời khuyên:** HRP là lựa chọn tốt cho nhà đầu tư thận trọng, muốn một danh mục 
        cân bằng và ổn định theo thời gian, đặc biệt khi không chắc chắn về dự báo thị trường.
        """)


def visualize_hrp_model(data, result):
    """
    Trực quan hóa kết quả mô hình HRP.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        result (dict): Kết quả từ hàm hrp_model()
    """
    if result is None:
        st.error("Không có kết quả để hiển thị.")
        return
    
    st.header("Kết Quả Mô Hình HRP (Hierarchical Risk Parity)")
    
    # Hiển thị các chỉ số chính
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Lợi Nhuận Kỳ Vọng",
            f"{result['Lợi nhuận kỳ vọng'] * 100:.2f}%",
            help="Lợi nhuận hàng năm dự kiến của danh mục"
        )
    
    with col2:
        st.metric(
            "Rủi Ro (Độ Lệch Chuẩn)",
            f"{result['Rủi ro (Độ lệch chuẩn)'] * 100:.2f}%",
            help="Biến động hàng năm của danh mục"
        )
    
    with col3:
        st.metric(
            "Tỷ Lệ Sharpe",
            f"{result['Tỷ lệ Sharpe']:.2f}",
            help="Lợi nhuận trên mỗi đơn vị rủi ro"
        )
    
    # Lấy trọng số để truyền vào dendrogram
    weights = result["Trọng số danh mục"]
    
    # Vẽ Dendrogram - Đây là phần quan trọng nhất của HRP
    plot_hrp_dendrogram(data, weights)


