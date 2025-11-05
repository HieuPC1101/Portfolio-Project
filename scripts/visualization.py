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


def plot_candlestick_chart(ohlc_data, ticker):
    """
    Vẽ biểu đồ nến (Candlestick Chart) cho một cổ phiếu với các chỉ báo kỹ thuật.
    
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
    
    # 5. Alpha (so với benchmark đầu tiên nếu có)
    alpha = 0
    if benchmark_data:
        first_benchmark = list(benchmark_data.keys())[0]
        benchmark_returns = benchmark_data[first_benchmark].pct_change().dropna()
        
        # Đảm bảo cùng index
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_aligned = portfolio_returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]
        
        # Tính beta
        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0][1]
        benchmark_variance = np.var(benchmark_aligned)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Tính alpha hàng năm
        portfolio_annual_return = portfolio_aligned.mean() * 252
        benchmark_annual_return = benchmark_aligned.mean() * 252
        alpha = (portfolio_annual_return - benchmark_annual_return) * 100  # Phần trăm
    
    # 6. ROI (Return on Investment)
    roi = total_return  # Giống Total Return cho backtesting
    
    # Tạo bảng thống kê tổng hợp
    st.markdown("### Bảng Thống kê Tổng hợp")
    
    metrics_data = {
        "Chỉ số": [
            "Lợi nhuận tổng (Total Return)",
            "Lợi nhuận hàng năm (Annualized Return)",
            "ROI (Return on Investment)",
            "Độ biến động (Volatility)",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Alpha",
            "Maximum Drawdown"
        ],
        "Giá trị": [
            f"{total_return:.2f}%",
            f"{annualized_return:.2f}%",
            f"{roi:.2f}%",
            f"{volatility:.2f}%",
            f"{sharpe_ratio:.4f}",
            f"{sortino_ratio:.4f}",
            f"{alpha:.2f}%",
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
        "ROI": roi,
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
            
            # Hiển thị các chỉ số đánh giá chất lượng dự báo nếu có
            if 'metrics' in forecast_result and forecast_result['metrics'] is not None:
                st.markdown("---")
                st.markdown("**Chỉ số đánh giá chất lượng dự báo**")
                
                metrics = forecast_result['metrics']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("MAE", f"{metrics['MAE']:.4f}")
                    st.caption("Mean Absolute Error")
                    
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                    st.caption("Root Mean Squared Error")
                
                with col3:
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    st.caption("Mean Absolute % Error")
                
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("MSE", f"{metrics['MSE']:.4f}")
                    st.caption("Mean Squared Error")
                    
                with col5:
                    r2_value = metrics['R2']
                    if r2_value > 0.9:
                        r2_label = "Rất tốt"
                    elif r2_value > 0.7:
                        r2_label = "Tốt"
                    elif r2_value > 0.5:
                        r2_label = "Chấp nhận được"
                    else:
                        r2_label = "Kém"
                    st.metric("R²", f"{r2_value:.4f}", delta=r2_label)
                    st.caption("Coefficient of Determination")
        
        # Giải thích các chỉ số (expander riêng bên ngoài)
        if forecast_result and 'metrics' in forecast_result and forecast_result['metrics'] is not None:
            with st.expander("Giải thích các chỉ số đánh giá", expanded=False):
                st.markdown("""
                **MAE (Mean Absolute Error):** Sai số tuyệt đối trung bình. Giá trị càng nhỏ càng tốt.
                
                **MSE (Mean Squared Error):** Sai số bình phương trung bình. Giá trị càng nhỏ càng tốt.
                
                **RMSE (Root Mean Squared Error):** Căn bậc hai của MSE. Có cùng đơn vị với dữ liệu gốc, dễ diễn giải hơn.
                
                **R² (Hệ số xác định):** Đo lường mức độ phù hợp của mô hình. Giá trị từ 0 đến 1, càng gần 1 càng tốt.
                - R² > 0.9: Rất tốt
                - R² > 0.7: Tốt
                - R² > 0.5: Chấp nhận được
                - R² < 0.5: Kém
                
                **MAPE (Mean Absolute Percentage Error):** Sai số phần trăm tuyệt đối trung bình. Giá trị càng nhỏ càng tốt.
                - MAPE < 10%: Rất tốt
                - MAPE < 20%: Tốt
                - MAPE < 50%: Chấp nhận được
                - MAPE > 50%: Kém
                """)
