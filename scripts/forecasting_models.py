"""
Module forecasting_models.py
Chứa các hàm dự báo chuỗi thời gian cho giá cổ phiếu.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Thư viện statsmodels chưa được cài đặt. Vui lòng chạy: pip install statsmodels")


def check_stationarity(data):
    """
    Kiểm tra tính dừng của chuỗi thời gian.
    
    Args:
        data (pd.Series): Chuỗi dữ liệu giá
        
    Returns:
        bool: True nếu chuỗi dừng
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(data.dropna())
        # p-value < 0.05 nghĩa là chuỗi dừng
        return result[1] < 0.05
    except:
        return False


def auto_arima_params(data, max_p=5, max_q=5):
    """
    Tự động tìm tham số tốt nhất cho ARIMA.
    
    Args:
        data (pd.Series): Chuỗi dữ liệu giá
        max_p (int): Giá trị p tối đa
        max_q (int): Giá trị q tối đa
        
    Returns:
        tuple: (p, d, q) tốt nhất
    """
    best_aic = np.inf
    best_params = (1, 1, 1)
    
    # Kiểm tra nếu chuỗi dừng
    d = 0 if check_stationarity(data) else 1
    
    # Tìm kiếm grid đơn giản
    for p in range(0, min(max_p, 4)):
        for q in range(0, min(max_q, 4)):
            try:
                model = ARIMA(data, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_params = (p, d, q)
            except:
                continue
    
    return best_params


def forecast_arima(data, ticker, forecast_periods=30, order=None):
    """
    Dự báo giá cổ phiếu sử dụng mô hình ARIMA.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu với cột 'time' và ticker
        ticker (str): Mã cổ phiếu
        forecast_periods (int): Số ngày dự báo
        order (tuple): Tham số ARIMA (p, d, q). Nếu None, tự động tìm
        
    Returns:
        dict: {
            'forecast': pd.Series với index là thời gian,
            'lower_bound': pd.Series - Khoảng tin cậy dưới,
            'upper_bound': pd.Series - Khoảng tin cậy trên,
            'model_name': str,
            'params': dict
        }
    """
    if not STATSMODELS_AVAILABLE:
        return None
    
    try:
        # Chuẩn bị dữ liệu
        if ticker not in data.columns:
            return None
        
        prices = data[ticker].dropna()
        if len(prices) < 30:
            return None
        
        # Tự động tìm tham số nếu không được cung cấp
        if order is None:
            order = auto_arima_params(prices)
        
        # Fit mô hình ARIMA
        model = ARIMA(prices, order=order)
        fitted_model = model.fit()
        
        # Dự báo
        forecast_result = fitted_model.forecast(steps=forecast_periods)
        
        # Tính khoảng tin cậy
        # Lấy forecast object để có confidence interval
        forecast_obj = fitted_model.get_forecast(steps=forecast_periods)
        forecast_ci = forecast_obj.conf_int(alpha=0.05)  # 95% confidence interval
        
        # Tạo index thời gian cho dự báo
        last_date = data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        # Tạo Series cho dự báo
        forecast_series = pd.Series(forecast_result.values, index=forecast_index)
        lower_bound = pd.Series(forecast_ci.iloc[:, 0].values, index=forecast_index)
        upper_bound = pd.Series(forecast_ci.iloc[:, 1].values, index=forecast_index)
        
        return {
            'forecast': forecast_series,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'model_name': f'ARIMA{order}',
            'params': {
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            },
            'historical': prices
        }
    
    except Exception as e:
        print(f"Lỗi khi dự báo ARIMA cho {ticker}: {str(e)}")
        return None


def forecast_exponential_smoothing(data, ticker, forecast_periods=30, seasonal_periods=None):
    """
    Dự báo giá cổ phiếu sử dụng Exponential Smoothing (Holt-Winters).
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        ticker (str): Mã cổ phiếu
        forecast_periods (int): Số ngày dự báo
        seasonal_periods (int): Chu kỳ mùa vụ (None = không có)
        
    Returns:
        dict: Tương tự forecast_arima
    """
    if not STATSMODELS_AVAILABLE:
        return None
    
    try:
        if ticker not in data.columns:
            return None
        
        prices = data[ticker].dropna()
        if len(prices) < 30:
            return None
        
        # Fit mô hình Exponential Smoothing
        if seasonal_periods and len(prices) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                prices,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add'
            )
        else:
            model = ExponentialSmoothing(prices, trend='add')
        
        fitted_model = model.fit()
        
        # Dự báo
        forecast_result = fitted_model.forecast(steps=forecast_periods)
        
        # Tạo index thời gian
        last_date = data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        forecast_series = pd.Series(forecast_result.values, index=forecast_index)
        
        # Tính khoảng tin cậy đơn giản (sử dụng std của residuals)
        residuals = fitted_model.fittedvalues - prices
        std_residual = residuals.std()
        
        lower_bound = forecast_series - 1.96 * std_residual
        upper_bound = forecast_series + 1.96 * std_residual
        
        model_name = 'Exponential Smoothing'
        if seasonal_periods:
            model_name += f' (mùa vụ={seasonal_periods})'
        
        return {
            'forecast': forecast_series,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'model_name': model_name,
            'params': {
                'seasonal_periods': seasonal_periods
            },
            'historical': prices
        }
    
    except Exception as e:
        print(f"Lỗi khi dự báo Exponential Smoothing cho {ticker}: {str(e)}")
        return None


def forecast_moving_average(data, ticker, forecast_periods=30, window=20):
    """
    Dự báo đơn giản sử dụng Moving Average.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        ticker (str): Mã cổ phiếu
        forecast_periods (int): Số ngày dự báo
        window (int): Cửa sổ trung bình động
        
    Returns:
        dict: Tương tự forecast_arima
    """
    try:
        if ticker not in data.columns:
            return None
        
        prices = data[ticker].dropna()
        if len(prices) < window:
            return None
        
        # Tính moving average
        ma = prices.rolling(window=window).mean()
        
        # Tính xu hướng từ MA gần đây
        recent_ma = ma.iloc[-window:]
        if len(recent_ma) < 2:
            trend = 0
        else:
            # Tính slope đơn giản
            x = np.arange(len(recent_ma))
            y = recent_ma.values
            trend = np.polyfit(x, y, 1)[0]
        
        # Dự báo: MA cuối + trend * số ngày
        last_ma = ma.iloc[-1]
        forecast_values = [last_ma + trend * i for i in range(1, forecast_periods + 1)]
        
        # Tạo index thời gian
        last_date = data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        forecast_series = pd.Series(forecast_values, index=forecast_index)
        
        # Tính khoảng tin cậy từ volatility
        volatility = prices.pct_change().std()
        price_std = prices.iloc[-1] * volatility * np.sqrt(np.arange(1, forecast_periods + 1))
        
        lower_bound = forecast_series - 1.96 * pd.Series(price_std, index=forecast_index)
        upper_bound = forecast_series + 1.96 * pd.Series(price_std, index=forecast_index)
        
        return {
            'forecast': forecast_series,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'model_name': f'Moving Average (MA{window})',
            'params': {
                'window': window,
                'trend': trend
            },
            'historical': prices
        }
    
    except Exception as e:
        print(f"Lỗi khi dự báo Moving Average cho {ticker}: {str(e)}")
        return None


def get_forecast(data, ticker, method='auto', forecast_periods=30, **kwargs):
    """
    Hàm tổng hợp để lấy dự báo theo phương pháp được chọn.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        ticker (str): Mã cổ phiếu
        method (str): Phương pháp dự báo ('auto', 'arima', 'exp_smoothing', 'moving_average')
        forecast_periods (int): Số ngày dự báo
        **kwargs: Tham số bổ sung cho từng phương pháp
        
    Returns:
        dict: Kết quả dự báo
    """
    if method == 'arima' or method == 'auto':
        result = forecast_arima(data, ticker, forecast_periods, kwargs.get('order'))
        if result:
            return result
    
    if method == 'exp_smoothing' or (method == 'auto' and not result):
        result = forecast_exponential_smoothing(
            data, ticker, forecast_periods, 
            kwargs.get('seasonal_periods')
        )
        if result:
            return result
    
    if method == 'moving_average' or (method == 'auto' and not result):
        result = forecast_moving_average(
            data, ticker, forecast_periods,
            kwargs.get('window', 20)
        )
        return result
    
    return None


def calculate_forecast_metrics(actual, forecast):
    """
    Tính các chỉ số đánh giá độ chính xác dự báo.
    
    Args:
        actual (pd.Series): Giá trị thực tế
        forecast (pd.Series): Giá trị dự báo
        
    Returns:
        dict: Các chỉ số MAE, MSE, RMSE, R², MAPE
    """
    if len(actual) != len(forecast):
        return None
    
    try:
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - forecast))
        
        # Mean Squared Error
        mse = np.mean((actual - forecast) ** 2)
        
        # Root Mean Square Error
        rmse = np.sqrt(mse)
        
        # R-squared (Coefficient of Determination)
        ss_res = np.sum((actual - forecast) ** 2)  # Residual sum of squares
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    except Exception as e:
        print(f"Lỗi khi tính metrics: {str(e)}")
        return None


def backtest_forecast(data, ticker, method='auto', test_periods=30, **kwargs):
    """
    Kiểm tra độ chính xác của mô hình dự báo bằng cách dự báo trên dữ liệu lịch sử.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu đầy đủ
        ticker (str): Mã cổ phiếu
        method (str): Phương pháp dự báo
        test_periods (int): Số ngày để test (sử dụng làm tập test)
        **kwargs: Tham số bổ sung cho phương pháp dự báo
        
    Returns:
        dict: {
            'metrics': dict - Các chỉ số đánh giá (MAE, MSE, RMSE, R², MAPE),
            'actual': pd.Series - Giá trị thực tế trong tập test,
            'predicted': pd.Series - Giá trị dự báo,
            'train_data': pd.DataFrame - Dữ liệu huấn luyện
        }
    """
    try:
        if ticker not in data.columns:
            return None
        
        # Chia dữ liệu thành train và test
        prices = data[ticker].dropna()
        if len(prices) < test_periods + 30:  # Cần ít nhất 30 ngày để train
            return None
        
        # Tách dữ liệu
        train_data = data.iloc[:-test_periods].copy()
        actual_prices = prices.iloc[-test_periods:]
        
        # Thực hiện dự báo trên tập train
        forecast_result = get_forecast(
            train_data, 
            ticker, 
            method=method, 
            forecast_periods=test_periods,
            **kwargs
        )
        
        if forecast_result is None:
            return None
        
        forecast_values = forecast_result['forecast']
        
        # Căn chỉnh index để so sánh
        # Lấy giá trị theo thứ tự thời gian
        actual_values = actual_prices.values
        predicted_values = forecast_values.values[:len(actual_values)]
        
        # Tính các chỉ số
        metrics = calculate_forecast_metrics(
            pd.Series(actual_values),
            pd.Series(predicted_values)
        )
        
        if metrics is None:
            return None
        
        return {
            'metrics': metrics,
            'actual': pd.Series(actual_values, index=actual_prices.index),
            'predicted': pd.Series(predicted_values, index=forecast_values.index[:len(actual_values)]),
            'train_data': train_data,
            'model_name': forecast_result.get('model_name', 'Unknown')
        }
        
    except Exception as e:
        print(f"Lỗi khi backtest cho {ticker}: {str(e)}")
        return None
