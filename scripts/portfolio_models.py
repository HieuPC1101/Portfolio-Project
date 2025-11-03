"""
Module portfolio_models.py
Chứa các hàm tối ưu hóa danh mục đầu tư: Markowitz, Max Sharpe, Min Volatility, Min CVaR, Min CDaR, HRP.
"""

import numpy as np
import pandas as pd
from pypfopt import (
    EfficientFrontier, 
    risk_models, 
    expected_returns, 
    DiscreteAllocation,
    EfficientCVaR, 
    EfficientCDaR, 
    HRPOpt
)


def run_integer_programming(weights, latest_prices, total_portfolio_value):
    """
    Sử dụng Integer Programming (LP) để tối ưu phân bổ cổ phiếu.
    
    Args:
        weights (dict): Trọng số của từng cổ phiếu
        latest_prices (pd.Series): Giá cổ phiếu mới nhất
        total_portfolio_value (float): Tổng giá trị danh mục đầu tư
        
    Returns:
        tuple: (allocation_lp, leftover_lp)
    """
    allocation = DiscreteAllocation(
        weights, 
        latest_prices, 
        total_portfolio_value=total_portfolio_value
    )
    allocation_lp, leftover_lp = allocation.lp_portfolio(
        reinvest=False, 
        verbose=True, 
        solver='ECOS_BB'
    )
    return allocation_lp, leftover_lp


def markowitz_optimization(price_data, total_investment, get_latest_prices_func):
    """
    Mô hình Markowitz: Tối ưu hóa giữa lợi nhuận và rủi ro.
    
    Args:
        price_data (pd.DataFrame): Dữ liệu giá cổ phiếu
        total_investment (float): Tổng số tiền đầu tư
        get_latest_prices_func (function): Hàm lấy giá cổ phiếu mới nhất
        
    Returns:
        dict: Kết quả tối ưu hóa
    """
    tickers = price_data.columns.tolist()
    num_assets = len(tickers)

    if num_assets == 0:
        print("Danh sách mã cổ phiếu đã chọn không hợp lệ. Vui lòng kiểm tra lại.")
        return None

    log_ret = np.log(price_data / price_data.shift(1)).dropna()
    n_portfolios = 10000
    all_weights = np.zeros((n_portfolios, num_assets))
    ret_arr = np.zeros(n_portfolios)
    vol_arr = np.zeros(n_portfolios)
    sharpe_arr = np.zeros(n_portfolios)

    mean_returns = log_ret.mean() * 252  # Lợi nhuận kỳ vọng hàng năm
    cov_matrix = log_ret.cov() * 252  # Ma trận hiệp phương sai hàng năm

    np.random.seed(42)  # Thiết lập giá trị seed để kết quả ổn định

    for i in range(n_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        all_weights[i, :] = weights

        rf = 0.02
        ret_arr[i] = np.dot(mean_returns, weights)
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_arr[i] = (ret_arr[i] - rf) / vol_arr[i]

    max_sharpe_idx = sharpe_arr.argmax()
    optimal_weights = all_weights[max_sharpe_idx]

    weight2 = dict(zip(tickers, optimal_weights))
    latest_prices = get_latest_prices_func(tickers)
    latest_prices_series = pd.Series(latest_prices)
    total_portfolio_value = total_investment
    allocation_lp, leftover_lp = run_integer_programming(
        weight2, 
        latest_prices_series, 
        total_portfolio_value
    )

    result = {
        "Trọng số danh mục": dict(zip(tickers, optimal_weights)),
        "Lợi nhuận kỳ vọng": ret_arr[max_sharpe_idx],
        "Rủi ro (Độ lệch chuẩn)": vol_arr[max_sharpe_idx],
        "Tỷ lệ Sharpe": sharpe_arr[max_sharpe_idx],
        "Số cổ phiếu cần mua": allocation_lp,
        "Số tiền còn lại": leftover_lp,
        "Giá cổ phiếu": latest_prices,
        # Thêm dữ liệu cho biểu đồ
        "ret_arr": ret_arr,
        "vol_arr": vol_arr,
        "sharpe_arr": sharpe_arr,
        "all_weights": all_weights,
        "max_sharpe_idx": max_sharpe_idx
    }

    return result


def max_sharpe(data, total_investment, get_latest_prices_func):
    """
    Mô hình Max Sharpe Ratio: Tối đa hóa tỷ lệ Sharpe.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        total_investment (float): Tổng số tiền đầu tư
        get_latest_prices_func (function): Hàm lấy giá cổ phiếu mới nhất
        
    Returns:
        dict: Kết quả tối ưu hóa
    """
    try:
        mean_returns = expected_returns.mean_historical_return(data)
        cov_matrix = risk_models.sample_cov(data)

        ef = EfficientFrontier(mean_returns, cov_matrix)
        weights = ef.max_sharpe()
        performance = ef.portfolio_performance(verbose=False)
        cleaned_weights = ef.clean_weights()

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices_func(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        allocation_lp, leftover_lp = run_integer_programming(
            weights, 
            latest_prices_series, 
            total_portfolio_value
        )

        return {
            "Trọng số danh mục": cleaned_weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro (Độ lệch chuẩn)": performance[1],
            "Tỷ lệ Sharpe": performance[2],
            "Số cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá cổ phiếu": latest_prices
        }
    except Exception as e:
        print(f"Lỗi trong mô hình Max Sharpe: {e}")
        return None


def min_volatility(data, total_investment, get_latest_prices_func):
    """
    Mô hình Min Volatility: Tối thiểu hóa độ lệch chuẩn (rủi ro).
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        total_investment (float): Tổng số tiền đầu tư
        get_latest_prices_func (function): Hàm lấy giá cổ phiếu mới nhất
        
    Returns:
        dict: Kết quả tối ưu hóa
    """
    try:
        mean_returns = expected_returns.mean_historical_return(data)
        cov_matrix = risk_models.sample_cov(data)

        ef = EfficientFrontier(mean_returns, cov_matrix)
        weights = ef.min_volatility()
        performance = ef.portfolio_performance(verbose=False)
        cleaned_weights = ef.clean_weights()

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices_func(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        allocation_lp, leftover_lp = run_integer_programming(
            weights, 
            latest_prices_series, 
            total_portfolio_value
        )

        return {
            "Trọng số danh mục": cleaned_weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro (Độ lệch chuẩn)": performance[1],
            "Tỷ lệ Sharpe": performance[2],
            "Số cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá cổ phiếu": latest_prices
        }
    except Exception as e:
        print(f"Lỗi trong mô hình Min Volatility: {e}")
        return None


def min_cvar(data, total_investment, get_latest_prices_func, beta=0.95):
    """
    Mô hình Min CVaR: Tối thiểu hóa Conditional Value at Risk.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        total_investment (float): Tổng số tiền đầu tư
        get_latest_prices_func (function): Hàm lấy giá cổ phiếu mới nhất
        beta (float): Mức độ tin cậy (mặc định 0.95)
        
    Returns:
        dict: Kết quả tối ưu hóa
    """
    try:
        mean_returns = expected_returns.mean_historical_return(data)
        returns = expected_returns.returns_from_prices(data).dropna()

        cvar_optimizer = EfficientCVaR(mean_returns, returns, beta=beta)
        weights = cvar_optimizer.min_cvar()
        performance = cvar_optimizer.portfolio_performance()
        
        # Tính ma trận hiệp phương sai
        cov_matrix = risk_models.sample_cov(data)

        # Tính độ lệch chuẩn của danh mục
        weights_array = np.array(list(weights.values()))
        portfolio_std = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        rf = 0.02
        sharpe_ratio = (performance[0] - rf) / portfolio_std

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices_func(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        allocation_lp, leftover_lp = run_integer_programming(
            weights, 
            latest_prices_series, 
            total_portfolio_value
        )

        return {
            "Trọng số danh mục": weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro CVaR": performance[1],
            "Số cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá cổ phiếu": latest_prices,
            "Rủi ro (Độ lệch chuẩn)": portfolio_std,
            "Tỷ lệ Sharpe": sharpe_ratio
        }
    except Exception as e:
        print(f"Lỗi trong mô hình Min CVaR: {e}")
        return None


def min_cdar(data, total_investment, get_latest_prices_func, beta=0.95):
    """
    Mô hình Min CDaR: Tối thiểu hóa Conditional Drawdown at Risk.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        total_investment (float): Tổng số tiền đầu tư
        get_latest_prices_func (function): Hàm lấy giá cổ phiếu mới nhất
        beta (float): Mức độ tin cậy (mặc định 0.95)
        
    Returns:
        dict: Kết quả tối ưu hóa
    """
    try:
        mean_returns = expected_returns.mean_historical_return(data)
        returns = expected_returns.returns_from_prices(data).dropna()

        cdar_optimizer = EfficientCDaR(mean_returns, returns, beta=beta)
        weights = cdar_optimizer.min_cdar()
        performance = cdar_optimizer.portfolio_performance()

        cov_matrix = risk_models.sample_cov(data)
        weights_array = np.array(list(weights.values()))
        portfolio_std = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        rf = 0.02
        sharpe_ratio = (performance[0] - rf) / portfolio_std

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices_func(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        allocation_lp, leftover_lp = run_integer_programming(
            weights, 
            latest_prices_series, 
            total_portfolio_value
        )

        return {
            "Trọng số danh mục": weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro CDaR": performance[1],
            "Số cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá cổ phiếu": latest_prices,
            "Rủi ro (Độ lệch chuẩn)": portfolio_std,
            "Tỷ lệ Sharpe": sharpe_ratio
        }
    except Exception as e:
        print(f"Lỗi trong mô hình Min CDaR: {e}")
        return None


def hrp_model(data, total_investment, get_latest_prices_func):
    """
    Mô hình HRP (Hierarchical Risk Parity): Phân bổ rủi ro phân cấp.
    
    Args:
        data (pd.DataFrame): Dữ liệu giá cổ phiếu
        total_investment (float): Tổng số tiền đầu tư
        get_latest_prices_func (function): Hàm lấy giá cổ phiếu mới nhất
        
    Returns:
        dict: Kết quả tối ưu hóa
    """
    try:
        returns = data.pct_change().dropna(how="all")
        hrp = HRPOpt(returns)
        weights = hrp.optimize(linkage_method="single")
        performance = hrp.portfolio_performance()

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices_func(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        allocation_lp, leftover_lp = run_integer_programming(
            weights, 
            latest_prices_series, 
            total_portfolio_value
        )

        return {
            "Trọng số danh mục": weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro (Độ lệch chuẩn)": performance[1],
            "Tỷ lệ Sharpe": performance[2],
            "Số cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá cổ phiếu": latest_prices
        }
    except Exception as e:
        print(f"Lỗi trong mô hình HRP: {e}")
        return None
