"""
Module portfolio_models.py
Chứa các hàm tối ưu hóa danh mục đầu tư: Markowitz, Max Sharpe, Min Volatility, Min CVaR, Min CDaR, HRP.
"""

import numpy as np
import pandas as pd
import logging
from pypfopt import (
    EfficientFrontier, 
    risk_models, 
    expected_returns, 
    DiscreteAllocation,
    EfficientCVaR, 
    EfficientCDaR, 
    HRPOpt
)
from scipy.optimize import minimize

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def optimize_hrp_allocation(target_weights, prices, total_investment):
    """
    Tối ưu hóa phân bổ cổ phiếu cho HRP bằng cách tối thiểu hóa
    tổng bình phương sai số giữa trọng số mục tiêu và trọng số thực tế.
    
    Args:
        target_weights (dict): Trọng số HRP mục tiêu {ticker: weight}
        prices (dict): Giá cổ phiếu {ticker: price}
        total_investment (float): Tổng số tiền đầu tư
    
    Returns:
        tuple: (allocation_dict, leftover)
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    logger.info("=" * 60)
    logger.info("BAT DAU TOI UU PHAN BO HRP")
    logger.info(f"So tien dau tu: {total_investment:,.0f} VND")
    logger.info(f"Trong so muc tieu: {target_weights}")
    logger.info(f"Gia co phieu: {prices}")
    
    # Chuyển đổi sang mảng numpy
    tickers = list(target_weights.keys())
    n = len(tickers)
    
    target_w = np.array([target_weights[t] for t in tickers])
    price_arr = np.array([prices[t] for t in tickers])
    
    # Hàm mục tiêu: Minimize Σ [(weight_i - (shares_i * price_i) / total_investment)^2]
    # Chuyển đổi thành dạng bậc hai: (1/total_investment^2) * Σ [(target_w_i * total_investment - shares_i * price_i)^2]
    # = (1/total_investment^2) * Σ [(a_i - shares_i * price_i)^2] với a_i = target_w_i * total_investment
    
    # Ma trận Q cho hàm mục tiêu bậc hai: minimize (1/2) * x^T * Q * x + c^T * x
    # Với x là shares, ta có: Σ [(a_i - x_i * price_i)^2]
    # = Σ [a_i^2 - 2*a_i*x_i*price_i + x_i^2*price_i^2]
    # = Σ [x_i^2 * price_i^2] - 2 * Σ [a_i * x_i * price_i] + const
    
    a = target_w * total_investment
    
    # Sử dụng scipy.optimize.minimize với phương pháp SLSQP
    # vì milp không hỗ trợ hàm mục tiêu bậc hai
    
    def objective(shares):
        """Hàm mục tiêu: tổng bình phương sai số"""
        actual_values = shares * price_arr
        actual_weights = actual_values / total_investment
        squared_errors = (target_w - actual_weights) ** 2
        return np.sum(squared_errors)
    
    def constraint_budget(shares):
        """Ràng buộc ngân sách: tổng chi tiêu <= total_investment"""
        return total_investment - np.sum(shares * price_arr)
    
    # Ràng buộc
    constraints = [
        {'type': 'ineq', 'fun': constraint_budget}  # inequality: constraint >= 0
    ]
    
    # Giới hạn: shares >= 0 và là số nguyên (sẽ làm tròn sau)
    bounds = [(0, None) for _ in range(n)]
    
    # Giá trị khởi tạo: phân bổ tối đa có thể theo trọng số
    x0 = np.floor((target_w * total_investment) / price_arr)
    
    # Tối ưu hóa liên tục trước
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        logger.warning(f"Toi uu khong hoi tu hoan toan: {result.message}")
    
    # Làm tròn xuống để đảm bảo không vượt ngân sách
    shares_optimal = np.floor(result.x).astype(int)
    
    # Kiểm tra và điều chỉnh nếu còn dư ngân sách
    total_spent = np.sum(shares_optimal * price_arr)
    remaining = total_investment - total_spent
    
    logger.info(f"Sau khi lam tron: Tong chi: {total_spent:,.0f} VND, Con lai: {remaining:,.0f} VND")
    
    # Cố gắng mua thêm cổ phiếu nếu còn tiền
    # Ưu tiên mua cổ phiếu có trọng số mục tiêu cao nhưng chưa đủ
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Tính trọng số hiện tại
        current_values = shares_optimal * price_arr
        current_weights = current_values / total_investment
        weight_errors = target_w - current_weights
        
        # Sắp xếp theo độ ưu tiên: cổ phiếu có sai số dương lớn nhất
        priority_indices = np.argsort(-weight_errors)
        
        for idx in priority_indices:
            if remaining >= price_arr[idx]:
                # Thử mua thêm 1 cổ phiếu
                shares_optimal[idx] += 1
                remaining -= price_arr[idx]
                improved = True
                logger.info(f"Mua them 1 {tickers[idx]} voi gia {price_arr[idx]:,.0f} VND")
                break
    
    # Tạo dictionary kết quả
    allocation = {tickers[i]: int(shares_optimal[i]) for i in range(n) if shares_optimal[i] > 0}
    leftover = remaining
    
    # Validation
    total_spent_final = sum(allocation[ticker] * prices[ticker] for ticker in allocation)
    
    logger.info("-" * 60)
    logger.info("KET QUA TOI UU HRP:")
    logger.info(f"Phan bo: {allocation}")
    logger.info(f"Tong tien da su dung: {total_spent_final:,.0f} VND")
    logger.info(f"So tien con lai: {leftover:,.0f} VND")
    logger.info(f"Tong cong: {total_spent_final + leftover:,.0f} VND")
    
    # Tính và hiển thị trọng số thực tế
    actual_weights = {}
    for ticker in allocation:
        actual_value = allocation[ticker] * prices[ticker]
        actual_weights[ticker] = actual_value / total_investment
    
    logger.info("-" * 60)
    logger.info("SO SANH TRONG SO:")
    for ticker in tickers:
        target = target_weights.get(ticker, 0)
        actual = actual_weights.get(ticker, 0)
        diff = actual - target
        logger.info(f"{ticker}: Muc tieu={target:.4f}, Thuc te={actual:.4f}, Chenh lech={diff:+.4f}")
    
    # Tính tổng bình phương sai số
    total_squared_error = sum((actual_weights.get(t, 0) - target_weights.get(t, 0))**2 for t in tickers)
    logger.info(f"Tong binh phuong sai so: {total_squared_error:.6f}")
    logger.info("=" * 60)
    
    return allocation, leftover


def run_integer_programming(weights, latest_prices, total_portfolio_value):
    """
    Sử dụng Integer Programming (LP) để tối ưu phân bổ mã cổ phiếu.
    
    Args:
    weights (dict): Trọng số của từng mã cổ phiếu
    latest_prices (pd.Series): Giá mã cổ phiếu mới nhất
        total_portfolio_value (float): Tổng giá trị danh mục đầu tư
        
    Returns:
        tuple: (allocation_lp, leftover_lp)
    """
    # Validation: Kiểm tra số tiền đầu tư
    if total_portfolio_value <= 0:
        logger.error(f"So tien dau tu phai lon hon 0. Nhan duoc: {total_portfolio_value:,.0f} VND")
        raise ValueError(f"So tien dau tu phai lon hon 0. Nhan duoc: {total_portfolio_value:,.0f} VND")
    
    logger.info("=" * 60)
    logger.info("BAT DAU PHAN BO DANH MUC DAU TU")
    logger.info(f"So tien dau tu: {total_portfolio_value:,.0f} VND")
    logger.info(f"Trong so nhan duoc: {weights}")
    logger.info(f"Tong trong so: {sum(weights.values()) if isinstance(weights, dict) else weights.sum():.6f}")
    logger.info(f"Gia co phieu nhan duoc: {latest_prices.to_dict() if isinstance(latest_prices, pd.Series) else latest_prices}")
    
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
    
    # Validation: Kiểm tra kết quả phân bổ
    total_spent = sum(allocation_lp[ticker] * latest_prices[ticker] for ticker in allocation_lp)
    total_check = total_spent + leftover_lp
    
    logger.info("-" * 60)
    logger.info("KET QUA PHAN BO:")
    logger.info(f"Tong tien da su dung: {total_spent:,.0f} VND")
    logger.info(f"So tien con lai: {leftover_lp:,.0f} VND")
    logger.info(f"Tong cong (kiem tra): {total_check:,.0f} VND")
    
    # Kiểm tra tổng tiền có khớp không
    if abs(total_check - total_portfolio_value) > 1:  # Cho phép sai số 1 VND
        logger.warning(f"Co su chenh lech: {abs(total_check - total_portfolio_value):,.0f} VND")
    else:
        logger.info("PHAN BO DANH MUC THANH CONG!")
    logger.info("=" * 60)
    
    return allocation_lp, leftover_lp


def markowitz_optimization(price_data, total_investment, get_latest_prices_func):
    """
    Mô hình Markowitz: Tối ưu hóa giữa lợi nhuận và rủi ro.
    
    Args:
    price_data (pd.DataFrame): Dữ liệu giá mã cổ phiếu
    total_investment (float): Tổng số tiền đầu tư
    get_latest_prices_func (function): Hàm lấy giá mã cổ phiếu mới nhất
        
    Returns:
        dict: Kết quả tối ưu hóa
    """
    logger.info(f"[MARKOWITZ] Nhan total_investment: {total_investment:,.0f} VND")
    
    tickers = price_data.columns.tolist()
    num_assets = len(tickers)

    if num_assets == 0:
        logger.error("Danh sach ma ma co phieu da chon khong hop le")
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
    
    logger.info(f"[MARKOWITZ] Truoc khi gan total_portfolio_value: {total_investment:,.0f} VND")
    total_portfolio_value = total_investment
    logger.info(f"[MARKOWITZ] Sau khi gan total_portfolio_value: {total_portfolio_value:,.0f} VND")
    
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
        "Số mã cổ phiếu cần mua": allocation_lp,
        "Số tiền còn lại": leftover_lp,
        "Giá mã cổ phiếu": latest_prices,
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
    logger.info(f"[MAX_SHARPE] Nhan total_investment: {total_investment:,.0f} VND")
    
    try:
        tickers = data.columns.tolist()
        num_assets = len(tickers)
        
        # Tính toán mean returns và covariance matrix
        mean_returns = expected_returns.mean_historical_return(data)
        cov_matrix = risk_models.sample_cov(data)

        # Tối ưu hóa Max Sharpe
        ef = EfficientFrontier(mean_returns, cov_matrix)
        weights = ef.max_sharpe()
        performance = ef.portfolio_performance(verbose=False)
        cleaned_weights = ef.clean_weights()
        
        # Đảm bảo trọng số được chuẩn hóa
        total_weight = sum(cleaned_weights.values())
        if abs(total_weight - 1.0) > 1e-5:
            logger.warning(f"[MAX_SHARPE] Tong trong so truoc khi chuan hoa: {total_weight}")
            cleaned_weights = {k: v / total_weight for k, v in cleaned_weights.items() if v > 1e-5}
            logger.info(f"[MAX_SHARPE] Da chuan hoa lai trong so. Tong moi: {sum(cleaned_weights.values())}")

        # Tạo 10,000 danh mục ngẫu nhiên để vẽ scatter plot
        log_ret = np.log(data / data.shift(1)).dropna()
        n_portfolios = 10000
        all_weights = np.zeros((n_portfolios, num_assets))
        ret_arr = np.zeros(n_portfolios)
        vol_arr = np.zeros(n_portfolios)
        sharpe_arr = np.zeros(n_portfolios)

        mean_returns_annual = log_ret.mean() * 252
        cov_matrix_annual = log_ret.cov() * 252

        np.random.seed(42)
        rf = 0.04  # Risk-free rate 4%

        for i in range(n_portfolios):
            w = np.random.random(num_assets)
            w /= np.sum(w)
            all_weights[i, :] = w

            ret_arr[i] = np.dot(mean_returns_annual, w)
            vol_arr[i] = np.sqrt(np.dot(w.T, np.dot(cov_matrix_annual, w)))
            sharpe_arr[i] = (ret_arr[i] - rf) / vol_arr[i]

        latest_prices = get_latest_prices_func(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        
        logger.info(f"[MAX_SHARPE] Truoc khi goi run_integer_programming:")
        logger.info(f"  - total_portfolio_value: {total_portfolio_value:,.0f} VND")
        logger.info(f"  - Tong trong so: {sum(cleaned_weights.values()):.6f}")
        
        allocation_lp, leftover_lp = run_integer_programming(
            cleaned_weights, 
            latest_prices_series, 
            total_portfolio_value
        )

        return {
            "Trọng số danh mục": cleaned_weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro (Độ lệch chuẩn)": performance[1],
            "Tỷ lệ Sharpe": performance[2],
            "Số mã cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá mã cổ phiếu": latest_prices,
            # Thêm dữ liệu cho biểu đồ Max Sharpe
            "ret_arr": ret_arr,
            "vol_arr": vol_arr,
            "sharpe_arr": sharpe_arr,
            "all_weights": all_weights,
            "risk_free_rate": rf
        }
    except Exception as e:
        logger.error(f"Loi trong mo hinh Max Sharpe: {e}")
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
    logger.info(f"[MIN_VOLATILITY] Nhan total_investment: {total_investment:,.0f} VND")
    
    try:
        tickers = data.columns.tolist()
        num_assets = len(tickers)
        
        mean_returns = expected_returns.mean_historical_return(data)
        cov_matrix = risk_models.sample_cov(data)

        # Tối ưu hóa Min Volatility
        ef = EfficientFrontier(mean_returns, cov_matrix)
        weights = ef.min_volatility()
        performance = ef.portfolio_performance(verbose=False)
        cleaned_weights = ef.clean_weights()
        
        # Đảm bảo trọng số được chuẩn hóa
        total_weight = sum(cleaned_weights.values())
        if abs(total_weight - 1.0) > 1e-5:
            logger.warning(f"[MIN_VOLATILITY] Tong trong so truoc khi chuan hoa: {total_weight}")
            cleaned_weights = {k: v / total_weight for k, v in cleaned_weights.items() if v > 1e-5}
            logger.info(f"[MIN_VOLATILITY] Da chuan hoa lai trong so. Tong moi: {sum(cleaned_weights.values())}")
        
        # Tối ưu hóa Max Sharpe để so sánh
        ef_sharpe = EfficientFrontier(mean_returns, cov_matrix)
        weights_sharpe = ef_sharpe.max_sharpe()
        performance_sharpe = ef_sharpe.portfolio_performance(verbose=False)
        cleaned_weights_sharpe = ef_sharpe.clean_weights()

        # Tạo 10,000 danh mục ngẫu nhiên để vẽ scatter plot
        log_ret = np.log(data / data.shift(1)).dropna()
        n_portfolios = 10000
        all_weights = np.zeros((n_portfolios, num_assets))
        ret_arr = np.zeros(n_portfolios)
        vol_arr = np.zeros(n_portfolios)
        sharpe_arr = np.zeros(n_portfolios)

        mean_returns_annual = log_ret.mean() * 252
        cov_matrix_annual = log_ret.cov() * 252

        np.random.seed(42)
        rf = 0.02  # Risk-free rate 2%

        for i in range(n_portfolios):
            w = np.random.random(num_assets)
            w /= np.sum(w)
            all_weights[i, :] = w

            ret_arr[i] = np.dot(mean_returns_annual, w)
            vol_arr[i] = np.sqrt(np.dot(w.T, np.dot(cov_matrix_annual, w)))
            sharpe_arr[i] = (ret_arr[i] - rf) / vol_arr[i]

        latest_prices = get_latest_prices_func(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        
        logger.info(f"[MIN_VOLATILITY] Truoc khi goi run_integer_programming:")
        logger.info(f"  - total_portfolio_value: {total_portfolio_value:,.0f} VND")
        logger.info(f"  - Tong trong so: {sum(cleaned_weights.values()):.6f}")
        
        allocation_lp, leftover_lp = run_integer_programming(
            cleaned_weights, 
            latest_prices_series, 
            total_portfolio_value
        )

        return {
            "Trọng số danh mục": cleaned_weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro (Độ lệch chuẩn)": performance[1],
            "Tỷ lệ Sharpe": performance[2],
            "Số mã cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá mã cổ phiếu": latest_prices,
            # Thêm dữ liệu cho biểu đồ Min Volatility
            "ret_arr": ret_arr,
            "vol_arr": vol_arr,
            "sharpe_arr": sharpe_arr,
            "all_weights": all_weights,
            "risk_free_rate": rf,
            # Thêm thông tin Max Sharpe để so sánh
            "max_sharpe_return": performance_sharpe[0],
            "max_sharpe_volatility": performance_sharpe[1],
            # Thêm trọng số thực sự của Min Volatility và Max Sharpe
            "min_vol_weights": cleaned_weights,
            "max_sharpe_weights": cleaned_weights_sharpe
        }
    except Exception as e:
        logger.error(f"Loi trong mo hinh Min Volatility: {e}")
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
    logger.info(f"[MIN_CVAR] Nhan total_investment: {total_investment:,.0f} VND")
    
    try:
        mean_returns = expected_returns.mean_historical_return(data)
        returns = expected_returns.returns_from_prices(data).dropna()

        cvar_optimizer = EfficientCVaR(mean_returns, returns, beta=beta)
        weights = cvar_optimizer.min_cvar()
        performance = cvar_optimizer.portfolio_performance()
        
        # Làm sạch và chuẩn hóa trọng số
        cleaned_weights = {k: v for k, v in weights.items() if v > 1e-5}
        total_weight = sum(cleaned_weights.values())
        if abs(total_weight - 1.0) > 1e-5:
            logger.warning(f"[MIN_CVAR] Tong trong so truoc khi chuan hoa: {total_weight}")
            cleaned_weights = {k: v / total_weight for k, v in cleaned_weights.items()}
            logger.info(f"[MIN_CVAR] Da chuan hoa lai trong so. Tong moi: {sum(cleaned_weights.values())}")
        
        # Lọc dữ liệu và ma trận hiệp phương sai theo các cổ phiếu trong cleaned_weights
        selected_tickers = list(cleaned_weights.keys())
        filtered_data = data[selected_tickers]
        cov_matrix = risk_models.sample_cov(filtered_data)

        # Tính độ lệch chuẩn của danh mục
        weights_array = np.array(list(cleaned_weights.values()))
        portfolio_std = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        rf = 0.02
        sharpe_ratio = (performance[0] - rf) / portfolio_std

        # Lấy giá chỉ cho các cổ phiếu trong cleaned_weights
        latest_prices = get_latest_prices_func(selected_tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        
        logger.info(f"[MIN_CVAR] Truoc khi goi run_integer_programming:")
        logger.info(f"  - total_portfolio_value: {total_portfolio_value:,.0f} VND")
        logger.info(f"  - Tong trong so: {sum(cleaned_weights.values()):.6f}")
        
        allocation_lp, leftover_lp = run_integer_programming(
            cleaned_weights, 
            latest_prices_series, 
            total_portfolio_value
        )

        return {
            "Trọng số danh mục": cleaned_weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro CVaR": performance[1],
            "Số mã cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá mã cổ phiếu": latest_prices,
            "Rủi ro (Độ lệch chuẩn)": portfolio_std,
            "Tỷ lệ Sharpe": sharpe_ratio
        }
    except Exception as e:
        logger.error(f"Loi trong mo hinh Min CVaR: {e}")
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
    logger.info(f"[MIN_CDAR] Nhan total_investment: {total_investment:,.0f} VND")
    
    try:
        mean_returns = expected_returns.mean_historical_return(data)
        returns = expected_returns.returns_from_prices(data).dropna()

        cdar_optimizer = EfficientCDaR(mean_returns, returns, beta=beta)
        weights = cdar_optimizer.min_cdar()
        performance = cdar_optimizer.portfolio_performance()
        
        # Làm sạch và chuẩn hóa trọng số
        cleaned_weights = {k: v for k, v in weights.items() if v > 1e-5}
        total_weight = sum(cleaned_weights.values())
        if abs(total_weight - 1.0) > 1e-5:
            logger.warning(f"[MIN_CDAR] Tong trong so truoc khi chuan hoa: {total_weight}")
            cleaned_weights = {k: v / total_weight for k, v in cleaned_weights.items()}
            logger.info(f"[MIN_CDAR] Da chuan hoa lai trong so. Tong moi: {sum(cleaned_weights.values())}")

        # Lọc dữ liệu và ma trận hiệp phương sai theo các cổ phiếu trong cleaned_weights
        selected_tickers = list(cleaned_weights.keys())
        filtered_data = data[selected_tickers]
        cov_matrix = risk_models.sample_cov(filtered_data)
        weights_array = np.array(list(cleaned_weights.values()))
        portfolio_std = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        rf = 0.02
        sharpe_ratio = (performance[0] - rf) / portfolio_std

        # Lấy giá chỉ cho các cổ phiếu trong cleaned_weights
        latest_prices = get_latest_prices_func(selected_tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        
        logger.info(f"[MIN_CDAR] Truoc khi goi run_integer_programming:")
        logger.info(f"  - total_portfolio_value: {total_portfolio_value:,.0f} VND")
        logger.info(f"  - Tong trong so: {sum(cleaned_weights.values()):.6f}")
        
        allocation_lp, leftover_lp = run_integer_programming(
            cleaned_weights, 
            latest_prices_series, 
            total_portfolio_value
        )

        return {
            "Trọng số danh mục": cleaned_weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro CDaR": performance[1],
            "Số mã cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá mã cổ phiếu": latest_prices,
            "Rủi ro (Độ lệch chuẩn)": portfolio_std,
            "Tỷ lệ Sharpe": sharpe_ratio
        }
    except Exception as e:
        logger.error(f"Loi trong mo hinh Min CDaR: {e}")
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
    logger.info(f"[HRP_MODEL] Nhan total_investment: {total_investment:,.0f} VND")
    
    try:
        returns = data.pct_change().dropna(how="all")
        hrp = HRPOpt(returns)
        weights = hrp.optimize(linkage_method="single")
        
        # Làm sạch weights và chuẩn hóa
        cleaned_weights = {k: v for k, v in weights.items() if v > 1e-5}
        total_weight = sum(cleaned_weights.values())
        if abs(total_weight - 1.0) > 1e-5:
            logger.warning(f"[HRP_MODEL] Tong trong so truoc khi chuan hoa: {total_weight}")
            cleaned_weights = {k: v / total_weight for k, v in cleaned_weights.items()}
            logger.info(f"[HRP_MODEL] Da chuan hoa lai trong so. Tong moi: {sum(cleaned_weights.values())}")
        
        performance = hrp.portfolio_performance()

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices_func(tickers)
        
        # Kiểm tra và log giá cổ phiếu
        logger.info(f"[HRP_MODEL] Gia co phieu: {latest_prices}")
        logger.info(f"[HRP_MODEL] Trong so: {cleaned_weights}")
        
        total_portfolio_value = total_investment
        
        logger.info(f"[HRP_MODEL] Truoc khi goi optimize_hrp_allocation:")
        logger.info(f"  - total_portfolio_value: {total_portfolio_value:,.0f} VND")
        logger.info(f"  - Tong trong so: {sum(cleaned_weights.values()):.6f}")
        
        # Sử dụng hàm tối ưu hóa HRP mới
        allocation_lp, leftover_lp = optimize_hrp_allocation(
            cleaned_weights, 
            latest_prices, 
            total_portfolio_value
        )

        return {
            "Trọng số danh mục": cleaned_weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro (Độ lệch chuẩn)": performance[1],
            "Tỷ lệ Sharpe": performance[2],
            "Số mã cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá mã cổ phiếu": latest_prices
        }
    except Exception as e:
        logger.error(f"Loi trong mo hinh HRP: {e}")
        print(f"Lỗi trong mô hình HRP: {e}")
        return None
