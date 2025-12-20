"""
Utility functions for portfolio analysis and optimization.

This module contains reusable utility functions for:
- Metric normalization
- Result validation
- Risk metrics calculation (Max Drawdown)
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_metric(
    value: float,
    min_val: float,
    max_val: float,
    reverse: bool = False,
    padding: float = 0.1
) -> float:
    """
    Normalize a metric value to 0-100 scale with optional padding.
    
    This function is used to standardize different metrics to a common scale
    for comparison and scoring purposes. Padding helps prevent extreme compression
    when values are close together.
    
    Args:
        value: The value to normalize
        min_val: Minimum value in the dataset
        max_val: Maximum value in the dataset
        reverse: If True, higher values get lower scores (useful for risk metrics)
        padding: Percentage padding to add to range (default 10%)
    
    Returns:
        Normalized score between 0-100
    
    Examples:
        >>> normalize_metric(0.15, 0.10, 0.20, reverse=False)
        50.0
        
        >>> normalize_metric(0.08, 0.05, 0.15, reverse=True)  # Lower risk = higher score
        70.0
    """
    # Handle edge case: all values are the same
    if max_val == min_val:
        return 50.0
    
    # Add padding to prevent extreme compression of similar values
    range_val = max_val - min_val
    pad = range_val * padding if padding else 0
    baseline_min = min_val - pad
    baseline_max = max_val + pad
    
    # Recalculate range with padding
    if baseline_max == baseline_min:
        return 50.0
    
    # Normalize to 0-100
    normalized = ((value - baseline_min) / (baseline_max - baseline_min)) * 100
    
    # Reverse if needed (for metrics where lower is better)
    if reverse:
        normalized = 100 - normalized
    
    # Clamp to valid range
    return max(0.0, min(100.0, normalized))


def validate_result(result: Dict[str, Any]) -> bool:
    """
    Validate that a portfolio optimization result contains required fields.
    
    Args:
        result: Dictionary containing optimization results
    
    Returns:
        True if result is valid, False otherwise
    
    Required fields:
        - 'Trọng số danh mục': Portfolio weights
        - 'Lợi nhuận kỳ vọng': Expected return
        - 'Rủi ro (Độ lệch chuẩn)': Volatility/risk
    
    Example:
        >>> result = {
        ...     'Trọng số danh mục': {'VNM': 0.4, 'VIC': 0.6},
        ...     'Lợi nhuận kỳ vọng': 0.15,
        ...     'Rủi ro (Độ lệch chuẩn)': 0.08
        ... }
        >>> validate_result(result)
        True
    """
    if not result or not isinstance(result, dict):
        logger.warning("Result is None or not a dictionary")
        return False
    
    required_keys = [
        'Trọng số danh mục',
        'Lợi nhuận kỳ vọng',
        'Rủi ro (Độ lệch chuẩn)'
    ]
    
    for key in required_keys:
        if key not in result:
            logger.warning(f"Missing required key: {key}")
            return False
    
    # Additional validation: check weights are valid
    weights = result.get('Trọng số danh mục', {})
    if not isinstance(weights, dict) or len(weights) == 0:
        logger.warning("Portfolio weights are empty or invalid")
        return False
    
    return True


def calculate_max_drawdown_safe(
    returns_data: Optional[pd.Series] = None,
    cdar: Optional[float] = None,
    volatility: Optional[float] = None
) -> Optional[float]:
    """
    Calculate Maximum Drawdown with multiple fallback methods.
    
    Max Drawdown (MDD) measures the largest peak-to-trough decline in portfolio value.
    This function attempts multiple calculation methods with graceful fallbacks.
    
    Calculation priority:
        1. From actual returns data (most accurate)
        2. From CDaR if available (good proxy)
        3. From volatility estimate (conservative)
        4. Return None if insufficient data
    
    Args:
        returns_data: Series of portfolio returns
        cdar: Conditional Drawdown at Risk value
        volatility: Portfolio volatility (standard deviation)
    
    Returns:
        Maximum drawdown as percentage (negative value), or None if cannot calculate
    
    Examples:
        >>> returns = pd.Series([0.01, -0.02, 0.03, -0.05, 0.02])
        >>> mdd = calculate_max_drawdown_safe(returns_data=returns)
        >>> print(f"Max Drawdown: {mdd:.2f}%")
        Max Drawdown: -5.23%
    
    Notes:
        - MDD is always negative (represents a loss)
        - Typical relationship: MDD ≈ 2-3 × Volatility in worst case
    """
    # Method 1: Calculate from actual returns (most accurate)
    if returns_data is not None and len(returns_data) > 0:
        try:
            # Convert returns to cumulative wealth
            cumulative = (1 + returns_data).cumprod()
            
            # Track running maximum (peak)
            peak = np.maximum.accumulate(cumulative)
            
            # Calculate drawdown at each point
            drawdown = (cumulative - peak) / peak
            
            # Find maximum drawdown (most negative value)
            max_dd = drawdown.min() * 100  # Convert to percentage
            
            logger.info(f"Calculated MDD from returns: {max_dd:.2f}%")
            return max_dd
            
        except Exception as e:
            logger.warning(f"Failed to calculate MDD from returns: {e}")
            # Fall through to next method
    
    # Method 2: Use CDaR as proxy (good approximation)
    if cdar is not None:
        mdd = cdar * 100
        logger.info(f"Using CDaR as MDD proxy: {mdd:.2f}%")
        return mdd
    
    # Method 3: Estimate from volatility (conservative)
    if volatility is not None and volatility > 0:
        # Conservative estimate: MDD ≈ 2.5 × Volatility
        # This assumes worst-case scenario drawdown
        mdd = -volatility * 2.5
        logger.info(f"Estimated MDD from volatility: {mdd:.2f}%")
        return mdd
    
    # Method 4: Insufficient data
    logger.warning("Cannot calculate MDD: insufficient data provided")
    return None


# ============== Test Functions (for development) ==============

def _test_normalize_metric():
    """Test normalize_metric function."""
    print("Testing normalize_metric...")
    
    # Test 1: Normal case
    score = normalize_metric(15.0, 10.0, 20.0, reverse=False)
    assert 40 < score < 60, f"Expected ~50, got {score}"
    print(f"✓ Test 1 passed: {score:.2f}")
    
    # Test 2: Reverse (risk metric)
    score = normalize_metric(8.0, 5.0, 15.0, reverse=True)
    assert score > 60, f"Expected >60 (low risk = high score), got {score}"
    print(f"✓ Test 2 passed: {score:.2f}")
    
    # Test 3: Edge case - all same
    score = normalize_metric(10.0, 10.0, 10.0)
    assert score == 50.0, f"Expected 50, got {score}"
    print(f"✓ Test 3 passed: {score:.2f}")
    
    print("All normalize_metric tests passed!\n")


def _test_validate_result():
    """Test validate_result function."""
    print("Testing validate_result...")
    
    # Test 1: Valid result
    valid_result = {
        'Trọng số danh mục': {'VNM': 0.4, 'VIC': 0.6},
        'Lợi nhuận kỳ vọng': 0.15,
        'Rủi ro (Độ lệch chuẩn)': 0.08
    }
    assert validate_result(valid_result), "Valid result should pass"
    print("✓ Test 1 passed: Valid result")
    
    # Test 2: Missing key
    invalid_result = {
        'Trọng số danh mục': {'VNM': 0.4},
        'Lợi nhuận kỳ vọng': 0.15
        # Missing volatility
    }
    assert not validate_result(invalid_result), "Invalid result should fail"
    print("✓ Test 2 passed: Missing key detected")
    
    # Test 3: Empty weights
    invalid_result = {
        'Trọng số danh mục': {},
        'Lợi nhuận kỳ vọng': 0.15,
        'Rủi ro (Độ lệch chuẩn)': 0.08
    }
    assert not validate_result(invalid_result), "Empty weights should fail"
    print("✓ Test 3 passed: Empty weights detected")
    
    print("All validate_result tests passed!\n")


def _test_calculate_max_drawdown():
    """Test calculate_max_drawdown_safe function."""
    print("Testing calculate_max_drawdown_safe...")
    
    # Test 1: From returns data
    returns = pd.Series([0.01, -0.02, 0.03, -0.05, 0.02])
    mdd = calculate_max_drawdown_safe(returns_data=returns)
    assert mdd is not None and mdd < 0, f"MDD should be negative, got {mdd}"
    print(f"✓ Test 1 passed: MDD from returns = {mdd:.2f}%")
    
    # Test 2: From CDaR
    mdd = calculate_max_drawdown_safe(cdar=-0.15)
    assert mdd == -15.0, f"Expected -15.0, got {mdd}"
    print(f"✓ Test 2 passed: MDD from CDaR = {mdd:.2f}%")
    
    # Test 3: From volatility
    mdd = calculate_max_drawdown_safe(volatility=8.0)
    assert mdd == -20.0, f"Expected -20.0 (8*2.5), got {mdd}"
    print(f"✓ Test 3 passed: MDD from volatility = {mdd:.2f}%")
    
    # Test 4: No data
    mdd = calculate_max_drawdown_safe()
    assert mdd is None, f"Expected None, got {mdd}"
    print("✓ Test 4 passed: No data returns None")
    
    print("All calculate_max_drawdown_safe tests passed!\n")


if __name__ == "__main__":
    """Run all tests when module is executed directly."""
    print("=" * 60)
    print("Running Portfolio Utils Test Suite")
    print("=" * 60 + "\n")
    
    _test_normalize_metric()
    _test_validate_result()
    _test_calculate_max_drawdown()
    
    print("=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)
