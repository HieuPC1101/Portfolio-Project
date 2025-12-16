"""Compatibility layer that re-exports the modular data helpers."""

from utils.config import (
    ANALYSIS_START_DATE,
    ANALYSIS_END_DATE,
)
from data_process.fetchers import (
    fetch_data_from_csv,
    create_vnstock_instance,
    fetch_stock_data2,
    get_latest_prices,
    fetch_ohlc_data,
    get_index_history,
    get_sector_snapshot,
    get_realtime_index_board,
)
from data_process.fundamentals import (
    fetch_fundamental_data,
    fetch_fundamental_data_batch,
)
from data_process.processors import (
    INDEX_LABELS,
    DEFAULT_INDEX_SYMBOLS,
    get_indices_history,
    get_market_indices_metrics,
    summarize_sector_performance,
    summarize_market_cap_distribution,
    get_foreign_flow_leaderboard,
    get_liquidity_leaders,
    get_sector_heatmap_matrix,
)
from data_process.quant import (
    calculate_metrics,
    get_return_correlation_matrix,
)

__all__ = [
    'fetch_data_from_csv',
    'create_vnstock_instance',
    'fetch_stock_data2',
    'get_latest_prices',
    'fetch_ohlc_data',
    'get_index_history',
    'get_sector_snapshot',
    'get_realtime_index_board',
    'fetch_fundamental_data',
    'fetch_fundamental_data_batch',
    'INDEX_LABELS',
    'DEFAULT_INDEX_SYMBOLS',
    'get_indices_history',
    'get_market_indices_metrics',
    'summarize_sector_performance',
    'summarize_market_cap_distribution',
    'get_foreign_flow_leaderboard',
    'get_liquidity_leaders',
    'get_sector_heatmap_matrix',
    'calculate_metrics',
    'get_return_correlation_matrix'
]
