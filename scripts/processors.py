"""Data processing helpers powering dashboard visualizations."""

from typing import Iterable, List, Optional

import pandas as pd

from scripts.fetchers import get_index_history

INDEX_LABELS = {
    "VNINDEX": "VN-Index",
    "VN30": "VN30",
    "HNXINDEX": "HNX-Index",
    "HNX30": "HNX30",
    "UPCOMINDEX": "UPCoM",
    "UPCOM": "UPCoM",
}
DEFAULT_INDEX_SYMBOLS = ("VNINDEX", "HNXINDEX", "UPCOMINDEX")


def _safe_pct_change(current: float, previous: float) -> float:
    if previous in (0, None):
        return 0.0
    try:
        return (current - previous) / previous * 100
    except Exception:
        return 0.0


def get_indices_history(symbols: Iterable[str] = DEFAULT_INDEX_SYMBOLS, months: int = 6,
                        source: str = "VCI", start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
    """Return long-format historical quotes for a list of indices."""
    frames = []
    for symbol in symbols:
        df = get_index_history(symbol, start_date=start_date, end_date=end_date,
                                months=months, source=source)
        if df.empty:
            continue
        df = df.copy()
        df['display'] = INDEX_LABELS.get(str(symbol).upper(), symbol)
        frames.append(df[['time', 'close', 'display']])

    if not frames:
        return pd.DataFrame(columns=['time', 'close', 'symbol'])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('time')
    combined = combined.rename(columns={'display': 'symbol'})
    return combined


def get_market_indices_metrics(symbols: Iterable[str] = DEFAULT_INDEX_SYMBOLS,
                               source: str = "VCI") -> List[dict]:
    """Prepare KPI cards for headline indices."""
    metrics: List[dict] = []
    for symbol in symbols:
        history = get_index_history(symbol, months=1, source=source)
        label = INDEX_LABELS.get(str(symbol).upper(), symbol)

        if history.empty:
            metrics.append({
                'symbol': symbol,
                'label': label,
                'value': None,
                'change': None,
                'pct_change': None,
                'note': 'Không có dữ liệu',
                'timestamp': None,
            })
            continue

        history = history.sort_values('time')
        last_row = history.iloc[-1]
        prev_row = history.iloc[-2] if len(history) > 1 else last_row
        change = last_row['close'] - prev_row['close']
        pct_change = _safe_pct_change(last_row['close'], prev_row['close'])

        volume = last_row.get('volume', None)
        note = f"KL {volume / 1_000_000:.1f} triệu cp" if pd.notna(volume) else "Chưa có số liệu thanh khoản"

        metrics.append({
            'symbol': symbol,
            'label': label,
            'value': last_row['close'],
            'change': change,
            'pct_change': pct_change,
            'note': note,
            'timestamp': last_row['time'],
        })

    return metrics


def summarize_sector_performance(snapshot: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
    """Aggregate sector-level growth and liquidity stats."""
    if snapshot is None or snapshot.empty:
        return pd.DataFrame(columns=[
            'industry', 'avg_growth_1w', 'avg_growth_1m', 'avg_liquidity', 'market_cap',
            'delta_growth_1w', 'delta_growth_1m', 'market_avg_1w', 'market_avg_1m'
        ])

    required_columns = {
        'price_growth_1w': pd.NA,
        'price_growth_1m': pd.NA,
        'avg_trading_value_20d': pd.NA,
        'market_cap': pd.NA,
    }

    # Guard against upstream collectors missing the derived growth columns.
    working = snapshot.copy()
    for column, default_value in required_columns.items():
        if column not in working.columns:
            working[column] = default_value

    numeric_growth_cols = ['price_growth_1w', 'price_growth_1m']
    for column in numeric_growth_cols:
        working[column] = pd.to_numeric(working[column], errors='coerce')

    market_avg_1w = working['price_growth_1w'].mean(skipna=True)
    market_avg_1m = working['price_growth_1m'].mean(skipna=True)
    market_avg_1w = float(market_avg_1w) if pd.notna(market_avg_1w) else 0.0
    market_avg_1m = float(market_avg_1m) if pd.notna(market_avg_1m) else 0.0

    group = working.groupby('industry').agg({
        'price_growth_1w': 'mean',
        'price_growth_1m': 'mean',
        'avg_trading_value_20d': 'mean',
        'market_cap': 'sum'
    }).reset_index()

    group = group.rename(columns={
        'price_growth_1w': 'avg_growth_1w',
        'price_growth_1m': 'avg_growth_1m',
        'avg_trading_value_20d': 'avg_liquidity'
    })

    group['avg_growth_1w'] = pd.to_numeric(group['avg_growth_1w'], errors='coerce')
    group['avg_growth_1m'] = pd.to_numeric(group['avg_growth_1m'], errors='coerce')

    group['delta_growth_1w'] = group['avg_growth_1w'] - market_avg_1w
    group['delta_growth_1m'] = group['avg_growth_1m'] - market_avg_1m
    group['market_avg_1w'] = market_avg_1w
    group['market_avg_1m'] = market_avg_1m

    group = group.sort_values('avg_growth_1m', ascending=True)

    if top_n is None:
        return group

    limit = max(top_n, 1)
    return group.head(limit)


def summarize_market_cap_distribution(snapshot: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Prepare treemap weights by sector market cap."""
    if snapshot is None or snapshot.empty:
        return pd.DataFrame(columns=['industry', 'market_cap', 'weight'])

    agg = snapshot.groupby('industry')['market_cap'].sum().reset_index()
    agg = agg.sort_values('market_cap', ascending=False).head(max(top_n, 1))
    total = agg['market_cap'].sum()
    agg['weight'] = agg['market_cap'] / total if total else 0
    return agg


def get_foreign_flow_leaderboard(snapshot: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Rank tickers by net foreign flow."""
    if snapshot is None or snapshot.empty or 'foreign_buysell_20s' not in snapshot.columns:
        return pd.DataFrame(columns=['ticker', 'industry', 'foreign_buysell_20s', 'avg_trading_value_20d'])

    df = snapshot[['ticker', 'industry', 'foreign_buysell_20s', 'avg_trading_value_20d']].copy()
    df['foreign_buysell_20s'] = pd.to_numeric(df['foreign_buysell_20s'], errors='coerce').fillna(0)
    df['avg_trading_value_20d'] = pd.to_numeric(df['avg_trading_value_20d'], errors='coerce').fillna(0)
    df = df.sort_values('foreign_buysell_20s', ascending=False)

    inflow = df.head(max(top_n, 1))
    outflow = df.tail(max(top_n, 1)).sort_values('foreign_buysell_20s')
    return pd.concat([inflow, outflow], ignore_index=True)


def get_liquidity_leaders(snapshot: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Return the most liquid tickers for scatter plots."""
    if snapshot is None or snapshot.empty:
        return pd.DataFrame(columns=['ticker', 'industry', 'avg_trading_value_20d', 'price_growth_1m', 'market_cap'])

    cols = ['ticker', 'industry', 'avg_trading_value_20d', 'price_growth_1m', 'market_cap']
    available_cols = [col for col in cols if col in snapshot.columns]
    data = snapshot[available_cols].copy()
    data['avg_trading_value_20d'] = pd.to_numeric(data.get('avg_trading_value_20d'), errors='coerce')
    data['price_growth_1m'] = pd.to_numeric(data.get('price_growth_1m'), errors='coerce')
    data['market_cap'] = pd.to_numeric(data.get('market_cap'), errors='coerce')
    data = data.dropna(subset=['avg_trading_value_20d', 'price_growth_1m'])
    data = data.sort_values('avg_trading_value_20d', ascending=False).head(max(top_n, 1))
    return data


def get_sector_heatmap_matrix(snapshot: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Pivot sector performance metrics for heatmap visuals."""
    performance = summarize_sector_performance(snapshot, top_n=top_n)
    if performance.empty:
        return pd.DataFrame(columns=['industry', 'metric', 'value'])

    melted = performance.melt(
        id_vars='industry',
        value_vars=['avg_growth_1w', 'avg_growth_1m', 'avg_liquidity'],
        var_name='metric',
        value_name='value'
    )
    metric_labels = {
        'avg_growth_1w': 'Tăng trưởng 1W (%)',
        'avg_growth_1m': 'Tăng trưởng 1M (%)',
        'avg_liquidity': 'Thanh khoản TB 20 phiên (tỷ)'
    }
    melted['metric'] = melted['metric'].map(lambda key: metric_labels.get(key, key))
    return melted


__all__ = [
    'INDEX_LABELS', 'DEFAULT_INDEX_SYMBOLS', 'get_indices_history', 'get_market_indices_metrics',
    'summarize_sector_performance', 'summarize_market_cap_distribution',
    'get_foreign_flow_leaderboard', 'get_liquidity_leaders', 'get_sector_heatmap_matrix'
]
