#!/usr/bin/env python3
"""
intelligence.py — Counterparty intelligence from flow_log

Analyses:
  - Per-counterparty profile (size, tenor, direction, timing)
  - KMeans clustering by trading style
  - Time-of-day heatmap (JST hours)
  - Pre-event behaviour (3 calendar days before BOJ / FOMC)
  - Quarter-end basis pattern detection

Register with Flask via:  register_intelligence(app)
Standalone test:          python3 intelligence.py
"""
from __future__ import annotations
import json
from datetime import datetime, date, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import data_capture as dc

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_ROWS        = 10   # rows in flow_log before we do any analysis
MIN_CP_CLUSTER  = 3    # distinct counterparties needed for KMeans
N_CLUSTERS      = 3    # Morning-Asia / Corp-hedger / Spec-fund style split

# Approximate calendar days from spot for each tenor label
TENOR_DAYS: dict[str, int] = {
    'ON': 1, 'TN': 3, 'SN': 1,
    '1W': 7, '2W': 14, '3W': 21,
    '1M': 31, '2M': 63, '3M': 93, '4M': 123, '5M': 153,
    '6M': 185, '7M': 214, '8M': 246, '9M': 277,
    '10M': 304, '11M': 336, '12M': 365,
    '15M': 458, '18M': 550, '21M': 644, '2Y': 731,
}

# Upcoming BOJ / FOMC meeting dates (extend as needed)
_BOJ_DATES = [
    '2026-01-24', '2026-03-14', '2026-05-01', '2026-06-13',
    '2026-07-17', '2026-09-18', '2026-10-30', '2026-12-18',
    '2027-01-22', '2027-03-19', '2027-04-28', '2027-06-18',
]
_FOMC_DATES = [
    '2026-01-29', '2026-03-19', '2026-05-07', '2026-06-18',
    '2026-07-30', '2026-09-17', '2026-11-05', '2026-12-17',
    '2027-01-28', '2027-03-18', '2027-05-06', '2027-06-17',
]

_EVENT_DATES: list[date] = sorted(
    [datetime.strptime(d, '%Y-%m-%d').date() for d in _BOJ_DATES + _FOMC_DATES]
)

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_flow() -> pd.DataFrame:
    df = dc._fetch_df('flow_log')
    if df.empty:
        return df
    df['ts'] = pd.to_datetime(df['ts'])
    df['hour_jst'] = df['ts'].dt.hour
    df['date_jst'] = df['ts'].dt.date
    df['tenor_days'] = df['tenor'].map(TENOR_DAYS).fillna(31)
    df['is_buy'] = (df['direction'].str.lower() == 'buy').astype(int)
    df['is_gvn'] = (df['my_side'].str.upper() == 'GVN').astype(int)
    return df

# ── Per-counterparty profiles ─────────────────────────────────────────────────

def _build_profiles(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby('counterparty')
    profiles = pd.DataFrame({
        'trade_count':    g['ts'].count(),
        'avg_hour_jst':   g['hour_jst'].mean(),
        'tenor_days_avg': g['tenor_days'].mean(),
        'size_avg':       g['size'].mean(),
        'size_total':     g['size'].sum(),
        'direction_bias': g['is_buy'].mean(),   # 1.0 = always buy, 0.0 = always sell
        'gvn_ratio':      g['is_gvn'].mean(),
        'rate_avg':       g['rate'].mean(),
        'preferred_tenor': g['tenor'].agg(lambda x: x.value_counts().index[0]),
        'last_seen':      g['date_jst'].max(),
    }).reset_index()

    # Dominant activity window label
    def _session(h):
        if h < 8:   return 'Early-TKO'
        if h < 12:  return 'TKO-morning'
        if h < 15:  return 'TKO-afternoon'
        if h < 18:  return 'London'
        return 'NY'

    profiles['peak_session'] = profiles['avg_hour_jst'].apply(_session)
    return profiles

# ── KMeans clustering ─────────────────────────────────────────────────────────

_CLUSTER_LABELS = {
    0: 'Active-Speculator',
    1: 'Corporate-Hedger',
    2: 'Passive-Anchor',
}

def _cluster(profiles: pd.DataFrame) -> pd.DataFrame:
    features = ['avg_hour_jst', 'tenor_days_avg', 'size_avg',
                'direction_bias', 'gvn_ratio']
    X = profiles[features].fillna(0).values
    k = min(N_CLUSTERS, len(profiles))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    profiles = profiles.copy()
    profiles['cluster_id'] = km.fit_predict(Xs)

    # Label clusters by their centroid characteristics
    # Highest direction_bias → spec, lowest size_avg → passive
    centres = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=features
    )
    # Sort by size_avg: largest = active spec, middle = corp, smallest = passive
    order = centres['size_avg'].rank(ascending=False).astype(int) - 1
    profiles['cluster_label'] = profiles['cluster_id'].map(
        lambda i: _CLUSTER_LABELS.get(int(order[i]), f'Group-{i}')
    )
    return profiles

# ── Time-of-day heatmap ───────────────────────────────────────────────────────

def _heatmap(df: pd.DataFrame) -> dict:
    pivot = (
        df.groupby(['counterparty', 'hour_jst'])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all 24 hours present
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = 0
    pivot = pivot.sort_index(axis=1)
    return {
        'counterparties': pivot.index.tolist(),
        'hours': list(range(24)),
        'data': {cp: pivot.loc[cp].tolist() for cp in pivot.index},
    }

# ── Pre-event analysis ────────────────────────────────────────────────────────

def _pre_event(df: pd.DataFrame) -> dict:
    def _days_to_event(d: date) -> int | None:
        upcoming = [e for e in _EVENT_DATES if e > d]
        if not upcoming:
            return None
        return (upcoming[0] - d).days

    df = df.copy()
    df['days_to_event'] = df['date_jst'].apply(_days_to_event)
    pre  = df[df['days_to_event'].between(1, 3)]
    norm = df[df['days_to_event'] > 3]

    if pre.empty or norm.empty:
        return {'status': 'insufficient_event_data',
                'pre_event_rows': len(pre), 'normal_rows': len(norm)}

    summary = {
        'pre_event': {
            'rows':           len(pre),
            'size_avg':       round(pre['size'].mean(), 0),
            'direction_bias': round(pre['is_buy'].mean(), 3),
            'top_tenors':     pre['tenor'].value_counts().head(3).to_dict(),
            'gvn_ratio':      round(pre['is_gvn'].mean(), 3),
        },
        'normal': {
            'rows':           len(norm),
            'size_avg':       round(norm['size'].mean(), 0),
            'direction_bias': round(norm['is_buy'].mean(), 3),
            'top_tenors':     norm['tenor'].value_counts().head(3).to_dict(),
            'gvn_ratio':      round(norm['is_gvn'].mean(), 3),
        },
    }

    # Delta signals
    size_delta = (pre['size'].mean() - norm['size'].mean()) / max(norm['size'].mean(), 1)
    bias_delta = pre['is_buy'].mean() - norm['is_buy'].mean()
    summary['signals'] = {
        'size_uplift_pct':    round(size_delta * 100, 1),
        'direction_shift':    round(bias_delta, 3),
        'interpretation': (
            'Larger sizes pre-event (defensive hedging)' if size_delta > 0.15
            else 'Size unchanged pre-event'
        ),
    }
    return summary

# ── Quarter-end pattern ───────────────────────────────────────────────────────

def _quarter_end(df: pd.DataFrame) -> dict:
    def _is_qe(d: date) -> bool:
        return d.month in (3, 6, 9, 12) and d.day >= 15

    df = df.copy()
    df['is_qe'] = df['date_jst'].apply(_is_qe)
    qe   = df[df['is_qe']]
    nqe  = df[~df['is_qe']]

    if qe.empty or nqe.empty:
        return {'status': 'insufficient_quarter_end_data',
                'qe_rows': len(qe), 'non_qe_rows': len(nqe)}

    # Focus on 3M (most common quarter-end tenor for basis)
    qe_3m  = qe[qe['tenor'] == '3M']
    nqe_3m = nqe[nqe['tenor'] == '3M']

    rate_qe  = qe_3m['rate'].mean()  if not qe_3m.empty  else None
    rate_nqe = nqe_3m['rate'].mean() if not nqe_3m.empty else None

    return {
        'qe_period': {
            'rows':           len(qe),
            'size_avg':       round(qe['size'].mean(), 0),
            'direction_bias': round(qe['is_buy'].mean(), 3),
            'rate_3m_avg':    round(rate_qe,  3) if rate_qe  is not None else None,
        },
        'non_qe_period': {
            'rows':           len(nqe),
            'size_avg':       round(nqe['size'].mean(), 0),
            'direction_bias': round(nqe['is_buy'].mean(), 3),
            'rate_3m_avg':    round(rate_nqe, 3) if rate_nqe is not None else None,
        },
        'basis_widening_detected': bool(
            rate_qe is not None and rate_nqe is not None
            and abs(rate_qe - rate_nqe) > 3.0   # >3 pip differential
        ),
        'rate_delta_3m': (
            round(rate_qe - rate_nqe, 3)
            if (rate_qe is not None and rate_nqe is not None) else None
        ),
    }

# ── Main analysis entry point ─────────────────────────────────────────────────

def run_analysis() -> dict:
    df = _load_flow()
    n  = len(df)

    if n < MIN_ROWS:
        return {
            'status':       'insufficient_data',
            'rows_have':    n,
            'rows_needed':  MIN_ROWS,
            'message':      f'Need {MIN_ROWS} flow_log rows for analysis ({n} logged so far).',
        }

    profiles_df = _build_profiles(df)
    n_cp        = len(profiles_df)

    # Clustering (skip if too few counterparties)
    if n_cp >= MIN_CP_CLUSTER:
        profiles_df = _cluster(profiles_df)
        clustering_status = 'ok'
    else:
        profiles_df['cluster_id']    = 0
        profiles_df['cluster_label'] = 'Unclustered'
        clustering_status = f'skipped (need ≥{MIN_CP_CLUSTER} counterparties, have {n_cp})'

    # Serialise profiles (convert non-JSON-safe types)
    profiles_list = []
    for _, row in profiles_df.iterrows():
        profiles_list.append({
            'counterparty':   row['counterparty'],
            'trade_count':    int(row['trade_count']),
            'avg_hour_jst':   round(float(row['avg_hour_jst']), 1),
            'peak_session':   row['peak_session'],
            'tenor_days_avg': round(float(row['tenor_days_avg']), 0),
            'preferred_tenor': row['preferred_tenor'],
            'size_avg':       round(float(row['size_avg']), 0),
            'size_total':     round(float(row['size_total']), 0),
            'direction_bias': round(float(row['direction_bias']), 3),
            'gvn_ratio':      round(float(row['gvn_ratio']), 3),
            'rate_avg':       round(float(row['rate_avg']), 3),
            'cluster_id':     int(row['cluster_id']),
            'cluster_label':  row['cluster_label'],
            'last_seen':      str(row['last_seen']),
        })

    return {
        'status':              'ok',
        'generated_at':        datetime.now(timezone.utc).isoformat(),
        'flow_rows_analysed':  n,
        'counterparty_count':  n_cp,
        'clustering':          clustering_status,
        'profiles':            profiles_list,
        'heatmap':             _heatmap(df),
        'pre_event':           _pre_event(df),
        'quarter_end':         _quarter_end(df),
    }

# ── Flask registration ────────────────────────────────────────────────────────

def register_intelligence(app):
    from flask import jsonify

    @app.route('/api/intelligence/profiles')
    def ep_intelligence_profiles():
        try:
            return jsonify(run_analysis())
        except Exception as exc:
            import traceback
            return jsonify({'status': 'error', 'error': str(exc),
                            'trace': traceback.format_exc()}), 500


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print('=== intelligence.py standalone test ===\n')

    # Check current data
    result = run_analysis()
    if result['status'] == 'insufficient_data':
        print(f"Not enough data yet: {result['message']}")
        print(f"Seeding synthetic flow_log data for testing...\n")

        # Seed synthetic data
        import random
        random.seed(42)
        counterparties = [
            ('MUFG',     7,  10_000_000, 'buy',  'TKN', 'hedge'),
            ('SMBC',     8,   5_000_000, 'sell', 'GVN', 'corp'),
            ('NOMURA',   9,  20_000_000, 'sell', 'TKN', 'spec'),
            ('DAIWA',   10,  15_000_000, 'buy',  'GVN', 'hedge'),
            ('MIZUHO',  14,   8_000_000, 'sell', 'TKN', 'corp'),
        ]
        tenors = ['1M', '3M', '6M', '3M', '1M', '6M', '12M', '3M']

        for i in range(35):
            cp, base_h, base_sz, base_dir, base_side, tag = random.choice(counterparties)
            tenor = random.choice(tenors)
            days_back = random.randint(0, 90)
            # Sprinkle some into quarter-end (March 2026) and pre-event windows
            if i < 5:
                days_back = random.randint(180, 200)  # back in late 2025 / early 2026

            ts_fake = datetime.now() - timedelta(days=days_back,
                                                  hours=random.randint(-2, 2))
            rate = TENOR_DAYS.get(tenor, 31) * -1.6 + random.uniform(-5, 5)
            direction = base_dir if random.random() > 0.3 else ('sell' if base_dir == 'buy' else 'buy')
            my_side   = base_side if random.random() > 0.2 else ('GVN' if base_side == 'TKN' else 'TKN')
            size = base_sz * random.uniform(0.5, 2.0)

            dc._lock.acquire()
            try:
                dc._get().execute(
                    "INSERT INTO flow_log VALUES (?,?,?,?,?,?,?,?,?)",
                    [ts_fake.replace(tzinfo=None), cp, tenor, direction,
                     round(size, -6), round(rate, 3), my_side, tag, ''],
                )
            finally:
                dc._lock.release()

        print(f"Seeded 35 rows. Re-running analysis...\n")
        result = run_analysis()

    # Print summary
    print(f"Status:          {result['status']}")
    if result['status'] != 'ok':
        print(json.dumps(result, indent=2))
        sys.exit(0)

    print(f"Rows analysed:   {result['flow_rows_analysed']}")
    print(f"Counterparties:  {result['counterparty_count']}")
    print(f"Clustering:      {result['clustering']}")

    print('\n--- Counterparty Profiles ---')
    print(f"  {'Name':>10}  {'Trades':>6}  {'AvgHr':>6}  {'Session':>14}  "
          f"{'PrefTenor':>9}  {'AvgSize':>12}  {'DirBias':>8}  {'Cluster'}")
    print('  ' + '-' * 95)
    for p in result['profiles']:
        print(f"  {p['counterparty']:>10}  {p['trade_count']:>6}  "
              f"{p['avg_hour_jst']:>6.1f}  {p['peak_session']:>14}  "
              f"{p['preferred_tenor']:>9}  {p['size_avg']:>12,.0f}  "
              f"{p['direction_bias']:>8.3f}  {p['cluster_label']}")

    print('\n--- Heatmap (top active hours per counterparty) ---')
    hm = result['heatmap']
    for cp in hm['counterparties']:
        counts = hm['data'][cp]
        top = sorted(enumerate(counts), key=lambda x: -x[1])[:3]
        top_str = ', '.join(f'{h:02d}h({n})' for h, n in top if n > 0)
        print(f"  {cp:>10}: {top_str}")

    print('\n--- Pre-event Analysis ---')
    pe = result['pre_event']
    if pe.get('status'):
        print(f"  {pe['status']}")
    else:
        print(f"  Pre-event ({pe['pre_event']['rows']} trades):  "
              f"size={pe['pre_event']['size_avg']:,.0f}  "
              f"bias={pe['pre_event']['direction_bias']:.3f}")
        print(f"  Normal    ({pe['normal']['rows']} trades):  "
              f"size={pe['normal']['size_avg']:,.0f}  "
              f"bias={pe['normal']['direction_bias']:.3f}")
        print(f"  Signal: {pe['signals']['interpretation']}  "
              f"(size +{pe['signals']['size_uplift_pct']}%)")

    print('\n--- Quarter-End Pattern ---')
    qe = result['quarter_end']
    if qe.get('status'):
        print(f"  {qe['status']}")
    else:
        print(f"  QE period:     size={qe['qe_period']['size_avg']:,.0f}  "
              f"3M rate={qe['qe_period']['rate_3m_avg']}")
        print(f"  Non-QE period: size={qe['non_qe_period']['size_avg']:,.0f}  "
              f"3M rate={qe['non_qe_period']['rate_3m_avg']}")
        print(f"  Basis widening detected: {qe['basis_widening_detected']}  "
              f"delta={qe['rate_delta_3m']}")
