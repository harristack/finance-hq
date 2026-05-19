#!/usr/bin/env python3
"""
forecast.py — Curve forecasting for Finance HQ

Models:
  Prophet  — quarter-end basis widening seasonality
  XGBoost  — 1-day-ahead curve movement (falls back to sklearn GBM if libomp missing)

Flask endpoints (registered via register_forecast(app)):
  GET /api/forecast/quarterend   — seasonal basis forecast
  GET /api/forecast/curve        — next-day curve movement prediction

Graceful degradation: returns a progress message when < 30 days of snapshots exist.
"""
from __future__ import annotations
import warnings
import json
from datetime import datetime, timezone, timedelta, date

import numpy as np
import pandas as pd

# XGBoost with libomp fallback
try:
    import xgboost as xgb
    _XGB_OK = True
    _BOOSTER = f'xgboost {xgb.__version__}'
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor as _GBR
    _XGB_OK = False
    _BOOSTER = 'sklearn GradientBoostingRegressor (xgboost/libomp unavailable)'

import data_capture as dc

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*pystan.*')

MIN_DAYS_FORECAST = 30   # minimum days of snapshots before forecasting
MIN_ROWS_SNAPSHOT = 10   # minimum snapshot rows

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_snapshots() -> pd.DataFrame:
    df = dc._fetch_df('curve_snapshots', limit=100_000)
    if df.empty:
        return df
    df['ts']      = pd.to_datetime(df['ts'])
    df['date']    = df['ts'].dt.date
    df['curve']   = df['curve_json'].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    return df

def _extract_tenor(df: pd.DataFrame, tenor: str) -> pd.DataFrame:
    """Pull liveMid for a specific tenor out of each snapshot row."""
    records = []
    for _, row in df.iterrows():
        for r in row['curve']:
            if r.get('key') == tenor:
                records.append({'ds': row['ts'].normalize(), 'y': r.get('liveMid')})
                break
    out = pd.DataFrame(records).dropna()
    out['ds'] = pd.to_datetime(out['ds'])
    return out.groupby('ds')['y'].mean().reset_index()

def _check_sufficient(df: pd.DataFrame) -> dict | None:
    n = len(df)
    if n < MIN_ROWS_SNAPSHOT:
        return {
            'status':        'insufficient_data',
            'rows_have':     n,
            'rows_needed':   MIN_ROWS_SNAPSHOT,
            'days_needed':   MIN_DAYS_FORECAST,
            'message':       (
                f'Need {MIN_DAYS_FORECAST}+ days of curve snapshots '
                f'({n} snapshots logged so far). '
                'Snapshots auto-log every 30 min when the dashboard is open.'
            ),
        }
    span = (df['ts'].max() - df['ts'].min()).days
    if span < MIN_DAYS_FORECAST:
        return {
            'status':       'insufficient_data',
            'rows_have':    n,
            'days_have':    span,
            'days_needed':  MIN_DAYS_FORECAST,
            'message':      (
                f'Need {MIN_DAYS_FORECAST} days of history — '
                f'only {span} days logged so far. '
                f'Keep the dashboard open daily; forecasts unlock automatically.'
            ),
        }
    return None

# ── Prophet: quarter-end basis ────────────────────────────────────────────────

def _run_prophet(series: pd.DataFrame, periods: int = 90) -> dict:
    """
    Fit Prophet on a daily liveMid series.
    Returns forecast dict with yhat, yhat_lower, yhat_upper.
    """
    from prophet import Prophet  # lazy import — slow to load

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=5.0,
        interval_width=0.80,
    )
    # Quarter-end custom seasonality (91-day cycle ≈ quarter)
    m.add_seasonality(name='quarter', period=91.25, fourier_order=3)
    m.fit(series)

    future   = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    # Last `periods` rows only (future)
    fwd = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return {
        'tenor':       '3M',
        'model':       'Prophet',
        'horizon_days': periods,
        'forecast': [
            {
                'date':       row['ds'].strftime('%Y-%m-%d'),
                'yhat':       round(row['yhat'],       3),
                'yhat_lower': round(row['yhat_lower'], 3),
                'yhat_upper': round(row['yhat_upper'], 3),
            }
            for _, row in fwd.iterrows()
        ],
        'last_actual': round(float(series['y'].iloc[-1]), 3),
    }

# ── XGBoost / GBM: 1-day ahead curve movement ────────────────────────────────

_FORECAST_TENORS = ['1M', '3M', '6M', '12M']

def _make_features(series: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Lag features (t-1 … t-5) + day-of-week + day-of-month.
    Target: next-day change (liveMid[t+1] − liveMid[t]).
    """
    df = series.copy().reset_index(drop=True)
    df['change'] = df['y'].diff()

    rows = []
    for i in range(5, len(df) - 1):
        feats = [
            df['change'].iloc[i],
            df['change'].iloc[i-1],
            df['change'].iloc[i-2],
            df['change'].iloc[i-3],
            df['change'].iloc[i-4],
            df['ds'].iloc[i].dayofweek,
            df['ds'].iloc[i].day,
        ]
        target = df['change'].iloc[i+1]
        rows.append((feats, target))

    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    return X, y

def _predict_next_day(series: pd.DataFrame) -> dict:
    if len(series) < 10:
        return {'status': 'insufficient_series', 'points': len(series)}

    X, y = _make_features(series)
    if len(X) < 5:
        return {'status': 'insufficient_series', 'points': len(X)}

    # Train / last-point split
    X_tr, y_tr = X[:-1], y[:-1]
    X_pred      = X[-1:].copy()
    last_actual = float(series['y'].iloc[-1])

    if _XGB_OK:
        model = xgb.XGBRegressor(
            n_estimators=60, max_depth=3, learning_rate=0.1,
            subsample=0.8, verbosity=0, random_state=42,
        )
    else:
        model = _GBR(n_estimators=60, max_depth=3, learning_rate=0.1,
                     subsample=0.8, random_state=42)

    model.fit(X_tr, y_tr)
    pred_change = float(model.predict(X_pred)[0])

    # Rough CI: ±1 std of recent changes
    recent_std = float(np.std(y[-10:]))

    return {
        'last_actual':    round(last_actual, 3),
        'predicted_change': round(pred_change, 3),
        'predicted_value':  round(last_actual + pred_change, 3),
        'ci_lower':         round(last_actual + pred_change - recent_std, 3),
        'ci_upper':         round(last_actual + pred_change + recent_std, 3),
        'model':            _BOOSTER,
        'train_rows':       len(X_tr),
    }

# ── Public forecast functions ─────────────────────────────────────────────────

def forecast_quarterend() -> dict:
    df = _load_snapshots()
    insufficient = _check_sufficient(df)
    if insufficient:
        return insufficient

    series = _extract_tenor(df, '3M')
    if len(series) < 10:
        return {'status': 'insufficient_data',
                'message': 'Need more 3M data points in snapshots.'}

    try:
        result = _run_prophet(series)
        result['status'] = 'ok'
        result['generated_at'] = datetime.now(timezone.utc).isoformat()
        return result
    except Exception as exc:
        return {'status': 'error', 'error': str(exc)}


def forecast_curve() -> dict:
    df = _load_snapshots()
    insufficient = _check_sufficient(df)
    if insufficient:
        return insufficient

    predictions = {}
    for tenor in _FORECAST_TENORS:
        series = _extract_tenor(df, tenor)
        predictions[tenor] = _predict_next_day(series)

    return {
        'status':        'ok',
        'generated_at':  datetime.now(timezone.utc).isoformat(),
        'booster':       _BOOSTER,
        'horizon':       '1 trading day',
        'predictions':   predictions,
    }

# ── Flask registration ────────────────────────────────────────────────────────

def register_forecast(app):
    from flask import jsonify

    @app.route('/api/forecast/quarterend')
    def ep_forecast_qe():
        try:
            return jsonify(forecast_quarterend())
        except Exception as exc:
            import traceback
            return jsonify({'status': 'error', 'error': str(exc),
                            'trace': traceback.format_exc()}), 500

    @app.route('/api/forecast/curve')
    def ep_forecast_curve():
        try:
            return jsonify(forecast_curve())
        except Exception as exc:
            import traceback
            return jsonify({'status': 'error', 'error': str(exc),
                            'trace': traceback.format_exc()}), 500

# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print(f'=== forecast.py  booster={_BOOSTER} ===\n')

    # Check raw snapshot count
    raw = dc._fetch_df('curve_snapshots')
    print(f'Snapshots in DB: {len(raw)}')

    result_qe = forecast_quarterend()
    print(f'\n/api/forecast/quarterend → status={result_qe["status"]}')
    if result_qe['status'] == 'insufficient_data':
        print(f'  {result_qe["message"]}')
        print('\nSeeding 60 days of synthetic curve snapshots...')

        import random
        random.seed(7)
        base_date = datetime.now() - timedelta(days=60)
        spot = 149.85

        # Realistic curve shape: -1.6 pips/day interpolated per tenor
        tenor_rates = {
            '1M': -47.5, '3M': -147.5, '6M': -290.0, '12M': -578.0,
            '15M': -730.0, '18M': -875.0, '2Y': -1165.0,
        }
        for day in range(61):
            ts = base_date + timedelta(days=day)
            # Add realistic drift + noise
            drift = day * 0.3
            rows = []
            for k, base in tenor_rates.items():
                noise = random.gauss(0, 2.5)
                # Quarter-end widening: amplify mid-March, mid-June
                qe_bump = 0
                if ts.month in (3, 6, 9, 12) and 10 <= ts.day <= 25:
                    qe_bump = -random.uniform(2, 8)
                rows.append({'key': k, 'liveMid': round(base + drift + noise + qe_bump, 3),
                             'source': 'THEO'})
            dc._lock.acquire()
            try:
                dc._get().execute(
                    "INSERT INTO curve_snapshots VALUES (?,?,?,?,?)",
                    [ts, spot, 4.33, 0.492, json.dumps(rows)],
                )
            finally:
                dc._lock.release()

        print('Seeded 61 snapshots. Re-running forecasts...')
        result_qe = forecast_quarterend()

    print(f'\n/api/forecast/quarterend → status={result_qe["status"]}')
    if result_qe['status'] == 'ok':
        fc = result_qe['forecast']
        print(f'  tenor={result_qe["tenor"]}  horizon={result_qe["horizon_days"]}d  '
              f'last_actual={result_qe["last_actual"]}')
        print(f'  First 5 forecast days:')
        for row in fc[:5]:
            print(f'    {row["date"]}  yhat={row["yhat"]:>8.3f}  '
                  f'[{row["yhat_lower"]:>8.3f} .. {row["yhat_upper"]:>8.3f}]')

    result_cv = forecast_curve()
    print(f'\n/api/forecast/curve → status={result_cv["status"]}')
    if result_cv['status'] == 'ok':
        print(f'  booster={result_cv["booster"]}')
        for tenor, p in result_cv['predictions'].items():
            if p.get('status'):
                print(f'  {tenor:>4}: {p["status"]}')
            else:
                print(f'  {tenor:>4}: last={p["last_actual"]:>8.3f}  '
                      f'Δ={p["predicted_change"]:>+7.3f}  '
                      f'→ {p["predicted_value"]:>8.3f}  '
                      f'CI[{p["ci_lower"]:>8.3f} .. {p["ci_upper"]:>8.3f}]')
