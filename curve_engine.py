#!/usr/bin/env python3
"""
curve_engine.py — USD/JPY forward curve builder using QuantLib
Flask API on port 5051

Endpoints:
  POST /api/curve            — calculate full forward curve
  GET  /api/health           — health check
  POST /api/log/price        — log per-tenor prices to DuckDB
  POST /api/log/snapshot     — log full curve snapshot to DuckDB
  POST /api/log/trade        — log executed trade to DuckDB
  POST /api/log/flow         — log counterparty flow to DuckDB
  GET  /api/data/export      — export table as CSV or JSON
                               ?table=price_log&format=csv
"""
from __future__ import annotations
import math
from datetime import date
from flask import Flask, request, jsonify, Response
import QuantLib as ql
import data_capture as dc
import intelligence as intel
import forecast as fc_mod

app = Flask(__name__)

# ── Calendars ────────────────────────────────────────────────────────────────
_TK  = ql.Japan()
_NY  = ql.UnitedStates(ql.UnitedStates.FederalReserve)
_CAL = ql.JointCalendar(_TK, _NY)   # holiday if EITHER market is closed
MF   = ql.ModifiedFollowing

def _ql(d: date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)

def _py(d: ql.Date) -> date:
    return date(d.year(), d.month(), d.dayOfMonth())

def spot_date(today: ql.Date) -> ql.Date:
    """T+2 using joint TK+NY calendar (advance 2 good business days)."""
    return _CAL.advance(today, 2, ql.Days, ql.Following)

def next_good(d: ql.Date) -> ql.Date:
    return _CAL.advance(d, 1, ql.Days, ql.Following)

def add_months(base: ql.Date, n: int) -> ql.Date:
    return _CAL.advance(base, n, ql.Months, MF)

def add_weeks(base: ql.Date, n: int) -> ql.Date:
    return _CAL.advance(base, n, ql.Weeks, MF)

def biz_days(d1: ql.Date, d2: ql.Date) -> int:
    """Calendar days between two QL dates (consistent with JS implementation)."""
    return int(d2 - d1)

# ── Tenor schedule ────────────────────────────────────────────────────────────
def build_tenors(today: ql.Date, spot: ql.Date) -> list[dict]:
    T: list[dict] = []

    on_far = next_good(today)
    T.append({'key': 'ON', 'label': 'O/N', 'near': today, 'far': on_far,
              'd': biz_days(today, on_far)})

    tom = next_good(today)
    T.append({'key': 'TN', 'label': 'T/N', 'near': tom, 'far': spot,
              'd': max(1, biz_days(tom, spot))})

    sn_far = next_good(spot)
    T.append({'key': 'SN', 'label': 'S/N', 'near': spot, 'far': sn_far,
              'd': max(1, biz_days(spot, sn_far))})

    for n, k in [(1, '1W'), (2, '2W'), (3, '3W')]:
        far = add_weeks(spot, n)
        T.append({'key': k, 'label': k, 'near': spot, 'far': far,
                  'd': biz_days(spot, far)})

    for m in range(1, 13):
        far = add_months(spot, m)
        T.append({'key': f'{m}M', 'label': f'{m}M', 'near': spot, 'far': far,
                  'd': biz_days(spot, far)})

    for m, k in [(15, '15M'), (18, '18M'), (21, '21M'), (24, '2Y')]:
        far = add_months(spot, m)
        T.append({'key': k, 'label': k, 'near': spot, 'far': far,
                  'd': biz_days(spot, far)})

    return T

# ── Interpolation ─────────────────────────────────────────────────────────────
def log_lin(d: float, d0: float, d1: float, v0: float, v1: float) -> float:
    if d0 <= 0 or d1 <= 0 or d0 == d1: return v0
    if d <= d0: return v0
    if d >= d1: return v1
    return v0 + math.log(d / d0) / math.log(d1 / d0) * (v1 - v0)

def interp_adj(d: float, anchors: list[tuple]) -> float:
    if not anchors: return 0.0
    s = sorted(anchors)
    if d <= s[0][0]:  return s[0][1]
    if d >= s[-1][0]: return s[-1][1]
    for i in range(len(s) - 1):
        if s[i][0] <= d <= s[i+1][0]:
            return log_lin(d, s[i][0], s[i+1][0], s[i][1], s[i+1][1])
    return 0.0

# ── Parse helper ──────────────────────────────────────────────────────────────
def _num(v) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

# ── OIS Step-Function Curve ───────────────────────────────────────────────────

def _parse_ql_date(s: str) -> ql.Date:
    from datetime import date as _date
    d = _date.fromisoformat(s)
    return ql.Date(d.day, d.month, d.year)


class StepCurve:
    """
    Piecewise-constant O/N rate curve built from meeting-dated OIS.
    Each segment (start_date, rate_decimal) applies flat until next segment.
    """
    def __init__(self, today: ql.Date, segments: list[tuple], day_count: int = 365):
        self.today     = today
        self.segs      = sorted(segments, key=lambda x: x[0])
        self.day_count = day_count

    def df(self, target: ql.Date) -> float:
        if target <= self.today:
            return 1.0
        acc  = 1.0
        segs = self.segs
        n    = len(segs)
        for i, (s_start, s_rate) in enumerate(segs):
            seg_end = segs[i + 1][0] if i + 1 < n else target
            t_start = s_start if s_start > self.today else self.today
            t_end   = min(seg_end, target)
            if t_end <= t_start:
                continue
            days = int(t_end - t_start)
            if days > 0:
                acc *= (1.0 + s_rate / self.day_count) ** days
            if t_end >= target:
                break
        return acc

    def zero_pct(self, target: ql.Date) -> float:
        """Simple-interest zero rate (%) from today to target."""
        if target <= self.today:
            return self.segs[0][1] * 100.0 if self.segs else 0.0
        df_val  = self.df(target)
        t_years = int(target - self.today) / self.day_count
        if t_years <= 0:
            return 0.0
        return (df_val - 1.0) / t_years * 100.0


def _build_step_curve(today: ql.Date, on_fallback: float,
                      inputs: dict, rate_key: str,
                      day_count: int = 365) -> StepCurve:
    """
    Generic step-curve builder for JPY (rate_key='mpmRates') or
    USD (rate_key='fomcRates').
    inputs: {ois1m?, meetings:[ISO], <rate_key>:[pct], ois1y?, ois2y?}
    """
    ois1m     = _num(inputs.get('ois1m'))
    meetings  = inputs.get('meetings', [])
    mt_rates  = inputs.get(rate_key, [])

    pre_rate = (ois1m if ois1m is not None else on_fallback) / 100.0
    segs: list[tuple] = [(today, pre_rate)]

    for d_str, r in zip(meetings, mt_rates):
        try:
            segs.append((_parse_ql_date(d_str), float(r) / 100.0))
        except Exception:
            pass

    ois1y = _num(inputs.get('ois1y'))
    ois2y = _num(inputs.get('ois2y'))
    if ois1y is not None:
        segs.append((_CAL.advance(today, 1, ql.Years, MF), ois1y / 100.0))
    if ois2y is not None:
        segs.append((_CAL.advance(today, 2, ql.Years, MF), ois2y / 100.0))

    return StepCurve(today, segs, day_count=day_count)


def _interp_basis(d: int, anchors: list) -> float:
    """Linear interpolation of XCCY basis (bps) for given day count."""
    if not anchors:
        return 0.0
    pts = sorted(anchors, key=lambda x: x[0])
    if d <= pts[0][0]:
        return float(pts[0][1])
    if d >= pts[-1][0]:
        return float(pts[-1][1])
    for i in range(len(pts) - 1):
        d0, v0 = pts[i][0], pts[i][1]
        d1, v1 = pts[i + 1][0], pts[i + 1][1]
        if d0 <= d <= d1:
            return v0 + (v1 - v0) * (d - d0) / (d1 - d0)
    return 0.0


# ── Core calculation ──────────────────────────────────────────────────────────
def calculate_curve(spot_fx: float, sofr: float, tona: float,
                    market_quotes: dict,
                    jpy_ois: dict | None = None,
                    usd_ois: dict | None = None,
                    xccy_basis: dict | None = None) -> dict:
    """
    Forward pts formula:
      theo = spot × (r_jpy_eff − r_usd) × days / 360   [pips, 0.01 JPY]
      outright = spot + liveMid / 100

    When jpyOis / usdOis are supplied, r_jpy / r_usd are OIS zero rates
    from step-function curves (meeting-dated); otherwise falls back to
    flat TONA / SOFR.  xccyBasis adds basis (bps) to r_jpy.

    Interpolation priority: MKT → INTERP (log-lin between MKT anchors)
                             → NYC+adj → THEO+adj  (+EXTRAP flag)
    """
    today  = _ql(date.today())
    spot_q = spot_date(today)
    tenors = build_tenors(today, spot_q)

    jpy_curve     = _build_step_curve(today, tona, jpy_ois, 'mpmRates',  day_count=365) if jpy_ois else None
    usd_curve     = _build_step_curve(today, sofr, usd_ois, 'fomcRates', day_count=360) if usd_ois else None
    basis_anchors = xccy_basis.get('anchors', []) if xccy_basis else []

    rows: list[dict] = []
    adj_anchors: list[tuple] = []

    # ── Pass 1: theo + market inputs ──────────────────────────────────────────
    for t in tenors:
        d = t['d']

        r_jpy      = jpy_curve.zero_pct(t['far']) if jpy_curve else tona
        r_usd      = usd_curve.zero_pct(t['far']) if usd_curve else sofr
        basis_bps  = _interp_basis(d, basis_anchors)
        r_jpy_eff  = r_jpy + basis_bps / 100.0
        theo       = spot_fx * (r_jpy_eff - r_usd) * d / 360

        q      = market_quotes.get(t['key'], {})
        nyc    = _num(q.get('nyc'))
        bid    = _num(q.get('bid'))
        offer  = _num(q.get('offer'))
        mkt_mid = (bid + offer) / 2 if (bid is not None and offer is not None) else None

        adj = (mkt_mid - (nyc if nyc is not None else theo)) if mkt_mid is not None else None
        if adj is not None:
            adj_anchors.append((d, adj))

        rows.append({
            'key':       t['key'],
            'label':     t['label'],
            'valueDate': _py(t['far']).isoformat(),
            'days':      d,
            'theo':      round(theo,       4),
            'rJpy':      round(r_jpy_eff,  4),
            'rUsd':      round(r_usd,      4),
            'basisBps':  round(basis_bps,  4) if basis_bps else None,
            'nyc':       nyc,
            'bid':       bid,
            'offer':     offer,
            'mktMid':    round(mkt_mid, 4) if mkt_mid is not None else None,
            'adj':       round(adj,     4) if adj     is not None else None,
        })

    # ── MKT anchor list sorted by days ────────────────────────────────────────
    mkt     = sorted([r for r in rows if r['mktMid'] is not None], key=lambda x: x['days'])
    mkt_min = mkt[0]['days']  if mkt else float('inf')
    mkt_max = mkt[-1]['days'] if mkt else float('-inf')

    # ── Pass 2: liveMid + source ──────────────────────────────────────────────
    for r in rows:
        d = r['days']

        if r['mktMid'] is not None:
            r.update(liveMid=r['mktMid'], source='MKT', extrap=False, interpBracket=None)

        else:
            # Strict MKT interpolation (log-linear, bracketed only)
            lo = hi = None
            for i in range(len(mkt) - 1):
                if mkt[i]['days'] < d < mkt[i+1]['days']:
                    lo, hi = mkt[i], mkt[i+1]
                    break

            if lo and hi:
                live = round(log_lin(d, lo['days'], hi['days'],
                                     lo['mktMid'], hi['mktMid']), 4)
                r.update(
                    liveMid=live, source='INTERP', extrap=False,
                    interpBracket={'nearLabel': lo['label'], 'nearDays': lo['days'],
                                   'farLabel':  hi['label'], 'farDays':  hi['days']},
                )
            else:
                if r['adj'] is None:
                    r['adj'] = round(interp_adj(d, adj_anchors), 4)
                extrap = bool(mkt) and (d < mkt_min or d > mkt_max)
                base   = r['nyc'] if r['nyc'] is not None else r['theo']
                live   = round(base + (r['adj'] or 0.0), 4)
                src    = 'NYC' if r['nyc'] is not None else 'THEO'
                r.update(liveMid=live, source=src, extrap=extrap, interpBracket=None)

        r['outright'] = round(spot_fx + r['liveMid'] / 100, 4)
        r['deltaNyc'] = round(r['liveMid'] - r['nyc'], 4) if r['nyc'] is not None else None

    curve_mode = 'OIS' if (jpy_curve or usd_curve) else 'FLAT'
    return {
        'ok':        True,
        'today':     date.today().isoformat(),
        'spotDate':  _py(spot_q).isoformat(),
        'spot':      spot_fx,
        'sofr':      sofr,
        'tona':      tona,
        'curveMode': curve_mode,
        'rows':      rows,
    }

# ── Flask endpoints ───────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({
        'status':     'ok',
        'service':    'curve_engine',
        'ql_version': ql.__version__,
        'today':      date.today().isoformat(),
        'spotDate':   _py(spot_date(_ql(date.today()))).isoformat(),
        'calendar':   'Japan + FederalReserve joint',
    })

@app.route('/api/curve', methods=['POST'])
def curve():
    try:
        body    = request.get_json(force=True) or {}
        spot_fx = float(body.get('spot',   145.0))
        sofr    = float(body.get('sofr',   4.33))
        tona    = float(body.get('tona',   0.492))
        mq      = body.get('marketQuotes', {})
        jpy_ois = body.get('jpyOis')
        usd_ois = body.get('usdOis')
        xccy    = body.get('xccyBasis')
        return jsonify(calculate_curve(spot_fx, sofr, tona, mq, jpy_ois, usd_ois, xccy))
    except Exception as exc:
        import traceback
        return jsonify({'ok': False, 'error': str(exc),
                        'trace': traceback.format_exc()}), 400

# ── Data-capture endpoints ────────────────────────────────────────────────────

@app.route('/api/log/price', methods=['POST'])
def ep_log_price():
    """
    Single tenor:  {spot, tenor, bid?, offer?, mid?, session?}
    Bulk (array):  {spot, session?, prices:[{tenor,bid?,offer?,mid?},...]}
    """
    try:
        body    = request.get_json(force=True) or {}
        spot    = float(body.get('spot', 0))
        session = str(body.get('session', ''))
        prices  = body.get('prices')
        if prices and isinstance(prices, list):
            n = dc.log_prices_bulk(prices, spot, session)
            return jsonify({'ok': True, 'logged': n})
        else:
            dc.log_price(
                tenor   = str(body.get('tenor', '')),
                bid     = _num(body.get('bid')),
                offer   = _num(body.get('offer')),
                mid     = _num(body.get('mid')),
                spot    = spot,
                session = session,
            )
            return jsonify({'ok': True, 'logged': 1})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/log/snapshot', methods=['POST'])
def ep_log_snapshot():
    """
    {spot, sofr, tona, rows:[...]}  — rows from /api/curve response
    """
    try:
        body = request.get_json(force=True) or {}
        dc.log_snapshot(
            spot = float(body.get('spot', 0)),
            sofr = float(body.get('sofr', 0)),
            tona = float(body.get('tona', 0)),
            rows = body.get('rows', []),
        )
        return jsonify({'ok': True})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/log/trade', methods=['POST'])
def ep_log_trade():
    """
    {tenor, direction, notional, rate, counterparty?, spot_ref?, outright?}
    direction: 'buy' | 'sell'
    """
    try:
        body = request.get_json(force=True) or {}
        dc.log_trade(
            tenor        = str(body.get('tenor', '')),
            direction    = str(body.get('direction', '')),
            notional     = float(body.get('notional', 0)),
            rate         = float(body.get('rate', 0)),
            counterparty = str(body.get('counterparty', '')),
            spot_ref     = float(body.get('spot_ref', 0)),
            outright     = float(body.get('outright', 0)),
        )
        return jsonify({'ok': True})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/log/flow', methods=['POST'])
def ep_log_flow():
    """
    {counterparty, tenor, direction, size, rate, my_side?, tag?, notes?}
    direction: 'buy' | 'sell'  (from counterparty's perspective)
    my_side:   'GVN' | 'TKN'
    """
    try:
        body = request.get_json(force=True) or {}
        dc.log_flow(
            counterparty = str(body.get('counterparty', '')),
            tenor        = str(body.get('tenor', '')),
            direction    = str(body.get('direction', '')),
            size         = float(body.get('size', 0)),
            rate         = float(body.get('rate', 0)),
            my_side      = str(body.get('my_side', '')),
            tag          = str(body.get('tag', '')),
            notes        = str(body.get('notes', '')),
        )
        return jsonify({'ok': True})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/data/export')
def ep_export():
    """
    ?table=price_log&format=csv   → CSV attachment
    ?table=price_log&format=json  → JSON array
    """
    table = request.args.get('table', 'price_log')
    fmt   = request.args.get('format', 'json').lower()
    try:
        if fmt == 'csv':
            csv = dc.export_csv(table)
            return Response(
                csv,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename="{table}.csv"'},
            )
        else:
            rows = dc.export_json_rows(table)
            return jsonify({'ok': True, 'table': table, 'rows': rows, 'count': len(rows)})
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/rate-history')
def ep_rate_history():
    """
    Returns historical liveMid per tenor from curve_snapshots.
    ?tenors=1M,3M,6M,12M&days=30  (days=0 means all history)
    """
    import json as _json, pandas as _pd
    tenors_param = request.args.get('tenors', '1M,3M,6M,12M')
    days_str     = request.args.get('days', '30')
    tenors = [t.strip() for t in tenors_param.split(',') if t.strip()]
    try:
        days = int(days_str)
    except (ValueError, TypeError):
        days = 30
    try:
        df = dc._fetch_df('curve_snapshots', limit=200_000)
        if df.empty:
            return jsonify({'ok': True, 'data': {t: [] for t in tenors}, 'message': 'No snapshots yet.'})
        df['ts'] = _pd.to_datetime(df['ts'])
        if days > 0:
            cutoff = _pd.Timestamp.now() - _pd.Timedelta(days=days)
            df = df[df['ts'] >= cutoff]
        df['date'] = df['ts'].dt.date
        result = {t: {} for t in tenors}
        for _, row in df.iterrows():
            try:
                curve = _json.loads(row['curve_json']) if isinstance(row['curve_json'], str) else []
                d = row['date'].isoformat()
                for entry in curve:
                    k = entry.get('key')
                    if k in result and entry.get('liveMid') is not None:
                        result[k][d] = round(float(entry['liveMid']), 4)
            except Exception:
                pass
        out = {t: [{'date': d, 'value': v} for d, v in sorted(result[t].items())] for t in tenors}
        return jsonify({'ok': True, 'data': out, 'days': days})
    except Exception as exc:
        import traceback
        return jsonify({'ok': False, 'error': str(exc), 'trace': traceback.format_exc()}), 500


@app.route('/api/brokerage/scan', methods=['POST'])
def ep_brokerage_scan():
    import urllib.request, urllib.error, json as _json, re as _re
    body    = request.get_json(force=True) or {}
    api_key = str(body.get('api_key', '')).strip()
    if not api_key:
        return jsonify({'ok': False, 'error': 'No API key provided'}), 400
    image_b64  = str(body.get('image', ''))
    media_type = str(body.get('media_type', 'image/jpeg'))
    prompt = (
        'This is a Japanese FX brokerage statement. '
        'Extract all bank or counterparty names and their yen brokerage fee/commission amounts. '
        'Identify the statement period (month and year). '
        'Reply with ONLY valid JSON, no explanation:\n'
        '{"month":"YYYY-MM","entries":[{"bank":"MUFG","amount":1250000}],"total":0}\n'
        'Amounts are plain integers in yen. If month is unclear, guess from context.'
    )
    payload = {
        'model': 'claude-sonnet-4-6',
        'max_tokens': 1024,
        'messages': [{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': image_b64}},
            {'type': 'text', 'text': prompt}
        ]}]
    }
    try:
        req = urllib.request.Request(
            'https://api.anthropic.com/v1/messages',
            data=_json.dumps(payload).encode('utf-8'),
            headers={'x-api-key': api_key, 'anthropic-version': '2023-06-01',
                     'content-type': 'application/json', 'accept': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = _json.loads(resp.read().decode('utf-8'))
        text = resp_data.get('content', [{}])[0].get('text', '')
        m = _re.search(r'\{[\s\S]*\}', text)
        if not m:
            return jsonify({'ok': False, 'error': 'No JSON in response: ' + text[:200]}), 500
        return jsonify({'ok': True, 'result': _json.loads(m.group())})
    except urllib.error.HTTPError as exc:
        err = exc.read().decode('utf-8', errors='replace')
        return jsonify({'ok': False, 'error': 'API ' + str(exc.code) + ': ' + err[:300]}), 500
    except Exception as exc:
        import traceback
        return jsonify({'ok': False, 'error': str(exc), 'trace': traceback.format_exc()}), 500

@app.route('/api/spot')
def ep_spot():
    """Live USDJPY spot + 30d history via yfinance."""
    import yfinance as yf, json as _json
    try:
        tk = yf.Ticker('USDJPY=X')
        hist = tk.history(period='30d', interval='1d')
        closes = [round(float(v), 4) for v in hist['Close'].dropna().tolist()]
        info = tk.fast_info
        price = round(float(info.last_price), 4) if info.last_price else (closes[-1] if closes else None)
        prev  = closes[-2] if len(closes) >= 2 else price
        change = round(price - prev, 4) if price and prev else 0.0
        change_pct = round(change / prev * 100, 4) if prev else 0.0
        return jsonify({'ok': True, 'price': price, 'change': change, 'changePct': change_pct, 'prices': closes[-30:]})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500

intel.register_intelligence(app)
fc_mod.register_forecast(app)

if __name__ == '__main__':
    print(f'curve_engine v1.2  QuantLib {ql.__version__}  :5051')
    print(f'database: {dc.DB_PATH}')
    app.run(host='0.0.0.0', port=5051, debug=False)
