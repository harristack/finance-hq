#!/usr/bin/env python3
"""
data_capture.py — DuckDB persistence layer for Finance HQ

Tables (all timestamps in JST, UTC+9):
  price_log       — per-tenor bid/offer snapshots
  curve_snapshots — full curve state every 30 min
  trade_log       — executed trades
  flow_log        — counterparty flow observations
"""
from __future__ import annotations
import json
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
import duckdb

DB_PATH = Path.home() / 'finance-hq' / 'data' / 'finhq.db'
JST     = timezone(timedelta(hours=9))

VALID_TABLES = {'price_log', 'curve_snapshots', 'trade_log', 'flow_log'}

# One connection per process, protected by a lock for thread safety
_lock  = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None

def _get() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(DB_PATH))
        _create_tables(_conn)
    return _conn

def _create_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS price_log (
            ts          TIMESTAMP NOT NULL,
            tenor       TEXT,
            bid         DOUBLE,
            offer       DOUBLE,
            mid         DOUBLE,
            spot        DOUBLE,
            session     TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS curve_snapshots (
            ts          TIMESTAMP NOT NULL,
            spot        DOUBLE,
            sofr        DOUBLE,
            tona        DOUBLE,
            curve_json  TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            ts          TIMESTAMP NOT NULL,
            tenor       TEXT,
            direction   TEXT,
            notional    DOUBLE,
            rate        DOUBLE,
            counterparty TEXT,
            spot_ref    DOUBLE,
            outright    DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS flow_log (
            ts          TIMESTAMP NOT NULL,
            counterparty TEXT,
            tenor        TEXT,
            direction    TEXT,
            size         DOUBLE,
            rate         DOUBLE,
            my_side      TEXT,
            tag          TEXT,
            notes        TEXT
        )
    """)

def now_jst() -> datetime:
    """Current time as naive datetime in JST (no tzinfo stored in DuckDB TIMESTAMP)."""
    return datetime.now(JST).replace(tzinfo=None)

# ── Writes ────────────────────────────────────────────────────────────────────

def log_price(tenor: str, bid: float | None, offer: float | None,
              mid: float | None, spot: float, session: str = '') -> None:
    ts = now_jst()
    with _lock:
        _get().execute(
            "INSERT INTO price_log VALUES (?,?,?,?,?,?,?)",
            [ts, tenor, bid, offer, mid, spot, session],
        )

def log_prices_bulk(rows: list[dict], spot: float, session: str = '') -> int:
    """Insert multiple price rows in a single transaction. Returns rows inserted."""
    ts = now_jst()
    params = [
        [ts,
         r.get('tenor') or r.get('key'),
         r.get('bid'),
         r.get('offer'),
         r.get('mid') if r.get('mid') is not None else r.get('mktMid'),
         spot,
         session]
        for r in rows
    ]
    with _lock:
        con = _get()
        for p in params:
            con.execute("INSERT INTO price_log VALUES (?,?,?,?,?,?,?)", p)
    return len(params)

def log_snapshot(spot: float, sofr: float, tona: float, rows: list[dict]) -> None:
    ts = now_jst()
    curve_json = json.dumps(rows, default=str)
    with _lock:
        _get().execute(
            "INSERT INTO curve_snapshots VALUES (?,?,?,?,?)",
            [ts, spot, sofr, tona, curve_json],
        )

def log_trade(tenor: str, direction: str, notional: float, rate: float,
              counterparty: str = '', spot_ref: float = 0.0,
              outright: float = 0.0) -> None:
    ts = now_jst()
    with _lock:
        _get().execute(
            "INSERT INTO trade_log VALUES (?,?,?,?,?,?,?,?)",
            [ts, tenor, direction, notional, rate, counterparty, spot_ref, outright],
        )

def log_flow(counterparty: str, tenor: str, direction: str, size: float,
             rate: float, my_side: str = '', tag: str = '',
             notes: str = '') -> None:
    ts = now_jst()
    with _lock:
        _get().execute(
            "INSERT INTO flow_log VALUES (?,?,?,?,?,?,?,?,?)",
            [ts, counterparty, tenor, direction, size, rate, my_side, tag, notes],
        )

# ── Reads / export ────────────────────────────────────────────────────────────

def _fetch_df(table: str, limit: int = 10_000):
    if table not in VALID_TABLES:
        raise ValueError(f"unknown table '{table}'")
    with _lock:
        return _get().execute(
            f"SELECT * FROM {table} ORDER BY ts DESC LIMIT {limit}"
        ).df()

def export_csv(table: str) -> str:
    df = _fetch_df(table)
    return df.to_csv(index=False)

def export_json_rows(table: str) -> list[dict]:
    df = _fetch_df(table)
    # Timestamps → ISO strings for JSON serialisation
    for col in df.select_dtypes(include=['datetime64[ns]', 'object']).columns:
        if 'ts' in col.lower():
            df[col] = df[col].astype(str)
    return df.to_dict(orient='records')

def row_counts() -> dict[str, int]:
    with _lock:
        con = _get()
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in VALID_TABLES}

# Ensure tables exist as soon as this module is imported
_get()
