# ============================================================
# MARKET BRAIN AI V2 — REPLIT READY
# Adaptive Multi-Timeframe Crypto Scanner + Trade Manager
# ============================================================
#
# TIMEFRAMES
# 1D  = Major Bias
# 4H  = Structure / Liquidity / SMC
# 1H  = Setup
# 30M = Entry Confirmation
#
# IMPORTANT
# - This is a statistical/rules engine, not a guaranteed-profit AI.
# - "Confidence" is a setup-quality score, NOT probability of profit.
# - Uses public Binance Spot klines; no API key is required for scanning.
# - Paper/demo trading is strongly recommended before real money.
#
# REPLIT:
#   pip install -r requirements.txt
#   python market_brain.py
#
# Dashboard:
#   http://0.0.0.0:8080
# ============================================================

import os
import time
import math
import sqlite3
import threading
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string

# ---------------- CONFIG ----------------

BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")
BINANCE_ENDPOINTS = []
for endpoint in [
    BASE_URL,
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]:
    if endpoint and endpoint not in BINANCE_ENDPOINTS:
        BINANCE_ENDPOINTS.append(endpoint)
PKT = ZoneInfo("Asia/Karachi")
PORT = int(os.getenv("PORT", "8080"))

SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "300"))
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "300"))

MIN_SETUP_SCORE = float(os.getenv("MIN_SETUP_SCORE", "6.5"))
MIN_RR = float(os.getenv("MIN_RR", "2.0"))
MAX_RR = float(os.getenv("MAX_RR", "5.0"))

# Never risk a target smaller than the initial stop.
MIN_SL_PCT = float(os.getenv("MIN_SL_PCT", "0.25"))
MAX_SL_PCT = float(os.getenv("MAX_SL_PCT", "3.0"))
MAX_TP_PCT = float(os.getenv("MAX_TP_PCT", "15.0"))

# Paper account parameters. These do NOT place real orders.
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "1000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))

DB_FILE = os.getenv("DB_FILE", "market_brain.db")

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "SUIUSDT", "OPUSDT", "ARBUSDT", "INJUSDT",
    "APTUSDT", "LTCUSDT", "TRXUSDT", "UNIUSDT", "ATOMUSDT"
]

TIMEFRAMES = ["1d", "4h", "1h", "30m"]

# Initial expert weights. They are later adjusted from closed-trade statistics,
# but only inside safe bounds to reduce overfitting.
BASE_WEIGHTS = {
    "bias_1d": 15.0,
    "structure_4h": 15.0,
    "bos_choch": 10.0,
    "liquidity": 10.0,
    "ob_fvg": 12.0,
    "location": 8.0,
    "momentum": 10.0,
    "volume_pressure": 8.0,
    "entry_30m": 5.0,
    "risk_rr": 7.0,
}

WEIGHT_BOUNDS = {
    "bias_1d": (8, 22),
    "structure_4h": (8, 22),
    "bos_choch": (5, 16),
    "liquidity": (5, 16),
    "ob_fvg": (6, 18),
    "location": (4, 12),
    "momentum": (5, 15),
    "volume_pressure": (4, 13),
    "entry_30m": (3, 10),
    "risk_rr": (4, 12),
}

# ---------------- APP / HTTP ----------------

app = Flask(__name__)
session = requests.Session()
session.headers.update({"User-Agent": "MARKET-BRAIN-AI/2.0"})

state_lock = threading.Lock()
STATE = {
    "last_scan": None,
    "scanning": False,
    "signals": [],
    "active_trades": [],
    "market": {},
    "errors": [],
    "status": "STARTING",
}

# ---------------- DATABASE ----------------

def db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        entry REAL NOT NULL,
        sl REAL NOT NULL,
        tp1 REAL NOT NULL,
        tp2 REAL NOT NULL,
        tp3 REAL NOT NULL,
        rr1 REAL NOT NULL,
        rr2 REAL NOT NULL,
        rr3 REAL NOT NULL,
        setup_score REAL NOT NULL,
        entry_score REAL NOT NULL,
        confidence_label TEXT NOT NULL,
        status TEXT NOT NULL,
        outcome TEXT,
        exit_price REAL,
        exit_reason TEXT,
        pnl_r REAL,
        hit_tp1 INTEGER DEFAULT 0,
        hit_tp2 INTEGER DEFAULT 0,
        hit_tp3 INTEGER DEFAULT 0,
        realized_r REAL DEFAULT 0,
        remaining_pct REAL DEFAULT 100,
        protected_stop REAL,
        max_favorable_r REAL DEFAULT 0,
        max_adverse_r REAL DEFAULT 0,
        factor_json TEXT,
        snapshot_json TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS factor_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER NOT NULL,
        factor TEXT NOT NULL,
        factor_score REAL NOT NULL,
        outcome TEXT,
        pnl_r REAL,
        FOREIGN KEY(trade_id) REFERENCES trades(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS scan_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        symbol TEXT NOT NULL,
        direction TEXT,
        setup_score REAL,
        entry_score REAL,
        decision TEXT,
        reasons TEXT
    )
    """)

    # Lightweight migration for an existing database created by an earlier version.
    existing = {r["name"] for r in cur.execute("PRAGMA table_info(trades)").fetchall()}
    migrations = {
        "realized_r": "ALTER TABLE trades ADD COLUMN realized_r REAL DEFAULT 0",
        "remaining_pct": "ALTER TABLE trades ADD COLUMN remaining_pct REAL DEFAULT 100",
        "protected_stop": "ALTER TABLE trades ADD COLUMN protected_stop REAL",
    }
    for name, sql in migrations.items():
        if name not in existing:
            cur.execute(sql)

    conn.commit()
    conn.close()

# ---------------- UTILITIES ----------------

def now():
    return datetime.now(PKT).isoformat(timespec="seconds")

def now_pkt_text():
    return datetime.now(PKT).strftime("%Y-%m-%d %H:%M:%S PKT")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_float(x, default=0.0):
    try:
        v = float(x)
        return default if not math.isfinite(v) else v
    except Exception:
        return default

def pct(a, b):
    if b == 0:
        return 0
    return (a - b) / b * 100.0

def normalize_weights(raw):
    total = sum(raw.values())
    if total <= 0:
        return BASE_WEIGHTS.copy()
    return {k: v * 100.0 / total for k, v in raw.items()}

# ---------------- BINANCE DATA ----------------

def fetch_klines(symbol, interval, limit=KLINE_LIMIT):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_error = None

    for endpoint in BINANCE_ENDPOINTS:
        url = f"{endpoint.rstrip('/')}/api/v3/klines"
        for attempt in range(2):
            try:
                r = session.get(url, params=params, timeout=15)
                r.raise_for_status()
                raw = r.json()
                if not isinstance(raw, list) or len(raw) < 80:
                    raise ValueError("Insufficient kline data")

                cols = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ]
                df = pd.DataFrame(raw, columns=cols)
                for c in ["open", "high", "low", "close", "volume",
                          "quote_volume", "taker_buy_base", "taker_buy_quote"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

                # Strict closed-candle-only rule.
                now_utc = datetime.now(timezone.utc)
                df = df[df["close_time"] <= now_utc].copy()
                if len(df) < 80:
                    raise ValueError("Not enough closed candles after filtering")

                return df.reset_index(drop=True)
            except Exception as e:
                last_error = e
                time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"{symbol} {interval}: all Binance endpoints failed: {last_error}")

# ---------------- INDICATORS ----------------

def add_indicators(df):
    d = df.copy()

    for n in [20, 50, 200]:
        d[f"ema{n}"] = d["close"].ewm(span=n, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    d["volume_ma20"] = d["volume"].rolling(20).mean()
    d["volume_ratio"] = (
        d["volume"] / d["volume_ma20"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1)

    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["body_ratio"] = (d["close"] - d["open"]).abs() / rng
    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]

    # Simple ADX-like trend strength using directional movement.
    up = d["high"].diff()
    down = -d["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr_safe = d["atr"].replace(0, np.nan)
    d["plus_di"] = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1/14, adjust=False).mean() / atr_safe
    d["minus_di"] = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1/14, adjust=False).mean() / atr_safe
    d["trend_strength"] = (d["plus_di"] - d["minus_di"]).abs().fillna(0)

    return d.replace([np.inf, -np.inf], np.nan).ffill().bfill()

def pressure(df, length=12):
    d = df.tail(length)
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    strength = ((d["close"] - d["open"]) / rng).fillna(0).abs()
    signed = np.sign(d["close"] - d["open"]) * strength * d["volume"]

    buy = float(signed.clip(lower=0).sum())
    sell = float((-signed.clip(upper=0)).sum())
    total = buy + sell

    if total == 0:
        return 50.0, 50.0
    return buy / total * 100, sell / total * 100

# ---------------- STRUCTURE ----------------

def pivots(df, left=3, right=3):
    d = df.copy()
    ph = pd.Series(False, index=d.index)
    pl = pd.Series(False, index=d.index)

    for i in range(left, len(d) - right):
        hi = d["high"].iloc[i]
        lo = d["low"].iloc[i]
        if hi >= d["high"].iloc[i-left:i+right+1].max():
            ph.iloc[i] = True
        if lo <= d["low"].iloc[i-left:i+right+1].min():
            pl.iloc[i] = True

    d["pivot_high"] = ph
    d["pivot_low"] = pl
    return d

def structure_info(df):
    d = pivots(df.tail(160).reset_index(drop=True))
    highs = d.loc[d["pivot_high"], "high"].tolist()
    lows = d.loc[d["pivot_low"], "low"].tolist()

    if len(highs) < 2 or len(lows) < 2:
        return {
            "trend": "NEUTRAL",
            "bos": None,
            "choch": None,
            "last_high": float(d["high"].iloc[-1]),
            "last_low": float(d["low"].iloc[-1]),
            "equal_high": False,
            "equal_low": False,
        }

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]
    close = float(d["close"].iloc[-1])

    tol = float(d["atr"].iloc[-1]) * 0.35 if "atr" in d else close * 0.002
    trend = "BULLISH" if h2 > h1 and l2 > l1 else \
            "BEARISH" if h2 < h1 and l2 < l1 else "NEUTRAL"

    bos = "BULLISH" if close > h1 else "BEARISH" if close < l1 else None

    # Previous structure direction, if enough pivots exist.
    prev_trend = None
    if len(highs) >= 3 and len(lows) >= 3:
        prev_trend = "BULLISH" if highs[-2] > highs[-3] and lows[-2] > lows[-3] \
                     else "BEARISH" if highs[-2] < highs[-3] and lows[-2] < lows[-3] else None

    choch = None
    if prev_trend == "BULLISH" and close < l1:
        choch = "BEARISH"
    elif prev_trend == "BEARISH" and close > h1:
        choch = "BULLISH"

    return {
        "trend": trend,
        "bos": bos,
        "choch": choch,
        "last_high": float(h2),
        "last_low": float(l2),
        "equal_high": abs(h2 - h1) <= tol,
        "equal_low": abs(l2 - l1) <= tol,
    }

# ---------------- SMC / CANDLE FEATURES ----------------

def smc_info(df):
    d = df.copy()
    if len(d) < 5:
        return {}

    # Latest three closed candles.
    a, b, c = d.iloc[-3], d.iloc[-2], d.iloc[-1]

    bull_fvg = float(c["low"]) > float(a["high"])
    bear_fvg = float(c["high"]) < float(a["low"])

    # Approximation: last opposite candle before an impulse candle.
    bull_ob = bool(
        float(b["close"]) < float(b["open"]) and
        float(c["close"]) > float(c["open"]) and
        float(c["body_ratio"]) >= 0.55
    )
    bear_ob = bool(
        float(b["close"]) > float(b["open"]) and
        float(c["close"]) < float(c["open"]) and
        float(c["body_ratio"]) >= 0.55
    )

    last50 = d.tail(50)
    eq = (float(last50["high"].max()) + float(last50["low"].min())) / 2
    price = float(c["close"])

    recent = d.iloc[:-1].tail(30)
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())

    bull_sweep = float(c["low"]) < recent_low and price > recent_low
    bear_sweep = float(c["high"]) > recent_high and price < recent_high

    return {
        "bull_fvg": bull_fvg,
        "bear_fvg": bear_fvg,
        "bull_ob": bull_ob,
        "bear_ob": bear_ob,
        "equilibrium": eq,
        "premium": price > eq,
        "discount": price < eq,
        "bull_sweep": bull_sweep,
        "bear_sweep": bear_sweep,
    }

def candle_info(df):
    c = df.iloc[-1]
    prev = df.iloc[-2]

    bull_engulf = (
        c["close"] > c["open"] and
        prev["close"] < prev["open"] and
        c["close"] >= prev["open"] and
        c["open"] <= prev["close"]
    )
    bear_engulf = (
        c["close"] < c["open"] and
        prev["close"] > prev["open"] and
        c["open"] >= prev["close"] and
        c["close"] <= prev["open"]
    )

    hammer = (
        c["lower_wick"] >= max(c["body_ratio"] * (c["high"] - c["low"]) * 1.5, 1e-12)
        and c["lower_wick"] > c["upper_wick"] * 1.5
    )
    shooting = (
        c["upper_wick"] > c["lower_wick"] * 1.5
        and c["upper_wick"] >= max(c["body_ratio"] * (c["high"] - c["low"]) * 1.5, 1e-12)
    )

    return {
        "bull_engulf": bool(bull_engulf),
        "bear_engulf": bool(bear_engulf),
        "hammer": bool(hammer),
        "shooting_star": bool(shooting),
        "body_ratio": float(c["body_ratio"]),
    }

# ---------------- ADAPTIVE LEARNING ----------------

def historical_factor_stats():
    conn = db()
    rows = conn.execute("""
        SELECT factor,
               COUNT(*) AS n,
               SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) AS wins,
               AVG(CASE WHEN outcome='WIN' THEN 1.0 ELSE 0.0 END) AS win_rate,
               AVG(COALESCE(pnl_r, 0)) AS avg_r
        FROM factor_results
        WHERE outcome IN ('WIN','LOSS')
        GROUP BY factor
    """).fetchall()
    conn.close()

    result = {}
    for r in rows:
        n = int(r["n"] or 0)
        wr = safe_float(r["win_rate"], 0.5)
        avg_r = safe_float(r["avg_r"], 0)
        # Reliability starts moving only after enough observations.
        confidence = min(1.0, n / 40.0)
        edge = 0.5 + (wr - 0.5) * confidence
        edge += clamp(avg_r / 10.0, -0.15, 0.15)
        result[r["factor"]] = {
            "n": n,
            "win_rate": wr,
            "avg_r": avg_r,
            "edge": clamp(edge, 0.25, 0.75)
        }
    return result

def adaptive_weights():
    stats = historical_factor_stats()
    raw = {}
    for factor, base in BASE_WEIGHTS.items():
        s = stats.get(factor)
        if not s or s["n"] < 10:
            raw[factor] = base
            continue

        # Move only moderately away from the expert prior.
        multiplier = 0.85 + 0.30 * ((s["edge"] - 0.25) / 0.50)
        raw[factor] = base * multiplier

    bounded = {}
    for factor, value in raw.items():
        lo, hi = WEIGHT_BOUNDS[factor]
        bounded[factor] = clamp(value, lo, hi)

    return normalize_weights(bounded)

# ---------------- SCORING ----------------

def directional_score(df, direction):
    c = df.iloc[-1]
    bull = direction == "LONG"
    score = 0.0

    ema_align = (
        c["ema20"] > c["ema50"] > c["ema200"]
        if bull else
        c["ema20"] < c["ema50"] < c["ema200"]
    )
    price_align = c["close"] > c["ema20"] if bull else c["close"] < c["ema20"]
    rsi_ok = 52 <= c["rsi"] <= 72 if bull else 28 <= c["rsi"] <= 48

    score += 4 if ema_align else 0
    score += 3 if price_align else 0
    score += 3 if rsi_ok else 0
    return score  # /10

def score_setup(data):
    d1, h4, h1, m30 = data["1d"], data["4h"], data["1h"], data["30m"]
    s1 = structure_info(d1)
    s4 = structure_info(h4)
    s1h = structure_info(h1)
    s30 = structure_info(m30)

    smc4 = smc_info(h4)
    smc1 = smc_info(h1)
    smc30 = smc_info(m30)
    ci30 = candle_info(m30)

    # Candidate direction from 1D/4H/1H agreement.
    long_votes = 0
    short_votes = 0
    for s in [s1, s4, s1h]:
        if s["trend"] == "BULLISH":
            long_votes += 1
        if s["trend"] == "BEARISH":
            short_votes += 1

    direction = "LONG" if long_votes > short_votes else \
                "SHORT" if short_votes > long_votes else "NONE"

    if direction == "NONE":
        return {"direction": "NONE", "score": 0, "factors": {}, "reasons": ["MTF direction conflict"]}

    bull = direction == "LONG"
    weights = adaptive_weights()

    factors = {}
    reasons = []

    # 1D bias
    f = 10 if (s1["trend"] == ("BULLISH" if bull else "BEARISH")) else 2
    factors["bias_1d"] = f
    if f >= 8: reasons.append("1D bias aligned")

    # 4H structure
    f = 10 if (s4["trend"] == ("BULLISH" if bull else "BEARISH")) else 3
    factors["structure_4h"] = f
    if f >= 8: reasons.append("4H structure aligned")

    # BOS / CHoCH
    desired = "BULLISH" if bull else "BEARISH"
    f = 10 if s4["bos"] == desired else 7 if s4["choch"] == desired else 3
    factors["bos_choch"] = f
    if f >= 8: reasons.append("BOS/CHoCH confirmation")

    # Liquidity
    sweep = smc4["bull_sweep"] if bull else smc4["bear_sweep"]
    equal = smc4["equal_low"] if bull else smc4["equal_high"]
    f = 10 if sweep else 7 if equal else 4
    factors["liquidity"] = f
    if sweep: reasons.append("liquidity sweep")

    # OB + FVG
    ob = smc1["bull_ob"] if bull else smc1["bear_ob"]
    fvg = smc1["bull_fvg"] if bull else smc1["bear_fvg"]
    f = 10 if ob and fvg else 8 if ob or fvg else 3
    factors["ob_fvg"] = f
    if ob or fvg: reasons.append("OB/FVG present")

    # Premium/discount
    location_ok = smc1["discount"] if bull else smc1["premium"]
    factors["location"] = 9 if location_ok else 4
    if location_ok: reasons.append("favorable premium/discount location")

    # Momentum
    factors["momentum"] = directional_score(h1, direction)

    # Volume/pressure
    buy1, sell1 = pressure(h1)
    buy30, sell30 = pressure(m30)
    if bull:
          p = (buy1 + buy30) / 2
        f = 10 if p >= 65 else 8 if p >= 55 else 5 if p >= 50 else 2
    else:
        p = (sell1 + sell30) / 2
        f = 10 if p >= 65 else 8 if p >= 55 else 5 if p >= 50 else 2
    factors["volume_pressure"] = f

    # 30M entry
    candle_ok = (
        ci30["bull_engulf"] or ci30["hammer"] or
        s30["bos"] == "BULLISH"
    ) if bull else (
        ci30["bear_engulf"] or ci30["shooting_star"] or
        s30["bos"] == "BEARISH"
    )
    f = 10 if candle_ok and s30["trend"] == desired else \
        8 if candle_ok else \
        5 if s30["trend"] == desired else 2
    factors["entry_30m"] = f
    if f >= 8: reasons.append("30M entry confirmation")

    # Risk/RR is scored after a preliminary risk model.
    risk_model = build_trade_levels(data, direction, factors, preliminary=True)
    rr1 = risk_model["rr1"]
    rr3 = risk_model["rr3"]
    rr_score = 10 if rr3 >= 3 else 9 if rr3 >= 2.5 else 7 if rr3 >= 2 else 4 if rr3 >= 1.5 else 1
    factors["risk_rr"] = rr_score

    weighted = sum((factors[k] / 10.0) * weights[k] for k in weights)
    final_score = weighted / 10.0

    # Hard gates for a high-quality entry.
    if s4["trend"] == "NEUTRAL":
        final_score -= 0.7
    if rr1 < MIN_RR:
        final_score -= 1.0
        reasons.append("TP1 R:R below minimum")
    if not candle_ok:
        final_score -= 0.4
        reasons.append("30M candle confirmation weak")

    final_score = clamp(final_score, 0, 10)

    return {
        "direction": direction,
        "score": final_score,
        "factors": factors,
        "weights": weights,
        "reasons": reasons,
        "pressure": {"1h_buy": buy1, "1h_sell": sell1, "30m_buy": buy30, "30m_sell": sell30},
        "structure": {"1d": s1, "4h": s4, "1h": s1h, "30m": s30},
        "smc": {"4h": smc4, "1h": smc1, "30m": smc30},
    }

# ---------------- DYNAMIC SL / TP ----------------

def build_trade_levels(data, direction, factors=None, preliminary=False):
    h1 = data["1h"]
    m30 = data["30m"]
    c = float(m30["close"].iloc[-1])

    atr = float(h1["atr"].iloc[-1])
    atr30 = float(m30["atr"].iloc[-1])
    atr = max(atr, atr30)

    s4 = structure_info(data["4h"])
    s1 = structure_info(h1)
    smc1 = smc_info(h1)
    bull = direction == "LONG"

    # Structural stop candidates.
    if bull:
        structural = min(s4["last_low"], s1["last_low"], float(m30["low"].tail(8).min()))
        raw_sl = min(c - atr * 1.15, structural - atr * 0.15)
        stop_distance = c - raw_sl
    else:
        structural = max(s4["last_high"], s1["last_high"], float(m30["high"].tail(8).max()))
        raw_sl = max(c + atr * 1.15, structural + atr * 0.15)
        stop_distance = raw_sl - c

    # Keep SL inside sane percentage limits.
    min_dist = c * MIN_SL_PCT / 100
    max_dist = c * MAX_SL_PCT / 100
    stop_distance = clamp(stop_distance, min_dist, max_dist)

    sl = c - stop_distance if bull else c + stop_distance

    # Small first target: designed to secure the first R before extending.
    # Larger targets are projected from ATR + structure + liquidity.
    score_hint = 7.0
    if factors:
        score_hint = sum(factors.values()) / len(factors)

    # Target multipliers increase gradually with setup quality.
    base_r1 = 1.05 + clamp((score_hint - 5) / 5, 0, 1) * 0.35
    base_r2 = 2.0 + clamp((score_hint - 6) / 4, 0, 1) * 0.75
    base_r3 = 2.8 + clamp((score_hint - 7) / 3, 0, 1) * 1.7

    # Historical average R can extend targets, but never below the minimum.
    stats = historical_factor_stats()
    if stats:
        avg_r = np.mean([v["avg_r"] for v in stats.values() if v["n"] >= 10]) if any(v["n"] >= 10 for v in stats.values()) else 0
        if avg_r > 1.5:
            base_r2 += min(0.35, (avg_r - 1.5) * 0.15)
            base_r3 += min(0.60, (avg_r - 1.5) * 0.25)

    base_r1 = max(1.0, base_r1)
    base_r2 = max(base_r1 + 0.5, base_r2)
    base_r3 = max(base_r2 + 0.5, base_r3)

    max_r3_by_price = (MAX_TP_PCT / 100 * c) / stop_distance
    base_r3 = min(base_r3, max_r3_by_price)
    base_r2 = min(base_r2, max(base_r3 - 0.4, base_r2))
    base_r1 = min(base_r1, max(base_r2 - 0.5, 1.0))

    tp1 = c + stop_distance * base_r1 if bull else c - stop_distance * base_r1
    tp2 = c + stop_distance * base_r2 if bull else c - stop_distance * base_r2
    tp3 = c + stop_distance * base_r3 if bull else c - stop_distance * base_r3

    rr1 = abs(tp1 - c) / stop_distance
    rr2 = abs(tp2 - c) / stop_distance
    rr3 = abs(tp3 - c) / stop_distance

    # Optional structural target context.
    target_note = "ATR/R ladder"
    if bull and smc1["premium"]:
        target_note = "ATR + premium/liquidity target"
    elif (not bull) and smc1["discount"]:
        target_note = "ATR + discount/liquidity target"

    risk_cash = ACCOUNT_BALANCE * RISK_PER_TRADE_PCT / 100
    position_size = risk_cash / stop_distance if stop_distance > 0 else 0

    return {
        "entry": c,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr1": rr1,
        "rr2": rr2,
        "rr3": rr3,
        "sl_pct": abs(c - sl) / c * 100,
        "tp1_pct": abs(tp1 - c) / c * 100,
        "tp2_pct": abs(tp2 - c) / c * 100,
        "tp3_pct": abs(tp3 - c) / c * 100,
        "risk_cash": risk_cash,
        "position_size": position_size,
        "target_note": target_note,
    }

# ---------------- SNAPSHOT ----------------

def compact_snapshot(data):
    out = {}
    for tf, df in data.items():
        c = df.iloc[-1]
        buy, sell = pressure(df)
        out[tf] = {
            "close": safe_float(c["close"]),
            "ema20": safe_float(c["ema20"]),
            "ema50": safe_float(c["ema50"]),
            "ema200": safe_float(c["ema200"]),
            "rsi": safe_float(c["rsi"]),
            "atr": safe_float(c["atr"]),
            "volume": safe_float(c["volume"]),
            "volume_ratio": safe_float(c["volume_ratio"]),
            "buy_pressure": buy,
            "sell_pressure": sell,
            "trend_strength": safe_float(c["trend_strength"]),
        }
    return out

# ---------------- SCANNING ----------------

def fetch_symbol_data(symbol):
    result = {}
    for tf in TIMEFRAMES:
        result[tf] = add_indicators(fetch_klines(symbol, tf))
    return result

def log_scan(symbol, result, decision):
    conn = db()
    conn.execute("""
        INSERT INTO scan_log
        (created_at, symbol, direction, setup_score, entry_score, decision, reasons)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        now(), symbol,
        result.get("direction"),
        result.get("score", 0),
        result.get("entry_score", 0),
        decision,
        "; ".join(result.get("reasons", []))
    ))
    conn.commit()
    conn.close()

def scan_symbol(symbol):
    data = fetch_symbol_data(symbol)
    result = score_setup(data)

    if result["direction"] == "NONE":
        log_scan(symbol, result, "NO_TRADE")
        return None

    levels = build_trade_levels(
        data, result["direction"], result["factors"]
    )
    result["levels"] = levels
    result["entry_score"] = result["factors"].get("entry_30m", 0)

    # Hard safety gates.
    if levels["rr1"] < MIN_RR:
        log_scan(symbol, result, "REJECT_RR")
        return None

    if result["score"] < MIN_SETUP_SCORE:
        log_scan(symbol, result, "REJECT_SCORE")
        return None

    # Entry confirmation should not be weak.
    if result["entry_score"] < 6:
        log_scan(symbol, result, "REJECT_ENTRY")
        return None

    label = (
        "VERY STRONG" if result["score"] >= 8.5 else
        "STRONG" if result["score"] >= 7.5 else
        "VALID"
    )
    result["label"] = label
    result["snapshot"] = compact_snapshot(data)
    log_scan(symbol, result, "SIGNAL")
    return result

# ---------------- TRADE RECORD / OUTCOME ----------------

def save_signal(symbol, result):
    lv = result["levels"]
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trades (
            created_at, symbol, direction, entry, sl, tp1, tp2, tp3,
            rr1, rr2, rr3, setup_score, entry_score, confidence_label,
            status, factor_json, snapshot_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now(), symbol, result["direction"],
        lv["entry"], lv["sl"], lv["tp1"], lv["tp2"], lv["tp3"],
        lv["rr1"], lv["rr2"], lv["rr3"],
        result["score"], result["entry_score"], result["label"],
        "PAPER_ACTIVE",
        str(result["factors"]), str(result["snapshot"])
    ))
    trade_id = cur.lastrowid

    for factor, factor_score in result["factors"].items():
        cur.execute("""
            INSERT INTO factor_results
            (trade_id, factor, factor_score, outcome, pnl_r)
            VALUES (?, ?, ?, NULL, NULL)
        """, (trade_id, factor, factor_score))

    conn.commit()
    conn.close()
    return trade_id

def update_factor_outcomes(trade_id, outcome, pnl_r):
    conn = db()
    conn.execute("""
        UPDATE factor_results
        SET outcome=?, pnl_r=?
        WHERE trade_id=?
    """, (outcome, pnl_r, trade_id))
    conn.commit()
    conn.close()

def active_trades():
    conn = db()
    rows = conn.execute("""
        SELECT * FROM trades
        WHERE status='PAPER_ACTIVE'
        ORDER BY id DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def monitor_trade(row):
    """Manage a paper trade using the latest CLOSED 30M candle.

    Uses candle high/low for level detection instead of close-only checks.
    If a stop and target are both touched in the same candle, the stop is
    treated as first (conservative, because candle sequence is unknown).
    """
    symbol = row["symbol"]
    direction = row["direction"]

    try:
        df = add_indicators(fetch_klines(symbol, "30m", 100))
    except Exception:
        return

    bar = df.iloc[-1]
    price = float(bar["close"])
    bar_high = float(bar["high"])
    bar_low = float(bar["low"])
    entry = float(row["entry"])
    original_sl = float(row["sl"])
    tp1 = float(row["tp1"])
    tp2 = float(row["tp2"])
    tp3 = float(row["tp3"])

    stop_dist = abs(entry - original_sl)
    if stop_dist <= 0:
        return

    hit_tp1 = bool(row["hit_tp1"])
    hit_tp2 = bool(row["hit_tp2"])
    hit_tp3 = bool(row["hit_tp3"])
    realized_r = safe_float(row["realized_r"], 0)
    remaining_pct = safe_float(row["remaining_pct"], 100)
    protected_stop = row["protected_stop"]
    protected_stop = safe_float(protected_stop, original_sl) if protected_stop is not None else original_sl

    if direction == "LONG":
        mfe = (bar_high - entry) / stop_dist
        mae = (entry - bar_low) / stop_dist
        stop_hit = bar_low <= (protected_stop if hit_tp1 else original_sl)
        active_stop = protected_stop if hit_tp1 else original_sl
        tp1_crossed = bar_high >= tp1
        tp2_crossed = bar_high >= tp2
        tp3_crossed = bar_high >= tp3
    else:
        mfe = (entry - bar_low) / stop_dist
        mae = (bar_high - entry) / stop_dist
        stop_hit = bar_high >= (protected_stop if hit_tp1 else original_sl)
        active_stop = protected_stop if hit_tp1 else original_sl
        tp1_crossed = bar_low <= tp1
        tp2_crossed = bar_low <= tp2
        tp3_crossed = bar_low <= tp3

    outcome = None
    exit_price = None
    exit_reason = None
    pnl_r = None

    # Conservative sequencing: if the initial stop and TP1 are both touched
    # before any partial profit has been booked, count the stop first.
    if not hit_tp1 and stop_hit:
        realized_r += -1.0
        pnl_r = realized_r
        outcome = "LOSS"
        exit_price = active_stop
        exit_reason = "SL_HIT_OR_SAME_CANDLE_SL_FIRST" if tp1_crossed else "SL_HIT"
        remaining_pct = 0
    else:
        # TP1 -> 30% realized, remaining 70% protected at breakeven.
        if not hit_tp1 and tp1_crossed:
            r1 = abs(tp1 - entry) / stop_dist
            realized_r += 0.30 * r1
            remaining_pct = 70
            hit_tp1 = True
            protected_stop = entry
            active_stop = protected_stop

        # If TP1 was reached in this candle and the same candle also touched
        # breakeven/protected stop, conservative handling is still protective.
        if hit_tp1 and not hit_tp2:
            active_stop = protected_stop
            protected_hit = (bar_low <= active_stop) if direction == "LONG" else (bar_high >= active_stop)
            if protected_hit and not tp2_crossed:
                stop_r = ((active_stop - entry) / stop_dist if direction == "LONG"
                          else (entry - active_stop) / stop_dist)
                realized_r += (remaining_pct / 100.0) * stop_r
                pnl_r = realized_r
                outcome = "WIN" if pnl_r > 0 else "LOSS"
                exit_price = active_stop
                exit_reason = "PROTECTED_STOP_AFTER_TP1"
                remaining_pct = 0
            elif tp2_crossed:
                r2 = abs(tp2 - entry) / stop_dist
                realized_r += 0.30 * r2
                remaining_pct = 40
                hit_tp2 = True
                protected_stop = tp1

        # TP3 -> remaining 40% realized.
        if outcome is None and hit_tp2 and not hit_tp3 and tp3_crossed:
            r3 = abs(tp3 - entry) / stop_dist
            realized_r += 0.40 * r3
            remaining_pct = 0
            hit_tp3 = True
            pnl_r = realized_r
            outcome = "WIN" if pnl_r > 0 else "LOSS"
            exit_price = tp3
            exit_reason = "TP3_HIT"

        # After TP2, protect remaining 40% at TP1. If that stop is touched
        # before TP3, book the remaining realized R and close the trade.
        if outcome is None and hit_tp2 and remaining_pct > 0:
            active_stop = protected_stop
            protected_hit = (bar_low <= active_stop) if direction == "LONG" else (bar_high >= active_stop)
            if protected_hit and not tp3_crossed:
                stop_r = ((active_stop - entry) / stop_dist if direction == "LONG"
                          else (entry - active_stop) / stop_dist)
                realized_r += (remaining_pct / 100.0) * stop_r
                pnl_r = realized_r
                outcome = "WIN" if pnl_r > 0 else "LOSS"
                exit_price = active_stop
                exit_reason = "PROTECTED_STOP_AFTER_TP2"
                remaining_pct = 0

    conn = db()
    if outcome:
        conn.execute("""
            UPDATE trades
            SET status='CLOSED', outcome=?, exit_price=?, exit_reason=?,
                pnl_r=?, realized_r=?, remaining_pct=?, protected_stop=?,
                hit_tp1=?, hit_tp2=?, hit_tp3=?,
                max_favorable_r=MAX(max_favorable_r, ?),
                max_adverse_r=MAX(max_adverse_r, ?)
            WHERE id=?
        """, (
            outcome, exit_price, exit_reason, pnl_r, realized_r, remaining_pct,
            protected_stop, int(hit_tp1), int(hit_tp2), int(hit_tp3),
            max(mfe, 0), max(mae, 0), row["id"]
        ))
        conn.commit()
        conn.close()
        update_factor_outcomes(row["id"], outcome, pnl_r)
        return

    conn.execute("""
        UPDATE trades
        SET hit_tp1=?, hit_tp2=?, hit_tp3=?, realized_r=?, remaining_pct=?,
            protected_stop=?, max_favorable_r=MAX(max_favorable_r, ?),
            max_adverse_r=MAX(max_adverse_r, ?)
        WHERE id=?
    """, (
        int(hit_tp1), int(hit_tp2), int(hit_tp3), realized_r, remaining_pct,
        protected_stop, max(mfe, 0), max(mae, 0), row["id"]
    ))
    conn.commit()
    conn.close()
# ---------------- DASHBOARD DATA ----------------

def stats():
    conn = db()
    total = conn.execute("SELECT COUNT(*) n FROM trades WHERE status='CLOSED'").fetchone()["n"]
    wins = conn.execute("SELECT COUNT(*) n FROM trades WHERE outcome='WIN'").fetchone()["n"]
    losses = conn.execute("SELECT COUNT(*) n FROM trades WHERE outcome='LOSS'").fetchone()["n"]
    avg_r = conn.execute("SELECT AVG(pnl_r) x FROM trades WHERE status='CLOSED'").fetchone()["x"]
    best = conn.execute("""
        SELECT factor, COUNT(*) n, AVG(pnl_r) avg_r,
               AVG(CASE WHEN outcome='WIN' THEN 1.0 ELSE 0.0 END) wr
        FROM factor_results
        WHERE outcome IS NOT NULL
        GROUP BY factor
        ORDER BY avg_r DESC
    """).fetchall()
    conn.close()

    weights = adaptive_weights()
    factor_stats = []
    for x in best:
        item = dict(x)
        item["wr"] = safe_float(item.get("wr"), 0)
        item["avg_r"] = safe_float(item.get("avg_r"), 0)
        item["weight"] = safe_float(weights.get(item.get("factor"), 0), 0)
        factor_stats.append(item)

    return {
        "total": int(total),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": (wins / total * 100) if total else 0,
        "avg_r": safe_float(avg_r),
        "factor_stats": factor_stats,
    }

def recent_trades():
    conn = db()
    rows = conn.execute("""
        SELECT * FROM trades
        ORDER BY id DESC LIMIT 50
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ---------------- WORKER ----------------

def scan_worker():
    while True:
        started = time.time()
        with state_lock:
            STATE["scanning"] = True
            STATE["status"] = "SCANNING"

        signals = []
        errors = []

        # First monitor existing paper trades.
        for t in active_trades():
            try:
                monitor_trade(t)
            except Exception:
                errors.append(f"monitor {t['symbol']}: {traceback.format_exc()[-500:]}")

        # Scan symbols.
        for symbol in SYMBOLS:
            try:
                result = scan_symbol(symbol)
                market_row = {"symbol": symbol, "status": "NO TRADE", "updated_at": now()}
                if result:
                    # Avoid duplicate active signal for same symbol/direction.
                    existing = [x for x in active_trades()
                                 if x["symbol"] == symbol and
                                 x["direction"] == result["direction"]]
                    if not existing:
                        trade_id = save_signal(symbol, result)
                        result["trade_id"] = trade_id
                    signals.append({"symbol": symbol, **result})
                    market_row.update({
                        "status": result.get("direction", "NO TRADE"),
                        "score": result.get("score", 0),
                        "entry_score": result.get("entry_score", 0),
                        "label": result.get("label", ""),
                        "reasons": result.get("reasons", []),
                        "levels": result.get("levels", {}),
                        "pressure": result.get("pressure", {}),
                        "structure": result.get("structure", {}),
                        "smc": result.get("smc", {}),
                        "snapshot": result.get("snapshot", {}),
                    })
                else:
                    # Still keep every one of the 20 coins visible on the dashboard.
                    market_row["reasons"] = ["No valid setup passed the safety gates"]
                with state_lock:
                    STATE["market"][symbol] = market_row
                time.sleep(0.15)
            except Exception as e:
                errors.append(f"{symbol}: {e}")
                with state_lock:
                    STATE["market"][symbol] = {"symbol": symbol, "status": "ERROR", "error": str(e), "updated_at": now()}

        with state_lock:
            STATE["signals"] = sorted(
                signals, key=lambda x: x.get("score", 0), reverse=True
            )
            STATE["active_trades"] = active_trades()
            STATE["last_scan"] = now()
            STATE["status"] = "ONLINE" if not errors else "ONLINE_WITH_WARNINGS"
            STATE["scanning"] = False
            STATE["errors"] = errors[-10:]

        elapsed = time.time() - started
        time.sleep(max(5, SCAN_SECONDS - elapsed))

# ---------------- FLASK DASHBOARD ----------------

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>MARKET BRAIN AI — Professional Dashboard</title>
<style>
:root{--bg:#070b12;--panel:#0e1622;--panel2:#111d2b;--line:#223044;--text:#edf4fb;--muted:#8fa0b4;--green:#20d38a;--red:#ff5d70;--amber:#f5b942;--blue:#4da3ff;--purple:#a78bfa}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#050910,#09111c 55%,#07101a);color:var(--text);font-family:Inter,Arial,sans-serif}
.wrap{max-width:1600px;margin:auto;padding:16px}.top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}.brand{font-size:24px;font-weight:800;letter-spacing:.3px}.sub{color:var(--muted);font-size:12px;margin-top:4px}.clock{font-size:14px;font-weight:700;color:#dce8f6;text-align:right}.badge{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);background:var(--panel);border-radius:999px;padding:7px 11px;font-size:12px}.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green)}
.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:16px 0}.card{background:linear-gradient(145deg,var(--panel),#0b121d);border:1px solid var(--line);border-radius:14px;padding:13px;box-shadow:0 8px 25px #0003}.k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.7px}.v{font-size:23px;font-weight:800;margin-top:5px}.green{color:var(--green)}.red{color:var(--red)}.amber{color:var(--amber)}.blue{color:var(--blue)}
.section{margin-top:18px}.section h2{font-size:15px;margin:0 0 9px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;min-width:1050px}th,td{padding:9px 10px;border-bottom:1px solid #1b2839;text-align:left;font-size:12px;white-space:nowrap}th{position:sticky;top:0;background:#101a27;color:#9fb0c4;font-size:10px;text-transform:uppercase;letter-spacing:.5px;z-index:2}tr:hover td{background:#ffffff05}
.coin{font-weight:800}.pill{display:inline-block;padding:4px 7px;border-radius:7px;font-weight:800;font-size:10px}.buy{background:#0b3929;color:var(--green);border:1px solid #165a42}.sell{background:#421d25;color:var(--red);border:1px solid #6c2d39}.wait{background:#272b31;color:#aab5c2;border:1px solid #39414c}.err{background:#432b12;color:var(--amber);border:1px solid #704d19}.num{font-variant-numeric:tabular-nums}.mini{font-size:10px;color:var(--muted)}
.reason{max-width:300px;white-space:normal;line-height:1.45}.chips{display:flex;gap:5px;flex-wrap:wrap}.chip{background:#162235;border:1px solid #26364c;border-radius:6px;padding:3px 6px;color:#b9c9db;font-size:10px}
.cards2{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.trade{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:13px}.tradeHead{display:flex;justify-content:space-between;align-items:center}.levels{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:10px}.level{background:var(--panel2);border-radius:8px;padding:8px}.level b{display:block;margin-top:3px}.footer{color:var(--muted);font-size:10px;margin:18px 0 5px;line-height:1.5}.bar{height:6px;background:#202c3b;border-radius:9px;overflow:hidden;margin-top:6px}.fill{height:100%;background:var(--blue)}
@media(max-width:1100px){.grid{grid-template-columns:repeat(3,1fr)}.cards2{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.wrap{padding:10px}.grid{grid-template-columns:repeat(2,1fr)}.brand{font-size:19px}.v{font-size:19px}.cards2{grid-template-columns:1fr}.clock{text-align:left}}
</style>
</head>
<body>
<div class="wrap">
<div class="top">
  <div><div class="brand">MARKET BRAIN AI</div><div class="sub">1D Bias → 4H Structure / Liquidity / SMC → 1H Setup → 30M Entry Confirmation</div></div>
  <div><div class="clock" id="pkt">Pakistan Time: {{ pkt }}</div><div style="text-align:right;margin-top:6px"><span class="badge"><span class="dot"></span>{{ status }}</span></div></div>
</div>

<div class="grid">
<div class="card"><div class="k">Closed Trades</div><div class="v">{{stats.total}}</div></div>
<div class="card"><div class="k">Wins</div><div class="v green">{{stats.wins}}</div></div>
<div class="card"><div class="k">Losses</div><div class="v red">{{stats.losses}}</div></div>
<div class="card"><div class="k">Historical Win Rate</div><div class="v">{{"%.1f"|format(stats.win_rate)}}%</div></div>
<div class="card"><div class="k">Average R</div><div class="v blue">{{"%.2f"|format(stats.avg_r)}}R</div></div>
<div class="card"><div class="k">Last Scan</div><div class="v" style="font-size:13px">{{last_scan}}</div></div>
</div>

<div class="section"><h2>20-COIN LIVE MARKET MATRIX</h2><div class="panel scroll"><table>
<tr><th>Coin</th><th>Decision</th><th>Setup Score</th><th>30M Entry</th><th>1D</th><th>4H</th><th>1H</th><th>30M</th><th>RSI 1H</th><th>Pressure 1H</th><th>Volume</th><th>Entry</th><th>SL</th><th>TP1</th><th>TP2</th><th>TP3</th><th>RR3</th><th>Reason</th></tr>
{% for sym in symbols %}{% set m=market.get(sym,{}) %}{% set snap=m.get('snapshot',{}) %}{% set st=m.get('structure',{}) %}{% set p=m.get('pressure',{}) %}{% set lv=m.get('levels',{}) %}
<tr>
<td class="coin">{{sym.replace('USDT','/USDT')}}</td>
<td><span class="pill {{'buy' if m.get('status')=='LONG' else 'sell' if m.get('status')=='SHORT' else 'err' if m.get('status')=='ERROR' else 'wait'}}">{{'BUY' if m.get('status')=='LONG' else 'SELL' if m.get('status')=='SHORT' else m.get('status','NO TRADE')}}</span></td>
<td class="num"><b>{{"%.2f"|format(m.get('score',0))}}</b>/10<div class="bar"><div class="fill" style="width:{{[m.get('score',0)*10,100]|min}}%"></div></div></td>
<td>{{"%.1f"|format(m.get('entry_score',0))}}/10</td>
<td>{{st.get('1d',{}).get('trend','—')}}</td><td>{{st.get('4h',{}).get('trend','—')}}</td><td>{{st.get('1h',{}).get('trend','—')}}</td><td>{{st.get('30m',{}).get('trend','—')}}</td>
<td class="num">{{"%.1f"|format(snap.get('1h',{}).get('rsi',0))}}</td>
<td class="num"><span class="green">B {{"%.0f"|format(p.get('1h_buy',0))}}</span> / <span class="red">S {{"%.0f"|format(p.get('1h_sell',0))}}</span></td>
<td class="num">{{"%.2f"|format(snap.get('1h',{}).get('volume_ratio',0))}}x</td>
<td class="num">{{"%.8g"|format(lv.get('entry',0)) if lv else '—'}}</td><td class="num">{{"%.2f"|format(lv.get('sl_pct',0)) if lv else 0}}%</td>
<td class="num">{{"%.2f"|format(lv.get('tp1_pct',0)) if lv else 0}}%</td><td class="num">{{"%.2f"|format(lv.get('tp2_pct',0)) if lv else 0}}%</td><td class="num">{{"%.2f"|format(lv.get('tp3_pct',0)) if lv else 0}}%</td><td class="num">{{"%.2f"|format(lv.get('rr3',0)) if lv else 0}}R</td>
<td class="reason">{% for r in m.get('reasons',[])[:4] %}<span class="chip">{{r}}</span>{% endfor %}</td>
</tr>{% endfor %}
</table></div></div>

<div class="section"><h2>ACTIVE PAPER TRADES</h2><div class="cards2">
{% for t in active %}<div class="trade"><div class="tradeHead"><b>{{t.symbol}}</b><span class="pill {{'buy' if t.direction=='LONG' else 'sell'}}">{{'BUY' if t.direction=='LONG' else 'SELL'}}</span></div>
<div class="mini" style="margin-top:7px">Entry {{"%.8g"|format(t.entry)}} · Initial SL {{"%.8g"|format(t.sl)}}</div>
<div class="levels"><div class="level"><span class="mini">TP1</span><b>{{"%.8g"|format(t.tp1)}}</b><span class="mini">1:{{"%.2f"|format(t.rr1)}}</span></div><div class="level"><span class="mini">TP2</span><b>{{"%.8g"|format(t.tp2)}}</b><span class="mini">1:{{"%.2f"|format(t.rr2)}}</span></div><div class="level"><span class="mini">TP3</span><b>{{"%.8g"|format(t.tp3)}}</b><span class="mini">1:{{"%.2f"|format(t.rr3)}}</span></div><div class="level"><span class="mini">Protected SL</span><b>{{"%.8g"|format(t.protected_stop or t.sl)}}</b></div></div>
<div class="mini" style="margin-top:9px">TP1 {{t.hit_tp1}} · TP2 {{t.hit_tp2}} · TP3 {{t.hit_tp3}} · Realized {{"%.2f"|format(t.realized_r or 0)}}R · MFE {{"%.2f"|format(t.max_favorable_r or 0)}}R</div>
</div>{% else %}<div class="trade"><span class="muted">No active paper trades.</span></div>{% endfor %}
</div></div>

<div class="section"><h2>LEARNING / FACTOR PERFORMANCE</h2><div class="panel scroll"><table><tr><th>Factor</th><th>Samples</th><th>Win Rate</th><th>Average R</th><th>Adaptive Weight</th></tr>{% for f in stats.factor_stats %}<tr><td>{{f.factor}}</td><td>{{f.n}}</td><td>{{"%.1f"|format(f.wr*100)}}%</td><td>{{"%.2f"|format(f.avg_r)}}R</td><td>{{"%.2f"|format(f.get('weight',0))}}</td></tr>{% else %}<tr><td colspan="5">Learning data will appear after closed trades.</td></tr>{% endfor %}</table></div></div>

<div class="section"><h2>RECENT TRADE HISTORY</h2><div class="panel scroll"><table><tr><th>Pakistan Time</th><th>Coin</th><th>Side</th><th>Setup</th><th>Outcome</th><th>Exit Reason</th><th>P/L</th><th>MFE</th><th>MAE</th></tr>{% for t in history %}<tr><td>{{t.created_at}}</td><td>{{t.symbol}}</td><td>{{'BUY' if t.direction=='LONG' else 'SELL'}}</td><td>{{"%.2f"|format(t.setup_score)}}/10</td><td class="{{'green' if t.outcome=='WIN' else 'red' if t.outcome=='LOSS' else 'amber'}}">{{t.outcome or t.status}}</td><td>{{t.exit_reason or '—'}}</td><td>{{"%.2f"|format(t.pnl_r or 0)}}R</td><td>{{"%.2f"|format(t.max_favorable_r or 0)}}R</td><td>{{"%.2f"|format(t.max_adverse_r or 0)}}R</td></tr>{% endfor %}</table></div></div>

<div class="footer">Professional paper-trading dashboard. No real orders are sent. “Confidence/Setup Score” is a rules/quality score, not a guaranteed win probability. Data decisions use closed candles only. Dashboard timestamps are Pakistan Standard Time (Asia/Karachi, UTC+05:00). Auto-refresh: 30 seconds.</div>
</div>
<script>
function tick(){const d=new Date();document.getElementById('pkt').textContent='Pakistan Time: '+new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Karachi',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(d)+' PKT';}tick();setInterval(tick,1000);setTimeout(()=>location.reload(),30000);
</script>
</body></html>
"""

@app.route("/")
def home():
    with state_lock:
        last_scan = STATE["last_scan"]
        signals = STATE["signals"]
        active = STATE["active_trades"]
    return render_template_string(
        HTML,
        stats=stats(),
        last_scan=last_scan or "Starting...",
        signals=signals,
        active=active,
        history=recent_trades(),
        market=STATE.get("market", {}),
        symbols=SYMBOLS,
        pkt=now_pkt_text(),
        status=STATE.get("status", "STARTING")
    )

@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({
            "last_scan": STATE["last_scan"],
            "scanning": STATE["scanning"],
            "signals": STATE["signals"],
            "active_trades": STATE["active_trades"],
            "errors": STATE["errors"],
            "market": STATE["market"],
            "status": STATE.get("status", "STARTING"),
            "pkt": now_pkt_text(),
            "stats": stats(),
        })

@app.route("/api/trades")
def api_trades():
    return jsonify(recent_trades())

@app.route("/api/weights")
def api_weights():
    return jsonify(adaptive_weights())

# ---------------- MAIN ----------------

if __name__ == "__main__":
    init_db()

    worker = threading.Thread(target=scan_worker, daemon=True)
    worker.start()

    # Replit/container-friendly server.
    app.run(host="0.0.0.0", port=PORT, debug=False)
      
