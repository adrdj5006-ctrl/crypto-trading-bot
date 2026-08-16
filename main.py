# MARKET BRAIN AI — Python / GitHub Edition
# ============================================================
# Multi-timeframe adaptive crypto market scanner
#
# 20 COINS
# 1D  = major bias
# 4H  = structure / liquidity / pressure
# 1H  = main setup / SMC / classical patterns
# 5M  = pure entry confirmation
#
# Includes:
# - HH / HL / LH / LL
# - BOS / CHoCH
# - liquidity sweeps
# - equal highs / equal lows
# - M / W / double top / double bottom
# - head & shoulders / inverse H&S
# - triple top / bottom
# - ascending / descending / symmetrical triangles
# - rising / falling wedges
# - flags / pennants
# - range / rectangle breakout
# - cup / handle approximation
# - FVG
# - order block
# - breaker-style confirmation
# - premium / discount
# - buyer / seller pressure
# - volume
# - EMA / RSI / ATR
# - support / resistance
# - candlestick confirmation
# - dynamic market-based Entry / SL / TP
# - minimum reward:risk protection
# - trade memory
# - adaptive learning from closed trades
# - immediate WIN / LOSS Gmail alert
# - 12-hour Gmail performance report
# - Pakistan time (Asia/Karachi)
#
# IMPORTANT:
# This is an adaptive statistical/scoring engine, NOT a neural network.
# It does not guarantee profit. Test on paper/demo before real money.
# ============================================================

import os
import time
import json
import math
import logging
import smtplib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
PKT = ZoneInfo("Asia/Karachi")

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "SUIUSDT", "OPUSDT", "ARBUSDT", "INJUSDT",
    "APTUSDT", "LTCUSDT", "TRXUSDT", "UNIUSDT", "ATOMUSDT",
]

TIMEFRAMES = {
    "1d": "1d",
    "4h": "4h",
    "1h": "1h",
    "5m": "5m",
}

CANDLE_LIMIT = 250
SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "60"))

# Dynamic risk controls
MIN_RR = float(os.getenv("MIN_RR", "2.0"))
MAX_RR = float(os.getenv("MAX_RR", "5.0"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.25"))
ATR_TARGET_MULT = float(os.getenv("ATR_TARGET_MULT", "2.75"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "65"))
DIRECTION_GAP = float(os.getenv("DIRECTION_GAP", "8"))
MAX_SL_PCT = float(os.getenv("MAX_SL_PCT", "3.0"))
MIN_SL_PCT = float(os.getenv("MIN_SL_PCT", "0.25"))
MAX_TARGET_PCT = float(os.getenv("MAX_TARGET_PCT", "12.0"))

# Learning
LEARNING_FILE = Path("ai_learning.json")
TRADE_MEMORY_FILE = Path("trade_memory.json")
TRADE_CSV_FILE = Path("trade_learning_log.csv")
STATE_FILE = Path("bot_state.json")

# Gmail
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
GMAIL_RECEIVER = os.environ.get("GMAIL_RECEIVER", GMAIL_USER or "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("MARKET_BRAIN")


# ============================================================
# UTILITIES
# ============================================================

def now_pkt():
    return datetime.now(PKT)


def iso_pkt(dt=None):
    return (dt or now_pkt()).isoformat()


def safe_div(a, b, default=0.0):
    try:
        if b is None or float(b) == 0:
            return default
        return float(a) / float(b)
    except Exception:
        return default


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def pct(x):
    return f"{x:.2f}%"


def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
    return default


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ============================================================
# GMAIL
# ============================================================

def send_email(subject, body):
    if not GMAIL_USER:
        logger.error("GMAIL_USER missing.")
        return False
    if not GMAIL_PASS:
        logger.error("GMAIL_PASS missing.")
        return False
    if not GMAIL_RECEIVER:
        logger.error("GMAIL_RECEIVER missing.")
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = GMAIL_USER
        msg["To"] = GMAIL_RECEIVER
        msg.set_content(body)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(msg)

        logger.info("Gmail sent: %s", subject)
        return True
    except Exception as e:
        logger.error("Gmail error: %s", e)
        return False


# ============================================================
# BINANCE DATA
# ============================================================

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MARKET-BRAIN-AI/1.0"})


def fetch_klines(symbol, interval, limit=CANDLE_LIMIT):
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    for attempt in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            r.raise_for_status()
            raw = r.json()

            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]
            df = pd.DataFrame(raw, columns=cols)

            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

            return df.dropna().reset_index(drop=True)
        except Exception as e:
            logger.warning("%s %s attempt %s: %s", symbol, interval, attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))

    return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def add_indicators(df):
    x = df.copy()
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi"] = rsi(x["close"], 14)
    x["atr"] = atr(x, 14)
    x["avg_volume"] = x["volume"].rolling(20).mean()
    x["volume_ratio"] = x["volume"] / x["avg_volume"].replace(0, np.nan)
    x["body"] = (x["close"] - x["open"]).abs()
    x["range"] = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_ratio"] = x["body"] / x["range"]
    x["upper_wick"] = x["high"] - x[["open", "close"]].max(axis=1)
    x["lower_wick"] = x[["open", "close"]].min(axis=1) - x["low"]
    return x


# ============================================================
# PRESSURE
# ============================================================

def pressure(df, length=12):
    d = df.tail(length)
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    strength = (d["close"] - d["open"]).abs() / rng
    weighted = strength.fillna(0) * d["volume"]

    buy = weighted[d["close"] > d["open"]].sum()
    sell = weighted[d["close"] < d["open"]].sum()
    total = buy + sell

    if total <= 0:
        return 50.0, 50.0

    return buy / total * 100, sell / total * 100


# ============================================================
# SWING / STRUCTURE
# ============================================================

def pivots(df, left=3, right=3):
    highs = []
    lows = []

    h = df["high"].values
    l = df["low"].values

    for i in range(left, len(df) - right):
        if h[i] == max(h[i-left:i+right+1]):
            highs.append((i, h[i]))
        if l[i] == min(l[i-left:i+right+1]):
            lows.append((i, l[i]))

    return highs, lows


def structure_info(df):
    highs, lows = pivots(df, 3, 3)

    last_h = prev_h = None
    last_l = prev_l = None

    if len(highs) >= 2:
        prev_h = highs[-2][1]
        last_h = highs[-1][1]

    if len(lows) >= 2:
        prev_l = lows[-2][1]
        last_l = lows[-1][1]

    hh = last_h is not None and prev_h is not None and last_h > prev_h
    lh = last_h is not None and prev_h is not None and last_h < prev_h
    hl = last_l is not None and prev_l is not None and last_l > prev_l
    ll = last_l is not None and prev_l is not None and last_l < prev_l

    close = float(df["close"].iloc[-1])

    bull_bos = last_h is not None and close > last_h and float(df["close"].iloc[-2]) <= last_h
    bear_bos = last_l is not None and close < last_l and float(df["close"].iloc[-2]) >= last_l

    # Previous structure direction is estimated from the last confirmed BOS.
    state = 0
    for i in range(max(20, len(df)-100), len(df)):
        sub = df.iloc[:i+1]
        hs, ls = pivots(sub, 3, 3)
        if hs and float(sub["close"].iloc[-1]) > hs[-1][1]:
            state = 1
        if ls and float(sub["close"].iloc[-1]) < ls[-1][1]:
            state = -1

    bull_choch = bull_bos and state == -1
    bear_choch = bear_bos and state == 1

    return {
        "last_high": last_h,
        "prev_high": prev_h,
        "last_low": last_l,
        "prev_low": prev_l,
        "HH": hh, "LH": lh, "HL": hl, "LL": ll,
        "bull_bos": bull_bos,
        "bear_bos": bear_bos,
        "bull_choch": bull_choch,
        "bear_choch": bear_choch,
        "highs": highs,
        "lows": lows,
    }


# ============================================================
# PATTERN ENGINE
# ============================================================

def detect_patterns(df, st):
    close = float(df["close"].iloc[-1])
    ema20 = float(df["ema20"].iloc[-1])

    last_h = st["last_high"]
    prev_h = st["prev_high"]
    last_l = st["last_low"]
    prev_l = st["prev_low"]

    equal_high = (
        last_h is not None and prev_h is not None and
        abs(last_h-prev_h) / max(last_h, 1e-12) < 0.002
    )
    equal_low = (
        last_l is not None and prev_l is not None and
        abs(last_l-prev_l) / max(last_l, 1e-12) < 0.002
    )

    double_top = (
        last_h is not None and prev_h is not None and
        abs(last_h-prev_h) / max(last_h, 1e-12) < 0.015 and
        close < float(df["low"].iloc[-11:-1].min())
    )

    double_bottom = (
        last_l is not None and prev_l is not None and
        abs(last_l-prev_l) / max(last_l, 1e-12) < 0.015 and
        close > float(df["high"].iloc[-11:-1].max())
    )

    head_shoulders = st["HH"] and st["LH"]
    inverse_hs = st["LL"] and st["HL"]

    triple_top = equal_high and st["LH"] and close < ema20
    triple_bottom = equal_low and st["HL"] and close > ema20

    ascending_triangle = equal_high and st["HL"]
    descending_triangle = equal_low and st["LH"]
    symmetrical_triangle = st["LH"] and st["HL"]

    rising_wedge = st["HH"] and st["HL"] and st["LH"]
    falling_wedge = st["LL"] and st["HL"] and st["LH"]

    bull_flag = (
        ema20 > float(df["ema50"].iloc[-1]) and
        df["low"].tail(8).min() > df["low"].tail(20).min() and
        close > ema20
    )
    bear_flag = (
        ema20 < float(df["ema50"].iloc[-1]) and
        df["high"].tail(8).max() < df["high"].tail(20).max() and
        close < ema20
    )

    range_high = float(df["high"].tail(50).max())
    range_low = float(df["low"].tail(50).min())
    range_width = safe_div(range_high-range_low, close)
    range_market = range_width < 0.06

    range_break_bull = range_market and close > float(df["high"].iloc[-2])
    range_break_bear = range_market and close < float(df["low"].iloc[-2])

    mid = (range_high + range_low) / 2
    cup_handle_bull = close > mid and close > ema20
    cup_handle_bear = close < mid and close < ema20

    return {
        "equal_high": equal_high,
        "equal_low": equal_low,
        "double_top": double_top,
        "double_bottom": double_bottom,
        "head_shoulders": head_shoulders,
        "inverse_hs": inverse_hs,
        "triple_top": triple_top,
        "triple_bottom": triple_bottom,
        "ascending_triangle": ascending_triangle,
        "descending_triangle": descending_triangle,
        "symmetrical_triangle": symmetrical_triangle,
        "rising_wedge": rising_wedge,
        "falling_wedge": falling_wedge,
        "bull_flag": bull_flag,
        "bear_flag": bear_flag,
        "range_market": range_market,
        "range_break_bull": range_break_bull,
        "range_break_bear": range_break_bear,
        "cup_handle_bull": cup_handle_bull,
        "cup_handle_bear": cup_handle_bear,
    }


# ============================================================
# SMC ENGINE
# ============================================================

def detect_smc(df, st):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    op = df["open"]

    bull_fvg = float(low.iloc[-1]) > float(high.iloc[-3])
    bear_fvg = float(high.iloc[-1]) < float(low.iloc[-3])

    bull_ob = (
        float(close.iloc[-1]) > float(op.iloc[-1]) and
        float(close.iloc[-1]) > float(high.iloc[-2]) and
        float(close.iloc[-2]) < float(op.iloc[-2])
    )

    bear_ob = (
        float(close.iloc[-1]) < float(op.iloc[-1]) and
        float(close.iloc[-1]) < float(low.iloc[-2]) and
        float(close.iloc[-2]) > float(op.iloc[-2])
    )

    bull_breaker = (
        len(df) >= 4 and
        float(close.iloc[-2]) < float(op.iloc[-2]) and
        float(close.iloc[-1]) > float(high.iloc[-2])
    )

    bear_breaker = (
        len(df) >= 4 and
        float(close.iloc[-2]) > float(op.iloc[-2]) and
        float(close.iloc[-1]) < float(low.iloc[-2])
    )

    last_low = st["last_low"]
    last_high = st["last_high"]

    bull_sweep = last_low is not None and float(low.iloc[-1]) < last_low and float(close.iloc[-1]) > last_low
    bear_sweep = last_high is not None and float(high.iloc[-1]) > last_high and float(close.iloc[-1]) < last_high

    range_high = float(high.tail(50).max())
    range_low = float(low.tail(50).min())
    equilibrium = (range_high + range_low) / 2
    price = float(close.iloc[-1])

    return {
        "bull_fvg": bull_fvg,
        "bear_fvg": bear_fvg,
        "bull_ob": bull_ob,
        "bear_ob": bear_ob,
        "bull_breaker": bull_breaker,
        "bear_breaker": bear_breaker,
        "bull_sweep": bull_sweep,
        "bear_sweep": bear_sweep,
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": equilibrium,
        "discount": price < equilibrium,
        "premium": price > equilibrium,
    }


# ============================================================
# CANDLESTICK ENGINE
# ============================================================

def candle_patterns(df):
    c = df.iloc[-1]
    p = df.iloc[-2]

    body = abs(float(c["close"] - c["open"]))
    rng = max(float(c["high"] - c["low"]), 1e-12)
    upper = float(c["high"] - max(c["open"], c["close"]))
    lower = float(min(c["open"], c["close"]) - c["low"])

    bull_engulf = (
        c["close"] > c["open"] and
        p["close"] < p["open"] and
        c["close"] >= p["open"] and
        c["open"] <= p["close"]
    )

    bear_engulf = (
        c["close"] < c["open"] and
        p["close"] > p["open"] and
        c["close"] <= p["open"] and
        c["open"] >= p["close"]
    )

    hammer = lower > body * 2 and upper <= max(body, 1e-12)
    shooting_star = upper > body * 2 and lower <= max(body, 1e-12)

    inside_bar = c["high"] < p["high"] and c["low"] > p["low"]
    outside_bar = c["high"] > p["high"] and c["low"] < p["low"]

    bull_rejection = lower > body * 1.5 and c["close"] > c["open"]
    bear_rejection = upper > body * 1.5 and c["close"] < c["open"]

    return {
        "bull_engulf": bull_engulf,
        "bear_engulf": bear_engulf,
        "hammer": hammer,
        "shooting_star": shooting_star,
        "inside_bar": inside_bar,
        "outside_bar": outside_bar,
        "bull_rejection": bull_rejection,
        "bear_rejection": bear_rejection,
        "body_ratio": body / rng,
    }


# ============================================================
# TIMEFRAME CONTEXT
# ============================================================

def timeframe_context(df):
    x = add_indicators(df)
    price = float(x["close"].iloc[-1])

    return {
        "price": price,
        "ema20": float(x["ema20"].iloc[-1]),
        "ema50": float(x["ema50"].iloc[-1]),
        "ema200": float(x["ema200"].iloc[-1]),
        "rsi": float(x["rsi"].iloc[-1]),
        "atr": float(x["atr"].iloc[-1]),
        "volume_ratio": float(x["volume_ratio"].iloc[-1]) if not pd.isna(x["volume_ratio"].iloc[-1]) else 1.0,
        "bull": price > float(x["ema20"].iloc[-1]) > float(x["ema50"].iloc[-1]),
        "bear": price < float(x["ema20"].iloc[-1]) < float(x["ema50"].iloc[-1]),
        "buyer_pressure": pressure(x)[0],
        "seller_pressure": pressure(x)[1],
        "df": x,
    }

# ============================================================
# ADAPTIVE LEARNING
# ============================================================

DEFAULT_LEARNING = {
    "global": {"wins": 0, "losses": 0, "win_rate": 50.0},
    "symbols": {},
    "setups": {},
    "directions": {"BUY": {"wins": 0, "losses": 0}, "SELL": {"wins": 0, "losses": 0}},
    "market_regimes": {},
}


def load_learning():
    data = load_json(LEARNING_FILE, DEFAULT_LEARNING.copy())
    return data


LEARNING = load_learning()


def ensure_symbol_learning(symbol):
    LEARNING.setdefault("symbols", {})
    LEARNING["symbols"].setdefault(symbol, {
        "wins": 0, "losses": 0, "profit_r": 0.0
    })


def ensure_setup_learning(name):
    LEARNING.setdefault("setups", {})
    LEARNING["setups"].setdefault(name, {
        "wins": 0, "losses": 0, "profit_r": 0.0
    })


def current_learning_factor(symbol, setup_names, direction):
    ensure_symbol_learning(symbol)

    records = []
    s = LEARNING["symbols"][symbol]
    if s["wins"] + s["losses"] >= 5:
        records.append(s["wins"] / (s["wins"] + s["losses"]) * 100)

    for name in setup_names:
        ensure_setup_learning(name)
        x = LEARNING["setups"][name]
        if x["wins"] + x["losses"] >= 5:
            records.append(x["wins"] / (x["wins"] + x["losses"]) * 100)

    d = LEARNING["directions"].setdefault(direction, {"wins": 0, "losses": 0})
    if d["wins"] + d["losses"] >= 5:
        records.append(d["wins"] / (d["wins"] + d["losses"]) * 100)

    if not records:
        return 1.0

    avg = float(np.mean(records))
    if avg >= 65:
        return 1.05
    if avg <= 40:
        return 0.95
    return 1.0


def update_learning(trade):
    symbol = trade["symbol"]
    direction = trade["direction"]
    result = trade["result"]
    r_multiple = float(trade.get("r_multiple", 0))

    ensure_symbol_learning(symbol)

    d = LEARNING["directions"].setdefault(direction, {"wins": 0, "losses": 0})

    if result == "WIN":
        LEARNING["global"]["wins"] += 1
        LEARNING["symbols"][symbol]["wins"] += 1
        d["wins"] += 1
    elif result == "LOSS":
        LEARNING["global"]["losses"] += 1
        LEARNING["symbols"][symbol]["losses"] += 1
        d["losses"] += 1

    LEARNING["symbols"][symbol]["profit_r"] += r_multiple

    for setup in trade.get("setups", []):
        ensure_setup_learning(setup)
        LEARNING["setups"][setup]["profit_r"] += r_multiple
        if result == "WIN":
            LEARNING["setups"][setup]["wins"] += 1
        elif result == "LOSS":
            LEARNING["setups"][setup]["losses"] += 1

    total = LEARNING["global"]["wins"] + LEARNING["global"]["losses"]
    LEARNING["global"]["win_rate"] = (
        LEARNING["global"]["wins"] / total * 100 if total else 50.0
    )
    save_json(LEARNING_FILE, LEARNING)


# ============================================================
# SCORE ENGINE
# ============================================================

def score_market(symbol, d1, h4, h1, m5):
    buy = 0.0
    sell = 0.0
    reasons_buy = []
    reasons_sell = []
    setup_names = []

    def add(side, points, reason):
        nonlocal buy, sell
        if side == "BUY":
            buy += points
            reasons_buy.append(reason)
        else:
            sell += points
            reasons_sell.append(reason)

    # 1D
    if d1["bull"]:
        add("BUY", 5, "1D BULLISH")
    if d1["bear"]:
        add("SELL", 5, "1D BEARISH")

    # 4H
    if h4["bull"]:
        add("BUY", 7, "4H BULLISH")
    if h4["bear"]:
        add("SELL", 7, "4H BEARISH")

    # 1H
    if h1["bull"]:
        add("BUY", 8, "1H TREND")
    if h1["bear"]:
        add("SELL", 8, "1H TREND")

    # Pressure
    if h1["buyer_pressure"] >= 55:
        add("BUY", 10, "BUYER PRESSURE")
    if h1["seller_pressure"] >= 55:
        add("SELL", 10, "SELLER PRESSURE")

    # Volume
    if h1["volume_ratio"] >= 1.10 and h1["price"] > h1["ema20"]:
        add("BUY", 8, "BUY VOLUME")
    if h1["volume_ratio"] >= 1.10 and h1["price"] < h1["ema20"]:
        add("SELL", 8, "SELL VOLUME")

    st = structure_info(h1["df"])

    if st["HH"] and st["HL"]:
        add("BUY", 8, "HH/HL")
    if st["LH"] and st["LL"]:
        add("SELL", 8, "LH/LL")

    if st["bull_bos"]:
        add("BUY", 12, "BULL BOS")
        setup_names.append("BOS")
    if st["bear_bos"]:
        add("SELL", 12, "BEAR BOS")
        setup_names.append("BOS")
    if st["bull_choch"]:
        add("BUY", 12, "BULL CHoCH")
        setup_names.append("CHoCH")
    if st["bear_choch"]:
        add("SELL", 12, "BEAR CHoCH")
        setup_names.append("CHoCH")

    smc = detect_smc(h1["df"], st)

    if smc["bull_sweep"]:
        add("BUY", 10, "LIQUIDITY SWEEP")
        setup_names.append("LIQUIDITY SWEEP")
    if smc["bear_sweep"]:
        add("SELL", 10, "LIQUIDITY SWEEP")
        setup_names.append("LIQUIDITY SWEEP")

    if smc["discount"]:
        add("BUY", 3, "DISCOUNT")
    if smc["premium"]:
        add("SELL", 3, "PREMIUM")

    if smc["bull_fvg"]:
        add("BUY", 7, "BULL FVG")
        setup_names.append("FVG")
    if smc["bear_fvg"]:
        add("SELL", 7, "BEAR FVG")
        setup_names.append("FVG")

    if smc["bull_ob"]:
        add("BUY", 7, "BULL ORDER BLOCK")
        setup_names.append("ORDER BLOCK")
    if smc["bear_ob"]:
        add("SELL", 7, "BEAR ORDER BLOCK")
        setup_names.append("ORDER BLOCK")

    if smc["bull_breaker"]:
        add("BUY", 5, "BULL BREAKER")
        setup_names.append("BREAKER")
    if smc["bear_breaker"]:
        add("SELL", 5, "BEAR BREAKER")
        setup_names.append("BREAKER")

    # Support / resistance
    support = float(h1["df"]["low"].iloc[-51:-1].min())
    resistance = float(h1["df"]["high"].iloc[-51:-1].max())
    price = h1["price"]

    if abs(price-support)/max(price, 1e-12) <= 0.008:
        add("BUY", 8, "SUPPORT")
        setup_names.append("SUPPORT")
    if abs(resistance-price)/max(price, 1e-12) <= 0.008:
        add("SELL", 8, "RESISTANCE")
        setup_names.append("RESISTANCE")

    # RSI
    if 45 <= h1["rsi"] <= 68 and h1["bull"]:
        add("BUY", 4, "RSI")
    if 32 <= h1["rsi"] <= 55 and h1["bear"]:
        add("SELL", 4, "RSI")

    candles = candle_patterns(h1["df"])
    if candles["bull_engulf"] or candles["hammer"] or candles["bull_rejection"]:
        add("BUY", 4, "BULL PRICE ACTION")
        setup_names.append("PRICE ACTION")
    if candles["bear_engulf"] or candles["shooting_star"] or candles["bear_rejection"]:
        add("SELL", 4, "BEAR PRICE ACTION")
        setup_names.append("PRICE ACTION")

    patterns = detect_patterns(h1["df"], st)

    pattern_scores = [
        ("double_bottom", "BUY", 10, "W / DOUBLE BOTTOM"),
        ("double_top", "SELL", 10, "M / DOUBLE TOP"),
        ("inverse_hs", "BUY", 9, "INVERSE H&S"),
        ("head_shoulders", "SELL", 9, "HEAD & SHOULDERS"),
        ("triple_bottom", "BUY", 7, "TRIPLE BOTTOM"),
        ("triple_top", "SELL", 7, "TRIPLE TOP"),
        ("ascending_triangle", "BUY", 6, "ASCENDING TRIANGLE"),
        ("descending_triangle", "SELL", 6, "DESCENDING TRIANGLE"),
        ("falling_wedge", "BUY", 6, "FALLING WEDGE"),
        ("rising_wedge", "SELL", 6, "RISING WEDGE"),
        ("bull_flag", "BUY", 5, "BULL FLAG"),
        ("bear_flag", "SELL", 5, "BEAR FLAG"),
        ("range_break_bull", "BUY", 7, "RANGE BREAKOUT"),
        ("range_break_bear", "SELL", 7, "RANGE BREAKDOWN"),
        ("cup_handle_bull", "BUY", 4, "CUP HANDLE"),
        ("cup_handle_bear", "SELL", 4, "INVERSE CUP"),
    ]

    for key, side, pts, label in pattern_scores:
        if patterns[key]:
            add(side, pts, label)
            setup_names.append(label)

    if patterns["symmetrical_triangle"]:
        if h1["bull"]:
            add("BUY", 5, "BULL SYMMETRICAL TRIANGLE")
        if h1["bear"]:
            add("SELL", 5, "BEAR SYMMETRICAL TRIANGLE")

    # Learning
    factor_buy = current_learning_factor(symbol, setup_names, "BUY")
    factor_sell = current_learning_factor(symbol, setup_names, "SELL")

    buy *= factor_buy
    sell *= factor_sell

    buy = clamp(buy, 0, 100)
    sell = clamp(sell, 0, 100)

    if buy >= MIN_SCORE and buy >= sell + DIRECTION_GAP:
        direction = "BUY"
    elif sell >= MIN_SCORE and sell >= buy + DIRECTION_GAP:
        direction = "SELL"
    else:
        direction = "NO TRADE"

    # 5M confirmation is intentionally only used after higher-timeframe setup.
    m5_confirm_buy = (
        m5["price"] > m5["ema20"] and
        m5["body_ratio"] >= 0.25 and
        m5["volume_ratio"] >= 1.10
    )
    m5_confirm_sell = (
        m5["price"] < m5["ema20"] and
        m5["body_ratio"] >= 0.25 and
        m5["volume_ratio"] >= 1.10
    )

    final_direction = direction
    if direction == "BUY" and not m5_confirm_buy:
        final_direction = "BUY WAIT 5M"
    elif direction == "SELL" and not m5_confirm_sell:
        final_direction = "SELL WAIT 5M"

    return {
        "buy_score": round(buy, 2),
        "sell_score": round(sell, 2),
        "direction": final_direction,
        "raw_direction": direction,
        "m5_buy_confirm": m5_confirm_buy,
        "m5_sell_confirm": m5_confirm_sell,
        "buy_reasons": reasons_buy,
        "sell_reasons": reasons_sell,
        "setups": list(dict.fromkeys(setup_names)),
        "support": support,
        "resistance": resistance,
        "structure": st,
        "smc": smc,
        "patterns": patterns,
        "candles": candles,
    }


# ============================================================
# DYNAMIC ENTRY / STOP / TARGET
# ============================================================

def nearest_support_resistance(df, structure):
    lows = [x[1] for x in structure["lows"][-8:]]
    highs = [x[1] for x in structure["highs"][-8:]]

    price = float(df["close"].iloc[-1])

    supports = [x for x in lows if x < price]
    resistances = [x for x in highs if x > price]

    support = max(supports) if supports else float(df["low"].tail(20).min())
    resistance = min(resistances) if resistances else float(df["high"].tail(20).max())

    return support, resistance


def dynamic_levels(direction, h1, m5, score):
    price = float(m5["price"])
    atr_value = max(float(h1["atr"]), price * 0.001)

    support, resistance = nearest_support_resistance(
        h1["df"],
        structure_info(h1["df"])
    )

    if direction == "BUY":
        structural_sl = support - atr_value * 0.25
        atr_sl = price - atr_value * ATR_SL_MULT
        stop = min(structural_sl, atr_sl)

        # Do not put SL absurdly far away.
        max_stop = price * (1 - MAX_SL_PCT/100)
        min_stop = price * (1 - MIN_SL_PCT/100)
        stop = clamp(stop, max_stop, min_stop)

        risk = price - stop

        structural_target = resistance
        atr_target = price + atr_value * ATR_TARGET_MULT
        target = max(structural_target, atr_target)

        min_target = price + risk * MIN_RR
        target = max(target, min_target)

        max_target = price * (1 + MAX_TARGET_PCT/100)
        target = min(target, max_target)

    else:
        structural_sl = resistance + atr_value * 0.25
        atr_sl = price + atr_value * ATR_SL_MULT
        stop = max(structural_sl, atr_sl)

        max_stop = price * (1 + MAX_SL_PCT/100)
        min_stop = price * (1 + MIN_SL_PCT/100)
        stop = clamp(stop, min_stop, max_stop)

        risk = stop - price

        structural_target = support
        atr_target = price - atr_value * ATR_TARGET_MULT
        target = min(structural_target, atr_target)

        min_target = price - risk * MIN_RR
        target = min(target, min_target)

        max_target = price * (1 - MAX_TARGET_PCT/100)
        target = max(target, max_target)

    risk_pct = abs(price-stop)/price*100
    reward_pct = abs(target-price)/price*100
    rr = safe_div(reward_pct, risk_pct)

    # Safety rule: target must be greater than risk.
    valid = rr >= MIN_RR and rr <= MAX_RR and risk_pct <= MAX_SL_PCT

    return {
        "entry": price,
        "stop_loss": stop,
        "target": target,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "rr": rr,
        "valid": valid,
        "support": support,
        "resistance": resistance,
        "atr": atr_value,
        "score": score,
    }


# ============================================================
# TRADE MEMORY
# ============================================================

def load_trades():
    return load_json(TRADE_MEMORY_FILE, [])


TRADES = load_trades()


def save_trade(trade):
    global TRADES
    TRADES.append(trade)
    save_json(TRADE_MEMORY_FILE, TRADES)

    row = {
        k: trade.get(k)
        for k in [
            "id", "symbol", "direction", "entry", "stop_loss", "target",
            "entry_time", "exit_time", "result", "exit_price",
            "pnl_pct", "r_multiple", "buy_score", "sell_score",
            "rr", "setups"
        ]
    }

    df = pd.DataFrame([row])
    header = not TRADE_CSV_FILE.exists()
    df.to_csv(TRADE_CSV_FILE, mode="a", header=header, index=False)


def make_trade_id(symbol):
    return f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"


# ============================================================
# OPEN TRADE MANAGEMENT
# ============================================================

OPEN_TRADES = {}


def trade_alert(trade):
    body = f"""
🧠 MARKET BRAIN AI — NEW TRADE

Symbol: {trade['symbol']}
Direction: {trade['direction']}

Entry Price: {trade['entry']:.8f}
Stop Loss: {trade['stop_loss']:.8f}
Target: {trade['target']:.8f}

Risk: {trade['risk_pct']:.2f}%
Reward: {trade['reward_pct']:.2f}%
R:R: 1:{trade['rr']:.2f}

BUY Score: {trade['buy_score']:.1f}
SELL Score: {trade['sell_score']:.1f}

1D Bias: {trade['d1_bias']}
4H Bias: {trade['h4_bias']}
1H Bias: {trade['h1_bias']}

Buyer Pressure: {trade['buyer_pressure']:.1f}%
Seller Pressure: {trade['seller_pressure']:.1f}%
Volume: {trade['volume_ratio']:.2f}x
RSI: {trade['rsi']:.1f}

FVG: {trade['fvg']}
Order Block: {trade['order_block']}
Liquidity: {trade['liquidity']}
Pattern: {trade['pattern']}

Support: {trade['support']:.8f}
Resistance: {trade['resistance']:.8f}
ATR: {trade['atr']:.8f}

Setups:
{', '.join(trade['setups']) if trade['setups'] else 'None'}

Time: {trade['entry_time']} PKT

⚠️ This is an algorithmic signal, not a guarantee.
"""
    send_email(
        f"🧠 MARKET BRAIN {trade['direction']} — {trade['symbol']}",
        body.strip()
    )


def closed_trade_alert(trade):
    emoji = "✅" if trade["result"] == "WIN" else "❌"

    body = f"""
🧠 MARKET BRAIN AI — TRADE CLOSED

{emoji} Result: {trade['result']}

Symbol: {trade['symbol']}
Direction: {trade['direction']}

Entry: {trade['entry']:.8f}
Exit: {trade['exit_price']:.8f}

Stop Loss: {trade['stop_loss']:.8f}
Target: {trade['target']:.8f}

P/L: {trade['pnl_pct']:.2f}%
R Multiple: {trade['r_multiple']:.2f}R
R:R Planned: 1:{trade['rr']:.2f}

Entry Time: {trade['entry_time']} PKT
Exit Time: {trade['exit_time']} PKT

BUY Score: {trade['buy_score']:.1f}
SELL Score: {trade['sell_score']:.1f}

Setups:
{', '.join(trade.get('setups', [])) or 'None'}

The result has been added to Trade Memory
and the learning engine.
"""
    send_email(
        f"{emoji} MARKET BRAIN {trade['result']} — {trade['symbol']}",
        body.strip()
    )


def check_open_trades():
    for trade_id, trade in list(OPEN_TRADES.items()):
        df = fetch_klines(trade["symbol"], "5m", 5)
        if df.empty:
            continue

        # Use latest completed/available candle range for simulation.
        high = float(df["high"].iloc[-1])
        low = float(df["low"].iloc[-1])

        result = None
        exit_price = None

        if trade["direction"] == "BUY":
            # Conservative: if both are touched in same candle,
            # assume SL first.
            if low <= trade["stop_loss"]:
                result = "LOSS"
                exit_price = trade["stop_loss"]
            elif high >= trade["target"]:
                result = "WIN"
                exit_price = trade["target"]

        else:
            if high >= trade["stop_loss"]:
                result = "LOSS"
                exit_price = trade["stop_loss"]
            elif low <= trade["target"]:
                result = "WIN"
                exit_price = trade["target"]

        if result:
            if trade["direction"] == "BUY":
                pnl_pct = (exit_price-trade["entry"]) / trade["entry"] * 100
            else:
                pnl_pct = (trade["entry"]-exit_price) / trade["entry"] * 100

            risk_pct = trade["risk_pct"]
            r_multiple = pnl_pct / risk_pct if risk_pct else 0

            trade["exit_price"] = exit_price
            trade["exit_time"] = iso_pkt()
            trade["result"] = result
            trade["pnl_pct"] = pnl_pct
            trade["r_multiple"] = r_multiple

            save_trade(trade)
            update_learning(trade)
            closed_trade_alert(trade)

            del OPEN_TRADES[trade_id]
            save_json(STATE_FILE, {"open_trades": list(OPEN_TRADES.values())})


# ============================================================
# 12-HOUR REPORT
# ============================================================

LAST_REPORT_FILE = Path("last_12h_report.json")


def report_due():
    last = load_json(LAST_REPORT_FILE, {})
    if not last:
        return True

    last_time = last.get("timestamp")
    if not last_time:
        return True

    try:
        dt = datetime.fromisoformat(last_time)
        return now_pkt() - dt >= timedelta(hours=12)
    except Exception:
        return True


def send_12h_report():
    global TRADES

    end = now_pkt()
    start = end - timedelta(hours=12)

    completed = []

    for t in TRADES:
        if t.get("result") not in ("WIN", "LOSS"):
            continue
        try:
            et = datetime.fromisoformat(t["exit_time"])
            if et >= start and et <= end:
                completed.append(t)
        except Exception:
            pass

    wins = sum(t["result"] == "WIN" for t in completed)
    losses = sum(t["result"] == "LOSS" for t in completed)
    total = wins + losses
    win_rate = wins / total * 100 if total else 0
    net_r = sum(float(t.get("r_multiple", 0)) for t in completed)
    net_pct = sum(float(t.get("pnl_pct", 0)) for t in completed)

    by_symbol = {}
    for t in completed:
        s = t["symbol"]
        by_symbol.setdefault(s, {"wins": 0, "losses": 0, "r": 0.0})
        by_symbol[s]["wins"] += int(t["result"] == "WIN")
        by_symbol[s]["losses"] += int(t["result"] == "LOSS")
        by_symbol[s]["r"] += float(t.get("r_multiple", 0))

    symbol_lines = []
    for s, v in sorted(by_symbol.items(), key=lambda kv: kv[1]["r"], reverse=True):
        n = v["wins"] + v["losses"]
        wr = v["wins"]/n*100 if n else 0
        symbol_lines.append(
            f"{s}: {n} trades | W {v['wins']} | L {v['losses']} | "
            f"WR {wr:.1f}% | {v['r']:+.2f}R"
        )

    setup_stats = {}
    for t in completed:
        for setup in t.get("setups", []):
            setup_stats.setdefault(setup, {"w": 0, "l": 0, "r": 0})
            setup_stats[setup]["w"] += int(t["result"] == "WIN")
            setup_stats[setup]["l"] += int(t["result"] == "LOSS")
            setup_stats[setup]["r"] += float(t.get("r_multiple", 0))

    setup_lines = []
    for s, v in sorted(setup_stats.items(), key=lambda kv: kv[1]["r"], reverse=True)[:10]:
        n = v["w"] + v["l"]
        wr = v["w"]/n*100 if n else 0
        setup_lines.append(f"{s}: {n} | WR {wr:.1f}% | {v['r']:+.2f}R")

    best_symbol = max(by_symbol, key=lambda x: by_symbol[x]["r"]) if by_symbol else "N/A"
    worst_symbol = min(by_symbol, key=lambda x: by_symbol[x]["r"]) if by_symbol else "N/A"

    body = f"""
🧠 MARKET BRAIN AI — 12 HOUR REPORT

Period:
{start.strftime('%d-%b-%Y %H:%M')} → {end.strftime('%d-%b-%Y %H:%M')} PKT

━━━━━━━━━━━━━━━━━━━
    PERFORMANCE
━━━━━━━━━━━━━━━━━━━━

Total Closed Trades: {total}
Wins: {wins}
Losses: {losses}
Win Rate: {win_rate:.2f}%

Net P/L: {net_pct:+.2f}%
Net R: {net_r:+.2f}R

━━━━━━━━━━━━━━━━━━━━
BEST / WORST COIN
━━━━━━━━━━━━━━━━━━━━

Best: {best_symbol}
Worst: {worst_symbol}

━━━━━━━━━━━━━━━━━━━━
COIN BREAKDOWN
━━━━━━━━━━━━━━━━━━━━

{chr(10).join(symbol_lines) if symbol_lines else 'No completed trades in this period.'}

━━━━━━━━━━━━━━━━━━━━
BEST SETUPS
━━━━━━━━━━━━━━━━━━━━

{chr(10).join(setup_lines) if setup_lines else 'No setup statistics yet.'}

━━━━━━━━━━━━━━━━━━━━
GLOBAL LEARNING
━━━━━━━━━━━━━━━━━━━━

Lifetime Wins: {LEARNING['global']['wins']}
Lifetime Losses: {LEARNING['global']['losses']}
Lifetime Win Rate: {LEARNING['global']['win_rate']:.2f}%

Learning Status: ACTIVE

The next scan will use the updated
historical statistics where enough
sample data exists.

━━━━━━━━━━━━━━━━━━━━

Report Time:
{end.strftime('%d-%b-%Y %H:%M:%S')} PKT
"""

    send_email(
        "🧠 MARKET BRAIN AI — 12 HOUR REPORT",
        body.strip()
    )

    save_json(LAST_REPORT_FILE, {"timestamp": iso_pkt()})


# ============================================================
# ANALYZE ONE SYMBOL
# ============================================================

def analyze_symbol(symbol):
    frames = {}

    for key, interval in TIMEFRAMES.items():
        df = fetch_klines(symbol, interval)
        if df.empty or len(df) < 210:
            return None
        frames[key] = timeframe_context(df)

    d1 = frames["1d"]
    h4 = frames["4h"]
    h1 = frames["1h"]
    m5 = frames["5m"]

    score = score_market(symbol, d1, h4, h1, m5)

    # Higher timeframes must not be ignored.
    if score["raw_direction"] == "BUY" and not (d1["bull"] or h4["bull"] or h1["bull"]):
        return None
    if score["raw_direction"] == "SELL" and not (d1["bear"] or h4["bear"] or h1["bear"]):
        return None

    direction = score["raw_direction"]

    if direction not in ("BUY", "SELL"):
        return {
            "symbol": symbol,
            "score": score,
            "d1": d1, "h4": h4, "h1": h1, "m5": m5,
        }

    # Pure 5M entry confirmation.
    if direction == "BUY" and not score["m5_buy_confirm"]:
        return {
            "symbol": symbol,
            "score": score,
            "d1": d1, "h4": h4, "h1": h1, "m5": m5,
        }

    if direction == "SELL" and not score["m5_sell_confirm"]:
        return {
            "symbol": symbol,
            "score": score,
            "d1": d1, "h4": h4, "h1": h1, "m5": m5,
        }

    levels = dynamic_levels(direction, h1, m5, max(score["buy_score"], score["sell_score"]))

    return {
        "symbol": symbol,
        "score": score,
        "d1": d1, "h4": h4, "h1": h1, "m5": m5,
        "levels": levels,
    }


# ============================================================
# TRADE CREATION
# ============================================================

def open_trade(result):
    symbol = result["symbol"]
    score = result["score"]
    levels = result["levels"]

    direction = score["raw_direction"]

    if not levels["valid"]:
        return False

    # One position per symbol.
    if any(t["symbol"] == symbol for t in OPEN_TRADES.values()):
        return False

    pattern = "NONE"
    p = score["patterns"]
    for key, name in [
        ("double_bottom", "W / DOUBLE BOTTOM"),
        ("double_top", "M / DOUBLE TOP"),
        ("inverse_hs", "INVERSE H&S"),
        ("head_shoulders", "HEAD & SHOULDERS"),
        ("ascending_triangle", "ASC TRIANGLE"),
        ("descending_triangle", "DESC TRIANGLE"),
        ("falling_wedge", "FALLING WEDGE"),
        ("rising_wedge", "RISING WEDGE"),
        ("bull_flag", "BULL FLAG"),
        ("bear_flag", "BEAR FLAG"),
    ]:
        if p.get(key):
            pattern = name
            break

    smc = score["smc"]

    trade = {
        "id": make_trade_id(symbol),
        "symbol": symbol,
        "direction": direction,
        "entry": levels["entry"],
        "stop_loss": levels["stop_loss"],
        "target": levels["target"],
        "risk_pct": levels["risk_pct"],
        "reward_pct": levels["reward_pct"],
        "rr": levels["rr"],
        "entry_time": iso_pkt(),

        "buy_score": score["buy_score"],
        "sell_score": score["sell_score"],

        "d1_bias": "BULLISH" if result["d1"]["bull"] else "BEARISH" if result["d1"]["bear"] else "NEUTRAL",
        "h4_bias": "BULLISH" if result["h4"]["bull"] else "BEARISH" if result["h4"]["bear"] else "NEUTRAL",
        "h1_bias": "BULLISH" if result["h1"]["bull"] else "BEARISH" if result["h1"]["bear"] else "NEUTRAL",

        "buyer_pressure": result["h1"]["buyer_pressure"],
        "seller_pressure": result["h1"]["seller_pressure"],
        "volume_ratio": result["h1"]["volume_ratio"],
        "rsi": result["h1"]["rsi"],

        "fvg": "BULLISH" if smc["bull_fvg"] else "BEARISH" if smc["bear_fvg"] else "NONE",
        "order_block": "BULLISH" if smc["bull_ob"] else "BEARISH" if smc["bear_ob"] else "NONE",
        "liquidity": "BUY SWEEP" if smc["bull_sweep"] else "SELL SWEEP" if smc["bear_sweep"] else "NONE",
        "pattern": pattern,

        "support": levels["support"],
        "resistance": levels["resistance"],
        "atr": levels["atr"],

        "setups": score["setups"],
        "result": "OPEN",
    }

    OPEN_TRADES[trade["id"]] = trade
    save_json(STATE_FILE, {"open_trades": list(OPEN_TRADES.values())})
    trade_alert(trade)

    logger.info(
        "OPEN %s %s | score %.1f/%.1f | entry %.8f | SL %.8f | TP %.8f | RR %.2f",
        symbol, direction,
        score["buy_score"], score["sell_score"],
        levels["entry"], levels["stop_loss"],
        levels["target"], levels["rr"]
    )

    return True


# ============================================================
# RESTORE OPEN TRADES
# ============================================================

def restore_open_trades():
    global OPEN_TRADES
    state = load_json(STATE_FILE, {})
    for t in state.get("open_trades", []):
        if t.get("result") == "OPEN":
            OPEN_TRADES[t["id"]] = t


# ============================================================
# RANKING
# ============================================================

def result_rank(x):
    score = x["score"]
    return max(score["buy_score"], score["sell_score"])


# ============================================================
# MAIN SCAN
# ============================================================

def scan_all():
    candidates = []

    logger.info("Scanning %d coins...", len(SYMBOLS))

    for symbol in SYMBOLS:
        try:
            result = analyze_symbol(symbol)
            if result is None:
                continue

            score = result["score"]

            logger.info(
                "%s | BUY %.1f | SELL %.1f | %s",
                symbol,
                score["buy_score"],
                score["sell_score"],
                score["direction"]
            )

            if (
                score["raw_direction"] in ("BUY", "SELL")
                and score["m5_buy_confirm"] if score["raw_direction"] == "BUY"
                else score["raw_direction"] in ("BUY", "SELL")
                and score["m5_sell_confirm"]
            ):
                if "levels" in result and result["levels"]["valid"]:
                    candidates.append(result)

        except Exception as e:
            logger.exception("Analysis failed for %s: %s", symbol, e)

    candidates.sort(key=result_rank, reverse=True)

    # Only top opportunities are allowed per scan.
    max_new_trades = int(os.getenv("MAX_NEW_TRADES_PER_SCAN", "3"))

    opened = 0
    for result in candidates:
        if opened >= max_new_trades:
            break
        if open_trade(result):
            opened += 1

    logger.info("Scan complete. Candidates=%d, opened=%d", len(candidates), opened)


# ============================================================
# STARTUP
# ============================================================

def main():
    restore_open_trades()

    logger.info("==============================================")
    logger.info("🧠 MARKET BRAIN AI STARTED")
    logger.info("20 COINS | 1D / 4H / 1H / 5M")
    logger.info("Dynamic SL/TP | Learning | Gmail | 12H Report")
    logger.info("Pakistan Time: %s", now_pkt().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("==============================================")

    while True:
        try:
            check_open_trades()

            if report_due():
                send_12h_report()

            scan_all()

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.exception("Main loop error: %s", e)

        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    main()
