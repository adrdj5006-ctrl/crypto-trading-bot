# MARKET BRAIN AI — Python / GitHub Edition v2
# ============================================================
# 1D = major bias
# 4H = structure / liquidity / zones / target map
# 1H = setup + entry + SL + TP
#
# NO 5M ANALYSIS
#
# Features:
# - HH / HL / LH / LL
# - BOS / CHoCH
# - liquidity sweeps
# - equal highs / lows
# - double top / bottom
# - head & shoulders / inverse H&S (structure approximation)
# - triangles / wedges / flags
# - range breakouts
# - FVG / Order Block / breaker-style confirmation
# - premium / discount
# - pressure / volume / EMA / RSI / ATR
# - support / resistance
# - dynamic 1H entry
# - 4H + 1H structural SL/TP
# - minimum 2R reward:risk
# - adaptive statistical learning
# - detailed trade memory
# - MAE / MFE tracking
# - immediate NEW TRADE Gmail alert
# - immediate WIN / LOSS / SL HIT Gmail alert
# - persistent state for GitHub restarts
#
# IMPORTANT:
# A 70-80% win rate cannot be guaranteed by code.
# The engine is designed to increase trade opportunity while
# rejecting weak/poor-RR setups and learning from closed trades.
# ============================================================

import os
import time
import json
import logging
import smtplib
from datetime import datetime, timedelta, timezone
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
}

CANDLE_LIMIT = 300
SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "60"))

# More opportunity than the old version, but still protected.
MIN_SCORE = float(os.getenv("MIN_SCORE", "58"))
DIRECTION_GAP = float(os.getenv("DIRECTION_GAP", "5"))

# Risk / reward
MIN_RR = float(os.getenv("MIN_RR", "2.0"))
MAX_RR = float(os.getenv("MAX_RR", "4.5"))
MIN_SL_PCT = float(os.getenv("MIN_SL_PCT", "0.20"))
MAX_SL_PCT = float(os.getenv("MAX_SL_PCT", "3.50"))
MAX_TARGET_PCT = float(os.getenv("MAX_TARGET_PCT", "15.0"))

# ATR is a buffer, not the main structural decision.
ATR_SL_BUFFER = float(os.getenv("ATR_SL_BUFFER", "0.25"))
ATR_TARGET_MULT = float(os.getenv("ATR_TARGET_MULT", "2.50"))

# Trade frequency / exposure
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "6"))
MAX_NEW_TRADES_PER_SCAN = int(os.getenv("MAX_NEW_TRADES_PER_SCAN", "2"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "90"))

# Entry quality
MIN_1H_VOLUME_RATIO = float(os.getenv("MIN_1H_VOLUME_RATIO", "0.85"))
MIN_1H_BODY_RATIO = float(os.getenv("MIN_1H_BODY_RATIO", "0.20"))

# Learning
LEARNING_FILE = Path("ai_learning.json")
TRADE_MEMORY_FILE = Path("trade_memory.json")
TRADE_CSV_FILE = Path("trade_learning_log.csv")
STATE_FILE = Path("bot_state.json")

# Gmail
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
GMAIL_RECEIVER = os.getenv("GMAIL_RECEIVER", GMAIL_USER or "")

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


def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
    return default


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    tmp.replace(path)


def parse_dt(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=PKT)
        return dt
    except Exception:
        return None


# ============================================================
# GMAIL
# ============================================================

def send_email(subject, body):
    if not GMAIL_USER or not GMAIL_PASS or not GMAIL_RECEIVER:
        logger.warning("Gmail credentials/receiver missing. Alert skipped.")
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
SESSION.headers.update({"User-Agent": "MARKET-BRAIN-AI/2.0"})


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

            df["open_time"] = pd.to_datetime(
                df["open_time"], unit="ms", utc=True
            )
            df["close_time"] = pd.to_datetime(
                df["close_time"], unit="ms", utc=True
            )

            return df.dropna().reset_index(drop=True)

        except Exception as e:
            logger.warning(
                "%s %s attempt %s: %s",
                symbol, interval, attempt + 1, e
            )
            time.sleep(1.5 * (attempt + 1))

    return pd.DataFrame()


def fetch_price(symbol):
    try:
        url = f"{BASE_URL}/api/v3/ticker/price"
        r = SESSION.get(
            url,
            params={"symbol": symbol},
            timeout=10
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        logger.warning("Price fetch failed %s: %s", symbol, e)
        return None


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
    x["volume_ratio"] = (
        x["volume"] /
        x["avg_volume"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    x["body"] = (x["close"] - x["open"]).abs()
    x["range"] = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_ratio"] = (
        x["body"] / x["range"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    x["upper_wick"] = (
        x["high"] - x[["open", "close"]].max(axis=1)
    )
    x["lower_wick"] = (
        x[["open", "close"]].min(axis=1) - x["low"]
    )

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
# SWING / MARKET STRUCTURE
# ============================================================

def pivots(df, left=3, right=3):
    highs = []
    lows = []

    h = df["high"].values
    l = df["low"].values

    for i in range(left, len(df) - right):
        if h[i] == max(h[i-left:i+right+1]):
            highs.append((i, float(h[i])))

        if l[i] == min(l[i-left:i+right+1]):
            lows.append((i, float(l[i])))

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

    hh = bool(last_h is not None and prev_h is not None and last_h > prev_h)
    lh = bool(last_h is not None and prev_h is not None and last_h < prev_h)
    hl = bool(last_l is not None and prev_l is not None and last_l > prev_l)
    ll = bool(last_l is not None and prev_l is not None and last_l < prev_l)

    close_now = float(df["close"].iloc[-1])
    close_prev = float(df["close"].iloc[-2])

    bull_bos = (
        last_h is not None and
        close_now > last_h and
        close_prev <= last_h
    )

    bear_bos = (
        last_l is not None and
        close_now < last_l and
        close_prev >= last_l
    )

    # Approximate prior direction using recent confirmed structure.
    state = 0
    start = max(20, len(df) - 100)

    for i in range(start, len(df)):
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
        "HH": hh,
        "HL": hl,
        "LH": lh,
        "LL": ll,
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
    ema20_now = float(df["ema20"].iloc[-1])
    ema50_now = float(df["ema50"].iloc[-1])

    last_h = st["last_high"]
    prev_h = st["prev_high"]
    last_l = st["last_low"]
    prev_l = st["prev_low"]

    equal_high = (
        last_h is not None and
        prev_h is not None and
        abs(last_h - prev_h) / max(last_h, 1e-12) < 0.003
    )

    equal_low = (
        last_l is not None and
        prev_l is not None and
        abs(last_l - prev_l) / max(last_l, 1e-12) < 0.003
    )

    recent_low = float(df["low"].iloc[-11:-1].min())
    recent_high = float(df["high"].iloc[-11:-1].max())

    double_top = (
        equal_high and
        close < recent_low
    )

    double_bottom = (
        equal_low and
        close > recent_high
    )

    head_shoulders = st["HH"] and st["LH"] and st["LL"]
    inverse_hs = st["LL"] and st["HL"] and st["HH"]

    triple_top = equal_high and st["LH"] and close < ema20_now
    triple_bottom = equal_low and st["HL"] and close > ema20_now

    ascending_triangle = equal_high and st["HL"]
    descending_triangle = equal_low and st["LH"]
    symmetrical_triangle = st["LH"] and st["HL"]

    rising_wedge = st["HH"] and st["HL"] and st["LH"]
    falling_wedge = st["LL"] and st["HL"] and st["LH"]

    bull_flag = (
        ema20_now > ema50_now and
        float(df["low"].tail(8).min()) >
        float(df["low"].tail(20).min()) and
        close > ema20_now
    )

    bear_flag = (
        ema20_now < ema50_now and
        float(df["high"].tail(8).max()) <
        float(df["high"].tail(20).max()) and
        close < ema20_now
    )

    range_high = float(df["high"].tail(50).max())
    range_low = float(df["low"].tail(50).min())
    range_width = safe_div(range_high - range_low, close)

    range_market = range_width < 0.06

    previous_range_high = float(df["high"].iloc[-51:-1].max())
    previous_range_low = float(df["low"].iloc[-51:-1].min())

    range_break_bull = (
        range_market and
        close > previous_range_high
    )

    range_break_bear = (
        range_market and
        close < previous_range_low
    )

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
    }


# ============================================================
# SMC
# ============================================================

def detect_smc(df, st):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    op = df["open"]

    bull_fvg = (
        len(df) >= 3 and
        float(low.iloc[-1]) > float(high.iloc[-3])
    )

    bear_fvg = (
        len(df) >= 3 and
        float(high.iloc[-1]) < float(low.iloc[-3])
    )

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
        float(close.iloc[-2]) < float(op.iloc[-2]) and
        float(close.iloc[-1]) > float(high.iloc[-2])
    )

    bear_breaker = (
        float(close.iloc[-2]) > float(op.iloc[-2]) and
        float(close.iloc[-1]) < float(low.iloc[-2])
    )

    last_low = st["last_low"]
    last_high = st["last_high"]

    bull_sweep = (
        last_low is not None and
        float(low.iloc[-1]) < last_low and
        float(close.iloc[-1]) > last_low
    )

    bear_sweep = (
        last_high is not None and
        float(high.iloc[-1]) > last_high and
        float(close.iloc[-1]) < last_high
    )

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
# CANDLE ENGINE
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

    hammer = (
        lower > body * 2 and
        upper <= max(body, 1e-12)
    )

    shooting_star = (
        upper > body * 2 and
        lower <= max(body, 1e-12)
    )

    bull_rejection = (
        lower > body * 1.5 and
        c["close"] > c["open"]
    )

    bear_rejection = (
        upper > body * 1.5 and
        c["close"] < c["open"]
    )

    return {
        "bull_engulf": bool(bull_engulf),
        "bear_engulf": bool(bear_engulf),
        "hammer": bool(hammer),
        "shooting_star": bool(shooting_star),
        "bull_rejection": bool(bull_rejection),
        "bear_rejection": bool(bear_rejection),
        "body_ratio": body / rng,
    }


# ============================================================
# TIMEFRAME CONTEXT
# ============================================================

def timeframe_context(df):
    x = add_indicators(df)
    price = float(x["close"].iloc[-1])

    buyer, seller = pressure(x)

    bull = (
        price > float(x["ema20"].iloc[-1]) >
        float(x["ema50"].iloc[-1])
    )

    bear = (
        price < float(x["ema20"].iloc[-1]) <
        float(x["ema50"].iloc[-1])
    )

    return {
        "price": price,
        "ema20": float(x["ema20"].iloc[-1]),
        "ema50": float(x["ema50"].iloc[-1]),
        "ema200": float(x["ema200"].iloc[-1]),
        "rsi": float(x["rsi"].iloc[-1]),
        "atr": float(x["atr"].iloc[-1]),
        "volume_ratio": float(x["volume_ratio"].iloc[-1]),
        "body_ratio": float(x["body_ratio"].iloc[-1]),
        "bull": bull,
        "bear": bear,
        "buyer_pressure": buyer,
        "seller_pressure": seller,
        "df": x,
    }


# ============================================================
# LEARNING
# ============================================================

DEFAULT_LEARNING = {
    "global": {
        "wins": 0,
        "losses": 0,
        "win_rate": 50.0
    },
    "symbols": {},
    "setups": {},
    "features": {},
    "directions": {
        "BUY": {"wins": 0, "losses": 0},
        "SELL": {"wins": 0, "losses": 0}
    }
}

LEARNING = load_json(LEARNING_FILE, DEFAULT_LEARNING)


def ensure_symbol_learning(symbol):
    LEARNING.setdefault("symbols", {})
    LEARNING["symbols"].setdefault(
        symbol,
        {"wins": 0, "losses": 0, "profit_r": 0.0}
    )


def ensure_setup_learning(name):
    LEARNING.setdefault("setups", {})
    LEARNING["setups"].setdefault(
        name,
        {"wins": 0, "losses": 0, "profit_r": 0.0}
    )


def current_learning_factor(symbol, setup_names, direction):
    ensure_symbol_learning(symbol)

    rates = []

    s = LEARNING["symbols"][symbol]
    total = s["wins"] + s["losses"]

    if total >= 8:
        rates.append(s["wins"] / total * 100)

    for name in setup_names:
        ensure_setup_learning(name)
        x = LEARNING["setups"][name]
        total = x["wins"] + x["losses"]

        if total >= 8:
            rates.append(x["wins"] / total * 100)

    d = LEARNING["directions"].setdefault(
        direction,
        {"wins": 0, "losses": 0}
    )
    total = d["wins"] + d["losses"]

    if total >= 8:
        rates.append(d["wins"] / total * 100)

    if not rates:
        return 1.0

    avg = float(np.mean(rates))

    if avg >= 68:
        return 1.06
    if avg <= 38:
        return 0.94

    return 1.0


def update_learning(trade):
    symbol = trade["symbol"]
    direction = trade["direction"]
    result = trade["result"]
    r_multiple = float(trade.get("r_multiple", 0))

    ensure_symbol_learning(symbol)

    d = LEARNING["directions"].setdefault(
        direction,
        {"wins": 0, "losses": 0}
    )

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

    total = (
        LEARNING["global"]["wins"] +
        LEARNING["global"]["losses"]
    )

    LEARNING["global"]["win_rate"] = (
        LEARNING["global"]["wins"] / total * 100
        if total else 50.0
    )

    save_json(LEARNING_FILE, LEARNING)


# ============================================================
# SCORE ENGINE
# ============================================================

def score_market(symbol, d1, h4, h1):
    buy = 0.0
    sell = 0.0

    reasons_buy = []
    reasons_sell = []
    setup_names = []

    def add(side, points, reason, setup=None):
        nonlocal buy, sell

        if side == "BUY":
            buy += points
            reasons_buy.append(reason)
        else:
            sell += points
            reasons_sell.append(reason)

        if setup:
            setup_names.append(setup)

    # --------------------------------------------------------
    # 1D = SOFT MAJOR BIAS
    # --------------------------------------------------------
    if d1["bull"]:
        add("BUY", 12, "1D BULLISH", "1D BIAS")
    elif d1["bear"]:
        add("SELL", 12, "1D BEARISH", "1D BIAS")

    # --------------------------------------------------------
    # 4H = STRUCTURE / LIQUIDITY
    # --------------------------------------------------------
    st4 = structure_info(h4["df"])

    if h4["bull"]:
        add("BUY", 8, "4H BULLISH", "4H TREND")

    if h4["bear"]:
        add("SELL", 8, "4H BEARISH", "4H TREND")

    if st4["HH"] and st4["HL"]:
        add("BUY", 10, "4H HH/HL", "4H STRUCTURE")

    if st4["LH"] and st4["LL"]:
        add("SELL", 10, "4H LH/LL", "4H STRUCTURE")

    if st4["bull_bos"]:
        add("BUY", 10, "4H BULL BOS", "4H BOS")

    if st4["bear_bos"]:
        add("SELL", 10, "4H BEAR BOS", "4H BOS")

    if st4["bull_choch"]:
        add("BUY", 8, "4H BULL CHoCH", "4H CHoCH")

    if st4["bear_choch"]:
        add("SELL", 8, "4H BEAR CHoCH", "4H CHoCH")

    smc4 = detect_smc(h4["df"], st4)

    if smc4["bull_sweep"]:
        add("BUY", 9, "4H LIQUIDITY SWEEP", "LIQUIDITY SWEEP")

    if smc4["bear_sweep"]:
        add("SELL", 9, "4H LIQUIDITY SWEEP", "LIQUIDITY SWEEP")

    if smc4["bull_fvg"]:
        add("BUY", 6, "4H BULL FVG", "FVG")

    if smc4["bear_fvg"]:
        add("SELL", 6, "4H BEAR FVG", "FVG")

    if smc4["bull_ob"]:
        add("BUY", 6, "4H BULL ORDER BLOCK", "ORDER BLOCK")

    if smc4["bear_ob"]:
        add("SELL", 6, "4H BEAR ORDER BLOCK", "ORDER BLOCK")

    if smc4["discount"]:
        add("BUY", 3, "4H DISCOUNT")

    if smc4["premium"]:
        add("SELL", 3, "4H PREMIUM")

    # --------------------------------------------------------
    # 1H = MAIN SETUP / ENTRY
    # --------------------------------------------------------
    st1 = structure_info(h1["df"])
    smc1 = detect_smc(h1["df"], st1)
    candles = candle_patterns(h1["df"])
    patterns = detect_patterns(h1["df"], st1)

    if h1["bull"]:
        add("BUY", 8, "1H BULLISH", "1H TREND")

    if h1["bear"]:
        add("SELL", 8, "1H BEARISH", "1H TREND")

    if st1["HH"] and st1["HL"]:
        add("BUY", 8, "1H HH/HL", "1H STRUCTURE")

    if st1["LH"] and st1["LL"]:
        add("SELL", 8, "1H LH/LL", "1H STRUCTURE")

    if st1["bull_bos"]:
        add("BUY", 11, "1H BULL BOS", "BOS")

    if st1["bear_bos"]:
        add("SELL", 11, "1H BEAR BOS", "BOS")

    if st1["bull_choch"]:
        add("BUY", 10, "1H BULL CHoCH", "CHoCH")

    if st1["bear_choch"]:
        add("SELL", 10, "1H BEAR CHoCH", "CHoCH")

    if smc1["bull_sweep"]:
        add("BUY", 9, "1H BUY LIQUIDITY SWEEP", "LIQUIDITY SWEEP")

    if smc1["bear_sweep"]:
        add("SELL", 9, "1H SELL LIQUIDITY SWEEP", "LIQUIDITY SWEEP")

    if smc1["bull_fvg"]:
        add("BUY", 6, "1H BULL FVG", "FVG")

    if smc1["bear_fvg"]:
        add("SELL", 6, "1H BEAR FVG", "FVG")

    if smc1["bull_ob"]:
        add("BUY", 6, "1H BULL ORDER BLOCK", "ORDER BLOCK")

    if smc1["bear_ob"]:
        add("SELL", 6, "1H BEAR ORDER BLOCK", "ORDER BLOCK")

    if smc1["bull_breaker"]:
        add("BUY", 4, "1H BULL BREAKER", "BREAKER")

    if smc1["bear_breaker"]:
        add("SELL", 4, "1H BEAR BREAKER", "BREAKER")

    # Price action
    if (
        candles["bull_engulf"] or
        candles["hammer"] or
        candles["bull_rejection"]
    ):
        add("BUY", 5, "1H BULL PRICE ACTION", "PRICE ACTION")

    if (
        candles["bear_engulf"] or
        candles["shooting_star"] or
        candles["bear_rejection"]
    ):
        add("SELL", 5, "1H BEAR PRICE ACTION", "PRICE ACTION")

    # Pressure
    if h1["buyer_pressure"] >= 52:
        add("BUY", 5, "1H BUYER PRESSURE", "PRESSURE")

    if h1["seller_pressure"] >= 52:
        add("SELL", 5, "1H SELLER PRESSURE", "PRESSURE")

    # Volume
    if h1["volume_ratio"] >= 1.05 and h1["price"] > h1["ema20"]:
        add("BUY", 5, "1H BUY VOLUME", "VOLUME")

    if h1["volume_ratio"] >= 1.05 and h1["price"] < h1["ema20"]:
        add("SELL", 5, "1H SELL VOLUME", "VOLUME")

    # RSI is confirmation, not a primary trigger.
    if 43 <= h1["rsi"] <= 70 and h1["bull"]:
        add("BUY", 3, "1H RSI SUPPORT")

    if 30 <= h1["rsi"] <= 57 and h1["bear"]:
        add("SELL", 3, "1H RSI SUPPORT")

    # Classical patterns
    pattern_scores = [
        ("double_bottom", "BUY", 8, "DOUBLE BOTTOM", "DOUBLE BOTTOM"),
        ("double_top", "SELL", 8, "DOUBLE TOP", "DOUBLE TOP"),
        ("inverse_hs", "BUY", 7, "INVERSE H&S", "INVERSE H&S"),
        ("head_shoulders", "SELL", 7, "HEAD & SHOULDERS", "HEAD & SHOULDERS"),
        ("triple_bottom", "BUY", 6, "TRIPLE BOTTOM", "TRIPLE BOTTOM"),
        ("triple_top", "SELL", 6, "TRIPLE TOP", "TRIPLE TOP"),
        ("ascending_triangle", "BUY", 5, "ASCENDING TRIANGLE", "TRIANGLE"),
        ("descending_triangle", "SELL", 5, "DESCENDING TRIANGLE", "TRIANGLE"),
        ("falling_wedge", "BUY", 5, "FALLING WEDGE", "WEDGE"),
        ("rising_wedge", "SELL", 5, "RISING WEDGE", "WEDGE"),
        ("bull_flag", "BUY", 4, "BULL FLAG", "FLAG"),
        ("bear_flag", "SELL", 4, "BEAR FLAG", "FLAG"),
        ("range_break_bull", "BUY", 6, "RANGE BREAKOUT", "RANGE BREAK"),
        ("range_break_bear", "SELL", 6, "RANGE BREAKDOWN", "RANGE BREAK"),
    ]

    for key, side, points, label, setup in pattern_scores:
        if patterns.get(key):
            add(side, points, label, setup)

    # Learning factor is intentionally mild.
    factor_buy = current_learning_factor(
        symbol, list(dict.fromkeys(setup_names)), "BUY"
    )
    factor_sell = current_learning_factor(
        symbol, list(dict.fromkeys(setup_names)), "SELL"
    )

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

    return {
        "buy_score": round(buy, 2),
        "sell_score": round(sell, 2),
        "direction": direction,
        "buy_reasons": reasons_buy,
        "sell_reasons": reasons_sell,
        "setups": list(dict.fromkeys(setup_names)),
        "st4": st4,
        "st1": st1,
        "smc4": smc4,
        "smc1": smc1,
        "patterns": patterns,
        "candles": candles,
}
    # ============================================================
# LEVEL ENGINE
# ============================================================

def nearest_levels(df, structure, price):
    lows = [x[1] for x in structure["lows"][-12:]]
    highs = [x[1] for x in structure["highs"][-12:]]

    supports = [x for x in lows if x < price]
    resistances = [x for x in highs if x > price]

    support = (
        max(supports)
        if supports
        else float(df["low"].tail(30).min())
    )

    resistance = (
        min(resistances)
        if resistances
        else float(df["high"].tail(30).max())
    )

    return float(support), float(resistance)


def build_levels(direction, h4, h1, score):
    price = float(h1["price"])

    atr1 = max(
        float(h1["atr"]),
        price * 0.001
    )

    st4 = structure_info(h4["df"])
    st1 = structure_info(h1["df"])

    s4, r4 = nearest_levels(
        h4["df"], st4, price
    )

    s1, r1 = nearest_levels(
        h1["df"], st1, price
    )

    # Combine nearby structure zones.
    if direction == "BUY":
        supports = [x for x in [s1, s4] if x < price]
        resistances = [x for x in [r1, r4] if x > price]

        support = max(supports) if supports else price - atr1
        target_zone = min(resistances) if resistances else price + atr1 * 3

        # SL below the strongest nearby structural support.
        stop = support - atr1 * ATR_SL_BUFFER

        # If structural stop is too tight, use ATR minimum.
        min_structural_stop = price - atr1 * 0.75
        stop = min(stop, min_structural_stop)

        # Keep stop within configured risk bounds.
        max_stop = price * (1 - MIN_SL_PCT / 100)
        far_stop = price * (1 - MAX_SL_PCT / 100)
        stop = clamp(stop, far_stop, max_stop)

        risk = price - stop

        # Target must respect 4H/1H resistance but also guarantee 2R.
        min_target = price + risk * MIN_RR
        atr_target = price + atr1 * ATR_TARGET_MULT

        target = max(target_zone, atr_target, min_target)

        max_target = price * (1 + MAX_TARGET_PCT / 100)
        target = min(target, max_target)

    else:
        supports = [x for x in [s1, s4] if x < price]
        resistances = [x for x in [r1, r4] if x > price]

        resistance = min(resistances) if resistances else price + atr1
        target_zone = max(supports) if supports else price - atr1 * 3

        stop = resistance + atr1 * ATR_SL_BUFFER

        max_structural_stop = price + atr1 * 0.75
        stop = max(stop, max_structural_stop)

        near_stop = price * (1 + MIN_SL_PCT / 100)
        far_stop = price * (1 + MAX_SL_PCT / 100)
        stop = clamp(stop, near_stop, far_stop)

        risk = stop - price

        min_target = price - risk * MIN_RR
        atr_target = price - atr1 * ATR_TARGET_MULT

        target = min(target_zone, atr_target, min_target)

        max_target = price * (1 - MAX_TARGET_PCT / 100)
        target = max(target, max_target)

    risk_pct = abs(price - stop) / price * 100
    reward_pct = abs(target - price) / price * 100
    rr = safe_div(reward_pct, risk_pct)

    valid = (
        risk_pct >= MIN_SL_PCT and
        risk_pct <= MAX_SL_PCT and
        rr >= MIN_RR and
        rr <= MAX_RR
    )

    return {
        "entry": price,
        "stop_loss": float(stop),
        "target": float(target),
        "risk_pct": float(risk_pct),
        "reward_pct": float(reward_pct),
        "rr": float(rr),
        "valid": bool(valid),
        "support_1h": s1,
        "resistance_1h": r1,
        "support_4h": s4,
        "resistance_4h": r4,
        "atr": atr1,
        "score": score,
    }


# ============================================================
# TRADE MEMORY
# ============================================================

TRADES = load_json(TRADE_MEMORY_FILE, [])
OPEN_TRADES = {}
LAST_ENTRY_BY_SYMBOL = {}


def save_trade(trade):
    global TRADES

    TRADES.append(trade)
    save_json(TRADE_MEMORY_FILE, TRADES)

    row_keys = [
        "id", "symbol", "direction",
        "entry", "stop_loss", "target",
        "entry_time", "exit_time",
        "result", "exit_reason",
        "exit_price", "pnl_pct",
        "r_multiple", "planned_rr",
        "mae_pct", "mfe_pct",
        "mae_r", "mfe_r",
        "buy_score", "sell_score",
        "setups"
    ]

    row = {k: trade.get(k) for k in row_keys}

    pd.DataFrame([row]).to_csv(
        TRADE_CSV_FILE,
        mode="a",
        header=not TRADE_CSV_FILE.exists(),
        index=False
    )


def make_trade_id(symbol):
    return (
        f"{symbol}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    )


def save_state():
    save_json(
        STATE_FILE,
        {
            "open_trades": list(OPEN_TRADES.values()),
            "last_entry_by_symbol": LAST_ENTRY_BY_SYMBOL,
            "updated_at": iso_pkt(),
        }
    )


# ============================================================
# ALERTS
# ============================================================

def trade_alert(trade):
    body = f"""
🧠 MARKET BRAIN AI — NEW TRADE

Symbol: {trade['symbol']}
Direction: {trade['direction']}

ENTRY: {trade['entry']:.8f}
STOP LOSS: {trade['stop_loss']:.8f}
TARGET: {trade['target']:.8f}

Risk: {trade['risk_pct']:.2f}%
Reward: {trade['reward_pct']:.2f}%
R:R: 1:{trade['planned_rr']:.2f}

BUY SCORE: {trade['buy_score']:.1f}
SELL SCORE: {trade['sell_score']:.1f}

1D: {trade['d1_bias']}
4H: {trade['h4_bias']}
1H: {trade['h1_bias']}

Buyer Pressure: {trade['buyer_pressure']:.1f}%
Seller Pressure: {trade['seller_pressure']:.1f}%
Volume: {trade['volume_ratio']:.2f}x
RSI: {trade['rsi']:.1f}

FVG: {trade['fvg']}
ORDER BLOCK: {trade['order_block']}
LIQUIDITY: {trade['liquidity']}
PATTERN: {trade['pattern']}

4H Support: {trade['support_4h']:.8f}
4H Resistance: {trade['resistance_4h']:.8f}

1H Support: {trade['support_1h']:.8f}
1H Resistance: {trade['resistance_1h']:.8f}

Setups:
{', '.join(trade['setups']) if trade['setups'] else 'None'}

Time: {trade['entry_time']} PKT
"""

    send_email(
        f"🧠 MARKET BRAIN NEW {trade['direction']} — {trade['symbol']}",
        body.strip()
    )


def closed_trade_alert(trade):
    if trade["result"] == "WIN":
        title = "✅ WIN"
        reason = "TARGET HIT"
    else:
        title = "❌ LOSS"
        reason = trade.get("exit_reason", "STOP LOSS HIT")

    body = f"""
🧠 MARKET BRAIN AI — TRADE CLOSED

{title}

Symbol: {trade['symbol']}
Direction: {trade['direction']}

ENTRY: {trade['entry']:.8f}
EXIT: {trade['exit_price']:.8f}

STOP LOSS: {trade['stop_loss']:.8f}
TARGET: {trade['target']:.8f}

RESULT: {trade['result']}
REASON: {reason}

P/L: {trade['pnl_pct']:+.2f}%
RESULT: {trade['r_multiple']:+.2f}R
PLANNED R:R: 1:{trade['planned_rr']:.2f}

MAE: {trade['mae_pct']:.2f}% ({trade['mae_r']:.2f}R)
MFE: {trade['mfe_pct']:.2f}% ({trade['mfe_r']:.2f}R)

ENTRY TIME:
{trade['entry_time']} PKT

EXIT TIME:
{trade['exit_time']} PKT

BUY SCORE: {trade['buy_score']:.1f}
SELL SCORE: {trade['sell_score']:.1f}

SETUPS:
{', '.join(trade.get('setups', [])) or 'None'}

The trade has been saved to Trade Memory
and the adaptive learning engine.
"""

    send_email(
        f"{title} MARKET BRAIN — {trade['symbol']}",
        body.strip()
    )


# ============================================================
# TRADE CREATION
# ============================================================

def build_trade(result):
    symbol = result["symbol"]
    score = result["score"]
    levels = result["levels"]

    direction = score["direction"]

    if direction not in ("BUY", "SELL"):
        return None

    if not levels["valid"]:
        return None

    if len(OPEN_TRADES) >= MAX_OPEN_TRADES:
        return None

    # One open position per symbol.
    if any(
        t["symbol"] == symbol
        for t in OPEN_TRADES.values()
    ):
        return None

    # Cooldown prevents duplicate entries on repeated scans.
    last_entry = LAST_ENTRY_BY_SYMBOL.get(symbol)
    if last_entry:
        dt = parse_dt(last_entry)
        if dt and now_pkt() - dt < timedelta(
            minutes=COOLDOWN_MINUTES
        ):
            return None

    p = score["patterns"]

    pattern = "NONE"

    for key, name in [
        ("double_bottom", "DOUBLE BOTTOM"),
        ("double_top", "DOUBLE TOP"),
        ("inverse_hs", "INVERSE H&S"),
        ("head_shoulders", "HEAD & SHOULDERS"),
        ("ascending_triangle", "ASCENDING TRIANGLE"),
        ("descending_triangle", "DESCENDING TRIANGLE"),
        ("falling_wedge", "FALLING WEDGE"),
        ("rising_wedge", "RISING WEDGE"),
        ("bull_flag", "BULL FLAG"),
        ("bear_flag", "BEAR FLAG"),
    ]:
        if p.get(key):
            pattern = name
            break

    smc1 = score["smc1"]

    trade = {
        "id": make_trade_id(symbol),
        "symbol": symbol,
        "direction": direction,

        "entry": levels["entry"],
        "stop_loss": levels["stop_loss"],
        "target": levels["target"],

        "risk_pct": levels["risk_pct"],
        "reward_pct": levels["reward_pct"],
        "planned_rr": levels["rr"],

        "entry_time": iso_pkt(),

        "buy_score": score["buy_score"],
        "sell_score": score["sell_score"],

        "d1_bias": (
            "BULLISH"
            if result["d1"]["bull"]
            else "BEARISH"
            if result["d1"]["bear"]
            else "NEUTRAL"
        ),

        "h4_bias": (
            "BULLISH"
            if result["h4"]["bull"]
            else "BEARISH"
            if result["h4"]["bear"]
            else "NEUTRAL"
        ),

        "h1_bias": (
            "BULLISH"
            if result["h1"]["bull"]
            else "BEARISH"
            if result["h1"]["bear"]
            else "NEUTRAL"
        ),

        "buyer_pressure": result["h1"]["buyer_pressure"],
        "seller_pressure": result["h1"]["seller_pressure"],
        "volume_ratio": result["h1"]["volume_ratio"],
        "rsi": result["h1"]["rsi"],

        "fvg": (
            "BULLISH"
            if smc1["bull_fvg"]
            else "BEARISH"
            if smc1["bear_fvg"]
            else "NONE"
        ),

        "order_block": (
            "BULLISH"
            if smc1["bull_ob"]
            else "BEARISH"
            if smc1["bear_ob"]
            else "NONE"
        ),

        "liquidity": (
            "BUY SWEEP"
            if smc1["bull_sweep"]
            else "SELL SWEEP"
            if smc1["bear_sweep"]
            else "NONE"
        ),

        "pattern": pattern,

        "support_1h": levels["support_1h"],
        "resistance_1h": levels["resistance_1h"],
        "support_4h": levels["support_4h"],
        "resistance_4h": levels["resistance_4h"],

        "atr": levels["atr"],
        "setups": score["setups"],

        # Live performance memory
        "highest_price": levels["entry"],
        "lowest_price": levels["entry"],
        "mae_pct": 0.0,
        "mfe_pct": 0.0,

        "result": "OPEN",
    }

    return trade


def open_trade(result):
    trade = build_trade(result)

    if not trade:
        return False

    OPEN_TRADES[trade["id"]] = trade
    LAST_ENTRY_BY_SYMBOL[trade["symbol"]] = trade["entry_time"]

    save_state()
    trade_alert(trade)

    logger.info(
        "OPEN %s %s | BUY %.1f | SELL %.1f | "
        "ENTRY %.8f | SL %.8f | TP %.8f | RR %.2f",
        trade["symbol"],
        trade["direction"],
        trade["buy_score"],
        trade["sell_score"],
        trade["entry"],
        trade["stop_loss"],
        trade["target"],
        trade["planned_rr"],
    )

    return True


# ============================================================
# OPEN TRADE MONITOR
# ============================================================

def update_trade_extremes(trade, current_price):
    entry = float(trade["entry"])
    price = float(current_price)

    if trade["direction"] == "BUY":
        favorable = (price - entry) / entry * 100
        adverse = (entry - price) / entry * 100
    else:
        favorable = (entry - price) / entry * 100
        adverse = (price - entry) / entry * 100

    trade["mfe_pct"] = max(
        float(trade.get("mfe_pct", 0)),
        favorable
    )

    trade["mae_pct"] = max(
        float(trade.get("mae_pct", 0)),
        adverse
    )

    if price > float(trade.get("highest_price", entry)):
        trade["highest_price"] = price

    if price < float(trade.get("lowest_price", entry)):
        trade["lowest_price"] = price


def close_trade(trade_id, trade, result, exit_price, reason):
    entry = float(trade["entry"])
    exit_price = float(exit_price)

    if trade["direction"] == "BUY":
        pnl_pct = (exit_price - entry) / entry * 100
    else:
        pnl_pct = (entry - exit_price) / entry * 100

    risk_pct = float(trade["risk_pct"])
    r_multiple = safe_div(pnl_pct, risk_pct)

    trade["exit_price"] = exit_price
    trade["exit_time"] = iso_pkt()
    trade["result"] = result
    trade["exit_reason"] = reason
    trade["pnl_pct"] = pnl_pct
    trade["r_multiple"] = r_multiple

    trade["mae_r"] = safe_div(
        trade.get("mae_pct", 0),
        risk_pct
    )

    trade["mfe_r"] = safe_div(
        trade.get("mfe_pct", 0),
        risk_pct
    )

    save_trade(trade)
    update_learning(trade)
    closed_trade_alert(trade)

    del OPEN_TRADES[trade_id]
    save_state()

    logger.info(
        "CLOSED %s %s %s | P/L %.2f%% | %.2fR | %s",
        trade["symbol"],
        trade["direction"],
        result,
        pnl_pct,
        r_multiple,
        reason,
    )


def check_open_trades():
    for trade_id, trade in list(OPEN_TRADES.items()):
        current_price = fetch_price(trade["symbol"])

        if current_price is None:
            continue

        update_trade_extremes(
            trade,
            current_price
        )

        if trade["direction"] == "BUY":
            if current_price <= trade["stop_loss"]:
                close_trade(
                    trade_id,
                    trade,
                    "LOSS",
                    trade["stop_loss"],
                    "STOP LOSS HIT"
                )
                continue

            if current_price >= trade["target"]:
                close_trade(
                    trade_id,
                    trade,
                    "WIN",
                    trade["target"],
                    "TARGET HIT"
                )
                continue

        else:
            if current_price >= trade["stop_loss"]:
                close_trade(
                    trade_id,
                    trade,
                    "LOSS",
                    trade["stop_loss"],
                    "STOP LOSS HIT"
                )
                continue

            if current_price <= trade["target"]:
                close_trade(
                    trade_id,
                    trade,
                    "WIN",
                    trade["target"],
                    "TARGET HIT"
                )
                continue

    save_state()


# ============================================================
# ANALYZE ONE SYMBOL
# ============================================================

def analyze_symbol(symbol):
    frames = {}

    for key, interval in TIMEFRAMES.items():
        df = fetch_klines(symbol, interval)

        if df.empty or len(df) < 220:
            return None

        # Use completed candles for analysis where possible.
        # Drop the currently forming candle when it is still open.
        if len(df) > 5:
            last_close_time = df["close_time"].iloc[-1]
            now_utc = datetime.now(timezone.utc)

            if last_close_time.to_pydatetime() > now_utc:
                df = df.iloc[:-1].copy()

        frames[key] = timeframe_context(df)

    d1 = frames["1d"]
    h4 = frames["4h"]
    h1 = frames["1h"]

    score = score_market(
        symbol,
        d1,
        h4,
        h1
    )

    if score["direction"] not in ("BUY", "SELL"):
        return {
            "symbol": symbol,
            "score": score,
            "d1": d1,
            "h4": h4,
            "h1": h1,
        }

    # Entry quality: 1H must have enough activity,
    # but volume is not required to be 1.10x every time.
    if h1["volume_ratio"] < MIN_1H_VOLUME_RATIO:
        return {
            "symbol": symbol,
            "score": score,
            "d1": d1,
            "h4": h4,
            "h1": h1,
        }

    if h1["body_ratio"] < MIN_1H_BODY_RATIO:
        return {
            "symbol": symbol,
            "score": score,
            "d1": d1,
            "h4": h4,
            "h1": h1,
        }

    levels = build_levels(
        score["direction"],
        h4,
        h1,
        max(
            score["buy_score"],
            score["sell_score"]
        )
    )

    return {
        "symbol": symbol,
        "score": score,
        "d1": d1,
        "h4": h4,
        "h1": h1,
        "levels": levels,
    }


# ============================================================
# SCAN
# ============================================================

def result_rank(result):
    score = result["score"]

    strength = (
        max(score["buy_score"], score["sell_score"])
        + result["levels"]["rr"] * 2
    )

    return strength


def scan_all():
    candidates = []

    logger.info(
        "Scanning %d coins | 1D / 4H / 1H",
        len(SYMBOLS)
    )

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
                score["direction"] in ("BUY", "SELL")
                and "levels" in result
                and result["levels"]["valid"]
            ):
                candidates.append(result)

        except Exception as e:
            logger.exception(
                "Analysis failed for %s: %s",
                symbol,
                e
            )

    candidates.sort(
        key=result_rank,
        reverse=True
    )

    opened = 0

    for result in candidates:
        if opened >= MAX_NEW_TRADES_PER_SCAN:
            break

        if len(OPEN_TRADES) >= MAX_OPEN_TRADES:
            break

        if open_trade(result):
            opened += 1

    logger.info(
        "Scan complete | candidates=%d | opened=%d | open=%d",
        len(candidates),
        opened,
        len(OPEN_TRADES)
    )


# ============================================================
# RESTORE
# ============================================================

def restore_state():
    global OPEN_TRADES, LAST_ENTRY_BY_SYMBOL

    state = load_json(
        STATE_FILE,
        {}
    )

    for trade in state.get("open_trades", []):
        if trade.get("result") == "OPEN":
            OPEN_TRADES[trade["id"]] = trade

    LAST_ENTRY_BY_SYMBOL = state.get(
        "last_entry_by_symbol",
        {}
    )

    logger.info(
        "Restored %d open trades.",
        len(OPEN_TRADES)
    )


# ============================================================
# STARTUP
# ============================================================

def main():
    restore_state()

    logger.info("==============================================")
    logger.info("🧠 MARKET BRAIN AI v2 STARTED")
    logger.info("20 COINS | 1D / 4H / 1H")
    logger.info("NO 5M")
    logger.info("Dynamic Entry / Structural SL / Dynamic TP")
    logger.info("Minimum RR: %.2f", MIN_RR)
    logger.info("Minimum Score: %.2f", MIN_SCORE)
    logger.info("Max Open Trades: %d", MAX_OPEN_TRADES)
    logger.info("New Trades / Scan: %d", MAX_NEW_TRADES_PER_SCAN)
    logger.info(
        "Pakistan Time: %s",
        now_pkt().strftime("%Y-%m-%d %H:%M:%S")
    )
    logger.info("==============================================")

    while True:
        try:
            # First manage existing positions.
            check_open_trades()

            # Then search for new opportunities.
            scan_all()

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break

        except Exception as e:
            logger.exception(
                "Main loop error: %s",
                e
            )

        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    main()
