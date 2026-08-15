import os
import time
import json
import math
import logging
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np


# ============================================================
# 🧠 MARKET BRAIN v11
# ADAPTIVE LEARNING TRADING ENGINE
# ============================================================
#
# 1D  -> MARKET CONTEXT
# 4H  -> MAJOR STRUCTURE
# 1H  -> MAIN SETUP
# 5M  -> ENTRY ONLY
#
# NORMAL RULES AT START
# AI LEARNS FROM CLOSED TRADES
#
# LEARNING WINDOW:
# LAST 50 CLOSED TRADES
#
# AI STUDIES:
# RSI
# EMA
# PRESSURE
# VOLUME
# STRUCTURE
# FVG
# ORDER BLOCK
# SUPPORT / RESISTANCE
# 5M ENTRY
#
# AI learns:
# WIN PATTERNS
# LOSS PATTERNS
# FEATURE PERFORMANCE
# MARKET REGIME
#
# DEFAULT:
# SL = 1%
# TP = 2%
#
# AI may adapt SL/TP only after enough evidence.
#
# REPORT:
# 08:00 AM Pakistan = previous 12 hours
# 08:00 PM Pakistan = previous 12 hours
#
# ============================================================


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

logger = logging.getLogger("MARKET_BRAIN")


# ============================================================
# PAKISTAN TIME
# ============================================================

PK_TZ = timezone(timedelta(hours=5))


def pakistan_datetime():
    return datetime.now(PK_TZ)


def pakistan_time():
    return pakistan_datetime().strftime(
        "%Y-%m-%d %I:%M:%S %p"
    )


def utc_timestamp():
    return time.time()


# ============================================================
# GMAIL
# ============================================================

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")

GMAIL_RECEIVER = os.environ.get(
    "GMAIL_RECEIVER",
    GMAIL_USER or ""
)


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

        with smtplib.SMTP(
            SMTP_SERVER,
            SMTP_PORT,
            timeout=30
        ) as server:

            server.ehlo()
            server.starttls()
            server.ehlo()

            server.login(
                GMAIL_USER,
                GMAIL_PASS
            )

            server.send_message(msg)

        logger.info(
            "Gmail sent: %s",
            subject
        )

        return True

    except Exception as e:

        logger.error(
            "Gmail error: %s",
            e
        )

        return False


# ============================================================
# CONFIG
# ============================================================

SCAN_INTERVAL_SECONDS = 120

BASE_TP_PERCENT = 2.0
BASE_SL_PERCENT = 1.0

# AI adaptive range
MIN_SL_PERCENT = 0.8
MAX_SL_PERCENT = 1.5

MIN_TP_PERCENT = 1.6
MAX_TP_PERCENT = 3.0

# Normal signal rules
MIN_SIGNAL_SCORE = 60
MIN_DIRECTION_GAP = 8

COOLDOWN_SECONDS = 15 * 60

# AI learning
LEARNING_WINDOW = 50
MIN_LEARNING_TRADES = 50

# Reports
REPORT_HOUR_MORNING = 8
REPORT_HOUR_EVENING = 20

REPORT_FILE = "report_schedule.json"

MEMORY_FILE = "market_brain_memory.json"
TRADE_LOG_FILE = "ai_trade_learning_log.json"
ACTIVE_TRADES_FILE = "active_trades.json"


# ============================================================
# 20 COINS
# ============================================================

ASSETS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "SUIUSDT",
    "OPUSDT",
    "ARBUSDT",
    "INJUSDT",
    "APTUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "POLUSDT",
    "PAXGUSDT"
]


# ============================================================
# HTTP
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
})


BINANCE_ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines"
]


# ============================================================
# TRACKER
# ============================================================

SIGNAL_TRACKER = {
    symbol: {
        "last_signal": None,
        "last_signal_time": 0,
        "active_trade": None
    }
    for symbol in ASSETS
}


# ============================================================
# JSON HELPERS
# ============================================================

def json_safe(value):

    if value is None:
        return None

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, str):
        return value

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):

        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            json_safe(v)
            for v in value
        ]

    return str(value)


def load_json_file(filename, default):

    if not os.path.exists(filename):
        return default

    try:

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        logger.warning(
            "%s load error: %s",
            filename,
            e
        )

        return default


def save_json_file(filename, data):

    temp = filename + ".tmp"

    try:

        with open(
            temp,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                json_safe(data),
                f,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

        os.replace(
            temp,
            filename
        )

        return True

    except Exception as e:

        logger.error(
            "%s save error: %s",
            filename,
            e
        )

        return False


# ============================================================
# 🧠 DEFAULT AI BRAIN
# ============================================================

def default_memory():

    return {

        "created_at": utc_timestamp(),

        "observations": 0,
        "signals": 0,

        "wins": 0,
        "losses": 0,
        "closed_trades": 0,

        "learning_cycles": 0,

        "last_learning_update": None,

        "last_learning_trade_count": 0,

        # ----------------------------------------------------
        # Adaptive feature weights
        # ----------------------------------------------------

        "weights": {

            "pressure": 1.00,
            "volume": 1.00,
            "ema": 1.00,
            "rsi": 1.00,
            "structure": 1.00,
            "fvg": 1.00,
            "order_block": 1.00,
            "support_resistance": 1.00,
            "entry_5m": 1.00
        },

        # ----------------------------------------------------
        # Feature statistics
        # ----------------------------------------------------

        "feature_stats": {},

        # ----------------------------------------------------
        # Market regime statistics
        # ----------------------------------------------------

        "regimes": {},

        # ----------------------------------------------------
        # SL / TP learning
        # ----------------------------------------------------

        "risk_model": {

            "sl_percent": BASE_SL_PERCENT,
            "tp_percent": BASE_TP_PERCENT,

            "reason": "BASE_SETTINGS",

            "sample_size": 0
        }
    }


BRAIN_MEMORY = load_json_file(
    MEMORY_FILE,
    default_memory()
)

if not isinstance(BRAIN_MEMORY, dict):
    BRAIN_MEMORY = default_memory()


def save_memory():

    BRAIN_MEMORY[
        "last_learning_update"
    ] = utc_timestamp()

    save_json_file(
        MEMORY_FILE,
        BRAIN_MEMORY
    )


# ============================================================
# TRADE LOG
# ============================================================

def load_trade_logs():

    data = load_json_file(
        TRADE_LOG_FILE,
        []
    )

    return data if isinstance(data, list) else []


def save_trade_logs(logs):

    save_json_file(
        TRADE_LOG_FILE,
        logs
    )


def add_trade_log(trade):

    logs = load_trade_logs()

    trade_id = trade.get(
        "trade_id"
    )

    for existing in logs:

        if existing.get(
            "trade_id"
        ) == trade_id:

            return False

    logs.append(
        json_safe(trade)
    )

    save_trade_logs(logs)

    return True


def update_trade_log(trade):

    logs = load_trade_logs()

    trade_id = trade.get(
        "trade_id"
    )

    found = False

    for i, existing in enumerate(logs):

        if existing.get(
            "trade_id"
        ) == trade_id:

            logs[i] = json_safe(
                trade
            )

            found = True
            break

    if not found:

        logs.append(
            json_safe(trade)
        )

    save_trade_logs(logs)


# ============================================================
# ACTIVE TRADES
# ============================================================

def save_active_trades():

    data = {}

    for symbol in ASSETS:

        trade = SIGNAL_TRACKER[
            symbol
        ].get(
            "active_trade"
        )

        if trade:

            data[symbol] = trade

    save_json_file(
        ACTIVE_TRADES_FILE,
        data
    )


def restore_active_trades():

    active = load_json_file(
        ACTIVE_TRADES_FILE,
        {}
    )

    if not isinstance(active, dict):
        return

    for symbol, trade in active.items():

        if symbol not in SIGNAL_TRACKER:
            continue

        if not isinstance(trade, dict):
            continue

        if trade.get("status") != "OPEN":
            continue

        SIGNAL_TRACKER[
            symbol
        ]["active_trade"] = trade

        SIGNAL_TRACKER[
            symbol
        ]["last_signal"] = trade.get(
            "direction"
        )

        SIGNAL_TRACKER[
            symbol
        ]["last_signal_time"] = trade.get(
            "created_at",
            0
        )

        logger.info(
            "Restored %s %s",
            symbol,
            trade.get("direction")
        )


# ============================================================
# MARKET DATA
# ============================================================

def fetch_candles(
    symbol,
    interval,
    limit=200
):

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    for endpoint in BINANCE_ENDPOINTS:

        try:

            response = SESSION.get(
                endpoint,
                params=params,
                timeout=12
            )

            if response.status_code != 200:
                continue

            raw = response.json()

            if not isinstance(raw, list):
                continue

            if len(raw) < 20:
                continue

            columns = [
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore"
            ]

            df = pd.DataFrame(
                raw,
                columns=columns
            )

            numeric = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_base",
                "taker_buy_quote"
            ]

            for col in numeric:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            df["time"] = pd.to_numeric(
                df["time"],
                errors="coerce"
            )

            df = df.dropna(
                subset=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume"
                ]
            )

            return df.reset_index(
                drop=True
            )

        except Exception:
            continue

    return None


# ============================================================
# PRESSURE
# ============================================================

def pressure_analysis(df, candles=12):

    result = {
        "buyer": 50.0,
        "seller": 50.0,
        "dominant": "NEUTRAL"
    }

    if df is None or len(df) < candles + 3:
        return result

    closed = df.iloc[:-1]
    data = closed.tail(candles)

    buyer = 0.0
    seller = 0.0

    for _, candle in data.iterrows():

        candle_range = (
            candle["high"] -
            candle["low"]
        )

        if candle_range <= 0:
            continue

        body = abs(
            candle["close"] -
            candle["open"]
        )

        body_ratio = (
            body /
            candle_range
        )

        volume = max(
            float(candle["volume"]),
            0
        )

        pressure = (
            body_ratio *
            volume
        )

        if candle["close"] > candle["open"]:
            buyer += pressure

        elif candle["close"] < candle["open"]:
            seller += pressure

    total = buyer + seller

    if total <= 0:
        return result

    buyer_pct = (
        buyer / total
    ) * 100

    seller_pct = (
        seller / total
    ) * 100

    if buyer_pct > seller_pct:
        dominant = "BUYER"

    elif seller_pct > buyer_pct:
        dominant = "SELLER"

    else:
        dominant = "NEUTRAL"

    return {
        "buyer": round(
            buyer_pct,
            2
        ),
        "seller": round(
            seller_pct,
            2
        ),
        "dominant": dominant
    }


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(df, lookback=20):

    result = {
        "ratio": 1.0,
        "strong": False,
        "direction": "NEUTRAL"
    }

    if df is None or len(df) < lookback + 3:
        return result

    closed = df.iloc[:-1]

    current = closed.iloc[-1]

    previous = closed[
        "volume"
    ].iloc[
        -(lookback + 1):-1
    ]

    average = previous.mean()

    if average <= 0:
        return result

    ratio = (
        float(current["volume"]) /
        float(average)
    )

    if current["close"] > current["open"]:
        direction = "BUYER"

    elif current["close"] < current["open"]:
        direction = "SELLER"

    else:
        direction = "NEUTRAL"

    return {
        "ratio": round(
            ratio,
            2
        ),
        "strong": bool(
            ratio >= 1.10
        ),
        "direction": direction
    }


# ============================================================
# RSI
# ============================================================

def calculate_rsi(df, period=14):

    if df is None or len(df) < period + 3:
        return 50.0

    closed = df.iloc[:-1]

    delta = closed[
        "close"
    ].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    loss_value = avg_loss.iloc[-1]

    if loss_value <= 0:
        return 100.0

    rs = (
        avg_gain.iloc[-1] /
        loss_value
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return round(
        float(
            max(
                0,
                min(
                    100,
                    rsi
                )
            )
        ),
        2
    )


# ============================================================
# EMA
# ============================================================

def ema_direction(df):

    if df is None or len(df) < 53:
        return "NEUTRAL"

    closed = df.iloc[:-1]

    ema20 = closed[
        "close"
    ].ewm(
        span=20,
        adjust=False
    ).mean().iloc[-1]

    ema50 = closed[
        "close"
    ].ewm(
        span=50,
        adjust=False
    ).mean().iloc[-1]

    if ema20 > ema50:
        return "BULLISH"

    if ema20 < ema50:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# STRUCTURE
# ============================================================

def one_hour_structure(
    df,
    lookback=50
):

    result = {
        "bullish": False,
        "bearish": False,
        "high": None,
        "low": None
    }

    if df is None or len(df) < lookback + 3:
        return result

    closed = df.iloc[:-1]

    current = closed.iloc[-1]

    previous = closed.iloc[
        -(lookback + 1):-1
    ]

    resistance = float(
        previous["high"].max()
    )

    support = float(
        previous["low"].min()
    )

    result["high"] = resistance
    result["low"] = support

    result["bullish"] = bool(
        current["high"] > resistance
        and
        current["close"] > resistance
    )

    result["bearish"] = bool(
        current["low"] < support
        and
        current["close"] < support
    )

    return result


# ============================================================
# FVG
# ============================================================

def detect_fvg(df):

    result = {
        "bullish": False,
        "bearish": False,
        "size_percent": 0.0,
        "age": 0
    }

    if df is None or len(df) < 10:
        return result

    closed = df.iloc[:-1]

    # Search recent candles for a valid FVG.
    for i in range(
        len(closed) - 1,
        max(2, len(closed) - 8),
        -1
    ):

        c1 = closed.iloc[i - 2]
        c3 = closed.iloc[i]

        if c3["low"] > c1["high"]:

            gap = (
                c3["low"] -
                c1["high"]
            )

            midpoint = (
                c3["low"] +
                c1["high"]
            ) / 2

            result.update({
                "bullish": True,
                "size_percent": round(
                    gap / midpoint * 100,
                    3
                ),
                "age": len(closed) - 1 - i
            })

            return result

        if c3["high"] < c1["low"]:

            gap = (
                c1["low"] -
                c3["high"]
            )

            midpoint = (
                c1["low"] +
                c3["high"]
            ) / 2

            result.update({
                "bearish": True,
                "size_percent": round(
                    gap / midpoint * 100,
                    3
                ),
                "age": len(closed) - 1 - i
            })

            return result

    return result


# ============================================================
# ORDER BLOCK
# ============================================================

def detect_order_block(df):

    result = {
        "bullish": False,
        "bearish": False,
        "strength": 0.0
    }

    if df is None or len(df) < 10:
        return result

    closed = df.iloc[:-1]

    previous = closed.iloc[-2]
    last = closed.iloc[-1]

    prev_range = (
        previous["high"] -
        previous["low"]
    )

    if prev_range <= 0:
        return result

    body = abs(
        previous["close"] -
        previous["open"]
    )

    strength = (
        body /
        prev_range
    )

    if (
        previous["close"] <
        previous["open"]
        and
        last["close"] >
        last["open"]
        and
        last["close"] >
        previous["high"]
    ):

        result.update({
            "bullish": True,
            "strength": round(
                strength,
                3
            )
        })

    elif (
        previous["close"] >
        previous["open"]
        and
        last["close"] <
        last["open"]
        and
        last["close"] <
        previous["low"]
    ):

        result.update({
            "bearish": True,
            "strength": round(
                strength,
                3
            )
        })

    return result


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def support_resistance(
    df,
    lookback=50
):

    result = {
        "support": None,
        "resistance": None,
        "near_support": False,
        "near_resistance": False
    }

    if df is None or len(df) < lookback + 3:
        return result

    closed = df.iloc[:-1]

    data = closed.tail(
        lookback
    )

    support = float(
        data["low"].min()
    )

    resistance = float(
        data["high"].max()
    )

    price = float(
        closed["close"].iloc[-1]
    )

    if price <= 0:
        return result

    result.update({

        "support": support,

        "resistance": resistance,

        "near_support": (
            abs(price - support) /
            price <= 0.008
        ),

        "near_resistance": (
            abs(resistance - price) /
            price <= 0.008
        )
    })

    return result


# ============================================================
# MARKET REGIME
# ============================================================

def market_regime(
    h1,
    h4
):

    if h1 is None:
        return "UNKNOWN"

    closed = h1.iloc[:-1]

    if len(closed) < 30:
        return "UNKNOWN"

    returns = closed[
        "close"
    ].pct_change().tail(20)

    volatility = (
        returns.std() * 100
    )

    ema = ema_direction(
        h1
    )

    if volatility >= 1.5:
        return "HIGH_VOLATILITY"

    if ema in [
        "BULLISH",
        "BEARISH"
    ]:

        return "TRENDING"

    return "RANGING"


# ============================================================
# 5M ENTRY ONLY
# ============================================================

def five_min_entry(
    df,
    direction
):

    result = {
        "confirmed": False,
        "price": None,
        "body_ratio": 0.0,
        "reason": "NO_CONFIRMATION"
    }

    if df is None or len(df) < 5:
        return result

    closed = df.iloc[:-1]

    last = closed.iloc[-1]

    price = float(
        last["close"]
    )

    candle_range = (
        last["high"] -
        last["low"]
    )

    if candle_range <= 0:
        return result

    body = abs(
        last["close"] -
        last["open"]
    )

    body_ratio = (
        body /
        candle_range
    )

    # NORMAL RULE:
    # Do NOT make this too strict.
    if direction == "BUY":

        if (
            last["close"] >
            last["open"]
            and
            body_ratio >= 0.25
        ):

            return {
                "confirmed": True,
                "price": price,
                "body_ratio": round(
                    body_ratio,
                    3
                ),
                "reason": "5M_BULLISH_CANDLE"
            }

    elif direction == "SELL":

        if (
            last["close"] <
            last["open"]
            and
            body_ratio >= 0.25
        ):

            return {
                "confirmed": True,
                "price": price,
                "body_ratio": round(
                    body_ratio,
                    3
                ),
                "reason": "5M_BEARISH_CANDLE"
            }

    return result


# ============================================================
# AI WEIGHT
# ============================================================

def get_weight(name):

    weights = BRAIN_MEMORY.get(
        "weights",
        {}
    )

    value = weights.get(
        name,
        1.0
    )

    try:
        value = float(value)
    except Exception:
        value = 1.0

    # IMPORTANT:
    # AI cannot destroy the strategy.
    return max(
        0.50,
        min(
            1.50,
            value
        )
    )


# ============================================================
# ADAPTIVE SCORE
# ============================================================

def weighted_score(
    points,
    feature
):

    return points * get_weight(
        feature
    )


# ============================================================
# TRADE LEVELS
# ============================================================

def get_risk_model():

    model = BRAIN_MEMORY.get(
        "risk_model",
        {}
    )

    sl = float(
        model.get(
            "sl_percent",
            BASE_SL_PERCENT
        )
    )

    tp = float(
        model.get(
            "tp_percent",
            BASE_TP_PERCENT
        )
    )

    sl = max(
        MIN_SL_PERCENT,
        min(
            MAX_SL_PERCENT,
            sl
        )
    )

    tp = max(
        MIN_TP_PERCENT,
        min(
            MAX_TP_PERCENT,
            tp
        )
    )

    return sl, tp


def calculate_trade_levels(
    direction,
    entry
):

    sl_percent, tp_percent = (
        get_risk_model()
    )

    if direction == "BUY":

        sl = entry * (
            1 - sl_percent / 100
        )

        tp = entry * (
            1 + tp_percent / 100
        )

    else:

        sl = entry * (
            1 + sl_percent / 100
        )

        tp = entry * (
            1 - tp_percent / 100
        )

    return (
        float(entry),
        float(sl),
        float(tp),
        sl_percent,
        tp_percent
    )


# ============================================================
# MAIN MARKET ANALYSIS
# ============================================================

def analyze_market(
    symbol,
    h1,
    m5,
    h4=None
):

    if h1 is None or m5 is None:
        return None

    pressure = pressure_analysis(
        h1
    )

    volume = volume_analysis(
        h1
    )

    structure = one_hour_structure(
        h1
    )

    fvg = detect_fvg(
        h1
    )

    ob = detect_order_block(
        h1
    )

    ema = ema_direction(
        h1
    )

    rsi = calculate_rsi(
        h1
    )

    sr = support_resistance(
        h1
    )

    regime = market_regime(
        h1,
        h4
    )

    buy_score = 0
    sell_score = 0

    buy_reasons = []
    sell_reasons = []

    # ========================================================
    # PRESSURE
    # ========================================================

    if pressure["buyer"] >= 55:

        buy_score += weighted_score(
            18,
            "pressure"
        )

        buy_reasons.append(
            "Buyer pressure"
        )

    elif pressure["seller"] >= 55:

        sell_score += weighted_score(
            18,
            "pressure"
        )

        sell_reasons.append(
            "Seller pressure"
        )

    # ========================================================
    # VOLUME
    # ========================================================

    if volume["strong"]:

        if volume["direction"] == "BUYER":

            buy_score += weighted_score(
                12,
                "volume"
            )

            buy_reasons.append(
                f"Volume {volume['ratio']:.2f}x"
            )

        elif volume["direction"] == "SELLER":

            sell_score += weighted_score(
                12,
                "volume"
            )

            sell_reasons.append(
                f"Volume {volume['ratio']:.2f}x"
            )

    # ========================================================
    # EMA
    # ========================================================

    if ema == "BULLISH":

        buy_score += weighted_score(
            10,
            "ema"
        )

        buy_reasons.append(
            "EMA bullish"
        )

    elif ema == "BEARISH":

        sell_score += weighted_score(
            10,
            "ema"
        )

        sell_reasons.append(
            "EMA bearish"
        )

    # ========================================================
    # STRUCTURE
    # ========================================================

    if structure["bullish"]:

        buy_score += weighted_score(
            18,
            "structure"
        )

        buy_reasons.append(
            "Bullish structure break"
        )

    if structure["bearish"]:

        sell_score += weighted_score(
            18,
            "structure"
        )

        sell_reasons.append(
            "Bearish structure break"
    )
        # ========================================================
    # FVG
    # ========================================================

    if fvg["bullish"]:

        buy_score += weighted_score(
            8,
            "fvg"
        )

        buy_reasons.append(
            "Bullish FVG"
        )

    if fvg["bearish"]:

        sell_score += weighted_score(
            8,
            "fvg"
        )

        sell_reasons.append(
            "Bearish FVG"
        )

    # ========================================================
    # ORDER BLOCK
    # ========================================================

    if ob["bullish"]:

        buy_score += weighted_score(
            8,
            "order_block"
        )

        buy_reasons.append(
            "Bullish Order Block"
        )

    if ob["bearish"]:

        sell_score += weighted_score(
            8,
            "order_block"
        )

        sell_reasons.append(
            "Bearish Order Block"
        )

    # ========================================================
    # RSI
    # ========================================================

    if 45 <= rsi <= 68:

        if ema == "BULLISH":

            buy_score += weighted_score(
                5,
                "rsi"
            )

            buy_reasons.append(
                "RSI supports BUY"
            )

        elif ema == "BEARISH":

            sell_score += weighted_score(
                5,
                "rsi"
            )

            sell_reasons.append(
                "RSI supports SELL"
            )

    # ========================================================
    # S/R
    # ========================================================

    if sr["near_support"]:

        buy_score += weighted_score(
            7,
            "support_resistance"
        )

        buy_reasons.append(
            "Near support"
        )

    if sr["near_resistance"]:

        sell_score += weighted_score(
            7,
            "support_resistance"
        )

        sell_reasons.append(
            "Near resistance"
        )

    # ========================================================
    # DIRECTION
    # ========================================================

    direction = "HOLD"

    if (
        buy_score >= MIN_SIGNAL_SCORE
        and
        buy_score >=
        sell_score + MIN_DIRECTION_GAP
    ):

        direction = "BUY"

    elif (
        sell_score >= MIN_SIGNAL_SCORE
        and
        sell_score >=
        buy_score + MIN_DIRECTION_GAP
    ):

        direction = "SELL"

    # ========================================================
    # 5M ENTRY
    # ========================================================

    entry_check = {
        "confirmed": False,
        "price": None,
        "body_ratio": 0,
        "reason": "NO_CONFIRMATION"
    }

    if direction in [
        "BUY",
        "SELL"
    ]:

        entry_check = five_min_entry(
            m5,
            direction
        )

    final_score = max(
        buy_score,
        sell_score
    )

    if entry_check["confirmed"]:

        final_score += weighted_score(
            5,
            "entry_5m"
        )

    # ========================================================
    # TRADE
    # ========================================================

    trade = None

    if (
        direction in [
            "BUY",
            "SELL"
        ]
        and
        entry_check["confirmed"]
    ):

        entry = entry_check[
            "price"
        ]

        (
            entry,
            sl,
            tp,
            sl_percent,
            tp_percent
        ) = calculate_trade_levels(
            direction,
            entry
        )

        trade = {

            "trade_id":
                f"{symbol}_{direction}_{int(time.time())}",

            "symbol":
                symbol,

            "direction":
                direction,

            "score":
                int(round(final_score)),

            "buy_score":
                int(round(buy_score)),

            "sell_score":
                int(round(sell_score)),

            "entry":
                entry,

            "stop_loss":
                sl,

            "take_profit":
                tp,

            "sl_percent":
                sl_percent,

            "tp_percent":
                tp_percent,

            "risk_reward":
                round(
                    tp_percent /
                    sl_percent,
                    2
                ),

            "created_at":
                utc_timestamp(),

            "created_time_pk":
                pakistan_time(),

            "status":
                "OPEN",

            "market_regime":
                regime,

            "entry_reason":
                entry_check["reason"],

            "entry_body_ratio":
                entry_check["body_ratio"],

            # ---------------------------------------------
            # MARKET FEATURES
            # ---------------------------------------------

            "buyer_pressure":
                pressure["buyer"],

            "seller_pressure":
                pressure["seller"],

            "dominant_pressure":
                pressure["dominant"],

            "volume_ratio":
                volume["ratio"],

            "volume_strong":
                volume["strong"],

            "volume_direction":
                volume["direction"],

            "h1_ema":
                ema,

            "h1_rsi":
                rsi,

            "h1_bullish_structure":
                structure["bullish"],

            "h1_bearish_structure":
                structure["bearish"],

            "fvg_bullish":
                fvg["bullish"],

            "fvg_bearish":
                fvg["bearish"],

            "fvg_size_percent":
                fvg["size_percent"],

            "fvg_age":
                fvg["age"],

            "ob_bullish":
                ob["bullish"],

            "ob_bearish":
                ob["bearish"],

            "ob_strength":
                ob["strength"],

            "near_support":
                sr["near_support"],

            "near_resistance":
                sr["near_resistance"],

            "support":
                sr["support"],

            "resistance":
                sr["resistance"],

            # ---------------------------------------------
            # SNAPSHOT OF AI WEIGHTS
            # ---------------------------------------------

            "weights_at_entry":
                dict(
                    BRAIN_MEMORY.get(
                        "weights",
                        {}
                    )
                ),

            # ---------------------------------------------
            # REASONS
            # ---------------------------------------------

            "reasons":
                (
                    buy_reasons
                    if direction == "BUY"
                    else sell_reasons
                )
        }

    return {

        "symbol":
            symbol,

        "direction":
            direction,

        "score":
            int(round(final_score)),

        "buy_score":
            int(round(buy_score)),

        "sell_score":
            int(round(sell_score)),

        "trade":
            trade,

        "regime":
            regime
    }


# ============================================================
# TRADE PROTECTION
# ============================================================

def should_send_trade(
    symbol,
    direction
):

    tracker = SIGNAL_TRACKER[
        symbol
    ]

    if tracker.get(
        "active_trade"
    ) is not None:

        return False

    now = time.time()

    last_direction = tracker.get(
        "last_signal"
    )

    last_time = tracker.get(
        "last_signal_time",
        0
    )

    if (
        last_direction == direction
        and
        now - last_time <
        COOLDOWN_SECONDS
    ):

        return False

    return True


# ============================================================
# REGISTER
# ============================================================

def register_trade(
    symbol,
    trade
):

    SIGNAL_TRACKER[
        symbol
    ]["last_signal"] = trade[
        "direction"
    ]

    SIGNAL_TRACKER[
        symbol
    ]["last_signal_time"] = time.time()

    SIGNAL_TRACKER[
        symbol
    ]["active_trade"] = trade

    BRAIN_MEMORY[
        "signals"
    ] += 1

    add_trade_log(
        trade
    )

    save_active_trades()
    save_memory()


# ============================================================
# TRADE EMAIL
# ============================================================

def create_trade_email(trade):

    direction = trade[
        "direction"
    ]

    if direction == "BUY":
        icon = "🟢"
        pressure = trade[
            "buyer_pressure"
        ]
    else:
        icon = "🔴"
        pressure = trade[
            "seller_pressure"
        ]

    subject = (
        f"{icon} MARKET BRAIN "
        f"{direction} — "
        f"{trade['symbol']}"
    )

    body = f"""
==================================================
🧠 MARKET BRAIN — {direction}
==================================================

Coin:
{trade["symbol"]}

Direction:
{direction}

Score:
{trade["score"]}/100

Market Regime:
{trade["market_regime"]}

==================================================
TRADE
==================================================

Entry:
{trade["entry"]:.10f}

Stop Loss:
{trade["stop_loss"]:.10f}
({trade["sl_percent"]:.2f}%)

Target:
{trade["take_profit"]:.10f}
({trade["tp_percent"]:.2f}%)

Risk / Reward:
1 : {trade["risk_reward"]}

==================================================
PRESSURE
==================================================

Buyer:
{trade["buyer_pressure"]:.1f}%

Seller:
{trade["seller_pressure"]:.1f}%

Dominant:
{trade["dominant_pressure"]}

==================================================
VOLUME
==================================================

Ratio:
{trade["volume_ratio"]:.2f}x

Direction:
{trade["volume_direction"]}

Status:
{"STRONG" if trade["volume_strong"] else "NORMAL"}

==================================================
1H ANALYSIS
==================================================

EMA:
{trade["h1_ema"]}

RSI:
{trade["h1_rsi"]:.2f}

Bullish Structure:
{trade["h1_bullish_structure"]}

Bearish Structure:
{trade["h1_bearish_structure"]}

Bullish FVG:
{trade["fvg_bullish"]}

Bearish FVG:
{trade["fvg_bearish"]}

FVG Size:
{trade["fvg_size_percent"]:.3f}%

FVG Age:
{trade["fvg_age"]}

Bullish OB:
{trade["ob_bullish"]}

Bearish OB:
{trade["ob_bearish"]}

OB Strength:
{trade["ob_strength"]}

Near Support:
{trade["near_support"]}

Near Resistance:
{trade["near_resistance"]}

==================================================
5M ENTRY
==================================================

Confirmation:
{trade["entry_reason"]}

Body Ratio:
{trade["entry_body_ratio"]}

5M is ENTRY ONLY.

==================================================
AI WEIGHTS AT ENTRY
==================================================

{json.dumps(
    trade["weights_at_entry"],
    indent=2
)}

==================================================
TIME
==================================================

{trade["created_time_pk"]}

Pakistan Time UTC+5

Trade ID:
{trade["trade_id"]}

==================================================
"""

    for reason in trade["reasons"]:

        body += (
            f"\n• {reason}"
        )

    return subject, body


# ============================================================
# CLOSE EMAIL
# ============================================================

def create_close_email(
    trade,
    result
):

    icon = (
        "🟢"
        if result == "WIN"
        else "🔴"
    )

    subject = (
        f"{icon} MARKET BRAIN "
        f"{result} — "
        f"{trade['symbol']}"
    )

    body = f"""
==================================================
🧠 MARKET BRAIN — TRADE CLOSED
==================================================

Coin:
{trade["symbol"]}

Direction:
{trade["direction"]}

Result:
{icon} {result}

Entry:
{trade["entry"]}

Closed Price:
{trade["closed_price"]}

SL:
{trade["stop_loss"]}

TP:
{trade["take_profit"]}

SL %:
{trade["sl_percent"]:.2f}%

TP %:
{trade["tp_percent"]:.2f}%

Created:
{trade["created_time_pk"]}

Closed:
{trade["closed_time_pk"]}

Market Regime:
{trade["market_regime"]}

Buyer Pressure:
{trade["buyer_pressure"]:.1f}%

Seller Pressure:
{trade["seller_pressure"]:.1f}%

Volume:
{trade["volume_ratio"]:.2f}x

RSI:
{trade["h1_rsi"]}

EMA:
{trade["h1_ema"]}

FVG:
{trade["fvg_bullish"] or trade["fvg_bearish"]}

Order Block:
{trade["ob_bullish"] or trade["ob_bearish"]}

==================================================

Trade saved to AI learning memory.
"""

    return subject, body


# ============================================================
# CHECK ACTIVE TRADE
# ============================================================

def check_active_trade(
    symbol,
    trade
):

    if trade is None:
        return None

    df = fetch_candles(
        symbol,
        "5m",
        10
    )

    if df is None:
        return None

    current = float(
        df["close"].iloc[-1]
    )

    direction = trade[
        "direction"
    ]

    result = None

    if direction == "BUY":

        if current >= trade[
            "take_profit"
        ]:

            result = "WIN"

        elif current <= trade[
            "stop_loss"
        ]:

            result = "LOSS"

    else:

        if current <= trade[
            "take_profit"
        ]:

            result = "WIN"

        elif current >= trade[
            "stop_loss"
        ]:

            result = "LOSS"

    if result:

        trade["status"] = result

        trade["closed_price"] = current

        trade[
            "closed_time_pk"
        ] = pakistan_time()

        trade[
            "closed_at"
        ] = utc_timestamp()

        trade[
            "realized_rr"
        ] = (
            trade["tp_percent"] /
            trade["sl_percent"]
            if result == "WIN"
            else -1.0
        )

        return result

    return None


# ============================================================
# 🧠 AI LEARNING
# ============================================================

FEATURES = [
    "pressure",
    "volume",
    "ema",
    "rsi",
    "structure",
    "fvg",
    "order_block",
    "support_resistance",
    "entry_5m"
]


def feature_present(
    trade,
    feature
):

    if feature == "pressure":

        return (
            trade.get(
                "buyer_pressure",
                50
            ) >= 55
            or
            trade.get(
                "seller_pressure",
                50
            ) >= 55
        )

    if feature == "volume":

        return bool(
            trade.get(
                "volume_strong",
                False
            )
        )

    if feature == "ema":

        return (
            trade.get(
                "h1_ema"
            )
            in
            ["BULLISH", "BEARISH"]
        )

    if feature == "rsi":

        rsi = float(
            trade.get(
                "h1_rsi",
                50
            )
        )

        return 45 <= rsi <= 68

    if feature == "structure":

        return bool(
            trade.get(
                "h1_bullish_structure",
                False
            )
            or
            trade.get(
                "h1_bearish_structure",
                False
            )
        )

    if feature == "fvg":

        return bool(
            trade.get(
                "fvg_bullish",
                False
            )
            or
            trade.get(
                "fvg_bearish",
                False
            )
        )

    if feature == "order_block":

        return bool(
            trade.get(
                "ob_bullish",
                False
            )
            or
            trade.get(
                "ob_bearish",
                False
            )
        )

    if feature == "support_resistance":

        return bool(
            trade.get(
                "near_support",
                False
            )
            or
            trade.get(
                "near_resistance",
                False
            )
        )

    if feature == "entry_5m":

        return bool(
            trade.get(
                "entry_reason"
            )
        )

    return False


def learn_from_last_50():

    logs = load_trade_logs()

    closed = [
        t
        for t in logs
        if t.get("status")
        in ["WIN", "LOSS"]
    ]

    closed = closed[
        -LEARNING_WINDOW:
    ]

    if len(closed) < MIN_LEARNING_TRADES:

        logger.info(
            "AI learning waiting: %s/%s closed trades",
            len(closed),
            MIN_LEARNING_TRADES
        )

        return

    logger.info(
        "🧠 AI LEARNING STARTED — "
        "last %s trades",
        len(closed)
    )

    stats = {}

# ========================================================
    # FEATURE ANALYSIS
    # ========================================================

    for feature in FEATURES:

        applicable = [
            t
            for t in closed
            if feature_present(
                t,
                feature
            )
        ]

        if not applicable:
            continue

        wins = sum(
            1
            for t in applicable
            if t.get("status") == "WIN"
        )

        losses = sum(
            1
            for t in applicable
            if t.get("status") == "LOSS"
        )

        total = wins + losses

        win_rate = (
            wins / total * 100
            if total
            else 0
        )

        stats[feature] = {

            "trades": total,

            "wins": wins,

            "losses": losses,

            "win_rate": round(
                win_rate,
                2
            )
        }

    BRAIN_MEMORY[
        "feature_stats"
    ] = stats

    # ========================================================
    # ADJUST WEIGHTS
    # ========================================================

    weights = BRAIN_MEMORY[
        "weights"
    ]

    for feature, data in stats.items():

        if data["trades"] < 8:
            continue

        win_rate = data[
            "win_rate"
        ]

        old = float(
            weights.get(
                feature,
                1.0
            )
        )

        # Good feature
        if win_rate >= 65:

            new = old + 0.05

        # Weak feature
        elif win_rate <= 42:

            new = old - 0.05

        else:

            new = old

        # Keep adaptation controlled
        new = max(
            0.50,
            min(
                1.50,
                new
            )
        )

        weights[
            feature
        ] = round(
            new,
            3
        )

    # ========================================================
    # REGIME LEARNING
    # ========================================================

    regimes = {}

    for trade in closed:

        regime = trade.get(
            "market_regime",
            "UNKNOWN"
        )

        if regime not in regimes:

            regimes[regime] = {
                "wins": 0,
                "losses": 0
            }

        if trade.get(
            "status"
        ) == "WIN":

            regimes[
                regime
            ]["wins"] += 1

        else:

            regimes[
                regime
            ]["losses"] += 1

    BRAIN_MEMORY[
        "regimes"
    ] = regimes

    # ========================================================
    # RISK MODEL LEARNING
    # ========================================================
    #
    # IMPORTANT:
    # Don't change SL/TP aggressively.
    #
    # The bot starts with:
    # SL 1%
    # TP 2%
    #
    # Only after 50 trades we inspect whether
    # current risk model is repeatedly failing.
    # ========================================================

    wins = sum(
        1
        for t in closed
        if t.get("status") == "WIN"
    )

    losses = sum(
        1
        for t in closed
        if t.get("status") == "LOSS"
    )

    win_rate = (
        wins /
        len(closed) *
        100
    )

    risk_model = BRAIN_MEMORY[
        "risk_model"
    ]

    current_sl = float(
        risk_model.get(
            "sl_percent",
            BASE_SL_PERCENT
        )
    )

    current_tp = float(
        risk_model.get(
            "tp_percent",
            BASE_TP_PERCENT
        )
    )

    # Conservative adaptation.
    #
    # If loss rate is unusually high:
    # slightly widen SL and TP.
    #
    # If performance is healthy:
    # maintain the base model.

    if win_rate < 35:

        current_sl += 0.05
        current_tp += 0.10

        reason = (
            "50-trade sample shows "
            "high loss rate"
        )

    elif win_rate >= 60:

        # Don't aggressively change a good system.
        reason = (
            "50-trade sample healthy; "
            "risk model maintained"
        )

    else:

        reason = (
            "50-trade sample neutral; "
            "risk model maintained"
        )

    current_sl = max(
        MIN_SL_PERCENT,
        min(
            MAX_SL_PERCENT,
            current_sl
        )
    )

    current_tp = max(
        MIN_TP_PERCENT,
        min(
            MAX_TP_PERCENT,
            current_tp
        )
    )

    risk_model.update({

        "sl_percent":
            round(
                current_sl,
                3
            ),

        "tp_percent":
            round(
                current_tp,
                3
            ),

        "reason":
            reason,

        "sample_size":
            len(closed),

        "win_rate":
            round(
                win_rate,
                2
            )
    })

    BRAIN_MEMORY[
        "learning_cycles"
    ] += 1

    BRAIN_MEMORY[
        "last_learning_trade_count"
    ] = len(closed)

    save_memory()

    logger.info(
        "🧠 AI LEARNING COMPLETE"
    )

    logger.info(
        "Win rate: %.1f%%",
        win_rate
    )

    logger.info(
        "Weights: %s",
        weights
    )

    logger.info(
        "SL %.2f%% | TP %.2f%%",
        current_sl,
        current_tp
    )


# ============================================================
# LEARN AFTER TRADE
# ============================================================

def learn_from_trade(
    trade,
    result
):

    BRAIN_MEMORY[
        "closed_trades"
    ] += 1

    if result == "WIN":

        BRAIN_MEMORY[
            "wins"
        ] += 1

    elif result == "LOSS":

        BRAIN_MEMORY[
            "losses"
        ] += 1

    update_trade_log(
        trade
    )

    save_memory()

    # Learn only when 50 closed trades exist.
    learn_from_last_50()


# ============================================================
# CHECK ALL TRADES
# ============================================================

def check_all_active_trades():

    changed = False

    for symbol in ASSETS:

        trade = SIGNAL_TRACKER[
            symbol
        ].get(
            "active_trade"
        )

        if trade is None:
            continue

        try:

            result = check_active_trade(
                symbol,
                trade
            )

            if result:

                logger.info(
                    "%s %s -> %s",
                    symbol,
                    trade["direction"],
                    result
                )

                learn_from_trade(
                    trade,
                    result
                )

                subject, body = (
                    create_close_email(
                        trade,
                        result
                    )
                )

                send_email(
                    subject,
                    body
                )

                SIGNAL_TRACKER[
                    symbol
                ]["active_trade"] = None

                changed = True

        except Exception as e:

            logger.error(
                "%s trade check: %s",
                symbol,
                e
            )

    if changed:

        save_active_trades()


# ============================================================
# SCAN SYMBOL
# ============================================================

def scan_symbol(symbol):

    logger.info(
        "Scanning %s",
        symbol
    )

    h1 = fetch_candles(
        symbol,
        "1h",
        100
    )

    h4 = fetch_candles(
        symbol,
        "4h",
        100
    )

    m5 = fetch_candles(
        symbol,
        "5m",
        30
    )

    if h1 is None or m5 is None:
        return

    analysis = analyze_market(
        symbol,
        h1,
        m5,
        h4
    )

    if analysis is None:
        return

    BRAIN_MEMORY[
        "observations"
    ] += 1

    logger.info(
        "%s | BUY=%s | SELL=%s | %s | SCORE=%s",
        symbol,
        analysis["buy_score"],
        analysis["sell_score"],
        analysis["direction"],
        analysis["score"]
    )

    trade = analysis[
        "trade"
    ]

    if trade is None:
        return

    if not should_send_trade(
        symbol,
        trade["direction"]
    ):

        return

    subject, body = (
        create_trade_email(
            trade
        )
    )

    sent = send_email(
        subject,
        body
    )

    if sent:

        register_trade(
            symbol,
            trade
        )

        logger.info(
            "🟢 TRADE CREATED %s %s",
            symbol,
            trade["direction"]
        )

    else:

        logger.warning(
            "%s Gmail failed; "
            "trade not registered.",
            symbol
        )


# ============================================================
# 12-HOUR REPORT DATA
# ============================================================

def create_12_hour_report():

    logs = load_trade_logs()

    now = time.time()

    start = now - (
        12 * 60 * 60
    )

    recent = []

    for trade in logs:

        created = trade.get(
            "created_at",
            0
        )

        try:
            created = float(created)
        except Exception:
            continue

        if created >= start:

            recent.append(
                trade
            )

    wins = sum(
        1
        for t in recent
        if t.get("status") == "WIN"
    )

    losses = sum(
        1
        for t in recent
        if t.get("status") == "LOSS"
    )

    open_trades = sum(
        1
        for t in recent
        if t.get("status") == "OPEN"
    )

    total = len(recent)

    closed = wins + losses

    win_rate = (
        wins /
        closed *
        100
        if closed
        else 0
    )

    net_rr = (
        wins * 2
        -
        losses
    )

    return {
        "recent": recent,
        "total": total,
        "wins": wins,
        "losses": losses,
        "open": open_trades,
        "closed": closed,
        "win_rate": win_rate,
        "net_rr": net_rr
    }


# ============================================================
# REPORT EMAIL
# ============================================================

def send_12_hour_report():

    report = create_12_hour_report()

    weights = BRAIN_MEMORY.get(
        "weights",
        {}
    )

    risk = BRAIN_MEMORY.get(
        "risk_model",
        {}
    )

    subject = (
        "🧠 MARKET BRAIN — "
        "12 HOUR REPORT"
    )

    body = f"""
==================================================
🧠 MARKET BRAIN
12 HOUR PERFORMANCE REPORT
==================================================

Pakistan Time:
{pakistan_time()}

==================================================
RESULT
==================================================

Total Trades:
{report["total"]}

🟢 WIN:
{report["wins"]}

🔴 LOSS:
{report["losses"]}

🟡 OPEN:
{report["open"]}

Closed:
{report["closed"]}

Win Rate:
{report["win_rate"]:.1f}%

Net R:R:
{report["net_rr"]:+.2f}R

==================================================
CURRENT AI RISK MODEL
==================================================

SL:
{risk.get("sl_percent", BASE_SL_PERCENT):.2f}%

TP:
{risk.get("tp_percent", BASE_TP_PERCENT):.2f}%

Learning Sample:
{risk.get("sample_size", 0)}

==================================================
AI WEIGHTS
==================================================

"""

    for name, value in weights.items():

        body += (
            f"{name}: "
            f"{float(value):.3f}\n"
        )

    body += """

==================================================
TRADE HISTORY
==================================================
"""

    if not report["recent"]:

        body += (
            "\nNo trades during this 12-hour period.\n"
        )

    else:

        for i, trade in enumerate(
            report["recent"],
            1
        ):

            status = trade.get(
                "status",
                "UNKNOWN"
            )

            icon = (
                "🟢"
                if status == "WIN"
                else
                "🔴"
                if status == "LOSS"
                else
                "🟡"
            )

            body += f"""

--------------------------------------------------
TRADE #{i}
--------------------------------------------------

Coin:
{trade.get("symbol")}

Direction:
{trade.get("direction")}

Score:
{trade.get("score")}

Result:
{icon} {status}

Entry:
{trade.get("entry")}

SL:
{trade.get("stop_loss")}

TP:
{trade.get("take_profit")}

Created:
{trade.get("created_time_pk")}

Closed:
{trade.get("closed_time_pk", "-")}

Buyer Pressure:
{trade.get("buyer_pressure", "-")}%

Seller Pressure:
{trade.get("seller_pressure", "-")}%

Volume:
{trade.get("volume_ratio", "-")}x

RSI:
{trade.get("h1_rsi", "-")}

EMA:
{trade.get("h1_ema", "-")}

FVG:
{trade.get("fvg_bullish", False)
 or trade.get("fvg_bearish", False)}

Order Block:
{trade.get("ob_bullish", False)
 or trade.get("ob_bearish", False)}

Market Regime:
{trade.get("market_regime", "-")}

"""

    body += """

==================================================
AI LEARNING
==================================================

The Brain stores every closed trade.

After 50 closed trades it evaluates:

• RSI
• EMA
• Pressure
• Volume
• Structure
• FVG
• Order Block
• Support / Resistance
• 5M Entry
• Market Regime

Winning features receive more weight.
Weak features receive less weight.

AI adaptation is limited so the strategy
does not become excessively restrictive.

==================================================

Report:
12-hour rolling period

Pakistan Time:
UTC+5

==================================================
"""

    sent = send_email(
        subject,
        body
    )

    return sent


# ============================================================
# FIXED 8AM / 8PM REPORT SCHEDULER
# ============================================================

def report_slot():

    now = pakistan_datetime()

    if now.hour < REPORT_HOUR_MORNING:

        slot_date = (
            now.date() -
            timedelta(days=1)
        )

        return (
            f"{slot_date}_20"
        )

    if now.hour < REPORT_HOUR_EVENING:

        return (
            f"{now.date()}_08"
        )

    return (
        f"{now.date()}_20"
    )


def should_send_scheduled_report():

    now = pakistan_datetime()

    # Only trigger close to 8 AM or 8 PM.
    #
    # Main loop scans every ~2 minutes,
    # so a 10-minute window is enough.

    if now.hour == 8 and now.minute < 10:

        slot = f"{now.date()}_08"

    elif now.hour == 20 and now.minute < 10:

        slot = f"{now.date()}_20"

    else:

        return False

    data = load_json_file(
        REPORT_FILE,
        {}
    )

    if data.get(
        "last_slot"
    ) == slot:

        return False

    return True


def mark_report_sent():

    now = pakistan_datetime()

    if now.hour == 8:

        slot = f"{now.date()}_08"

    else:

        slot = f"{now.date()}_20"

    save_json_file(
        REPORT_FILE,
        {
            "last_slot": slot,
            "sent_at": utc_timestamp(),
            "sent_time_pk": pakistan_time()
        }
    )


def scheduled_report_check():

    if not should_send_scheduled_report():
        return

    logger.info(
        "🧠 Scheduled 12-hour report triggered."
    )

    sent = send_12_hour_report()

    if sent:

        mark_report_sent()

        logger.info(
            "12-hour report sent."
        )

    else:

        logger.warning(
            "12-hour report Gmail failed."
        )


# ============================================================
# START
# ============================================================

def main():

    logger.info(
        "================================================"
    )

    logger.info(
        "🧠 MARKET BRAIN v11 STARTED"
    )

    logger.info(
        "Coins: %s",
        len(ASSETS)
    )

    logger.info(
        "1H = Main setup"
    )

    logger.info(
        "5M = Entry only"
    )

    logger.info(
        "Base SL = %.2f%%",
        BASE_SL_PERCENT
    )

    logger.info(
        "Base TP = %.2f%%",
        BASE_TP_PERCENT
    )

    logger.info(
        "AI learning window = %s trades",
        LEARNING_WINDOW
    )

    logger.info(
        "Reports = 08:00 AM / 08:00 PM PKT"
    )

    logger.info(
        "================================================"
    )

    restore_active_trades()

    while True:

        cycle_start = time.time()

        try:

            # --------------------------------------------
            # 1. Existing trades first
            # --------------------------------------------

            check_all_active_trades()

            # --------------------------------------------
            # 2. Fixed report scheduler
            # --------------------------------------------

            scheduled_report_check()

            # --------------------------------------------
            # 3. Scan coins
            # --------------------------------------------

            for symbol in ASSETS:

                try:

                    scan_symbol(
                        symbol
                    )

                except Exception as e:

                    logger.error(
                        "%s scan error: %s",
                        symbol,
                        e
                    )

                time.sleep(
                    0.5
                )

            save_memory()

        except KeyboardInterrupt:

            logger.info(
                "Bot stopped."
            )

            save_active_trades()
            save_memory()

            break

        except Exception as e:

            logger.error(
                "Main loop error: %s",
                e
            )

        elapsed = (
            time.time() -
            cycle_start
        )

        sleep_time = max(
            1,
            SCAN_INTERVAL_SECONDS -
            elapsed
        )

        logger.info(
            "Next scan in %.0f seconds.",
            sleep_time
        )

        time.sleep(
            sleep_time
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        logger.critical(
            "Fatal error: %s",
            e
        )
