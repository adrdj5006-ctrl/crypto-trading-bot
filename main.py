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
# 🧠 MARKET BRAIN — FLEXIBLE MULTI-TIMEFRAME ENGINE
# ============================================================
#
# TIMEFRAME JOBS
#
# 1D  -> Buyer/Seller pressure + Volume
# 4H  -> Buyer/Seller pressure + Volume
# 1H  -> Last 50 candles + High/Low break + FVG + Order Block
# 30M  -> Support / Resistance
# 5M  -> Entry confirmation
#
# TRADE
# BUY:
# Entry = current 5M market price
# SL    = -1%
# TP    = +2%
#
# SELL:
# Entry = current 5M market price
# SL    = +1%
# TP    = -2%
#
# SCAN = every 2 minutes
# SYMBOLS = 20
# TIMEZONE = Pakistan
#
# ============================================================


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)


# ============================================================
# PAKISTAN TIME
# ============================================================

PK_TZ = timezone(timedelta(hours=5))


def pakistan_time():
    return datetime.now(PK_TZ).strftime(
        "%Y-%m-%d %I:%M:%S %p"
    )


def pakistan_datetime():
    return datetime.now(PK_TZ)


# ============================================================
# GMAIL
# ============================================================

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")

GMAIL_RECEIVER = os.environ.get(
    "GMAIL_RECEIVER",
    GMAIL_USER
)


def send_email(subject, body):

    if not GMAIL_USER or not GMAIL_PASS:
        logging.error(
            "GMAIL_USER / GMAIL_PASS missing."
        )
        return False

    if not GMAIL_RECEIVER:
        logging.error(
            "GMAIL_RECEIVER missing."
        )
        return False

    try:

        message = EmailMessage()

        message["Subject"] = subject
        message["From"] = GMAIL_USER
        message["To"] = GMAIL_RECEIVER

        message.set_content(body)

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

            server.send_message(message)

        logging.info(
            "Gmail alert sent."
        )

        return True

    except Exception as e:

        logging.error(
            f"Gmail error: {e}"
        )

        return False


# ============================================================
# CONFIG
# ============================================================

SCAN_INTERVAL_SECONDS = 120

TAKE_PROFIT_PERCENT = 2.0
STOP_LOSS_PERCENT = 1.0

MIN_SIGNAL_SCORE = 60

COOLDOWN_SECONDS = 15 * 60

MEMORY_FILE = "market_brain_memory.json"

TRADE_LOG_FILE = "ai_trade_learning_log.json"

LAST_ALERT_FILE = "last_trade_alerts.json"


# ============================================================
# 20 BINANCE ASSETS
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
    "MATICUSDT",
    "PAXGUSDT"
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
# HTTP
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
})


# ============================================================
# JSON SAFE
# ============================================================

def json_safe(value):

    if value is None:
        return None

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, str):
        return value

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return value.tolist()

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


# ============================================================
# MEMORY
# ============================================================

def default_memory():

    return {
        "created_at": time.time(),
        "observations": 0,
        "wins": 0,
        "losses": 0,
        "expired": 0,
        "closed_trades": 0,
        "symbols_seen": [],
        "patterns": {},
        "last_learning_update": None
    }


def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return default_memory()

    try:

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        base = default_memory()

        if isinstance(data, dict):
            base.update(data)

        return base

    except Exception as e:

        logging.warning(
            f"Memory load error: {e}"
        )

        return default_memory()


BRAIN_MEMORY = load_memory()


def save_memory():

    global BRAIN_MEMORY

    BRAIN_MEMORY = json_safe(
        BRAIN_MEMORY
    )

    BRAIN_MEMORY[
        "last_learning_update"
    ] = time.time()

    temp = MEMORY_FILE + ".tmp"

    try:

        with open(
            temp,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                BRAIN_MEMORY,
                f,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

        os.replace(
            temp,
            MEMORY_FILE
        )

    except Exception as e:

        logging.error(
            f"Memory save error: {e}"
        )


# ============================================================
# TRADE LOG
# ============================================================

def load_trade_logs():

    if not os.path.exists(TRADE_LOG_FILE):
        return []

    try:

        with open(
            TRADE_LOG_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return data if isinstance(
            data,
            list
        ) else []

    except Exception:

        return []


def save_trade_log(trade):

    logs = load_trade_logs()

    logs.append(
        json_safe(trade)
    )

    try:

        with open(
            TRADE_LOG_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                logs,
                f,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

    except Exception as e:

        logging.error(
            f"Trade log error: {e}"
        )


# ============================================================
# BINANCE ENDPOINTS
# ============================================================

BINANCE_ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines"
]


# ============================================================
# FETCH CANDLES
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
                timeout=10
            )

            if response.status_code != 200:
                continue

            raw = response.json()

            if not isinstance(
                raw,
                list
            ):
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

        except Exception as e:

            logging.debug(
                f"{symbol} {interval}: {e}"
            )

    return None


# ============================================================
# PRESSURE
# ============================================================

def pressure_analysis(
    df,
    candles=10
):

    if df is None or len(df) < candles + 2:

        return {
            "buyer": 50.0,
            "seller": 50.0,
            "dominant": "NEUTRAL"
        }

    # Ignore currently forming candle
    data = df.iloc[
        -(candles + 1):-1
    ]

    buyer = 0.0
    seller = 0.0

    for _, candle in data.iterrows():

        body = abs(
            candle["close"] -
            candle["open"]
        )

        total_range = (
            candle["high"] -
            candle["low"]
        )

        if total_range <= 0:
            continue

        body_strength = (
            body / total_range
        )

        weighted = (
            body_strength *
            candle["volume"]
        )

        if candle["close"] > candle["open"]:
            buyer += weighted

        elif candle["close"] < candle["open"]:
            seller += weighted

    total = buyer + seller

    if total <= 0:

        return {
            "buyer": 50.0,
            "seller": 50.0,
            "dominant": "NEUTRAL"
        }

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
            1
        ),
        "seller": round(
            seller_pct,
            1
        ),
        "dominant": dominant
    }


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(
    df,
    lookback=20
):

    if df is None or len(df) < lookback + 2:

        return {
            "ratio": 1.0,
            "strong": False
        }

    closed = df.iloc[:-1]

    current = closed.iloc[-1]

    average = closed[
        "volume"
    ].tail(lookback).mean()

    if average <= 0:

        return {
            "ratio": 1.0,
            "strong": False
        }

    ratio = (
        current["volume"] /
        average
    )

    return {
        "ratio": round(
            float(ratio),
            2
        ),
        "strong": bool(
            ratio >= 1.10
        )
    }


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    df,
    period=14
):

    if df is None or len(df) < period + 2:
        return 50.0

    delta = df["close"].diff()

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

    loss_value = avg_loss.iloc[-2]

    if loss_value <= 0:
        return 100.0

    rs = (
        avg_gain.iloc[-2] /
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

def ema_direction(
    df,
    fast=20,
    slow=50
):

    if df is None or len(df) < slow + 2:
        return "NEUTRAL"

    closed = df.iloc[:-1]

    fast_ema = closed[
        "close"
    ].ewm(
        span=fast,
        adjust=False
    ).mean().iloc[-1]

    slow_ema = closed[
        "close"
    ].ewm(
        span=slow,
        adjust=False
    ).mean().iloc[-1]

    if fast_ema > slow_ema:
        return "BULLISH"

    if fast_ema < slow_ema:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# 1H BREAK
# ============================================================

def one_hour_break(
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

    resistance = previous[
        "high"
    ].max()

    support = previous[
        "low"
    ].min()

    bullish = (
        current["high"] > resistance
        and current["close"] > resistance
    )

    bearish = (
        current["low"] < support
        and current["close"] < support
    )

    return {
        "bullish": bool(bullish),
        "bearish": bool(bearish),
        "high": float(resistance),
        "low": float(support)
    }


# ============================================================
# FVG
# ============================================================

def detect_fvg(df):

    result = {
        "bullish": False,
        "bearish": False,
        "low": None,
        "high": None
    }

    if df is None or len(df) < 10:
        return result

    closed = df.iloc[:-1]

    if len(closed) < 4:
        return result

    c1 = closed.iloc[-3]
    c3 = closed.iloc[-1]

    # Bullish FVG:
    # latest candle low > candle 1 high
    if c3["low"] > c1["high"]:

        result["bullish"] = True

        result["low"] = float(
            c1["high"]
        )

        result["high"] = float(
            c3["low"]
        )

    # Bearish FVG:
    # latest candle high < candle 1 low
    elif c3["high"] < c1["low"]:

        result["bearish"] = True

        result["low"] = float(
            c3["high"]
        )

        result["high"] = float(
            c1["low"]
        )

    return result


# ============================================================
# ORDER BLOCK
# ============================================================

def detect_order_block(df):

    result = {
        "bullish": False,
        "bearish": False,
        "low": None,
        "high": None
    }

    if df is None or len(df) < 10:
        return result

    closed = df.iloc[:-1]

    last = closed.iloc[-1]
    previous = closed.iloc[-2]

    # Previous bearish candle followed by bullish move
    if (
        previous["close"] < previous["open"]
        and last["close"] > last["open"]
        and last["close"] > previous["high"]
    ):

        result["bullish"] = True

        result["low"] = float(
            previous["low"]
        )

        result["high"] = float(
            previous["open"]
        )

    # Previous bullish candle followed by bearish move
    elif (
        previous["close"] > previous["open"]
        and last["close"] < last["open"]
        and last["close"] < previous["low"]
    ):

        result["bearish"] = True

        result["low"] = float(
            previous["open"]
        )

        result["high"] = float(
            previous["high"]
        )

    return result


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def support_resistance(
    df,
    lookback=50
):

    if df is None or len(df) < lookback + 2:

        return {
            "support": None,
            "resistance": None,
            "near_support": False,
            "near_resistance": False
        }

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

    distance_support = (
        abs(price - support) /
        price
    )

    distance_resistance = (
        abs(resistance - price) /
        price
    )

    return {
        "support": support,
        "resistance": resistance,
        "near_support": bool(
            distance_support <= 0.008
        ),
        "near_resistance": bool(
            distance_resistance <= 0.008
        )
    }


# ============================================================
# 5M ENTRY
# ============================================================

def five_min_entry(
    df,
    direction
):

    result = {
        "confirmed": False,
        "price": None,
        "reason": "NO_CONFIRMATION"
    }

    if df is None or len(df) < 10:
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
        body / candle_range
    )

    if direction == "BUY":

        bullish = (
            last["close"] >
            last["open"]
        )

        if bullish and body_ratio >= 0.25:

            return {
                "confirmed": True,
                "price": price,
                "reason": "5M_BULLISH_CONFIRMATION"
            }

    elif direction == "SELL":

        bearish = (
            last["close"] <
            last["open"]
        )

        if bearish and body_ratio >= 0.25:

            return {
                "confirmed": True,
                "price": price,
                "reason": "5M_BEARISH_CONFIRMATION"
            }

    return result


# ============================================================
# TRADE LEVELS
# ============================================================

def calculate_trade_levels(
    direction,
    entry
):

    if direction == "BUY":

        stop_loss = entry * (
            1 - STOP_LOSS_PERCENT / 100
        )

        take_profit = entry * (
            1 + TAKE_PROFIT_PERCENT / 100
        )

    else:

        stop_loss = entry * (
            1 + STOP_LOSS_PERCENT / 100
        )

        take_profit = entry * (
            1 - TAKE_PROFIT_PERCENT / 100
        )

    return (
        float(entry),
        float(stop_loss),
        float(take_profit)
    )


# ============================================================
# FLEXIBLE SIGNAL ENGINE
# ============================================================

def analyze_market(
    symbol,
    d1,
    h4,
    h1,
    m30,
    m5
):

    if any(
        x is None
        for x in [
            d1,
            h4,
            h1,
            m30,
            m5
        ]
    ):
        return None

    # --------------------------------------------------------
    # 1D
    # --------------------------------------------------------

    daily_pressure = pressure_analysis(
        d1,
        candles=8
    )

    daily_volume = volume_analysis(
        d1,
        lookback=8
    )

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    h4_pressure = pressure_analysis(
        h4,
        candles=10
    )

    h4_volume = volume_analysis(
        h4,
        lookback=10
    )

    # --------------------------------------------------------
    # 1H / LAST 50
    # --------------------------------------------------------

    h1_break = one_hour_break(
        h1,
        lookback=50
    )

    h1_fvg = detect_fvg(
        h1
    )

    h1_ob = detect_order_block(
        h1
    )

    h1_ema = ema_direction(
        h1
    )

    h1_rsi = calculate_rsi(
        h1
    )

    # --------------------------------------------------------
    # 30M
    # --------------------------------------------------------

    sr = support_resistance(
        m30,
        lookback=50
    )

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    current_price = float(
        m5["close"].iloc[-1]
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    buy_score = 0
    sell_score = 0

    buy_reasons = []
    sell_reasons = []

    # ========================================================
    # DAILY PRESSURE
    # ========================================================

    if daily_pressure["buyer"] >= 55:

        buy_score += 12
        buy_reasons.append(
            "1D buyer pressure"
        )

    elif daily_pressure["seller"] >= 55:

        sell_score += 12
        sell_reasons.append(
            "1D seller pressure"
        )

    # Daily volume is supportive, not mandatory
    if daily_volume["strong"]:

        if daily_pressure["dominant"] == "BUYER":
            buy_score += 5

        elif daily_pressure["dominant"] == "SELLER":
            sell_score += 5

    # ========================================================
    # 4H PRESSURE
    # ========================================================

    if h4_pressure["buyer"] >= 55:

        buy_score += 12
        buy_reasons.append(
            "4H buyer pressure"
        )

    elif h4_pressure["seller"] >= 55:

        sell_score += 12
        sell_reasons.append(
            "4H seller pressure"
        )

    if h4_volume["strong"]:

        if h4_pressure["dominant"] == "BUYER":
            buy_score += 5

        elif h4_pressure["dominant"] == "SELLER":
            sell_score += 5

    # ========================================================
    # 1H BREAK
    # ========================================================

    if h1_break["bullish"]:

        buy_score += 18

        buy_reasons.append(
            "1H high breakout close"
        )

    if h1_break["bearish"]:

        sell_score += 18

        sell_reasons.append(
            "1H low breakdown close"
        )

    # ========================================================
    # FVG
    # ========================================================

    if h1_fvg["bullish"]:

        buy_score += 8

        buy_reasons.append(
            "1H bullish FVG"
        )

    if h1_fvg["bearish"]:

        sell_score += 8

        sell_reasons.append(
            "1H bearish FVG"
        )

    # ========================================================
    # ORDER BLOCK
    # ========================================================

    if h1_ob["bullish"]:

        buy_score += 8

        buy_reasons.append(
            "1H bullish order block"
        )

    if h1_ob["bearish"]:

        sell_score += 8

        sell_reasons.append(
            "1H bearish order block"
        )

    # ========================================================
    # EMA
    # ========================================================

    if h1_ema == "BULLISH":

        buy_score += 6

        buy_reasons.append(
            "1H EMA bullish"
        )

    elif h1_ema == "BEARISH":

        sell_score += 6

        sell_reasons.append(
            "1H EMA bearish"
        )

    # ========================================================
    # RSI
    #
    # NORMAL — NOT STRICT
    # ========================================================

    if 45 <= h1_rsi <= 68:

        if h1_ema == "BULLISH":
            buy_score += 5

        elif h1_ema == "BEARISH":
            sell_score += 5

    # ========================================================
    # 30M SUPPORT
    # ========================================================

    if sr["near_support"]:

        buy_score += 10

        buy_reasons.append(
            "30M near support"
        )

    if sr["near_resistance"]:

        sell_score += 10

        sell_reasons.append(
            "30M near resistance"
        )

    # ========================================================
    # 5M ENTRY
    # ========================================================

    if buy_score >= MIN_SIGNAL_SCORE:

        entry_check = five_min_entry(
            m5,
            "BUY"
        )

        if entry_check["confirmed"]:

            buy_score += 5

    if sell_score >= MIN_SIGNAL_SCORE:

        entry_check = five_min_entry(
            m5,
            "SELL"
        )

        if entry_check["confirmed"]:

            sell_score += 5

    # ========================================================
    # FINAL DECISION
    # ========================================================

    direction = "HOLD"
    score = max(
        buy_score,
        sell_score
    )

    reasons = []

    if (
        buy_score >= MIN_SIGNAL_SCORE
        and buy_score > sell_score + 5
    ):

        entry_check = five_min_entry(
            m5,
            "BUY"
        )

        if entry_check["confirmed"]:

            direction = "BUY"
            score = buy_score
            reasons = buy_reasons

            entry = entry_check["price"]

        else:

            direction = "WATCH"
            entry = current_price

    elif (
        sell_score >= MIN_SIGNAL_SCORE
        and sell_score > buy_score + 5
    ):

        entry_check = five_min_entry(
            m5,
            "SELL"
        )

        if entry_check["confirmed"]:

            direction = "SELL"
            score = sell_score
            reasons = sell_reasons

            entry = entry_check["price"]

        else:

            direction = "WATCH"
            entry = current_price

    else:

        entry = current_price

    # ========================================================
    # TRADE LEVELS
    # ========================================================

    trade = None

    if direction in [
        "BUY",
        "SELL"
    ]:

        entry, sl, tp = calculate_trade_levels(
            direction,
            entry
        )

        trade = {
            "symbol": symbol,
            "direction": direction,
            "score": int(score),
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "created_at": time.time(),
            "created_time_pk": pakistan_time(),
            "status": "OPEN",
            "reasons": reasons
        }

    return {
        "symbol": symbol,
        "direction": direction,
        "score": int(score),
        "buy_score": int(buy_score),
        "sell_score": int(sell_score),
        "price": current_price,
        "trade": trade,

        "daily_buyer": daily_pressure["buyer"],
        "daily_seller": daily_pressure["seller"],

        "h4_buyer": h4_pressure["buyer"],
        "h4_seller": h4_pressure["seller"],

        "h1_rsi": h1_rsi,
        "h1_ema": h1_ema,

        "h1_high_break": h1_break["bullish"],
        "h1_low_break": h1_break["bearish"],

        "fvg_bullish": h1_fvg["bullish"],
        "fvg_bearish": h1_fvg["bearish"],

        "ob_bullish": h1_ob["bullish"],
        "ob_bearish": h1_ob["bearish"],

        "near_support": sr["near_support"],
        "near_resistance": sr["near_resistance"],

        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons
    }


# ============================================================
# DUPLICATE SIGNAL PROTECTION
# ============================================================

def should_send_trade(
    symbol,
    direction,
    score
):

    tracker = SIGNAL_TRACKER[
        symbol
    ]

    now = time.time()

    if (
        tracker["last_signal"] == direction
        and
        now -
        tracker["last_signal_time"]
        < COOLDOWN_SECONDS
    ):

        return False

    tracker["last_signal"] = direction

    tracker["last_signal_time"] = now

    return True


# ============================================================
# TRADE EMAIL
# ============================================================

def create_trade_email(
    trade
):

    direction = trade[
        "direction"
    ]

    symbol = trade[
        "symbol"
    ]

    subject = (
        f"🧠 MARKET BRAIN "
        f"{direction} — {symbol}"
    )

    body = f"""
🧠 MARKET BRAIN TRADE ALERT

Coin:
{symbol}

Signal:
{direction}

Score:
{trade["score"]}

Entry:
{trade["entry"]:.8f}

Stop Loss:
{trade["stop_loss"]:.8f}
(-1.00%)

Target:
{trade["take_profit"]:.8f}
(+2.00%)

Pakistan Time:
{trade["created_time_pk"]}

Reasons:
"""

    for reason in trade[
        "reasons"
    ]:

        body += (
            f"\n• {reason}"
        )

    body += """

Risk / Reward:
1 : 2

Timeframes:
1D → Pressure + Volume
4H → Pressure + Volume
1H → 50 Candle Break + FVG + OB
30M → Support / Resistance
5M → Entry Confirmation

This is a market-analysis signal.
"""

    return subject, body


# ============================================================
# CHECK OPEN TRADES
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
        5
    )

    if df is None or len(df) < 2:
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

    elif direction == "SELL":

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

        trade[
            "closed_price"
        ] = current

        trade[
            "closed_time_pk"
        ] = pakistan_time()

        return result

    return None


# ============================================================
# UPDATE LEARNING
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

    pattern_key = (
        f'{trade["direction"]}_'
        f'{trade["score"]}'
    )

    patterns = BRAIN_MEMORY[
        "patterns"
    ]

    if pattern_key not in patterns:

        patterns[pattern_key] = {
            "wins": 0,
            "losses": 0
        }

    if result == "WIN":

        patterns[
            pattern_key
        ]["wins"] += 1

    elif result == "LOSS":

        patterns[
            pattern_key
        ]["losses"] += 1

    save_memory()

    save_trade_log(
        trade
    )


# ============================================================
# 24 HOUR REPORT
# ============================================================

def send_24_hour_report():

    logs = load_trade_logs()

    now = time.time()

    day_ago = now - (
        24 * 60 * 60
    )

    recent = []

    for trade in logs:

        created = trade.get(
            "created_at",
            0
        )

        if created >= day_ago:

            recent.append(
                trade
            )

    total = len(recent)

    wins = sum(
        1
        for x in recent
        if x.get("status") == "WIN"
    )

    losses = sum(
        1
        for x in recent
        if x.get("status") == "LOSS"
    )

    open_trades = sum(
        1
        for x in recent
        if x.get("status") == "OPEN"
    )

    if total > 0:

        win_rate = (
            wins / total
        ) * 100

    else:

        win_rate = 0

    subject = (
        "🧠 MARKET BRAIN — "
        "24 HOUR REPORT"
    )

    body = f"""
🧠 MARKET BRAIN 24 HOUR REPORT

Pakistan Time:
{pakistan_time()}

Total Signals:
{total}

Wins:
{wins}

Losses:
{losses}

Open:
{open_trades}

Win Rate:
{win_rate:.1f}%

==================================================
TRADES
==================================================
"""

    for trade in recent:

        body += f"""

{trade.get("symbol")}
{trade.get("direction")}
Score: {trade.get("score")}
Entry: {trade.get("entry")}
SL: {trade.get("stop_loss")}
TP: {trade.get("take_profit")}
Status: {trade.get("status")}
Created: {trade.get("created_time_pk")}
Closed: {trade.get("closed_time_pk", "-")}
"""

    send_email(
        subject,
        body
    )


# ============================================================
# MAIN SCANNER
# ============================================================

def scan_symbol(
    symbol
):

    logging.info(
        f"Scanning {symbol}"
    )

    # --------------------------------------------------------
    # 1D
    # --------------------------------------------------------

    d1 = fetch_candles(
        symbol,
        "1d",
        30
    )

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    h4 = fetch_candles(
        symbol,
        "4h",
        50
    )

    # --------------------------------------------------------
    # 1H — 50 CANDLES
    # --------------------------------------------------------

    h1 = fetch_candles(
        symbol,
        "1h",
        60
    )

    # --------------------------------------------------------
    # 30M
    # --------------------------------------------------------

    m30 = fetch_candles(
        symbol,
        "30m",
        60
    )

    # --------------------------------------------------------
    # 5M
    # --------------------------------------------------------

    m5 = fetch_candles(
        symbol,
        "5m",
        30
    )

    if any(
        x is None
        for x in [
            d1,
            h4,
            h1,
            m30,
            m5
        ]
    ):

        logging.warning(
            f"{symbol}: incomplete market data"
        )

        return

    analysis = analyze_market(
        symbol,
        d1,
        h4,
        h1,
        m30,
        m5
    )

    if analysis is None:
        return

    BRAIN_MEMORY[
        "observations"
    ] += 1

    if symbol not in BRAIN_MEMORY[
        "symbols_seen"
    ]:

        BRAIN_MEMORY[
            "symbols_seen"
        ].append(symbol)

    logging.info(
        f"{symbol} | "
        f"BUY={analysis['buy_score']} | "
        f"SELL={analysis['sell_score']} | "
        f"{analysis['direction']}"
    )

    # ========================================================
    # NEW TRADE
    # ========================================================

    trade = analysis[
        "trade"
    ]

    if trade is not None:

        if should_send_trade(
            symbol,
            trade["direction"],
            trade["score"]
        ):

            subject, body = create_trade_email(
                trade
            )

            sent = send_email(
                subject,
                body
            )

            if sent:

                SIGNAL_TRACKER[
                    symbol
                ]["active_trade"] = trade

                save_trade_log(
                    trade
                )

                logging.info(
                    f"TRADE CREATED: "
                    f"{symbol} "
                    f"{trade['direction']}"
                )

    save_memory()


# ============================================================
# CHECK ALL ACTIVE TRADES
# ============================================================

def check_all_active_trades():

    for symbol in ASSETS:

        trade = SIGNAL_TRACKER[
            symbol
        ]["active_trade"]

        if trade is None:
            continue

        result = check_active_trade(
            symbol,
            trade
        )

        if result:

            logging.info(
                f"{symbol} trade "
                f"closed: {result}"
            )

            learn_from_trade(
                trade,
                result
            )

            SIGNAL_TRACKER[
                symbol
            ]["active_trade"] = None


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    logging.info(
        "=========================================="
    )

    logging.info(
        "🧠 MARKET BRAIN STARTED"
    )

    logging.info(
        f"Assets: {len(ASSETS)}"
    )

    logging.info(
        "Scan interval: 2 minutes"
    )

    logging.info(
        "TP: 2%"
    )

    logging.info(
        "SL: 1%"
    )

    logging.info(
        "Timezone: Pakistan"
    )

    logging.info(
        "=========================================="
    )

    last_report = time.time()

    while True:

        cycle_start = time.time()

        try:

            # ------------------------------------------------
            # Check existing trades
            # ------------------------------------------------

            check_all_active_trades()

            # ------------------------------------------------
            # Scan 20 coins
            # ------------------------------------------------

            for symbol in ASSETS:

                try:

                    scan_symbol(
                        symbol
                    )

                except Exception as e:

                    logging.error(
                        f"{symbol} scan error: {e}"
                    )

                time.sleep(
                    0.5
                )

            # ------------------------------------------------
            # 24 Hour Gmail report
            # ------------------------------------------------

            if (
                time.time() -
                last_report
                >= HOURLY_REPORT_SECONDS
            ):

                send_24_hour_report()

                last_report = time.time()

        except KeyboardInterrupt:

            logging.info(
                "Market Brain stopped."
            )

            break

        except Exception as e:

            logging.error(
                f"Main loop error: {e}"
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

        logging.info(
            f"Next scan in "
            f"{sleep_time:.0f} seconds."
        )

        time.sleep(
            sleep_time
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
