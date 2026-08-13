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
# 🧠 MARKET BRAIN v10
# ============================================================
#
# CORE STRUCTURE
#
# 1H  -> MAIN MARKET DIRECTION
#        Pressure
#        Volume
#        EMA
#        RSI
#        High/Low structure
#        FVG
#        Order Block
#        Support / Resistance
#
# 5M  -> ENTRY ONLY
#        Candle direction
#        Candle body confirmation
#        NO FVG
#        NO ORDER BLOCK
#        NO RSI
#        NO EMA
#
# TRADE
#
# BUY:
# Entry = 5M confirmation price
# SL    = -1%
# TP    = +2%
#
# SELL:
# Entry = 5M confirmation price
# SL    = +1%
# TP    = -2%
#
# 20 COINS
# SCAN = every 2 minutes
# TIMEZONE = Pakistan
#
# Gmail:
# New trade alert
# Trade close result
# Rolling 24-hour report
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
# GMAIL CONFIG
# ============================================================

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")

GMAIL_RECEIVER = os.environ.get(
    "GMAIL_RECEIVER",
    GMAIL_USER or ""
)


# ============================================================
# SEND EMAIL
# ============================================================

def send_email(subject, body):

    if not GMAIL_USER:
        logger.error("GMAIL_USER is missing.")
        return False

    if not GMAIL_PASS:
        logger.error("GMAIL_PASS is missing.")
        return False

    if not GMAIL_RECEIVER:
        logger.error("GMAIL_RECEIVER is missing.")
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

        logger.info(
            "Gmail alert sent successfully."
        )

        return True

    except Exception as e:

        logger.error(
            f"Gmail error: {e}"
        )

        return False


# ============================================================
# CONFIG
# ============================================================

SCAN_INTERVAL_SECONDS = 120

TAKE_PROFIT_PERCENT = 2.0
STOP_LOSS_PERCENT = 1.0

# Strong signal threshold
MIN_SIGNAL_SCORE = 65

# Required difference between directions
MIN_DIRECTION_GAP = 10

# Same direction cooldown
COOLDOWN_SECONDS = 15 * 60

# 24-hour report
REPORT_INTERVAL_SECONDS = 24 * 60 * 60

# Files
MEMORY_FILE = "market_brain_memory.json"
TRADE_LOG_FILE = "ai_trade_learning_log.json"
ACTIVE_TRADES_FILE = "active_trades.json"
LAST_REPORT_FILE = "last_24h_report.json"


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
    "POLUSDT",
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
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
})


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
# GENERIC JSON LOAD
# ============================================================

def load_json_file(
    filename,
    default
):

    if not os.path.exists(filename):
        return default

    try:

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return data

    except Exception as e:

        logger.warning(
            f"{filename} load error: {e}"
        )

        return default


# ============================================================
# GENERIC JSON SAVE
# ============================================================

def save_json_file(
    filename,
    data
):

    temp = filename + ".tmp"

    try:

        safe_data = json_safe(data)

        with open(
            temp,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                safe_data,
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
            f"{filename} save error: {e}"
        )

        return False


# ============================================================
# MEMORY
# ============================================================

def default_memory():

    return {
        "created_at": utc_timestamp(),
        "observations": 0,
        "signals": 0,
        "wins": 0,
        "losses": 0,
        "closed_trades": 0,
        "symbols_seen": [],
        "patterns": {},
        "last_learning_update": None
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

    if not isinstance(data, list):
        return []

    return data


def save_trade_logs(logs):

    save_json_file(
        TRADE_LOG_FILE,
        logs
    )


# ============================================================
# ADD TRADE LOG
# ============================================================

def add_trade_log(trade):

    logs = load_trade_logs()

    trade_id = trade.get(
        "trade_id"
    )

    # Prevent duplicate trade records
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


# ============================================================
# UPDATE EXISTING TRADE
# ============================================================

def update_trade_log(trade):

    logs = load_trade_logs()

    trade_id = trade.get(
        "trade_id"
    )

    found = False

    for index, existing in enumerate(logs):

        if existing.get(
            "trade_id"
        ) == trade_id:

            logs[index] = json_safe(
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
# LOAD ACTIVE TRADES
# ============================================================

def load_active_trades():

    data = load_json_file(
        ACTIVE_TRADES_FILE,
        {}
    )

    if not isinstance(data, dict):
        return {}

    return data


# ============================================================
# SAVE ACTIVE TRADES
# ============================================================

def save_active_trades():

    data = {}

    for symbol in ASSETS:

        trade = SIGNAL_TRACKER[
            symbol
        ].get("active_trade")

        if trade is not None:

            data[symbol] = trade

    save_json_file(
        ACTIVE_TRADES_FILE,
        data
    )


# ============================================================
# RESTORE ACTIVE TRADES
# ============================================================

def restore_active_trades():

    active = load_active_trades()

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
            f"Restored active trade: "
            f"{symbol} "
            f"{trade.get('direction')}"
        )


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
                timeout=12
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

            numeric_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_base",
                "taker_buy_quote"
            ]

            for col in numeric_columns:

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

            df = df.reset_index(
                drop=True
            )

            return df

        except Exception as e:

            logger.debug(
                f"{symbol} {interval}: {e}"
            )

    return None


# ============================================================
# PRESSURE ANALYSIS
# ============================================================

def pressure_analysis(
    df,
    candles=12
):

    neutral = {
        "buyer": 50.0,
        "seller": 50.0,
        "dominant": "NEUTRAL"
    }

    if df is None:
        return neutral

    if len(df) < candles + 3:
        return neutral

    # Ignore currently forming candle
    closed = df.iloc[:-1]

    data = closed.tail(
        candles
    )

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

        body_strength = (
            body /
            candle_range
        )

        volume = max(
            float(candle["volume"]),
            0.0
        )

        weighted_pressure = (
            body_strength *
            volume
        )

        if candle["close"] > candle["open"]:

            buyer += weighted_pressure

        elif candle["close"] < candle["open"]:

            seller += weighted_pressure

    total = buyer + seller

    if total <= 0:
        return neutral

    buyer_pct = (
        buyer /
        total
    ) * 100

    seller_pct = (
        seller /
        total
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
# VOLUME ANALYSIS
# ============================================================

def volume_analysis(
    df,
    lookback=20
):

    if df is None:
        return {
            "ratio": 1.0,
            "strong": False,
            "direction": "NEUTRAL"
        }

    if len(df) < lookback + 3:

        return {
            "ratio": 1.0,
            "strong": False,
            "direction": "NEUTRAL"
        }

    closed = df.iloc[:-1]

    current = closed.iloc[-1]

    previous_volume = closed[
        "volume"
    ].iloc[
        -(lookback + 1):-1
    ]

    average = previous_volume.mean()

    if average <= 0:

        return {
            "ratio": 1.0,
            "strong": False,
            "direction": "NEUTRAL"
        }

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
# RSI — 1H ONLY
# ============================================================

def calculate_rsi(
    df,
    period=14
):

    if df is None:
        return 50.0

    if len(df) < period + 3:
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
        100 /
        (1 + rs)
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
# EMA — 1H ONLY
# ============================================================

def ema_direction(
    df,
    fast=20,
    slow=50
):

    if df is None:
        return "NEUTRAL"

    if len(df) < slow + 3:
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
# 1H MARKET STRUCTURE
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

    if df is None:
        return result

    if len(df) < lookback + 3:
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

    bullish = (
        current["high"] > resistance
        and
        current["close"] > resistance
    )

    bearish = (
        current["low"] < support
        and
        current["close"] < support
    )

    return {
        "bullish": bool(bullish),
        "bearish": bool(bearish),
        "high": resistance,
        "low": support
    }

   # ============================================================
# 1H FVG
# ============================================================

def detect_fvg(df):

    result = {
        "bullish": False,
        "bearish": False
    }

    if df is None:
        return result

    if len(df) < 10:
        return result

    closed = df.iloc[:-1]

    if len(closed) < 4:
        return result

    c1 = closed.iloc[-3]
    c3 = closed.iloc[-1]

    if c3["low"] > c1["high"]:

        result["bullish"] = True

    elif c3["high"] < c1["low"]:

        result["bearish"] = True

    return result


# ============================================================
# 1H ORDER BLOCK
# ============================================================

def detect_order_block(df):

    result = {
        "bullish": False,
        "bearish": False
    }

    if df is None:
        return result

    if len(df) < 10:
        return result

    closed = df.iloc[:-1]

    last = closed.iloc[-1]
    previous = closed.iloc[-2]

    # Bullish OB
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

        result["bullish"] = True

    # Bearish OB
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

        result["bearish"] = True

    return result


# ============================================================
# 1H SUPPORT / RESISTANCE
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

    if df is None:
        return result

    if len(df) < lookback + 3:
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

    support_distance = (
        abs(price - support) /
        price
    )

    resistance_distance = (
        abs(resistance - price) /
        price
    )

    result.update({
        "support": support,
        "resistance": resistance,
        "near_support": bool(
            support_distance <= 0.008
        ),
        "near_resistance": bool(
            resistance_distance <= 0.008
        )
    })

    return result


# ============================================================
# 5M ENTRY ONLY
# ============================================================
#
# IMPORTANT:
#
# No RSI
# No EMA
# No FVG
# No Order Block
# No Support/Resistance
#
# Only candle confirmation.
#
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

    if df is None:
        return result

    if len(df) < 5:
        return result

    # Ignore currently forming candle
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

    # --------------------------------------------
    # BUY
    # --------------------------------------------

    if direction == "BUY":

        bullish = (
            last["close"] >
            last["open"]
        )

        if (
            bullish
            and
            body_ratio >= 0.25
        ):

            return {
                "confirmed": True,
                "price": price,
                "reason": "5M_BULLISH_ENTRY"
            }

    # --------------------------------------------
    # SELL
    # --------------------------------------------

    if direction == "SELL":

        bearish = (
            last["close"] <
            last["open"]
        )

        if (
            bearish
            and
            body_ratio >= 0.25
        ):

            return {
                "confirmed": True,
                "price": price,
                "reason": "5M_BEARISH_ENTRY"
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

        stop_loss = (
            entry *
            (
                1 -
                STOP_LOSS_PERCENT /
                100
            )
        )

        take_profit = (
            entry *
            (
                1 +
                TAKE_PROFIT_PERCENT /
                100
            )
        )

    else:

        stop_loss = (
            entry *
            (
                1 +
                STOP_LOSS_PERCENT /
                100
            )
        )

        take_profit = (
            entry *
            (
                1 -
                TAKE_PROFIT_PERCENT /
                100
            )
        )

    return (
        float(entry),
        float(stop_loss),
        float(take_profit)
    )


# ============================================================
# SIGNAL ENGINE
# ============================================================

def analyze_market(
    symbol,
    h1,
    m5
):

    if h1 is None or m5 is None:
        return None

    # ========================================================
    # 1H ANALYSIS
    # ========================================================

    pressure = pressure_analysis(
        h1,
        candles=12
    )

    volume = volume_analysis(
        h1,
        lookback=20
    )

    structure = one_hour_structure(
        h1,
        lookback=50
    )

    fvg = detect_fvg(
        h1
    )

    order_block = detect_order_block(
        h1
    )

    ema = ema_direction(
        h1
    )

    rsi = calculate_rsi(
        h1
    )

    sr = support_resistance(
        h1,
        lookback=50
    )

    # ========================================================
    # SCORES
    # ========================================================

    buy_score = 0
    sell_score = 0

    buy_reasons = []
    sell_reasons = []

    # --------------------------------------------------------
    # PRESSURE
    # --------------------------------------------------------

    if pressure["buyer"] >= 55:

        buy_score += 20

        buy_reasons.append(
            f"1H Buyer Pressure "
            f"{pressure['buyer']:.1f}%"
        )

    elif pressure["seller"] >= 55:

        sell_score += 20

        sell_reasons.append(
            f"1H Seller Pressure "
            f"{pressure['seller']:.1f}%"
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if volume["strong"]:

        if volume["direction"] == "BUYER":

            buy_score += 12

            buy_reasons.append(
                f"Strong buyer volume "
                f"{volume['ratio']:.2f}x"
            )

        elif volume["direction"] == "SELLER":

            sell_score += 12

            sell_reasons.append(
                f"Strong seller volume "
                f"{volume['ratio']:.2f}x"
            )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    if ema == "BULLISH":

        buy_score += 10

        buy_reasons.append(
            "1H EMA bullish"
        )

    elif ema == "BEARISH":

        sell_score += 10

        sell_reasons.append(
            "1H EMA bearish"
        )

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    if structure["bullish"]:

        buy_score += 18

        buy_reasons.append(
            "1H bullish structure break"
        )

    if structure["bearish"]:

        sell_score += 18

        sell_reasons.append(
            "1H bearish structure break"
        )

    # --------------------------------------------------------
    # FVG
    # --------------------------------------------------------

    if fvg["bullish"]:

        buy_score += 8

        buy_reasons.append(
            "1H bullish FVG"
        )

    if fvg["bearish"]:

        sell_score += 8

        sell_reasons.append(
            "1H bearish FVG"
        )

    # --------------------------------------------------------
    # ORDER BLOCK
    # --------------------------------------------------------

    if order_block["bullish"]:

        buy_score += 8

        buy_reasons.append(
            "1H bullish order block"
        )

    if order_block["bearish"]:

        sell_score += 8

        sell_reasons.append(
            "1H bearish order block"
        )

    # --------------------------------------------------------
    # RSI
    #
    # Flexible / supportive only.
    # --------------------------------------------------------

    if 45 <= rsi <= 68:

        if ema == "BULLISH":

            buy_score += 5

            buy_reasons.append(
                "1H RSI supports BUY"
            )

        elif ema == "BEARISH":

            sell_score += 5

            sell_reasons.append(
                "1H RSI supports SELL"
            )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    if sr["near_support"]:

        buy_score += 7

        buy_reasons.append(
            "1H near support"
        )

    if sr["near_resistance"]:

        sell_score += 7

        sell_reasons.append(
            "1H near resistance"
        )

    # ========================================================
    # DECIDE DIRECTION
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
    #
    # ONLY AFTER 1H DIRECTION IS STRONG
    # ========================================================

    entry_check = {
        "confirmed": False,
        "price": None,
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

        if entry_check["confirmed"]:

            final_score = max(
                buy_score,
                sell_score
            ) + 5

        else:

            final_score = max(
                buy_score,
                sell_score
            )

    else:

        final_score = max(
            buy_score,
            sell_score
        )

    # ========================================================
    # FINAL TRADE
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

        entry = entry_check["price"]

        entry, sl, tp = calculate_trade_levels(
            direction,
            entry
        )

        trade = {
            "trade_id": (
                f"{symbol}_"
                f"{direction}_"
                f"{int(time.time())}"
            ),

            "symbol": symbol,

            "direction": direction,

            "score": int(
                final_score
            ),

            "buy_score": int(
                buy_score
            ),

            "sell_score": int(
                sell_score
            ),

            "entry": entry,

            "stop_loss": sl,

            "take_profit": tp,

            "risk_reward": "1:2",

            "created_at": utc_timestamp(),

            "created_time_pk": pakistan_time(),

            "status": "OPEN",

            "entry_reason": entry_check[
                "reason"
            ],

            "buyer_pressure": pressure[
                "buyer"
            ],

            "seller_pressure": pressure[
                "seller"
            ],

            "dominant_pressure": pressure[
                "dominant"
            ],

            "volume_ratio": volume[
                "ratio"
            ],

            "volume_strong": volume[
                "strong"
            ],

            "volume_direction": volume[
                "direction"
            ],

            "h1_ema": ema,

            "h1_rsi": rsi,

            "h1_bullish_structure": structure[
                "bullish"
            ],

            "h1_bearish_structure": structure[
                "bearish"
            ],

            "h1_bullish_fvg": fvg[
                "bullish"
            ],

            "h1_bearish_fvg": fvg[
                "bearish"
            ],

            "h1_bullish_ob": order_block[
                "bullish"
            ],

            "h1_bearish_ob": order_block[
                "bearish"
            ],

            "reasons": (
                buy_reasons
                if direction == "BUY"
                else sell_reasons
            )
        }

    return {
        "symbol": symbol,

        "direction": direction,

        "score": int(
            final_score
        ),

        "buy_score": int(
            buy_score
        ),

        "sell_score": int(
            sell_score
        ),

        "trade": trade,

        "buyer_pressure": pressure[
            "buyer"
        ],

        "seller_pressure": pressure[
            "seller"
        ],

        "dominant_pressure": pressure[
            "dominant"
        ],

        "volume_ratio": volume[
            "ratio"
        ],

        "volume_strong": volume[
            "strong"
        ],

        "volume_direction": volume[
            "direction"
        ],

        "h1_ema": ema,

        "h1_rsi": rsi,

        "buy_reasons": buy_reasons,

        "sell_reasons": sell_reasons
    }


# ============================================================
# DUPLICATE / ACTIVE TRADE PROTECTION
# ============================================================

def should_send_trade(
    symbol,
    direction
):

    tracker = SIGNAL_TRACKER[
        symbol
    ]

    # --------------------------------------------
    # NEVER create another trade while one is open
    # --------------------------------------------

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
# REGISTER SIGNAL
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
# GMAIL TRADE ALERT
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

    if direction == "BUY":

        icon = "🟢"

        sl_text = "-1.00%"
        tp_text = "+2.00%"

        pressure_name = (
            "Buyer Pressure"
        )

        pressure_value = (
            trade["buyer_pressure"]
        )

    else:

        icon = "🔴"

        sl_text = "+1.00%"
        tp_text = "-2.00%"

        pressure_name = (
            "Seller Pressure"
        )

        pressure_value = (
            trade["seller_pressure"]
        )

    subject = (
        f"{icon} MARKET BRAIN "
        f"STRONG {direction} — "
        f"{symbol}"
    )

    body = f"""
==================================================
🧠 MARKET BRAIN — STRONG {direction}
==================================================

COIN
{symbol}

DIRECTION
{icon} {direction}

SIGNAL SCORE
{trade["score"]}/100

==================================================
TRADE SETUP
==================================================

ENTRY PRICE
{trade["entry"]:.10f}

STOP LOSS
{trade["stop_loss"]:.10f}
{sl_text}

TARGET PRICE
{trade["take_profit"]:.10f}
{tp_text}

RISK / REWARD
1 : 2

==================================================
MARKET PRESSURE
==================================================

{pressure_name}
{pressure_value:.1f}%

Buyer Pressure
{trade["buyer_pressure"]:.1f}%

Seller Pressure
{trade["seller_pressure"]:.1f}%

Dominant Pressure
{trade["dominant_pressure"]}

==================================================
VOLUME
==================================================

Volume Ratio
{trade["volume_ratio"]:.2f}x Average

Volume Status
{"STRONG" if trade["volume_strong"] else "NORMAL"}

Volume Direction
{trade["volume_direction"]}

==================================================
1H MARKET ANALYSIS
==================================================

1H EMA
{trade["h1_ema"]}

1H RSI
{trade["h1_rsi"]:.2f}

1H Bullish Structure
{trade["h1_bullish_structure"]}

1H Bearish Structure
{trade["h1_bearish_structure"]}

1H Bullish FVG
{trade["h1_bullish_fvg"]}

1H Bearish FVG
{trade["h1_bearish_fvg"]}

1H Bullish Order Block
{trade["h1_bullish_ob"]}

1H Bearish Order Block
{trade["h1_bearish_ob"]}

==================================================
5M ENTRY
==================================================

5M Entry Confirmation
{trade["entry_reason"]}

IMPORTANT:
5M is used ONLY for entry confirmation.

No 5M FVG
No 5M Order Block
No 5M RSI
No 5M EMA

==================================================
SIGNAL REASONS
==================================================
"""

    for reason in trade[
        "reasons"
    ]:

        body += (
            f"\n• {reason}"
        )

    body += f"""

==================================================
TIME
==================================================

Trade Created
{trade["created_time_pk"]}

Timezone
Pakistan Time (UTC+5)

Trade ID
{trade["trade_id"]}

==================================================

SL = 1%
TP = 2%
Risk / Reward = 1:2

This is a market-analysis signal.
"""

    return subject, body


# ============================================================
# TRADE CLOSE EMAIL
# ============================================================

def create_close_email(
    trade,
    result
):

    if result == "WIN":

        icon = "🟢"

    else:

        icon = "🔴"

    subject = (
        f"{icon} MARKET BRAIN "
        f"TRADE {result} — "
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
{trade["entry"]:.10f}

Stop Loss:
{trade["stop_loss"]:.10f}

Target:
{trade["take_profit"]:.10f}

Closed Price:
{trade["closed_price"]:.10f}

Created:
{trade["created_time_pk"]}

Closed:
{trade["closed_time_pk"]}

Buyer Pressure:
{trade["buyer_pressure"]:.1f}%

Seller Pressure:
{trade["seller_pressure"]:.1f}%

Volume:
{trade["volume_ratio"]:.2f}x

Score:
{trade["score"]}/100

==================================================

Result recorded in Market Brain learning memory.
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

    if len(df) < 2:
        return None

    # Use the latest available price
    current = float(
        df["close"].iloc[-1]
    )

    direction = trade[
        "direction"
    ]

    result = None

    # ========================================================
    # BUY
    # ========================================================

    if direction == "BUY":

        if current >= trade[
            "take_profit"
        ]:

            result = "WIN"

        elif current <= trade[
            "stop_loss"
        ]:

            result = "LOSS"

    # ========================================================
    # SELL
    # ========================================================

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

        trade[
            "status"
        ] = result

        trade[
            "closed_price"
        ] = current

        trade[
            "closed_time_pk"
        ] = pakistan_time()

        trade[
            "closed_at"
        ] = utc_timestamp()

        if result == "WIN":

            trade[
                "realized_rr"
            ] = 2.0

        else:

            trade[
                "realized_rr"
            ] = -1.0

        return result

    return None


# ============================================================
# LEARNING
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

        patterns[
            pattern_key
        ] = {
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

    update_trade_log(
        trade
    )

    save_memory()


# ============================================================
# CHECK ALL ACTIVE TRADES
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
                    f"{symbol} "
                    f"{trade['direction']} "
                    f"closed: {result}"
                )

                learn_from_trade(
                    trade,
                    result
                )

                # Send close notification
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
                f"{symbol} trade check error: {e}"
            )

    if changed:

        save_active_trades()


# ============================================================
# SCAN SYMBOL
# ============================================================

def scan_symbol(
    symbol
):

    logger.info(
        f"Scanning {symbol}"
    )

    # ========================================================
    # ONLY TWO TIMEFRAMES
    # ========================================================

    h1 = fetch_candles(
        symbol,
        "1h",
        80
    )

    m5 = fetch_candles(
        symbol,
        "5m",
        30
    )

    if h1 is None or m5 is None:

        logger.warning(
            f"{symbol}: incomplete data"
        )

        return

    analysis = analyze_market(
        symbol,
        h1,
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
        ].append(
            symbol
        )

    logger.info(
        f"{symbol} | "
        f"BUY={analysis['buy_score']} | "
        f"SELL={analysis['sell_score']} | "
        f"FINAL={analysis['direction']} | "
        f"SCORE={analysis['score']}"
    )

    trade = analysis[
        "trade"
    ]

    if trade is None:
        return

    # ========================================================
    # ACTIVE TRADE PROTECTION
    # ========================================================

    if not should_send_trade(
        symbol,
        trade["direction"]
    ):

        logger.info(
            f"{symbol}: "
            f"trade blocked by protection."
        )

        return

    # ========================================================
    # SEND EMAIL FIRST
    # ========================================================

    subject, body = (
        create_trade_email(
            trade
        )
    )

    sent = send_email(
        subject,
        body
    )

    # ========================================================
    # ONLY REGISTER TRADE IF EMAIL SENT
    # ========================================================

    if sent:

        register_trade(
            symbol,
            trade
        )

        logger.info(
            f"TRADE CREATED: "
            f"{symbol} "
            f"{trade['direction']} "
            f"Score={trade['score']}"
        )

    else:

        logger.warning(
            f"{symbol}: "
            f"trade NOT registered because "
            f"Gmail failed."
        )


# ============================================================
# 24-HOUR REPORT
# ============================================================

def create_24_hour_report():

    logs = load_trade_logs()

    now = time.time()

    start_time = (
        now -
        REPORT_INTERVAL_SECONDS
    )

    recent = []

    for trade in logs:

        created_at = trade.get(
            "created_at",
            0
        )

        try:
            created_at = float(
                created_at
            )
        except Exception:
            continue

        if created_at >= start_time:

            recent.append(
                trade
            )

    total = len(recent)

    wins = sum(
        1
        for trade in recent
        if trade.get("status") == "WIN"
    )

    losses = sum(
        1
        for trade in recent
        if trade.get("status") == "LOSS"
    )

    open_trades = sum(
        1
        for trade in recent
        if trade.get("status") == "OPEN"
    )

    buy_trades = sum(
        1
        for trade in recent
        if trade.get("direction") == "BUY"
    )

    sell_trades = sum(
        1
        for trade in recent
        if trade.get("direction") == "SELL"
    )

    closed = wins + losses

    if closed > 0:

        win_rate = (
            wins /
            closed
        ) * 100

    else:

        win_rate = 0.0

    total_rr = (
        wins * 2.0
        -
        losses * 1.0
    )

    return {
        "recent": recent,
        "total": total,
        "wins": wins,
        "losses": losses,
        "open": open_trades,
        "buy": buy_trades,
        "sell": sell_trades,
        "closed": closed,
        "win_rate": win_rate,
        "net_rr": total_rr
    }


# ============================================================
# SEND 24-HOUR REPORT
# ============================================================

def send_24_hour_report():

    report = create_24_hour_report()

    subject = (
        "🧠 MARKET BRAIN — "
        "24 HOUR PERFORMANCE"
    )

    body = f"""
==================================================
🧠 MARKET BRAIN
24 HOUR PERFORMANCE REPORT
==================================================

Pakistan Time:
{pakistan_time()}

==================================================
SUMMARY
==================================================

Total Trades:
{report["total"]}

BUY Trades:
{report["buy"]}

SELL Trades:
{report["sell"]}

Wins:
🟢 {report["wins"]}

Losses:
🔴 {report["losses"]}

Open:
🟡 {report["open"]}

Closed:
{report["closed"]}

Win Rate:
{report["win_rate"]:.1f}%

Net R:R:
{report["net_rr"]:+.1f}R

==================================================
TRADE HISTORY
==================================================
"""

    if not report["recent"]:

        body += (
            "\nNo trades in the last 24 hours.\n"
        )

    else:

        for number, trade in enumerate(
            report["recent"],
            start=1
        ):

            result = trade.get(
                "status",
                "UNKNOWN"
            )

            if result == "WIN":
                result_icon = "🟢"

            elif result == "LOSS":
                result_icon = "🔴"

            else:
                result_icon = "🟡"

            body += f"""

--------------------------------------------------
TRADE #{number}
--------------------------------------------------

Coin:
{trade.get("symbol")}

Direction:
{trade.get("direction")}

Score:
{trade.get("score")}/100

Entry:
{trade.get("entry")}

Stop Loss:
{trade.get("stop_loss")}

Target:
{trade.get("take_profit")}

Result:
{result_icon} {result}

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

"""

    body += """

==================================================

Report period:
Rolling previous 24 hours

Timezone:
Pakistan Time (UTC+5)

Market Brain report.
"""

    sent = send_email(
        subject,
        body
    )

    if sent:

        save_json_file(
            LAST_REPORT_FILE,
            {
                "sent_at": utc_timestamp(),
                "sent_time_pk": pakistan_time()
            }
        )

    return sent


# ============================================================
# SHOULD SEND 24H REPORT
# ============================================================

def should_send_24h_report():

    data = load_json_file(
        LAST_REPORT_FILE,
        {}
    )

    if not isinstance(data, dict):
        return True

    last_sent = data.get(
        "sent_at",
        0
    )

    try:
        last_sent = float(
            last_sent
        )
    except Exception:
        return True

    return (
        time.time() -
        last_sent
        >= REPORT_INTERVAL_SECONDS
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    logger.info(
        "================================================"
    )

    logger.info(
        "🧠 MARKET BRAIN v10 STARTED"
    )

    logger.info(
        f"Assets: {len(ASSETS)}"
    )

    logger.info(
        "Main timeframe: 1H"
    )

    logger.info(
        "Entry timeframe: 5M"
    )

    logger.info(
        "TP: 2%"
    )

    logger.info(
        "SL: 1%"
    )

    logger.info(
        "Risk/Reward: 1:2"
    )

    logger.info(
        "Timezone: Pakistan UTC+5"
    )

    logger.info(
        "================================================"
    )

    # Restore trades after restart
    restore_active_trades()

    while True:

        cycle_start = time.time()

        try:

            # =================================================
            # CHECK EXISTING TRADES FIRST
            # =================================================

            check_all_active_trades()

            # =================================================
            # SCAN ALL 20 COINS
            # =================================================

            for symbol in ASSETS:

                try:

                    scan_symbol(
                        symbol
                    )

                except Exception as e:

                    logger.error(
                        f"{symbol} "
                        f"scan error: {e}"
                    )

                # Small delay to avoid hammering API
                time.sleep(
                    0.5
                )

            # =================================================
            # 24 HOUR REPORT
            # =================================================

            if should_send_24h_report():

                try:

                    send_24_hour_report()

                except Exception as e:

                    logger.error(
                        f"24h report error: {e}"
                    )

            save_memory()

        except KeyboardInterrupt:

            logger.info(
                "Market Brain stopped manually."
            )

            save_active_trades()
            save_memory()

            break

        except Exception as e:

            logger.error(
                f"Main loop error: {e}"
            )

        # =====================================================
        # MAINTAIN 2-MINUTE SCAN CYCLE
        # =====================================================

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

    try:

        main()

    except Exception as e:

        logger.critical(
            f"Fatal error: {e}"
    )
    
