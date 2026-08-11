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
# 🧠 MARKET BRAIN — COMPLETE TRADING ENGINE
# ============================================================
#
# Features:
# 1D / 4H / 1H / 30M / 15M / 5M / 2M
# Market Structure
# HH / HL / LH / LL
# BOS / CHoCH
# Liquidity sweep
# Order Block
# Fair Value Gap
# Support / Resistance
# Buyer / Seller pressure
# Volume
# Candle structure
# Adaptive learning memory
# Learning Progress %
# Trade tracking
# Correct BUY / SELL SL & TP
# Gmail trade alerts
# Gmail hourly Brain report
# JSON-safe NumPy/Pandas memory
# 20 Binance assets
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
# TIMEZONE
# ============================================================

PK_TZ = timezone(timedelta(hours=5))


def get_pakistan_time():
    return datetime.now(PK_TZ).strftime(
        "%Y-%m-%d %I:%M:%S %p"
    )


# ============================================================
# GMAIL
# ============================================================

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = os.environ.get("GMAIL_USER")
SENDER_PASSWORD = os.environ.get("GMAIL_PASS")

RECEIVER_EMAIL = os.environ.get(
    "GMAIL_RECEIVER",
    SENDER_EMAIL
)


def send_trade_email(subject, message_body):

    if not SENDER_EMAIL or not SENDER_PASSWORD:

        logging.error(
            "Gmail credentials missing. "
            "Set GMAIL_USER and GMAIL_PASS."
        )

        return False

    try:

        msg = EmailMessage()

        msg.set_content(message_body)

        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        with smtplib.SMTP(
            SMTP_SERVER,
            SMTP_PORT,
            timeout=30
        ) as server:

            server.ehlo()
            server.starttls()
            server.ehlo()

            server.login(
                SENDER_EMAIL,
                SENDER_PASSWORD
            )

            server.send_message(msg)

        logging.info(
            "Email Alert Sent Successfully!"
        )

        return True

    except Exception as e:

        logging.error(
            f"Gmail SMTP Error: {e}"
        )

        return False


# ============================================================
# CONFIG
# ============================================================

SCAN_INTERVAL_SECONDS = 120

HOURLY_REPORT_SECONDS = 3600

STOP_LOSS_PERCENT = 1.0

TARGET_PERCENT = 2.0

MIN_SIGNAL_SCORE = 65

MEMORY_FILE = "market_brain_memory.json"

REPORT_STATE_FILE = "brain_report_state.json"

TRADE_LOG_FILE = "ai_trade_learning_log.json"


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
# TRADE TRACKER
# ============================================================

SIGNAL_TRACKER = {
    asset: {
        "last_signal": None,
        "active_trade": None,
        "last_signal_time": 0
    }
    for asset in ASSETS
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
# JSON SAFE CONVERTER
# ============================================================

def make_json_safe(obj):

    if obj is None:
        return None

    if isinstance(obj, bool):
        return bool(obj)

    if isinstance(obj, str):
        return obj

    if isinstance(obj, int):
        return int(obj)

    if isinstance(obj, float):

        if math.isnan(obj) or math.isinf(obj):
            return None

        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):

        value = float(obj)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, dict):

        return {
            str(k): make_json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):

        return [
            make_json_safe(v)
            for v in obj
        ]

    return str(obj)


# ============================================================
# BRAIN MEMORY
# ============================================================

def default_memory():

    return {
        "created_at": time.time(),

        "observations": 0,

        "wins": 0,

        "losses": 0,

        "closed_trades": 0,

        "symbols_seen": [],

        "patterns": {},

        "structure_stats": {},

        "signal_stats": {},

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
            f"Memory load failed: {e}"
        )

        return default_memory()


BRAIN_MEMORY = load_memory()


def save_memory():

    global BRAIN_MEMORY

    BRAIN_MEMORY = make_json_safe(
        BRAIN_MEMORY
    )

    BRAIN_MEMORY[
        "last_learning_update"
    ] = time.time()

    temp_file = MEMORY_FILE + ".tmp"

    try:

        with open(
            temp_file,
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
            temp_file,
            MEMORY_FILE
        )

    except Exception as e:

        logging.error(
            f"Memory save error: {e}"
        )


# ============================================================
# TRADE LOG
# ============================================================

def save_trade_result(data):

    logs = []

    if os.path.exists(TRADE_LOG_FILE):

        try:

            with open(
                TRADE_LOG_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                logs = json.load(f)

            if not isinstance(logs, list):
                logs = []

        except Exception:

            logs = []

    logs.append(
        make_json_safe(data)
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
            f"Trade log save error: {e}"
        )


# ============================================================
# BINANCE DATA
# ============================================================

BINANCE_ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines"
]


def fetch_binance_candles(
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

            if not isinstance(raw, list):
                continue

            if len(raw) < 10:
                continue

            df = pd.DataFrame(
                raw,
                columns=[
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore"
                ]
            )

            numeric_cols = [
                "open",
                "high",
                "low",
                "close",
                "vol",
                "quote_volume",
                "taker_buy_base",
                "taker_buy_quote"
            ]

            for col in numeric_cols:

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
                    "vol"
                ]
            )

            return df.reset_index(
                drop=True
            )

        except Exception as e:

            logging.debug(
                f"{symbol} {interval} endpoint error: {e}"
            )

            continue

    logging.warning(
        f"Binance data unavailable: "
        f"{symbol} {interval}"
    )

    return None


# ============================================================
# 2-MINUTE AGGREGATION
# ============================================================

def make_2m_from_1m(df_1m):

    if df_1m is None or len(df_1m) < 4:
        return None

    df = df_1m.copy()

    df["datetime"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True
    )

    df = df.set_index("datetime")

    aggregated = df.resample(
        "2min"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum"
    })

    aggregated = aggregated.dropna()

    aggregated["time"] = (
        aggregated.index.astype(
            "int64"
        ) // 10**6
    )

    aggregated = aggregated.reset_index(
        drop=True
    )

    return aggregated[
        [
            "time",
            "open",
            "high",
            "low",
            "close",
            "vol"
        ]
    ]


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    df,
    periods=14
):

    if df is None or len(df) < periods + 2:
        return 50.0

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        com=periods - 1,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        com=periods - 1,
        adjust=False
    ).mean()

    if avg_loss.iloc[-1] == 0:
        return 100.0

    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]

    rsi = 100 - (
        100 / (1 + rs)
    )

    return float(
        max(0, min(100, rsi))
    )


# ============================================================
# EMA TREND
# ============================================================

def ema_trend(
    df,
    fast=20,
    slow=50
):

    if df is None or len(df) < slow:
        return "NEUTRAL"

    ema_fast = df["close"].ewm(
        span=fast,
        adjust=False
    ).mean().iloc[-2]

    ema_slow = df["close"].ewm(
        span=slow,
        adjust=False
    ).mean().iloc[-2]

    if ema_fast > ema_slow:
        return "BULLISH"

    if ema_fast < ema_slow:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(df):

    if df is None or len(df) < 20:

        return {
            "confirmed": False,
            "ratio": 0.0
        }

    # Ignore currently forming candle
    previous = df.iloc[:-1]

    if len(previous) < 20:

        return {
            "confirmed": False,
            "ratio": 0.0
        }

    avg_volume = previous[
        "vol"
    ].tail(20).mean()

    current_volume = previous[
        "vol"
    ].iloc[-1]

    if avg_volume <= 0:

        return {
            "confirmed": False,
            "ratio": 0.0
        }

    ratio = current_volume / avg_volume

    return {
        "confirmed": bool(
            ratio >= 1.15
        ),
        "ratio": round(
            float(ratio),
            2
        )
    }


# ============================================================
# BUYER / SELLER PRESSURE
# ============================================================

def pressure_analysis(
    df,
    lookback=20
):

    if df is None or len(df) < lookback + 2:

        return {
            "buyer": 50.0,
            "seller": 50.0,
            "dominant": "NEUTRAL"
        }

    data = df.iloc[
        -(lookback + 1):-1
    ]

    buyer = 0.0
    seller = 0.0

    for _, candle in data.iterrows():

        body = abs(
            candle["close"] -
            candle["open"]
        )

        weighted_body = (
            body *
            candle["vol"]
        )

        if candle["close"] > candle["open"]:

            buyer += weighted_body

        elif candle["close"] < candle["open"]:

            seller += weighted_body

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
            float(buyer_pct),
            1
        ),
        "seller": round(
            float(seller_pct),
            1
        ),
        "dominant": dominant
    }


# ============================================================
# SWING POINTS
# ============================================================

def find_swings(
    df,
    left=2,
    right=2
):

    if df is None or len(df) < 20:

        return [], []

    highs = []
    lows = []

    h = df["high"].values
    l = df["low"].values

    for i in range(
        left,
        len(df) - right
    ):

        high_window = h[
            i-left:i+right+1
        ]

        low_window = l[
            i-left:i+right+1
        ]

        if h[i] == max(
            high_window
        ):

            highs.append(
                (i, float(h[i]))
            )

        if l[i] == min(
            low_window
        ):

            lows.append(
                (i, float(l[i]))
            )

    return highs, lows


# ============================================================
# MARKET STRUCTURE
# ============================================================

def market_structure(df):

    if df is None or len(df) < 30:

        return {
            "trend": "UNKNOWN",
            "structure": "UNKNOWN",
            "bos": False,
            "bos_direction": "NONE",
            "choch": False,
            "liquidity_sweep": "NONE",
            "swing_high": None,
            "swing_low": None
        }

    highs, lows = find_swings(df)

    if len(highs) < 3 or len(lows) < 3:

        trend = ema_trend(
            df,
            20,
            50
        )

        return {
            "trend": trend,
            "structure": "MIXED",
            "bos": False,
            "bos_direction": "NONE",
            "choch": False,
            "liquidity_sweep": "NONE",
            "swing_high": float(
                df["high"].iloc[:-1].tail(20).max()
            ),
            "swing_low": float(
                df["low"].iloc[:-1].tail(20).min()
            )
        }

    last_highs = highs[-3:]
    last_lows = lows[-3:]

    h1 = last_highs[-2][1]
    h2 = last_highs[-1][1]

    l1 = last_lows[-2][1]
    l2 = last_lows[-1][1]

    if h2 > h1 and l2 > l1:

        structure = "HH_HL"
        trend = "BULLISH"

    elif h2 < h1 and l2 < l1:

        structure = "LH_LL"
        trend = "BEARISH"

    else:

        structure = "MIXED"
        trend = ema_trend(
            df,
            20,
            50
        )

    # Last closed candle
    last = df.iloc[-2]

    previous_high = max(
        x[1]
        for x in highs[:-1]
    )

    previous_low = min(
        x[1]
        for x in lows[:-1]
    )

    bos_up = (
        last["close"] >
        previous_high
    )

    bos_down = (
        last["close"] <
        previous_low
    )

    bos = bos_up or bos_down

    if bos_up:
        bos_direction = "UP"

    elif bos_down:
        bos_direction = "DOWN"

    else:
        bos_direction = "NONE"

    # CHoCH approximation:
    # Break opposite to established structure.
    choch = False

    if trend == "BULLISH" and bos_down:
        choch = True

    elif trend == "BEARISH" and bos_up:
        choch = True

    # Liquidity sweep
    prior_high = max(
        df["high"].iloc[-12:-2]
    )

    prior_low = min(
        df["low"].iloc[-12:-2]
    )

    liquidity_sweep = "NONE"

    if (
        last["high"] > prior_high
        and last["close"] < prior_high
    ):
        liquidity_sweep = "SELL_SIDE"

    elif (
        last["low"] < prior_low
        and last["close"] > prior_low
    ):
        liquidity_sweep = "BUY_SIDE"

    return {
        "trend": trend,
        "structure": structure,
        "bos": bool(bos),
        "bos_direction": bos_direction,
        "choch": bool(choch),
        "liquidity_sweep": liquidity_sweep,
        "swing_high": float(h2),
        "swing_low": float(l2)
    }


# ============================================================
# FAIR VALUE GAP
# ============================================================

def detect_fvg(df):

    if df is None or len(df) < 10:

        return {
            "bullish": False,
            "bearish": False,
            "low": None,
            "high": None
        }

    # Use three CLOSED candles:
    c1 = df.iloc[-4]
    c2 = df.iloc[-3]
    c3 = df.iloc[-2]

    bullish = (
        c3["low"] >
        c1["high"]
    )

    bearish = (
        c3["high"] <
        c1["low"]
    )

    if bullish:

        return {
            "bullish": True,
            "bearish": False,
            "low": float(c1["high"]),
            "high": float(c3["low"])
        }

    if bearish:

        return {
            "bullish": False,
            "bearish": True,
            "low": float(c3["high"]),
            "high": float(c1["low"])
        }

    return {
        "bullish": False,
        "bearish":
