# ============================================================
# MARKET BRAIN AI — ADAPTIVE 1D / 4H / 1H EDITION
# ============================================================
#
# 1D = Major Market Bias
# 4H = Structure / Liquidity / SMC / Pressure
# 1H = Setup + Entry + Dynamic SL + Dynamic TP
#
# NO 5M
# NO 12-HOUR REPORT
#
# ALERTS:
#   1) NEW TRADE
#   2) TRADE CLOSED -> WIN / LOSS
#
# FEATURES:
# - HH / HL / LH / LL
# - BOS / CHoCH
# - Liquidity Sweeps
# - Equal High / Equal Low
# - Double Top / Double Bottom
# - Head & Shoulders
# - Triple Top / Bottom
# - Triangles
# - Wedges
# - Flags
# - Range Breakout
# - FVG
# - Order Block
# - Breaker
# - Premium / Discount
# - Buyer / Seller Pressure
# - Volume
# - EMA
# - RSI
# - ATR
# - Support / Resistance
# - Candlestick confirmation
#
# ADAPTIVE LEARNING:
# - Learns each indicator/setup separately
# - Learns BUY vs SELL
# - Learns each symbol
# - Learns setup combinations
# - Rewards indicators involved in WINs
# - Penalizes indicators involved in LOSSes
# - Uses sample-size protection
# - Stores MAE / MFE
# - Stores failure information
#
# TRADE PROTECTION:
# - Persistent trade memory
# - Duplicate trade protection
# - Duplicate email protection
# - Candle-based signal identity
# - Open-trade restoration
#
# IMPORTANT:
# This is a statistical adaptive engine, NOT a neural network.
# No strategy can guarantee 70%-80% win rate.
# ============================================================

import os
import time
import json
import uuid
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

# Binance normal API can return HTTP 451 in some regions.
# Public market-data endpoint is used by default.
BASE_URL = os.getenv(
    "BINANCE_BASE_URL",
    "https://data-api.binance.vision"
)

PKT = ZoneInfo("Asia/Karachi")

SYMBOLS = [
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
    "UNIUSDT",
    "ATOMUSDT",
]

TIMEFRAMES = {
    "1d": "1d",
    "4h": "4h",
    "1h": "1h",
}

CANDLE_LIMIT = 300

# Bot scans every minute.
# This does NOT mean 1-minute analysis.
# It only means the 1H trade is checked more frequently.
SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "60"))

# ============================================================
# ENTRY / SCORE SETTINGS
# ============================================================

# Lower than old 65 to avoid "no trades for a week".
MIN_SCORE = float(os.getenv("MIN_SCORE", "58"))

# Smaller direction gap = more opportunities.
DIRECTION_GAP = float(os.getenv("DIRECTION_GAP", "5"))

# Maximum new trades per scan.
MAX_NEW_TRADES_PER_SCAN = int(
    os.getenv("MAX_NEW_TRADES_PER_SCAN", "3")
)

# Maximum simultaneous positions.
MAX_OPEN_TRADES = int(
    os.getenv("MAX_OPEN_TRADES", "8")
)

# ============================================================
# RISK SETTINGS
# ============================================================

MIN_RR = float(os.getenv("MIN_RR", "1.8"))
MAX_RR = float(os.getenv("MAX_RR", "4.5"))

ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.20"))
ATR_TARGET_MULT = float(os.getenv("ATR_TARGET_MULT", "2.80"))

MIN_SL_PCT = float(os.getenv("MIN_SL_PCT", "0.35"))
MAX_SL_PCT = float(os.getenv("MAX_SL_PCT", "3.00"))

MIN_TARGET_PCT = float(os.getenv("MIN_TARGET_PCT", "0.70"))
MAX_TARGET_PCT = float(os.getenv("MAX_TARGET_PCT", "12.0"))

# ============================================================
# LEARNING SETTINGS
# ============================================================

LEARNING_WIN_REWARD = float(
    os.getenv("LEARNING_WIN_REWARD", "0.035")
)

LEARNING_LOSS_PENALTY = float(
    os.getenv("LEARNING_LOSS_PENALTY", "0.025")
)

MIN_LEARNING_SAMPLES = int(
    os.getenv("MIN_LEARNING_SAMPLES", "3")
)

MIN_INDICATOR_WEIGHT = float(
    os.getenv("MIN_INDICATOR_WEIGHT", "0.70")
)

MAX_INDICATOR_WEIGHT = float(
    os.getenv("MAX_INDICATOR_WEIGHT", "1.35")
)

# ============================================================
# PERSISTENT FILES
# ============================================================

DATA_DIR = Path(
    os.getenv("DATA_DIR", "market_brain_data")
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

LEARNING_FILE = DATA_DIR / "ai_learning.json"
TRADE_MEMORY_FILE = DATA_DIR / "trade_memory.json"
TRADE_CSV_FILE = DATA_DIR / "trade_learning_log.csv"
STATE_FILE = DATA_DIR / "bot_state.json"

# ============================================================
# GMAIL
# ============================================================

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
GMAIL_RECEIVER = os.getenv(
    "GMAIL_RECEIVER",
    GMAIL_USER or ""
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("MARKET_BRAIN")


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "MARKET-BRAIN-AI/2.0",
    "Accept": "application/json",
})


# ============================================================
# UTILITIES
# ============================================================

def now_pkt():
    return datetime.now(PKT)


def iso_pkt(dt=None):
    return (dt or now_pkt()).isoformat()


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def safe_div(a, b, default=0.0):
    try:
        b = float(b)
        if b == 0:
            return default
        return float(a) / b
    except Exception:
        return default


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def pct(value):
    return f"{safe_float(value):.2f}%"


def load_json(path, default):
    try:
        if path.exists():
            return json.loads(
                path.read_text(encoding="utf-8")
            )
    except Exception as e:
        logger.warning(
            "Could not load %s: %s",
            path,
            e
        )

    return default


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    tmp.replace(path)


# ============================================================
# EMAIL
# ============================================================

def send_email(subject, body):

    if not GMAIL_USER:
        logger.error("GMAIL_USER missing")
        return False

    if not GMAIL_PASS:
        logger.error("GMAIL_PASS missing")
        return False

    if not GMAIL_RECEIVER:
        logger.error("GMAIL_RECEIVER missing")
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
            "Email sent: %s",
            subject
        )

        return True

    except Exception as e:

        logger.error(
            "Email error: %s",
            e
        )

        return False


# ============================================================
# BINANCE DATA
# ============================================================

def fetch_klines(
    symbol,
    interval,
    limit=CANDLE_LIMIT
):

    url = (
        f"{BASE_URL}/api/v3/klines"
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    for attempt in range(3):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=20
            )

            response.raise_for_status()

            raw = response.json()

            if not isinstance(raw, list):
                return pd.DataFrame()

            columns = [
                "open_time",
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
                "ignore",
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
            ]

            for col in numeric_columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            df["open_time"] = pd.to_datetime(
                df["open_time"],
                unit="ms",
                utc=True
            )

            df["close_time"] = pd.to_datetime(
                df["close_time"],
                unit="ms",
                utc=True
            )

            df = df.dropna().reset_index(
                drop=True
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Remove currently forming candle.
            # All signals use CLOSED candles only.
            # ------------------------------------------------

            now_utc = pd.Timestamp.now(
                tz="UTC"
            )

            if not df.empty:

                last_close = df[
                    "close_time"
                ].iloc[-1]

                if last_close > now_utc:

                    df = df.iloc[:-1].copy()

            return df.reset_index(
                drop=True
            )

        except Exception as e:

            logger.warning(
                "%s %s attempt %d: %s",
                symbol,
                interval,
                attempt + 1,
                e
            )

            time.sleep(
                1.5 * (attempt + 1)
            )

    return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def ema(series, length):

    return series.ewm(
        span=length,
        adjust=False
    ).mean()


def rsi(series, length=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    output = 100 - (
        100 / (1 + rs)
    )

    return output.fillna(50)


def atr(df, length=14):

    previous_close = df[
        "close"
    ].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"] -
                previous_close
            ).abs(),
            (
                df["low"] -
                previous_close
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()


def add_indicators(df):

    x = df.copy()

    x["ema20"] = ema(
        x["close"],
        20
    )

    x["ema50"] = ema(
        x["close"],
        50
    )

    x["ema200"] = ema(
        x["close"],
        200
    )

    x["rsi"] = rsi(
        x["close"],
        14
    )

    x["atr"] = atr(
        x,
        14
    )

    x["avg_volume"] = (
        x["volume"]
        .rolling(20)
        .mean()
    )

    x["volume_ratio"] = (
        x["volume"] /
        x["avg_volume"].replace(
            0,
            np.nan
        )
    )

    x["body"] = (
        x["close"] -
        x["open"]
    ).abs()

    x["range"] = (
        x["high"] -
        x["low"]
    ).replace(
        0,
        np.nan
    )

    x["body_ratio"] = (
        x["body"] /
        x["range"]
    )

    x["upper_wick"] = (
        x["high"] -
        x[["open", "close"]].max(
            axis=1
        )
    )

    x["lower_wick"] = (
        x[["open", "close"]].min(
            axis=1
        ) -
        x["low"]
    )

    return x


# ============================================================
# PRESSURE
# ============================================================

def pressure(
    df,
    length=12
):

    d = df.tail(length)

    candle_range = (
        d["high"] -
        d["low"]
    ).replace(
        0,
        np.nan
    )

    strength = (
        (
            d["close"] -
            d["open"]
        ).abs() /
        candle_range
    )

    weighted = (
        strength.fillna(0) *
        d["volume"]
    )

    buy = weighted[
        d["close"] >
        d["open"]
    ].sum()

    sell = weighted[
        d["close"] <
        d["open"]
    ].sum()

    total = buy + sell

    if total <= 0:

        return 50.0, 50.0

    return (
        buy / total * 100,
        sell / total * 100
    )


# ============================================================
# SWINGS
# ============================================================

def pivots(
    df,
    left=3,
    right=3
):

    highs = []
    lows = []

    high_values = df[
        "high"
    ].values

    low_values = df[
        "low"
    ].values

    for i in range(
        left,
        len(df) - right
    ):

        if high_values[i] == max(
            high_values[
                i-left:i+right+1
            ]
        ):

            highs.append(
                (
                    i,
                    float(high_values[i])
                )
            )

        if low_values[i] == min(
            low_values[
                i-left:i+right+1
            ]
        ):

            lows.append(
                (
                    i,
                    float(low_values[i])
                )
            )

    return highs, lows


def structure_info(df):

    highs, lows = pivots(
        df,
        3,
        3
    )

    last_high = None
    previous_high = None

    last_low = None
    previous_low = None

    if len(highs) >= 2:

        previous_high = highs[-2][1]
        last_high = highs[-1][1]

    if len(lows) >= 2:

        previous_low = lows[-2][1]
        last_low = lows[-1][1]

    hh = (
        last_high is not None and
        previous_high is not None and
        last_high > previous_high
    )

    lh = (
        last_high is not None and
        previous_high is not None and
        last_high < previous_high
    )

    hl = (
        last_low is not None and
        previous_low is not None and
        last_low > previous_low
    )

    ll = (
        last_low is not None and
        previous_low is not None and
        last_low < previous_low
    )

    close = float(
        df["close"].iloc[-1]
    )

    previous_close = float(
        df["close"].iloc[-2]
    )

    bull_bos = (
        last_high is not None and
        close > last_high and
        previous_close <= last_high
    )

    bear_bos = (
        last_low is not None and
        close < last_low and
        previous_close >= last_low
    )

    # --------------------------------------------------------
    # CHoCH approximation
    # --------------------------------------------------------

    previous_state = 0

    start = max(
        10,
        len(df) - 80
    )

    for i in range(
        start,
        len(df)
    ):

        sub = df.iloc[
            :i+1
        ]

        hs, ls = pivots(
            sub,
            3,
            3
        )

        if hs:

            if float(
                sub["close"].iloc[-1]
            ) > hs[-1][1]:

                previous_state = 1

        if ls:

            if float(
                sub["close"].iloc[-1]
            ) < ls[-1][1]:

                previous_state = -1

    bull_choch = (
        bull_bos and
        previous_state == -1
    )

    bear_choch = (
        bear_bos and
        previous_state == 1
    )

    return {
        "last_high": last_high,
        "previous_high": previous_high,
        "last_low": last_low,
        "previous_low": previous_low,

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
# SMC
# ============================================================

def detect_smc(
    df,
    structure
):

    high = df["high"]
    low = df["low"]
    close = df["close"]
    op = df["open"]

    bull_fvg = (
        float(low.iloc[-1]) >
        float(high.iloc[-3])
    )

    bear_fvg = (
        float(high.iloc[-1]) <
        float(low.iloc[-3])
    )

    bull_ob = (
        float(close.iloc[-1]) >
        float(op.iloc[-1])
        and
        float(close.iloc[-1]) >
        float(high.iloc[-2])
        and
        float(close.iloc[-2]) <
        float(op.iloc[-2])
    )

    bear_ob = (
        float(close.iloc[-1]) <
        float(op.iloc[-1])
        and
        float(close.iloc[-1]) <
        float(low.iloc[-2])
        and
        float(close.iloc[-2]) >
        float(op.iloc[-2])
    )

    bull_breaker = (
        float(close.iloc[-2]) <
        float(op.iloc[-2])
        and
        float(close.iloc[-1]) >
        float(high.iloc[-2])
    )

    bear_breaker = (
        float(close.iloc[-2]) >
        float(op.iloc[-2])
        and
        float(close.iloc[-1]) <
        float(low.iloc[-2])
    )

    last_low = structure[
        "last_low"
    ]

    last_high = structure[
        "last_high"
    ]

    bull_sweep = (
        last_low is not None
        and
        float(low.iloc[-1]) <
        last_low
        and
        float(close.iloc[-1]) >
        last_low
    )

    bear_sweep = (
        last_high is not None
        and
        float(high.iloc[-1]) >
        last_high
        and
        float(close.iloc[-1]) <
        last_high
    )

    range_high = float(
        high.tail(50).max()
    )

    range_low = float(
        low.tail(50).min()
    )

    equilibrium = (
        range_high +
        range_low
    ) / 2

    price = float(
        close.iloc[-1]
    )

    return {

        "bull_fvg": bull_fvg,
        "bear_fvg": bear_fvg,

        "bull_ob": bull_ob,
        "bear_ob": bear_ob,

        "bull_breaker":
            bull_breaker,

        "bear_breaker":
            bear_breaker,

        "bull_sweep":
            bull_sweep,

        "bear_sweep":
            bear_sweep,

        "range_high":
            range_high,

        "range_low":
            range_low,

        "equilibrium":
            equilibrium,

        "discount":
            price < equilibrium,

        "premium":
            price > equilibrium,
    }


# ============================================================
# CANDLE PATTERNS
# ============================================================

def candle_patterns(df):

    c = df.iloc[-1]
    p = df.iloc[-2]

    body = abs(
        float(
            c["close"] -
            c["open"]
        )
    )

    candle_range = max(
        float(
            c["high"] -
            c["low"]
        ),
        1e-12
    )

    upper = float(
        c["high"] -
        max(
            c["open"],
            c["close"]
        )
    )

    lower = float(
        min(
            c["open"],
            c["close"]
        ) -
        c["low"]
    )

    bull_engulf = (
        c["close"] >
        c["open"]
        and
        p["close"] <
        p["open"]
        and
        c["close"] >=
        p["open"]
        and
        c["open"] <=
        p["close"]
    )

    bear_engulf = (
        c["close"] <
        c["open"]
        and
        p["close"] >
        p["open"]
        and
        c["close"] <=
        p["open"]
        and
        c["open"] >=
        p["close"]
    )

    hammer = (
        lower > body * 2
        and
        upper <= max(
            body,
            1e-12
        )
    )

    shooting_star = (
        upper > body * 2
        and
        lower <= max(
            body,
            1e-12
        )
    )

    bull_rejection = (
        lower > body * 1.5
        and
        c["close"] >
        c["open"]
    )

    bear_rejection = (
        upper > body * 1.5
        and
        c["close"] <
        c["open"]
    )

    return {
        "bull_engulf":
            bool(bull_engulf),

        "bear_engulf":
            bool(bear_engulf),

        "hammer":
            bool(hammer),

        "shooting_star":
            bool(shooting_star),

        "bull_rejection":
            bool(bull_rejection),

        "bear_rejection":
            bool(bear_rejection),

        "body_ratio":
            body / candle_range,
    }


# ============================================================
# PATTERN ENGINE
# ============================================================

def detect_patterns(
    df,
    st
):

    close = float(
        df["close"].iloc[-1]
    )

    ema20 = float(
        df["ema20"].iloc[-1]
    )

    last_h = st[
        "last_high"
    ]

    prev_h = st[
        "previous_high"
    ]

    last_l = st[
        "last_low"
    ]

    prev_l = st[
        "previous_low"
    ]

    equal_high = (
        last_h is not None
        and
        prev_h is not None
        and
        abs(last_h - prev_h) /
        max(last_h, 1e-12)
        < 0.003
    )

    equal_low = (
        last_l is not None
        and
        prev_l is not None
        and
        abs(last_l - prev_l) /
        max(last_l, 1e-12)
        < 0.003
    )

    double_top = (
        equal_high
        and
        close <
        float(
            df["low"].iloc[-11:-1].min()
        )
    )

    double_bottom = (
        equal_low
        and
        close >
        float(
            df["high"].iloc[-11:-1].max()
        )
    )

    head_shoulders = (
        st["HH"] and
        st["LH"]
    )

    inverse_hs = (
        st["LL"] and
        st["HL"]
    )

    triple_top = (
        equal_high
        and
        st["LH"]
        and
        close < ema20
    )

    triple_bottom = (
        equal_low
        and
        st["HL"]
        and
        close > ema20
    )

    ascending_triangle = (
        equal_high and
        st["HL"]
    )

    descending_triangle = (
        equal_low and
        st["LH"]
    )

    symmetrical_triangle = (
        st["LH"] and
        st["HL"]
    )

    rising_wedge = (
        st["HH"] and
        st["HL"] and
        st["LH"]
    )

    falling_wedge = (
        st["LL"] and
        st["HL"] and
        st["LH"]
    )

    bull_flag = (
        ema20 >
        float(
            df["ema50"].iloc[-1]
        )
        and
        df["low"].tail(8).min() >
        df["low"].tail(20).min()
        and
        close > ema20
    )

    bear_flag = (
        ema20 <
        float(
            df["ema50"].iloc[-1]
        )
        and
        df["high"].tail(8).max() <
        df["high"].tail(20).max()
        and
        close < ema20
    )

    range_high = float(
        df["high"].tail(50).max()
    )

    range_low = float(
        df["low"].tail(50).min()
    )

    range_width = safe_div(
        range_high - range_low,
        close
    )

    range_market = (
        range_width < 0.08
    )

    previous_high = float(
        df["high"].iloc[-2]
    )

    previous_low = float(
        df["low"].iloc[-2]
    )

    range_break_bull = (
        range_market
        and
        close > previous_high
    )

    range_break_bear = (
        range_market
        and
        close < previous_low
    )

    mid = (
        range_high +
        range_low
    ) / 2

    cup_handle_bull = (
        close > mid
        and
        close > ema20
    )

    cup_handle_bear = (
        close < mid
        and
        close < ema20
    )

    return {

        "equal_high":
            equal_high,

        "equal_low":
            equal_low,

        "double_top":
            double_top,

        "double_bottom":
            double_bottom,

        "head_shoulders":
            head_shoulders,

        "inverse_hs":
            inverse_hs,

        "triple_top":
            triple_top,

        "triple_bottom":
            triple_bottom,

        "ascending_triangle":
            ascending_triangle,

        "descending_triangle":
            descending_triangle,

        "symmetrical_triangle":
            symmetrical_triangle,

        "rising_wedge":
            rising_wedge,

        "falling_wedge":
            falling_wedge,

        "bull_flag":
            bull_flag,

        "bear_flag":
            bear_flag,

        "range_market":
            range_market,

        "range_break_bull":
            range_break_bull,

        "range_break_bear":
            range_break_bear,

        "cup_handle_bull":
            cup_handle_bull,

        "cup_handle_bear":
            cup_handle_bear,
    }


# ============================================================
# TIMEFRAME CONTEXT
# ============================================================

def timeframe_context(df):

    x = add_indicators(df)

    price = float(
        x["close"].iloc[-1]
    )

    buyer_pressure, seller_pressure = (
        pressure(x)
    )

    return {

        "price": price,

        "ema20":
            float(x["ema20"].iloc[-1]),

        "ema50":
            float(x["ema50"].iloc[-1]),

        "ema200":
            float(x["ema200"].iloc[-1]),

        "rsi":
            float(x["rsi"].iloc[-1]),

        "atr":
            float(x["atr"].iloc[-1]),

        "volume_ratio":
            safe_float(
                x["volume_ratio"].iloc[-1],
                1.0
            ),

        "body_ratio":
            safe_float(
                x["body_ratio"].iloc[-1],
                0.0
            ),

        "bull":
            price >
            float(x["ema20"].iloc[-1]) >
            float(x["ema50"].iloc[-1]),

        "bear":
            price <
            float(x["ema20"].iloc[-1]) <
            float(x["ema50"].iloc[-1]),

        "buyer_pressure":
            buyer_pressure,

        "seller_pressure":
            seller_pressure,

        "df":
            x,
    }


# ============================================================
# DEFAULT LEARNING
# ============================================================

DEFAULT_LEARNING = {

    "global": {
        "wins": 0,
        "losses": 0,
        "win_rate": 50.0,
        "net_r": 0.0,
    },

    "symbols": {},

    "directions": {
        "BUY": {
            "wins": 0,
            "losses": 0,
            "profit_r": 0.0,
        },

        "SELL": {
            "wins": 0,
            "losses": 0,
            "profit_r": 0.0,
        },
    },

    "indicators": {},

    "combinations": {},

    "regimes": {},
}


LEARNING = load_json(
    LEARNING_FILE,
    DEFAULT_LEARNING
)


# ============================================================
# LEARNING HELPERS
# ============================================================

def ensure_learning_structure():

    LEARNING.setdefault(
        "global",
        DEFAULT_LEARNING["global"].copy()
    )

    LEARNING.setdefault(
        "symbols",
        {}
    )

    LEARNING.setdefault(
        "directions",
        {
            "BUY": {
                "wins": 0,
                "losses": 0,
                "profit_r": 0.0
            },
            "SELL": {
                "wins": 0,
                "losses": 0,
                "profit_r": 0.0
            },
        }
    )

    LEARNING.setdefault(
        "indicators",
        {}
    )

    LEARNING.setdefault(
        "combinations",
        {}
    )

    LEARNING.setdefault(
        "regimes",
        {}
    )


ensure_learning_structure()


def ensure_symbol(symbol):

    LEARNING[
        "symbols"
    ].setdefault(
        symbol,
        {
            "wins": 0,
            "losses": 0,
            "profit_r": 0.0,
        }
    )


def ensure_indicator(name):

    LEARNING[
        "indicators"
    ].setdefault(
        name,
        {
            "wins": 0,
            "losses": 0,
            "profit_r": 0.0,
            "weight": 1.0,
        }
    )


def ensure_combination(name):

    LEARNING[
        "combinations"
    ].setdefault(
        name,
        {
            "wins": 0,
            "losses": 0,
            "profit_r": 0.0,
            "weight": 1.0,
        }
    )


# ============================================================
# INDICATOR WEIGHT
# ============================================================

def indicator_weight(name):

    ensure_indicator(name)

    item = LEARNING[
        "indicators"
    ][name]

    samples = (
        item["wins"] +
        item["losses"]
    )

    if samples < MIN_LEARNING_SAMPLES:
        return 1.0

    return clamp(
        safe_float(
            item.get(
                "weight",
                1.0
            ),
            1.0
        ),
        MIN_INDICATOR_WEIGHT,
        MAX_INDICATOR_WEIGHT
    )


def combination_weight(names):

    if not names:
        return 1.0

    key = "|".join(
        sorted(
            set(names)
        )
    )

    ensure_combination(key)

    item = LEARNING[
        "combinations"
    ][key]

    samples = (
        item["wins"] +
        item["losses"]
    )

    if samples < MIN_LEARNING_SAMPLES:
        return 1.0

    return clamp(
        safe_float(
            item.get(
                "weight",
                1.0
            ),
            1.0
        ),
        MIN_INDICATOR_WEIGHT,
        MAX_INDICATOR_WEIGHT
)
    # ============================================================
# LEARNING FACTOR
# ============================================================

def learning_factor(
    symbol,
    direction
):

    ensure_symbol(symbol)

    records = []

    symbol_data = LEARNING[
        "symbols"
    ][symbol]

    symbol_samples = (
        symbol_data["wins"] +
        symbol_data["losses"]
    )

    if symbol_samples >= 5:

        records.append(
            symbol_data["wins"] /
            symbol_samples *
            100
        )

    direction_data = LEARNING[
        "directions"
    ][direction]

    direction_samples = (
        direction_data["wins"] +
        direction_data["losses"]
    )

    if direction_samples >= 5:

        records.append(
            direction_data["wins"] /
            direction_samples *
            100
        )

    if not records:
        return 1.0

    avg = float(
        np.mean(records)
    )

    if avg >= 65:
        return 1.04

    if avg <= 40:
        return 0.96

    return 1.0


# ============================================================
# SCORE ENGINE
# ============================================================

def score_market(
    symbol,
    d1,
    h4,
    h1
):

    buy = 0.0
    sell = 0.0

    buy_reasons = []
    sell_reasons = []

    buy_indicators = []
    sell_indicators = []

    setup_names = []

    def add(
        side,
        points,
        reason,
        indicator=None
    ):

        nonlocal buy
        nonlocal sell

        weight = 1.0

        if indicator:
            weight = indicator_weight(
                indicator
            )

        final_points = (
            points * weight
        )

        if side == "BUY":

            buy += final_points

            buy_reasons.append(
                reason
            )

            if indicator:
                buy_indicators.append(
                    indicator
                )

        else:

            sell += final_points

            sell_reasons.append(
                reason
            )

            if indicator:
                sell_indicators.append(
                    indicator
                )

    # --------------------------------------------------------
    # 1D
    # --------------------------------------------------------

    if d1["bull"]:

        add(
            "BUY",
            7,
            "1D BULLISH",
            "1D_BIAS"
        )

    if d1["bear"]:

        add(
            "SELL",
            7,
            "1D BEARISH",
            "1D_BIAS"
        )

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    if h4["bull"]:

        add(
            "BUY",
            9,
            "4H TREND",
            "4H_TREND"
        )

    if h4["bear"]:

        add(
            "SELL",
            9,
            "4H TREND",
            "4H_TREND"
        )

    # --------------------------------------------------------
    # 1H
    # --------------------------------------------------------

    if h1["bull"]:

        add(
            "BUY",
            10,
            "1H TREND",
            "1H_TREND"
        )

    if h1["bear"]:

        add(
            "SELL",
            10,
            "1H TREND",
            "1H_TREND"
        )

    # --------------------------------------------------------
    # Pressure
    # --------------------------------------------------------

    if h1["buyer_pressure"] >= 54:

        add(
            "BUY",
            8,
            "BUYER PRESSURE",
            "PRESSURE"
        )

        setup_names.append(
            "PRESSURE"
        )

    if h1["seller_pressure"] >= 54:

        add(
            "SELL",
            8,
            "SELLER PRESSURE",
            "PRESSURE"
        )

        setup_names.append(
            "PRESSURE"
        )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if (
        h1["volume_ratio"] >= 1.05
        and
        h1["price"] > h1["ema20"]
    ):

        add(
            "BUY",
            6,
            "BUY VOLUME",
            "VOLUME"
        )

        setup_names.append(
            "VOLUME"
        )

    if (
        h1["volume_ratio"] >= 1.05
        and
        h1["price"] < h1["ema20"]
    ):

        add(
            "SELL",
            6,
            "SELL VOLUME",
            "VOLUME"
        )

        setup_names.append(
            "VOLUME"
        )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    st = structure_info(
        h1["df"]
    )

    if st["HH"] and st["HL"]:

        add(
            "BUY",
            8,
            "HH / HL",
            "HH_HL"
        )

        setup_names.append(
            "HH_HL"
        )

    if st["LH"] and st["LL"]:

        add(
            "SELL",
            8,
            "LH / LL",
            "LH_LL"
        )

        setup_names.append(
            "LH_LL"
        )

    if st["bull_bos"]:

        add(
            "BUY",
            12,
            "BULL BOS",
            "BOS"
        )

        setup_names.append(
            "BOS"
        )

    if st["bear_bos"]:

        add(
            "SELL",
            12,
            "BEAR BOS",
            "BOS"
        )

        setup_names.append(
            "BOS"
        )

    if st["bull_choch"]:

        add(
            "BUY",
            10,
            "BULL CHoCH",
            "CHOCH"
        )

        setup_names.append(
            "CHOCH"
        )

    if st["bear_choch"]:

        add(
            "SELL",
            10,
            "BEAR CHoCH",
            "CHOCH"
        )

        setup_names.append(
            "CHOCH"
        )

    # --------------------------------------------------------
    # SMC
    # --------------------------------------------------------

    smc = detect_smc(
        h1["df"],
        st
    )

    if smc["bull_sweep"]:

        add(
            "BUY",
            10,
            "LIQUIDITY SWEEP",
            "LIQUIDITY_SWEEP"
        )

        setup_names.append(
            "LIQUIDITY_SWEEP"
        )

    if smc["bear_sweep"]:

        add(
            "SELL",
            10,
            "LIQUIDITY SWEEP",
            "LIQUIDITY_SWEEP"
        )

        setup_names.append(
            "LIQUIDITY_SWEEP"
        )

    if smc["discount"]:

        add(
            "BUY",
            4,
            "DISCOUNT",
            "DISCOUNT"
        )

    if smc["premium"]:

        add(
            "SELL",
            4,
            "PREMIUM"
        )

    if smc["bull_fvg"]:

        add(
            "BUY",
            7,
            "BULL FVG",
            "FVG"
        )

        setup_names.append(
            "FVG"
        )

    if smc["bear_fvg"]:

        add(
            "SELL",
            7,
            "BEAR FVG",
            "FVG"
        )

        setup_names.append(
            "FVG"
        )

    if smc["bull_ob"]:

        add(
            "BUY",
            7,
            "BULL ORDER BLOCK",
            "ORDER_BLOCK"
        )

        setup_names.append(
            "ORDER_BLOCK"
        )

    if smc["bear_ob"]:

        add(
            "SELL",
            7,
            "BEAR ORDER BLOCK",
            "ORDER_BLOCK"
        )

        setup_names.append(
            "ORDER_BLOCK"
        )

    if smc["bull_breaker"]:

        add(
            "BUY",
            5,
            "BULL BREAKER",
            "BREAKER"
        )

        setup_names.append(
            "BREAKER"
        )

    if smc["bear_breaker"]:

        add(
            "SELL",
            5,
            "BEAR BREAKER",
            "BREAKER"
        )

        setup_names.append(
            "BREAKER"
        )

    # --------------------------------------------------------
    # Support / Resistance
    # --------------------------------------------------------

    support = float(
        h1["df"]["low"]
        .iloc[-51:-1]
        .min()
    )

    resistance = float(
        h1["df"]["high"]
        .iloc[-51:-1]
        .max()
    )

    price = h1["price"]

    if (
        abs(price - support) /
        max(price, 1e-12)
        <= 0.010
    ):

        add(
            "BUY",
            7,
            "SUPPORT",
            "SUPPORT"
        )

        setup_names.append(
            "SUPPORT"
        )

    if (
        abs(resistance - price) /
        max(price, 1e-12)
        <= 0.010
    ):

        add(
            "SELL",
            7,
            "RESISTANCE",
            "RESISTANCE"
        )

        setup_names.append(
            "RESISTANCE"
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if (
        44 <= h1["rsi"] <= 68
        and
        h1["bull"]
    ):

        add(
            "BUY",
            4,
            "RSI",
            "RSI"
        )

    if (
        32 <= h1["rsi"] <= 56
        and
        h1["bear"]
    ):

        add(
            "SELL",
            4,
            "RSI",
            "RSI"
        )

    # --------------------------------------------------------
    # Candles
    # --------------------------------------------------------

    candles = candle_patterns(
        h1["df"]
    )

    if (
        candles["bull_engulf"]
        or
        candles["hammer"]
        or
        candles["bull_rejection"]
    ):

        add(
            "BUY",
            5,
            "BULL PRICE ACTION",
            "PRICE_ACTION"
        )

        setup_names.append(
            "PRICE_ACTION"
        )

    if (
        candles["bear_engulf"]
        or
        candles["shooting_star"]
        or
        candles["bear_rejection"]
    ):

        add(
            "SELL",
            5,
            "BEAR PRICE ACTION",
            "PRICE_ACTION"
        )

        setup_names.append(
            "PRICE_ACTION"
        )

    # --------------------------------------------------------
    # Classical patterns
    # --------------------------------------------------------

    patterns = detect_patterns(
        h1["df"],
        st
    )

    pattern_scores = [

        (
            "double_bottom",
            "BUY",
            9,
            "DOUBLE BOTTOM",
            "DOUBLE_BOTTOM"
        ),

        (
            "double_top",
            "SELL",
            9,
            "DOUBLE TOP",
            "DOUBLE_TOP"
        ),

        (
            "inverse_hs",
            "BUY",
            8,
            "INVERSE H&S",
            "INVERSE_HS"
        ),

        (
            "head_shoulders",
            "SELL",
            8,
            "HEAD & SHOULDERS",
            "HEAD_SHOULDERS"
        ),

        (
            "triple_bottom",
            "BUY",
            7,
            "TRIPLE BOTTOM",
            "TRIPLE_BOTTOM"
        ),

        (
            "triple_top",
            "SELL",
            7,
            "TRIPLE TOP",
            "TRIPLE_TOP"
        ),

        (
            "ascending_triangle",
            "BUY",
            6,
            "ASCENDING TRIANGLE",
            "ASC_TRIANGLE"
        ),

        (
            "descending_triangle",
            "SELL",
            6,
            "DESCENDING TRIANGLE",
            "DESC_TRIANGLE"
        ),

        (
            "falling_wedge",
            "BUY",
            6,
            "FALLING WEDGE",
            "FALLING_WEDGE"
        ),

        (
            "rising_wedge",
            "SELL",
            6,
            "RISING WEDGE",
            "RISING_WEDGE"
        ),

        (
            "bull_flag",
            "BUY",
            5,
            "BULL FLAG",
            "BULL_FLAG"
        ),

        (
            "bear_flag",
            "SELL",
            5,
            "BEAR FLAG",
            "BEAR_FLAG"
        ),

        (
            "range_break_bull",
            "BUY",
            7,
            "RANGE BREAKOUT",
            "RANGE_BREAKOUT"
        ),

        (
            "range_break_bear",
            "SELL",
            7,
            "RANGE BREAKDOWN",
            "RANGE_BREAKDOWN"
        ),
    ]

    for (
        key,
        side,
        points,
        label,
        indicator
    ) in pattern_scores:

        if patterns.get(key):

            add(
                side,
                points,
                label,
                indicator
            )

            setup_names.append(
                indicator
            )

    # --------------------------------------------------------
    # Combination learning
    # --------------------------------------------------------

    unique_setups = list(
        dict.fromkeys(
            setup_names
        )
    )

    combo_factor = (
        combination_weight(
            unique_setups
        )
    )

    buy *= combo_factor
    sell *= combo_factor

    # Symbol / direction learning
    buy *= learning_factor(
        symbol,
        "BUY"
    )

    sell *= learning_factor(
        symbol,
        "SELL"
    )

    buy = clamp(
        buy,
        0,
        100
    )

    sell = clamp(
        sell,
        0,
        100
    )

    if (
        buy >= MIN_SCORE
        and
        buy >= sell + DIRECTION_GAP
    ):

        direction = "BUY"

    elif (
        sell >= MIN_SCORE
        and
        sell >= buy + DIRECTION_GAP
    ):

        direction = "SELL"

    else:

        direction = "NO TRADE"

    return {

        "buy_score":
            round(buy, 2),

        "sell_score":
            round(sell, 2),

        "direction":
            direction,

        "buy_reasons":
            buy_reasons,

        "sell_reasons":
            sell_reasons,

        "buy_indicators":
            list(dict.fromkeys(
                buy_indicators
            )),

        "sell_indicators":
            list(dict.fromkeys(
                sell_indicators
            )),

        "setups":
            list(dict.fromkeys(
                setup_names
            )),

        "support":
            support,

        "resistance":
            resistance,

        "structure":
            st,

        "smc":
            smc,

        "patterns":
            patterns,

        "candles":
            candles,
    }


# ============================================================
# DYNAMIC LEVELS
# ============================================================

def nearest_levels(
    df,
    structure
):

    price = float(
        df["close"].iloc[-1]
    )

    lows = [
        x[1]
        for x in structure[
            "lows"
        ][-10:]
    ]

    highs = [
        x[1]
        for x in structure[
            "highs"
        ][-10:]
    ]

    supports = [
        x for x in lows
        if x < price
    ]

    resistances = [
        x for x in highs
        if x > price
    ]

    support = (
        max(supports)
        if supports
        else float(
            df["low"].tail(20).min()
        )
    )

    resistance = (
        min(resistances)
        if resistances
        else float(
            df["high"].tail(20).max()
        )
    )

    return (
        support,
        resistance
    )


def dynamic_levels(
    direction,
    h1,
    score
):

    price = float(
        h1["price"]
    )

    atr_value = max(
        float(h1["atr"]),
        price * 0.001
    )

    structure = structure_info(
        h1["df"]
    )

    support, resistance = (
        nearest_levels(
            h1["df"],
            structure
        )
    )

    if direction == "BUY":

        structural_sl = (
            support -
            atr_value * 0.20
        )

        atr_sl = (
            price -
            atr_value * ATR_SL_MULT
        )

        stop = min(
            structural_sl,
            atr_sl
        )

        max_stop = (
            price *
            (1 - MAX_SL_PCT / 100)
        )

        min_stop = (
            price *
            (1 - MIN_SL_PCT / 100)
        )

        stop = clamp(
            stop,
            max_stop,
            min_stop
        )

        risk = price - stop

        structural_target = resistance

        atr_target = (
            price +
            atr_value *
            ATR_TARGET_MULT
        )

        target = max(
            structural_target,
            atr_target
        )

        min_target = (
            price +
            risk * MIN_RR
        )

        target = max(
            target,
            min_target
        )

        max_target = (
            price *
            (1 + MAX_TARGET_PCT / 100)
        )

        target = min(
            target,
            max_target
        )

    else:

        structural_sl = (
            resistance +
            atr_value * 0.20
        )

        atr_sl = (
            price +
            atr_value * ATR_SL_MULT
        )

        stop = max(
            structural_sl,
            atr_sl
        )

        max_stop = (
            price *
            (1 + MAX_SL_PCT / 100)
        )

        min_stop = (
            price *
            (1 + MIN_SL_PCT / 100)
        )

        stop = clamp(
            stop,
            min_stop,
            max_stop
        )

        risk = stop - price

        structural_target = support

        atr_target = (
            price -
            atr_value *
            ATR_TARGET_MULT
        )

        target = min(
            structural_target,
            atr_target
        )

        min_target = (
            price -
            risk * MIN_RR
        )

        target = min(
            target,
            min_target
        )

        max_target = (
            price *
            (1 - MAX_TARGET_PCT / 100)
        )

        target = max(
            target,
            max_target
        )

    risk_pct = (
        abs(price - stop) /
        price *
        100
    )

    reward_pct = (
        abs(target - price) /
        price *
        100
    )

    rr = safe_div(
        reward_pct,
        risk_pct
    )

    valid = (
        rr >= MIN_RR
        and
        rr <= MAX_RR
        and
        risk_pct >= MIN_SL_PCT
        and
        risk_pct <= MAX_SL_PCT
        and
        reward_pct >= MIN_TARGET_PCT
        and
        target != price
        and
        stop != price
    )

    return {

        "entry":
            price,

        "stop_loss":
            stop,

        "target":
            target,

        "risk_pct":
            risk_pct,

        "reward_pct":
            reward_pct,

        "rr":
            rr,

        "valid":
            valid,

        "support":
            support,

        "resistance":
            resistance,

        "atr":
            atr_value,

        "score":
            score,
        }
    # ============================================================
# TRADE MEMORY
# ============================================================

TRADES = load_json(
    TRADE_MEMORY_FILE,
    []
)

OPEN_TRADES = {}

EMAIL_SENT = load_json(
    STATE_FILE,
    {}
)


# ============================================================
# TRADE ID
# ============================================================

def make_signal_id(
    symbol,
    direction,
    signal_candle
):

    candle_key = str(
        signal_candle
    )

    return (
        f"{symbol}|"
        f"{direction}|"
        f"{candle_key}"
    )


def make_trade_id():

    return (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d%H%M%S%f"
        )
        + "_"
        + uuid.uuid4().hex[:8]
    )


# ============================================================
# DUPLICATE PROTECTION
# ============================================================

def same_open_trade_exists(
    symbol,
    direction,
    signal_id
):

    for trade in OPEN_TRADES.values():

        if (
            trade.get("symbol") ==
            symbol
            and
            trade.get("direction") ==
            direction
            and
            trade.get("signal_id") ==
            signal_id
        ):

            return True

    return False


def symbol_has_open_trade(
    symbol
):

    for trade in OPEN_TRADES.values():

        if trade.get("symbol") == symbol:

            return True

    return False


def trade_already_closed(
    signal_id
):

    for trade in TRADES:

        if (
            trade.get("signal_id") ==
            signal_id
            and
            trade.get("result")
            in ("WIN", "LOSS")
        ):

            return True

    return False


# ============================================================
# INDICATOR SNAPSHOT
# ============================================================

def build_indicator_snapshot(
    score
):

    indicators = set()

    if score["direction"] == "BUY":

        indicators.update(
            score["buy_indicators"]
        )

    elif score["direction"] == "SELL":

        indicators.update(
            score["sell_indicators"]
        )

    return sorted(
        indicators
    )


# ============================================================
# SAVE TRADE
# ============================================================

def save_trade(trade):

    global TRADES

    signal_id = trade.get(
        "signal_id"
    )

    # --------------------------------------------------------
    # NEVER append duplicate closed trade
    # --------------------------------------------------------

    for existing in TRADES:

        if (
            existing.get(
                "signal_id"
            ) == signal_id
            and
            existing.get(
                "result"
            ) == trade.get(
                "result"
            )
            and
            trade.get(
                "result"
            ) in ("WIN", "LOSS")
        ):

            logger.warning(
                "Duplicate trade ignored: %s",
                signal_id
            )

            return False

    TRADES.append(
        trade
    )

    save_json(
        TRADE_MEMORY_FILE,
        TRADES
    )

    row = {

        "id":
            trade.get("id"),

        "signal_id":
            trade.get("signal_id"),

        "symbol":
            trade.get("symbol"),

        "direction":
            trade.get("direction"),

        "entry":
            trade.get("entry"),

        "stop_loss":
            trade.get("stop_loss"),

        "target":
            trade.get("target"),

        "entry_time":
            trade.get("entry_time"),

        "exit_time":
            trade.get("exit_time"),

        "result":
            trade.get("result"),

        "reason":
            trade.get("reason"),

        "exit_price":
            trade.get("exit_price"),

        "pnl_pct":
            trade.get("pnl_pct"),

        "r_multiple":
            trade.get("r_multiple"),

        "mae_pct":
            trade.get("mae_pct"),

        "mae_r":
            trade.get("mae_r"),

        "mfe_pct":
            trade.get("mfe_pct"),

        "mfe_r":
            trade.get("mfe_r"),

        "buy_score":
            trade.get("buy_score"),

        "sell_score":
            trade.get("sell_score"),

        "rr":
            trade.get("rr"),

        "setups":
            ",".join(
                trade.get(
                    "setups",
                    []
                )
            ),

        "indicators":
            ",".join(
                trade.get(
                    "indicators",
                    []
                )
            ),
    }

    row_df = pd.DataFrame(
        [row]
    )

    header = not (
        TRADE_CSV_FILE.exists()
    )

    row_df.to_csv(
        TRADE_CSV_FILE,
        mode="a",
        header=header,
        index=False
    )

    return True


# ============================================================
# PERSIST OPEN STATE
# ============================================================

def save_state():

    data = {

        "open_trades":
            list(
                OPEN_TRADES.values()
            ),

        "email_sent":
            EMAIL_SENT,

        "updated_at":
            iso_pkt(),
    }

    save_json(
        STATE_FILE,
        data
    )


def restore_open_trades():

    global OPEN_TRADES
    global EMAIL_SENT

    state = load_json(
        STATE_FILE,
        {}
    )

    EMAIL_SENT = state.get(
        "email_sent",
        {}
    )

    OPEN_TRADES = {}

    for trade in state.get(
        "open_trades",
        []
    ):

        if (
            trade.get(
                "result"
            ) == "OPEN"
        ):

            OPEN_TRADES[
                trade["id"]
            ] = trade

    logger.info(
        "Restored %d open trades",
        len(OPEN_TRADES)
    )


# ============================================================
# LEARNING UPDATE
# ============================================================

def update_learning(
    trade
):

    ensure_learning_structure()

    result = trade.get(
        "result"
    )

    if result not in (
        "WIN",
        "LOSS"
    ):

        return

    symbol = trade[
        "symbol"
    ]

    direction = trade[
        "direction"
    ]

    r_multiple = safe_float(
        trade.get(
            "r_multiple",
            0
        )
    )

    ensure_symbol(
        symbol
    )

    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------

    if result == "WIN":

        LEARNING[
            "global"
        ]["wins"] += 1

    else:

        LEARNING[
            "global"
        ]["losses"] += 1

    LEARNING[
        "global"
    ]["net_r"] += r_multiple

    # --------------------------------------------------------
    # Symbol
    # --------------------------------------------------------

    if result == "WIN":

        LEARNING[
            "symbols"
        ][symbol]["wins"] += 1

    else:

        LEARNING[
            "symbols"
        ][symbol]["losses"] += 1

    LEARNING[
        "symbols"
    ][symbol]["profit_r"] += r_multiple

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    d = LEARNING[
        "directions"
    ][direction]

    if result == "WIN":

        d["wins"] += 1

    else:

        d["losses"] += 1

    d["profit_r"] += r_multiple

    # --------------------------------------------------------
    # Indicator learning
    # --------------------------------------------------------

    indicators = trade.get(
        "indicators",
        []
    )

    for indicator in indicators:

        ensure_indicator(
            indicator
        )

        item = LEARNING[
            "indicators"
        ][indicator]

        if result == "WIN":

            item["wins"] += 1

            item["weight"] = clamp(
                item["weight"] *
                (
                    1 +
                    LEARNING_WIN_REWARD
                ),
                MIN_INDICATOR_WEIGHT,
                MAX_INDICATOR_WEIGHT
            )

        else:

            item["losses"] += 1

            item["weight"] = clamp(
                item["weight"] *
                (
                    1 -
                    LEARNING_LOSS_PENALTY
                ),
                MIN_INDICATOR_WEIGHT,
                MAX_INDICATOR_WEIGHT
            )

        item["profit_r"] += (
            r_multiple
        )

    # --------------------------------------------------------
    # Combination learning
    # --------------------------------------------------------

    combination = sorted(
        set(indicators)
    )

    if combination:

        combo_key = "|".join(
            combination
        )

        ensure_combination(
            combo_key
        )

        combo = LEARNING[
            "combinations"
        ][combo_key]

        if result == "WIN":

            combo["wins"] += 1

            combo["weight"] = clamp(
                combo["weight"] *
                (
                    1 +
                    LEARNING_WIN_REWARD
                ),
                MIN_INDICATOR_WEIGHT,
                MAX_INDICATOR_WEIGHT
            )

        else:

            combo["losses"] += 1

            combo["weight"] = clamp(
                combo["weight"] *
                (
                    1 -
                    LEARNING_LOSS_PENALTY
                ),
                MIN_INDICATOR_WEIGHT,
                MAX_INDICATOR_WEIGHT
            )

        combo["profit_r"] += (
            r_multiple
        )

    # --------------------------------------------------------
    # Global win rate
    # --------------------------------------------------------

    wins = LEARNING[
        "global"
    ]["wins"]

    losses = LEARNING[
        "global"
    ]["losses"]

    total = wins + losses

    LEARNING[
        "global"
    ]["win_rate"] = (
        wins / total * 100
        if total
        else 50.0
    )

    save_json(
        LEARNING_FILE,
        LEARNING
    )


# ============================================================
# TRADE OPEN EMAIL
# ============================================================

def send_trade_open_alert(
    trade
):

    signal_id = trade[
        "signal_id"
    ]

    email_key = (
        "OPEN|" +
        signal_id
    )

    if EMAIL_SENT.get(
        email_key
    ):

        return

    body = f"""
🧠 MARKET BRAIN AI — NEW TRADE

━━━━━━━━━━━━━━━━━━━━
TRADE
━━━━━━━━━━━━━━━━━━━━

Symbol: {trade['symbol']}
Direction: {trade['direction']}

ENTRY:
{trade['entry']:.8f}

STOP LOSS:
{trade['stop_loss']:.8f}

TARGET:
{trade['target']:.8f}

RISK:
{trade['risk_pct']:.2f}%

REWARD:
{trade['reward_pct']:.2f}%

R:R:
1:{trade['rr']:.2f}

━━━━━━━━━━━━━━━━━━━━
SCORES
━━━━━━━━━━━━━━━━━━━━

BUY SCORE: {trade['buy_score']:.1f}
SELL SCORE: {trade['sell_score']:.1f}

1D: {trade['d1_bias']}
4H: {trade['h4_bias']}
1H: {trade['h1_bias']}

━━━━━━━━━━━━━━━━━━━━
MARKET
━━━━━━━━━━━━━━━━━━━━

Buyer Pressure:
{trade['buyer_pressure']:.1f}%

Seller Pressure:
{trade['seller_pressure']:.1f}%

Volume:
{trade['volume_ratio']:.2f}x

RSI:
{trade['rsi']:.1f}

━━━━━━━━━━━━━━━━━━━━
SETUPS
━━━━━━━━━━━━━━━━━━━━

{', '.join(trade['setups']) or 'None'}

━━━━━━━━━━━━━━━━━━━━
LEARNING INDICATORS
━━━━━━━━━━━━━━━━━━━━

{', '.join(trade['indicators']) or 'None'}

Entry Candle:
{trade['signal_candle']}

Time:
{trade['entry_time']} PKT
"""

    sent = send_email(
        (
            "🧠 MARKET BRAIN "
            f"{trade['direction']} — "
            f"{trade['symbol']}"
        ),
        body.strip()
    )

    if sent:

        EMAIL_SENT[
            email_key
        ] = iso_pkt()

        save_state()


# ============================================================
# CLOSED TRADE EMAIL
# ============================================================

def send_trade_closed_alert(
    trade
):

    signal_id = trade[
        "signal_id"
    ]

    email_key = (
        "CLOSED|" +
        signal_id
    )

    if EMAIL_SENT.get(
        email_key
    ):

        return

    result = trade[
        "result"
    ]

    if result == "WIN":

        emoji = "✅"

    else:

        emoji = "❌"

    body = f"""
🧠 MARKET BRAIN AI — TRADE CLOSED

{emoji} {result}

━━━━━━━━━━━━━━━━━━━━
TRADE
━━━━━━━━━━━━━━━━━━━━

Symbol:
{trade['symbol']}

Direction:
{trade['direction']}

━━━━━━━━━━━━━━━━━━━━
PRICES
━━━━━━━━━━━━━━━━━━━━

Entry:
{trade['entry']:.8f}

Exit:
{trade['exit_price']:.8f}

Stop Loss:
{trade['stop_loss']:.8f}

Target:
{trade['target']:.8f}

━━━━━━━━━━━━━━━━━━━━
RESULT
━━━━━━━━━━━━━━━━━━━━

RESULT:
{result}

REASON:
{trade['reason']}

P/L:
{trade['pnl_pct']:+.2f}%

RESULT:
{trade['r_multiple']:+.2f}R

PLANNED R:R:
1:{trade['rr']:.2f}

━━━━━━━━━━━━━━━━━━━━
TRADE BEHAVIOUR
━━━━━━━━━━━━━━━━━━━━

MAE:
{trade['mae_pct']:.2f}% ({trade['mae_r']:.2f}R)

MFE:
{trade['mfe_pct']:.2f}% ({trade['mfe_r']:.2f}R)

━━━━━━━━━━━━━━━━━━━━
TIME
━━━━━━━━━━━━━━━━━━━━

Entry:
{trade['entry_time']} PKT

Exit:
{trade['exit_time']} PKT

━━━━━━━━━━━━━━━━━━━━
SCORE
━━━━━━━━━━━━━━━━━━━━

BUY:
{trade['buy_score']:.1f}

SELL:
{trade['sell_score']:.1f}

━━━━━━━━━━━━━━━━━━━━
SETUPS
━━━━━━━━━━━━━━━━━━━━

{', '.join(trade['setups']) or 'None'}

━━━━━━━━━━━━━━━━━━━━
LEARNING
━━━━━━━━━━━━━━━━━━━━

Indicators used:

{', '.join(trade['indicators']) or 'None'}

The result has been recorded.

Successful indicators receive
positive learning weight.

Failed indicators receive
negative learning weight.

The learning engine will use
this result on future trades.
"""

    sent = send_email(
        (
            f"{emoji} MARKET BRAIN "
            f"{result} — "
            f"{trade['symbol']}"
        ),
        body.strip()
    )

    if sent:

        EMAIL_SENT[
            email_key
        ] = iso_pkt()

        save_state()


# ============================================================
# OPEN TRADE CREATION
# ============================================================

def open_trade(
    result
):

    global OPEN_TRADES

    symbol = result[
        "symbol"
    ]

    score = result[
        "score"
    ]

    levels = result[
        "levels"
    ]

    direction = score[
        "direction"
    ]

    if direction not in (
        "BUY",
        "SELL"
    ):

        return False

    if not levels[
        "valid"
    ]:

        return False

    # --------------------------------------------------------
    # Maximum open trades
    # --------------------------------------------------------

    if len(
        OPEN_TRADES
    ) >= MAX_OPEN_TRADES:

        return False

    # --------------------------------------------------------
    # One symbol = one open trade
    # --------------------------------------------------------

    if symbol_has_open_trade(
        symbol
    ):

        return False

    signal_candle = result[
        "signal_candle"
    ]

    signal_id = make_signal_id(
        symbol,
        direction,
        signal_candle
    )

    # --------------------------------------------------------
    # Duplicate protection
    # --------------------------------------------------------

    if same_open_trade_exists(
        symbol,
        direction,
        signal_id
    ):

        return False

    if trade_already_closed(
        signal_id
    ):

        logger.info(
            "Signal already closed: %s",
            signal_id
        )

        return False

    smc = score[
        "smc"
    ]

    patterns = score[
        "patterns"
    ]

    pattern = "NONE"

    pattern_map = [

        (
            "double_bottom",
            "DOUBLE BOTTOM"
        ),

        (
            "double_top",
            "DOUBLE TOP"
        ),

        (
            "inverse_hs",
            "INVERSE H&S"
        ),

        (
            "head_shoulders",
            "HEAD & SHOULDERS"
        ),

        (
            "ascending_triangle",
            "ASC TRIANGLE"
        ),

        (
            "descending_triangle",
            "DESC TRIANGLE"
        ),

        (
            "falling_wedge",
            "FALLING WEDGE"
        ),

        (
            "rising_wedge",
            "RISING WEDGE"
        ),

        (
            "bull_flag",
            "BULL FLAG"
        ),

        (
            "bear_flag",
            "BEAR FLAG"
        ),
    ]

    for key, label in pattern_map:

        if patterns.get(key):

            pattern = label
            break

    indicators = build_indicator_snapshot(
        score
    )

    trade = {

        "id":
            make_trade_id(),

        "signal_id":
            signal_id,

        "signal_candle":
            signal_candle,

        "symbol":
            symbol,

        "direction":
            direction,

        "entry":
            levels["entry"],

        "stop_loss":
            levels["stop_loss"],

        "target":
            levels["target"],

        "risk_pct":
            levels["risk_pct"],

        "reward_pct":
            levels["reward_pct"],

        "rr":
            levels["rr"],

        "entry_time":
            iso_pkt(),

        "exit_time":
            None,

        "result":
            "OPEN",

        "reason":
            None,

        "exit_price":
            None,

        "pnl_pct":
            None,

        "r_multiple":
            None,

        "mae_pct":
            0.0,

        "mae_r":
            0.0,

        "mfe_pct":
            0.0,

        "mfe_r":
            0.0,

        "buy_score":
            score["buy_score"],

        "sell_score":
            score["sell_score"],

        "d1_bias":
            (
                "BULLISH"
                if result["d1"]["bull"]
                else
                "BEARISH"
                if result["d1"]["bear"]
                else
                "NEUTRAL"
            ),

        "h4_bias":
            (
                "BULLISH"
                if result["h4"]["bull"]
                else
                "BEARISH"
                if result["h4"]["bear"]
                else
                "NEUTRAL"
            ),

        "h1_bias":
            (
                "BULLISH"
                if result["h1"]["bull"]
                else
                "BEARISH"
                if result["h1"]["bear"]
                else
                "NEUTRAL"
            ),

        "buyer_pressure":
            result["h1"][
                "buyer_pressure"
            ],

        "seller_pressure":
            result["h1"][
                "seller_pressure"
            ],

        "volume_ratio":
            result["h1"][
                "volume_ratio"
            ],

        "rsi":
            result["h1"]["rsi"],

        "fvg":
            (
                "BULLISH"
                if smc["bull_fvg"]
                else
                "BEARISH"
                if smc["bear_fvg"]
                else
                "NONE"
            ),

        "order_block":
            (
                "BULLISH"
                if smc["bull_ob"]
                else
                "BEARISH"
                if smc["bear_ob"]
                else
                "NONE"
            ),

        "liquidity":
            (
                "BUY SWEEP"
                if smc["bull_sweep"]
                else
                "SELL SWEEP"
                if smc["bear_sweep"]
                else
                "NONE"
            ),

        "pattern":
            pattern,

        "support":
            levels["support"],

        "resistance":
            levels["resistance"],

        "atr":
            levels["atr"],

        "setups":
            score["setups"],

        "indicators":
            indicators,

        # Track highest/lowest excursion
        "best_price":
            levels["entry"],

        "worst_price":
            levels["entry"],

        "last_checked_candle":
            signal_candle,
    }

    OPEN_TRADES[
        trade["id"]
    ] = trade

    save_state()

    # --------------------------------------------------------
    # NEW TRADE EMAIL ONLY ONCE
    # --------------------------------------------------------

    send_trade_open_alert(
        trade
    )

    logger.info(
        "OPEN %s %s | "
        "score %.1f/%.1f | "
        "Entry %.8f | "
        "SL %.8f | "
        "TP %.8f | "
        "RR %.2f",
        symbol,
        direction,
        score["buy_score"],
        score["sell_score"],
        levels["entry"],
        levels["stop_loss"],
        levels["target"],
        levels["rr"]
    )

    return True


# ============================================================
# TRADE MONITORING
# ============================================================

def update_trade_excursion(
    trade,
    high,
    low
):

    entry = float(
        trade["entry"]
    )

    direction = trade[
        "direction"
    ]

    if direction == "BUY":

        favorable_pct = (
            high - entry
        ) / entry * 100

        adverse_pct = (
            entry - low
        ) / entry * 100

    else:

        favorable_pct = (
            entry - low
        ) / entry * 100

        adverse_pct = (
            high - entry
        ) / entry * 100

    favorable_pct = max(
        0.0,
        favorable_pct
    )

    adverse_pct = max(
        0.0,
        adverse_pct
    )

    trade[
        "mfe_pct"
    ] = max(
        safe_float(
            trade.get(
                "mfe_pct",
                0
            )
        ),
        favorable_pct
    )

    trade[
        "mae_pct"
    ] = max(
        safe_float(
            trade.get(
                "mae_pct",
                0
            )
        ),
        adverse_pct
    )

    risk_pct = max(
        safe_float(
            trade["risk_pct"]
        ),
        1e-12
    )

    trade[
        "mfe_r"
    ] = (
        trade["mfe_pct"] /
        risk_pct
    )

    trade[
        "mae_r"
    ] = (
        trade["mae_pct"] /
        risk_pct
    )


def close_trade(
    trade,
    result,
    reason,
    exit_price
):

    direction = trade[
        "direction"
    ]

    entry = float(
        trade["entry"]
    )

    exit_price = float(
        exit_price
    )

    if direction == "BUY":

        pnl_pct = (
            exit_price -
            entry
        ) / entry * 100

    else:

        pnl_pct = (
            entry -
            exit_price
        ) / entry * 100

    risk_pct = max(
        safe_float(
            trade["risk_pct"]
        ),
        1e-12
    )

    r_multiple = (
        pnl_pct /
        risk_pct
    )

    trade[
        "exit_price"
    ] = exit_price

    trade[
        "exit_time"
    ] = iso_pkt()

    trade[
        "result"
    ] = result

    trade[
        "reason"
    ] = reason

    trade[
        "pnl_pct"
    ] = pnl_pct

    trade[
        "r_multiple"
    ] = r_multiple

    # --------------------------------------------------------
    # Remove from open first
    # --------------------------------------------------------

    trade_id = trade[
        "id"
    ]

    if trade_id in OPEN_TRADES:

        del OPEN_TRADES[
            trade_id
        ]

    # --------------------------------------------------------
    # Save closed trade
    # --------------------------------------------------------

    saved = save_trade(
        trade
    )

    if not saved:

        logger.warning(
            "Closed trade already saved: %s",
            trade.get(
                "signal_id"
            )
        )

        save_state()

        return

    # --------------------------------------------------------
    # Learn
    # --------------------------------------------------------

    update_learning(
        trade
    )

    # --------------------------------------------------------
    # ONE close email
    # --------------------------------------------------------

    send_trade_closed_alert(
        trade
    )

    save_state()

    logger.info(
        "CLOSED %s %s | %s | "
        "P/L %.2f%% | %.2fR",
        trade["symbol"],
        trade["direction"],
        result,
        pnl_pct,
        r_multiple
    )


def check_open_trades():

    if not OPEN_TRADES:

        return

    for trade_id, trade in list(
        OPEN_TRADES.items()
    ):

        symbol = trade[
            "symbol"
        ]

        df = fetch_klines(
            symbol,
            "1h",
            5
        )

        if df.empty:

            continue

        # ----------------------------------------------------
        # Use latest CLOSED 1H candle.
        # ----------------------------------------------------

        candle = df.iloc[-1]

        high = float(
            candle["high"]
        )

        low = float(
            candle["low"]
        )

        candle_time = str(
            candle["open_time"]
        )

        update_trade_excursion(
            trade,
            high,
            low
        )

        direction = trade[
            "direction"
        ]

        result = None
        reason = None
        exit_price = None

        if direction == "BUY":

            sl_hit = (
                low <=
                trade["stop_loss"]
            )

            tp_hit = (
                high >=
                trade["target"]
            )

            # Conservative rule:
            # if both are touched inside
            # same candle, SL first.
            if sl_hit:

                result = "LOSS"
                reason = "STOP LOSS HIT"

                exit_price = (
                    trade["stop_loss"]
                )

            elif tp_hit:

                result = "WIN"
                reason = "TARGET HIT"

                exit_price = (
                    trade["target"]
                )

        else:

            sl_hit = (
                high >=
                trade["stop_loss"]
            )

            tp_hit = (
                low <=
                trade["target"]
            )

            if sl_hit:

                result = "LOSS"
                reason = "STOP LOSS HIT"

                exit_price = (
                    trade["stop_loss"]
                )

            elif tp_hit:

                result = "WIN"
                reason = "TARGET HIT"

                exit_price = (
                    trade["target"]
                )

        trade[
            "last_checked_candle"
        ] = candle_time

        if result:

            close_trade(
                trade,
                result,
                reason,
                exit_price
            )

        else:

            # Save latest MAE/MFE
            save_state()


# ============================================================
# ANALYZE SYMBOL
# ============================================================

def analyze_symbol(
    symbol
):

    frames = {}

    for key, interval in (
        TIMEFRAMES.items()
    ):

        df = fetch_klines(
            symbol,
            interval
        )

        if df.empty:

            logger.warning(
                "%s %s returned no data",
                symbol,
                interval
            )

            return None

        if len(df) < 210:

            logger.warning(
                "%s %s insufficient candles: %d",
                symbol,
                interval,
                len(df)
            )

            return None

        frames[
            key
        ] = timeframe_context(
            df
        )

    d1 = frames[
        "1d"
    ]

    h4 = frames[
        "4h"
    ]

    h1 = frames[
        "1h"
    ]

    score = score_market(
        symbol,
        d1,
        h4,
        h1
    )

    # --------------------------------------------------------
    # Must have at least some HTF agreement.
    # But do not require all three to agree.
    # --------------------------------------------------------

    direction = score[
        "direction"
    ]

    if direction == "BUY":

        bullish_context = sum(
            [
                int(d1["bull"]),
                int(h4["bull"]),
                int(h1["bull"]),
            ]
        )

        if bullish_context == 0:

            return None

    elif direction == "SELL":

        bearish_context = sum(
            [
                int(d1["bear"]),
                int(h4["bear"]),
                int(h1["bear"]),
            ]
        )

        if bearish_context == 0:

            return None

    else:

        return {
            "symbol":
                symbol,

            "score":
                score,

            "d1":
                d1,

            "h4":
                h4,

            "h1":
                h1,
        }

    # --------------------------------------------------------
    # Entry is 1H.
    # No 5M confirmation.
    # --------------------------------------------------------

    levels = dynamic_levels(
        direction,
        h1,
        max(
            score["buy_score"],
            score["sell_score"]
        )
    )

    if not levels[
        "valid"
    ]:

        return {
            "symbol":
                symbol,

            "score":
                score,

            "d1":
                d1,

            "h4":
                h4,

            "h1":
                h1,

            "levels":
                levels,
        }

    # --------------------------------------------------------
    # The signal candle is the latest closed 1H candle.
    # --------------------------------------------------------

    signal_candle = str(
        h1["df"][
            "open_time"
        ].iloc[-1]
    )

    return {

        "symbol":
            symbol,

        "score":
            score,

        "d1":
            d1,

        "h4":
            h4,

        "h1":
            h1,

        "levels":
            levels,

        "signal_candle":
            signal_candle,
    }


# ============================================================
# SCAN ALL
# ============================================================

def scan_all():

    candidates = []

    logger.info(
        "Scanning %d symbols...",
        len(SYMBOLS)
    )

    for symbol in SYMBOLS:

        try:

            result = analyze_symbol(
                symbol
            )

            if result is None:

                continue

            score = result[
                "score"
            ]

            logger.info(
                "%s | BUY %.1f | SELL %.1f | %s",
                symbol,
                score["buy_score"],
                score["sell_score"],
                score["direction"]
            )

            if (
                score["direction"]
                in ("BUY", "SELL")
                and
                "levels" in result
                and
                result["levels"][
                    "valid"
                ]
            ):

                candidates.append(
                    result
                )

        except Exception as e:

            logger.exception(
                "Analysis failed for %s: %s",
                symbol,
                e
            )

    candidates.sort(
        key=lambda x:
        max(
            x["score"]["buy_score"],
            x["score"]["sell_score"]
        ),
        reverse=True
    )

    opened = 0

    for result in candidates:

        if opened >= (
            MAX_NEW_TRADES_PER_SCAN
        ):

            break

        if open_trade(
            result
        ):

            opened += 1

    logger.info(
        "Scan complete | "
        "Candidates=%d | "
        "Opened=%d | "
        "Open=%d",
        len(candidates),
        opened,
        len(OPEN_TRADES)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    restore_open_trades()

    logger.info(
        "============================================"
    )

    logger.info(
        "🧠 MARKET BRAIN AI 2.0 STARTED"
    )

    logger.info(
        "20 COINS | 1D / 4H / 1H"
    )

    logger.info(
        "NO 5M | NO 12H REPORT"
    )

    logger.info(
        "Adaptive Learning ACTIVE"
    )

    logger.info(
        "Duplicate Protection ACTIVE"
    )

    logger.info(
        "Persistent Memory: %s",
        DATA_DIR
    )

    logger.info(
        "Pakistan Time: %s",
        now_pkt().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    logger.info(
        "============================================"
    )

    while True:

        try:

            # First check existing trades.
            check_open_trades()

            # Then search for new trades.
            scan_all()

        except KeyboardInterrupt:

            logger.info(
                "Stopped by user."
            )

            break

        except Exception as e:

            logger.exception(
                "Main loop error: %s",
                e
            )

        time.sleep(
            SCAN_SECONDS
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
