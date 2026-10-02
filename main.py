# ============================================================
# MARKET BRAIN AI — ADAPTIVE MULTI-TIMEFRAME + ANTI-LOSS ENGINE
# ============================================================
#
# 1D = Major Bias
# 4H = Structure / Liquidity / SMC / Pressure
# 1H = Setup / Entry / SL / TP
#
# CORE:
# - HH / HL / LH / LL
# - BOS / CHoCH
# - Liquidity Sweeps
# - Equal High / Equal Low
# - FVG
# - Order Block
# - Breaker
# - Premium / Discount
# - Buyer / Seller Pressure
# - Volume
# - EMA 20 / 50 / 200
# - RSI
# - ATR
# - Support / Resistance
# - Candlestick confirmation
# - Classical patterns
#
# ADAPTIVE LEARNING:
# - Indicator learning
# - Symbol learning
# - BUY / SELL learning
# - Combination learning
# - WIN learning
# - LOSS learning
# - Failure-pattern learning
# - Anti-loss filters
# - Sample-size protection
# - MAE / MFE
#
# IMPORTANT:
# This is a statistical adaptive engine.
# It does NOT guarantee profit or a specific win rate.
# It does NOT place real Binance orders.
# Trades are paper/signal trades.
# ============================================================

import os
import time
import json
import uuid
import logging
import smtplib

from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_URL = os.getenv(
    "BINANCE_BASE_URL",
    "https://data-api.binance.vision"
)

PKT = ZoneInfo("Asia/Karachi")

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "SUIUSDT", "OPUSDT", "ARBUSDT", "INJUSDT",
    "APTUSDT", "LTCUSDT", "TRXUSDT", "UNIUSDT", "ATOMUSDT"
]

TIMEFRAMES = {
    "1d": "1d",
    "4h": "4h",
    "1h": "1h"
}

CANDLE_LIMIT = 300
SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "60"))

# ENTRY
MIN_SCORE = float(os.getenv("MIN_SCORE", "65"))
DIRECTION_GAP = float(os.getenv("DIRECTION_GAP", "8"))
MAX_NEW_TRADES_PER_SCAN = int(
    os.getenv("MAX_NEW_TRADES_PER_SCAN", "3")
)
MAX_OPEN_TRADES = int(
    os.getenv("MAX_OPEN_TRADES", "8")
)

# RISK
MIN_RR = float(os.getenv("MIN_RR", "1.8"))
MAX_RR = float(os.getenv("MAX_RR", "4.5"))

ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.20"))
ATR_TARGET_MULT = float(os.getenv("ATR_TARGET_MULT", "2.80"))

MIN_SL_PCT = float(os.getenv("MIN_SL_PCT", "0.35"))
MAX_SL_PCT = float(os.getenv("MAX_SL_PCT", "3.00"))

MIN_TARGET_PCT = float(os.getenv("MIN_TARGET_PCT", "0.70"))
MAX_TARGET_PCT = float(os.getenv("MAX_TARGET_PCT", "12.0"))

# LEARNING
MIN_LEARNING_SAMPLES = int(
    os.getenv("MIN_LEARNING_SAMPLES", "5")
)

LEARNING_WIN_REWARD = float(
    os.getenv("LEARNING_WIN_REWARD", "0.035")
)

LEARNING_LOSS_PENALTY = float(
    os.getenv("LEARNING_LOSS_PENALTY", "0.025")
)

MIN_INDICATOR_WEIGHT = float(
    os.getenv("MIN_INDICATOR_WEIGHT", "0.70")
)

MAX_INDICATOR_WEIGHT = float(
    os.getenv("MAX_INDICATOR_WEIGHT", "1.35")
)

# FAILURE LEARNING
MIN_FAILURE_SAMPLES = int(
    os.getenv("MIN_FAILURE_SAMPLES", "5")
)

FAILURE_REJECT_RATE = float(
    os.getenv("FAILURE_REJECT_RATE", "0.70")
)

FAILURE_WARNING_RATE = float(
    os.getenv("FAILURE_WARNING_RATE", "0.55")
)

MAX_ANTI_LOSS_PENALTY = float(
    os.getenv("MAX_ANTI_LOSS_PENALTY", "22")
)


# ============================================================
# FILES
# ============================================================

DATA_DIR = Path(
    os.getenv("DATA_DIR", "market_brain_data")
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LEARNING_FILE = DATA_DIR / "ai_learning.json"
TRADE_MEMORY_FILE = DATA_DIR / "trade_memory.json"
TRADE_CSV_FILE = DATA_DIR / "trade_learning_log.csv"
STATE_FILE = DATA_DIR / "bot_state.json"


# ============================================================
# EMAIL
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
# HTTP
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "MARKET-BRAIN-AI/3.0",
    "Accept": "application/json"
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


def normalize_key(text):
    return str(text).upper().replace(" ", "_")


# ============================================================
# JSON
# ============================================================

def load_json(path, default):
    try:
        if path.exists():
            return json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
    except Exception as e:
        logger.warning(
            "Could not load %s: %s",
            path,
            e
        )

    return default


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

    url = f"{BASE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
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
                "ignore"
            ]

            df = pd.DataFrame(
                raw,
                columns=columns
            )

            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]:
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

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

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
            ).abs()
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

def pressure(df, length=12):

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
        d["close"] > d["open"]
    ].sum()

    sell = weighted[
        d["close"] < d["open"]
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
    left=4,
    right=4
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
                    float(
                        high_values[i]
                    )
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
                    float(
                        low_values[i]
                    )
                )
            )

    return highs, lows


def structure_info(df):

    highs, lows = pivots(
        df,
        4,
        4
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
        last_high is not None
        and previous_high is not None
        and last_high > previous_high
    )

    lh = (
        last_high is not None
        and previous_high is not None
        and last_high < previous_high
    )

    hl = (
        last_low is not None
        and previous_low is not None
        and last_low > previous_low
    )

    ll = (
        last_low is not None
        and previous_low is not None
        and last_low < previous_low
    )

    close = float(
        df["close"].iloc[-1]
    )

    previous_close = float(
        df["close"].iloc[-2]
    )

    bull_bos = (
        last_high is not None
        and close > last_high
        and previous_close <= last_high
    )

    bear_bos = (
        last_low is not None
        and close < last_low
        and previous_close >= last_low
    )

    previous_state = 0

    start = max(
        20,
        len(df) - 100
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
            4,
            4
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
        "lows": lows
    }


# ============================================================
# SMC
# ============================================================

def detect_smc(df, st):

    price = float(
        df["close"].iloc[-1]
    )

    bull_fvg = False
    bear_fvg = False

    bull_ob = False
    bear_ob = False

    bull_breaker = False
    bear_breaker = False

    bull_sweep = False
    bear_sweep = False

    if len(df) >= 4:

        bull_fvg = (
            float(df["low"].iloc[-1]) >
            float(df["high"].iloc[-4])
        )

        bear_fvg = (
            float(df["high"].iloc[-1]) <
            float(df["low"].iloc[-4])
        )

    if len(df) >= 2:

        current_open = float(
            df["open"].iloc[-1]
        )

        current_close = float(
            df["close"].iloc[-1]
        )

        previous_open = float(
            df["open"].iloc[-2]
        )

        previous_close = float(
            df["close"].iloc[-2]
        )

        previous_high = float(
            df["high"].iloc[-2]
        )

        previous_low = float(
            df["low"].iloc[-2]
        )

        bull_ob = (
            current_close > current_open
            and previous_close < previous_open
            and current_close > previous_high
        )

        bear_ob = (
            current_close < current_open
            and previous_close > previous_open
            and current_close < previous_low
        )

        bull_breaker = (
            previous_close < previous_open
            and current_close > previous_high
        )

        bear_breaker = (
            previous_close > previous_open
            and current_close < previous_low
        )

    if st["last_low"] is not None:

        bull_sweep = (
            float(df["low"].iloc[-1])
            < st["last_low"]
            and price > st["last_low"]
        )

    if st["last_high"] is not None:

        bear_sweep = (
            float(df["high"].iloc[-1])
            > st["last_high"]
            and price < st["last_high"]
        )

    recent = df.tail(50)

    range_high = float(
        recent["high"].max()
    )

    range_low = float(
        recent["low"].min()
    )

    equilibrium = (
        range_high +
        range_low
    ) / 2

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
        "premium": price > equilibrium
    }


# ============================================================
# CANDLE PATTERNS
# ============================================================

def candle_patterns(df):

    if len(df) < 2:
        return {}

    c = df.iloc[-1]
    p = df.iloc[-2]

    c_open = float(c["open"])
    c_close = float(c["close"])
    c_high = float(c["high"])
    c_low = float(c["low"])

    p_open = float(p["open"])
    p_close = float(p["close"])

    candle_range = max(
        c_high - c_low,
        1e-12
    )

    body = abs(
        c_close - c_open
    )

    upper_wick = (
        c_high -
        max(c_open, c_close)
    )

    lower_wick = (
        min(c_open, c_close) -
        c_low
    )

    body_ratio = (
        body /
        candle_range
    )

    bullish_engulfing = (
        c_close > c_open
        and p_close < p_open
        and c_close >= p_open
        and c_open <= p_close
)
    bearish_engulfing = (
        c_close < c_open
        and p_close > p_open
        and c_open >= p_close
        and c_close <= p_open
    )

    hammer = (
        lower_wick >= body * 2
        and upper_wick <= body
    )

    shooting_star = (
        upper_wick >= body * 2
        and lower_wick <= body
    )

    bullish_rejection = (
        lower_wick > body * 1.5
        and c_close > c_open
    )

    bearish_rejection = (
        upper_wick > body * 1.5
        and c_close < c_open
    )

    return {
        "bullish_engulfing": bullish_engulfing,
        "bearish_engulfing": bearish_engulfing,
        "hammer": hammer,
        "shooting_star": shooting_star,
        "bullish_rejection": bullish_rejection,
        "bearish_rejection": bearish_rejection,
        "body_ratio": body_ratio
    }


# ============================================================
# CLASSICAL PATTERNS
# ============================================================

def detect_patterns(df, st):

    patterns = {}

    highs = st["highs"]
    lows = st["lows"]

    equal_high = False
    equal_low = False

    if len(highs) >= 2:

        a = highs[-1][1]
        b = highs[-2][1]

        equal_high = (
            abs(a - b) /
            max(abs(b), 1e-12)
            <= 0.003
        )

    if len(lows) >= 2:

        a = lows[-1][1]
        b = lows[-2][1]

        equal_low = (
            abs(a - b) /
            max(abs(b), 1e-12)
            <= 0.003
        )

    patterns["equal_high"] = equal_high
    patterns["equal_low"] = equal_low

    recent_high = float(
        df["high"].tail(10).max()
    )

    recent_low = float(
        df["low"].tail(10).min()
    )

    close = float(
        df["close"].iloc[-1]
    )

    patterns["double_top"] = (
        equal_high
        and close < recent_high
    )

    patterns["double_bottom"] = (
        equal_low
        and close > recent_low
    )

    patterns["head_shoulders"] = (
        st["HH"] and st["LH"]
    )

    patterns["inverse_hs"] = (
        st["LL"] and st["HL"]
    )

    patterns["triple_top"] = (
        len(highs) >= 3
        and abs(
            highs[-1][1] -
            highs[-2][1]
        ) / max(
            abs(highs[-2][1]),
            1e-12
        ) <= 0.005
        and abs(
            highs[-2][1] -
            highs[-3][1]
        ) / max(
            abs(highs[-3][1]),
            1e-12
        ) <= 0.005
    )

    patterns["triple_bottom"] = (
        len(lows) >= 3
        and abs(
            lows[-1][1] -
            lows[-2][1]
        ) / max(
            abs(lows[-2][1]),
            1e-12
        ) <= 0.005
        and abs(
            lows[-2][1] -
            lows[-3][1]
        ) / max(
            abs(lows[-3][1]),
            1e-12
        ) <= 0.005
    )

    last_50 = df.tail(50)

    range_high = float(
        last_50["high"].max()
    )

    range_low = float(
        last_50["low"].min()
    )

    range_width = safe_div(
        range_high - range_low,
        max(abs(close), 1e-12)
    ) * 100

    patterns["range_market"] = (
        range_width < 8
    )

    previous_range_high = float(
        df["high"].iloc[-21:-1].max()
    )

    previous_range_low = float(
        df["low"].iloc[-21:-1].min()
    )

    patterns["range_breakout"] = (
        close > previous_range_high
    )

    patterns["range_breakdown"] = (
        close < previous_range_low
    )

    patterns["ascending_triangle"] = (
        equal_high and st["HL"]
    )

    patterns["descending_triangle"] = (
        equal_low and st["LH"]
    )

    patterns["symmetrical_triangle"] = (
        st["LH"] and st["HL"]
    )

    patterns["rising_wedge"] = (
        st["HH"] and st["HL"] and st["LH"]
    )

    patterns["falling_wedge"] = (
        st["LL"] and st["LH"] and st["HL"]
    )

    last_10 = df.tail(10)

    bull_flag = (
        float(last_10["close"].iloc[-1])
        >
        float(last_10["close"].iloc[0])
    )

    bear_flag = (
        float(last_10["close"].iloc[-1])
        <
        float(last_10["close"].iloc[0])
    )

    patterns["bull_flag"] = bull_flag
    patterns["bear_flag"] = bear_flag

    return patterns


# ============================================================
# TIMEFRAME CONTEXT
# ============================================================

def timeframe_context(df):

    x = add_indicators(df)

    price = float(
        x["close"].iloc[-1]
    )

    ema20 = float(
        x["ema20"].iloc[-1]
    )

    ema50 = float(
        x["ema50"].iloc[-1]
    )

    ema200 = float(
        x["ema200"].iloc[-1]
    )

    rsi_value = float(
        x["rsi"].iloc[-1]
    )

    volume_ratio = safe_float(
        x["volume_ratio"].iloc[-1],
        1.0
    )

    buy_pressure, sell_pressure = pressure(
        x,
        12
    )

    bull = (
        price > ema20
        and ema20 > ema50
    )

    bear = (
        price < ema20
        and ema20 < ema50
    )

    return {
        "df": x,
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi_value,
        "volume_ratio": volume_ratio,
        "buy_pressure": buy_pressure,
        "sell_pressure": sell_pressure,
        "bull": bull,
        "bear": bear
    }


# ============================================================
# DEFAULT LEARNING
# ============================================================

DEFAULT_LEARNING = {
    "global": {
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "net_r": 0.0
    },
    "symbols": {},
    "directions": {
        "BUY": {},
        "SELL": {}
    },
    "indicators": {},
    "combinations": {},
    "failure_patterns": {},
    "regimes": {},
    "anti_loss_rules": {}
}


def load_learning():

    data = load_json(
        LEARNING_FILE,
        DEFAULT_LEARNING.copy()
    )

    if not isinstance(data, dict):
        data = DEFAULT_LEARNING.copy()

    for key, value in DEFAULT_LEARNING.items():

        if key not in data:
            data[key] = value

    return data


LEARNING = load_learning()


# ============================================================
# LEARNING HELPERS
# ============================================================

def ensure_indicator(name):

    name = normalize_key(name)

    if name not in LEARNING["indicators"]:

        LEARNING["indicators"][name] = {
            "wins": 0,
            "losses": 0,
            "weight": 1.0
        }

    return LEARNING["indicators"][name]


def ensure_symbol(symbol):

    if symbol not in LEARNING["symbols"]:

        LEARNING["symbols"][symbol] = {
            "wins": 0,
            "losses": 0,
            "net_r": 0.0
        }

    return LEARNING["symbols"][symbol]


def ensure_direction(direction):

    if direction not in LEARNING["directions"]:

        LEARNING["directions"][direction] = {
            "wins": 0,
            "losses": 0,
            "net_r": 0.0
        }

    return LEARNING["directions"][direction]


def ensure_combination(key):

    key = normalize_key(key)

    if key not in LEARNING["combinations"]:

        LEARNING["combinations"][key] = {
            "wins": 0,
            "losses": 0,
            "weight": 1.0
        }

    return LEARNING["combinations"][key]


def ensure_failure(key):

    key = normalize_key(key)

    if key not in LEARNING["failure_patterns"]:

        LEARNING["failure_patterns"][key] = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "failure_rate": 0.0
        }

    return LEARNING["failure_patterns"][key]


# ============================================================
# FAILURE FEATURES
# ============================================================

def build_failure_features(
    symbol,
    direction,
    d1,
    h4,
    h1,
    score_result,
    levels
):

    features = []

    def add(condition, name):

        if condition:
            features.append(
                normalize_key(name)
            )

    add(
        direction == "BUY"
        and not d1["bull"],
        "HTF_1D_AGAINST_BUY"
    )

    add(
        direction == "SELL"
        and not d1["bear"],
        "HTF_1D_AGAINST_SELL"
    )

    add(
        direction == "BUY"
        and not h4["bull"],
        "HTF_4H_AGAINST_BUY"
    )

    add(
        direction == "SELL"
        and not h4["bear"],
        "HTF_4H_AGAINST_SELL"
    )

    add(
        direction == "BUY"
        and not h1["bull"],
        "ENTRY_TREND_NOT_BULL"
    )

    add(
        direction == "SELL"
        and not h1["bear"],
        "ENTRY_TREND_NOT_BEAR"
    )

    add(
        direction == "BUY"
        and h1["sell_pressure"] >= 54,
        "SELL_PRESSURE_AGAINST_BUY"
    )

    add(
        direction == "SELL"
        and h1["buy_pressure"] >= 54,
        "BUY_PRESSURE_AGAINST_SELL"
    )

    add(
        h1["volume_ratio"] < 1.0,
        "WEAK_VOLUME"
    )

    add(
        direction == "BUY"
        and h1["rsi"] >= 70,
        "BUY_OVEREXTENDED_RSI"
    )

    add(
        direction == "SELL"
        and h1["rsi"] <= 30,
        "SELL_OVEREXTENDED_RSI"
    )

    st = score_result["structure"]
    smc = score_result["smc"]

    add(
        direction == "BUY"
        and st["bear_choch"],
        "BEARISH_CHOCH_AGAINST_BUY"
    )

    add(
        direction == "SELL"
        and st["bull_choch"],
        "BULLISH_CHOCH_AGAINST_SELL"
    )

    add(
        direction == "BUY"
        and smc["bear_sweep"],
        "BEARISH_SWEEP_AGAINST_BUY"
    )

    add(
        direction == "SELL"
        and smc["bull_sweep"],
        "BULLISH_SWEEP_AGAINST_SELL"
    )

    add(
        direction == "BUY"
        and smc["premium"],
        "BUY_IN_PREMIUM"
    )

    add(
        direction == "SELL"
        and smc["discount"],
        "SELL_IN_DISCOUNT"
    )

    risk_pct = safe_float(
        levels.get("risk_pct"),
        0
    )

    add(
        risk_pct > 2.0,
        "HIGH_RISK_DISTANCE"
    )

    return sorted(
        set(features)
    )


# ============================================================
# FAILURE ANALYSIS
# ============================================================

def failure_risk(
    failure_features
):

    penalty = 0.0
    warnings = []
    reject_reasons = []

    for feature in failure_features:

        record = ensure_failure(
            feature
        )

        trades = int(
            record.get("trades", 0)
        )

        losses = int(
            record.get("losses", 0)
        )

        wins = int(
            record.get("wins", 0)
        )

        if trades < MIN_FAILURE_SAMPLES:
            continue

        failure_rate = safe_div(
            losses,
            trades
        )

        if failure_rate >= FAILURE_REJECT_RATE:

            penalty += min(
                8.0,
                MAX_ANTI_LOSS_PENALTY
            )

            reject_reasons.append(
                f"{feature}:{failure_rate:.0%}"
            )

        elif failure_rate >= FAILURE_WARNING_RATE:

            penalty += 3.0

            warnings.append(
                f"{feature}:{failure_rate:.0%}"
            )

    penalty = min(
        penalty,
        MAX_ANTI_LOSS_PENALTY
    )

    return {
        "penalty": penalty,
        "warnings": warnings,
        "reject_reasons": reject_reasons
    }
    # ============================================================
# SCORE ENGINE
# ============================================================

def score_market(
    symbol,
    d1,
    h4,
    h1
):

    buy_score = 0.0
    sell_score = 0.0

    buy_reasons = []
    sell_reasons = []

    buy_indicators = []
    sell_indicators = []

    def add(
        side,
        points,
        reason,
        indicator=None
    ):

        nonlocal buy_score
        nonlocal sell_score

        if indicator:

            record = ensure_indicator(
                indicator
            )

            weight = safe_float(
                record.get("weight"),
                1.0
            )

        else:

            weight = 1.0

        value = points * weight

        if side == "BUY":

            buy_score += value
            buy_reasons.append(
                reason
            )

            if indicator:
                buy_indicators.append(
                    normalize_key(
                        indicator
                    )
                )

        else:

            sell_score += value
            sell_reasons.append(
                reason
            )

            if indicator:
                sell_indicators.append(
                    normalize_key(
                        indicator
                    )
                )

    # --------------------------------------------------------
    # 1D
    # --------------------------------------------------------

    if d1["bull"]:

        add(
            "BUY",
            7,
            "1D BULLISH",
            "1D_TREND"
        )

    if d1["bear"]:

        add(
            "SELL",
            7,
            "1D BEARISH",
            "1D_TREND"
        )

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    if h4["bull"]:

        add(
            "BUY",
            9,
            "4H BULLISH",
            "4H_TREND"
        )

    if h4["bear"]:

        add(
            "SELL",
            9,
            "4H BEARISH",
            "4H_TREND"
        )

    # --------------------------------------------------------
    # 1H
    # --------------------------------------------------------

    if h1["bull"]:

        add(
            "BUY",
            10,
            "1H BULLISH",
            "1H_TREND"
        )

    if h1["bear"]:

        add(
            "SELL",
            10,
            "1H BEARISH",
            "1H_TREND"
        )

    # --------------------------------------------------------
    # PRESSURE
    # --------------------------------------------------------

    if h1["buy_pressure"] >= 54:

        add(
            "BUY",
            8,
            "BUYER PRESSURE",
            "PRESSURE"
        )

    if h1["sell_pressure"] >= 54:

        add(
            "SELL",
            8,
            "SELLER PRESSURE",
            "PRESSURE"
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if h1["volume_ratio"] >= 1.05:

        if h1["price"] > h1["ema20"]:

            add(
                "BUY",
                6,
                "VOLUME CONFIRMS BUY",
                "VOLUME"
            )

        if h1["price"] < h1["ema20"]:

            add(
                "SELL",
                6,
                "VOLUME CONFIRMS SELL",
                "VOLUME"
            )

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    st = structure_info(
        h1["df"]
    )

    if st["HH"] and st["HL"]:

        add(
            "BUY",
            8,
            "HH + HL",
            "STRUCTURE"
        )

    if st["LH"] and st["LL"]:

        add(
            "SELL",
            8,
            "LH + LL",
            "STRUCTURE"
        )

    if st["bull_bos"]:

        add(
            "BUY",
            12,
            "BULLISH BOS",
            "BOS"
        )

    if st["bear_bos"]:

        add(
            "SELL",
            12,
            "BEARISH BOS",
            "BOS"
        )

    if st["bull_choch"]:

        add(
            "BUY",
            10,
            "BULLISH CHOCH",
            "CHOCH"
        )

    if st["bear_choch"]:

        add(
            "SELL",
            10,
            "BEARISH CHOCH",
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
            "BULLISH LIQUIDITY SWEEP",
            "LIQUIDITY_SWEEP"
        )

    if smc["bear_sweep"]:

        add(
            "SELL",
            10,
            "BEARISH LIQUIDITY SWEEP",
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
            "PREMIUM",
            "PREMIUM"
        )

    if smc["bull_fvg"]:

        add(
            "BUY",
            7,
            "BULLISH FVG",
            "FVG"
        )

    if smc["bear_fvg"]:

        add(
            "SELL",
            7,
            "BEARISH FVG",
            "FVG"
        )

    if smc["bull_ob"]:

        add(
            "BUY",
            7,
            "BULLISH ORDER BLOCK",
            "ORDER_BLOCK"
        )

    if smc["bear_ob"]:

        add(
            "SELL",
            7,
            "BEARISH ORDER BLOCK",
            "ORDER_BLOCK"
        )

    if smc["bull_breaker"]:

        add(
            "BUY",
            5,
            "BULLISH BREAKER",
            "BREAKER"
        )

    if smc["bear_breaker"]:

        add(
            "SELL",
            5,
            "BEARISH BREAKER",
            "BREAKER"
        )

    # --------------------------------------------------------
    # CANDLES
    # --------------------------------------------------------

    candles = candle_patterns(
        h1["df"]
    )

    if (
        candles.get("bullish_engulfing")
        or candles.get("hammer")
        or candles.get("bullish_rejection")
    ):

        add(
            "BUY",
            5,
            "BULLISH PRICE ACTION",
            "CANDLE"
        )

    if (
        candles.get("bearish_engulfing")
        or candles.get("shooting_star")
        or candles.get("bearish_rejection")
    ):

        add(
            "SELL",
            5,
            "BEARISH PRICE ACTION",
            "CANDLE"
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if (
        45 <= h1["rsi"] <= 65
        and h1["bull"]
    ):

        add(
            "BUY",
            4,
            "BUY RSI CONTEXT",
            "RSI"
        )

    if (
        35 <= h1["rsi"] <= 55
        and h1["bear"]
    ):

        add(
            "SELL",
            4,
            "SELL RSI CONTEXT",
            "RSI"
        )

    # --------------------------------------------------------
    # PATTERNS
    # --------------------------------------------------------

    patterns = detect_patterns(
        h1["df"],
        st
    )

    pattern_points = {
        "double_bottom": (
            "BUY",
            9,
            "DOUBLE BOTTOM"
        ),
        "double_top": (
            "SELL",
            9,
            "DOUBLE TOP"
        ),
        "inverse_hs": (
            "BUY",
            8,
            "INVERSE HEAD & SHOULDERS"
        ),
        "head_shoulders": (
            "SELL",
            8,
            "HEAD & SHOULDERS"
        ),
        "triple_bottom": (
            "BUY",
            7,
            "TRIPLE BOTTOM"
        ),
        "triple_top": (
            "SELL",
            7,
            "TRIPLE TOP"
        ),
        "ascending_triangle": (
            "BUY",
            6,
            "ASCENDING TRIANGLE"
        ),
        "descending_triangle": (
            "SELL",
            6,
            "DESCENDING TRIANGLE"
        ),
        "falling_wedge": (
            "BUY",
            6,
            "FALLING WEDGE"
        ),
        "rising_wedge": (
            "SELL",
            6,
            "RISING WEDGE"
        ),
        "bull_flag": (
            "BUY",
            5,
            "BULL FLAG"
        ),
        "bear_flag": (
            "SELL",
            5,
            "BEAR FLAG"
        ),
        "range_breakout": (
            "BUY",
            7,
            "RANGE BREAKOUT"
        ),
        "range_breakdown": (
            "SELL",
            7,
            "RANGE BREAKDOWN"
        )
    }

    for name, data in pattern_points.items():

        if patterns.get(name):

            add(
                data[0],
                data[1],
                data[2],
                name
            )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    price = h1["price"]

    supports = [
        value
        for _, value in st["lows"]
        if value < price
    ]

    resistances = [
        value
        for _, value in st["highs"]
        if value > price
    ]

    support = (
        max(supports)
        if supports
        else float(
            h1["df"]["low"].tail(20).min()
        )
    )

    resistance = (
        min(resistances)
        if resistances
        else float(
            h1["df"]["high"].tail(20).max()
        )
    )

    atr_value = max(
        safe_float(
            h1["df"]["atr"].iloc[-1]
        ),
        price * 0.001
    )

    support_distance = safe_div(
        price - support,
        price
    ) * 100

    resistance_distance = safe_div(
        resistance - price,
        price
    ) * 100

    if support_distance <= 1.5:

        add(
            "BUY",
            7,
            "NEAR SUPPORT",
            "SUPPORT"
        )

    if resistance_distance <= 1.5:

        add(
            "SELL",
            7,
            "NEAR RESISTANCE",
            "RESISTANCE"
        )

    # --------------------------------------------------------
    # CORRELATION CONTROL
    # --------------------------------------------------------

    buy_score = min(
        buy_score,
        100
    )

    sell_score = min(
        sell_score,
        100
    )

    # --------------------------------------------------------
    # COMBINATION
    # --------------------------------------------------------

    buy_indicators = sorted(
        set(buy_indicators)
    )

    sell_indicators = sorted(
        set(sell_indicators)
    )

    buy_combo = "|".join(
        buy_indicators[:8]
    )

    sell_combo = "|".join(
        sell_indicators[:8]
    )

    if buy_combo:

        combo = ensure_combination(
            buy_combo
        )

        if (
            combo["wins"] +
            combo["losses"]
        ) >= MIN_LEARNING_SAMPLES:

            buy_score *= clamp(
                safe_float(
                    combo.get("weight"),
                    1.0
                ),
                0.80,
                1.15
            )

    if sell_combo:

        combo = ensure_combination(
            sell_combo
        )

        if (
            combo["wins"] +
            combo["losses"]
        ) >= MIN_LEARNING_SAMPLES:

            sell_score *= clamp(
                safe_float(
                    combo.get("weight"),
                    1.0
                ),
                0.80,
                1.15
            )

    buy_score = clamp(
        buy_score,
        0,
        100
    )

    sell_score = clamp(
        sell_score,
        0,
        100
    )

    if (
        buy_score >= MIN_SCORE
        and buy_score >=
        sell_score + DIRECTION_GAP
    ):

        direction = "BUY"

    elif (
        sell_score >= MIN_SCORE
        and sell_score >=
        buy_score + DIRECTION_GAP
    ):

        direction = "SELL"

    else:

        direction = "NO TRADE"

    return {
        "symbol": symbol,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "direction": direction,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons,
        "buy_indicators": buy_indicators,
        "sell_indicators": sell_indicators,
        "setups": sorted(
            set(
                list(patterns.keys())
            )
        ),
        "structure": st,
        "smc": smc,
        "patterns": patterns,
        "candles": candles,
        "support": support,
        "resistance": resistance,
        "atr": atr_value,
        "pressure": {
            "buy": h1["buy_pressure"],
            "sell": h1["sell_pressure"]
        }
    }
    # ============================================================
# DYNAMIC LEVELS
# ============================================================

def dynamic_levels(
    direction,
    h1,
    score
):

    price = float(
        h1["price"]
    )

    atr_value = max(
        safe_float(h1["df"]["atr"].iloc[-1]),
        price * 0.001
    )

    support = safe_float(
        score["support"],
        price * 0.99
    )

    resistance = safe_float(
        score["resistance"],
        price * 1.01
    )

    if direction == "BUY":

        structural_sl = (
            support -
            0.20 * atr_value
        )

        atr_sl = (
            price -
            ATR_SL_MULT * atr_value
        )

        stop = min(
            structural_sl,
            atr_sl
        )

        min_stop = (
            price *
            (1 - MAX_SL_PCT / 100)
        )

        max_stop = (
            price *
            (1 - MIN_SL_PCT / 100)
        )

        stop = clamp(
            stop,
            min_stop,
            max_stop
        )

        risk = price - stop

        structural_target = resistance

        atr_target = (
            price +
            ATR_TARGET_MULT *
            atr_value
        )

        target = max(
            structural_target,
            atr_target
        )

        min_target = (
            price +
            risk * MIN_RR
        )

        max_target = (
            price *
            (1 + MAX_TARGET_PCT / 100)
        )

        target = max(
            target,
            min_target
        )

        target = min(
            target,
            max_target
        )

    else:

        structural_sl = (
            resistance +
            0.20 * atr_value
        )

        atr_sl = (
            price +
            ATR_SL_MULT * atr_value
        )

        stop = max(
            structural_sl,
            atr_sl
        )

        min_stop = (
            price *
            (1 + MIN_SL_PCT / 100)
        )

        max_stop = (
            price *
            (1 + MAX_SL_PCT / 100)
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
            ATR_TARGET_MULT *
            atr_value
        )

        target = min(
            structural_target,
            atr_target
        )

        min_target = (
            price -
            risk * MIN_RR
        )

        max_target = (
            price *
            (1 - MAX_TARGET_PCT / 100)
        )

        target = min(
            target,
            min_target
        )

        target = max(
            target,
            max_target
        )

    if risk <= 0:
        return {
            "valid": False
        }

    reward = (
        target - price
        if direction == "BUY"
        else price - target
    )

    risk_pct = safe_div(
        risk,
        price
    ) * 100

    reward_pct = safe_div(
        reward,
        price
    ) * 100

    rr = safe_div(
        reward,
        risk
    )

    valid = (
        MIN_RR <= rr <= MAX_RR
        and MIN_SL_PCT <= risk_pct <= MAX_SL_PCT
        and MIN_TARGET_PCT <= reward_pct <= MAX_TARGET_PCT
        and target != price
        and stop != price
    )

    return {
        "valid": valid,
        "entry": price,
        "stop": stop,
        "target": target,
        "risk": risk,
        "reward": reward,
        "rr": rr,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct
    }


# ============================================================
# TRADE STORAGE
# ============================================================

TRADES = load_json(
    TRADE_MEMORY_FILE,
    []
)

STATE = load_json(
    STATE_FILE,
    {}
)

OPEN_TRADES = STATE.get(
    "open_trades",
    []
)

EMAIL_SENT = STATE.get(
    "email_sent",
    {}
)


# ============================================================
# STATE
# ============================================================

def save_state():

    save_json(
        STATE_FILE,
        {
            "open_trades": OPEN_TRADES,
            "email_sent": EMAIL_SENT,
            "updated_at": iso_pkt()
        }
    )


# ============================================================
# CSV
# ============================================================

def save_trade_csv(trade):

    row = pd.DataFrame(
        [trade]
    )

    if TRADE_CSV_FILE.exists():

        row.to_csv(
            TRADE_CSV_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        row.to_csv(
            TRADE_CSV_FILE,
            index=False
        )


# ============================================================
# TRADE SAVE
# ============================================================

def save_closed_trade(trade):

    global TRADES

    signal_id = trade.get(
        "signal_id"
    )

    result = trade.get(
        "result"
    )

    for old in TRADES:

        if (
            old.get("signal_id") ==
            signal_id
            and old.get("result") ==
            result
        ):

            return

    TRADES.append(
        trade
    )

    save_json(
        TRADE_MEMORY_FILE,
        TRADES
    )

    try:
        save_trade_csv(
            trade
        )
    except Exception as e:
        logger.warning(
            "CSV error: %s",
            e
        )


# ============================================================
# LEARNING UPDATE
# ============================================================

def update_learning_from_trade(
    trade
):

    global LEARNING

    result = trade.get(
        "result"
    )

    symbol = trade.get(
        "symbol"
    )

    direction = trade.get(
        "direction"
    )

    r_value = safe_float(
        trade.get("r"),
        0
    )

    indicators = trade.get(
        "learning_indicators",
        []
    )

    failure_features = trade.get(
        "failure_features",
        []
    )

    combination = trade.get(
        "learning_combination",
        ""
    )

    # --------------------------------------------------------
    # GLOBAL
    # --------------------------------------------------------

    if result == "WIN":

        LEARNING["global"]["wins"] += 1

    elif result == "LOSS":

        LEARNING["global"]["losses"] += 1

    total = (
        LEARNING["global"]["wins"] +
        LEARNING["global"]["losses"]
    )

    LEARNING["global"]["win_rate"] = (
        safe_div(
            LEARNING["global"]["wins"],
            total
        ) * 100
    )

    LEARNING["global"]["net_r"] += r_value

    # --------------------------------------------------------
    # SYMBOL
    # --------------------------------------------------------

    symbol_record = ensure_symbol(
        symbol
    )

    if result == "WIN":
        symbol_record["wins"] += 1

    elif result == "LOSS":
        symbol_record["losses"] += 1

    symbol_record["net_r"] += r_value

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    direction_record = ensure_direction(
        direction
    )

    if result == "WIN":
        direction_record["wins"] += 1

    elif result == "LOSS":
        direction_record["losses"] += 1

    direction_record["net_r"] += r_value

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    for indicator in set(
        indicators
    ):

        record = ensure_indicator(
            indicator
        )

        if result == "WIN":

            record["wins"] += 1

            if (
                record["wins"] +
                record["losses"]
            ) >= MIN_LEARNING_SAMPLES:

                record["weight"] *= (
                    1 +
                    LEARNING_WIN_REWARD
                )

        elif result == "LOSS":

            record["losses"] += 1

            if (
                record["wins"] +
                record["losses"]
            ) >= MIN_LEARNING_SAMPLES:

                record["weight"] *= (
                    1 -
                    LEARNING_LOSS_PENALTY
                )

        record["weight"] = clamp(
            record["weight"],
            MIN_INDICATOR_WEIGHT,
            MAX_INDICATOR_WEIGHT
        )

    # --------------------------------------------------------
    # COMBINATION
    # --------------------------------------------------------

    if combination:

        combo = ensure_combination(
            combination
        )

        if result == "WIN":

            combo["wins"] += 1

            if (
                combo["wins"] +
                combo["losses"]
            ) >= MIN_LEARNING_SAMPLES:

                combo["weight"] *= 1.03

        elif result == "LOSS":

            combo["losses"] += 1

            if (
                combo["wins"] +
                combo["losses"]
            ) >= MIN_LEARNING_SAMPLES:

                combo["weight"] *= 0.97

        combo["weight"] = clamp(
            combo["weight"],
            0.70,
            1.30
        )

    # --------------------------------------------------------
    # FAILURE PATTERNS
    # --------------------------------------------------------

    for feature in set(
        failure_features
    ):

        record = ensure_failure(
            feature
        )

        record["trades"] += 1

        if result == "WIN":
            record["wins"] += 1

        elif result == "LOSS":
            record["losses"] += 1

        record["failure_rate"] = safe_div(
            record["losses"],
            record["trades"]
        )

    # --------------------------------------------------------
    # ANTI-LOSS RULES
    # --------------------------------------------------------

    if result == "LOSS":

        for feature in set(
            failure_features
        ):

            record = ensure_failure(
                feature
            )

            if (
                record["trades"] >=
                MIN_FAILURE_SAMPLES
            ):

                if (
                    record["failure_rate"] >=
                    FAILURE_REJECT_RATE
                ):

                    LEARNING[
                        "anti_loss_rules"
                    ][feature] = {
                        "enabled": True,
                        "failure_rate":
                            record[
                                "failure_rate"
                            ],
                        "trades":
                            record["trades"]
                    }

    elif result == "WIN":

        for feature in set(
            failure_features
        ):

            record = ensure_failure(
                feature
            )

            if (
                record["trades"] >=
                MIN_FAILURE_SAMPLES
                and
                record["failure_rate"] <
                FAILURE_WARNING_RATE
            ):

                LEARNING[
                    "anti_loss_rules"
                ].pop(
                    feature,
                    None
                )

    save_json(
        LEARNING_FILE,
        LEARNING
    )
    # ============================================================
# TRADE HEALTH / FAILURE DETECTION
# ============================================================

def evaluate_trade_health(
    trade,
    h1
):

    direction = trade[
        "direction"
    ]

    df = h1["df"]

    st = structure_info(
        df
    )

    smc = detect_smc(
        df,
        st
    )

    score = 100.0

    warnings = []

    critical = []

    if direction == "BUY":

        if not h1["bull"]:

            score -= 15
            warnings.append(
                "1H BULL TREND LOST"
            )

        if h1["sell_pressure"] >= 55:

            score -= 15
            warnings.append(
                "SELLER PRESSURE DOMINANT"
            )

        if st["bear_choch"]:

            score -= 25
            critical.append(
                "BEARISH CHOCH"
            )

        if st["bear_bos"]:

            score -= 30
            critical.append(
                "BEARISH BOS"
            )

        if (
            float(df["close"].iloc[-1])
            <
            float(df["ema20"].iloc[-1])
        ):

            score -= 10
            warnings.append(
                "PRICE BELOW EMA20"
            )

        if (
            float(df["close"].iloc[-1])
            <
            float(df["ema50"].iloc[-1])
        ):

            score -= 10
            warnings.append(
                "PRICE BELOW EMA50"
            )

        if smc["bear_sweep"]:

            score -= 12
            warnings.append(
                "BEARISH LIQUIDITY SWEEP"
            )

    else:

        if not h1["bear"]:

            score -= 15
            warnings.append(
                "1H BEAR TREND LOST"
            )

        if h1["buy_pressure"] >= 55:

            score -= 15
            warnings.append(
                "BUYER PRESSURE DOMINANT"
            )

        if st["bull_choch"]:

            score -= 25
            critical.append(
                "BULLISH CHOCH"
            )

        if st["bull_bos"]:

            score -= 30
            critical.append(
                "BULLISH BOS"
            )

        if (
            float(df["close"].iloc[-1])
            >
            float(df["ema20"].iloc[-1])
        ):

            score -= 10
            warnings.append(
                "PRICE ABOVE EMA20"
            )

        if (
            float(df["close"].iloc[-1])
            >
            float(df["ema50"].iloc[-1])
        ):

            score -= 10
            warnings.append(
                "PRICE ABOVE EMA50"
            )

        if smc["bull_sweep"]:

            score -= 12
            warnings.append(
                "BULLISH LIQUIDITY SWEEP"
            )

    score = clamp(
        score,
        0,
        100
    )

    if score >= 70:

        status = "HEALTHY"

    elif score >= 45:

        status = "WEAKENING"

    elif score >= 25:

        status = "HIGH LOSS RISK"

    else:

        status = "CRITICAL"

    return {
        "health_score": score,
        "status": status,
        "warnings": warnings,
        "critical": critical,
        "structure": st,
        "smc": smc,
        "checked_at": iso_pkt()
    }


# ============================================================
# OPEN TRADE
# ============================================================

def symbol_has_open_trade(
    symbol
):

    return any(
        t.get("symbol") == symbol
        and t.get("result") == "OPEN"
        for t in OPEN_TRADES
    )


def signal_already_used(
    signal_id
):

    for t in TRADES:

        if t.get(
            "signal_id"
        ) == signal_id:

            return True

    for t in OPEN_TRADES:

        if t.get(
            "signal_id"
        ) == signal_id:

            return True

    return False


def open_trade(
    result
):

    global OPEN_TRADES
    global EMAIL_SENT

    if result["direction"] not in (
        "BUY",
        "SELL"
    ):
        return False

    if len(OPEN_TRADES) >= MAX_OPEN_TRADES:
        return False

    symbol = result[
        "symbol"
    ]

    if symbol_has_open_trade(
        symbol
    ):
        return False

    signal_id = result[
        "signal_id"
    ]

    if signal_already_used(
        signal_id
    ):
        return False

    direction = result[
        "direction"
    ]

    levels = result[
        "levels"
    ]

    if not levels.get(
        "valid",
        False
    ):
        return False

    failure_features = result[
        "failure_features"
    ]

    anti_loss = result[
        "anti_loss"
    ]

    if anti_loss[
        "reject_reasons"
    ]:

        logger.info(
            "%s %s rejected by anti-loss: %s",
            symbol,
            direction,
            anti_loss[
                "reject_reasons"
            ]
        )

        return False

    if direction == "BUY":

        selected_indicators = (
            result["buy_indicators"]
        )

    else:

        selected_indicators = (
            result["sell_indicators"]
        )

    combination = "|".join(
        selected_indicators[:8]
    )

    trade = {
        "trade_id": str(
            uuid.uuid4()
        ),

        "signal_id": signal_id,

        "symbol": symbol,

        "direction": direction,

        "result": "OPEN",

        "entry": levels["entry"],
        "stop": levels["stop"],
        "target": levels["target"],

        "risk": levels["risk"],
        "reward": levels["reward"],
        "rr": levels["rr"],

        "risk_pct": levels["risk_pct"],
        "reward_pct": levels["reward_pct"],

        "open_time": iso_pkt(),

        "signal_candle":
            result["signal_candle"],

        "buy_score":
            result["buy_score"],

        "sell_score":
            result["sell_score"],

        "d1_bias":
            result["d1_bias"],

        "h4_bias":
            result["h4_bias"],

        "h1_bias":
            result["h1_bias"],

        "pressure_buy":
            result["pressure_buy"],

        "pressure_sell":
            result["pressure_sell"],

        "volume_ratio":
            result["volume_ratio"],

        "rsi":
            result["rsi"],

        "setups":
            result["setups"],

        "learning_indicators":
            selected_indicators,

        "learning_combination":
            combination,

        "failure_features":
            failure_features,

        "best_price":
            levels["entry"],

        "worst_price":
            levels["entry"],

        "mae": 0.0,

        "mfe": 0.0,

        "last_health":
            result["health"],

        "last_checked_candle":
            result["signal_candle"]
    }

    OPEN_TRADES.append(
        trade
    )

    save_state()

    email_key = (
        "OPEN|" +
        signal_id
    )

    if not EMAIL_SENT.get(
        email_key
    ):

        body = f"""
MARKET BRAIN AI — NEW TRADE

Symbol: {symbol}
Direction: {direction}

Entry: {levels["entry"]:.8f}
SL: {levels["stop"]:.8f}
TP: {levels["target"]:.8f}

Risk: {levels["risk_pct"]:.2f}%
Reward: {levels["reward_pct"]:.2f}%
RR: {levels["rr"]:.2f}

BUY Score: {result["buy_score"]:.2f}
SELL Score: {result["sell_score"]:.2f}

1D: {result["d1_bias"]}
4H: {result["h4_bias"]}
1H: {result["h1_bias"]}

Buyer Pressure:
{result["pressure_buy"]:.2f}%

Seller Pressure:
{result["pressure_sell"]:.2f}%

Volume:
{result["volume_ratio"]:.2f}x

RSI:
{result["rsi"]:.2f}

Setups:
{", ".join(result["setups"])}

Learning Indicators:
{", ".join(selected_indicators)}

Anti-Loss Warnings:
{", ".join(anti_loss["warnings"]) or "None"}

Failure Features:
{", ".join(failure_features) or "None"}

Trade Health:
{result["health"]["health_score"]:.1f}/100

Signal Candle:
{result["signal_candle"]}
"""

        sent = send_email(
            f"MARKET BRAIN NEW TRADE | {symbol} | {direction}",
            body
        )

        if sent:

            EMAIL_SENT[
                email_key
            ] = iso_pkt()

            save_state()

    logger.info(
        "OPEN %s %s score %.1f / %.1f",
        symbol,
        direction,
        result["buy_score"],
        result["sell_score"]
    )

    return True
    # ============================================================
# CLOSE TRADE
# ============================================================

def close_trade(
    trade,
    exit_price,
    reason,
    candle
):

    global OPEN_TRADES

    direction = trade[
        "direction"
    ]

    entry = safe_float(
        trade["entry"]
    )

    risk = max(
    safe_float(
     trade.get("risk", 1e-12)
        ),
        1e-12
        )

    if direction == "BUY":

        pnl = (
            exit_price -
            entry
        )

    else:

        pnl = (
            entry -
            exit_price
        )

    r_value = safe_div(
        pnl,
        risk
    )

    if r_value > 0:

        result = "WIN"

    else:

        result = "LOSS"

    trade["result"] = result

    trade["exit"] = exit_price

    trade["exit_time"] = iso_pkt()

    trade["exit_reason"] = reason

    trade["pnl"] = pnl

    trade["r"] = r_value

    trade["last_checked_candle"] = (
        candle
    )

    trade["close_health"] = (
        evaluate_trade_health(
            trade,
            timeframe_context(
                fetch_klines(
                    trade["symbol"],
                    "1h",
                    100
                )
            )
        )
        if False
        else trade.get(
            "last_health",
            {}
        )
    )

    update_learning_from_trade(
        trade
    )

    save_closed_trade(
        trade
    )

    OPEN_TRADES = [
        t for t in OPEN_TRADES
        if t.get(
            "trade_id"
        ) != trade.get(
            "trade_id"
        )
    ]

    save_state()

    email_key = (
        "CLOSED|" +
        trade["signal_id"]
    )

    if not EMAIL_SENT.get(
        email_key
    ):

        body = f"""
MARKET BRAIN AI — TRADE CLOSED

Symbol: {trade["symbol"]}
Direction: {direction}

Result: {result}

Entry: {entry:.8f}
Exit: {exit_price:.8f}

SL: {trade["stop"]:.8f}
TP: {trade["target"]:.8f}

Reason:
{reason}

P/L:
{pnl:.8f}

R:
{r_value:.2f}

MAE:
{trade.get("mae", 0):.2f}%

MFE:
{trade.get("mfe", 0):.2f}%

Failure Features:
{", ".join(trade.get("failure_features", []))}

Learning Indicators:
{", ".join(trade.get("learning_indicators", []))}
"""

        sent = send_email(
            (
                f"MARKET BRAIN CLOSED | "
                f"{trade['symbol']} | "
                f"{result}"
            ),
            body
        )

        if sent:

            EMAIL_SENT[
                email_key
            ] = iso_pkt()

            save_state()

    logger.info(
        "CLOSED %s %s -> %s | R %.2f",
        trade["symbol"],
        direction,
        result,
        r_value
    )


# ============================================================
# OPEN TRADE MONITOR
# ============================================================

def check_open_trades():

    if not OPEN_TRADES:
        return

    for trade in list(
        OPEN_TRADES
    ):

        symbol = trade[
            "symbol"
        ]

        df = fetch_klines(
            symbol,
            "1h",
            100
        )

        if df.empty:
            continue

        h1 = timeframe_context(
            df
        )

        candle = df.iloc[-1]

        candle_time = str(
            candle["open_time"]
        )

        close = float(
            candle["close"]
        )

        high = float(
            candle["high"]
        )

        low = float(
            candle["low"]
        )

        entry = float(
            trade["entry"]
        )

        # ----------------------------------------------------
        # MFE / MAE
        # ----------------------------------------------------

        if trade["direction"] == "BUY":

            favorable = (
                high - entry
            )

            adverse = (
                entry - low
            )

        else:

            favorable = (
                entry - low
            )

            adverse = (
                high - entry
            )

        mfe = safe_div(
            favorable,
            entry
        ) * 100

        mae = safe_div(
            adverse,
            entry
        ) * 100

        trade["mfe"] = max(
            safe_float(
                trade.get("mfe")
            ),
            mfe
        )

        trade["mae"] = max(
            safe_float(
                trade.get("mae")
            ),
            mae
        )

        # ----------------------------------------------------
        # TRADE HEALTH
        # ----------------------------------------------------

        health = evaluate_trade_health(
            trade,
            h1
        )

        trade["last_health"] = health

        trade[
            "last_checked_candle"
        ] = candle_time

         # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        stop = 0.0
        stop_val = trade.get("stop")
        if stop_val is not None:
            stop = float(stop_val)

        target = 0.0
        target_val = trade.get("target")
        if target_val is not None:
            target = float(target_val)

        hit_stop = False
        hit_target = False

        if trade["direction"] == "BUY":
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            hit_stop = high >= stop
            hit_target = low <= target

        if hit_stop and hit_target:
            close_trade(
                trade,
                stop,
                "SL_AND_TP_SAME_CANDLE_SL_FIRST",
                candle_time
            )
            continue

        if hit_stop:
            close_trade(
                trade,
                stop,
                "STOP_LOSS",
                candle_time
            )
            continue

        if hit_target:
            close_trade(
                trade,
                target,
                "TAKE_PROFIT",
                candle_time
            )
            continue

        # ----------------------------------------------------
        # HEALTH WARNING
        # ----------------------------------------------------


        if health["status"] in (
            "HIGH LOSS RISK",
            "CRITICAL"
        ):

            logger.warning(
                "%s %s | %s | Health %.1f",
                symbol,
                trade["direction"],
                health["status"],
                health["health_score"]
            )

            logger.warning(
                "Warnings: %s",
                ", ".join(
                    health["warnings"]
                )
            )

            logger.warning(
                "Critical: %s",
                ", ".join(
                    health["critical"]
                )
            )

    save_state()


# ============================================================
# ANALYZE SYMBOL
# ============================================================

def analyze_symbol(symbol):

    frames = {}

    for name, interval in (
        TIMEFRAMES.items()
    ):

        df = fetch_klines(
            symbol,
            interval,
            CANDLE_LIMIT
        )

        if len(df) < 210:

            return None

        frames[name] = (
            timeframe_context(df)
        )

    d1 = frames["1d"]
    h4 = frames["4h"]
    h1 = frames["1h"]

    score = score_market(
        symbol,
        d1,
        h4,
        h1
    )

    direction = score[
        "direction"
    ]

    if direction == "NO TRADE":
        return None

    # --------------------------------------------------------
    # MULTI-TIMEFRAME AGREEMENT
    # --------------------------------------------------------

    if direction == "BUY":

        aligned = sum(
            [
                bool(d1["bull"]),
                bool(h4["bull"]),
                bool(h1["bull"])
            ]
        )

        if aligned < 2:
            return None

    else:

        aligned = sum(
            [
                bool(d1["bear"]),
                bool(h4["bear"]),
                bool(h1["bear"])
            ]
        )

        if aligned < 2:
            return None

    # --------------------------------------------------------
    # LEVELS
    # --------------------------------------------------------

    levels = dynamic_levels(
        direction,
        h1,
        score
    )

    if not levels.get(
        "valid",
        False
    ):

        return None

    signal_candle = str(
        h1["df"][
            "open_time"
        ].iloc[-1]
    )

    signal_id = (
        f"{symbol}|"
        f"{direction}|"
        f"{signal_candle}"
    )

    # --------------------------------------------------------
    # FAILURE FEATURES
    # --------------------------------------------------------

    temp_result = {
        "structure":
            score["structure"],
        "smc":
            score["smc"]
    }

    failure_features = (
        build_failure_features(
            symbol,
            direction,
            d1,
            h4,
            h1,
            temp_result,
            levels
        )
    )

    anti_loss = failure_risk(
        failure_features
    )

    # --------------------------------------------------------
    # ADDITIONAL HARD FILTER
    # --------------------------------------------------------

    if anti_loss[
        "reject_reasons"
    ]:

        logger.info(
            "%s %s rejected by historical failure patterns: %s",
            symbol,
            direction,
            ", ".join(
                anti_loss[
                    "reject_reasons"
                ]
            )
        )

        return None

    health = {
        "health_score": 100,
        "status": "HEALTHY",
        "warnings": anti_loss[
            "warnings"
        ],
        "critical": [],
        "checked_at": iso_pkt()
    }

    return {
        "symbol": symbol,

        "direction": direction,

        "signal_id": signal_id,

        "signal_candle":
            signal_candle,

        "buy_score":
            score["buy_score"],

        "sell_score":
            score["sell_score"],

        "buy_reasons":
            score["buy_reasons"],

        "sell_reasons":
            score["sell_reasons"],

        "buy_indicators":
            score["buy_indicators"],

        "sell_indicators":
            score["sell_indicators"],

        "setups":
            score["setups"],

        "structure":
            score["structure"],

        "smc":
            score["smc"],

        "patterns":
            score["patterns"],

        "candles":
            score["candles"],

        "support":
            score["support"],

        "resistance":
            score["resistance"],

        "atr":
            score["atr"],

        "levels":
            levels,

        "d1_bias":
            "BULL" if d1["bull"]
            else "BEAR"
            if d1["bear"]
            else "NEUTRAL",

        "h4_bias":
            "BULL" if h4["bull"]
            else "BEAR"
            if h4["bear"]
            else "NEUTRAL",

        "h1_bias":
            "BULL" if h1["bull"]
            else "BEAR"
            if h1["bear"]
            else "NEUTRAL",

        "pressure_buy":
            h1["buy_pressure"],

        "pressure_sell":
            h1["sell_pressure"],

        "volume_ratio":
            h1["volume_ratio"],

        "rsi":
            h1["rsi"],

        "failure_features":
            failure_features,

        "anti_loss":
            anti_loss,

        "health":
            health
    }


# ============================================================
# SCAN
# ============================================================

def scan_all():

    candidates = []

    for symbol in SYMBOLS:

        try:

            result = analyze_symbol(
                symbol
            )

            if result is None:
                continue

            candidates.append(
                result
            )

            logger.info(
                "%s | BUY %.1f | SELL %.1f | %s",
                symbol,
                result["buy_score"],
                result["sell_score"],
                result["direction"]
            )

        except Exception as e:

            logger.exception(
                "Analyze error %s: %s",
                symbol,
                e
            )

    candidates.sort(
        key=lambda x: max(
            x["buy_score"],
            x["sell_score"]
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


# ============================================================
# RESTORE
# ============================================================

def restore_open_trades():

    global OPEN_TRADES
    global EMAIL_SENT

    state = load_json(
        STATE_FILE,
        {}
    )

    restored = state.get(
        "open_trades",
        []
    )

    if isinstance(
        restored,
        list
    ):

        OPEN_TRADES = [
            t for t in restored
            if t.get(
                "result"
            ) == "OPEN"
        ]

    EMAIL_SENT = state.get(
        "email_sent",
        {}
    )

    logger.info(
        "Restored %d open trades",
        len(OPEN_TRADES)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    restore_open_trades()

    logger.info(
        "================================================"
    )

    logger.info(
        "MARKET BRAIN AI STARTED"
    )

    logger.info(
        "Symbols: %d",
        len(SYMBOLS)
    )

    logger.info(
        "Anti-Loss Learning: ENABLED"
    )

    logger.info(
        "Trade Health Monitoring: ENABLED"
    )

    logger.info(
        "Failure Pattern Learning: ENABLED"
    )

    logger.info(
        "================================================"
    )

    while True:

        cycle_start = time.time()

        try:

            check_open_trades()

            scan_all()

        except KeyboardInterrupt:

            logger.info(
                "Bot stopped by user."
            )

            save_state()

            break

        except Exception as e:

            logger.exception(
                "Main loop error: %s",
                e
            )

            save_state()

        elapsed = (
            time.time() -
            cycle_start
        )

        sleep_for = max(
            1,
            SCAN_SECONDS - elapsed
        )

        time.sleep(
            sleep_for
        )


if __name__ == "__main__":
    main()
