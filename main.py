import os
import json
import time
import threading
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template

# ==========================================================
# TRADING BOT BACKEND
# Multi-timeframe market scanner + paper trade tracker
# Timeframes: 1D / 4H / 1H / 15M
# Paper trading only. No real orders are placed.
# ==========================================================

app = Flask(__name__)

# ---------------- CONFIGURATION ----------------

APP_NAME = "MARKET BRAIN AI"
PORT = int(os.environ.get("PORT", "5000"))

SCAN_SECONDS = int(os.environ.get("SCAN_SECONDS", "60"))
CANDLE_LIMIT = 250
MAX_NEW_TRADES_PER_SCAN = 3
MAX_OPEN_TRADES = 8

MIN_SCORE = 65
MIN_RR = 2.0

# Public market-data endpoint
BINANCE_BASE = os.environ.get(
    "BINANCE_BASE",
    "https://data-api.binance.vision"
).rstrip("/")

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "DOTUSDT", "LINKUSDT", "NEARUSDT", "SUIUSDT",
    "OPUSDT", "ARBUSDT", "INJUSDT", "APTUSDT",
    "LTCUSDT", "TRXUSDT", "UNIUSDT", "ATOMUSDT"
]

TIMEFRAMES = ["1d", "4h", "1h", "15m"]

PKT = ZoneInfo("Asia/Karachi")

DATA_DIR = os.environ.get("DATA_DIR", "market_brain_data")
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")
TRADES_FILE = os.path.join(DATA_DIR, "paper_trades.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("market_brain")

session = requests.Session()
session.headers.update({
    "User-Agent": "MARKET-BRAIN-AI/3.0"
})

# ---------------- GLOBAL STATE ----------------

lock = threading.RLock()

state = {
    "app": APP_NAME,
    "status": "STARTING",
    "last_scan": None,
    "last_error": None,
    "market": {},
    "signals": [],
    "logs": [],
    "scan_count": 0
}

trades = {
    "active": [],
    "closed": []
}


# ---------------- TIME HELPERS ----------------

def now_utc():
    return datetime.now(timezone.utc)


def now_pkt():
    return datetime.now(PKT).isoformat(timespec="seconds")


def utc_string():
    return now_utc().isoformat(timespec="seconds")


# ---------------- PERSISTENT DATA ----------------

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path, data):
    temporary = path + ".tmp"

    try:
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        os.replace(temporary, path)

    except OSError:
        logger.exception("Could not save %s", path)


def load_saved_data():
    global state, trades

    saved_state = load_json(STATE_FILE, {})
    saved_trades = load_json(TRADES_FILE, {})

    if isinstance(saved_state, dict):
        state.update(saved_state)

    if isinstance(saved_trades, dict):
        trades["active"] = saved_trades.get("active", [])
        trades["closed"] = saved_trades.get("closed", [])


def save_all():
    with lock:
        save_json(STATE_FILE, state)
        save_json(TRADES_FILE, trades)


def add_log(message, level="INFO"):
    with lock:
        state["logs"].insert(0, {
            "time": now_pkt(),
            "level": level,
            "message": str(message)
        })

        state["logs"] = state["logs"][:100]


# ---------------- BINANCE MARKET DATA ----------------

def get_candles(symbol, interval, limit=CANDLE_LIMIT):
    url = BINANCE_BASE + "/api/v3/klines"

    response = session.get(
        url,
        params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        },
        timeout=15
    )

    response.raise_for_status()
    rows = response.json()

    if not rows:
        raise ValueError("No candles returned")

    columns = [
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote",
        "ignore"
    ]

    df = pd.DataFrame(rows, columns=columns)

    numeric_columns = [
        "open", "high", "low", "close",
        "volume", "quote_volume",
        "taker_buy_base", "taker_buy_quote"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(
        df["open_time"], unit="ms", utc=True
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"], unit="ms", utc=True
    )

    # Never use a candle that is still forming.
    df = df[df["close_time"] <= pd.Timestamp.now(tz="UTC")]

    return df.dropna().reset_index(drop=True)


# ---------------- INDICATORS ----------------

def add_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    previous_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    df["volume_avg"] = df["volume"].rolling(
        20, min_periods=5
    ).mean()

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    candle_range = (high - low).replace(0, np.nan)

    df["body"] = close - df["open"]

    df["body_ratio"] = (
        df["body"].abs() / candle_range
    ).fillna(0)

    df["upper_wick"] = (
        high - pd.concat([df["open"], close], axis=1).max(axis=1)
    )

    df["lower_wick"] = (
        pd.concat([df["open"], close], axis=1).min(axis=1) - low
    )

    return df


# ---------------- MARKET STRUCTURE ----------------

def analyze_timeframe(df):
    df = add_indicators(df)

    if len(df) < 205:
        return {
            "bias": "UNKNOWN",
            "reason": "Not enough candles",
            "price": float(df["close"].iloc[-1])
        }

    last = df.iloc[-1]
    previous = df.iloc[-2]

    bullish = (
        last["close"] > last["ema50"]
        and last["ema20"] > last["ema50"]
        and last["ema50"] > last["ema200"]
    )

    bearish = (
        last["close"] < last["ema50"]
        and last["ema20"] < last["ema50"]
        and last["ema50"] < last["ema200"]
    )

    if bullish:
        bias = "BULLISH"
    elif bearish:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    recent = df.iloc[-21:-1]

    resistance = float(recent["high"].max())
    support = float(recent["low"].min())

    liquidity_sweep_low = (
        last["low"] < recent["low"].min()
        and last["close"] > recent["low"].min()
    )

    liquidity_sweep_high = (
        last["high"] > recent["high"].max()
        and last["close"] < recent["high"].max()
    )

    bullish_bos = last["close"] > resistance
    bearish_bos = last["close"] < support

    # Simple three-candle fair value gap approximation.
    candle3 = df.iloc[-3]

    bullish_fvg = last["low"] > candle3["high"]
    bearish_fvg = last["high"] < candle3["low"]

    return {
        "bias": bias,
        "price": float(last["close"]),
        "ema20": float(last["ema20"]),
        "ema50": float(last["ema50"]),
        "ema200": float(last["ema200"]),
        "rsi": float(last["rsi"]),
        "atr": float(last["atr"]),
        "volume_ratio": float(last["volume_ratio"]),
        "support": support,
        "resistance": resistance,
        "bullish_bos": bool(bullish_bos),
        "bearish_bos": bool(bearish_bos),
        "sweep_low": bool(liquidity_sweep_low),
        "sweep_high": bool(liquidity_sweep_high),
        "bullish_fvg": bool(bullish_fvg),
        "bearish_fvg": bool(bearish_fvg),
        "body_ratio": float(last["body_ratio"]),
        "bullish_candle": bool(last["close"] > last["open"]),
        "bearish_candle": bool(last["close"] < last["open"]),
        "previous_close": float(previous["close"]),
        "reason": bias
    }


# ---------------- SIGNAL SCORING ----------------

def build_signal(symbol, analyses):
    daily = analyses["1d"]
    four_hour = analyses["4h"]
    hourly = analyses["1h"]
    fifteen = analyses["15m"]

    price = fifteen["price"]

    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []

    # Major trend: 1D
    if daily["bias"] == "BULLISH":
        buy_score += 20
        buy_reasons.append("1D bullish trend")

    elif daily["bias"] == "BEARISH":
        sell_score += 20
        sell_reasons.append("1D bearish trend")

    # Structure: 4H
    if four_hour["bias"] == "BULLISH":
        buy_score += 15
        buy_reasons.append("4H bullish structure")

    elif four_hour["bias"] == "BEARISH":
        sell_score += 15
        sell_reasons.append("4H bearish structure")

    if four_hour["bullish_bos"]:
        buy_score += 12
        buy_reasons.append("4H bullish breakout")

    if four_hour["bearish_bos"]:
        sell_score += 12
        sell_reasons.append("4H bearish breakdown")

    if four_hour["sweep_low"]:
        buy_score += 10
        buy_reasons.append("4H sell-side liquidity sweep")

    if four_hour["sweep_high"]:
        sell_score += 10
        sell_reasons.append("4H buy-side liquidity sweep")

    # Setup: 1H
    if hourly["bias"] == "BULLISH":
        buy_score += 12
        buy_reasons.append("1H bullish trend")

    elif hourly["bias"] == "BEARISH":
        sell_score += 12
        sell_reasons.append("1H bearish trend")

    if hourly["bullish_fvg"]:
        buy_score += 6
        buy_reasons.append("1H bullish FVG")

    if hourly["bearish_fvg"]:
        sell_score += 6
        sell_reasons.append("1H bearish FVG")

    # Entry confirmation: 15M
    if fifteen["bullish_candle"]:
        buy_score += 5

    if fifteen["bearish_candle"]:
        sell_score += 5

    if fifteen["bullish_bos"]:
        buy_score += 10
        buy_reasons.append("15M bullish breakout")

    if fifteen["bearish_bos"]:
        sell_score += 10
        sell_reasons.append("15M bearish breakdown")

    if fifteen["sweep_low"]:
        buy_score += 8
        buy_reasons.append("15M liquidity sweep")

    if fifteen["sweep_high"]:
        sell_score += 8
        sell_reasons.append("15M liquidity sweep")

    if fifteen["volume_ratio"] >= 1.5:
        if fifteen["bullish_candle"]:
            buy_score += 5
            buy_reasons.append("Strong bullish volume")

        elif fifteen["bearish_candle"]:
            sell_score += 5
            sell_reasons.append("Strong bearish volume")

    buy_score = min(100, buy_score)
    sell_score = min(100, sell_score)

    if buy_score >= sell_score:
        direction = "BUY"
        score = buy_score
        reasons = buy_reasons
        atr = hourly["atr"]
        support = hourly["support"]
    else:
        direction = "SELL"
        score = sell_score
        reasons = sell_reasons
        atr = hourly["atr"]
        support = hourly["resistance"]

    # No signal without reasonable agreement between timeframes.
    if score < MIN_SCORE:
        return None

    if abs(buy_score - sell_score) < 10:
        return None

    # Require 1H and 15M direction confirmation.
    if direction == "BUY":
        if hourly["bias"] != "BULLISH":
            return None

        if fifteen["bias"] not in ("BULLISH", "NEUTRAL"):
            return None

        if daily["bias"] == "BEARISH":
            return None

    else:
        if hourly["bias"] != "BEARISH":
            return None

        if fifteen["bias"] not in ("BEARISH", "NEUTRAL"):
            return None

        if daily["bias"] == "BULLISH":
            return None

    if not np.isfinite(atr) or atr <= 0:
        return None

    # ATR-based paper-trade risk levels.
    if direction == "BUY":
        entry = price
        stop = min(support, entry - 1.2 * atr)

        if stop >= entry:
            stop = entry - 1.2 * atr

        risk = entry - stop
        target = entry + max(MIN_RR * risk, 2.0 * atr)

    else:
        entry = price
        stop = max(support, entry + 1.2 * atr)

        if stop <= entry:
            stop = entry + 1.2 * atr

        risk = stop - entry
        target = entry - max(MIN_RR * risk, 2.0 * atr)

    if risk <= 0 or target <= 0:
        return None

    rr = abs(target - entry) / risk

    if rr < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "score": int(score),
        "entry": float(entry),
        "stop_loss": float(stop),
        "take_profit": float(target),
        "risk_reward": round(float(rr), 2),
        "reasons": reasons,
        "status": "SIGNAL",
        "created_at": now_pkt(),
        "timeframe": "15M",
        "score_note": "Rule-based score; not a win probability."
    }


# ---------------- PAPER TRADE TRACKER ----------------

def trade_exists(signal):
    return any(
        t.get("symbol") == signal["symbol"]
        and t.get("direction") == signal["direction"]
        and t.get("status") == "ACTIVE"
        for t in trades["active"]
    )


def create_paper_trade(signal):
    if trade_exists(signal):
        return False

    if len(trades["active"]) >= MAX_OPEN_TRADES:
        return False

    trade = dict(signal)
    trade["id"] = (
        signal["symbol"] + "-"
        + signal["direction"] + "-"
        + str(int(time.time()))
    )

    trade["status"] = "ACTIVE"
    trade["opened_at"] = now_pkt()
    trade["exit_price"] = None
    trade["pnl_percent"] = None

    with lock:
        trades["active"].insert(0, trade)

    add_log(
        f"NEW PAPER TRADE: {trade['symbol']} {trade['direction']} "
        f"Score {trade['score']}"
    )

    save_all()
    return True


def update_paper_trades():
    with lock:
        active_snapshot = list(trades["active"])

    for trade in active_snapshot:
        symbol = trade["symbol"]

        try:
            response = session.get(
                BINANCE_BASE + "/api/v3/ticker/price",
                params={"symbol": symbol},
                timeout=10
            )

            response.raise_for_status()
            price = float(response.json()["price"])

        except Exception as exc:
            logger.warning("Price update failed for %s: %s", symbol, exc)
            continue

        direction = trade["direction"]
        stop = float(trade["stop_loss"])
        target = float(trade["take_profit"])
        entry = float(trade["entry"])

        outcome = None

        if direction == "BUY":
            if price <= stop:
                outcome = "LOSS"
            elif price >= target:
                outcome = "WIN"

            pnl = (price - entry) / entry * 100

        else:
            if price >= stop:
                outcome = "LOSS"
            elif price <= target:
                outcome = "WIN"

            pnl = (entry - price) / entry * 100

        with lock:
            for current in trades["active"]:
                if current.get("id") == trade.get("id"):
                    current["current_price"] = price
                    current["pnl_percent"] = round(pnl, 4)
                    break

        if outcome:
            close_trade(trade["id"], price, outcome)

    save_all()


def close_trade(trade_id, exit_price, outcome):
    with lock:
        trade = next(
            (
                t for t in trades["active"]
                if t.get("id") == trade_id
            ),
            None
        )

        if trade is None:
            return

        trades["active"].remove(trade)

        entry = float(trade["entry"])

        if trade["direction"] == "BUY":
            pnl = (exit_price - entry) / entry * 100
        else:
            pnl = (entry - exit_price) / entry * 100

        trade["exit_price"] = float(exit_price)
        trade["pnl_percent"] = round(pnl, 4)
        trade["result"] = outcome
        trade["status"] = "CLOSED"
        trade["closed_at"] = now_pkt()

        trades["closed"].insert(0, trade)
        trades["closed"] = trades["closed"][:1000]

    add_log(
        f"TRADE CLOSED: {trade['symbol']} "
        f"{outcome} PnL {trade['pnl_percent']}%"
    )

    save_all()


# ---------------- MARKET SCANNER ----------------

def scan_symbol(symbol):
    analyses = {}

    for timeframe in TIMEFRAMES:
        candles = get_candles(symbol, timeframe)
        analyses[timeframe] = analyze_timeframe(candles)

    signal = build_signal(symbol, analyses)

    with lock:
        state["market"][symbol] = {
            "symbol": symbol,
            "price": analyses["15m"]["price"],
            "timeframes": analyses,
            "signal": signal,
            "updated_at": now_pkt()
        }

    return signal


def scanner_loop():
    while True:
        try:
            found_signals = []

            for symbol in SYMBOLS:
                try:
                    signal = scan_symbol(symbol)

                    if signal:
                        found_signals.append(signal)

                except Exception as exc:
                    logger.warning(
                        "Scan failed for %s: %s",
                        symbol,
                        exc
                    )

                time.sleep(0.15)

            found_signals.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            with lock:
                state["signals"] = found_signals
                state["last_scan"] = now_pkt()
                state["scan_count"] += 1
                state["status"] = "RUNNING"
                state["last_error"] = None

            created = 0

            for signal in found_signals:
                if created >= MAX_NEW_TRADES_PER_SCAN:
                    break

                if create_paper_trade(signal):
                    created += 1

            update_paper_trades()
            save_all()

        except Exception as exc:
            logger.exception("Scanner loop error")
            with lock:
                state["status"] = "DEGRADED"
                state["last_error"] = str(exc)

            add_log("Scanner error: " + str(exc), "ERROR")

        time.sleep(max(10, SCAN_SECONDS))


# ---------------- DASHBOARD API ----------------

def performance_summary():
    closed = trades["closed"]

    wins = sum(1 for t in closed if t.get("result") == "WIN")
    losses = sum(1 for t in closed if t.get("result") == "LOSS")
    total = wins + losses

     return {
        "wins": wins,
        "losses": losses,
        "closed_trades": len(closed),
        "active_trades": len(trades["active"]),
        "win_rate": round(wins / total * 100, 2) if total else 0.0,
        "total_pnl_percent": round(
            sum(float(t.get("pnl_percent") or 0) for t in closed),
            4
        )
    }


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with lock:
        return jsonify({
            "app": APP_NAME,
            "status": state["status"],
            "last_scan": state["last_scan"],
            "last_error": state["last_error"],
            "scan_count": state["scan_count"],
            "server_time_pkt": now_pkt(),
            "symbols_count": len(SYMBOLS),
            "timeframes": TIMEFRAMES
        })


@app.route("/api/dashboard")
def api_dashboard():
    with lock:
        return jsonify({
            "app": APP_NAME,
            "status": state["status"],
            "last_scan": state["last_scan"],
            "server_time_pkt": now_pkt(),
            "symbols_count": len(SYMBOLS),
            "signals": state["signals"],
            "market": state["market"],
            "active_trades": trades["active"],
            "closed_trades": trades["closed"][:100],
            "performance": performance_summary(),
            "logs": state["logs"][:50]
        })


@app.route("/api/signals")
def api_signals():
    with lock:
        return jsonify(state["signals"])


@app.route("/api/trades")
def api_trades():
    with lock:
        return jsonify({
            "active": trades["active"],
            "closed": trades["closed"][:100],
            "performance": performance_summary()
        })


@app.route("/health")
def health():
    return jsonify({"ok": True, "time": now_pkt()})


# ---------------- STARTUP ----------------

load_saved_data()


def start_background_scanner():
    thread = threading.Thread(
        target=scanner_loop,
        daemon=True,
        name="market-scanner"
    )
    thread.start()


if __name__ == "__main__":
    start_background_scanner()

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False
                 )
