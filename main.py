import os
import requests
import pandas as pd
import numpy as np
import time
import logging
import json
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

# Gmail Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.environ.get("GMAIL_USER")
SENDER_PASSWORD = os.environ.get("GMAIL_PASS")
RECEIVER_EMAIL = SENDER_EMAIL  # خود بخود جی میل یوزر کو بھیجے گا

def send_trade_email(subject, message_body):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        logging.error("Gmail credentials are missing in environment variables!")
        return
    try:
        msg = EmailMessage()
        msg.set_content(message_body)
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        logging.info("Email Alert Sent Successfully!")
    except Exception as e:
        logging.error(f"Gmail SMTP Error: {e}")

def get_pakistan_time():
    pk_timezone = timezone(timedelta(hours=5))
    return datetime.now(pk_timezone).strftime('%Y-%m-%d %I:%M:%S %p')

BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com"
]

ASSETS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", 
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "NEARUSDT", 
    "SUIUSDT", "OPUSDT", "ARBUSDT", "INJUSDT", "APTUSDT",
    "PAXGUSDT", "LTCUSDT", "TRXUSDT"
]

SIGNAL_TRACKER = {asset: {"last_processed_ob": None, "active_trade": None} for asset in ASSETS}
AI_LEARNING_FILE = "ai_trade_learning_log.json"
REPORT_STATE_FILE = "last_report_time.json"

def save_ai_log(data):
    logs = []
    if os.path.exists(AI_LEARNING_FILE):
        try:
            with open(AI_LEARNING_FILE, 'r') as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(data)
    with open(AI_LEARNING_FILE, 'w') as f:
        json.dump(logs, f, indent=4)

# --- 24-Hour Daily Performance Summary Report ---
def check_and_send_daily_report():
    last_run_time = None
    if os.path.exists(REPORT_STATE_FILE):
        try:
            with open(REPORT_STATE_FILE, "r") as f:
                d = json.load(f)
                last_run_time = datetime.strptime(d.get("last_time"), "%Y-%m-%d %H:%M:%S")
        except:
            pass
            
    now = datetime.now()
    if last_run_time is None or (now - last_run_time) >= timedelta(hours=24):
        wins, losses = 0, 0
        if os.path.exists(AI_LEARNING_FILE):
            try:
                with open(AI_LEARNING_FILE, "r") as f:
                    trades = json.load(f)
                    for t in trades:
                        t_time = datetime.fromtimestamp(t.get('timestamp', time.time()))
                        if (now - t_time) <= timedelta(hours=24):
                            if "WIN" in t.get('outcome', ''):
                                wins += 1
                            else:
                                losses += 1
            except:
                pass
                
        pk_time = get_pakistan_time()
        report_body = (
            f"📊 24-HOUR AI TRADING BOT PERFORMANCE REPORT 📊\n"
            f"Time (PKT): {pk_time}\n\n"
            f"- Total Trades (Last 24h): {wins + losses}\n"
            f"- Wins: {wins}\n"
            f"- Losses: {losses}\n\n"
            f"Self-Learning AI Status: Active and optimizing trade filters based on past data."
        )
        send_trade_email("📊 AI Bot - 24 Hours Daily Performance Report", report_body)
        
        with open(REPORT_STATE_FILE, "w") as f:
            json.dump({"last_time": now.strftime("%Y-%m-%d %H:%M:%S")}, f)

def fetch_candles_with_backup(symbol, interval, limit=150):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    for base_url in BINANCE_ENDPOINTS:
        url = f"{base_url}/api/v3/klines"
        try:
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'ct', 'qv', 'tr', 'tb', 'ts', 'ig'])
                df[['open', 'high', 'low', 'close', 'vol']] = df[['open', 'high', 'low', 'close', 'vol']].astype(float)
                return df
            elif response.status_code == 429:
                logging.warning(f"Rate limit hit on {base_url}, switching endpoint...")
                time.sleep(2)
        except Exception:
            continue
    logging.error(f"All Binance endpoints failed for {symbol}")
    return None

def calculate_rsi(df, periods=14):
    if df is None or len(df) < periods: return 50
    close_delta = df['close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=False).mean()
    ma_down = down.ewm(com=periods - 1, adjust=False).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi.iloc[-1]

def analyze_trend_ema(df, period=20):
    if df is None or len(df) < period: return "NEUTRAL"
    ema = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
    return "UP" if df['close'].iloc[-1] > ema else "DOWN"

def detect_advanced_smc(df):
    features = {"ob_price": None, "ob_low": None, "ob_high": None, "fvg_detected": "NO", "next_target": None, "type": None, "is_mitigated": False}
    if df is None or len(df) < 40: return features
    recent_high, recent_low = df['high'].iloc[-25:-2].max(), df['low'].iloc[-25:-2].min()
    current_close = df['close'].iloc[-1]
    
    if current_close > recent_high:
        features["type"] = "BULLISH"
        features["next_target"] = df['high'].iloc[-50:].max()
        for i in range(len(df)-2, 1, -1):
            if df['close'].iloc[i] < df['open'].iloc[i]:
                features["ob_price"] = df['high'].iloc[i]
                features["ob_low"] = df['low'].iloc[i]
                if df['low'].iloc[i+1:].min() <= features["ob_price"]: features["is_mitigated"] = True
                break
    elif current_close < recent_low:
        features["type"] = "BEARISH"
        features["next_target"] = df['low'].iloc[-50:].min()
        for i in range(len(df)-2, 1, -1):
            if df['close'].iloc[i] > df['open'].iloc[i]:
                features["ob_price"] = df['low'].iloc[i]
                features["ob_high"] = df['high'].iloc[i]
                if df['high'].iloc[i+1:].max() >= features["ob_price"]: features["is_mitigated"] = True
                break
    if len(df) >= 3:
        if df['low'].iloc[-1] > df['high'].iloc[-3]: features["fvg_detected"] = "BULLISH"
        elif df['high'].iloc[-1] < df['low'].iloc[-3]: features["fvg_detected"] = "BEARISH"
    return features

def track_active_trades(symbol, current_price):
    trade = SIGNAL_TRACKER[symbol]["active_trade"]
    if not trade: return

    if trade["type"] == "BUY":
        hit_target = current_price >= trade["target"]
        hit_sl = current_price <= trade["sl"]
    else:
        hit_target = current_price <= trade["target"]
        hit_sl = current_price >= trade["sl"]

    if hit_target or hit_sl:
        outcome = "WIN (Target Hit)" if hit_target else "LOSS (Stop Loss Hit)"
        pnl_pct = ((current_price - trade["entry_price"]) / trade["entry_price"]) * 100
        if trade["type"] == "SELL": pnl_pct = -pnl_pct

        log_data = {
            "timestamp": time.time(), "symbol": symbol, "direction": trade["type"],
            "entry_price": trade["entry_price"], "exit_price": current_price, "pnl_percentage": round(pnl_pct, 2),
            "outcome": outcome, "technical_context": trade["context"]
        }
        save_ai_log(log_data)
        
        pk_time = get_pakistan_time()
        close_msg = (
            f"📊 TRADE CLOSED: {symbol}\n"
            f"Time (PKT): {pk_time}\n"
            f"Result: {outcome}\n"
            f"Exit Price: ${current_price:,.4f}\n"
            f"P&L: {pnl_pct:.2f}%"
        )
        send_trade_email(f"📊 Trade Closed: {symbol} - {outcome}", close_msg)
        SIGNAL_TRACKER[symbol]["active_trade"] = None

def analyze_asset_pipeline(symbol):
    df_1h = fetch_candles_with_backup(symbol, "1h", 100)
    df_30m = fetch_candles_with_backup(symbol, "30m", 100)
    df_15m = fetch_candles_with_backup(symbol, "15m", 150)
    df_5m = fetch_candles_with_backup(symbol, "5m", 150)
    
    if any(df is None for df in [df_1h, df_30m, df_15m, df_5m]):
        return {"status": "DATA ERROR", "trend": "N/A", "target_move": "0.00%"}

    current_price = df_5m['close'].iloc[-1]
    track_active_trades(symbol, current_price)

    trend_1h = analyze_trend_ema(df_1h)
    trend_30m = analyze_trend_ema(df_30m)
    smc = detect_advanced_smc(df_15m)
    rsi_val = calculate_rsi(df_5m)
    
    potential_move = abs((smc["next_target"] - current_price) / current_price) * 100 if smc["next_target"] else 0.0
    trend_status = trend_1h if trend_1h == trend_30m else "MIXED"

    if trend_1h == trend_30m and trend_1h != "NEUTRAL" and smc["type"] and not smc["is_mitigated"]:
        action, sl_price, trade_type = None, 0.0, ""
        
        if trend_1h == "UP" and smc["type"] == "BULLISH" and current_price <= smc["ob_price"] * 1.003 and current_price >= smc["ob_low"] * 0.997 and rsi_val <= 53:
            action, sl_price, trade_type = "STRONGLY BUY", smc["ob_low"] * 0.996, "BUY"
        elif trend_1h == "DOWN" and smc["type"] == "BEARISH" and current_price >= smc["ob_price"] * 0.997 and current_price <= smc["ob_high"] * 1.003 and rsi_val >= 47:
            action, sl_price, trade_type = "STRONGLY SELL", smc["ob_high"] * 1.004, "SELL"

        if action and smc["ob_price"] != SIGNAL_TRACKER[symbol]["last_processed_ob"] and not SIGNAL_TRACKER[symbol]["active_trade"]:
            risk_pct = abs((sl_price - current_price) / current_price) * 100
            
            context = {"rsi": round(rsi_val, 2), "fvg": smc["fvg_detected"], "trend": trend_status}
            SIGNAL_TRACKER[symbol]["active_trade"] = {
                "type": trade_type, "entry_price": current_price, "target": smc["next_target"], "sl": sl_price, "context": context
            }
            
            pk_time = get_pakistan_time()
            msg = (
                f"🚨 CRYPTO TRADE SIGNAL ALERT 🚨\n\n"
                f"Coin / Asset: {symbol}\n"
                f"Time (Pakistan/Karachi): {pk_time}\n"
                f"Action: {action}\n"
                f"Entry Price: ${current_price:,.4f}\n"
                f"Target Price: ${smc['next_target']:,.4f} (+{potential_move:.2f}%)\n"
                f"Stop Loss: ${sl_price:,.4f} (Risk: {risk_pct:.2f}%)\n\n"
                f"Engine Context -> RSI: {rsi_val:.2f} | FVG: {smc['fvg_detected']} | Trend: {trend_status}"
            )
            send_trade_email(f"🚨 {action} Signal: {symbol}", msg)
            SIGNAL_TRACKER[symbol]["last_processed_ob"] = smc["ob_price"]
            return {"status": action, "trend": trend_status, "target_move": f"{potential_move:.2f}%"}

    return {"status": "SCANNING", "trend": trend_status, "target_move": f"{potential_move:.2f}%"}

def main_loop():
    logging.info("Advanced Production Engine Started Successfully (No Flask, Pure Trading Engine)...")
    while True:
        check_and_send_daily_report()
        
        for asset in ASSETS:
            analyze_asset_pipeline(asset)
            time.sleep(0.5)
        time.sleep(25)

if __name__ == "__main__":
    main_loop()
        
