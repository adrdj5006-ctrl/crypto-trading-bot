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
RECEIVER_EMAIL = SENDER_EMAIL

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

ASSETS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "NEARUSDT",
    "SUIUSDT", "OPUSDT", "ARBUSDT", "INJUSDT", "APTUSDT",
    "PAXGUSDT", "LTCUSDT", "TRXUSDT"
]

SIGNAL_TRACKER = {asset: {"last_processed_pattern": None, "active_trade": None} for asset in ASSETS}
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
            f"Self-Learning AI Status: Active with Multi-TF Volume, RSI, Dynamic 1%-2% Risk & 3%-5% Targets."  
        )  
        send_trade_email("📊 AI Bot - 24 Hours Daily Performance Report", report_body)  
          
        with open(REPORT_STATE_FILE, "w") as f:  
            json.dump({"last_time": now.strftime("%Y-%m-%d %H:%M:%S")}, f)

def fetch_candles_with_backup(symbol, interval, limit=150):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    binance_urls = [  
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",  
        f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",  
        f"https://api1.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",  
        f"https://api3.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"  
    ]  
      
    for url in binance_urls:  
        try:  
            response = requests.get(url, headers=headers, timeout=6)  
            if response.status_code == 200:  
                data = response.json()  
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'ct', 'qv', 'tr', 'tb', 'ts', 'ig'])  
                df[['open', 'high', 'low', 'close', 'vol']] = df[['open', 'high', 'low', 'close', 'vol']].astype(float)  
                return df  
        except Exception:  
            continue  

    try:  
        mexc_interval_map = {"1d": "1d", "4h": "4h", "1h": "60m", "30m": "30m", "15m": "15m", "5m": "5m"}  
        mexc_iv = mexc_interval_map.get(interval, "15m")  
        mexc_url = f"https://www.mexc.com/open/api/v2/market/kline?symbol={symbol}&interval={mexc_iv}"  
        response = requests.get(mexc_url, headers=headers, timeout=6)  
        if response.status_code == 200:  
            res_json = response.json()  
            if "data" in res_json and res_json["data"]:  
                rows = []  
                for item in res_json["data"]:  
                    rows.append([item[0], float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])])  
                df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'vol'])  
                return df.tail(limit)  
    except Exception:  
        pass  

    logging.error(f"All endpoints (Binance + Backups) failed for {symbol}")  
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

def check_volume_strength(df):
    if df is None or len(df) < 10: return False
    avg_vol = df['vol'].iloc[-10:-1].mean()
    curr_vol = df['vol'].iloc[-1]
    return curr_vol >= (avg_vol * 0.8)

def detect_advanced_patterns_and_structure(df_1h, df_15m):
    analysis = {
        "pattern": None, 
        "structure": "NEUTRAL", 
        "ob_price": None, 
        "ob_low": None,
        "fvg": "NO",
        "next_target": None
    }
    if df_15m is None or len(df_15m) < 50: return analysis

    recent_high = df_15m['high'].iloc[-30:-2].max()
    recent_low = df_15m['low'].iloc[-30:-2].min()
    current_close = df_15m['close'].iloc[-1]

    if current_close > recent_high:
        analysis["structure"] = "BULLISH_BOS"
        analysis["next_target"] = df_15m['high'].iloc[-50:].max()
    elif current_close < recent_low:
        analysis["structure"] = "BEARISH_BOS"
        analysis["next_target"] = df_15m['low'].iloc[-50:].min()

    lows = df_15m['low'].iloc[-40:].values
    min_idx1 = np.argmin(lows[:20])
    min_idx2 = 20 + np.argmin(lows[20:])
    val1, val2 = lows[min_idx1], lows[min_idx2]
    
    if abs(val1 - val2) / val1 < 0.008:
        if current_close > df_15m['high'].iloc[min_idx1:min_idx2].max():
            analysis["pattern"] = "W_PATTERN_BULLISH"

    highs = df_15m['high'].iloc[-40:].values
    max_idx1 = np.argmax(highs[:20])
    max_idx2 = 20 + np.argmax(highs[20:])
    h_val1, h_val2 = highs[max_idx1], highs[max_idx2]

    if abs(h_val1 - h_val2) / h_val1 < 0.008:
        if current_close < df_15m['low'].iloc[max_idx1:max_idx2].min():
            analysis["pattern"] = "M_PATTERN_BEARISH"

    if df_1h is not None and len(df_1h) >= 3:
        for i in range(len(df_1h)-2, 1, -1):
            if df_1h['close'].iloc[i] < df_1h['open'].iloc[i]:
                analysis["ob_price"] = df_1h['high'].iloc[i]
                analysis["ob_low"] = df_1h['low'].iloc[i]
                break

        if df_1h['low'].iloc[-1] > df_1h['high'].iloc[-3]: 
            analysis["fvg"] = "BULLISH"  
        elif df_1h['high'].iloc[-1] < df_1h['low'].iloc[-3]: 
            analysis["fvg"] = "BEARISH"

    return analysis

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
    df_1d = fetch_candles_with_backup(symbol, "1d", 30)
    df_4h = fetch_candles_with_backup(symbol, "4h", 50)
    df_1h = fetch_candles_with_backup(symbol, "1h", 100)
    df_30m = fetch_candles_with_backup(symbol, "30m", 100)
    df_15m = fetch_candles_with_backup(symbol, "15m", 150)
    df_5m = fetch_candles_with_backup(symbol, "5m", 150)

    if any(df is None for df in [df_1d, df_4h, df_1h, df_30m, df_15m, df_5m]):  
        return {"status": "DATA ERROR"}  

    current_price = df_5m['close'].iloc[-1]  
    track_active_trades(symbol, current_price)  

    prev_day_close = df_1d['close'].iloc[-2]
    curr_day_close = df_1d['close'].iloc[-1]
    daily_bias = "BULLISH" if curr_day_close > prev_day_close else "BEARISH"

    vol_1d_ok = check_volume_strength(df_1d)
    vol_4h_ok = check_volume_strength(df_4h)
    vol_1h_ok = check_volume_strength(df_1h)
    vol_30m_ok = check_volume_strength(df_30m)
    volume_confirmed = (vol_1d_ok or vol_4h_ok) and (vol_1h_ok or vol_30m_ok)

    trend_4h = analyze_trend_ema(df_4h, period=20)
    market_data = detect_advanced_patterns_and_structure(df_1h, df_15m)  

    rsi_1d = calculate_rsi(df_1d)
    rsi_1h = calculate_rsi(df_1h)
    rsi_5m = calculate_rsi(df_5m)

    action, sl_price, target_price, trade_type = None, 0.0, 0.0, ""

    if volume_confirmed:
        if (daily_bias == "BULLISH" or trend_4h == "UP") and (market_data["pattern"] == "W_PATTERN_BULLISH" or market_data["structure"] == "BULLISH_BOS"):
            if rsi_5m <= 65 and rsi_1h >= 40:
                action = "STRONGLY BUY"
                trade_type = "BUY"
                
                # Dynamic Risk: Stop Loss between 1% and 2% based on structure/order block
                sl_price = market_data["ob_low"] if market_data["ob_low"] and market_data["ob_low"] < current_price else current_price * 0.985
                risk_pct = abs((current_price - sl_price) / current_price) * 100
                if risk_pct < 1.0:
                    sl_price = current_price * 0.990  # Min 1% Stop Loss
                elif risk_pct > 2.0:
                    sl_price = current_price * 0.980  # Max 2% Stop Loss
                
                # Dynamic Target: Big Target between 3% to 5% (Ensuring target is strictly above entry)
                if market_data["next_target"] and market_data["next_target"] > (current_price * 1.03):
                    target_price = market_data["next_target"]
                else:
                    target_price = current_price * 1.04  # Solid 4% Target

        elif (daily_bias == "BEARISH" or trend_4h == "DOWN") and (market_data["pattern"] == "M_PATTERN_BEARISH" or market_data["structure"] == "BEARISH_BOS"):
            if rsi_5m >= 35 and rsi_1h <= 60:
                action = "STRONGLY SELL"
                trade_type = "SELL"
                
                # Dynamic Risk: Stop Loss between 1% and 2%
                sl_price = market_data["ob_price"] if market_data["ob_price"] and market_data["ob_price"] > current_price else current_price * 1.015
                risk_pct = abs((sl_price - current_price) / current_price) * 100
                if risk_pct < 1.0:
                    sl_price = current_price * 1.010  # Min 1% Stop Loss
                elif risk_pct > 2.0:
                    sl_price = current_price * 1.020  # Max 2% Stop Loss
                
                # Dynamic Target: Big Target between 3% to 5% (Ensuring target is strictly below entry)
                if market_data["next_target"] and market_data["next_target"] < (current_price * 0.97):
                    target_price = market_data["next_target"]
                else:
                    target_price = current_price * 0.96  # Solid 4% Target

    if action and not SIGNAL_TRACKER[symbol]["active_trade"]:
        risk_pct = abs((sl_price - current_price) / current_price) * 100  
        potential_move = abs((target_price - current_price) / current_price) * 100  

        # Strict Validation: Target must not equal entry price and must provide good reward
        if potential_move >= 2.0 and current_price != target_price:
            context = {
                "rsi_5m": round(rsi_5m, 2),
                "rsi_1h": round(rsi_1h, 2),
                "rsi_1d": round(rsi_1d, 2),
                "pattern": market_data["pattern"], 
                "structure": market_data["structure"], 
                "daily_bias": daily_bias,
                "volume_confirmed": volume_confirmed
            }  
            SIGNAL_TRACKER[symbol]["active_trade"] = {  
                "type": trade_type, "entry_price": current_price, "target": target_price, "sl": sl_price, "context": context  
            }  
              
            pk_time = get_pakistan_time()  
            msg = (  
                f"🚨 CRYPTO TRADE SIGNAL ALERT 🚨\n\n"  
                f"Coin / Asset: {symbol}\n"  
                f"Time (Pakistan/Karachi): {pk_time}\n"  
                f"Action: {action}\n"  
                f"Pattern / Structure: {market_data['pattern'] or market_data['structure']}\n"  
                f"Entry Price: ${current_price:,.4f}\n"  
                f"Target Price: ${target_price:,.4f} (+{potential_move:.2f}%)\n"  
                f"Stop Loss: ${sl_price:,.4f} (Risk: {risk_pct:.2f}%)\n\n"  
                f"Engine Context -> Daily Bias: {daily_bias} | Volume Confirmed: {volume_confirmed} | RSI: {rsi_5m:.1f}"  
            )  
            send_trade_email(f"🚨 {action} Signal: {symbol}", msg)  
            SIGNAL_TRACKER[symbol]["last_processed_pattern"] = market_data["pattern"]  
            return {"status": action}  

    return {"status": "SCANNING"}

def main_loop():
    logging.info("Advanced Multi-TF Engine with Volume, RSI, 1%-2% Risk & 4% Target Started...")
    while True:
        check_and_send_daily_report()

        for asset in ASSETS:  
            analyze_asset_pipeline(asset)  
            time.sleep(1)  
        
        time.sleep(180)

if __name__ == "__main__":
    main_loop()
