import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import pytz

# =========================
# CONFIG
# =========================

SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
    "SOLUSDT","DOGEUSDT","MATICUSDT","DOTUSDT","LTCUSDT",
    "AVAXUSDT","TRXUSDT","LINKUSDT","ATOMUSDT","NEARUSDT",
    "FTMUSDT","APEUSDT","SANDUSDT","GALAUSDT","OPUSDT"
]

EMAIL = "your@gmail.com"
PASSWORD = "your_app_password"
TO_EMAIL = "your@gmail.com"

# =========================
# TIME (Pakistan)
# =========================

def get_pak_time():
    tz = pytz.timezone("Asia/Karachi")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

# =========================
# DATA (Dummy - replace later)
# =========================

def get_data(symbol, tf):
    return pd.DataFrame({
        "close": np.random.rand(100),
        "high": np.random.rand(100),
        "low": np.random.rand(100)
    })

# =========================
# SMC LOGIC
# =========================

def fvg(df):
    return df["low"].iloc[-2] > df["high"].iloc[-3]

def ob(df):
    return df["close"].iloc[-1] > df["close"].mean()

def choch(df):
    return df["close"].iloc[-1] > df["close"].iloc[-5]

def liquidity(df):
    return df["high"].iloc[-1] == df["high"].rolling(10).max().iloc[-1]

# =========================
# AI WEIGHTS
# =========================

weights = {
    "fvg": 1.0,
    "ob": 1.0,
    "choch": 1.0,
    "liq": 1.0
}

def update_weights(win):
    global weights
    for k in weights:
        weights[k] += 0.1 if win else -0.05

# =========================
# STRATEGY
# =========================

def strategy(symbol):

    df1 = get_data(symbol,"1h")    # seller pressure
    df30 = get_data(symbol,"30m")  # support
    df15 = get_data(symbol,"15m")  # structure
    df5 = get_data(symbol,"5m")    # entry

    score = 0

    if fvg(df15):
        score += weights["fvg"]

    if ob(df15):
        score += weights["ob"]

    if choch(df15):
        score += weights["choch"]

    if liquidity(df15):
        score += weights["liq"]

    trend_sell = df1["close"].iloc[-1] < df1["close"].mean()

    price = df5["close"].iloc[-1]

    # ENTRY / SL / TP
    entry = price
    sl = price * 0.98
    tp = price * 1.04

    if score > 2.5 and trend_sell:
        return "SELL", entry, sl, tp, score

    elif score > 2.5:
        return "BUY", entry, sl, tp, score

    return "NO", None, None, None, score

# =========================
# EMAIL ALERT
# =========================

def send_email(message):

    msg = MIMEText(message)
    msg["Subject"] = "🚀 Trade Alert"
    msg["From"] = EMAIL
    msg["To"] = TO_EMAIL

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL, PASSWORD)
    server.send_message(msg)
    server.quit()

# =========================
# MAIN BOT
# =========================

def run_bot():

    for symbol in SYMBOLS:

        signal, entry, sl, tp, score = strategy(symbol)

        if signal != "NO":

            time_now = get_pak_time()

            message = f"""
📊 TRADE SIGNAL

Symbol: {symbol}
Signal: {signal}

Entry: {entry:.4f}
Stop Loss: {sl:.4f}
Take Profit: {tp:.4f}

Score: {score:.2f}

Time (Pakistan): {time_now}
"""

            print(message)
            send_email(message)

            update_weights(True)

        else:
            update_weights(False)

# =========================
if __name__ == "__main__":
    run_bot()
