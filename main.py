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

# ================= CONFIG =================
TZ = timezone(timedelta(hours=5))

ASSETS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","AVAXUSDT","DOTUSDT","LINKUSDT","NEARUSDT",
    "SUIUSDT","OPUSDT","ARBUSDT","INJUSDT","APTUSDT",
    "PAXGUSDT","LTCUSDT","TRXUSDT"
]

AI_FILE = "ai_learning.json"
TRADE_LOG = "trades.json"

# ================= EMAIL =================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL = os.environ.get("GMAIL_USER")
PASS = os.environ.get("GMAIL_PASS")

def send_email(subject, msg):
    try:
        m = EmailMessage()
        m.set_content(msg)
        m["Subject"] = subject
        m["From"] = EMAIL
        m["To"] = EMAIL

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            s.login(EMAIL, PASS)
            s.send_message(m)
    except:
        pass

# ================= TIME =================
def pk_time():
    return datetime.now(TZ).strftime("%Y-%m-%d %I:%M:%S %p")

# ================= AI WEIGHTS =================
def load_ai():
    if os.path.exists(AI_FILE):
        return json.load(open(AI_FILE))
    return {"rsi":1,"trend":1,"fvg":1,"ob":1,"pressure":1}

def save_ai(w):
    json.dump(w, open(AI_FILE,"w"), indent=4)

AI_WEIGHTS = load_ai()

# ================= FETCH =================
def get_data(symbol, tf):
    try:
        url=f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&limit=100"
        d=requests.get(url,timeout=5).json()
        df=pd.DataFrame(d,columns=['t','o','h','l','c','v','ct','q','n','tb','tq','i'])
        df[['o','h','l','c']]=df[['o','h','l','c']].astype(float)
        return df
    except:
        return None

# ================= INDICATORS =================
def rsi(df):
    delta=df['c'].diff()
    up=delta.clip(lower=0)
    down=-delta.clip(upper=0)
    ma_up=up.ewm(com=13).mean()
    ma_down=down.ewm(com=13).mean()
    rs=ma_up/ma_down
    return 100-(100/(1+rs)).iloc[-1]

def trend(df):
    ema=df['c'].ewm(span=20).mean().iloc[-1]
    return "UP" if df['c'].iloc[-1]>ema else "DOWN"

def detect_smc(df):
    if df is None: return {}
    high=df['h'].iloc[-20:].max()
    low=df['l'].iloc[-20:].min()
    price=df['c'].iloc[-1]

    if price>high: return {"type":"BULLISH","target":high*1.02}
    if price<low: return {"type":"BEARISH","target":low*0.98}
    return {"type":None,"target":None}

def pressure(df):
    body=abs(df['c'].iloc[-1]-df['o'].iloc[-1])
    range_=df['h'].iloc[-1]-df['l'].iloc[-1]
    return body/range_ if range_>0 else 0

# ================= AI ENGINE =================
def score(symbol):
    df1=get_data(symbol,"1h")
    df5=get_data(symbol,"5m")
    if df1 is None or df5 is None:
        return None

    s=0
    rs=rsi(df5)
    tr=trend(df1)
    smc=detect_smc(df5)
    pr=pressure(df5)

    if rs<60: s+=15*AI_WEIGHTS["rsi"]
    if tr=="UP": s+=25*AI_WEIGHTS["trend"]
    if smc["type"]=="BULLISH": s+=20*AI_WEIGHTS["ob"]
    if pr>0.5: s+=10*AI_WEIGHTS["pressure"]

    return {
        "score":s,
        "price":df5['c'].iloc[-1],
        "rsi":rs,
        "trend":tr,
        "target": smc["target"] if smc["target"] else df5['c'].iloc[-1]*1.02
    }

# ================= LEARNING =================
def learn(result):
    if result=="WIN":
        for k in AI_WEIGHTS:
            AI_WEIGHTS[k]*=1.02
    else:
        for k in AI_WEIGHTS:
            AI_WEIGHTS[k]*=0.98
    save_ai(AI_WEIGHTS)

# ================= TRADING =================
ACTIVE_TRADES={}

def trade(symbol):
    data=score(symbol)
    if not data: return

    sc=data["score"]

    if sc>60 and symbol not in ACTIVE_TRADES:
        ACTIVE_TRADES[symbol]=data
        send_email("BUY "+symbol, str(data))

    if symbol in ACTIVE_TRADES:
        entry=ACTIVE_TRADES[symbol]["price"]
        cur=data["price"]

        if cur>=ACTIVE_TRADES[symbol]["target"]:
            learn("WIN")
            send_email("WIN "+symbol,str(data))
            del ACTIVE_TRADES[symbol]

        elif cur<=entry*0.98:
            learn("LOSS")
            send_email("LOSS "+symbol,str(data))
            del ACTIVE_TRADES[symbol]

# ================= DASHBOARD =================
        "time": pk_time(),
        "active_trades": ACTIVE_TRADES,
        "ai_weights": AI_WEIGHTS
    })

# ================= MAIN =================
def run_bot():
    while True:
        for s in ASSETS:
            trade(s)
            time.sleep(1)
        time.sleep(20)

# ================= START =================
if __name__=="__main__":
    import threading
    threading.Thread(target=run_bot).start()
    app.run(host="0.0.0.0", port=5000)
