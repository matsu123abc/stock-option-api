from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from math import log, sqrt, exp
from scipy.stats import norm
import yfinance as yf
import numpy as np

app = FastAPI()

# -----------------------------
# ① ブラック–ショールズ（コール）
# -----------------------------
@app.get("/api/bs_call")
def bs_call(S: float, K: float, T: float, r: float, sigma: float):
    try:
        d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * sqrt(T)

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -----------------------------
# ② 日経225の現在値
# -----------------------------
@app.get("/api/nk225_params")
def nk225_params():
    try:
        yf_ticker = yf.Ticker("^N225")
        info = yf_ticker.info

        return {
            "price": info.get("regularMarketPrice"),
            "previous_close": info.get("regularMarketPreviousClose")
        }
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# ③ 日経225のボラティリティ
# -----------------------------
@app.get("/api/nk225_vol")
def nk225_vol(days: int = 20):
    try:
        yf_ticker = yf.Ticker("^N225")
        hist = yf_ticker.history(period=f"{days+1}d")

        if len(hist) < days + 1:
            return {"error": "データ不足"}

        close = hist["Close"].values
        log_returns = np.log(close[1:] / close[:-1])
        vol = np.std(log_returns) * np.sqrt(252)

        return {"days": days, "volatility": vol}
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# ④ ブル・プット・クレジットスプレッド API
# -----------------------------
@app.get("/api/bull_put")
def bull_put(S: float, K_short: float, K_long: float,
             premium_short: float, premium_long: float):

    credit = premium_short - premium_long
    max_profit = credit
    max_loss = (K_short - K_long) - credit
    breakeven = K_short - credit

    if S >= K_short:
        profit = max_profit
    elif S <= K_long:
        profit = -max_loss
    else:
        profit = credit - (K_short - S)

    return {
        "credit": credit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "profit_at_S": profit
    }


# -----------------------------
# ⑤ ベア・コール・クレジットスプレッド API
# -----------------------------
@app.get("/api/bear_call")
def bear_call(S: float, K_short: float, K_long: float,
              premium_short: float, premium_long: float):

    credit = premium_short - premium_long
    max_profit = credit
    max_loss = (K_long - K_short) - credit
    breakeven = K_short + credit

    if S <= K_short:
        profit = max_profit
    elif S >= K_long:
        profit = -max_loss
    else:
        profit = credit - (S - K_short)

    return {
        "credit": credit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "profit_at_S": profit
    }


# -----------------------------
# ⑥ ブルプットのストライク候補 API（現在値 S と σ）
# -----------------------------
@app.get("/api/bull_put_strikes")
def bull_put_strikes(T: float = 0.1):
    import math

    yf_ticker = yf.Ticker("^N225")
    info = yf_ticker.info
    S = info.get("regularMarketPrice")

    hist = yf_ticker.history(period="21d")
    close = hist["Close"].values
    log_returns = np.log(close[1:] / close[:-1])
    sigma = np.std(log_returns) * np.sqrt(252)

    one_sigma = S * (1 - sigma * math.sqrt(T))
    two_sigma = S * (1 - 2 * sigma * math.sqrt(T))
    ten_percent = S * 0.90

    return {
        "S": round(S, 2),
        "sigma": round(sigma, 4),
        "safe_1sigma": round(one_sigma, 2),
        "super_safe_2sigma": round(two_sigma, 2),
        "aggressive_10percent": round(ten_percent, 2)
    }


# -----------------------------
# ⑦ ベアコールのストライク候補 API（現在値 S と σ）
# -----------------------------
@app.get("/api/bear_call_strikes")
def bear_call_strikes(T: float = 0.1):
    import math

    yf_ticker = yf.Ticker("^N225")
    info = yf_ticker.info
    S = info.get("regularMarketPrice")

    hist = yf_ticker.history(period="21d")
    close = hist["Close"].values
    log_returns = np.log(close[1:] / close[:-1])
    sigma = np.std(log_returns) * np.sqrt(252)

    one_sigma = S * (1 + sigma * math.sqrt(T))
    two_sigma = S * (1 + 2 * sigma * math.sqrt(T))
    ten_percent = S * 1.10

    return {
        "S": round(S, 2),
        "sigma": round(sigma, 4),
        "safe_1sigma": round(one_sigma, 2),
        "super_safe_2sigma": round(two_sigma, 2),
        "aggressive_10percent": round(ten_percent, 2)
    }

@app.get("/api/bull_put_premium_candidates")
def bull_put_premium_candidates(T: float = 0.1, r: float = 0.001):
    import math

    # 現在値とボラティリティ
    yf_ticker = yf.Ticker("^N225")
    info = yf_ticker.info
    S = info.get("regularMarketPrice")

    hist = yf_ticker.history(period="21d")
    close = hist["Close"].values
    log_returns = np.log(close[1:] / close[:-1])
    sigma = np.std(log_returns) * np.sqrt(252)

    # ストライク候補
    K1 = S * (1 - sigma * math.sqrt(T))      # 1σ
    K2 = S * (1 - 2 * sigma * math.sqrt(T))  # 2σ
    K3 = S * 0.90                             # 10%下

    # プット価格（BSモデル）
    def put_price(S, K, T, r, sigma):
        d1 = (math.log(S/K) + (r + sigma*sigma/2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    return {
        "S": round(S, 2),
        "sigma": round(sigma, 4),
        "strike_1sigma": round(K1, 2),
        "premium_1sigma": round(put_price(S, K1, T, r, sigma), 2),
        "strike_2sigma": round(K2, 2),
        "premium_2sigma": round(put_price(S, K2, T, r, sigma), 2),
        "strike_10percent": round(K3, 2),
        "premium_10percent": round(put_price(S, K3, T, r, sigma), 2)
    }

@app.get("/api/bear_call_premium_candidates")
def bear_call_premium_candidates(T: float = 0.1, r: float = 0.001):
    import math

    # 現在値とボラティリティ
    yf_ticker = yf.Ticker("^N225")
    info = yf_ticker.info
    S = info.get("regularMarketPrice")

    hist = yf_ticker.history(period="21d")
    close = hist["Close"].values
    log_returns = np.log(close[1:] / close[:-1])
    sigma = np.std(log_returns) * np.sqrt(252)

    # ストライク候補
    K1 = S * (1 + sigma * math.sqrt(T))      # 1σ
    K2 = S * (1 + 2 * sigma * math.sqrt(T))  # 2σ
    K3 = S * 1.10                             # 10%上

    # コール価格（BSモデル）
    def call_price(S, K, T, r, sigma):
        d1 = (math.log(S/K) + (r + sigma*sigma/2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

    return {
        "S": round(S, 2),
        "sigma": round(sigma, 4),
        "strike_1sigma": round(K1, 2),
        "premium_1sigma": round(call_price(S, K1, T, r, sigma), 2),
        "strike_2sigma": round(K2, 2),
        "premium_2sigma": round(call_price(S, K2, T, r, sigma), 2),
        "strike_10percent": round(K3, 2),
        "premium_10percent": round(call_price(S, K3, T, r, sigma), 2)
    }


# -----------------------------
# ⑧ UI（スマホ最適化 + ブルプット/ベアコール）
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>日経225 オプション分析ツール</title>

<style>
  :root{
    --bg:#ffffff;
    --panel:#f2f2f2;
    --accent:#0078ff;
    --text:#000;
  }

  body{
    margin:0;
    background:var(--bg);
    color:var(--text);
    font-family:system-ui, -apple-system, "Hiragino Kaku Gothic ProN", sans-serif;
    padding:16px;
    font-size:22px;
  }

  h2, h3{
    font-size:28px;
    margin-bottom:12px;
  }

  select, input{
    width:100%;
    font-size:24px;
    padding:16px;
    margin:10px 0;
    border-radius:10px;
    border:1px solid #ccc;
    background:#fff;
  }

  button{
    width:100%;
    font-size:26px;
    padding:18px;
    border-radius:12px;
    margin-top:16px;
    background:var(--accent);
    color:#fff;
    border:none;
  }

  #infoBox, #bullPutBox, #bearCallBox{
    background:var(--panel);
    padding:16px;
    border-radius:10px;
    font-size:24px;
    margin-top:16px;
  }

  pre{
    background:var(--panel);
    padding:16px;
    border-radius:10px;
    font-size:24px;
    white-space:pre-wrap;
  }
</style>
</head>

<body>

<h2>日経225 オプション分析ツール</h2>

<h3>メニュー</h3>
<select id="menu" onchange="onMenuChange()">
    <option value="">選択してください</option>
    <option value="basic">基本情報（株価・ボラティリティ）</option>
    <option value="bull_put">ブル・プット・クレジットスプレッド</option>
    <option value="bear_call">ベア・コール・クレジットスプレッド</option>
</select>

<div id="infoBox"></div>

<!-- ★ ブルプット UI ★ -->
<div id="bullPutBox" style="display:none;">
    <h3>ブル・プット・クレジットスプレッド</h3>

    株価 S（日経225・任意入力可）:<br>
    <input id="bp_S" type="number">

    売りプットのストライク（K_short）:<br>
    <input id="bp_K_short" type="number">

    買いプットのストライク（K_long）:<br>
    <input id="bp_K_long" type="number">

    売りプットのプレミアム:<br>
    <input id="bp_premium_short" type="number">

    買いプットのプレミアム:<br>
    <input id="bp_premium_long" type="number">

    <button onclick="calcBullPut()">ブル・プット損益計算</button>

    <button onclick="loadBullPutStrikes()">ストライク候補を表示</button>
    <pre id="bullPutStrikes"></pre>

    <button onclick="loadBullPutPremiums()">プレミアム候補を表示</button>
    <pre id="bullPutPremiums"></pre>

    <pre id="bullPutResult"></pre>
</div>

<!-- ★ ベアコール UI ★ -->
<div id="bearCallBox" style="display:none;">
    <h3>ベア・コール・クレジットスプレッド</h3>

    株価 S（日経225・任意入力可）:<br>
    <input id="bc_S" type="number">

    売りコールのストライク（K_short）:<br>
    <input id="bc_K_short" type="number">

    買いコールのストライク（K_long）:<br>
    <input id="bc_K_long" type="number">

    売りコールのプレミアム:<br>
    <input id="bc_premium_short" type="number">

    買いコールのプレミアム:<br>
    <input id="bc_premium_long" type="number">

    <button onclick="calcBearCall()">ベア・コール損益計算</button>

    <button onclick="loadBearCallStrikes()">ストライク候補を表示</button>
    <pre id="bearCallStrikes"></pre>

    <button onclick="loadBearCallPremiums()">プレミアム候補を表示</button>
    <pre id="bearCallPremiums"></pre>

    <pre id="bearCallResult"></pre>
</div>

<hr>

<script>
async function onMenuChange(){
    const menu = document.getElementById("menu").value;

    document.getElementById("bullPutBox").style.display = "none";
    document.getElementById("bearCallBox").style.display = "none";

    if(menu === "bull_put"){
        document.getElementById("bullPutBox").style.display = "block";
    }
    if(menu === "bear_call"){
        document.getElementById("bearCallBox").style.display = "block";
    }

    if(menu === ""){
        document.getElementById("infoBox").innerHTML = "";
        return;
    }

    const S = await fetch("/api/nk225_params").then(r=>r.json());
    const V = await fetch("/api/nk225_vol?days=20").then(r=>r.json());

    document.getElementById("infoBox").innerHTML =
        "📌 株価 S: " + S.price + "<br>" +
        "📌 ボラティリティ σ: " + V.volatility.toFixed(4) + "<br>" +
        "📌 メニュー: " + menu;
}

async function loadBullPutStrikes(){
    const T = 0.1;
    const data = await fetch(`/api/bull_put_strikes?T=${T}`).then(r=>r.json());

    document.getElementById("bullPutStrikes").textContent =
        "📌 現在値 S: " + data.S + "\\n" +
        "📌 ボラティリティ σ: " + data.sigma + "\\n\\n" +
        "📌 ストライク候補（ブルプット）\\n" +
        "安全（1σ）: " + data.safe_1sigma + "\\n" +
        "超安全（2σ）: " + data.super_safe_2sigma + "\\n" +
        "やや攻め（10%下）: " + data.aggressive_10percent;
}

async function loadBearCallStrikes(){
    const T = 0.1;
    const data = await fetch(`/api/bear_call_strikes?T=${T}`).then(r=>r.json());

    document.getElementById("bearCallStrikes").textContent =
        "📌 現在値 S: " + data.S + "\\n" +
        "📌 ボラティリティ σ: " + data.sigma + "\\n\\n" +
        "📌 ストライク候補（ベアコール）\\n" +
        "安全（1σ）: " + data.safe_1sigma + "\\n" +
        "超安全（2σ）: " + data.super_safe_2sigma + "\\n" +
        "やや攻め（10%上）: " + data.aggressive_10percent;
}

async function calcBullPut(){
    const S = Number(document.getElementById("bp_S").value || 0);
    const K_short = Number(document.getElementById("bp_K_short").value);
    const K_long = Number(document.getElementById("bp_K_long").value);
    const premium_short = Number(document.getElementById("bp_premium_short").value);
    const premium_long = Number(document.getElementById("bp_premium_long").value);

    const url = `/api/bull_put?S=${S}&K_short=${K_short}&K_long=${K_long}&premium_short=${premium_short}&premium_long=${premium_long}`;
    const data = await fetch(url).then(r=>r.json());

    document.getElementById("bullPutResult").textContent =
        "受取クレジット: " + data.credit.toFixed(2) + "\\n" +
        "最大利益: " + data.max_profit.toFixed(2) + "\\n" +
        "最大損失: " + data.max_loss.toFixed(2) + "\\n" +
        "損益分岐点: " + data.breakeven.toFixed(2) + "\\n" +
        "現在の株価での損益: " + data.profit_at_S.toFixed(2);
}

async function calcBearCall(){
    const S = Number(document.getElementById("bc_S").value || 0);
    const K_short = Number(document.getElementById("bc_K_short").value);
    const K_long = Number(document.getElementById("bc_K_long").value);
    const premium_short = Number(document.getElementById("bc_premium_short").value);
    const premium_long = Number(document.getElementById("bc_premium_long").value);

    const url = `/api/bear_call?S=${S}&K_short=${K_short}&K_long=${K_long}&premium_short=${premium_short}&premium_long=${premium_long}`;
    const data = await fetch(url).then(r=>r.json());

    document.getElementById("bearCallResult").textContent =
        "受取クレジット: " + data.credit.toFixed(2) + "\\n" +
        "最大利益: " + data.max_profit.toFixed(2) + "\\n" +
        "最大損失: " + data.max_loss.toFixed(2) + "\\n" +
        "損益分岐点: " + data.breakeven.toFixed(2) + "\\n" +
        "現在の株価での損益: " + data.profit_at_S.toFixed(2);
}

async function loadBullPutPremiums(){
    const T = 0.1;
    const data = await fetch(`/api/bull_put_premium_candidates?T=${T}`).then(r=>r.json());

    document.getElementById("bullPutPremiums").textContent =
        "📌 現在値 S: " + data.S + "\\n" +
        "📌 σ: " + data.sigma + "\\n\\n" +
        "📌 プレミアム候補（ブルプット）\\n" +
        "1σ ストライク: " + data.strike_1sigma + " → プレミアム: " + data.premium_1sigma + "\\n" +
        "2σ ストライク: " + data.strike_2sigma + " → プレミアム: " + data.premium_2sigma + "\\n" +
        "10%下: " + data.strike_10percent + " → プレミアム: " + data.premium_10percent;
}


async function loadBearCallPremiums(){
    const T = 0.1;   // 残存日数（年換算）
    const data = await fetch(`/api/bear_call_premium_candidates?T=${T}`).then(r=>r.json());

    document.getElementById("bearCallPremiums").textContent =
        "📌 現在値 S: " + data.S + "\\n" +
        "📌 σ: " + data.sigma + "\\n\\n" +
        "📌 プレミアム候補（ベアコール）\\n" +
        "1σ ストライク: " + data.strike_1sigma + " → プレミアム: " + data.premium_1sigma + "\\n" +
        "2σ ストライク: " + data.strike_2sigma + " → プレミアム: " + data.premium_2sigma + "\\n" +
        "10%上: " + data.strike_10percent + " → プレミアム: " + data.premium_10percent;
}

</script>

</body>
</html>
"""
