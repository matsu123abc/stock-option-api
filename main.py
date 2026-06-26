
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from math import log, sqrt, exp
from scipy.stats import norm
import yfinance as yf
import numpy as np
import pandas as pd

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
# ⑥ ブルプットのストライク候補 API（3年間の月末終値ベース）
# -----------------------------
@app.get("/api/bull_put_strikes")
def bull_put_strikes():
    import pandas as pd

    ticker = yf.Ticker("^N225")

    # --- 3年分のデータを確実に取得 ---
    try:
        hist = ticker.history(period="1095d", interval="1d")
    except Exception as e:
        return {"error": f"yfinance error: {str(e)}"}

    if hist is None or hist.empty:
        return {"error": "yfinance が 3年分のデータを取得できませんでした"}

    # --- 月末終値（pandas 2.0 以降は 'ME' を使用） ---
    try:
        monthly = hist["Close"].resample("ME").last()
    except Exception as e:
        return {"error": f"resample error: {str(e)}"}

    if len(monthly) < 12:
        return {"error": "月末データが不足しています"}

    # --- 月次リターン ---
    returns = monthly.pct_change().dropna()
    if returns.empty:
        return {"error": "月次リターンが計算できません"}

    # --- 下落月のみ抽出 ---
    negative_returns = returns[returns < 0]
    if negative_returns.empty:
        return {"error": "下落月が存在しません"}

    avg_drop = negative_returns.mean()

    # --- 現在値 ---
    S = ticker.info.get("regularMarketPrice")
    if S is None:
        return {"error": "現在値が取得できません"}

    # --- ストライク計算 ---
    K_safe = S * (1 + avg_drop)
    K_super_safe = S * (1 + avg_drop * 1.5)
    K_aggressive = S * (1 + avg_drop * 0.7)

    return {
        "S": round(S, 2),
        "avg_drop_rate": round(avg_drop, 4),
        "strike_safe": round(K_safe, 2),
        "strike_super_safe": round(K_super_safe, 2),
        "strike_aggressive": round(K_aggressive, 2)
    }

# -----------------------------
# ベアコールのストライク候補 API（3年データベース）
# -----------------------------
@app.get("/api/bear_call_strikes")
def bear_call_strikes():
    import pandas as pd
    import math

    ticker = yf.Ticker("^N225")

    # --- 3年分のデータを確実に取得 ---
    hist = ticker.history(period="1095d", interval="1d")
    if hist is None or hist.empty:
        return {"error": "yfinance がデータを取得できませんでした"}

    # --- 月末終値（pandas 2.0 以降は ME） ---
    monthly = hist["Close"].resample("ME").last()
    if len(monthly) < 12:
        return {"error": "月末データが不足しています"}

    # --- 月次リターン ---
    returns = monthly.pct_change().dropna()
    if returns.empty:
        return {"error": "月次リターンが計算できません"}

    # --- 上昇月のみ抽出 ---
    positive_returns = returns[returns > 0]
    if positive_returns.empty:
        return {"error": "上昇月が存在しません"}

    avg_rise = positive_returns.mean()

    # --- 現在値 ---
    S = ticker.info.get("regularMarketPrice")
    if S is None:
        return {"error": "現在値が取得できません"}

    # --- ストライク計算（ブルプットと対称） ---
    K_safe = S * (1 + avg_rise)
    K_super_safe = S * (1 + avg_rise * 1.5)
    K_aggressive = S * (1 + avg_rise * 0.7)

    return {
        "S": round(S, 2),
        "avg_rise_rate": round(avg_rise, 4),
        "strike_safe": round(K_safe, 2),
        "strike_super_safe": round(K_super_safe, 2),
        "strike_aggressive": round(K_aggressive, 2)
    }

# -----------------------------
# ⑦ 買いプット候補 API
# -----------------------------
@app.get("/api/bull_put_long_candidates")
def bull_put_long_candidates(K_short: float):
    return {
        "short_strike": K_short,
        "long_safe": K_short - 4000,
        "long_standard": K_short - 2000,
        "long_aggressive": K_short - 1000
    }

# -----------------------------
# ブルプットのプレミアム候補 API（3年データからボラ推定）
# -----------------------------
@app.get("/api/bull_put_premium_candidates_new")
def bull_put_premium_candidates_new(T: float = 0.1, r: float = 0.001):
    import pandas as pd
    import math

    ticker = yf.Ticker("^N225")

    # --- 3年分のデータを確実に取得 ---
    hist = ticker.history(period="1095d", interval="1d")
    if hist is None or hist.empty:
        return {"error": "yfinance がデータを取得できませんでした"}

    # --- 月末終値 ---
    monthly = hist["Close"].resample("ME").last()
    if len(monthly) < 12:
        return {"error": "月末データが不足しています"}

    # --- 月次リターン ---
    returns = monthly.pct_change().dropna()
    if returns.empty:
        return {"error": "月次リターンが計算できません"}

    # --- ボラティリティ推定（年率換算） ---
    sigma_monthly = returns.std()
    sigma = sigma_monthly * math.sqrt(12)

    # --- 現在値 ---
    S = ticker.info.get("regularMarketPrice")
    if S is None:
        return {"error": "現在値が取得できません"}

    # --- ストライク（下落率ベース） ---
    negative_returns = returns[returns < 0]
    avg_drop = negative_returns.mean()

    K_safe = S * (1 + avg_drop)
    K_super_safe = S * (1 + avg_drop * 1.5)
    K_aggressive = S * (1 + avg_drop * 0.7)

    # --- プット価格（BSモデル） ---
    def put_price(S, K, T, r, sigma):
        d1 = (math.log(S/K) + (r + sigma*sigma/2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    return {
        "S": round(S, 2),
        "sigma_estimated": round(sigma, 4),

        "strike_safe": round(K_safe, 2),
        "premium_safe": round(put_price(S, K_safe, T, r, sigma), 2),

        "strike_super_safe": round(K_super_safe, 2),
        "premium_super_safe": round(put_price(S, K_super_safe, T, r, sigma), 2),

        "strike_aggressive": round(K_aggressive, 2),
        "premium_aggressive": round(put_price(S, K_aggressive, T, r, sigma), 2)
    }

# -----------------------------
# ベアコールのストライク & プレミアム候補 API（3年データベース）
# -----------------------------
@app.get("/api/bear_call_premium_candidates_new")
def bear_call_premium_candidates_new(T: float = 0.1, r: float = 0.001):
    import pandas as pd
    import math

    ticker = yf.Ticker("^N225")

    # --- 3年分のデータを確実に取得 ---
    hist = ticker.history(period="1095d", interval="1d")
    if hist is None or hist.empty:
        return {"error": "yfinance がデータを取得できませんでした"}

    # --- 月末終値（pandas 2.0 以降は ME） ---
    monthly = hist["Close"].resample("ME").last()
    if len(monthly) < 12:
        return {"error": "月末データが不足しています"}

    # --- 月次リターン ---
    returns = monthly.pct_change().dropna()
    if returns.empty:
        return {"error": "月次リターンが計算できません"}

    # --- 上昇月のみ抽出 ---
    positive_returns = returns[returns > 0]
    if positive_returns.empty:
        return {"error": "上昇月が存在しません"}

    avg_rise = positive_returns.mean()

    # --- ボラティリティ推定（年率換算） ---
    sigma_monthly = returns.std()
    sigma = sigma_monthly * math.sqrt(12)

    # --- 現在値 ---
    S = ticker.info.get("regularMarketPrice")
    if S is None:
        return {"error": "現在値が取得できません"}

    # --- ストライク（上昇率ベース） ---
    K_safe = S * (1 + avg_rise)
    K_super_safe = S * (1 + avg_rise * 1.5)
    K_aggressive = S * (1 + avg_rise * 0.7)

    # --- コール価格（BSモデル） ---
    def call_price(S, K, T, r, sigma):
        d1 = (math.log(S/K) + (r + sigma*sigma/2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

    return {
        "S": round(S, 2),
        "sigma_estimated": round(sigma, 4),
        "avg_rise_rate": round(avg_rise, 4),

        "strike_safe": round(K_safe, 2),
        "premium_safe": round(call_price(S, K_safe, T, r, sigma), 2),

        "strike_super_safe": round(K_super_safe, 2),
        "premium_super_safe": round(call_price(S, K_super_safe, T, r, sigma), 2),

        "strike_aggressive": round(K_aggressive, 2),
        "premium_aggressive": round(call_price(S, K_aggressive, T, r, sigma), 2)
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

    <button onclick="loadBullPutLongCandidates()">買いプット候補を表示</button>
    <pre id="bullPutLongCandidates"></pre>

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
    const data = await fetch(`/api/bull_put_strikes`).then(r=>r.json());

    document.getElementById("bullPutStrikes").textContent =
        "📌 現在値 S: " + data.S + "\\n" +
        "📌 平均下落率（3年・月末）: " + (data.avg_drop_rate * 100).toFixed(2) + "%\\n\\n" +
        "📌 ストライク候補（ブルプット：下落率ベース）\\n" +
        "安全（平均下落率）: " + data.strike_safe + "\\n" +
        "超安全（1.5倍）: " + data.strike_super_safe + "\\n" +
        "やや攻め（0.7倍）: " + data.strike_aggressive;
}

async function loadBearCallStrikes(){
    const data = await fetch(`/api/bear_call_strikes`).then(r => r.json());

    document.getElementById("bearCallStrikes").textContent =
        "📌 現在値 S: " + data.S + "\\n" +
        "📌 平均上昇率（3年・月末）: " + (data.avg_rise_rate * 100).toFixed(2) + "%\\n\\n" +

        "📌 ストライク候補（ベアコール：3年データベース）\\n" +
        "安全（平均上昇率）: " + data.strike_safe + "\\n" +
        "超安全（1.5倍）: " + data.strike_super_safe + "\\n" +
        "やや攻め（0.7倍）: " + data.strike_aggressive;
}


async function loadBearCallPremiums(){
    const T = 0.1;

    const data = await fetch(`/api/bear_call_premium_candidates_new?T=${T}`)
        .then(r => r.json());

    document.getElementById("bearCallPremiums").textContent =
        "📌 現在値 S: " + data.S + "\\n" +
        "📌 推定ボラティリティ σ: " + data.sigma_estimated + "\\n" +
        "📌 平均上昇率（3年・月末）: " + (data.avg_rise_rate * 100).toFixed(2) + "%\\n\\n" +

        "📌 プレミアム候補（ベアコール：3年データベース）\\n" +
        "安全（平均上昇率）: " + data.strike_safe +
        " → プレミアム: " + data.premium_safe + "\\n" +

        "超安全（1.5倍）: " + data.strike_super_safe +
        " → プレミアム: " + data.premium_super_safe + "\\n" +

        "やや攻め（0.7倍）: " + data.strike_aggressive +
        " → プレミアム: " + data.premium_aggressive;
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

    const data = await fetch(`/api/bull_put_premium_candidates_new?T=${T}`)
        .then(r => r.json());

    document.getElementById("bullPutPremiums").textContent =
        "📌 現在値 S: " + data.S + "\\n" +
        "📌 推定ボラティリティ σ: " + data.sigma_estimated + "\\n\\n" +

        "📌 プレミアム候補（ブルプット：3年データベース）\\n" +
        "安全（平均下落率）: " + data.strike_safe +
        " → プレミアム: " + data.premium_safe + "\\n" +

        "超安全（1.5倍）: " + data.strike_super_safe +
        " → プレミアム: " + data.premium_super_safe + "\\n" +

        "やや攻め（0.7倍）: " + data.strike_aggressive +
        " → プレミアム: " + data.premium_aggressive;
}

async function loadBearCallPremiums(){
    const T = 0.1;   // 残存日数（年換算）

    const data = await fetch(`/api/bear_call_premium_candidates_new?T=${T}`)
        .then(r => r.json());

    document.getElementById("bearCallPremiums").textContent =
        "📌 現在値 S: " + data.S + "\\n" +
        "📌 推定ボラティリティ σ: " + data.sigma_estimated + "\\n" +
        "📌 平均上昇率（3年・月末）: " + (data.avg_rise_rate * 100).toFixed(2) + "%\\n\\n" +

        "📌 プレミアム候補（ベアコール：3年データベース）\\n" +
        "安全（平均上昇率）: " + data.strike_safe +
        " → プレミアム: " + data.premium_safe + "\\n" +

        "超安全（1.5倍）: " + data.strike_super_safe +
        " → プレミアム: " + data.premium_super_safe + "\\n" +

        "やや攻め（0.7倍）: " + data.strike_aggressive +
        " → プレミアム: " + data.premium_aggressive;
}


async function loadBullPutLongCandidates(){
    const K_short = Number(document.getElementById("bp_K_short").value);

    const data = await fetch(`/api/bull_put_long_candidates?K_short=${K_short}`)
        .then(r=>r.json());

    document.getElementById("bullPutLongCandidates").textContent =
        "📌 売りプット: " + data.short_strike + "\\n\\n" +
        "📌 買いプット候補\\n" +
        "安全（Wide）: " + data.long_safe + "\\n" +
        "標準（Medium）: " + data.long_standard + "\\n" +
        "攻め（Narrow）: " + data.long_aggressive;
}


</script>

</body>
</html>
"""
