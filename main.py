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
# ② 日経225の現在値（info 方式）
# -----------------------------
@app.get("/api/nk225_params")
def nk225_params():
    try:
        yf_ticker = yf.Ticker("^N225")
        info = yf_ticker.info

        price = info.get("regularMarketPrice")
        previous_close = info.get("regularMarketPreviousClose")

        return {
            "price": price,
            "previous_close": previous_close
        }
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# ③ 日経225のボラティリティ（過去20日）
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

        return {
            "days": days,
            "volatility": vol
        }
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# ④ UI（HTML + JS）
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>日経225 オプション分析ツール</title>

    <!-- ★ スマホ最適化 CSS ★ -->
    <style>
        @media (max-width: 480px) {
            body {
                font-size: 22px;
                padding: 16px;
            }
            select, input {
                width: 100%;
                font-size: 22px;
                padding: 12px;
                margin: 8px 0;
                border-radius: 10px;
            }
            button {
                width: 100%;
                font-size: 22px;
                padding: 14px;
                border-radius: 12px;
                margin-top: 12px;
            }
            #infoBox {
                font-size: 22px;
                padding: 14px;
            }
            h2, h3 {
                font-size: 26px;
            }
        }
    </style>
</head>

<body style="font-family: sans-serif; padding: 20px;">

<h2>日経225 オプション分析ツール</h2>

<!-- メニュー -->
<h3>メニュー</h3>
<select id="menu" onchange="onMenuChange()">
    <option value="">選択してください</option>
    <option value="basic">基本情報（株価・ボラティリティ）</option>
    <option value="long_call">ロングコール</option>
    <option value="long_put">ロングプット</option>
    <option value="straddle">ストラドル</option>

    <!-- ★ クレジットスプレッド追加 ★ -->
    <option value="bull_put">ブル・プット・クレジットスプレッド</option>
    <option value="bear_call">ベア・コール・クレジットスプレッド</option>
</select>

<!-- 情報表示エリア -->
<div id="infoBox" style="margin-top:20px; background:#f0f0f0; padding:10px;"></div>

<hr>

<!-- ブラック–ショールズ計算 -->
<h3>ブラック–ショールズ計算</h3>

株価 S（日経225）：<input id="S" value="40000"><br><br>
権利行使価格 K：<input id="K" value="41000"><br><br>
残存日数 T（年換算）：<input id="T" value="0.1"><br><br>
金利 r：<input id="r" value="0.001"><br><br>
ボラティリティ σ：<input id="sigma" value="0.2"><br><br>

<button onclick="calc()">計算する</button>

<h3>計算結果</h3>
<pre id="result" style="background:#f5f5f5; padding:10px;"></pre>

<script>
async function onMenuChange() {
    const menu = document.getElementById("menu").value;

    if (menu === "") {
        document.getElementById("infoBox").innerHTML = "";
        return;
    }

    // 株価とボラティリティを取得
    const S = await fetch("/api/nk225_params").then(r => r.json());
    const V = await fetch("/api/nk225_vol?days=20").then(r => r.json());

    document.getElementById("infoBox").innerHTML =
        "📌 株価 S（日経225）: " + S.price + "<br>" +
        "📌 ボラティリティ σ（20日HV）: " + V.volatility.toFixed(4) + "<br>" +
        "📌 選択中のメニュー: " + menu;
}

async function calc() {
    const S = document.getElementById("S").value;
    const K = document.getElementById("K").value;
    const T = document.getElementById("T").value;
    const r = document.getElementById("r").value;
    const sigma = document.getElementById("sigma").value;

    const url = `/api/bs_call?S=${S}&K=${K}&T=${T}&r=${r}&sigma=${sigma}`;
    const res = await fetch(url);
    const data = await res.json();

    if (data.error) {
        document.getElementById("result").textContent = "エラー: " + data.error;
        return;
    }

    document.getElementById("result").textContent =
        "理論価格: " + data.price.toFixed(2) + "\\n" +
        "デルタ: " + data.delta.toFixed(4) + "\\n" +
        "ガンマ: " + data.gamma.toFixed(6) + "\\n" +
        "セータ: " + data.theta.toFixed(2) + "\\n" +
        "ベガ: " + data.vega.toFixed(2);
}
</script>

</body>
</html>
"""
