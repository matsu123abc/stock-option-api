import os
import math
import json
from functools import lru_cache
from typing import Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------
# App 初期化
# -------------------------
app = FastAPI(title="Nikkei225 Option Tool - Integrated")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------------
# ユーティリティ / キャッシュ
# -------------------------
@lru_cache(maxsize=64)
def get_ticker_history(symbol: str, period: str = "1095d", interval: str = "1d"):
    t = yf.Ticker(symbol)
    try:
        hist = t.history(period=period, interval=interval)
        return hist
    except Exception:
        return pd.DataFrame()

def safe_info_price(symbol: str):
    t = yf.Ticker(symbol)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    price = info.get("regularMarketPrice")
    if price is None:
        try:
            hist = t.history(period="1d")
            if hist is not None and not hist.empty:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            price = None
    return price

def resample_monthly_close(hist: pd.DataFrame):
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    try:
        monthly = hist["Close"].resample("ME").last()
    except Exception:
        monthly = hist["Close"].resample("M").last()
    return monthly.dropna()

# -------------------------
# Pydantic モデル（POST 用）
# -------------------------
class RollCandidateRequest(BaseModel):
    S: float = Field(..., gt=0)
    short_put: float = Field(..., gt=0)
    long_put: float = Field(..., gt=0)
    credit: float = Field(..., ge=0)
    iv: Optional[float] = Field(0.20, ge=0)
    market_bias: Optional[int] = Field(0)

class RollPnlRequest(BaseModel):
    S: float = Field(..., gt=0)
    new_short_put: float = Field(..., gt=0)
    new_long_put: float = Field(..., gt=0)
    new_credit: float = Field(..., ge=0)
    iv: Optional[float] = Field(0.20, ge=0)
    market_bias: Optional[int] = Field(0)

# -------------------------
# ① ブラック–ショールズ（コール/プット） & ギリシャ
# -------------------------
def bs_prices_and_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * (norm.cdf(d2) if option_type=="call" else -norm.cdf(-d2))
    vega = S * norm.pdf(d1) * math.sqrt(T)
    rho = K * T * math.exp(-r * T) * (norm.cdf(d2) if option_type=="call" else -norm.cdf(-d2))
    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho)
    }

@app.get("/api/bs_call")
def bs_call(S: float = Query(..., gt=0), K: float = Query(..., gt=0), T: float = Query(..., gt=0), r: float = 0.001, sigma: float = Query(..., gt=0)):
    try:
        return bs_prices_and_greeks(S, K, T, r, sigma, option_type="call")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="BS calculation error")

@app.get("/api/greeks")
def calc_greeks(S: float = Query(..., gt=0), K: float = Query(..., gt=0), T: float = Query(..., gt=0), r: float = 0.001, sigma: float = Query(..., gt=0), option_type: str = "put"):
    try:
        return bs_prices_and_greeks(S, K, T, r, sigma, option_type=option_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Greeks calculation failed")

# -------------------------
# ② 日経225 基本データ / ボラ
# -------------------------
@app.get("/api/nk225_params")
def nk225_params():
    price = safe_info_price("^N225")
    if price is None:
        return JSONResponse({"error": "現在値が取得できません"}, status_code=500)
    return {"price": price}

@app.get("/api/nk225_vol")
def nk225_vol(days: int = 20):
    hist = get_ticker_history("^N225", period=f"{days+1}d")
    if hist is None or hist.empty:
        return JSONResponse({"error": "データ不足"}, status_code=500)
    close = hist["Close"].values
    if len(close) < 2:
        return JSONResponse({"error": "データ不足"}, status_code=500)
    log_returns = np.log(close[1:] / close[:-1])
    vol = float(np.std(log_returns, ddof=1) * math.sqrt(252))
    return {"days": days, "volatility": vol}

# -------------------------
# ③ ブルプット / ベアコール 損益計算
# -------------------------
@app.get("/api/bull_put")
def bull_put(S: float = Query(...), K_short: float = Query(...), K_long: float = Query(...), premium_short: float = Query(...), premium_long: float = Query(...)):
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
    return {"credit": credit, "max_profit": max_profit, "max_loss": max_loss, "breakeven": breakeven, "profit_at_S": profit}

@app.get("/api/bear_call")
def bear_call(S: float = Query(...), K_short: float = Query(...), K_long: float = Query(...), premium_short: float = Query(...), premium_long: float = Query(...)):
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
    return {"credit": credit, "max_profit": max_profit, "max_loss": max_loss, "breakeven": breakeven, "profit_at_S": profit}

# -------------------------
# ④ ストライク候補 / プレミアム候補（3年データベース）
# -------------------------
@app.get("/api/bull_put_strikes")
def bull_put_strikes():
    hist = get_ticker_history("^N225", period="1095d")
    monthly = resample_monthly_close(hist)
    if monthly.empty or len(monthly) < 12:
        return JSONResponse({"error": "月末データが不足しています"}, status_code=500)
    returns = monthly.pct_change().dropna()
    negative_returns = returns[returns < 0]
    if negative_returns.empty:
        avg_drop = float(negative_returns.mean()) if not negative_returns.empty else -0.01
    else:
        avg_drop = float(negative_returns.mean())
    S = safe_info_price("^N225")
    if S is None:
        return JSONResponse({"error": "現在値が取得できません"}, status_code=500)
    K_safe = S * (1 + avg_drop)
    K_super_safe = S * (1 + avg_drop * 1.5)
    K_aggressive = S * (1 + avg_drop * 0.7)
    return {"S": round(S,2), "avg_drop_rate": round(avg_drop,4), "strike_safe": round(K_safe,2), "strike_super_safe": round(K_super_safe,2), "strike_aggressive": round(K_aggressive,2)}

@app.get("/api/bear_call_strikes")
def bear_call_strikes():
    hist = get_ticker_history("^N225", period="1095d")
    monthly = resample_monthly_close(hist)
    if monthly.empty or len(monthly) < 12:
        return JSONResponse({"error": "月末データが不足しています"}, status_code=500)
    returns = monthly.pct_change().dropna()
    positive_returns = returns[returns > 0]
    if positive_returns.empty:
        avg_rise = float(positive_returns.mean()) if not positive_returns.empty else 0.01
    else:
        avg_rise = float(positive_returns.mean())
    S = safe_info_price("^N225")
    if S is None:
        return JSONResponse({"error": "現在値が取得できません"}, status_code=500)
    K_safe = S * (1 + avg_rise)
    K_super_safe = S * (1 + avg_rise * 1.5)
    K_aggressive = S * (1 + avg_rise * 0.7)
    return {"S": round(S,2), "avg_rise_rate": round(avg_rise,4), "strike_safe": round(K_safe,2), "strike_super_safe": round(K_super_safe,2), "strike_aggressive": round(K_aggressive,2)}

# -------------------------
# ⑤ ロング候補 / 理論価格（BS）
# -------------------------
@app.get("/api/bull_put_long_candidates")
def bull_put_long_candidates(K_short: float = Query(...)):
    hist = get_ticker_history("^N225", period="1095d")
    monthly = resample_monthly_close(hist)
    returns = monthly.pct_change().dropna()
    negative_returns = returns[returns < 0]
    if negative_returns.empty:
        avg_drop = -0.01
        max_drop = -0.05
    else:
        avg_drop = float(negative_returns.mean())
        max_drop = float(negative_returns.min())
    K_wide = K_short * (1 + max_drop)
    K_medium = K_short * (1 + avg_drop * 2)
    K_narrow = K_short * (1 + avg_drop)
    return {"short_strike": K_short, "avg_drop_rate": round(avg_drop,4), "max_drop_rate": round(max_drop,4), "long_safe": round(K_wide,2), "long_standard": round(K_medium,2), "long_aggressive": round(K_narrow,2)}

@app.get("/api/bear_call_long_candidates")
def bear_call_long_candidates(K_short: float = Query(...)):
    hist = get_ticker_history("^N225", period="1095d")
    monthly = resample_monthly_close(hist)
    returns = monthly.pct_change().dropna()
    positive_returns = returns[returns > 0]
    if positive_returns.empty:
        avg_rise = 0.01
        max_rise = 0.03
    else:
        avg_rise = float(positive_returns.mean())
        max_rise = float(positive_returns.max())
    K_wide = K_short * (1 + max_rise)
    K_medium = K_short * (1 + avg_rise * 2)
    K_narrow = K_short * (1 + avg_rise)
    return {"short_strike": K_short, "avg_rise_rate": round(avg_rise,4), "max_rise_rate": round(max_rise,4), "long_safe": round(K_wide,2), "long_standard": round(K_medium,2), "long_aggressive": round(K_narrow,2)}

@app.get("/api/bull_put_long_premium")
def bull_put_long_premium(K_long: float = Query(...), T: float = 0.1, r: float = 0.001):
    S = safe_info_price("^N225")
    if S is None:
        return JSONResponse({"error": "現在値が取得できません"}, status_code=500)
    hist = get_ticker_history("^N225", period="1095d")
    monthly = resample_monthly_close(hist)
    returns = monthly.pct_change().dropna()
    if returns.empty:
        sigma = 0.2
    else:
        sigma = float(returns.std() * math.sqrt(12))
    try:
        d = bs_prices_and_greeks(S, K_long, T, r, sigma, option_type="put")
        return {"S": round(float(S),2), "sigma_estimated": round(float(sigma),4), "K_long": round(float(K_long),2), "premium_theoretical": round(d["price"],2)}
    except Exception as e:
        return JSONResponse({"error": "BS計算エラー"}, status_code=500)

@app.get("/api/bear_call_long_premium")
def bear_call_long_premium(K_long: float = Query(...), T: float = 0.1, r: float = 0.001):
    S = safe_info_price("^N225")
    if S is None:
        return JSONResponse({"error": "現在値が取得できません"}, status_code=500)
    hist = get_ticker_history("^N225", period="1095d")
    monthly = resample_monthly_close(hist)
    returns = monthly.pct_change().dropna()
    if returns.empty:
        sigma = 0.2
    else:
        sigma = float(returns.std() * math.sqrt(12))
    try:
        d = bs_prices_and_greeks(S, K_long, T, r, sigma, option_type="call")
        return {"S": round(float(S),2), "sigma_estimated": round(float(sigma),4), "K_long": round(float(K_long),2), "premium_theoretical": round(d["price"],2)}
    except Exception:
        return JSONResponse({"error": "BS計算エラー"}, status_code=500)

# -------------------------
# ⑥ Market Insights（簡易） - GPT 呼び出しは省略（既存関数を呼ぶ想定）
# -------------------------
@app.get("/api/market_insights")
def market_insights():
    S = safe_info_price("^N225")
    sigma = None
    try:
        sigma = get_ticker_history("^N225", period="21d")["Close"].pct_change().dropna().std() * math.sqrt(252)
        sigma = float(sigma)
    except Exception:
        sigma = None
    if S is None or sigma is None:
        return JSONResponse({"error": "市場データが不足しています"}, status_code=500)
    hist = get_ticker_history("^N225", period="1y")
    monthly = resample_monthly_close(hist)
    if monthly.empty or len(monthly) < 12:
        return JSONResponse({"error": "過去データ不足"}, status_code=500)
    prices = [float(x) for x in monthly.tolist()]
    returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
    avg_rise = float(np.mean([r for r in returns if r > 0])) if any(r > 0 for r in returns) else 0.0
    avg_drop = float(np.mean([r for r in returns if r < 0])) if any(r < 0 for r in returns) else 0.0
    range_low = min(prices); range_high = max(prices)
    position_percent = (S - range_low) / (range_high - range_low + 1e-9)
    # 簡易バックテスト
    bull_results = []; bear_results = []
    for i in range(len(prices)-1):
        S_entry = prices[i]; S_exit = prices[i+1]
        K_short_bp = int(S_entry * 0.97); K_long_bp = K_short_bp - 500
        pnl_bp = (200 if S_exit >= K_short_bp else (-((K_short_bp - K_long_bp) - 200) if S_exit <= K_long_bp else 200 - (K_short_bp - S_exit)))
        bull_results.append(pnl_bp)
        K_short_bc = int(S_entry * 1.03); K_long_bc = K_short_bc + 500
        pnl_bc = (200 if S_exit <= K_short_bc else (-((K_long_bc - K_short_bc) - 200) if S_exit >= K_long_bc else 200 - (S_exit - K_short_bc)))
        bear_results.append(pnl_bc)
    bull_win_rate = sum(1 for x in bull_results if x > 0) / len(bull_results) if bull_results else 0.0
    bear_win_rate = sum(1 for x in bear_results if x > 0) / len(bear_results) if bear_results else 0.0
    # GPT 呼び出しは省略して空のヒントを返す（実運用では gpt_market_hint を呼ぶ）
    hint = {"strategy": "", "expert_reason": "", "beginner_explanation": "", "beginner_caution": "", "next_step": ""}
    return {
        "S": S, "sigma": sigma, "avg_rise": avg_rise, "avg_drop": avg_drop,
        "max_rise": max(returns) if returns else 0.0, "max_drop": min(returns) if returns else 0.0,
        "bull_put_win_rate": bull_win_rate, "bear_call_win_rate": bear_win_rate,
        "range_low": range_low, "range_high": range_high, "position_percent": position_percent,
        "strategy": hint.get("strategy",""), "expert_reason": hint.get("expert_reason",""),
        "beginner_explanation": hint.get("beginner_explanation",""), "beginner_caution": hint.get("beginner_caution",""),
        "next_step": hint.get("next_step",""), "hint": hint, "hint_text": json.dumps(hint, ensure_ascii=False)
    }

# -------------------------
# ⑦ Rollout / Rolldown（POST） - Pydantic を使用（上で定義済）
# -------------------------
# roll endpoints are defined above in roll router style but included here for single-file simplicity
@app.post("/api/rollout_candidates")
def rollout_candidates(req: RollCandidateRequest):
    S = req.S; short_put = req.short_put; long_put = req.long_put; credit = req.credit; iv = req.iv; market_bias = req.market_bias
    shifts = [300, 500, 700]; candidates = []
    for shift in shifts:
        new_short_put = short_put - shift; new_long_put = long_put - shift
        iv_factor = 1 + iv; bias_factor = 1 + (market_bias * 0.1)
        estimated_credit = round(credit * iv_factor * bias_factor, 2)
        candidates.append({"shift": shift, "new_short_put": new_short_put, "new_long_put": new_long_put, "estimated_credit": estimated_credit, "distance_from_S": round(S - new_short_put,2), "iv": iv, "market_bias": market_bias})
    return {"S": S, "short_put": short_put, "long_put": long_put, "credit": credit, "iv": iv, "market_bias": market_bias, "candidates": candidates}

@app.post("/api/rollout_pnl")
def rollout_pnl(req: RollPnlRequest):
    S = req.S; new_short_put = req.new_short_put; new_long_put = req.new_long_put; new_credit = req.new_credit; iv = req.iv; market_bias = req.market_bias
    if new_short_put <= new_long_put:
        raise HTTPException(status_code=400, detail="new_short_put must be greater than new_long_put")
    width = new_short_put - new_long_put; max_profit = new_credit; max_loss = width - new_credit; breakeven = new_short_put - new_credit
    safety_distance = round(S - new_short_put,2); iv_effect = round(iv * 100,2)
    bias_comment = ("市場は弱気なので、ストライクを下げたロールアウトが有利です。" if market_bias < 0 else "市場は強気なので、ストライクをあまり下げないロールアウトが有利です。" if market_bias > 0 else "市場は中立です。一般的なロールアウトが適しています。")
    return {"new_short_put": new_short_put, "new_long_put": new_long_put, "new_credit": new_credit, "max_profit": round(max_profit,2), "max_loss": -round(max_loss,2), "breakeven": round(breakeven,2), "safety_distance": safety_distance, "iv_effect": iv_effect, "market_bias": market_bias, "bias_comment": bias_comment, "comment": "ロールアウト後の損益構造を計算しました。"}

@app.post("/api/rolldown_candidates")
def rolldown_candidates(req: RollCandidateRequest):
    S = req.S; short_put = req.short_put; long_put = req.long_put; credit = req.credit; iv = req.iv; market_bias = req.market_bias
    shifts = [300, 500, 700]; candidates = []
    for shift in shifts:
        new_short_put = short_put - shift; new_long_put = long_put - shift
        iv_factor = max(0.0, 1 - (iv * 0.3)); bias_factor = 1 + (market_bias * 0.1)
        estimated_credit = round(credit * iv_factor * bias_factor, 2)
        candidates.append({"shift": shift, "new_short_put": new_short_put, "new_long_put": new_long_put, "estimated_credit": estimated_credit, "distance_from_S": round(S - new_short_put,2), "iv": iv, "market_bias": market_bias})
    return {"S": S, "short_put": short_put, "long_put": long_put, "credit": credit, "iv": iv, "market_bias": market_bias, "candidates": candidates}

@app.post("/api/rolldown_pnl")
def rolldown_pnl(req: RollPnlRequest):
    S = req.S; new_short_put = req.new_short_put; new_long_put = req.new_long_put; new_credit = req.new_credit; iv = req.iv; market_bias = req.market_bias
    if new_short_put <= new_long_put:
        raise HTTPException(status_code=400, detail="new_short_put must be greater than new_long_put")
    width = new_short_put - new_long_put; max_profit = new_credit; max_loss = width - new_credit; breakeven = new_short_put - new_credit
    safety_distance = round(S - new_short_put,2); iv_effect = round(iv * 100,2)
    bias_comment = ("市場は弱気なので、ストライクを下げたロールダウンが有利です。" if market_bias < 0 else "市場は強気なので、ストライクをあまり下げないロールダウンが有利です。" if market_bias > 0 else "市場は中立です。一般的なロールダウンが適しています。")
    return {"new_short_put": new_short_put, "new_long_put": new_long_put, "new_credit": new_credit, "max_profit": round(max_profit,2), "max_loss": -round(max_loss,2), "breakeven": round(breakeven,2), "safety_distance": safety_distance, "iv_effect": iv_effect, "market_bias": market_bias, "bias_comment": bias_comment, "comment": "ロールダウン後の損益構造を計算しました。"}



# -----------------------------
# ⑧ UI（スマホ最適化 + ブルプット/ベアコール）
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
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
    font-family:system-ui, -apple-system, "Hiragino Kaku Gothic ProN", "Hiragino Kaku Gothic Pro", "Yu Gothic", "Meiryo", sans-serif;
    padding:16px;
    font-size:18px;
    line-height:1.5;
  }

  h2, h3{
    font-size:22px;
    margin-bottom:8px;
  }

  select, input{
    width:100%;
    font-size:18px;
    padding:12px;
    margin:8px 0;
    border-radius:8px;
    border:1px solid #ccc;
    background:#fff;
    box-sizing:border-box;
  }

  button{
    width:100%;
    font-size:18px;
    padding:12px;
    border-radius:10px;
    margin-top:12px;
    background:var(--accent);
    color:#fff;
    border:none;
    cursor:pointer;
  }

  button:disabled{
    opacity:0.6;
    cursor:not-allowed;
  }

  #infoBox, #bullPutBox, #bearCallBox, #rolloutBox, #rolldownBox{
    background:var(--panel);
    padding:12px;
    border-radius:10px;
    font-size:16px;
    margin-top:12px;
  }

  pre{
    background:var(--panel);
    padding:12px;
    border-radius:10px;
    font-size:15px;
    white-space:pre-wrap;
    overflow:auto;
  }

  .small{
    font-size:14px;
    color:#444;
  }

  @media (min-width:720px){
    body{ font-size:20px; }
    h2,h3{ font-size:26px; }
    select,input{ font-size:20px; padding:14px; }
    button{ font-size:20px; padding:14px; }
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
    <option value="rollout">ロールアウト（期限延長）</option>
    <option value="rolldown">ロールダウン（ストライク調整）</option>
</select>

<div id="infoBox" class="small"></div>

<h3>基本情報（Market Insights）</h3>

<div id="insightsSection" style="display:none;">
    <button id="btnLoadMarket" onclick="loadMarketInsights()">最新情報を取得</button>

    <div id="insightsBox" style="margin-top:12px;"></div>
</div>

<div id="greeksBox" style="margin-top:12px;"></div>

<!-- ブルプット UI -->
<div id="bullPutBox" style="display:none;">
    <h3>ブル・プット・クレジットスプレッド</h3>

    株価 S（日経225・任意入力可）:<br>
    <input id="bp_S" type="number" placeholder="例: 69000">

    売りプットのストライク（K_short）:<br>
    <input id="bp_K_short" type="number" placeholder="例: 68000">

    買いプットのストライク（K_long）:<br>
    <input id="bp_K_long" type="number" placeholder="例: 67500">

    売りプットのプレミアム:<br>
    <input id="bp_premium_short" type="number" placeholder="例: 200">

    買いプットのプレミアム:<br>
    <input id="bp_premium_long" type="number" placeholder="例: 50">

    <button id="btnCalcBullPut" onclick="calcBullPut()">ブル・プット損益計算</button>
    <pre id="bullPutResult"></pre>

    <button id="btnBullPutStrikes" onclick="loadBullPutStrikes()">ストライク候補を表示</button>
    <pre id="bullPutStrikes"></pre>

    <button id="btnBullPutPremiums" onclick="loadBullPutPremiums()">プレミアム候補を表示</button>
    <pre id="bullPutPremiums"></pre>

    <div class="small" style="margin-top:8px; color:#333;">
    📘 買いプット候補の出し方<br>
    1. 売りプットのストライク（K_short）を入力してください。<br>
    2. 「買いプット候補を表示」を押すと、過去3年の下落率から自動計算されます。
    </div>

    <button id="btnBullPutLongCandidates" onclick="loadBullPutLongCandidates()">買いプット候補を表示</button>
    <pre id="bullPutLongCandidates"></pre>

    <button id="btnCalcBullPutLongPremium" onclick="calcBullPutLongPremium()">買いプットプレミアムを自動計算</button>
</div>

<!-- ベアコール UI -->
<div id="bearCallBox" style="display:none;">
    <h3>ベア・コール・クレジットスプレッド</h3>

    株価 S（日経225・任意入力可）:<br>
    <input id="bc_S" type="number" placeholder="例: 69000">

    売りコールのストライク（K_short）:<br>
    <input id="bc_K_short" type="number" placeholder="例: 70000">

    買いコールのストライク（K_long）:<br>
    <input id="bc_K_long" type="number" placeholder="例: 70500">

    売りコールのプレミアム:<br>
    <input id="bc_premium_short" type="number" placeholder="例: 200">

    買いコールのプレミアム:<br>
    <input id="bc_premium_long" type="number" placeholder="例: 50">

    <button id="btnCalcBearCall" onclick="calcBearCall()">ベア・コール損益計算</button>
    <pre id="bearCallResult"></pre>

    <button id="btnBearCallStrikes" onclick="loadBearCallStrikes()">ストライク候補を表示</button>
    <pre id="bearCallStrikes"></pre>

    <button id="btnBearCallPremiums" onclick="loadBearCallPremiums()">プレミアム候補を表示</button>
    <pre id="bearCallPremiums"></pre>

    <div class="small" style="margin-top:8px; color:#333;">
    📘 買いコール候補の出し方<br>
    1. 売りコールのストライク（K_short）を入力してください。<br>
    2. 「買いコール候補を表示」を押すと、過去3年のデータから自動計算されます。
    </div>

    <button id="btnBearCallLongCandidates" onclick="loadBearCallLongCandidates()">買いコール候補を表示</button>
    <pre id="bearCallLongCandidates"></pre>

    <button id="btnCalcBearCallLongPremium" onclick="calcBearCallLongPremium()">買いコールプレミアムを自動計算</button>
</div>

<!-- ロールアウト UI -->
<div id="rolloutBox" style="display:none;">
    <h3>ロールアウト（期限延長）</h3>

    株価 S:<br>
    <input id="ro_S" type="number" placeholder="例: 69000">

    売りプットのストライク（K_short）:<br>
    <input id="ro_K_short" type="number" placeholder="例: 68000">

    買いプットのストライク（K_long）:<br>
    <input id="ro_K_long" type="number" placeholder="例: 67500">

    現在の受取クレジット:<br>
    <input id="ro_credit" type="number" placeholder="例: 200">

    IV（任意）:<br>
    <input id="ro_iv" type="number" placeholder="例: 0.20">

    市場バイアス（-1 弱気 / 0 中立 / +1 強気）:<br>
    <input id="ro_bias" type="number" placeholder="例: -1">

    <button id="btnLoadRolloutCandidates" onclick="loadRolloutCandidates()">ロールアウト候補を表示</button>
    <pre id="rolloutCandidates"></pre>

    <button id="btnCalcRolloutPNL" onclick="calcRolloutPNL()">ロールアウト損益計算</button>
    <pre id="rolloutPNL"></pre>
</div>

<!-- ロールダウン UI -->
<div id="rolldownBox" style="display:none;">
    <h3>ロールダウン（ストライク調整）</h3>

    株価 S:<br>
    <input id="rd_S" type="number" placeholder="例: 69000">

    売りプットのストライク（K_short）:<br>
    <input id="rd_K_short" type="number" placeholder="例: 68000">

    買いプットのストライク（K_long）:<br>
    <input id="rd_K_long" type="number" placeholder="例: 67500">

    現在の受取クレジット:<br>
    <input id="rd_credit" type="number" placeholder="例: 200">

    IV（任意）:<br>
    <input id="rd_iv" type="number" placeholder="例: 0.20">

    市場バイアス（-1 弱気 / 0 中立 / +1 強気）:<br>
    <input id="rd_bias" type="number" placeholder="例: -1">

    <button id="btnLoadRolldownCandidates" onclick="loadRolldownCandidates()">ロールダウン候補を表示</button>
    <pre id="rolldownCandidates"></pre>

    <button id="btnCalcRolldownPNL" onclick="calcRolldownPNL()">ロールダウン損益計算</button>
    <pre id="rolldownPNL"></pre>
</div>

<hr style="margin-top:18px; margin-bottom:18px;">

<script>
/* ---------------------------
   共通 fetch ラッパー（必須）
   --------------------------- */
async function apiFetch(url, options = {}) {
    try {
        const res = await fetch(url, options);
        const text = await res.text();
        let body = null;
        try { body = text ? JSON.parse(text) : null; } catch(e) { body = text; }
        if (!res.ok) {
            console.error("API error", res.status, body);
            return { __error: true, status: res.status, body: body };
        }
        return { __error: false, status: res.status, body: body };
    } catch (e) {
        console.error("Network or fetch error", e);
        return { __error: true, status: 0, body: String(e) };
    }
}

/* ---------------------------
   UI 切替
   --------------------------- */
function onMenuChange(){
    const menu = document.getElementById("menu").value;

    // 非表示リスト
    const ids = ["insightsSection","bullPutBox","bearCallBox","rolloutBox","rolldownBox"];
    ids.forEach(id => { document.getElementById(id).style.display = "none"; });

    if(menu === "basic"){
        document.getElementById("insightsSection").style.display = "block";
        return;
    }
    if(menu === "bull_put"){
        document.getElementById("bullPutBox").style.display = "block";
        return;
    }
    if(menu === "bear_call"){
        document.getElementById("bearCallBox").style.display = "block";
        return;
    }
    if(menu === "rollout"){
        document.getElementById("rolloutBox").style.display = "block";
        return;
    }
    if(menu === "rolldown"){
        document.getElementById("rolldownBox").style.display = "block";
        return;
    }
}

/* ---------------------------
   ユーティリティ
   --------------------------- */
function safeNumber(v, fallback = 0) {
    const n = Number(v);
    return isFinite(n) ? n : fallback;
}

/* ---------------------------
   Market Insights / Greeks
   --------------------------- */
async function loadMarketInsights(){
    const btn = document.getElementById("btnLoadMarket");
    btn.disabled = true;
    document.getElementById("insightsBox").textContent = "取得中...";
    const res = await apiFetch("/api/market_insights");
    btn.disabled = false;
    if(res.__error){
        document.getElementById("insightsBox").textContent = "市場情報の取得に失敗しました";
        console.error(res);
        return;
    }
    const info = res.body || {};
    if(!isFinite(Number(info.S)) || !isFinite(Number(info.sigma))){
        document.getElementById("insightsBox").textContent = "市場データが不完全です";
        return;
    }

    const posPercent = isFinite(Number(info.position_percent)) ? (Number(info.position_percent)*100).toFixed(1) + "%" : "—";
    const bullWin = isFinite(Number(info.bull_put_win_rate)) ? (Number(info.bull_put_win_rate)*100).toFixed(1) + "%" : "—";
    const bearWin = isFinite(Number(info.bear_call_win_rate)) ? (Number(info.bear_call_win_rate)*100).toFixed(1) + "%" : "—";

    document.getElementById("insightsBox").innerHTML =
        "<b>📌 株価 S:</b> " + Number(info.S).toLocaleString() + "<br>" +
        "<b>📌 ボラティリティ σ:</b> " + Number(info.sigma).toFixed(4) + "<br><br>" +
        "<b>【過去1年の傾向】</b><br>" +
        "平均上昇率: " + (isFinite(Number(info.avg_rise)) ? (Number(info.avg_rise)*100).toFixed(2) + "%" : "—") + "<br>" +
        "平均下落率: " + (isFinite(Number(info.avg_drop)) ? (Number(info.avg_drop)*100).toFixed(2) + "%" : "—") + "<br>" +
        "<b>【勝率（簡易バックテスト）】</b><br>" +
        "ブルプット勝率: " + bullWin + "<br>" +
        "ベアコール勝率: " + bearWin + "<br><br>" +
        "<b>【現在の位置】</b><br>" +
        "過去1年レンジ: " + (info.range_low || "—") + " ～ " + (info.range_high || "—") + "<br>" +
        "現在値の位置: " + posPercent + "<br><br>" +
        "<b>【戦略ヒント】</b><br>" + (info.hint_text || "");

    // ギリシャ指標を安全に取得
    await loadGreeksSafe(info.S, info.sigma);
}

async function loadGreeksSafe(S, sigma){
    document.getElementById("greeksBox").textContent = "ギリシャ指標を取得中...";
    if(!isFinite(Number(S)) || !isFinite(Number(sigma))){
        document.getElementById("greeksBox").textContent = "ギリシャ指標に必要な値が不足しています";
        return;
    }
    const r = 0.001;
    const T = 0.1;
    const K = S;
    const url = `/api/greeks?S=${encodeURIComponent(S)}&K=${encodeURIComponent(K)}&T=${encodeURIComponent(T)}&r=${encodeURIComponent(r)}&sigma=${encodeURIComponent(sigma)}&option_type=put`;
    const res = await apiFetch(url);
    if(res.__error){
        document.getElementById("greeksBox").textContent = "ギリシャ指標の取得に失敗しました";
        console.error(res);
        return;
    }
    const g = res.body || {};
    if(!g || g.delta == null){
        document.getElementById("greeksBox").textContent = "ギリシャ指標が不完全です";
        return;
    }
    document.getElementById("greeksBox").innerHTML =
        "<b>📘 ギリシャ指標（ATM近似）</b><br>" +
        "Δ Delta: " + Number(g.delta).toFixed(4) + "<br>" +
        "Γ Gamma: " + Number(g.gamma).toFixed(6) + "<br>" +
        "Θ Theta: " + Number(g.theta).toFixed(4) + "<br>" +
        "ν Vega: " + Number(g.vega).toFixed(2) + "<br>" +
        "ρ Rho: " + Number(g.rho).toFixed(2) + "<br>";
}

/* ---------------------------
   ブルプット / ベアコール 関数群（堅牢版）
   --------------------------- */
async function loadBullPutStrikes(){
    const res = await apiFetch("/api/bull_put_strikes");
    if(res.__error){ document.getElementById("bullPutStrikes").textContent = "ストライク候補の取得に失敗しました"; return; }
    const data = res.body || {};
    if(!data.S){ document.getElementById("bullPutStrikes").textContent = "データ不足"; return; }
    const avg_drop = isFinite(Number(data.avg_drop_rate)) ? Number(data.avg_drop_rate) : 0;
    const txt = "📌 現在値 S: " + data.S + "\n" +
        "📌 平均下落率（3年・月末）: " + (avg_drop * 100).toFixed(2) + "%\n\n" +
        "📌 ストライク候補（ブルプット：下落率ベース）\n" +
        "安全（平均下落率）: " + (data.strike_safe || "—") + "\n" +
        "超安全（1.5倍）: " + (data.strike_super_safe || "—") + "\n" +
        "やや攻め（0.7倍）: " + (data.strike_aggressive || "—");
    document.getElementById("bullPutStrikes").textContent = txt;
}

async function loadBearCallStrikes(){
    const res = await apiFetch("/api/bear_call_strikes");
    if(res.__error){ document.getElementById("bearCallStrikes").textContent = "ストライク候補の取得に失敗しました"; return; }
    const data = res.body || {};
    if(!data.S){ document.getElementById("bearCallStrikes").textContent = "データ不足"; return; }
    const avg_rise = isFinite(Number(data.avg_rise_rate)) ? Number(data.avg_rise_rate) : 0;
    const txt = "📌 現在値 S: " + data.S + "\n" +
        "📌 平均上昇率（3年・月末）: " + (avg_rise * 100).toFixed(2) + "%\n\n" +
        "📌 ストライク候補（ベアコール：3年データベース）\n" +
        "安全（平均上昇率）: " + (data.strike_safe || "—") + "\n" +
        "超安全（1.5倍）: " + (data.strike_super_safe || "—") + "\n" +
        "やや攻め（0.7倍）: " + (data.strike_aggressive || "—");
    document.getElementById("bearCallStrikes").textContent = txt;
}

async function calcBullPut(){
    const btn = document.getElementById("btnCalcBullPut");
    btn.disabled = true;
    const S = safeNumber(document.getElementById("bp_S").value, 0);
    const K_short = safeNumber(document.getElementById("bp_K_short").value, NaN);
    const K_long = safeNumber(document.getElementById("bp_K_long").value, NaN);
    const premium_short = safeNumber(document.getElementById("bp_premium_short").value, NaN);
    const premium_long = safeNumber(document.getElementById("bp_premium_long").value, NaN);

    if(!isFinite(K_short) || !isFinite(K_long) || !isFinite(premium_short) || !isFinite(premium_long)){
        alert("入力が不正です。数値を確認してください。");
        btn.disabled = false;
        return;
    }

    const url = "/api/bull_put?S=" + encodeURIComponent(S) + "&K_short=" + encodeURIComponent(K_short) + "&K_long=" + encodeURIComponent(K_long) + "&premium_short=" + encodeURIComponent(premium_short) + "&premium_long=" + encodeURIComponent(premium_long);
    const res = await apiFetch(url);
    btn.disabled = false;
    if(res.__error){ document.getElementById("bullPutResult").textContent = "損益計算に失敗しました"; return; }
    const data = res.body || {};

    const credit = isFinite(Number(data.credit)) ? Number(data.credit).toFixed(2) : "—";
    const max_profit = isFinite(Number(data.max_profit)) ? Number(data.max_profit).toFixed(2) : "—";
    const max_loss = isFinite(Number(data.max_loss)) ? Number(data.max_loss).toFixed(2) : "—";
    const breakeven = isFinite(Number(data.breakeven)) ? Number(data.breakeven).toFixed(2) : "—";
    const profit_at_S = isFinite(Number(data.profit_at_S)) ? Number(data.profit_at_S).toFixed(2) : "—";

    document.getElementById("bullPutResult").textContent =
        "受取クレジット: " + credit + "\n" +
        "最大利益: " + max_profit + "\n" +
        "最大損失: " + max_loss + "\n" +
        "損益分岐点: " + breakeven + "\n" +
        "現在の株価での損益: " + profit_at_S;
}

async function calcBearCall(){
    const btn = document.getElementById("btnCalcBearCall");
    btn.disabled = true;
    const S = safeNumber(document.getElementById("bc_S").value, 0);
    const K_short = safeNumber(document.getElementById("bc_K_short").value, NaN);
    const K_long = safeNumber(document.getElementById("bc_K_long").value, NaN);
    const premium_short = safeNumber(document.getElementById("bc_premium_short").value, NaN);
    const premium_long = safeNumber(document.getElementById("bc_premium_long").value, NaN);

    if(!isFinite(K_short) || !isFinite(K_long) || !isFinite(premium_short) || !isFinite(premium_long)){
        alert("入力が不正です。数値を確認してください。");
        btn.disabled = false;
        return;
    }

    const url = "/api/bear_call?S=" + encodeURIComponent(S) + "&K_short=" + encodeURIComponent(K_short) + "&K_long=" + encodeURIComponent(K_long) + "&premium_short=" + encodeURIComponent(premium_short) + "&premium_long=" + encodeURIComponent(premium_long);
    const res = await apiFetch(url);
    btn.disabled = false;
    if(res.__error){ document.getElementById("bearCallResult").textContent = "損益計算に失敗しました"; return; }
    const data = res.body || {};

    const credit = isFinite(Number(data.credit)) ? Number(data.credit).toFixed(2) : "—";
    const max_profit = isFinite(Number(data.max_profit)) ? Number(data.max_profit).toFixed(2) : "—";
    const max_loss = isFinite(Number(data.max_loss)) ? Number(data.max_loss).toFixed(2) : "—";
    const breakeven = isFinite(Number(data.breakeven)) ? Number(data.breakeven).toFixed(2) : "—";
    const profit_at_S = isFinite(Number(data.profit_at_S)) ? Number(data.profit_at_S).toFixed(2) : "—";

    document.getElementById("bearCallResult").textContent =
        "受取クレジット: " + credit + "\n" +
        "最大利益: " + max_profit + "\n" +
        "最大損失: " + max_loss + "\n" +
        "損益分岐点: " + breakeven + "\n" +
        "現在の株価での損益: " + profit_at_S;
}

/* プレミアム候補等 */
async function loadBullPutPremiums(){
    const T = 0.1;
    const res = await apiFetch("/api/bull_put_premium_candidates_new?T=" + encodeURIComponent(T));
    if(res.__error){ document.getElementById("bullPutPremiums").textContent = "プレミアム候補の取得に失敗しました"; return; }
    const data = res.body || {};
    const txt = "📌 現在値 S: " + (data.S || "—") + "\n" +
        "📌 推定ボラティリティ σ: " + (isFinite(Number(data.sigma_estimated)) ? Number(data.sigma_estimated).toFixed(4) : "—") + "\n\n" +
        "📌 プレミアム候補（ブルプット：3年データベース）\n" +
        "安全（平均下落率）: " + (data.strike_safe || "—") + " → プレミアム: " + (data.premium_safe || "—") + "\n" +
        "超安全（1.5倍）: " + (data.strike_super_safe || "—") + " → プレミアム: " + (data.premium_super_safe || "—") + "\n" +
        "やや攻め（0.7倍）: " + (data.strike_aggressive || "—") + " → プレミアム: " + (data.premium_aggressive || "—");
    document.getElementById("bullPutPremiums").textContent = txt;
}

async function loadBearCallPremiums(){
    const T = 0.1;
    const res = await apiFetch("/api/bear_call_premium_candidates_new?T=" + encodeURIComponent(T));
    if(res.__error){ document.getElementById("bearCallPremiums").textContent = "プレミアム候補の取得に失敗しました"; return; }
    const data = res.body || {};
    const avgRise = isFinite(Number(data.avg_rise_rate)) ? (Number(data.avg_rise_rate)*100).toFixed(2) + "%" : "—";
    const txt = "📌 現在値 S: " + (data.S || "—") + "\n" +
        "📌 推定ボラティリティ σ: " + (isFinite(Number(data.sigma_estimated)) ? Number(data.sigma_estimated).toFixed(4) : "—") + "\n" +
        "📌 平均上昇率（3年・月末）: " + avgRise + "\n\n" +
        "📌 プレミアム候補（ベアコール：3年データベース）\n" +
        "安全（平均上昇率）: " + (data.strike_safe || "—") + " → プレミアム: " + (data.premium_safe || "—") + "\n" +
        "超安全（1.5倍）: " + (data.strike_super_safe || "—") + " → プレミアム: " + (data.premium_super_safe || "—") + "\n" +
        "やや攻め（0.7倍）: " + (data.strike_aggressive || "—") + " → プレミアム: " + (data.premium_aggressive || "—");
    document.getElementById("bearCallPremiums").textContent = txt;
}

/* ロング候補・プレミアム自動計算 */
async function loadBullPutLongCandidates(){
    const K_short = safeNumber(document.getElementById("bp_K_short").value, NaN);
    if(!isFinite(K_short)){ alert("売りプットのストライクを入力してください"); return; }
    const res = await apiFetch("/api/bull_put_long_candidates?K_short=" + encodeURIComponent(K_short));
    if(res.__error){ document.getElementById("bullPutLongCandidates").textContent = "候補取得に失敗しました"; return; }
    const data = res.body || {};
    const txt = "📌 売りプット（ショート）: " + (data.short_strike || "—") + "\n" +
        "📌 平均下落率（3年・月末）: " + (isFinite(Number(data.avg_drop_rate)) ? (Number(data.avg_drop_rate)*100).toFixed(2) + "%" : "—") + "\n" +
        "📌 最大下落率（3年・月末）: " + (isFinite(Number(data.max_drop_rate)) ? (Number(data.max_drop_rate)*100).toFixed(2) + "%" : "—") + "\n\n" +
        "📌 買いプット候補（保険ロジック）\n" +
        "安全（Wide: 最悪の下落に備える）: " + (data.long_safe || "—") + "\n" +
        "標準（Medium: 平均下落 ×2）: " + (data.long_standard || "—") + "\n" +
        "攻め（Narrow: 平均下落 ×1）: " + (data.long_aggressive || "—");
    document.getElementById("bullPutLongCandidates").textContent = txt;
}

async function loadBearCallLongCandidates(){
    const K_short = safeNumber(document.getElementById("bc_K_short").value, NaN);
    if(!isFinite(K_short)){ alert("売りコールのストライクを入力してください"); return; }
    const res = await apiFetch("/api/bear_call_long_candidates?K_short=" + encodeURIComponent(K_short));
    if(res.__error){ document.getElementById("bearCallLongCandidates").textContent = "候補取得に失敗しました"; return; }
    const data = res.body || {};
    const txt = "📌 売りコール（ショート）: " + (data.short_strike || "—") + "\n" +
        "📌 平均上昇率（3年・月末）: " + (isFinite(Number(data.avg_rise_rate)) ? (Number(data.avg_rise_rate)*100).toFixed(2) + "%" : "—") + "\n" +
        "📌 最大上昇率（3年・月末）: " + (isFinite(Number(data.max_rise_rate)) ? (Number(data.max_rise_rate)*100).toFixed(2) + "%" : "—") + "\n\n" +
        "📌 買いコール候補（保険ロジック）\n" +
        "安全（Wide: 最悪の上昇に備える）: " + (data.long_safe || "—") + "\n" +
        "標準（Medium: 平均上昇 ×2）: " + (data.long_standard || "—") + "\n" +
        "攻め（Narrow: 平均上昇 ×1）: " + (data.long_aggressive || "—");
    document.getElementById("bearCallLongCandidates").textContent = txt;
}

async function calcBullPutLongPremium(){
    const K_long = safeNumber(document.getElementById("bp_K_long").value, NaN);
    if(!isFinite(K_long)){ alert("買いプットのストライクを入力してください"); return; }
    const T = 0.1;
    const res = await apiFetch("/api/bull_put_long_premium?K_long=" + encodeURIComponent(K_long) + "&T=" + encodeURIComponent(T));
    if(res.__error){ alert("プレミアム計算に失敗しました"); return; }
    const data = res.body || {};
    if(data.error){ alert("エラー: " + data.error); return; }
    document.getElementById("bp_premium_long").value = data.premium_theoretical || "";
}

async function calcBearCallLongPremium(){
    const K_long = safeNumber(document.getElementById("bc_K_long").value, NaN);
    if(!isFinite(K_long)){ alert("買いコールのストライクを入力してください"); return; }
    const T = 0.1;
    const res = await apiFetch("/api/bear_call_long_premium?K_long=" + encodeURIComponent(K_long) + "&T=" + encodeURIComponent(T));
    if(res.__error){ alert("プレミアム計算に失敗しました"); return; }
    const data = res.body || {};
    if(data.error){ alert("エラー: " + data.error); return; }
    document.getElementById("bc_premium_long").value = data.premium_theoretical || "";
}

/* ロールアウト / ロールダウン（POST） */
async function loadRolloutCandidates(){
    const btn = document.getElementById("btnLoadRolloutCandidates");
    btn.disabled = true;
    const payload = {
        S: safeNumber(document.getElementById("ro_S").value, 0),
        short_put: safeNumber(document.getElementById("ro_K_short").value, 0),
        long_put: safeNumber(document.getElementById("ro_K_long").value, 0),
        credit: safeNumber(document.getElementById("ro_credit").value, 0),
        iv: safeNumber(document.getElementById("ro_iv").value, 0.20),
        market_bias: safeNumber(document.getElementById("ro_bias").value, 0)
    };
    const res = await apiFetch("/api/rollout_candidates", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });
    btn.disabled = false;
    if(res.__error){ document.getElementById("rolloutCandidates").textContent = "候補取得に失敗しました"; return; }
    const data = res.body || {};
    if(!Array.isArray(data.candidates)){ document.getElementById("rolloutCandidates").textContent = "候補データが不正です"; return; }
    let txt = "📌 ロールアウト候補一覧\n\n";
    data.candidates.forEach(function(c){
        txt += "shift: " + c.shift + "\n";
        txt += "new_short_put: " + c.new_short_put + "\n";
        txt += "new_long_put: " + c.new_long_put + "\n";
        txt += "estimated_credit: " + c.estimated_credit + "\n";
        txt += "distance_from_S: " + c.distance_from_S + "\n\n";
    });
    document.getElementById("rolloutCandidates").textContent = txt;
}

async function calcRolloutPNL(){
    const btn = document.getElementById("btnCalcRolloutPNL");
    btn.disabled = true;
    const payload = {
        S: safeNumber(document.getElementById("ro_S").value, 0),
        new_short_put: safeNumber(document.getElementById("ro_K_short").value, 0),
        new_long_put: safeNumber(document.getElementById("ro_K_long").value, 0),
        new_credit: safeNumber(document.getElementById("ro_credit").value, 0),
        iv: safeNumber(document.getElementById("ro_iv").value, 0.20),
        market_bias: safeNumber(document.getElementById("ro_bias").value, 0)
    };
    const res = await apiFetch("/api/rollout_pnl", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });
    btn.disabled = false;
    if(res.__error){ document.getElementById("rolloutPNL").textContent = "損益計算に失敗しました"; return; }
    const data = res.body || {};
    document.getElementById("rolloutPNL").textContent =
        "最大利益: " + (data.max_profit ?? "—") + "\n" +
        "最大損失: " + (data.max_loss ?? "—") + "\n" +
        "ブレークイーブン: " + (data.breakeven ?? "—") + "\n" +
        "安全度: " + (data.safety_distance ?? "—") + "\n" +
        "IV効果: " + (data.iv_effect ?? "—") + "%\n" +
        "市場コメント: " + (data.bias_comment ?? "");
}

async function loadRolldownCandidates(){
    const btn = document.getElementById("btnLoadRolldownCandidates");
    btn.disabled = true;
    const payload = {
        S: safeNumber(document.getElementById("rd_S").value, 0),
        short_put: safeNumber(document.getElementById("rd_K_short").value, 0),
        long_put: safeNumber(document.getElementById("rd_K_long").value, 0),
        credit: safeNumber(document.getElementById("rd_credit").value, 0),
        iv: safeNumber(document.getElementById("rd_iv").value, 0.20),
        market_bias: safeNumber(document.getElementById("rd_bias").value, 0)
    };
    const res = await apiFetch("/api/rolldown_candidates", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });
    btn.disabled = false;
    if(res.__error){ document.getElementById("rolldownCandidates").textContent = "候補取得に失敗しました"; return; }
    const data = res.body || {};
    if(!Array.isArray(data.candidates)){ document.getElementById("rolldownCandidates").textContent = "候補データが不正です"; return; }
    let txt = "📌 ロールダウン候補一覧\n\n";
    data.candidates.forEach(function(c){
        txt += "shift: " + c.shift + "\n";
        txt += "new_short_put: " + c.new_short_put + "\n";
        txt += "new_long_put: " + c.new_long_put + "\n";
        txt += "estimated_credit: " + c.estimated_credit + "\n";
        txt += "distance_from_S: " + c.distance_from_S + "\n\n";
    });
    document.getElementById("rolldownCandidates").textContent = txt;
}

async function calcRolldownPNL(){
    const btn = document.getElementById("btnCalcRolldownPNL");
    btn.disabled = true;
    const payload = {
        S: safeNumber(document.getElementById("rd_S").value, 0),
        new_short_put: safeNumber(document.getElementById("rd_K_short").value, 0),
        new_long_put: safeNumber(document.getElementById("rd_K_long").value, 0),
        new_credit: safeNumber(document.getElementById("rd_credit").value, 0),
        iv: safeNumber(document.getElementById("rd_iv").value, 0.20),
        market_bias: safeNumber(document.getElementById("rd_bias").value, 0)
    };
    const res = await apiFetch("/api/rolldown_pnl", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });
    btn.disabled = false;
    if(res.__error){ document.getElementById("rolldownPNL").textContent = "損益計算に失敗しました"; return; }
    const data = res.body || {};
    document.getElementById("rolldownPNL").textContent =
        "最大利益: " + (data.max_profit ?? "—") + "\n" +
        "最大損失: " + (data.max_loss ?? "—") + "\n" +
        "ブレークイーブン: " + (data.breakeven ?? "—") + "\n" +
        "安全度: " + (data.safety_distance ?? "—") + "\n" +
        "IV効果: " + (data.iv_effect ?? "—") + "%\n" +
        "市場コメント: " + (data.bias_comment ?? "");
}

/* 初期表示 */
(function init(){
    document.getElementById("infoBox").textContent = "メニューから機能を選んでください";
})();
</script>

</body>
</html>
 
"""
