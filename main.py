import os
import json
from openai import AzureOpenAI
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
# 買いプット候補 API（3年データ × 保険ロジック）
# -----------------------------
@app.get("/api/bull_put_long_candidates")
def bull_put_long_candidates(K_short: float):
    import pandas as pd
    import math

    ticker = yf.Ticker("^N225")

    # --- 3年分のデータを取得 ---
    hist = ticker.history(period="1095d", interval="1d")
    if hist is None or hist.empty:
        return {"error": "yfinance がデータを取得できませんでした"}

    # --- 月末終値 ---
    monthly = hist["Close"].resample("ME").last()
    returns = monthly.pct_change().dropna()

    # --- 下落月のみ抽出 ---
    negative_returns = returns[returns < 0]
    avg_drop = negative_returns.mean()      # 平均下落率
    max_drop = negative_returns.min()       # 最大下落率（最悪の月）

    # --- 保険ロジックで距離を決める ---
    # Wide（最悪の下落率）
    K_wide = K_short * (1 + max_drop)

    # Medium（平均下落率 × 2）
    K_medium = K_short * (1 + avg_drop * 2)

    # Narrow（平均下落率 × 1）
    K_narrow = K_short * (1 + avg_drop)

    return {
        "short_strike": K_short,
        "avg_drop_rate": round(avg_drop, 4),
        "max_drop_rate": round(max_drop, 4),

        "long_safe": round(K_wide, 2),
        "long_standard": round(K_medium, 2),
        "long_aggressive": round(K_narrow, 2)
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
# 買いコール候補 API（3年データ × 保険ロジック）
# -----------------------------
@app.get("/api/bear_call_long_candidates")
def bear_call_long_candidates(K_short: float):
    import pandas as pd
    import math

    ticker = yf.Ticker("^N225")

    # --- 3年分のデータを取得 ---
    hist = ticker.history(period="1095d", interval="1d")
    if hist is None or hist.empty:
        return {"error": "yfinance がデータを取得できませんでした"}

    # --- 月末終値 ---
    monthly = hist["Close"].resample("ME").last()
    returns = monthly.pct_change().dropna()

    # --- 上昇月のみ抽出 ---
    positive_returns = returns[returns > 0]
    avg_rise = positive_returns.mean()      # 平均上昇率
    max_rise = positive_returns.max()       # 最大上昇率（最も上がった月）

    # --- 保険ロジックで距離を決める ---
    # Wide（最大上昇率）
    K_wide = K_short * (1 + max_rise)

    # Medium（平均上昇率 × 2）
    K_medium = K_short * (1 + avg_rise * 2)

    # Narrow（平均上昇率 × 1）
    K_narrow = K_short * (1 + avg_rise)

    return {
        "short_strike": K_short,
        "avg_rise_rate": round(avg_rise, 4),
        "max_rise_rate": round(max_rise, 4),

        "long_safe": round(K_wide, 2),
        "long_standard": round(K_medium, 2),
        "long_aggressive": round(K_narrow, 2)
    }

# -----------------------------
# 買いプット理論価格 API（任意の K_long）
# -----------------------------
from scipy.stats import norm
import yfinance as yf
import math
import pandas as pd

@app.get("/api/bull_put_long_premium")
def bull_put_long_premium(K_long: float, T: float = 0.1, r: float = 0.001):

    # --- 日経225の現在値（info は不安定なので history から取得） ---
    ticker = yf.Ticker("^N225")
    try:
        S = ticker.history(period="1d")["Close"].iloc[-1]
    except:
        return {"error": "現在値が取得できません（history 失敗）"}

    if S is None:
        return {"error": "現在値が取得できません（None）"}

    # --- 過去3年のデータからボラティリティ推定 ---
    hist = ticker.history(period="1095d", interval="1d")
    if hist is None or hist.empty:
        return {"error": "yfinance がデータを取得できませんでした"}

    try:
        monthly = hist["Close"].resample("ME").last()
    except:
        return {"error": "月末終値の計算に失敗しました（resample エラー）"}

    returns = monthly.pct_change().dropna()
    if returns.empty:
        return {"error": "月次リターンが計算できません"}

    sigma_monthly = returns.std()
    sigma = sigma_monthly * math.sqrt(12)

    # --- BSモデル（プット） ---
    def put_price(S, K, T, r, sigma):
        try:
            d1 = (math.log(S/K) + (r + sigma*sigma/2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        except Exception as e:
            return None

    premium = put_price(S, K_long, T, r, sigma)
    if premium is None:
        return {"error": "BSモデル計算中にエラーが発生しました"}

    return {
        "S": round(float(S), 2),
        "sigma_estimated": round(float(sigma), 4),
        "K_long": round(float(K_long), 2),
        "premium_theoretical": round(float(premium), 2)
    }

# -----------------------------
# 買いコール理論価格 API（任意の K_long）
# -----------------------------
from scipy.stats import norm
import yfinance as yf
import math
import pandas as pd

@app.get("/api/bear_call_long_premium")
def bear_call_long_premium(K_long: float, T: float = 0.1, r: float = 0.001):

    ticker = yf.Ticker("^N225")

    # --- 現在値 S（info は不安定なので history から取得） ---
    try:
        S = ticker.history(period="1d")["Close"].iloc[-1]
    except:
        return {"error": "現在値が取得できません"}

    # --- 過去3年のデータからボラティリティ推定 ---
    hist = ticker.history(period="1095d", interval="1d")
    if hist is None or hist.empty:
        return {"error": "価格データが取得できません"}

    try:
        monthly = hist["Close"].resample("ME").last()
    except:
        return {"error": "月末終値の計算に失敗しました"}

    returns = monthly.pct_change().dropna()
    if returns.empty:
        return {"error": "月次リターンが計算できません"}

    sigma_monthly = returns.std()
    sigma = sigma_monthly * math.sqrt(12)

    # --- コール理論価格（BSモデル） ---
    def call_price(S, K, T, r, sigma):
        d1 = (math.log(S/K) + (r + sigma*sigma/2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        return S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)

    try:
        premium = call_price(S, K_long, T, r, sigma)
    except Exception as e:
        return {"error": f"BSモデル計算中にエラー: {e}"}

    return {
        "S": round(float(S), 2),
        "sigma_estimated": round(float(sigma), 4),
        "K_long": round(float(K_long), 2),
        "premium_theoretical": round(float(premium), 2)
    }

# -----------------------------
# market_insights API（最新データ対応）
# -----------------------------

import numpy as np
import yfinance as yf

# ▼ 最新の株価を取得（既存APIと同じロジック）
def get_nk225_price():
    ticker = yf.Ticker("^N225")
    info = ticker.info
    return info.get("regularMarketPrice")

# ▼ 最新のボラティリティを取得（既存APIと同じロジック）
def get_volatility_20d():
    ticker = yf.Ticker("^N225")
    hist = ticker.history(period="21d")

    if len(hist) < 2:
        return None

    close = hist["Close"].values
    log_returns = np.log(close[1:] / close[:-1])
    vol = np.std(log_returns) * np.sqrt(252)
    return float(vol)

# ▼ 過去1年の月末データを取得（自動）
def get_monthly_prices_1y():
    ticker = yf.Ticker("^N225")
    hist = ticker.history(period="1y")

    if hist.empty:
        return []

    # Pandas 2.2 以降は "M" が廃止 → "ME" を使う
    monthly = hist["Close"].resample("ME").last()

    return monthly.astype(int).tolist()

# ▼ 損益計算（既存ロジック）
def bull_put_pnl(S, K_short, K_long, credit):
    if S >= K_short:
        return credit
    if S <= K_long:
        return credit - (K_short - K_long)
    return credit - (K_short - S)

def bear_call_pnl(S, K_short, K_long, credit):
    if S <= K_short:
        return credit
    if S >= K_long:
        return credit - (K_long - K_short)
    return credit - (S - K_short)

def gpt_market_hint(S, sigma, avg_rise, avg_drop,
                    bull_win_rate, bear_win_rate, position_percent):

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    prompt = f"""
あなたはオプション戦略の専門家であり、同時に初心者向けの講師でもあります。

以下の市場データを分析し、
1. 最適な戦略（ベアコール / ブルプット など）
2. 専門家としての判断理由（2〜3行）
3. 初心者向けに、できるだけわかりやすく噛み砕いた解説（3〜6行）
4. 初心者が注意すべきポイント（1〜2行）

を JSON 形式で返してください。

返答は必ず次の形式のみ：

{{
  "strategy": "戦略名",
  "expert_reason": "専門家としての理由を2〜3行で",
  "beginner_explanation": "初心者向けに3〜6行でわかりやすく解説",
  "beginner_caution": "初心者が注意すべきポイントを1〜2行で"
}}

【市場データ】
株価: {S}
ボラティリティ: {sigma}
平均上昇率: {avg_rise}
平均下落率: {avg_drop}
ブルプット勝率: {bull_win_rate}
ベアコール勝率: {bear_win_rate}
現在値の位置: {position_percent}
"""

    try:
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = res.choices[0].message.content.strip()

        # JSON抽出
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        json_text = raw[json_start:json_end]
        json_text = json_text.replace("```json", "").replace("```", "").strip()

        return json.loads(json_text)["hint"]

    except Exception as e:
        return f"GPTエラー: {str(e)}"

# ▼ Main.py では router ではなく app を使う
@app.get("/api/market_insights")
def market_insights():

    # 現在の株価・ボラティリティ（最新）
    S = get_nk225_price()
    sigma = get_volatility_20d()

    # 過去1年の月末データ（最新）
    prices = get_monthly_prices_1y()
    if len(prices) < 12:
        return {"error": "過去データ不足"}

    # 月次リターン
    returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]

    avg_rise = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
    avg_drop = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
    max_rise = max(returns)
    max_drop = min(returns)

    # 過去1年レンジ
    range_low = min(prices)
    range_high = max(prices)
    position_percent = (S - range_low) / (range_high - range_low + 1e-9)

    # σ の位置（簡易）
    sigma_percentile = 0.5

    # バックテスト
    bull_results = []
    bear_results = []

    for i in range(len(prices)-1):
        S_entry = prices[i]
        S_exit = prices[i+1]

        # ブルプット
        K_short_bp = int(S_entry * 0.97)
        K_long_bp = K_short_bp - 500
        pnl_bp = bull_put_pnl(S_exit, K_short_bp, K_long_bp, 200)
        bull_results.append(pnl_bp)

        # ベアコール
        K_short_bc = int(S_entry * 1.03)
        K_long_bc = K_short_bc + 500
        pnl_bc = bear_call_pnl(S_exit, K_short_bc, K_long_bc, 200)
        bear_results.append(pnl_bc)

    bull_win_rate = sum(1 for x in bull_results if x > 0) / len(bull_results)
    bear_win_rate = sum(1 for x in bear_results if x > 0) / len(bear_results)

    # ▼ GPT 戦略ヒント生成（ここが新しい）
    hint_text = gpt_market_hint(
        S, sigma, avg_rise, avg_drop,
        bull_win_rate, bear_win_rate, position_percent
    )

    # ▼ JSON 返却
    return {
        "S": S,
        "sigma": sigma,
        "avg_rise": avg_rise,
        "avg_drop": avg_drop,
        "max_rise": max_rise,
        "max_drop": max_drop,
        "bull_put_win_rate": bull_win_rate,
        "bear_call_win_rate": bear_win_rate,
        "range_low": range_low,
        "range_high": range_high,
        "position_percent": position_percent,
        "sigma_percentile": sigma_percentile,
        "hint_text": hint_text
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

<h3>基本情報（Market Insights）</h3>

<div id="insightsSection" style="display:none;">
    <button onclick="loadMarketInsights()">最新情報を取得</button>

    <div id="insightsBox" style="
        background:#f2f2f2;
        padding:16px;
        border-radius:10px;
        margin-top:16px;
        font-size:22px;
    "></div>
</div>

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
    <pre id="bullPutResult"></pre>

    <button onclick="loadBullPutStrikes()">ストライク候補を表示</button>
    <pre id="bullPutStrikes"></pre>

    <button onclick="loadBullPutPremiums()">プレミアム候補を表示</button>
    <pre id="bullPutPremiums"></pre>

    <div style="font-size:20px; margin-top:10px; color:#333;">
    📘 買いプット候補の出し方<br>
    1. 売りプットのストライク（K_short）を入力してください。<br>
    2. 「買いプット候補を表示」を押すと、過去3年の下落率から自動計算されます。
    </div>

    <button onclick="loadBullPutLongCandidates()">買いプット候補を表示</button>
    <pre id="bullPutLongCandidates"></pre>

    <button onclick="calcBullPutLongPremium()">買いプットプレミアムを自動計算</button>
    
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
    <pre id="bearCallResult"></pre>

    <button onclick="loadBearCallStrikes()">ストライク候補を表示</button>
    <pre id="bearCallStrikes"></pre>

    <button onclick="loadBearCallPremiums()">プレミアム候補を表示</button>
    <pre id="bearCallPremiums"></pre>

    <div style="font-size:20px; margin-top:10px; color:#333;">
    📘 買いコール候補の出し方<br>
    1. 売りコールのストライク（K_short）を入力してください。<br>
    2. 「買いコール候補を表示」を押すと、過去3年の下落率から自動計算されます。
    </div>

    <button onclick="loadBearCallLongCandidates()">買いコール候補を表示</button>
    <pre id="bearCallLongCandidates"></pre>

    <button onclick="calcBearCallLongPremium()">買いコールプレミアムを自動計算</button>

    </div>

<hr>

<script>
async function onMenuChange(){
    const menu = document.getElementById("menu").value;

    // すべて非表示
    document.getElementById("insightsSection").style.display = "none";
    document.getElementById("bullPutBox").style.display = "none";
    document.getElementById("bearCallBox").style.display = "none";

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

async function loadBullPutLongCandidates(){
    const K_short = Number(document.getElementById("bp_K_short").value);

    const data = await fetch(`/api/bull_put_long_candidates?K_short=${K_short}`)
        .then(r => r.json());

    document.getElementById("bullPutLongCandidates").textContent =
        "📌 売りプット（ショート）: " + data.short_strike + "\\n" +
        "📌 平均下落率（3年・月末）: " + (data.avg_drop_rate * 100).toFixed(2) + "%\\n" +
        "📌 最大下落率（3年・月末）: " + (data.max_drop_rate * 100).toFixed(2) + "%\\n\\n" +

        "📌 買いプット候補（保険ロジック）\\n" +
        "安全（Wide: 最悪の下落に備える）: " + data.long_safe + "\\n" +
        "標準（Medium: 平均下落 ×2）: " + data.long_standard + "\\n" +
        "攻め（Narrow: 平均下落 ×1）: " + data.long_aggressive;
}

async function loadBearCallLongCandidates(){
    const K_short = Number(document.getElementById("bc_K_short").value);

    const data = await fetch(`/api/bear_call_long_candidates?K_short=${K_short}`)
        .then(r => r.json());

    document.getElementById("bearCallLongCandidates").textContent =
        "📌 売りコール（ショート）: " + data.short_strike + "\\n" +
        "📌 平均上昇率（3年・月末）: " + (data.avg_rise_rate * 100).toFixed(2) + "%\\n" +
        "📌 最大上昇率（3年・月末）: " + (data.max_rise_rate * 100).toFixed(2) + "%\\n\\n" +

        "📌 買いコール候補（保険ロジック）\\n" +
        "安全（Wide: 最悪の上昇に備える）: " + data.long_safe + "\\n" +
        "標準（Medium: 平均上昇 ×2）: " + data.long_standard + "\\n" +
        "攻め（Narrow: 平均上昇 ×1）: " + data.long_aggressive;
}

async function calcBullPutLongPremium(){
    const K_long = Number(document.getElementById("bp_K_long").value);
    const T = 0.1;  // 残存期間（年換算）

    const data = await fetch(`/api/bull_put_long_premium?K_long=${K_long}&T=${T}`)
        .then(r => r.json());

    if(data.error){
        alert("エラー: " + data.error);
        return;
    }

    // 買いプットのプレミアム欄に自動入力
    document.getElementById("bp_premium_long").value = data.premium_theoretical;
}

async function calcBearCallLongPremium(){
    const K_long = Number(document.getElementById("bc_K_long").value);
    const T = 0.1;

    const data = await fetch(`/api/bear_call_long_premium?K_long=${K_long}&T=${T}`)
        .then(r => r.json());

    if(data.error){
        alert("エラー: " + data.error);
        return;
    }

    // 買いコールのプレミアム欄に自動入力
    document.getElementById("bc_premium_long").value = data.premium_theoretical;
}

async function loadMarketInsights(){
    const info = await fetch("/api/market_insights").then(r=>r.json());

    document.getElementById("insightsBox").innerHTML = `
        <b>📌 株価 S:</b> ${info.S}<br>
        <b>📌 ボラティリティ σ:</b> ${info.sigma.toFixed(4)}<br><br>

        <b>【過去1年の傾向】</b><br>
        平均上昇率: ${(info.avg_rise * 100).toFixed(2)}%<br>
        平均下落率: ${(info.avg_drop * 100).toFixed(2)}%<br>
        最大上昇率: ${(info.max_rise * 100).toFixed(2)}%<br>
        最大下落率: ${(info.max_drop * 100).toFixed(2)}%<br><br>

        <b>【勝率（簡易バックテスト）】</b><br>
        ブルプット勝率: ${(info.bull_put_win_rate * 100).toFixed(1)}%<br>
        ベアコール勝率: ${(info.bear_call_win_rate * 100).toFixed(1)}%<br><br>

        <b>【現在の位置】</b><br>
        過去1年レンジ: ${info.range_low} ～ ${info.range_high}<br>
        現在値の位置: ${(info.position_percent * 100).toFixed(1)}%<br><br>

        <b>【戦略ヒント】</b><br>
        ${info.hint_text}
    `;
}

</script>

</body>
</html>
"""
