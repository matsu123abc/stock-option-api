from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import numpy as np

app = FastAPI()

# ============================================================
# 1. 損益計算ロジック（ひながた）
# ============================================================

def simulate_call_spread(
    spot,
    k_short,
    premium_short,
    k_long,
    premium_long,
    size,
    iv,
    delta,
    gamma,
    theta,
    vega,
    adjustment
):
    """
    ひながたのコールスプレッド損益計算。
    実際のロジックはここに肉付けしていく。
    """

    # ネットプレミアム（受け取りが +、支払いが -）
    net_premium = premium_short + premium_long

    # 最大利益
    max_profit = net_premium

    # 最大損失
    spread_width = k_long - k_short
    max_loss = spread_width - net_premium

    # ブレークイーブン
    breakeven = k_short + net_premium

    # 満期損益（spot を使った簡易版）
    intrinsic = max(spot - k_short, 0)
    intrinsic = min(intrinsic, spread_width)
    pnl = intrinsic - net_premium

    # 調整案コメント
    if adjustment == "none":
        adj_comment = "調整なし（現状維持）"
    elif adjustment == "roll_up":
        adj_comment = "ロールアップ案（ショートストライクを上にずらす）"
    elif adjustment == "roll_down":
        adj_comment = "ロールダウン案（ショートストライクを下にずらす）"
    elif adjustment == "leg_out":
        adj_comment = "レッグ外し案（ロング側を外すなど）"
    else:
        adj_comment = "不明な調整案"

    # Greeks（ひながた）
    greeks = {
        "iv": iv,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega
    }

    return {
        "spot": spot,
        "k_short": k_short,
        "premium_short": premium_short,
        "k_long": k_long,
        "premium_long": premium_long,
        "size": size,
        "net_premium": net_premium,
        "max_profit": max_profit * size,
        "max_loss": max_loss * size,
        "breakeven": breakeven,
        "pnl_at_spot": pnl * size,
        "adjustment": adjustment,
        "adjustment_comment": adj_comment,
        "greeks": greeks
    }

# ============================================================
# 2. API エンドポイント
# ============================================================

@app.post("/api/simulate")
def api_simulate(payload: dict):

    spot = float(payload.get("spot", 0))

    k_short = float(payload.get("k_short", 0))
    premium_short = float(payload.get("premium_short", 0))

    k_long = float(payload.get("k_long", 0))
    premium_long = float(payload.get("premium_long", 0))

    size = int(payload.get("size", 1))

    iv = float(payload.get("iv", 0))
    delta = float(payload.get("delta", 0))
    gamma = float(payload.get("gamma", 0))
    theta = float(payload.get("theta", 0))
    vega = float(payload.get("vega", 0))

    adjustment = payload.get("adjustment", "none")

    result = simulate_call_spread(
        spot,
        k_short,
        premium_short,
        k_long,
        premium_long,
        size,
        iv,
        delta,
        gamma,
        theta,
        vega,
        adjustment
    )

    return result

# ============================================================
# 3. UI（ひながた）
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>コールスプレッド調整シミュレーション（ひながた）</title>
<style>
body { font-family: sans-serif; padding: 20px; font-size: 16px; }
input, select { width: 100%; padding: 6px; margin: 4px 0; font-size: 16px; }
button { width: 100%; padding: 10px; margin-top: 10px; font-size: 16px; }
table { border-collapse: collapse; margin-top: 10px; }
th, td { border: 1px solid #999; padding: 6px 10px; }
.section { margin-bottom: 16px; }
h2 { margin-top: 0; }
</style>
</head>
<body>

<h2>📌 コールスプレッド調整シミュレーション（ひながた）</h2>

<div class="section">
  <h3>① 市場データ（手入力）</h3>
  <label>現在の日経225株価（spot）</label>
  <input id="spot" type="number" step="0.1" />

  <label>IV</label>
  <input id="iv" type="number" step="0.01" />

  <label>Delta</label>
  <input id="delta" type="number" step="0.0001" />

  <label>Gamma</label>
  <input id="gamma" type="number" step="0.0001" />

  <label>Theta</label>
  <input id="theta" type="number" step="0.0001" />

  <label>Vega</label>
  <input id="vega" type="number" step="0.0001" />
</div>

<div class="section">
  <h3>② 戦略条件（手入力）</h3>
  <label>K_short（売りストライク）</label>
  <input id="k_short" type="number" step="5" />

  <label>premium_short（受け取り）</label>
  <input id="premium_short" type="number" step="0.1" />

  <label>K_long（買いストライク）</label>
  <input id="k_long" type="number" step="5" />

  <label>premium_long（支払い）</label>
  <input id="premium_long" type="number" step="0.1" />

  <label>size（枚数）</label>
  <input id="size" type="number" step="1" value="1" />
</div>

<div class="section">
  <h3>③ 調整ロジック案（選択）</h3>
  <select id="adjustment">
    <option value="none">調整なし（現状維持）</option>
    <option value="roll_up">ロールアップ</option>
    <option value="roll_down">ロールダウン</option>
    <option value="leg_out">レッグ外し</option>
  </select>
</div>

<button onclick="simulate()">シミュレーションする</button>

<h3>④ シミュレーション結果</h3>
<div id="result_area"></div>

<script>
async function simulate(){
  const payload = {
    spot: parseFloat(document.getElementById("spot").value || 0),

    k_short: parseFloat(document.getElementById("k_short").value || 0),
    premium_short: parseFloat(document.getElementById("premium_short").value || 0),

    k_long: parseFloat(document.getElementById("k_long").value || 0),
    premium_long: parseFloat(document.getElementById("premium_long").value || 0),

    size: parseInt(document.getElementById("size").value || 1),

    iv: parseFloat(document.getElementById("iv").value || 0),
    delta: parseFloat(document.getElementById("delta").value || 0),
    gamma: parseFloat(document.getElementById("gamma").value || 0),
    theta: parseFloat(document.getElementById("theta").value || 0),
    vega: parseFloat(document.getElementById("vega").value || 0),

    adjustment: document.getElementById("adjustment").value
  };

  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const data = await res.json();

  let html = "";

  // 基本損益指標
  html += "<h4>基本損益指標</h4>";
  html += "<table><tr><th>項目</th><th>値</th></tr>";
  html += `<tr><td>ネットプレミアム</td><td>${data.net_premium}</td></tr>`;
  html += `<tr><td>最大利益</td><td>${data.max_profit}</td></tr>`;
  html += `<tr><td>最大損失</td><td>${data.max_loss}</td></tr>`;
  html += `<tr><td>ブレークイーブン</td><td>${data.breakeven}</td></tr>`;
  html += `<tr><td>現在値での損益</td><td>${data.pnl_at_spot}</td></tr>`;
  html += "</table>";

  // Greeks
  html += "<h4>Greeks</h4>";
  html += "<table><tr><th>指標</th><th>値</th></tr>";
  html += `<tr><td>IV</td><td>${data.greeks.iv}</td></tr>`;
  html += `<tr><td>Delta</td><td>${data.greeks.delta}</td></tr>`;
  html += `<tr><td>Gamma</td><td>${data.greeks.gamma}</td></tr>`;
  html += `<tr><td>Theta</td><td>${data.greeks.theta}</td></tr>`;
  html += `<tr><td>Vega</td><td>${data.greeks.vega}</td></tr>`;
  html += "</table>";

  // 調整案コメント
  html += "<h4>調整案</h4>";
  html += `<p>${data.adjustment_comment}</p>`;

  document.getElementById("result_area").innerHTML = html;
}
</script>

</body>
</html>
"""
