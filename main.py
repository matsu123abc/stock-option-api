from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import math

app = FastAPI()

# ==============================
# 1. 損益計算ロジック（ひながた）
# ==============================

def simulate_call_spread(
    spot: float,
    k_short: float,
    k_long: float,
    size: int,
    premium: float,
    iv: float,
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    adjustment: str
):
    """
    ひながた用の簡易シミュレーション。
    実際にはここに詳細な損益曲線・Greeks合成などを実装していく。
    """

    # ベースの損益（満期時のざっくり計算）
    # コールスプレッド：ショートK_short、ロングK_long
    # payoff = min(max(spot - k_short, 0), k_long - k_short) - premium
    intrinsic = max(spot - k_short, 0)
    intrinsic = min(intrinsic, k_long - k_short)
    base_pnl = intrinsic - premium

    # 調整案ごとの簡易評価（ひながた）
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

    # ここではひながたとして、Greeksはそのまま返すだけ
    greeks = {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "iv": iv,
    }

    return {
        "spot": spot,
        "k_short": k_short,
        "k_long": k_long,
        "size": size,
        "premium": premium,
        "base_pnl": base_pnl * size,
        "adjustment": adjustment,
        "adjustment_comment": adj_comment,
        "greeks": greeks,
    }

# ==============================
# 2. API エンドポイント
# ==============================

@app.post("/api/simulate")
def api_simulate(payload: dict):
    spot    = float(payload.get("spot", 0))
    k_short = float(payload.get("k_short", 0))
    k_long  = float(payload.get("k_long", 0))
    size    = int(payload.get("size", 1))
    premium = float(payload.get("premium", 0))

    iv      = float(payload.get("iv", 0))
    delta   = float(payload.get("delta", 0))
    gamma   = float(payload.get("gamma", 0))
    theta   = float(payload.get("theta", 0))
    vega    = float(payload.get("vega", 0))

    adjustment = payload.get("adjustment", "none")

    result = simulate_call_spread(
        spot, k_short, k_long, size, premium,
        iv, delta, gamma, theta, vega,
        adjustment
    )
    return result

# ==============================
# 3. UI（ひながた）
# ==============================

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
pre { background: #f0f0f0; padding: 10px; border-radius: 6px; white-space: pre-wrap; }
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

  <label>K_long（買いストライク）</label>
  <input id="k_long" type="number" step="5" />

  <label>size（枚数）</label>
  <input id="size" type="number" step="1" value="1" />

  <label>premium（受け取り or 支払い）</label>
  <input id="premium" type="number" step="0.1" />
</div>

<div class="section">
  <h3>③ 調整ロジック案（選択）</h3>
  <select id="adjustment">
    <option value="none">調整なし（現状維持）</option>
    <option value="roll_up">ロールアップ（ショートストライクを上にずらす）</option>
    <option value="roll_down">ロールダウン（ショートストライクを下にずらす）</option>
    <option value="leg_out">レッグ外し（ロング側を外すなど）</option>
  </select>
</div>

<button onclick="simulate()">シミュレーションする</button>

<h3>④ シミュレーション結果</h3>
<pre id="result"></pre>

<script>
async function simulate(){
  const payload = {
    spot:    parseFloat(document.getElementById("spot").value || 0),
    k_short: parseFloat(document.getElementById("k_short").value || 0),
    k_long:  parseFloat(document.getElementById("k_long").value || 0),
    size:    parseInt(document.getElementById("size").value || 1),
    premium: parseFloat(document.getElementById("premium").value || 0),

    iv:      parseFloat(document.getElementById("iv").value || 0),
    delta:   parseFloat(document.getElementById("delta").value || 0),
    gamma:   parseFloat(document.getElementById("gamma").value || 0),
    theta:   parseFloat(document.getElementById("theta").value || 0),
    vega:    parseFloat(document.getElementById("vega").value || 0),

    adjustment: document.getElementById("adjustment").value
  };

  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  document.getElementById("result").textContent =
    JSON.stringify(data, null, 2);
}
</script>

</body>
</html>
"""
