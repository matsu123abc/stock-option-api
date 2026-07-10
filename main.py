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
    コールスプレッド損益計算（プレミアムは両方とも正の値で入力）。
    premium_short: 受け取り（+）
    premium_long:  支払い（+）として入力し、計算時に差し引く。
    """

    # ネットプレミアム（受け取り − 支払い）
    net_premium = premium_short - premium_long  # 例: 2300 - 1830 = 470

    # スプレッド幅
    spread_width = k_long - k_short  # 例: 70000 - 69000 = 1000

    # 最大利益（ネットプレミアム）
    max_profit = net_premium  # 例: 470

    # 最大損失（スプレッド幅 − ネットプレミアム）
    max_loss = spread_width - net_premium  # 例: 1000 - 470 = 530

    # ブレークイーブン（K_short + ネットプレミアム）
    breakeven = k_short + net_premium  # 例: 69000 + 470 = 69470

    # 満期損益（現在の spot での損益）
    intrinsic = max(spot - k_short, 0)
    intrinsic = min(intrinsic, spread_width)
    pnl = intrinsic - net_premium  # 例: spot=68557 → intrinsic=0 → -470

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
<title>コールスプレッド調整シミュレーション（満期損益表）</title>
<style>
body { font-family: sans-serif; padding: 20px; font-size: 16px; }
input, select { width: 100%; padding: 6px; margin: 4px 0; font-size: 16px; }
button { width: 100%; padding: 10px; margin-top: 10px; font-size: 16px; }
table { border-collapse: collapse; margin-top: 10px; width: 100%; }
th, td { border: 1px solid #999; padding: 6px 10px; text-align: right; }
th { background: #f7f7f7; text-align: center; }
.section { margin-bottom: 16px; }
h2 { margin-top: 0; }
.highlight { background:#fff3cd; font-weight:700; }
.small { font-size: 0.9em; color:#555; text-align:left; }
</style>
</head>
<body>

<h2>📌 コールスプレッド調整シミュレーション（満期損益表）</h2>

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
function fmt(n){
  if (n === null || n === undefined || isNaN(n)) return "-";
  const sign = n > 0 ? "+" : (n < 0 ? "−" : "");
  const abs = Math.abs(Math.round(n));
  return sign + abs.toLocaleString();
}

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

  // 基本指標表示
  let html = "";
  html += "<table><tr><th class='small'>項目</th><th>値</th></tr>";
  html += `<tr><td class='small'>ネットプレミアム</td><td>${fmt(data.net_premium)}</td></tr>`;
  html += `<tr><td class='small'>最大利益</td><td>${fmt(data.max_profit)}</td></tr>`;
  html += `<tr><td class='small'>最大損失</td><td>${fmt(-Math.abs(data.max_loss))}</td></tr>`;
  html += `<tr><td class='small'>ブレークイーブン</td><td>${(data.breakeven).toLocaleString()}</td></tr>`;
  html += `<tr><td class='small'>現在値での満期想定損益</td><td>${fmt(data.net_premium - Math.min(Math.max(data.spot - data.k_short,0), data.k_long - data.k_short))}</td></tr>`;
  html += "</table>";

  // Greeks
  html += "<h4>Greeks</h4>";
  html += "<table><tr><th class='small'>指標</th><th>値</th></tr>";
  html += `<tr><td class='small'>IV</td><td>${data.greeks.iv}</td></tr>`;
  html += `<tr><td class='small'>Delta</td><td>${data.greeks.delta}</td></tr>`;
  html += `<tr><td class='small'>Gamma</td><td>${data.greeks.gamma}</td></tr>`;
  html += `<tr><td class='small'>Theta</td><td>${data.greeks.theta}</td></tr>`;
  html += `<tr><td class='small'>Vega</td><td>${data.greeks.vega}</td></tr>`;
  html += "</table>";

  // 満期損益表（代表的な SQ 値）
  html += "<h4>満期損益表（代表的な SQ 値）</h4>";

  // SQ 値のレンジを作る（K_short -2000 〜 K_long +2000、刻み 500）
  const kshort = Number(data.k_short);
  const klong = Number(data.k_long);
  const net = Number(data.net_premium);
  const spread = klong - kshort;

  let sqs = [];
  const start = Math.max(0, kshort - 2000);
  const end = klong + 2000;
  for(let s = start; s <= end; s += 500) sqs.push(s);

  // 重要点を確実に含める（spot, k_short, breakeven, k_long）
  const important = [Math.round(data.spot), kshort, data.breakeven, klong];
  important.forEach(v => { if (!sqs.includes(v)) sqs.push(v); });

  // ソート
  sqs = Array.from(new Set(sqs)).sort((a,b)=>a-b);

  // テーブル作成
  html += "<table><tr><th>SQ</th><th>intrinsic</th><th>満期損益</th></tr>";
  for(const sq of sqs){
    const intrinsic = Math.min(Math.max(sq - kshort, 0), spread);
    // 満期損益の定義：受取クレジット − intrinsic（受取がプラス）
    const pnl = net - intrinsic;
    // 強調条件
    const isSpot = (sq === Math.round(data.spot));
    const isKshort = (sq === kshort);
    const isKlong = (sq === klong);
    const isBE = (sq === data.breakeven);

    let trClass = "";
    if (isBE) trClass = "highlight";
    html += `<tr${trClass? " class='"+trClass+"'" : ""}>`;
    html += `<td style="text-align:center">${sq.toLocaleString()}</td>`;
    html += `<td>${intrinsic.toLocaleString()}</td>`;
    html += `<td>${fmt(pnl)}</td>`;
    html += "</tr>";
  }
  html += "</table>";

  // 補足説明
  html += "<p class='small'>注：満期損益は「受取クレジット − intrinsic」で計算しています。正の値は利益、負の値は損失を示します。</p>";

  document.getElementById("result_area").innerHTML = html;
}
</script>

</body>
</html>
"""
