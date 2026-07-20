from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

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
    adjustment,
    # short out
    close_short_now=False,
    market_price_short=0.0,
    commission_per_leg_short=0.0,
    slippage_short=0.0,
    # roll params (common)
    roll_amount=1000.0,
    assumed_premium_change=0.0,
    # detailed roll inputs
    use_detailed_roll=False,
    buyback_price_short=0.0,
    buyback_commission_short=0.0,
    buyback_slippage_short=0.0,
    new_sell_premium=0.0
):
    """
    コールスプレッド損益計算（premium は正の値で入力）
    - short_out: ショート買戻し（ロングのみ残す）
    - roll_up / roll_down: ロール（簡易 or 詳細）
      * 簡易: net_premium' = net_premium + assumed_premium_change
      * 詳細: net_premium' = net_premium - buyback_cost + new_sell_premium
    """

    # 基本スプレッド指標
    net_premium = premium_short - premium_long
    spread_width = k_long - k_short
    max_profit = net_premium
    max_loss = spread_width - net_premium
    breakeven = k_short + net_premium

    # 現在値での満期想定損益（スプレッド）
    intrinsic_spot = max(spot - k_short, 0)
    intrinsic_spot = min(intrinsic_spot, spread_width)
    pnl_at_spot = net_premium - intrinsic_spot

    greeks = {
        "iv": iv,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega
    }

    result = {
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
        "pnl_at_spot": pnl_at_spot * size,
        "adjustment": adjustment,
        "adjustment_comment": "",
        "greeks": greeks,
        "pnl_curve": [],
        "shortout": None,
        "roll_up": None,
        "roll_down": None
    }

    # 調整コメント
    if adjustment == "none":
        result["adjustment_comment"] = "調整なし（現状維持）"
    elif adjustment == "roll_up":
        result["adjustment_comment"] = "ロールアップ案"
    elif adjustment == "roll_down":
        result["adjustment_comment"] = "ロールダウン案"
    elif adjustment == "short_out":
        result["adjustment_comment"] = "ショート外し案（ショートを買い戻す）"
    else:
        result["adjustment_comment"] = "不明な調整案"

    # 満期損益表（スプレッド）
    start = max(0, int(k_short) - 2000)
    end = int(k_long) + 2000
    step = 500
    sqs = list(range(start, end + 1, step))
    for v in [int(round(spot)), int(k_short), int(breakeven), int(k_long)]:
        if v not in sqs:
            sqs.append(v)
    sqs = sorted(set(sqs))

    pnl_curve = []
    for sq in sqs:
        intrinsic_sq = min(max(sq - k_short, 0), spread_width)
        pnl_sq = net_premium - intrinsic_sq
        pnl_curve.append({"sq": sq, "intrinsic": intrinsic_sq, "pnl": pnl_sq * size})
    result["pnl_curve"] = pnl_curve

    # ショート外しシナリオ（既存）
    if adjustment == "short_out":
        if close_short_now:
            buyback_cost = market_price_short + commission_per_leg_short + slippage_short
            cash_after = net_premium - buyback_cost
            shortout_curve = []
            for sq in sqs:
                payoff_long = max(sq - k_long, 0)
                pnl_sq = cash_after + payoff_long
                shortout_curve.append({"sq": sq, "payoff_long": payoff_long, "pnl": pnl_sq * size})
            result["shortout"] = {
                "close_short_now": True,
                "market_price_short": market_price_short,
                "commission_per_leg_short": commission_per_leg_short,
                "slippage_short": slippage_short,
                "buyback_cost": buyback_cost,
                "cash_after": cash_after * size,
                "shortout_curve": shortout_curve
            }
        else:
            result["shortout"] = {"close_short_now": False, "comment": "ショートを今買い戻さない設定です。"}

    # ロール計算ヘルパー
    def calc_roll(k_short_new, k_long_new, net_premium_new):
        spread_w = k_long_new - k_short_new
        curve = []
        for sq in sqs:
            intrinsic_sq = min(max(sq - k_short_new, 0), spread_w)
            pnl_sq = net_premium_new - intrinsic_sq
            curve.append({"sq": sq, "intrinsic": intrinsic_sq, "pnl": pnl_sq * size})
        intrinsic_spot_new = min(max(spot - k_short_new, 0), spread_w)
        pnl_at_spot_new = (net_premium_new - intrinsic_spot_new) * size
        return {
            "k_short": k_short_new,
            "k_long": k_long_new,
            "net_premium": net_premium_new,
            "spread_width": spread_w,
            "pnl_at_spot": pnl_at_spot_new,
            "pnl_curve": curve
        }

    # ロールアップ／ロールダウン
    if adjustment in ("roll_up", "roll_down"):
        # 新しいストライク
        direction = 1 if adjustment == "roll_up" else -1
        k_short_new = k_short + direction * roll_amount
        k_long_new = k_long + direction * roll_amount

        # net_premium の算出方法
        if use_detailed_roll:
            # 買戻しコスト（ショート買戻し）を計算
            buyback_cost = buyback_price_short + buyback_commission_short + buyback_slippage_short
            # 現金フロー：既存 net_premium から買戻しを引き、新規売りで受け取る
            net_premium_new = net_premium - buyback_cost + new_sell_premium
            roll_detail = {
                "use_detailed_roll": True,
                "buyback_price_short": buyback_price_short,
                "buyback_commission_short": buyback_commission_short,
                "buyback_slippage_short": buyback_slippage_short,
                "buyback_cost": buyback_cost,
                "new_sell_premium": new_sell_premium,
                "net_premium_new": net_premium_new
            }
        else:
            # 簡易ロール：net_premium に assumed_premium_change を加える
            net_premium_new = net_premium + assumed_premium_change
            roll_detail = {
                "use_detailed_roll": False,
                "assumed_premium_change": assumed_premium_change,
                "net_premium_new": net_premium_new
            }

        # 計算して結果格納
        roll_result = calc_roll(k_short_new, k_long_new, net_premium_new)
        roll_result.update(roll_detail)
        if adjustment == "roll_up":
            result["roll_up"] = roll_result
        else:
            result["roll_down"] = roll_result

    return result

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

    # short out params
    close_short_now = bool(payload.get("close_short_now", False))
    market_price_short = float(payload.get("market_price_short", 0.0))
    commission_per_leg_short = float(payload.get("commission_per_leg_short", 0.0))
    slippage_short = float(payload.get("slippage_short", 0.0))

    # roll params
    roll_amount = float(payload.get("roll_amount", 1000))
    assumed_premium_change = float(payload.get("assumed_premium_change", 0.0))
    use_detailed_roll = bool(payload.get("use_detailed_roll", False))
    buyback_price_short = float(payload.get("buyback_price_short", 0.0))
    buyback_commission_short = float(payload.get("buyback_commission_short", 0.0))
    buyback_slippage_short = float(payload.get("buyback_slippage_short", 0.0))
    new_sell_premium = float(payload.get("new_sell_premium", 0.0))

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
        adjustment,
        close_short_now=close_short_now,
        market_price_short=market_price_short,
        commission_per_leg_short=commission_per_leg_short,
        slippage_short=slippage_short,
        roll_amount=roll_amount,
        assumed_premium_change=assumed_premium_change,
        use_detailed_roll=use_detailed_roll,
        buyback_price_short=buyback_price_short,
        buyback_commission_short=buyback_commission_short,
        buyback_slippage_short=buyback_slippage_short,
        new_sell_premium=new_sell_premium
    )

    return result

# ============================================================
# ) HTML（スマホ最適化 UI）
# ============================================================
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">

<!-- スマホ最適化の必須設定 -->
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">

<title>コールスプレッド シミュレーション（詳細ロール対応）</title>

<style>
/* PC用（デフォルト） */
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
.hidden { display: none; }

/* ▼ スマホ対応（画面幅 600px 以下） ▼ */
@media (max-width: 600px) {

  body {
    padding: 10px;
    font-size: 14px;
    line-height: 1.4;
  }

  h2, h3, h4, h5 {
    font-size: 1.2em;
    margin-bottom: 10px;
    margin-top: 18px;
  }

  /* ★ ラベル文字を大きくする（最重要） */
  label {
    font-weight: 600;
    margin-top: 12px;
    display: block;
    font-size: 20px;   /* ← ここを追加 */
  }

  input, select {
    width: 100%;
    padding: 12px;
    margin: 10px 0;
    font-size: 18px;
    border-radius: 8px;
  }

  button {
    width: 100%;
    padding: 16px;
    font-size: 20px;
    margin-top: 18px;
    background: #007bff;
    color: white;
    border-radius: 8px;
    border: none;
  }

  table {
    font-size: 14px;
    display: block;
    overflow-x: auto;
    margin-top: 16px;
  }

  th, td {
    padding: 10px;
  }

  .section {
    margin-bottom: 24px;
  }

  #result_area {
    margin-top: 20px;
    font-size: 15px;
  }
}
/* ▲ スマホ対応 ▲ */

</style>
   
</head>
<body>

<h2>📌 コールスプレッド シミュレーション（詳細ロール対応）</h2>

<div class="section">
  <h3>① 市場データ（手入力）</h3>

  <label>現在の日経225株価（spot）</label>
  <input id="spot" type="number" step="0.1" placeholder="例：68557" />

  <label>IV</label>
  <input id="iv" type="number" step="0.01" placeholder="例：31.4" />

  <label>Delta</label>
  <input id="delta" type="number" step="0.0001" placeholder="例：0.43" />

  <label>Gamma</label>
  <input id="gamma" type="number" step="0.0001" placeholder="例：0.0000" />

  <label>Theta</label>
  <input id="theta" type="number" step="0.0001" placeholder="例：-37" />

  <label>Vega</label>
  <input id="vega" type="number" step="0.0001" placeholder="例：83" />
</div>

<div class="section">
  <h3>② 戦略条件（手入力）</h3>

  <label>K_short（売りストライク）</label>
  <input id="k_short" type="number" step="5" placeholder="例：69000" />

  <label>premium_short（受け取り）</label>
  <input id="premium_short" type="number" step="0.1" placeholder="例：2300" />

  <label>K_long（買いストライク）</label>
  <input id="k_long" type="number" step="5" placeholder="例：70000" />

  <label>premium_long（支払い）</label>
  <input id="premium_long" type="number" step="0.1" placeholder="例：1830" />

  <label>size（枚数）</label>
  <input id="size" type="number" step="1" value="1" placeholder="例：1" />
</div>


<div class="section">
  <h3>③ 調整ロジック案（選択）</h3>
  <select id="adjustment" onchange="onAdjustmentChange()">
    <option value="none">調整なし（現状維持）</option>
    <option value="roll_up">ロールアップ</option>
    <option value="roll_down">ロールダウン</option>
    <option value="short_out">ショート外し（ショートを買い戻す）</option>
  </select>

  <!-- ロール追加入力 -->
  <div id="roll_inputs" class="hidden">
    <h4>ロールの追加入力</h4>
    <label>ロール幅 roll_amount（例 1000）</label>
    <input id="roll_amount" type="number" step="1" value="1000" />

    <label>簡易想定差 assumed_premium_change（net_premium に直接加算）</label>
    <input id="assumed_premium_change" type="number" step="1" value="0" />

    <label><input id="use_detailed_roll" type="checkbox" /> 詳細ロールを使う（買戻しコストと新規売りプレミアムを個別入力）</label>

    <div id="detailed_roll_inputs" class="hidden">
      <h5>買戻し（ショート）想定</h5>
      <label>ショート買戻し想定価格 buyback_price_short</label>
      <input id="buyback_price_short" type="number" step="0.1" />

      <label>買戻し手数料 buyback_commission_short（片側）</label>
      <input id="buyback_commission_short" type="number" step="0.1" value="0" />

      <label>買戻しスリッページ buyback_slippage_short</label>
      <input id="buyback_slippage_short" type="number" step="0.1" value="0" />

      <h5>新規売り（上のストライク）想定</h5>
      <label>新規売りプレミアム new_sell_premium（上のストライクで受け取る想定）</label>
      <input id="new_sell_premium" type="number" step="0.1" value="0" />

      <p class="small">注：詳細ロールでは買戻しコストと新規売りプレミアムを個別に入力し、現金フローを正確に計算します。</p>
    </div>
  </div>

  <!-- ショート外し用追加入力欄 -->
  <div id="shortout_inputs" class="hidden">
    <h4>ショート外しの追加入力</h4>
    <label><input id="close_short_now" type="checkbox" /> ショートを今すぐ買い戻す（チェックすると下の入力を使用）</label>

    <div id="shortout_now_inputs" class="hidden">
      <label>ショート買戻し想定価格 market_price_short</label>
      <input id="market_price_short" type="number" step="0.1" />

      <label>手数料 commission_per_leg_short（買戻しにかかる片側手数料）</label>
      <input id="commission_per_leg_short" type="number" step="0.1" value="0" />

      <label>スリッページ slippage_short（買戻し時の不利な価格変動）</label>
      <input id="slippage_short" type="number" step="0.1" value="0" />
    </div>

    <p class="small">注：ショートを買い戻すと裸ロング（ロングのみ保有）になります。買戻しコストが大きいと即時損失が発生します。</p>
  </div>
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

function onAdjustmentChange(){
  const adj = document.getElementById("adjustment").value;
  const rollDiv = document.getElementById("roll_inputs");
  const shortDiv = document.getElementById("shortout_inputs");
  if (adj === "roll_up" || adj === "roll_down"){
    rollDiv.classList.remove("hidden");
  } else {
    rollDiv.classList.add("hidden");
    document.getElementById("use_detailed_roll").checked = false;
    document.getElementById("detailed_roll_inputs").classList.add("hidden");
  }
  if (adj === "short_out"){
    shortDiv.classList.remove("hidden");
  } else {
    shortDiv.classList.add("hidden");
    document.getElementById("close_short_now").checked = false;
    document.getElementById("shortout_now_inputs").classList.add("hidden");
  }
}

document.getElementById("use_detailed_roll")?.addEventListener("change", function(){
  const checked = this.checked;
  const nowDiv = document.getElementById("detailed_roll_inputs");
  if (checked) nowDiv.classList.remove("hidden"); else nowDiv.classList.add("hidden");
});

document.getElementById("close_short_now")?.addEventListener("change", function(){
  const checked = this.checked;
  const nowDiv = document.getElementById("shortout_now_inputs");
  if (checked) nowDiv.classList.remove("hidden"); else nowDiv.classList.add("hidden");
});

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

    adjustment: document.getElementById("adjustment").value,

    // short out params
    close_short_now: document.getElementById("close_short_now").checked,
    market_price_short: parseFloat(document.getElementById("market_price_short").value || 0),
    commission_per_leg_short: parseFloat(document.getElementById("commission_per_leg_short").value || 0),
    slippage_short: parseFloat(document.getElementById("slippage_short").value || 0),

    // roll params
    roll_amount: parseFloat(document.getElementById("roll_amount").value || 1000),
    assumed_premium_change: parseFloat(document.getElementById("assumed_premium_change").value || 0),
    use_detailed_roll: document.getElementById("use_detailed_roll").checked,
    buyback_price_short: parseFloat(document.getElementById("buyback_price_short").value || 0),
    buyback_commission_short: parseFloat(document.getElementById("buyback_commission_short").value || 0),
    buyback_slippage_short: parseFloat(document.getElementById("buyback_slippage_short").value || 0),
    new_sell_premium: parseFloat(document.getElementById("new_sell_premium").value || 0)
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
  html += `<tr><td class='small'>現在値での満期想定損益</td><td>${fmt(data.pnl_at_spot)}</td></tr>`;
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

  // 満期損益表（スプレッド）
  html += "<h4>満期損益表（スプレッド）</h4>";
  html += "<table><tr><th>SQ</th><th>intrinsic</th><th>満期損益</th></tr>";
  data.pnl_curve.forEach(row => {
    const isBE = row.sq === data.breakeven;
    html += `<tr${isBE? " class='highlight'": ""}><td style='text-align:center'>${row.sq.toLocaleString()}</td><td>${row.intrinsic.toLocaleString()}</td><td>${fmt(row.pnl)}</td></tr>`;
  });
  html += "</table>";

  // ショート外し結果
  if (data.shortout){
    html += "<h4>ショート外しシナリオ</h4>";
    if (data.shortout.close_short_now){
      html += "<table><tr><th class='small'>項目</th><th>値</th></tr>";
      html += `<tr><td class='small'>ショート買戻し想定価格</td><td>${(data.shortout.market_price_short).toLocaleString()}</td></tr>`;
      html += `<tr><td class='small'>スリッページ</td><td>${(data.shortout.slippage_short).toLocaleString()}</td></tr>`;
      html += `<tr><td class='small'>手数料（片側）</td><td>${(data.shortout.commission_per_leg_short).toLocaleString()}</td></tr>`;
      html += `<tr><td class='small'>買戻しコスト合計</td><td>${(Math.round(data.shortout.buyback_cost)).toLocaleString()}</td></tr>`;
      html += `<tr><td class='small'>買戻し後の手元現金</td><td>${fmt(data.shortout.cash_after)}</td></tr>`;
      html += "</table>";

      html += "<h5>ショート外し後の満期損益表（ロングのみ）</h5>";
      html += "<table><tr><th>SQ</th><th>payoff_long</th><th>満期損益</th></tr>";
      data.shortout.shortout_curve.forEach(row => {
        html += `<tr><td style='text-align:center'>${row.sq.toLocaleString()}</td><td>${row.payoff_long.toLocaleString()}</td><td>${fmt(row.pnl)}</td></tr>`;
      });
      html += "</table>";
    } else {
      html += `<p class='small'>${data.shortout.comment}</p>`;
    }
  }

  // ロール結果（存在する場合）
  if (data.roll_up){
    html += "<h4>ロールアップ結果</h4>";
    html += "<table><tr><th class='small'>項目</th><th>値</th></tr>";
    html += `<tr><td class='small'>新売りストライク</td><td>${(data.roll_up.k_short).toLocaleString()}</td></tr>`;
    html += `<tr><td class='small'>新買いストライク</td><td>${(data.roll_up.k_long).toLocaleString()}</td></tr>`;
    html += `<tr><td class='small'>想定 net_premium</td><td>${fmt(data.roll_up.net_premium)}</td></tr>`;
    html += `<tr><td class='small'>現在値での満期想定損益</td><td>${fmt(data.roll_up.pnl_at_spot)}</td></tr>`;
    html += "</table>";

    html += "<h5>ロールアップ満期損益表</h5>";
    html += "<table><tr><th>SQ</th><th>intrinsic</th><th>満期損益</th></tr>";
    data.roll_up.pnl_curve.forEach(row => {
      html += `<tr><td style='text-align:center'>${row.sq.toLocaleString()}</td><td>${row.intrinsic.toLocaleString()}</td><td>${fmt(row.pnl)}</td></tr>`;
    });
    html += "</table>";
  }

  if (data.roll_down){
    html += "<h4>ロールダウン結果</h4>";
    html += "<table><tr><th class='small'>項目</th><th>値</th></tr>";
    html += `<tr><td class='small'>新売りストライク</td><td>${(data.roll_down.k_short).toLocaleString()}</td></tr>`;
    html += `<tr><td class='small'>新買いストライク</td><td>${(data.roll_down.k_long).toLocaleString()}</td></tr>`;
    html += `<tr><td class='small'>想定 net_premium</td><td>${fmt(data.roll_down.net_premium)}</td></tr>`;
    html += `<tr><td class='small'>現在値での満期想定損益</td><td>${fmt(data.roll_down.pnl_at_spot)}</td></tr>`;
    html += "</table>";

    html += "<h5>ロールダウン満期損益表</h5>";
    html += "<table><tr><th>SQ</th><th>intrinsic</th><th>満期損益</th></tr>";
    data.roll_down.pnl_curve.forEach(row => {
      html += `<tr><td style='text-align:center'>${row.sq.toLocaleString()}</td><td>${row.intrinsic.toLocaleString()}</td><td>${fmt(row.pnl)}</td></tr>`;
    });
    html += "</table>";
  }

  document.getElementById("result_area").innerHTML = html;
}
</script>

</body>
</html>
"""

# uvicorn で直接起動する場合のエントリポイント（任意）
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
