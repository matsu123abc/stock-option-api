from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import re
import jaconv

app = FastAPI()

# ---------------------------------------------------------
# ① 前処理
# ---------------------------------------------------------
def preprocess(raw: str):
    raw = jaconv.z2h(raw, digit=True, ascii=True)
    lines = raw.split("\n")
    clean = [l.strip() for l in lines if l.strip() != ""]
    return clean

# ---------------------------------------------------------
# ② 行分類
# ---------------------------------------------------------
def classify_line(line: str):
    if "IV" in line:
        return "iv"
    if "Delta" in line or "Gamma" in line or "Theta" in line or "Vega" in line:
        return "greeks"
    if "売気配" in line or "買気配" in line:
        return "orderbook"
    if "現値" in line:
        return "last"
    if re.search(r"\d{5}", line):
        return "strike"
    return "other"

# ---------------------------------------------------------
# ③ 各行のパース
# ---------------------------------------------------------
def parse_strike(line):
    m = re.search(r"(\d{5})", line)
    return int(m.group(1)) if m else None

def parse_iv(line):
    m = re.search(r"IV\s+([\d\.]+)", line)
    return float(m.group(1)) if m else None

def parse_greeks(line):
    delta = float(re.search(r"Delta\s+([\d\.\-]+)", line).group(1))
    gamma = float(re.search(r"Gamma\s+([\d\.\-]+)", line).group(1))
    theta = float(re.search(r"Theta\s+([\d\.\-]+)", line).group(1))
    vega  = float(re.search(r"Vega\s+([\d\.\-]+)", line).group(1))
    return delta, gamma, theta, vega

def parse_orderbook(line):
    nums = re.findall(r"(\d+\.?\d*)\s*\((\d+)\)", line)
    if len(nums) >= 2:
        bid, bid_size = nums[0]
        ask, ask_size = nums[1]
        return float(bid), int(bid_size), float(ask), int(ask_size)
    return None

def parse_last(line):
    m = re.search(r"現値\s+([\d\.]+)", line)
    return float(m.group(1)) if m else None

def parse_volume(line):
    m = re.search(r"売買高\s+(\d+)", line)
    return int(m.group(1)) if m else None

# ---------------------------------------------------------
# ④ 行をまとめて JSON 化
# ---------------------------------------------------------
def parse_lines(lines):
    result = {
        "strike": None,
        "iv": None,
        "delta": None,
        "gamma": None,
        "theta": None,
        "vega": None,
        "bid": None,
        "bid_size": None,
        "ask": None,
        "ask_size": None,
        "last": None,
        "volume": None
    }

    for line in lines:
        t = classify_line(line)

        if t == "strike":
            result["strike"] = parse_strike(line)

        elif t == "iv":
            result["iv"] = parse_iv(line)

        elif t == "greeks":
            delta, gamma, theta, vega = parse_greeks(line)
            result.update({
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega
            })

        elif t == "orderbook":
            parsed = parse_orderbook(line)
            if parsed:
                bid, bid_size, ask, ask_size = parsed
                result.update({
                    "bid": bid,
                    "bid_size": bid_size,
                    "ask": ask,
                    "ask_size": ask_size
                })

        elif t == "last":
            result["last"] = parse_last(line)
            result["volume"] = parse_volume(line)

    return result

# ---------------------------------------------------------
# ⑤ API：画面コピー → JSON
# ---------------------------------------------------------
@app.post("/api/parse_market_text")
def parse_market_text(payload: dict):
    raw = payload.get("text", "")
    lines = preprocess(raw)
    parsed = parse_lines(lines)
    return parsed

# ---------------------------------------------------------
# ⑥ UI（画面コピー貼り付け → JSON化）
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>市場データ JSON化ツール</title>
<style>
body { font-family: sans-serif; padding: 20px; font-size: 20px; }
textarea { width: 100%; height: 300px; font-size: 18px; }
button { padding: 12px; font-size: 22px; width: 100%; margin-top: 10px; }
pre { background: #f0f0f0; padding: 15px; border-radius: 10px; }
</style>
</head>
<body>

<h2>📌 市場データ（画面コピー → JSON化）</h2>

<textarea id="rawText" placeholder="ここに証券会社の画面コピーを貼り付け"></textarea>

<button onclick="convert()">JSON化する</button>

<h3>📘 JSON 出力</h3>
<pre id="jsonOutput"></pre>

<script>
async function convert(){
    const raw = document.getElementById("rawText").value;

    const res = await fetch("/api/parse_market_text", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text: raw})
    });

    const data = await res.json();
    document.getElementById("jsonOutput").textContent =
        JSON.stringify(data, null, 2);
}
</script>

</body>
</html>
"""

