from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import jaconv
import re

app = FastAPI()

def preprocess(raw: str):
    raw = jaconv.z2h(raw, digit=True, ascii=True)
    lines = raw.split("\n")
    clean = [l.strip() for l in lines if l.strip() != ""]
    return clean

def to_float(s):
    try:
        return float(s.replace(",", ""))
    except:
        return None

def to_int(s):
    try:
        return int(s.replace(",", ""))
    except:
        return None

def qty(s):
    m = re.search(r"\((\d+)\)", s)
    return int(m.group(1)) if m else None

# -------------------------
# コール18行
# -------------------------
def parse_call(block):
    return {
        "iv": to_float(block[1]),
        "theta": to_float(block[2]),
        "vega": to_float(block[3]),
        "theoretical": to_float(block[4]),
        "delta": to_float(block[5]),
        "gamma": to_float(block[6]),
        "open": to_float(block[7]),
        "high": to_float(block[8]),
        "low": to_float(block[9]),
        "bid": to_float(block[10]),
        "bid_size": qty(block[11]),
        "ask": to_float(block[12]),
        "ask_size": qty(block[13]),
        "last": to_float(block[14]),
        "change": to_float(block[15]),
        "volume": to_int(block[16]),
        "strike": to_int(block[17])
    }

# -------------------------
# プット17行
# -------------------------
def parse_put(block):
    return {
        "last": to_float(block[0]),
        "change": to_float(block[1]),
        "volume": to_int(block[2]),
        "ask": to_float(block[3]),
        "ask_size": qty(block[4]),
        "bid": to_float(block[5]),
        "bid_size": qty(block[6]),
        "open": to_float(block[7]),
        "high": to_float(block[8]),
        "low": to_float(block[9]),
        "theoretical": to_float(block[10]),
        "delta": to_float(block[11]),
        "gamma": to_float(block[12]),
        "iv": to_float(block[13]),
        "theta": to_float(block[14]),
        "vega": to_float(block[15])
    }

# -------------------------
# ヘッダー除去 → 35行セット抽出
# -------------------------
def parse_lines(lines):
    results = []

    # ① 「新規」行の位置を探す
    start_indices = [i for i, line in enumerate(lines) if line.startswith("新規")]

    for start in start_indices:
        # ② コール18行＋プット17行が揃っているか
        if start + 34 < len(lines):
            call_block = lines[start:start+18]
            put_block  = lines[start+18:start+35]

            call_json = parse_call(call_block)
            put_json  = parse_put(put_block)

            results.append({
                "strike": call_json["strike"],
                "call": call_json,
                "put": put_json
            })

    return results

@app.post("/api/parse_market_text")
def parse_market_text(payload: dict):
    raw = payload.get("text", "")
    lines = preprocess(raw)
    parsed = parse_lines(lines)
    return parsed

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>市場データ JSON化ツール（コール＋プット対応）</title>
<style>
body { font-family: sans-serif; padding: 20px; font-size: 18px; }
textarea { width: 100%; height: 320px; font-size: 16px; }
button { padding: 10px; font-size: 18px; width: 100%; margin-top: 10px; }
pre { background: #f0f0f0; padding: 15px; border-radius: 8px; white-space: pre-wrap; }
</style>
</head>
<body>

<h2>📌 日経225オプション市場データ（コール＋プット）JSON化</h2>

<textarea id="rawText" placeholder="ここに画面コピーを貼り付け"></textarea>

<button onclick="convert()">JSON化する</button>

<h3>📘 JSON 出力</h3>
<pre id="jsonOutput"></pre>

<script>
async function convert(){
    const raw = document.getElementById("rawText").value;
    const url = window.location.origin + "/api/parse_market_text";

    const res = await fetch(url, {
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
