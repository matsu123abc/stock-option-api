from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import jaconv
import re

app = FastAPI()

# ---------------------------------------------------------
# 前処理：全角→半角・行分割
# ---------------------------------------------------------
def preprocess(raw: str):
    raw = jaconv.z2h(raw, digit=True, ascii=True)
    lines = raw.split("\n")
    clean = [l.strip() for l in lines if l.strip() != ""]
    return clean

# ---------------------------------------------------------
# 数値変換ヘルパー
# ---------------------------------------------------------
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
    # "(25)" → 25
    m = re.search(r"\((\d+)\)", s)
    return int(m.group(1)) if m else None

# ---------------------------------------------------------
# コール18行セットをパース
# ---------------------------------------------------------
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

        # 売気配（価格＋数量）
        "bid": to_float(block[10]),
        "bid_size": qty(block[11]),

        # 買気配（価格＋数量）
        "ask": to_float(block[12]),
        "ask_size": qty(block[13]),

        "last": to_float(block[14]),
        "change": to_float(block[15]),
        "volume": to_int(block[16]),

        # ★ strike 正しく取得
        "strike": to_int(block[17])
    }

# ---------------------------------------------------------
# プット17行セットをパース
# ---------------------------------------------------------
def parse_put(block):
    return {
        "last": to_float(block[0]),
        "change": to_float(block[1]),
        "volume": to_int(block[2]),

        # 買気配（価格＋数量）
        "ask": to_float(block[3]),
        "ask_size": qty(block[4]),

        # 売気配（価格＋数量）
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

# ---------------------------------------------------------
# 全行から 35行セット（コール18＋プット17）を抽出
# ---------------------------------------------------------
def parse_lines(lines):
    results = []
    i = 0
    n = len(lines)

    while i < n:
        if lines[i].startswith("新規"):
            # コール18行＋プット17行＝35行必要
            if i + 34 < n:
                call_block = lines[i:i+18]
                put_block  = lines[i+18:i+35]

                call_json = parse_call(call_block)
                put_json  = parse_put(put_block)

                results.append({
                    "strike": call_json["strike"],
                    "call": call_json,
                    "put": put_json
                })

                i += 35
            else:
                break
        else:
            i += 1

    return results

# ---------------------------------------------------------
# API：画面コピー → JSON
# ---------------------------------------------------------
@app.post("/api/parse_market_text")
def parse_market_text(payload: dict):
    raw = payload.get("text", "")
    lines = preprocess(raw)
    parsed = parse_lines(lines)
    return parsed

# ---------------------------------------------------------
# UI：画面コピー貼り付け → JSON化
# ---------------------------------------------------------
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
