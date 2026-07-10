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

# 売気配／買気配（価格＋数量）
def parse_price_size(s):
    m = re.search(r"([\d\.]+)\s*\((\d+)\)", s)
    if m:
        return to_float(m.group(1)), to_int(m.group(2))
    try:
        return to_float(s), None
    except:
        return None, None

# ---------------------------------------------------------
# コール17行セットをパース
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
        "bid": parse_price_size(block[10])[0],
        "bid_size": parse_price_size(block[10])[1],
        "ask": parse_price_size(block[11])[0],
        "ask_size": parse_price_size(block[11])[1],
        "last": to_float(block[12]),
        "change": to_float(block[13]),
        "volume": to_int(block[14])
    }

# ---------------------------------------------------------
# プット17行セットをパース
# ---------------------------------------------------------
def parse_put(block):
    return {
        "last": to_float(block[0]),
        "change": to_float(block[1]),
        "volume": to_int(block[2]),
        "bid": parse_price_size(block[4])[0],
        "bid_size": parse_price_size(block[4])[1],
        "ask": parse_price_size(block[3])[0],
        "ask_size": parse_price_size(block[3])[1],
        "open": to_float(block[5]),
        "high": to_float(block[6]),
        "low": to_float(block[7]),
        "theoretical": to_float(block[8]),
        "delta": to_float(block[9]),
        "gamma": to_float(block[10]),
        "iv": to_float(block[11]),
        "theta": to_float(block[12]),
        "vega": to_float(block[13])
    }

# ---------------------------------------------------------
# 全行から 34行セット（コール17＋プット17）を抽出
# ---------------------------------------------------------
def parse_lines(lines):
    results = []
    i = 0
    n = len(lines)

    while i < n:
        # コール側の開始は「新規」
        if lines[i].startswith("新規"):
            # 34行揃っているか確認
            if i + 33 < n:
                call_block = lines[i:i+17]
                put_block  = lines[i+17:i+34]

                call_json = parse_call(call_block)
                put_json  = parse_put(put_block)

                # ストライクはコール側の17行目
                strike = to_int(call_block[16])

                results.append({
                    "strike": strike,
                    "call": call_json,
                    "put": put_json
                })

                i += 34
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
