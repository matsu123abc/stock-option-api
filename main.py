from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import jaconv

app = FastAPI()

# ---------------------------------------------------------
# ① 前処理：全角→半角・行分割
# ---------------------------------------------------------
def preprocess(raw: str):
    # 数字・ASCII を半角に統一（全角混在対策）
    raw = jaconv.z2h(raw, digit=True, ascii=True)
    lines = raw.split("\n")
    clean = [l.strip() for l in lines if l.strip() != ""]
    return clean

# ---------------------------------------------------------
# ② 1ストライク分（17行セット）をパース
#    行順は章さんのフォーマットに合わせて固定
# ---------------------------------------------------------
def parse_block(block):
    """
    block: 長さ17のリスト
    0: '新規'（またはその他）
    1: IV
    2: セータ
    3: ベガ
    4: 理論価
    5: デルタ
    6: ガンマ
    7: 始値
    8: 高値
    9: 安値
    10: 売気配 (数量付き)
    11: 買気配 (数量付き)
    12: 現値
    13: 前日比
    14: 売買高
    15: （コール側の現値などの場合もあるがここでは無視）
    16: 行使価格（ストライク）
    """

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

    # 売気配・買気配は「価格 (数量)」形式
    def parse_price_size(s):
        # 例: "1335.0 (25)" → price=1335.0, size=25
        import re
        m = re.search(r"([\d\.]+)\s*\((\d+)\)", s)
        if m:
            return to_float(m.group(1)), to_int(m.group(2))
        # 価格だけの場合
        try:
            return to_float(s), None
        except:
            return None, None

    iv      = to_float(block[1])
    theta   = to_float(block[2])
    vega    = to_float(block[3])
    theo    = to_float(block[4])
    delta   = to_float(block[5])
    gamma   = to_float(block[6])
    open_   = to_float(block[7])
    high    = to_float(block[8])
    low     = to_float(block[9])

    bid, bid_size = parse_price_size(block[10])
    ask, ask_size = parse_price_size(block[11])

    last    = to_float(block[12])
    change  = to_float(block[13])
    volume  = to_int(block[14])

    strike  = to_int(block[16])

    return {
        "strike": strike,
        "iv": iv,
        "theta": theta,
        "vega": vega,
        "theoretical": theo,
        "delta": delta,
        "gamma": gamma,
        "open": open_,
        "high": high,
        "low": low,
        "bid": bid,
        "bid_size": bid_size,
        "ask": ask,
        "ask_size": ask_size,
        "last": last,
        "change": change,
        "volume": volume,
    }

# ---------------------------------------------------------
# ③ 全行から 17行セットを順に抜き出してパース
# ---------------------------------------------------------
def parse_lines(lines):
    results = []
    i = 0
    n = len(lines)

    while i < n:
        # 「新規」行を起点にする（多少ゆるくしてもOK）
        if lines[i].startswith("新規"):
            # 17行揃っているか確認
            if i + 16 < n:
                block = lines[i:i+17]
                parsed = parse_block(block)
                # strike が取れているものだけ採用
                if parsed["strike"] is not None:
                    results.append(parsed)
                i += 17
            else:
                # 足りない場合は終了
                break
        else:
            i += 1

    return results

# ---------------------------------------------------------
# ④ API：画面コピー → JSON（複数ストライク）
# ---------------------------------------------------------
@app.post("/api/parse_market_text")
def parse_market_text(payload: dict):
    raw = payload.get("text", "")
    lines = preprocess(raw)
    parsed = parse_lines(lines)
    return parsed

# ---------------------------------------------------------
# ⑤ UI：画面コピー貼り付け → JSON化ボタン
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>市場データ JSON化ツール（日本語フォーマット対応）</title>
<style>
body { font-family: sans-serif; padding: 20px; font-size: 18px; }
textarea { width: 100%; height: 320px; font-size: 16px; }
button { padding: 10px; font-size: 18px; width: 100%; margin-top: 10px; }
pre { background: #f0f0f0; padding: 15px; border-radius: 8px; white-space: pre-wrap; }
</style>
</head>
<body>

<h2>📌 日経225オプション市場データ（画面コピー → JSON化）</h2>

<p>証券会社の取引画面の「コール/プット一覧」をコピーして、下のテキストエリアに貼り付けてください。</p>

<textarea id="rawText" placeholder="ここに画面コピーを貼り付け"></textarea>

<button onclick="convert()">JSON化する</button>

<h3>📘 JSON 出力（ストライクごと）</h3>
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
