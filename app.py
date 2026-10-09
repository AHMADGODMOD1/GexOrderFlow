import sys
import io
import time
import threading
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from biquote import Biquote

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)

bq = Biquote()
SYMBOL = "XAUUSD"

TF_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d",
}

cache = {
    "price": None, "change": None, "spread": None, "bid": None, "ask": None,
    "poc": None, "hvn": [], "lvn": [], "profile": [],
    "candles": {},
    "clusters": [],
    "last_update": None,
}

# ═══════════════════════════════════════════════
# FETCH CANDLES
# ═══════════════════════════════════════════════
def fetch_candles(tf, limit=200, retries=3):
    for attempt in range(retries):
        try:
            raw = bq.ohlc(SYMBOL, interval=tf, limit=limit)
            if not raw:
                print(f"[CANDLE EMPTY {tf}] attempt {attempt+1}")
                time.sleep(2)
                continue
            candles = []
            for row in raw:
                candles.append({
                    "time": row['openTime'],
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row.get('tickVolume', 0) or 0),
                })
            return candles
        except Exception as e:
            print(f"[CANDLE ERROR {tf} attempt {attempt+1}] {e}")
            time.sleep(2)
    return []

# ═══════════════════════════════════════════════
# BUILD CLUSTERS
# ═══════════════════════════════════════════════
def build_clusters(candles_1m, num_clusters=15):
    if not candles_1m or len(candles_1m) < 10:
        return []
    df = pd.DataFrame(candles_1m)
    df = df[df['volume'] > 0].reset_index(drop=True)
    if len(df) < 10:
        return []

    price_min = df['low'].min()
    price_max = df['high'].max()
    rng = price_max - price_min
    if rng <= 0:
        return []

    bin_size = rng / num_clusters
    clusters = []
    for i in range(num_clusters):
        clusters.append({
            "price_low": price_min + i * bin_size,
            "price_high": price_min + (i + 1) * bin_size,
            "price_mid": price_min + (i + 0.5) * bin_size,
            "buy_vol": 0.0,
            "sell_vol": 0.0,
            "total": 0.0,
        })

    for _, row in df.iterrows():
        rng_bar = max(row['high'] - row['low'], 0.0001)
        close_pos = (row['close'] - row['low']) / rng_bar
        close_pos = max(0.0, min(1.0, close_pos))
        buy_vol = row['volume'] * close_pos
        sell_vol = row['volume'] * (1 - close_pos)

        lo_idx = max(0, min(num_clusters - 1, int((row['low'] - price_min) / bin_size)))
        hi_idx = max(0, min(num_clusters - 1, int((row['high'] - price_min) / bin_size)))

        if hi_idx > lo_idx:
            n_bins = hi_idx - lo_idx + 1
            for b in range(lo_idx, hi_idx + 1):
                clusters[b]["buy_vol"] += buy_vol / n_bins
                clusters[b]["sell_vol"] += sell_vol / n_bins
                clusters[b]["total"] += row['volume'] / n_bins
        else:
            clusters[lo_idx]["buy_vol"] += buy_vol
            clusters[lo_idx]["sell_vol"] += sell_vol
            clusters[lo_idx]["total"] += row['volume']

    max_total = max(c["total"] for c in clusters) if clusters else 1.0
    max_side = max(max(c["buy_vol"], c["sell_vol"]) for c in clusters) if clusters else 1.0

    for c in clusters:
        c["buy_pct"] = c["buy_vol"] / max_side * 100 if max_side > 0 else 0
        c["sell_pct"] = c["sell_vol"] / max_side * 100 if max_side > 0 else 0
        c["total_pct"] = c["total"] / max_total * 100 if max_total > 0 else 0

        if c["buy_vol"] > c["sell_vol"] * 1.5:
            c["dominance"] = "buy"
        elif c["sell_vol"] > c["buy_vol"] * 1.5:
            c["dominance"] = "sell"
        else:
            c["dominance"] = "balanced"

        total = c["total"]
        if total >= max_total * 0.7:
            c["size"] = "big"
        elif total >= max_total * 0.35:
            c["size"] = "medium"
        else:
            c["size"] = "small"

        c["delta"] = c["buy_vol"] - c["sell_vol"]

    return clusters

# ═══════════════════════════════════════════════
# BUILD VOLUME PROFILE
# ═══════════════════════════════════════════════
def build_profile_from_candles(candles, bins=50):
    if not candles or len(candles) < 5:
        return None
    df = pd.DataFrame(candles)
    df = df[df['volume'] > 0].reset_index(drop=True)
    if len(df) < 5:
        return None

    price_min = df['low'].min()
    price_max = df['high'].max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vols = np.zeros(bins)
    buy_vols = np.zeros(bins)
    sell_vols = np.zeros(bins)

    for _, row in df.iterrows():
        rng = max(row['high'] - row['low'], 0.0001)
        cp = (row['close'] - row['low']) / rng
        cp = max(0.0, min(1.0, cp))
        bv = row['volume'] * cp
        sv = row['volume'] * (1 - cp)

        li = max(0, np.searchsorted(bin_edges, row['low']) - 1)
        hi = min(bins - 1, np.searchsorted(bin_edges, row['high']) - 1)
        if hi > li:
            n = hi - li + 1
            for i in range(li, hi + 1):
                vols[i] += row['volume'] / n
                buy_vols[i] += bv / n
                sell_vols[i] += sv / n
        else:
            vols[li] += row['volume']
            buy_vols[li] += bv
            sell_vols[li] += sv

    max_vol = vols.max()
    if max_vol == 0:
        return None

    poc_idx = int(np.argmax(vols))
    poc = float(centers[poc_idx])
    hvn = [float(c) for c in centers[vols >= max_vol * 0.7]]
    lvn = [float(c) for c in centers[vols <= max_vol * 0.2]]

    profile_data = []
    for i in range(len(centers)):
        p = float(centers[i])
        v = float(vols[i])
        bv = float(buy_vols[i])
        sv = float(sell_vols[i])
        if abs(p - poc) < 0.5:
            t = "poc"
        elif p in hvn:
            t = "hvn"
        elif p in lvn:
            t = "lvn"
        else:
            t = "normal"
        profile_data.append({
            "price": p, "volume": v, "type": t,
            "buy": bv, "sell": sv,
        })

    return {"poc": poc, "hvn": hvn, "lvn": lvn, "profile": profile_data}

# ═══════════════════════════════════════════════
# BACKGROUND THREADS
# ═══════════════════════════════════════════════
def refresh_data():
    while True:
        try:
            print(f"[REFRESH] Starting cycle...")
            for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                candles = fetch_candles(TF_MAP[tf], limit=200)
                if candles:
                    cache["candles"][tf] = candles
                    print(f"[REFRESH] {tf}: {len(candles)} candles")

            c1m = cache["candles"].get("1m", [])
            if c1m:
                cache["clusters"] = build_clusters(c1m, num_clusters=15)
                print(f"[REFRESH] Clusters: {len(cache['clusters'])}")

            h1 = cache["candles"].get("1h", [])
            if h1:
                result = build_profile_from_candles(h1)
                if result:
                    cache["poc"] = result["poc"]
                    cache["hvn"] = result["hvn"]
                    cache["lvn"] = result["lvn"]
                    cache["profile"] = result["profile"]
                    print(f"[REFRESH] POC={result['poc']:.2f}")
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")
        time.sleep(60)

def ticker():
    while True:
        try:
            t = bq.tick(SYMBOL)
            if t and t.get("mid"):
                cache["price"] = t.get("mid")
                cache["bid"] = t.get("bid")
                cache["ask"] = t.get("ask")
                cache["change"] = t.get("dayDiffPercent")
                cache["spread"] = t.get("spread")
                cache["last_update"] = time.strftime('%H:%M:%S')
                print(f"[TICK] {cache['price']}")
            else:
                # ─── FALLBACK: Use last 1m candle close ───
                c1m = cache["candles"].get("1m", [])
                if c1m:
                    last = c1m[-1]
                    cache["price"] = last["close"]
                    cache["bid"] = last["close"]
                    cache["ask"] = last["close"]
                    cache["spread"] = 0
                    cache["change"] = 0
                    cache["last_update"] = time.strftime('%H:%M:%S')
                    print(f"[TICK-FALLBACK] {cache['price']}")
        except Exception as e:
            print(f"[TICK ERROR] {e}")
            # ─── Fallback on error too ───
            c1m = cache["candles"].get("1m", [])
            if c1m:
                last = c1m[-1]
                cache["price"] = last["close"]
                cache["last_update"] = time.strftime('%H:%M:%S')
        time.sleep(10)

# ═══════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    return jsonify({
        "price": cache["price"], "change": cache["change"],
        "spread": cache["spread"], "bid": cache["bid"], "ask": cache["ask"],
        "poc": cache["poc"], "hvn": cache["hvn"], "lvn": cache["lvn"],
        "profile": cache["profile"], "last_update": cache["last_update"],
    })

@app.route("/api/candles")
def api_candles():
    tf = request.args.get("tf", "1h")
    if tf not in TF_MAP:
        return jsonify({"error": "Invalid timeframe"}), 400
    return jsonify({"tf": tf, "candles": cache["candles"].get(tf, [])})

@app.route("/api/clusters")
def api_clusters():
    return jsonify({"clusters": cache["clusters"]})

# ═══════════════════════════════════════════════
# INITIAL LOAD + BACKGROUND THREADS (MODULE LEVEL)
# ═══════════════════════════════════════════════
print("[START] GEX Order Flow initial load...")

for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
    candles = fetch_candles(TF_MAP[tf], limit=200)
    if candles:
        cache["candles"][tf] = candles
        print(f"[START] {tf}: {len(candles)} candles")

c1m = cache["candles"].get("1m", [])
if c1m:
    cache["clusters"] = build_clusters(c1m, num_clusters=15)
    print(f"[START] Clusters: {len(cache['clusters'])}")

h1 = cache["candles"].get("1h", [])
if h1:
    result = build_profile_from_candles(h1)
    if result:
        cache["poc"] = result["poc"]
        cache["hvn"] = result["hvn"]
        cache["lvn"] = result["lvn"]
        cache["profile"] = result["profile"]
        print(f"[START] POC={result['poc']:.2f}")

# ═══ BACKGROUND THREADS START ═══
threading.Thread(target=refresh_data, daemon=True).start()
threading.Thread(target=ticker, daemon=True).start()
print("[START] Background threads started")

if __name__ == "__main__":
    print("\n[LOCAL] http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
