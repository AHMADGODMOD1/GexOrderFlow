import sys
import io
import time
import threading
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from biquote import Biquote

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

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
    "poc": None, "hvn": [], "lvn": [], "vah": None, "val": None,
    "profile": [],
    "candles": {},
    "clusters": [],
    "last_update": None,
    "tick_history": [],
    "order_book": {"bids": [], "asks": [], "current": None, "type": "APPROX",
                   "total_buy": 0, "total_sell": 0, "delta": 0},
    "ob_type": "APPROX",
    "delta_data": {},
    "absorptions": [],
    "footprint": {},
}

last_known_price = None
last_known_bid = None
last_known_ask = None
last_known_spread = None
last_known_change = None

# ═══════════════════════════════════════════════
# FETCH CANDLES
# ═══════════════════════════════════════════════
def fetch_candles(tf, limit=1000, retries=2):
    for attempt in range(retries):
        try:
            raw = bq.ohlc(SYMBOL, interval=tf, limit=limit)
            if not raw:
                time.sleep(0.2)
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
        except Exception:
            time.sleep(0.2)
    return []

# ═══════════════════════════════════════════════
# BUILD DELTA + CVD
# ═══════════════════════════════════════════════
def build_delta_cvd(candles):
    if not candles or len(candles) < 5:
        return None

    df = pd.DataFrame(candles)
    df = df[df['volume'] > 0].reset_index(drop=True)
    if len(df) < 5:
        return None

    df['range'] = (df['high'] - df['low']).clip(lower=0.0001)
    df['close_pos'] = ((df['close'] - df['low']) / df['range']).clip(0, 1)
    df['buy_vol'] = df['volume'] * df['close_pos']
    df['sell_vol'] = df['volume'] * (1 - df['close_pos'])
    df['delta'] = df['buy_vol'] - df['sell_vol']
    df['cvd'] = df['delta'].cumsum()

    data = []
    for _, row in df.iterrows():
        data.append({
            "time": row['time'],
            "delta": round(float(row['delta']), 2),
            "cvd": round(float(row['cvd']), 2),
            "buy": round(float(row['buy_vol']), 2),
            "sell": round(float(row['sell_vol']), 2),
            "volume": round(float(row['volume']), 2),
        })

    return data

# ═══════════════════════════════════════════════
# BUILD ABSORPTIONS
# ═══════════════════════════════════════════════
def build_absorptions(candles, lookback=50):
    if not candles or len(candles) < 10:
        return []

    df = pd.DataFrame(candles)
    df = df[df['volume'] > 0].reset_index(drop=True)
    if len(df) < 10:
        return []

    df['avg_vol'] = df['volume'].rolling(lookback, min_periods=5).mean()
    df['range'] = (df['high'] - df['low']).clip(lower=0.0001)
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['avg_range'] = df['range'].rolling(lookback, min_periods=5).mean()

    df['close_pos'] = ((df['close'] - df['low']) / df['range']).clip(0, 1)
    df['buy_vol'] = df['volume'] * df['close_pos']
    df['sell_vol'] = df['volume'] * (1 - df['close_pos'])
    df['delta'] = df['buy_vol'] - df['sell_vol']
    df['avg_delta'] = df['delta'].abs().rolling(lookback, min_periods=5).mean()

    absorptions = []

    for idx, row in df.iterrows():
        if pd.isna(row['avg_vol']) or row['avg_vol'] == 0:
            continue

        vol_mult = row['volume'] / row['avg_vol']
        move_mult = row['range'] / row['avg_range'] if row['avg_range'] > 0 else 1
        delta_mult = abs(row['delta']) / row['avg_delta'] if row['avg_delta'] > 0 else 0

        high_vol = vol_mult >= 1.2
        low_move = move_mult <= 1.2
        has_lower_wick = row['lower_wick'] > 0.3
        has_upper_wick = row['upper_wick'] > 0.3

        if high_vol and low_move and has_lower_wick:
            vol_score = min(vol_mult / 3.0, 1.0)
            delta_score = min(delta_mult / 2.0, 1.0)
            move_score = 1.0 - min(move_mult / 1.2, 1.0)
            wick_score = min(row['lower_wick'] / row['range'] * 2.0, 1.0)

            score = (vol_score * 0.3 + delta_score * 0.25 + move_score * 0.25 + wick_score * 0.2) * 10.0
            score = max(0, min(10, score))

            if score >= 1.5:
                absorptions.append({
                    "time": row['time'],
                    "price": float(row['low']),
                    "side": "buy",
                    "score": round(score, 1),
                    "volume": round(float(row['volume']), 0),
                    "vol_mult": round(vol_mult, 2),
                })

        elif high_vol and low_move and has_upper_wick:
            vol_score = min(vol_mult / 3.0, 1.0)
            delta_score = min(delta_mult / 2.0, 1.0)
            move_score = 1.0 - min(move_mult / 1.2, 1.0)
            wick_score = min(row['upper_wick'] / row['range'] * 2.0, 1.0)

            score = (vol_score * 0.3 + delta_score * 0.25 + move_score * 0.25 + wick_score * 0.2) * 10.0
            score = max(0, min(10, score))

            if score >= 1.5:
                absorptions.append({
                    "time": row['time'],
                    "price": float(row['high']),
                    "side": "sell",
                    "score": round(score, 1),
                    "volume": round(float(row['volume']), 0),
                    "vol_mult": round(vol_mult, 2),
                })

    return absorptions[-50:]

# ═══════════════════════════════════════════════
# BUILD FOOTPRINT (for last candle)
# ═══════════════════════════════════════════════
def build_footprint(candles, levels=10):
    """
    Build footprint for each candle:
    - Divide candle range into N levels
    - Distribute volume across levels
    - Estimate buy/sell split per level
    """
    if not candles or len(candles) < 2:
        return {}

    df = pd.DataFrame(candles)
    df = df[df['volume'] > 0].reset_index(drop=True)
    if len(df) < 2:
        return {}

    footprints = {}

    for idx, row in df.iterrows():
        rng = max(row['high'] - row['low'], 0.0001)
        body = abs(row['close'] - row['open'])
        close_pos = (row['close'] - row['low']) / rng
        close_pos = max(0.0, min(1.0, close_pos))

        # Buy/Sell split for whole candle
        total_buy = row['volume'] * close_pos
        total_sell = row['volume'] * (1 - close_pos)

        # Divide candle into levels
        level_size = rng / levels
        candle_levels = []

        for lvl in range(levels):
            level_low = row['low'] + lvl * level_size
            level_high = level_low + level_size
            level_mid = (level_low + level_high) / 2

            # How much of candle body passes through this level
            body_low = min(row['open'], row['close'])
            body_high = max(row['open'], row['close'])
            overlap = max(0, min(level_high, body_high) - max(level_low, body_low))
            body_overlap_ratio = overlap / max(body, 0.0001)

            # Volume distribution (heavier near body)
            level_vol_ratio = 0.3 + 0.7 * body_overlap_ratio
            level_vol = (row['volume'] / levels) * level_vol_ratio

            # Buy/sell split (proportional to close position)
            level_buy = level_vol * close_pos
            level_sell = level_vol * (1 - close_pos)

            candle_levels.append({
                "price": round(level_mid, 2),
                "buy": round(level_buy, 0),
                "sell": round(level_sell, 0),
                "total": round(level_vol, 0),
                "delta": round(level_buy - level_sell, 0),
            })

        # Sort descending (high price first)
        candle_levels = sorted(candle_levels, key=lambda x: -x['price'])

        footprints[row['time']] = {
            "time": row['time'],
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume'],
            "total_buy": round(total_buy, 0),
            "total_sell": round(total_sell, 0),
            "delta": round(total_buy - total_sell, 0),
            "levels": candle_levels,
        }

    return footprints

# ═══════════════════════════════════════════════
# REAL ORDER BOOK
# ═══════════════════════════════════════════════
def fetch_real_order_book():
    try:
        for method in ['depth', 'orderbook', 'book', 'get_depth']:
            if hasattr(bq, method):
                fn = getattr(bq, method)
                ob = fn(SYMBOL)
                if ob and isinstance(ob, dict) and (ob.get('bids') or ob.get('asks')):
                    return ob
    except Exception:
        pass
    return None

def build_order_book(ticks, num_levels=20):
    if not ticks or len(ticks) < 10:
        return {"bids": [], "asks": [], "current": None, "type": "APPROX",
                "total_buy": 0, "total_sell": 0, "delta": 0}

    current_price = ticks[-1]["price"]
    prices = [t["price"] for t in ticks]
    price_range = max(prices) - min(prices)
    if price_range <= 0:
        return {"bids": [], "asks": [], "current": current_price, "type": "APPROX",
                "total_buy": 0, "total_sell": 0, "delta": 0}

    bucket_size = price_range / num_levels
    if bucket_size <= 0:
        bucket_size = 0.10

    buckets = {}
    for i in range(1, len(ticks)):
        prev_price = ticks[i-1]["price"]
        curr_price = ticks[i]["price"]
        delta = curr_price - prev_price

        bucket_idx = int((curr_price - min(prices)) / bucket_size)
        bucket_price = min(prices) + bucket_idx * bucket_size

        if bucket_price not in buckets:
            buckets[bucket_price] = {"buy": 0.0, "sell": 0.0, "count": 0}

        if delta > 0:
            buckets[bucket_price]["buy"] += abs(delta) * 1000
        elif delta < 0:
            buckets[bucket_price]["sell"] += abs(delta) * 1000
        buckets[bucket_price]["count"] += 1

    bids = []
    asks = []
    total_buy = 0.0
    total_sell = 0.0

    for price, data in sorted(buckets.items()):
        level = {
            "price": round(price, 2),
            "buy": round(data["buy"], 2),
            "sell": round(data["sell"], 2),
            "total": round(data["buy"] + data["sell"], 2),
            "count": data["count"],
        }
        total_buy += data["buy"]
        total_sell += data["sell"]
        if price < current_price:
            bids.append(level)
        else:
            asks.append(level)

    bids = sorted(bids, key=lambda x: -x["price"])[:num_levels]
    asks = sorted(asks, key=lambda x: x["price"])[:num_levels]

    return {
        "bids": bids, "asks": asks,
        "current": round(current_price, 2),
        "type": "APPROX",
        "total_buy": round(total_buy, 2),
        "total_sell": round(total_sell, 2),
        "delta": round(total_buy - total_sell, 2),
    }

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

    total_vol = vols.sum()
    va_target = total_vol * 0.70
    va_vol = vols[poc_idx]
    va_lo = poc_idx
    va_hi = poc_idx
    while va_vol < va_target and (va_lo > 0 or va_hi < bins - 1):
        up_vol = vols[va_hi + 1] if va_hi < bins - 1 else 0
        dn_vol = vols[va_lo - 1] if va_lo > 0 else 0
        if up_vol >= dn_vol and va_hi < bins - 1:
            va_hi += 1
            va_vol += up_vol
        elif va_lo > 0:
            va_lo -= 1
            va_vol += dn_vol
        else:
            break

    vah = float(centers[va_hi])
    val = float(centers[va_lo])

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

        in_va = va_lo <= i <= va_hi

        profile_data.append({
            "price": p, "volume": v, "type": t,
            "buy": bv, "sell": sv,
            "in_va": in_va,
        })

    return {
        "poc": poc, "hvn": hvn, "lvn": lvn,
        "vah": vah, "val": val,
        "profile": profile_data,
    }

# ═══════════════════════════════════════════════
# TICKER
# ═══════════════════════════════════════════════
def ticker():
    global last_known_price, last_known_bid, last_known_ask, last_known_spread, last_known_change
    while True:
        try:
            t = bq.tick(SYMBOL)
            if t and t.get("mid"):
                mid = t.get("mid")
                cache["price"] = mid
                cache["bid"] = t.get("bid")
                cache["ask"] = t.get("ask")
                cache["change"] = t.get("dayDiffPercent")
                cache["spread"] = t.get("spread")
                cache["last_update"] = time.strftime('%H:%M:%S')

                last_known_price = mid
                last_known_bid = t.get("bid")
                last_known_ask = t.get("ask")
                last_known_spread = t.get("spread")
                last_known_change = t.get("dayDiffPercent")

                cache["tick_history"].append({"time": time.time(), "price": mid})
                if len(cache["tick_history"]) > 5000:
                    cache["tick_history"] = cache["tick_history"][-5000:]
            else:
                if last_known_price:
                    cache["price"] = last_known_price
                    cache["bid"] = last_known_bid
                    cache["ask"] = last_known_ask
                    cache["spread"] = last_known_spread
                    cache["change"] = last_known_change
        except Exception:
            if last_known_price:
                cache["price"] = last_known_price
        time.sleep(0.2)

# ═══════════════════════════════════════════════
# REFRESH
# ═══════════════════════════════════════════════
def refresh_data():
    cycle = 0
    while True:
        try:
            cycle += 1

            candles_1m = fetch_candles(TF_MAP["1m"], limit=1000)
            if candles_1m:
                cache["candles"]["1m"] = candles_1m
                if not cache["price"]:
                    cache["price"] = candles_1m[-1]["close"]
                cache["clusters"] = build_clusters(candles_1m[-200:], num_clusters=15)

                delta_cvd = build_delta_cvd(candles_1m)
                if delta_cvd:
                    cache["delta_data"]["1m"] = delta_cvd

                cache["absorptions"] = build_absorptions(candles_1m, lookback=50)
                cache["footprint"]["1m"] = build_footprint(candles_1m, levels=10)

            real_ob = fetch_real_order_book()
            if real_ob:
                cache["order_book"] = real_ob
                cache["ob_type"] = "REAL"
            else:
                if len(cache["tick_history"]) > 10:
                    cache["order_book"] = build_order_book(cache["tick_history"], num_levels=20)
                    cache["ob_type"] = "APPROX"

            if cycle % 5 == 0:
                for tf in ["5m", "15m", "30m"]:
                    candles = fetch_candles(TF_MAP[tf], limit=1000)
                    if candles:
                        cache["candles"][tf] = candles
                        delta_cvd = build_delta_cvd(candles)
                        if delta_cvd:
                            cache["delta_data"][tf] = delta_cvd
                        cache["footprint"][tf] = build_footprint(candles, levels=10)

            if cycle % 30 == 0:
                for tf in ["1h", "4h", "1d"]:
                    candles = fetch_candles(TF_MAP[tf], limit=1000)
                    if candles:
                        cache["candles"][tf] = candles
                        delta_cvd = build_delta_cvd(candles)
                        if delta_cvd:
                            cache["delta_data"][tf] = delta_cvd
                        cache["footprint"][tf] = build_footprint(candles, levels=10)

                h1 = cache["candles"].get("1h", [])
                if h1:
                    result = build_profile_from_candles(h1)
                    if result:
                        cache["poc"] = result["poc"]
                        cache["hvn"] = result["hvn"]
                        cache["lvn"] = result["lvn"]
                        cache["vah"] = result["vah"]
                        cache["val"] = result["val"]
                        cache["profile"] = result["profile"]
        except Exception:
            pass
        time.sleep(1)

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
        "vah": cache["vah"], "val": cache["val"],
        "profile": cache["profile"], "last_update": cache["last_update"],
        "order_book": cache["order_book"],
        "ob_type": cache.get("ob_type", "APPROX"),
        "absorptions": cache.get("absorptions", []),
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

@app.route("/api/orderbook")
def api_orderbook():
    return jsonify(cache["order_book"])

@app.route("/api/delta")
def api_delta():
    tf = request.args.get("tf", "1h")
    if tf not in TF_MAP:
        return jsonify({"error": "Invalid timeframe"}), 400
    return jsonify({"tf": tf, "delta_data": cache["delta_data"].get(tf, [])})

@app.route("/api/footprint")
def api_footprint():
    tf = request.args.get("tf", "1h")
    fp = cache["footprint"].get(tf, {})
    return jsonify({"tf": tf, "footprint": fp})

@app.route("/api/ticks")
def api_ticks():
    return jsonify({"ticks": cache["tick_history"][-500:]})

# ═══════════════════════════════════════════════
# INITIAL LOAD
# ═══════════════════════════════════════════════
print("[START] GEX Order Flow initial load...")

for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
    candles = fetch_candles(TF_MAP[tf], limit=1000)
    if candles:
        cache["candles"][tf] = candles
        delta_cvd = build_delta_cvd(candles)
        if delta_cvd:
            cache["delta_data"][tf] = delta_cvd
        cache["footprint"][tf] = build_footprint(candles, levels=10)
        print(f"[START] {tf}: {len(candles)} candles")

c1m = cache["candles"].get("1m", [])
if c1m:
    cache["clusters"] = build_clusters(c1m[-200:], num_clusters=15)
    cache["price"] = c1m[-1]["close"]
    cache["absorptions"] = build_absorptions(c1m, lookback=50)
    print(f"[START] Absorptions: {len(cache['absorptions'])}")

h1 = cache["candles"].get("1h", [])
if h1:
    result = build_profile_from_candles(h1)
    if result:
        cache["poc"] = result["poc"]
        cache["hvn"] = result["hvn"]
        cache["lvn"] = result["lvn"]
        cache["vah"] = result["vah"]
        cache["val"] = result["val"]
        cache["profile"] = result["profile"]

threading.Thread(target=refresh_data, daemon=True).start()
threading.Thread(target=ticker, daemon=True).start()
print("[START] Background threads started")

if __name__ == "__main__":
    print("\n[LOCAL] http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
