#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MotoMarket — Flask backend
Linear Regression (Python) + giao diện gốc từ index.html
"""

import re, json, math, os
from flask import Flask, request, jsonify, send_file

# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────
COND_MULT = {"zin": 1.0, "worn": 0.87, "engine": 0.78, "accident": 0.62}
REG_MULT  = {"hcm": 1.05, "hn": 1.0,  "other":  0.93}
AGE_MAX, KM_MAX, SCORE_MAX = 16, 200_000, 4
DEP_RATES = {1: 0.12, 2: 0.10, 3: 0.09, 4: 0.08, 5: 0.07}

# ─────────────────────────────────────────
#  LOAD DATA & TRAIN LR
# ─────────────────────────────────────────
HTML_PATH      = os.path.join(os.path.dirname(__file__), "index.html")       # served to browser
DATA_HTML_PATH = os.path.join(os.path.dirname(__file__), "data_source.html")  # original with DATA

def load_data():
    with open(DATA_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    m = re.search(r"const DATA = (\[.*?\]);", html, re.DOTALL)
    if not m:
        raise ValueError("Không tìm thấy const DATA trong data_source.html")
    return json.loads(m.group(1))

def clip(v, lo, hi): return max(lo, min(hi, v))

def train_lr(data):
    models    = sorted(set(d["model"] for d in data))
    model_idx = {m: i for i, m in enumerate(models)}
    p = 6 + len(models)
    n = len(data)

    def featurize(d):
        row    = [0.0] * p
        row[0] = 1.0
        row[1] = clip(2026 - d["year"], 0, AGE_MAX) / AGE_MAX
        row[2] = clip(d["km"],          0, KM_MAX)   / KM_MAX
        row[3] = COND_MULT.get(d["condition"], 0.87)
        row[4] = REG_MULT.get(d["region"],     1.0)
        row[5] = (clip(d["score"], 1, 5) - 1)        / SCORE_MAX
        idx = model_idx.get(d["model"])
        if idx is not None: row[6 + idx] = 1.0
        return row

    X = [featurize(d) for d in data]
    y = [float(d["price"]) for d in data]

    # XᵀX + ridge
    XtX = [[sum(X[k][i]*X[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    for i in range(1, p): XtX[i][i] += 1e-4

    Xty = [sum(X[k][i]*y[k] for k in range(n)) for i in range(p)]

    # Gaussian elimination (augmented)
    A = [XtX[i][:] + [Xty[i]] for i in range(p)]
    for col in range(p):
        mr = max(range(col, p), key=lambda r: abs(A[r][col]))
        A[col], A[mr] = A[mr], A[col]
        piv = A[col][col]
        if abs(piv) < 1e-12: continue
        for j in range(col, p+1): A[col][j] /= piv
        for r in range(p):
            if r == col: continue
            f = A[r][col]
            for j in range(col, p+1): A[r][j] -= f * A[col][j]

    beta = [A[i][p] for i in range(p)]

    # R²
    yh = [sum(beta[j]*featurize(d)[j] for j in range(p)) for d in data]
    ym = sum(y)/n
    r2 = 1 - sum((y[k]-yh[k])**2 for k in range(n)) / sum((yi-ym)**2 for yi in y)

    def predict(inp):
        row = featurize(inp)
        return max(500_000, round(sum(beta[j]*row[j] for j in range(p))))

    return predict, r2, models, model_idx, featurize, beta, p

# ── boot ──
print("⏳ Loading data & training Linear Regression...")
DATA        = load_data()
predict_fn, R2, MODELS, MODEL_IDX, featurize, BETA, P = train_lr(DATA)
N = len(DATA)
print(f"✓ {N:,} samples · R² = {R2:.4f} · {P} features")

# ─────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return send_file(HTML_PATH)

@app.route("/predict", methods=["POST"])
def predict_route():
    inp = request.json

    # validate
    required = ["brand","model","year","km","condition","region","score"]
    for k in required:
        if k not in inp:
            return jsonify(error=f"Missing field: {k}"), 400

    inp["year"]  = int(inp["year"])
    inp["km"]    = int(inp["km"])
    inp["score"] = int(inp["score"])

    predicted = predict_fn(inp)

    # market prices
    market = sorted(
        d["price"] for d in DATA
        if d["model"] == inp["model"]
        or (d["brand"] == inp["brand"] and abs(d["year"] - inp["year"]) <= 3)
    )

    # depreciation
    def project(base, years):
        p = float(base)
        for i in range(1, years+1): p *= 1 - DEP_RATES.get(i, 0.07)
        return round(p)

    depreciation = [{"year": 2026+i, "price": project(predicted, i)} for i in range(6)]

    # model stats
    n_samples = sum(1 for d in DATA if d["brand"] == inp["brand"] and d["model"] == inp["model"])

    return jsonify(
        predicted    = predicted,
        marketPrices = market,
        depreciation = depreciation,
        nSamples     = n_samples,
        r2           = round(R2, 4),
        nTotal       = N,
    )

@app.route("/brands")
def brands_route():
    from collections import defaultdict
    bm = defaultdict(set)
    cnt = {}
    for d in DATA:
        bm[d["brand"]].add(d["model"])
        key = f"{d['brand']}||{d['model']}"
        cnt[key] = cnt.get(key, 0) + 1
    brand_models = {b: sorted(ms) for b, ms in sorted(bm.items())}
    return jsonify(brandModels=brand_models, modelCnt=cnt)

if __name__ == "__main__":
    print("🚀  http://localhost:5000")
    app.run(debug=False, port=5000)
