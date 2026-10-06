"""StoreLab Sim: agent-based Monte-Carlo of shopper journeys under a candidate layout.

Gemini never touches this. The simulator takes a layout (baseline + validated changes) and
the *learned* Model, walks N statistical shoppers through the store, applies the learned
purchase model, then pushes buyers through a single-register queue to measure congestion.

Counterfactual layouts re-weight the learned baseline transition probabilities:
    P'(z | cur, cluster) ∝ P(z | cur, cluster) * w_candidate(cluster, z) / w_baseline(cluster, z)
    w(cluster, z) = 1 + PULL * Σ affinity(cluster, category) * display_strength(type)
PULL is a modelling assumption (shared with the demo world) — real pilots should recalibrate it.

Common random numbers: every shopper draws from fixed per-slot random streams, so baseline and
candidate runs are tightly paired and per-day deltas have small variance.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import store as S
from .history import HOUR_W, OPEN_HOUR
from .model import BOUNCE, K, Model

PULL = 0.8  # must match the demo world's pull; recalibrate from physical pilots
DAYS = 14
PER_DAY = 1400
MAXS = len(S.VISITABLE)
SERVICE_BASE, SERVICE_ITEM, SERVICE_REGISTER_IMPULSE = 20.0, 8.0, 12.0  # seconds
PEAK = (17 * 3600, 19 * 3600)
_T90 = {1: 6.314, 2: 2.92, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.86, 9: 1.833,
        10: 1.812, 11: 1.796, 12: 1.782, 14: 1.761, 19: 1.729, 29: 1.699}


def _tcrit(df: int) -> float:
    for k in sorted(_T90):
        if df <= k:
            return _T90[k]
    return 1.645


@lru_cache(maxsize=4)
def _draws(days: int, per_day: int, seed: int):
    rng = np.random.default_rng(seed)
    n = days * per_day
    return {
        "cl": rng.random(n).tolist(), "h": rng.random(n).tolist(), "frac": rng.random(n).tolist(),
        "end": rng.random((n, MAXS + 1)).tolist(), "nxt": rng.random((n, MAXS)).tolist(),
        "z": rng.standard_normal((n, MAXS)).tolist(), "buy": rng.random((n, len(S.CAT_LIST))).tolist(),
        "srv": rng.standard_normal(n).tolist(),
    }


def _weights(model: Model, zcats: dict) -> list[list[float]]:
    return [[1 + PULL * sum(model.aff[c][cat] * S.STRENGTH[t] for cat, t in zcats.get(z, []))
             for z in S.VISITABLE] for c in range(K + 1)]


def simulate(model: Model, layout: list[dict], days: int = DAYS, per_day: int = PER_DAY,
             seed: int = 11, n_paths: int = 0) -> dict:
    D = _draws(days, per_day, seed)
    zc = S.layout_index(layout)
    wB = _weights(model, S.layout_index(S.BASE_LAYOUT))
    wC = _weights(model, zc)
    Wc = {c: {cur: [model.T[c][cur][k] * wC[c][k] / wB[c][k] for k in range(MAXS)] for cur in model.T[c]}
          for c in range(K + 1)}
    mu = {c: [model.dwell_a + model.dwell_b * model.x_dwell(c, zc.get(z, [])) for z in S.VISITABLE]
          for c in range(K + 1)}
    sig = model.dwell_sigma
    edw = {c: [model.expected_dwell(c, zc.get(z, [])) for z in S.VISITABLE] for c in range(K + 1)}
    cum, acc = [], 0.0
    for s in model.cluster_share:
        acc += s
        cum.append(acc)
    hcum = np.cumsum(HOUR_W).tolist()
    # fast logit tables from the fitted pooled regression
    b, nb, nc = model.beta, model._nb, len(S.CAT_LIST)
    base = {cat: [float(b[i * (K + 1) + c]) for c in range(K + 1)] for i, cat in enumerate(S.CAT_LIST)}
    b_end, b_sig, b_reg, b_dw = float(b[nb]), float(b[nb + 1]), float(b[nb + 2]), float(b[nb + 3])
    c_before = {cat: float(b[nb + 4 + i]) for i, cat in enumerate(S.CAT_LIST)}
    c_adj = {cat: float(b[nb + 4 + nc + i]) for i, cat in enumerate(S.CAT_LIST)}

    out = {k: [] for k in ("visitors", "buyers", "revenue", "margin", "peak_wait", "mean_wait", "items", "peak_load")}
    cat_units = {c: [] for c in S.CAT_LIST}
    exposure = {c: [] for c in S.CAT_LIST}
    zone_visits = {z: 0 for z in S.VISITABLE}
    paths = []
    n_total = 0
    for day in range(days):
        units = {c: 0 for c in S.CAT_LIST}
        expo = {c: 0 for c in S.CAT_LIST}
        revenue = margin = items_total = buyers = 0
        queue = []
        for k in range(per_day):
            i = day * per_day + k
            n_total += 1
            cl = next((c for c, v in enumerate(cum) if D["cl"][i] <= v), BOUNCE)
            visited = [False] * MAXS
            stops, hat, cur, tsum = [], [], "entrance", 0.0
            for s in range(MAXS):
                if D["end"][i][s] < model.p_end[cl][s]:
                    break
                w = Wc[cl][cur]
                tot = sum(w[q] for q in range(MAXS) if not visited[q])
                if tot <= 0:
                    break
                r, q = D["nxt"][i][s] * tot, -1
                for q in range(MAXS):
                    if visited[q]:
                        continue
                    r -= w[q]
                    if r <= 0:
                        break
                visited[q] = True
                z = S.VISITABLE[q]
                dwell = min(max(math.exp(mu[cl][q] + sig * D["z"][i][s]), 2.0), 240.0)
                stops.append((z, dwell))
                hat.append((z, edw[cl][q]))
                zone_visits[z] += 1
                tsum += dwell
                cur = z
            full = hat + [("checkout", 4.0)]
            bought, reg_items = [], 0
            for ci, cat in enumerate(S.CAT_LIST):
                f = S.purchase_features(full, zc, cat)
                if not f:
                    continue
                expo[cat] += 1
                lg = (base[cat][cl] + b_dw * f["dwell10"] + b_end * f["endcap"] + b_sig * f["signage"] + b_reg * f["register"]
                      + c_before[cat] * f["comp_before"] + c_adj[cat] * f["comp_adj"])
                if D["buy"][i][ci] < 1.0 / (1.0 + math.exp(-lg)):
                    bought.append(cat)
                    units[cat] += 1
                    reg_items += f["register_only"]
            if bought:
                buyers += 1
                items_total += len(bought)
                revenue += sum(S.CATS[c]["price"] for c in bought)
                margin += sum(S.CATS[c]["price"] * S.CATS[c]["margin"] for c in bought)
                t0 = (OPEN_HOUR + next((h for h, v in enumerate(hcum) if D["h"][i] <= v), len(hcum) - 1)
                      + D["frac"][i]) * 3600
                svc = (SERVICE_BASE + SERVICE_ITEM * len(bought) + SERVICE_REGISTER_IMPULSE * reg_items) \
                    * math.exp(0.25 * D["srv"][i] - 0.03)
                queue.append((t0 + tsum + 8 * len(stops) + 20, svc))
            if day == 0 and len(paths) < n_paths and stops:
                paths.append({"stops": [[z, round(d, 1)] for z, d in stops], "bought": bought})
        # single-register FIFO queue for this day
        queue.sort()
        free, waits, peak, load = 0.0, [], [], 0.0
        for arr, svc in queue:
            start = max(arr, free)
            free = start + svc
            waits.append(start - arr)
            if PEAK[0] <= arr < PEAK[1]:
                peak.append(start - arr)
                load += svc
        out["visitors"].append(per_day)
        out["buyers"].append(buyers)
        out["revenue"].append(revenue)
        out["margin"].append(margin)
        out["items"].append(items_total)
        out["peak_wait"].append(float(np.mean(peak)) if peak else 0.0)
        out["peak_load"].append(load / (PEAK[1] - PEAK[0]))  # register utilisation in the peak window
        out["mean_wait"].append(float(np.mean(waits)) if waits else 0.0)
        for c in S.CAT_LIST:
            cat_units[c].append(units[c])
            exposure[c].append(expo[c] / per_day)
    out["cat_units"], out["exposure"] = cat_units, exposure
    out["cat_revenue"] = {c: [u * S.CATS[c]["price"] for u in v] for c, v in cat_units.items()}
    out["zone_visit_rate"] = {z: v / n_total for z, v in zone_visits.items()}
    out["paths"] = paths
    out["n_shoppers"] = n_total
    return out


def _pct(base: list[float], cand: list[float]) -> dict:
    n = len(base)
    d = np.array(cand) - np.array(base)
    mb = float(np.mean(base))
    if mb == 0:
        return {"base": 0.0, "cand": float(np.mean(cand)), "pct": 0.0, "lo": 0.0, "hi": 0.0}
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    t = _tcrit(n - 1)
    return {"base": mb, "cand": float(np.mean(cand)), "pct": float(d.mean() / mb * 100),
            "lo": float((d.mean() - t * se) / mb * 100), "hi": float((d.mean() + t * se) / mb * 100)}


def compare(base: dict, cand: dict, target: str) -> dict:
    """Paired per-day deltas (90% t-interval) between a candidate run and the baseline run."""
    basket = lambda r: [rev / b if b else 0 for rev, b in zip(r["revenue"], r["buyers"])]
    conv = lambda r: [b / v for b, v in zip(r["buyers"], r["visitors"])]
    attach = lambda r: [u / b if b else 0 for u, b in zip(r["cat_units"][target], r["buyers"])]
    return {
        "target_sales": _pct(base["cat_revenue"][target], cand["cat_revenue"][target]),
        "total_revenue": _pct(base["revenue"], cand["revenue"]),
        "gross_margin": _pct(base["margin"], cand["margin"]),
        "conversion": _pct(conv(base), conv(cand)),
        "basket_value": _pct(basket(base), basket(cand)),
        "target_attach": _pct(attach(base), attach(cand)),
        "target_exposure": _pct(base["exposure"][target], cand["exposure"][target]),
        "congestion": _pct(base["peak_load"], cand["peak_load"]),  # constrained metric (linear in demand)
        "peak_wait": _pct(base["peak_wait"], cand["peak_wait"]),
        "margin_delta_per_day": float(np.mean(np.array(cand["margin"]) - np.array(base["margin"]))),
    }
