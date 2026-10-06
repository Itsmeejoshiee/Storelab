"""Synthetic history: 10,000 anonymous shopper journeys + POS transactions.

THIS IS DEMO DATA. It is generated from a hidden "ground-truth" shopper world that the
rest of StoreLab never reads. model.py has to *learn* behaviour from the journeys it
produces, exactly as it would from real CCTV + POS data.

Production replacement: journeys from cv/video_to_journeys.py, transactions from the POS.
"""
from __future__ import annotations

import math

import numpy as np

from . import store as S

MISSIONS = ["breakfast", "snack_run", "drink_grab", "top_up", "browse"]
MISSION_W = [0.28, 0.27, 0.20, 0.15, 0.10]
INTEREST = {
    "breakfast":  {"bakery": .9, "coffee": .8, "snacks": .1, "beverages": .2, "household": .0},
    "snack_run":  {"bakery": .1, "coffee": .1, "snacks": .8, "beverages": .7, "household": .0},
    "drink_grab": {"bakery": .0, "coffee": .15, "snacks": .3, "beverages": .9, "household": .0},
    "top_up":     {"bakery": .1, "coffee": .0, "snacks": .3, "beverages": .4, "household": .9},
    "browse":     {"bakery": .35, "coffee": .35, "snacks": .35, "beverages": .35, "household": .35},
}
A_PULL = 0.8
END_P = [0.08, 0.25, 0.45, 0.65, 0.85, 1.0]  # P(stop shopping | stops so far)
B0 = {"coffee": -3.1, "bakery": -3.1, "snacks": -3.3, "beverages": -2.9, "household": -3.5}
COMP_BEFORE = {"coffee": 0.7, "bakery": 0.4, "snacks": 0.65, "beverages": 0.3, "household": 0.0}
COMP_ADJ = {"coffee": 0.5, "bakery": 0.2, "snacks": 0.5, "beverages": 0.1, "household": 0.0}

HOUR_W = np.array([.03, .05, .07, .05, .05, .06, .06, .05, .06, .08, .14, .14, .09, .07])  # 07:00-21:00
HOUR_W = HOUR_W / HOUR_W.sum()
OPEN_HOUR = 7


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def true_logit(cat: str, mission: str, f: dict) -> float:
    return (B0[cat] + 3.0 * INTEREST[mission][cat] + 0.50 * f["dwell10"] + 0.45 * f["endcap"]
            + 0.10 * f["signage"] - 1.3 * f["register"]
            + COMP_BEFORE[cat] * f["comp_before"] + COMP_ADJ[cat] * f["comp_adj"])


def day_layout(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """A store-day: baseline layout plus occasional past promotions (this is the natural
    variation that lets the model learn what endcaps / signage / register displays do)."""
    promos: list[dict] = []
    if rng.random() > 0.45:
        for _ in range(int(rng.integers(1, 3))):
            kind = rng.choice(["endcap", "signage", "register"], p=[.4, .4, .2])
            if kind == "endcap":
                promos.append({"category": rng.choice(S.CAT_LIST), "zone": str(rng.choice(
                    ["promo_bev_end", "promo_center", "promo_bakery_exit"])), "type": "endcap"})
            elif kind == "signage":
                promos.append({"category": rng.choice(S.CAT_LIST),
                               "zone": str(rng.choice(S.VISITABLE)), "type": "signage"})
            else:
                promos.append({"category": str(rng.choice(["coffee", "snacks", "beverages"])),
                               "zone": "checkout", "type": "register"})
    for p in promos:
        p["category"] = str(p["category"])
    return S.BASE_LAYOUT + promos, promos


def _walk(rng: np.random.Generator, mission: str, zcats: dict) -> list[tuple[str, float]]:
    cur, visited, stops = "entrance", set(), []
    while True:
        n = len(stops)
        if rng.random() < END_P[min(n, len(END_P) - 1)]:
            break
        cands = [z for z in S.VISITABLE if z not in visited]
        if not cands:
            break
        w = []
        for z in cands:
            attract = sum(INTEREST[mission][c] * S.STRENGTH[t] for c, t in zcats.get(z, []))
            pf = 0.35 if z in S.PASS_THROUGH else 1.0
            w.append(math.exp(-S.dist(cur, z) / 5.0) * pf * (1 + A_PULL * attract))
        w = np.array(w) / sum(w)
        z = cands[int(rng.choice(len(cands), p=w))]
        x = max([INTEREST[mission][c] * S.DWELL_STRENGTH[t] for c, t in zcats.get(z, [])] or [0.0])
        mean = 5 + 35 * x
        dwell = float(min(max(math.exp(math.log(mean) - 0.08 + 0.4 * rng.standard_normal()), 2.0), 240.0))
        stops.append((z, dwell))
        visited.add(z)
        cur = z
    return stops


def generate(n_journeys: int = 10_000, n_days: int = 50, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    per_day = n_journeys // n_days
    days, journeys = [], []
    for d in range(n_days):
        layout, promos = day_layout(rng)
        days.append({"day": d, "layout": layout, "promos": promos, "clean": not promos})
        zcats = S.layout_index(layout)
        hours = rng.choice(len(HOUR_W), size=per_day, p=HOUR_W)
        for k in range(per_day):
            mission = MISSIONS[int(rng.choice(len(MISSIONS), p=MISSION_W))]
            stops = _walk(rng, mission, zcats)
            full = stops + [("checkout", 4.0)]
            bought = {}
            for cat in S.CAT_LIST:
                f = S.purchase_features(full, zcats, cat)
                if f and rng.random() < _sigmoid(true_logit(cat, mission, f)):
                    bought[cat] = {"register_only": bool(f["register_only"])}
            t0 = float((OPEN_HOUR + hours[k] + rng.random()) * 3600)
            journeys.append({
                "id": f"anon_{len(journeys):05d}", "day": d, "t0": t0, "stops": stops,
                "bought": bought, "_mission": mission,  # ground truth — never used by the model
            })
    return {"days": days, "journeys": journeys, "seed": seed}


def journey_events(history: dict, seed: int = 3) -> list[dict]:
    """Rows for the `journey_events` table (same schema cv/video_to_journeys.py writes)."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in history["journeys"]:
        t = j["t0"]
        for z, dwell in j["stops"]:
            x, y, w, h = S.ZONES[z]["rect"]
            rows.append({"anonymous_track_id": j["id"], "timestamp": round(t, 1), "store_id": "demo_store_01",
                         "zone_id": z, "x": round(float(x + rng.random() * w), 2),
                         "y": round(float(y + rng.random() * h), 2), "event_type": "zone_visit",
                         "dwell_seconds": round(dwell, 1)})
            t += dwell + 8
    return rows


def transactions(history: dict) -> list[dict]:
    rows, tid = [], 0
    for j in history["journeys"]:
        if not j["bought"]:
            continue
        tid += 1
        for cat in j["bought"]:
            rows.append({"transaction_id": f"T{tid:06d}", "store_id": "demo_store_01",
                         "timestamp": round(j["t0"] + sum(d + 8 for _, d in j["stops"]), 1),
                         "sku": f"{cat}_std", "category": cat, "quantity": 1,
                         "revenue": S.CATS[cat]["price"], "discount": 0.0})
    return rows
