"""Store geometry, product catalogue, layouts and layout changes.

Everything here is deterministic. The LLM never edits a layout directly: it proposes
structured *changes*, which are checked by validator.py and applied by apply_changes().
"""
from __future__ import annotations

import copy
import math

STORE_W, STORE_H = 20.0, 12.0
CURRENCY = "₱"
MIN_AISLE_M = 1.2  # minimum clear aisle width after placing a display

# rect = (x, y, w, h) in metres. kind: gate | aisle | island | slot | register
ZONES = {
    "entrance": dict(name="Entrance", rect=(0, 4, 2, 4), kind="gate", restricted=True, capacity=0),
    "bakery": dict(name="Bakery", rect=(3, 0.5, 5, 3), kind="aisle", capacity=2,
                   clearance={"shelf": 1.6, "endcap": 1.4, "signage": 1.6}),
    "coffee": dict(name="Coffee", rect=(8.5, 0.5, 4, 3), kind="aisle", capacity=2,
                   clearance={"shelf": 1.6, "endcap": 1.4, "signage": 1.6}),
    "snacks": dict(name="Snacks", rect=(13, 0.5, 5, 3), kind="aisle", capacity=2,
                   clearance={"shelf": 1.6, "endcap": 1.4, "signage": 1.6}),
    "household": dict(name="Household", rect=(3, 8.5, 5, 3), kind="aisle", capacity=2,
                      clearance={"shelf": 1.6, "endcap": 1.4, "signage": 1.6}),
    "beverages": dict(name="Beverages (refrigeration)", rect=(9, 8.5, 8, 3), kind="aisle", capacity=2,
                      fixed=True, clearance={"shelf": 1.6, "endcap": 1.4, "signage": 1.6}),
    "promo_center": dict(name="Centre island", rect=(9.5, 5, 3, 2), kind="island", capacity=1,
                         clearance={"shelf": 1.3, "endcap": 1.0, "signage": 1.6}),
    "promo_bakery_exit": dict(name="Bakery exit", rect=(6.5, 4, 2, 1.8), kind="slot", capacity=1,
                              clearance={"shelf": 1.5, "endcap": 1.4, "signage": 1.6}),
    "promo_bev_end": dict(name="Beverage end-cap", rect=(13.5, 6.5, 2.5, 1.8), kind="slot", capacity=1,
                          clearance={"shelf": 1.5, "endcap": 1.5, "signage": 1.6}),
    "checkout": dict(name="Checkout", rect=(17, 4.5, 3, 3.5), kind="register", capacity=1, fixed=True,
                     clearance={"shelf": 1.4, "endcap": 1.2, "signage": 1.6}),
}
VISITABLE = ["bakery", "coffee", "snacks", "household", "beverages",
             "promo_center", "promo_bakery_exit", "promo_bev_end"]
CORE_ZONES = ["bakery", "coffee", "snacks", "household", "beverages"]
PASS_THROUGH = {"promo_center", "promo_bakery_exit", "promo_bev_end"}

CATS = {
    "bakery":    dict(name="Bakery",    price=60,  margin=0.45, fixed_fixture=False),
    "coffee":    dict(name="Coffee",    price=85,  margin=0.55, fixed_fixture=False),
    "snacks":    dict(name="Snacks",    price=55,  margin=0.35, fixed_fixture=False),
    "beverages": dict(name="Beverages", price=45,  margin=0.30, fixed_fixture=True),   # refrigeration
    "household": dict(name="Household", price=150, margin=0.25, fixed_fixture=False),
}
CAT_LIST = list(CATS)
COMP = {"coffee": "bakery", "bakery": "coffee", "snacks": "beverages", "beverages": "snacks", "household": None}

# how strongly a display type pulls shoppers toward a zone / holds them there
STRENGTH = {"shelf": 1.0, "endcap": 1.2, "signage": 0.3, "register": 0.0}
DWELL_STRENGTH = {"shelf": 1.0, "endcap": 1.15, "signage": 0.2, "register": 0.0}

# Cost model (PHP) used by the validator
COSTS = {"move_display": 6000, "move_to_island_or_slot": 2500, "move_to_register": 3000,
         "add_endcap": 28000, "add_signage": 3500}

BASE_LAYOUT = [
    {"category": "bakery", "zone": "bakery", "type": "shelf"},
    {"category": "coffee", "zone": "coffee", "type": "shelf"},
    {"category": "snacks", "zone": "snacks", "type": "shelf"},
    {"category": "beverages", "zone": "beverages", "type": "shelf"},
    {"category": "household", "zone": "household", "type": "shelf"},
]

CHANGE_TYPES = ("move_display", "add_endcap", "add_signage")


def centroid(zone: str) -> tuple[float, float]:
    x, y, w, h = ZONES[zone]["rect"]
    return (x + w / 2, y + h / 2)


def dist(a: str, b: str) -> float:
    ax, ay = centroid(a)
    bx, by = centroid(b)
    return math.hypot(ax - bx, ay - by)


def layout_index(layout: list[dict]) -> dict[str, list[tuple[str, str]]]:
    idx: dict[str, list[tuple[str, str]]] = {}
    for d in layout:
        idx.setdefault(d["zone"], []).append((d["category"], d["type"]))
    return idx


def change_cost(change: dict) -> int:
    t, zone = change["type"], change.get("to_zone")
    if t == "add_endcap":
        return COSTS["add_endcap"]
    if t == "add_signage":
        return COSTS["add_signage"]
    cost = COSTS["move_display"]
    kind = ZONES[zone]["kind"]
    if kind in ("island", "slot"):
        cost += COSTS["move_to_island_or_slot"]
    elif kind == "register":
        cost += COSTS["move_to_register"]
    return cost


def apply_changes(layout: list[dict], changes: list[dict]) -> list[dict]:
    """Apply already-validated changes to a copy of the layout."""
    new = copy.deepcopy(layout)
    for ch in changes:
        cat, zone, t = ch["category"], ch["to_zone"], ch["type"]
        if t == "move_display":
            primary = next((d for d in new if d["category"] == cat and d["type"] in ("shelf", "register")), None)
            if primary is not None:
                new.remove(primary)
            new.append({"category": cat, "zone": zone,
                        "type": "register" if ZONES[zone]["kind"] == "register" else "shelf"})
        elif t == "add_endcap":
            new.append({"category": cat, "zone": zone, "type": "endcap"})
        elif t == "add_signage":
            new.append({"category": cat, "zone": zone, "type": "signage"})
    return new


def purchase_features(stops: list[tuple[str, float]], zcats: dict, cat: str):
    """Exposure features for one (journey, category).

    `stops` is the ordered list of (zone, dwell_seconds) *including* the final checkout pass.
    Returns None when the shopper was never exposed to the category.
    Shared by the ground-truth generator and the simulator so both speak the same language.
    """
    exposed = [i for i, (z, _) in enumerate(stops) if any(c == cat for c, _t in zcats.get(z, []))]
    if not exposed:
        return None
    first = exposed[0]
    dwell = 0.0
    endcap = signage = register = 0
    for i in exposed:
        z, d = stops[i]
        dwell += d
        for c, t in zcats.get(z, []):
            if c != cat:
                continue
            endcap |= t == "endcap"
            signage |= t == "signage"
            register |= t == "register"
    comp = COMP[cat]
    comp_before = comp_adj = 0
    if comp:
        comp_before = int(any(any(c == comp for c, _t in zcats.get(stops[j][0], [])) for j in range(first)))
        comp_adj = int(first > 0 and any(c == comp for c, _t in zcats.get(stops[first - 1][0], [])))
    return {
        "dwell10": min(dwell, 60.0) / 10.0,
        "endcap": int(endcap), "signage": int(signage), "register": int(register),
        "comp_before": comp_before, "comp_adj": comp_adj,
        "register_only": int(stops[first][0] == "checkout" and all(stops[i][0] == "checkout" for i in exposed)),
    }
