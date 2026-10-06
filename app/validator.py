"""Deterministic constraint validator. The LLM proposes; this decides what is physically legal."""
from __future__ import annotations

from . import store as S

MAX_CHANGES = 3


def validate(changes: list[dict], goal: dict, layout: list[dict] | None = None) -> dict:
    """Return {ok, errors:[{code,message}], cost, layout} for a proposed list of changes."""
    layout = layout or S.BASE_LAYOUT
    errors: list[dict] = []

    def err(code: str, msg: str):
        errors.append({"code": code, "message": msg})

    if not isinstance(changes, list) or not changes:
        err("empty", "An experiment needs at least one change.")
        return {"ok": False, "errors": errors, "cost": 0, "layout": layout}
    if len(changes) > MAX_CHANGES:
        err("too_many", f"At most {MAX_CHANGES} changes per experiment.")

    fixed_cats = set(goal.get("fixed_categories") or []) | {c for c, v in S.CATS.items() if v["fixed_fixture"]}
    fixed_zones = set(goal.get("fixed_zones") or [])
    clean: list[dict] = []
    for ch in changes:
        t, cat, zone = ch.get("type"), ch.get("category"), ch.get("to_zone")
        if t not in S.CHANGE_TYPES:
            err("bad_type", f"Unknown change type '{t}'. Use one of {list(S.CHANGE_TYPES)}.")
            continue
        if cat not in S.CATS:
            err("bad_category", f"Unknown category '{cat}'. Use one of {S.CAT_LIST}.")
            continue
        if zone not in S.ZONES:
            err("bad_zone", f"Unknown zone '{zone}'. Use one of {list(S.ZONES)}.")
            continue
        z = S.ZONES[zone]
        if z.get("restricted"):
            err("restricted_zone", f"{z['name']} is an emergency egress path — nothing may be placed there.")
            continue
        if t == "move_display":
            if cat in fixed_cats:
                why = "refrigeration cannot be moved" if S.CATS[cat]["fixed_fixture"] else "marked as fixed by the manager"
                err("fixed_fixture", f"{S.CATS[cat]['name']} cannot be relocated ({why}).")
                continue
            if z.get("fixed") and z["kind"] != "register":
                err("fixed_zone", f"{z['name']} is a fixed fixture zone; displays cannot be moved into it.")
                continue
        if zone in fixed_zones and t == "move_display":
            err("fixed_zone", f"{z['name']} is marked fixed by the manager.")
            continue
        if t == "add_endcap" and z["kind"] == "register":
            err("bad_placement", "End-caps cannot go at the register; use move_display to the checkout instead.")
            continue
        clean.append({"type": t, "category": cat, "to_zone": zone})

    if errors:
        return {"ok": False, "errors": errors, "cost": 0, "layout": layout}

    new_layout = S.apply_changes(layout, clean)
    # capacity + aisle width after applying
    per_zone: dict[str, list[dict]] = {}
    for d in new_layout:
        per_zone.setdefault(d["zone"], []).append(d)
    touched = {c["to_zone"] for c in clean}
    for zone in touched:
        z = S.ZONES[zone]
        phys = [d for d in per_zone.get(zone, []) if d["type"] != "signage"]
        if len(phys) > z["capacity"]:
            err("capacity", f"{z['name']} holds at most {z['capacity']} physical display(s); this would make {len(phys)}.")
        for d in per_zone.get(zone, []):
            width = z.get("clearance", {}).get(d["type"], 9.9)
            if d["type"] != "register" and width < S.MIN_AISLE_M:
                err("aisle_width", f"A {d['type']} in {z['name']} leaves a {width:.1f} m aisle, below the "
                                   f"{S.MIN_AISLE_M:.1f} m minimum.")
            if d["type"] == "register" and z["clearance"]["shelf"] < S.MIN_AISLE_M:
                err("aisle_width", "Register display would block the queue lane.")
    seen = set()
    for d in new_layout:
        key = (d["category"], d["zone"], d["type"])
        if key in seen:
            err("duplicate", f"{S.CATS[d['category']]['name']} already has a {d['type']} display in {S.ZONES[d['zone']]['name']}.")
        seen.add(key)

    cost = sum(S.change_cost(c) for c in clean)
    budget = goal.get("budget")
    if budget is not None and cost > budget:
        err("budget", f"Estimated cost {S.CURRENCY}{cost:,} exceeds the {S.CURRENCY}{budget:,} budget.")
    return {"ok": not errors, "errors": errors, "cost": cost, "layout": new_layout, "changes": clean}
