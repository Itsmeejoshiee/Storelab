"""StoreLab Agent: objective -> diagnosis -> propose -> validate -> simulate -> critique -> recommend.

The loop is explicit code (not hidden inside a framework) so every step is a visible tool call:
    get_store_metrics · validate_experiment · run_simulation · compare_experiments · recommend_physical_test
Gemini (when configured) acts as the planner/critic; otherwise a rule-based planner fills in.
Gemini never produces numbers: every figure shown comes from the simulator or the analytics.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Iterator

from . import history as H
from . import store as S
from .llm import LLM, parse_goal
from .model import Model
from .sim import DAYS, PER_DAY, compare, simulate
from .validator import validate

N_PATHS = 70
MAX_ATTEMPTS = 6
HORIZON_DAYS = 90  # value horizon used to rank candidates (net margin vs. cost)


class Lab:
    """Everything loaded once at start-up: synthetic history, fitted model, baseline run, LLM."""

    def __init__(self, seed: int = 7):
        self.history = H.generate(seed=seed)
        self.model = Model(self.history)
        self.analytics = self.model.analytics()
        self.baseline = simulate(self.model, S.BASE_LAYOUT, n_paths=N_PATHS)
        self.llm = LLM()
        self.calibration = self._calibration()

    def _calibration(self) -> dict:
        clean = self.model._clean()
        n = len(clean)
        vr = Counter()
        for j in clean:
            for z in {z for z, _ in j["stops"]}:
                vr[z] += 1
        b = self.baseline
        vis = sum(b["visitors"])
        rows = []
        for c in S.CAT_LIST:
            a = sum(c in j["bought"] for j in clean) / n
            s = sum(b["cat_units"][c]) / vis
            rows.append({"metric": f"{S.CATS[c]['name']} purchases / visitor", "history": a, "simulated": s,
                         "error_pct": (s - a) / a * 100 if a else 0})
        for z in S.VISITABLE:
            a, s = vr[z] / n, b["zone_visit_rate"][z]
            rows.append({"metric": f"{S.ZONES[z]['name']} visit rate", "history": a, "simulated": s,
                         "error_pct": (s - a) / a * 100 if a else 0})
        a = sum(bool(j["bought"]) for j in clean) / n
        s = sum(b["buyers"]) / vis
        rows.append({"metric": "Conversion", "history": a, "simulated": s, "error_pct": (s - a) / a * 100})
        mape = sum(abs(r["error_pct"]) for r in rows) / len(rows)
        return {"rows": rows, "mean_abs_error_pct": mape, "note": "Baseline-layout days only. Fitted on 10,000 synthetic journeys."}


# ------------------------------------------------------------------------------------ diagnosis
def diagnose(lab: Lab, goal: dict) -> list[dict]:
    a, t = lab.analytics, goal["target_category"]
    zones = a["zones"]
    tz = zones[t]
    core = [zones[z] for z in S.CORE_ZONES]
    avg_conv = sum(z["conversion"] for z in core) / len(core)
    items = []
    items.append({
        "observation": f"{tz['name']} is passed by {tz['visit_rate']:.0%} of shoppers and converts {tz['conversion']:.0%} of them "
                       f"(average aisle: {avg_conv:.0%}).",
        "pattern": f"{tz['engaged']:,} of {tz['visitors']:,} visitors slowed down to engage; {tz['purchases']:,} bought "
                   f"{S.CATS[t]['name'].lower()}.",
        "hypothesis": f"Exposure to {S.CATS[t]['name'].lower()} is the limiting factor, so adding exposure on common routes may raise sales.",
        "evidence": {"visit_rate": tz["visit_rate"], "conversion": tz["conversion"]}})
    cl = next((c for c in a["complement_lift"] if c["category"] == t), None)
    if cl:
        items.append({
            "observation": f"Shoppers who saw {S.CATS[cl['after']]['name']} earlier on their path bought {S.CATS[t]['name']} "
                           f"{cl['with_comp_rate']:.0%} of the time, versus {cl['without_comp_rate']:.0%} otherwise.",
            "pattern": f"Purchase likelihood is {cl['lift']:.1f}× higher after {S.CATS[cl['after']]['name']} "
                       f"({cl['n_with']:,} vs {cl['n_without']:,} journeys).",
            "hypothesis": f"Placing {S.CATS[t]['name'].lower()} where {S.CATS[cl['after']]['name'].lower()} shoppers walk next may lift attachment.",
            "evidence": cl})
    slots = sorted((zones[z] for z in S.PASS_THROUGH), key=lambda z: -z["visit_rate"])
    items.append({
        "observation": f"The {slots[0]['name'].lower()} area sees {slots[0]['visit_rate']:.0%} of shoppers but holds no product "
                       f"(avg stop {slots[0]['avg_dwell']:.0f}s).",
        "pattern": "Empty pass-through space on busy routes is unused selling space.",
        "hypothesis": "A display in a high-traffic slot may reach shoppers who never enter the category aisle.",
        "evidence": {"zone": slots[0]["zone"], "visit_rate": slots[0]["visit_rate"]}})
    b = lab.baseline
    peak_hrs = sorted(a["hourly"], key=lambda h: -h["journeys"])[:2]
    items.append({
        "observation": f"Checkout is busiest around {min(h['hour'] for h in peak_hrs)}:00–{max(h['hour'] for h in peak_hrs) + 1}:00; the single "
                       f"register runs at {sum(b['peak_load']) / len(b['peak_load']):.0%} utilisation with ~{sum(b['peak_wait']) / len(b['peak_wait']):.0f}s average wait.",
        "pattern": "Every extra item adds service time, and impulse buys at the register add more.",
        "hypothesis": "Experiments that add purchases at the register are likely to worsen congestion.",
        "evidence": {"peak_load": sum(b["peak_load"]) / len(b["peak_load"])}})
    if lab.llm.live:
        out = lab.llm.json(
            "Rewrite each retail-analytics item so 'hypothesis' is phrased as a hypothesis, never a conclusion. Keep every "
            "number exactly as given. Write in the language of the manager's request. Return JSON {\"items\":[{observation,pattern,hypothesis}]} "
            "with the same number of items in the same order.",
            json.dumps({"request_language": goal.get("language", "en"),
                        "items": [{k: i[k] for k in ("observation", "pattern", "hypothesis")} for i in items]}))
        if out and isinstance(out.get("items"), list) and len(out["items"]) == len(items):
            for src, new in zip(items, out["items"]):
                for k in ("observation", "pattern", "hypothesis"):
                    if isinstance(new.get(k), str) and new[k].strip():
                        src[k] = new[k].strip()
    return items


# ------------------------------------------------------------------------------------ planners
def _slot_for(cat: str) -> str:
    comp = S.COMP.get(cat)
    return {"bakery": "promo_bakery_exit", "beverages": "promo_bev_end"}.get(comp, "promo_bakery_exit"
                                                                              if cat != "snacks" else "promo_bev_end")


def offline_templates(goal: dict) -> list[dict]:
    t = goal["target_category"]
    name = S.CATS[t]["name"]
    slot = _slot_for(t)
    slot_name = S.ZONES[slot]["name"].lower()
    comp = S.COMP.get(t)
    comp_txt = f" right after {S.CATS[comp]['name'].lower()} shoppers" if comp else ""
    other = "promo_center" if slot != "promo_center" else "promo_bakery_exit"
    return [
        {"key": "register", "name": f"{name} at the register",
         "rationale": f"Every shopper passes the checkout, so moving {name.lower()} there maximises exposure.",
         "changes": [{"type": "move_display", "category": t, "to_zone": "checkout"}]},
        {"key": "relocate", "name": f"Relocate {name} to the {slot_name}",
         "rationale": f"Move the whole {name.lower()} display to the {slot_name}{comp_txt}.",
         "changes": [{"type": "move_display", "category": t, "to_zone": slot}]},
        {"key": "endcap_center", "name": f"{name} end-cap on the centre island",
         "rationale": "The centre island sees steady traffic; add an end-cap display there and keep the aisle display.",
         "changes": [{"type": "add_endcap", "category": t, "to_zone": "promo_center"}]},
        {"key": "endcap", "name": f"{name} promotional end-cap at the {slot_name}",
         "rationale": f"Keep the aisle display and add an end-cap at the {slot_name}{comp_txt}.",
         "changes": [{"type": "add_endcap", "category": t, "to_zone": slot}]},
        {"key": "signage", "name": f"{name} signage at the {slot_name}",
         "rationale": f"A low-cost sign at the {slot_name} points shoppers back to {name.lower()} without moving fixtures.",
         "changes": [{"type": "add_signage", "category": t, "to_zone": slot}]},
        {"key": "combo", "name": f"{name} end-cap plus signage on a second route",
         "rationale": f"Combine the {slot_name} end-cap with signage at {S.ZONES[other]['name'].lower()} to reach two routes.",
         "changes": [{"type": "add_endcap", "category": t, "to_zone": slot},
                     {"type": "add_signage", "category": t, "to_zone": other}]},
    ]


def _signature(changes: list[dict]) -> str:
    return "|".join(sorted(f"{c.get('type')}:{c.get('category')}:{c.get('to_zone')}" for c in changes))


def offline_propose(goal: dict, attempts: list[dict]) -> dict | None:
    done = {a["signature"] for a in attempts}
    banned_zones, banned_types = set(), set()
    for a in attempts:
        codes = {e["code"] for e in a.get("errors", [])}
        if a["verdict"] == "violates" or "aisle_width" in codes:
            for c in a["changes"]:
                if c["to_zone"] == "checkout" or "aisle_width" in codes:
                    banned_zones.add(c["to_zone"])
    for tpl in offline_templates(goal):
        sig = _signature(tpl["changes"])
        if sig in done:
            continue
        if any(c["to_zone"] in banned_zones for c in tpl["changes"]):
            continue
        # skip combos we already know can't fit the budget
        cost = sum(S.change_cost(c) for c in tpl["changes"])
        if tpl["key"] == "combo" and goal.get("budget") and cost > goal["budget"]:
            continue
        return tpl
    return None


PLANNER_SYSTEM = f"""You are StoreLab's spatial experiment designer for a convenience store. You propose ONE physical store experiment per call,
learning from the simulator results of earlier attempts. You never invent numbers; a deterministic simulator evaluates your proposal.

Allowed change types (max 3 changes per experiment):
 move_display {{category, to_zone}}  relocate a category's main display (beverages refrigeration is fixed)
 add_endcap   {{category, to_zone}}  add an end-cap display (about {S.CURRENCY}{S.COSTS['add_endcap']:,}); not allowed at the register
 add_signage  {{category, to_zone}}  add signage (about {S.CURRENCY}{S.COSTS['add_signage']:,}); no floor space needed
Zones: {", ".join(f"{k} ({v['name']}, {v['kind']})" for k, v in S.ZONES.items())}.
Categories: {", ".join(S.CAT_LIST)}. 'entrance' is an emergency egress path.
Rules enforced by a validator: budget, aisle width >= {S.MIN_AISLE_M} m, display capacity per zone, fixed fixtures, and a simulator-measured
peak checkout-load limit. Learn from rejections and congestion violations; do not repeat an attempt.
Return ONLY JSON: {{"critique": "<1-2 sentences on what the last result teaches, or empty for the first attempt>",
 "experiment": {{"name": "...", "rationale": "...", "changes": [{{"type": "...", "category": "...", "to_zone": "..."}}]}}, "stop": false}}.
Set "stop": true (and still return an experiment equal to your best earlier one) only when further attempts are unlikely to help."""


def llm_propose(lab: Lab, goal: dict, attempts: list[dict]) -> dict | None:
    zones = lab.analytics["zones"]
    ctx = {
        "goal": goal,
        "baseline_layout": S.BASE_LAYOUT,
        "zone_traffic": {z: {"visit_rate": round(v["visit_rate"], 3), "conversion": round(v["conversion"], 3)} for z, v in zones.items()},
        "complement_lift": lab.analytics["complement_lift"],
        "learned_effects": lab.analytics["learned_effects"],
        "previous_attempts": [{
            "id": a["id"], "name": a["name"], "changes": a["changes"], "verdict": a["verdict"],
            "validator_errors": [e["message"] for e in a.get("errors", [])],
            "target_sales_pct": round(a["metrics"]["target_sales"]["pct"], 1) if a.get("metrics") else None,
            "peak_load_pct": round(a["metrics"]["congestion"]["pct"], 1) if a.get("metrics") else None,
            "cost": a.get("cost")} for a in attempts],
    }
    out = lab.llm.json(PLANNER_SYSTEM, json.dumps(ctx, ensure_ascii=False), temperature=0.5)
    if not isinstance(out, dict) or not isinstance(out.get("experiment"), dict):
        return None
    exp = out["experiment"]
    return {"name": str(exp.get("name") or "Gemini experiment")[:90], "rationale": str(exp.get("rationale") or "")[:400],
            "changes": exp.get("changes") if isinstance(exp.get("changes"), list) else [],
            "critique": str(out.get("critique") or ""), "stop": bool(out.get("stop"))}


# ------------------------------------------------------------------------------------ evaluation
def _summarise(m: dict) -> dict:
    """Trim a compare() result to what the UI/LLM need."""
    return m


def evaluate(goal: dict, cost: int, metrics: dict) -> tuple[str, list[str], bool]:
    violations = []
    cong = metrics["congestion"]["pct"]
    if cong > goal["max_congestion_pct"]:
        violations.append(f"Peak checkout load {cong:+.1f}% exceeds the +{goal['max_congestion_pct']:g}% limit")
    meets = metrics["target_sales"]["pct"] >= goal["uplift_pct"]
    if violations:
        return "violates", violations, meets
    if meets:
        return "qualifies", [], True
    if metrics["target_sales"]["pct"] > 0:
        return "below_goal", [], False
    return "loses_sales", [], False


def offline_critique(goal: dict, a: dict) -> str:
    t = S.CATS[goal["target_category"]]["name"].lower()
    if a["verdict"] == "rejected":
        msg = a["errors"][0]["message"] if a["errors"] else "invalid"
        return f"Rejected before simulation — {msg}"
    m = a["metrics"]
    tgt, cong, expo = m["target_sales"]["pct"], m["congestion"]["pct"], m["target_exposure"]["pct"]
    if a["verdict"] == "violates":
        at_register = any(c["to_zone"] == "checkout" for c in a["changes"])
        why = ("Impulse purchases at the register lengthen service time; avoid register placements."
               if at_register else "The extra purchases add service time at the single register; look for a cheaper or smaller lift.")
        return (f"Sales {tgt:+.0f}% but peak checkout load {cong:+.1f}% breaks the +{goal['max_congestion_pct']:g}% limit. {why}")
    if a["verdict"] == "loses_sales":
        return (f"Net loss: {t} sales {tgt:+.0f}% because exposure changes by {expo:+.0f}% — shoppers who used to walk to the "
                f"{t} aisle no longer find it. Keep the aisle display and add exposure instead of relocating.")
    if a["verdict"] == "below_goal":
        return f"Positive ({tgt:+.0f}%) but short of the {goal['uplift_pct']:g}% goal."
    return (f"Meets the goal: {t} sales {tgt:+.0f}% (90% CI {m['target_sales']['lo']:+.0f}…{m['target_sales']['hi']:+.0f}%) with "
            f"peak load {cong:+.1f}% for {S.CURRENCY}{a['cost']:,}.")


def net_value(a: dict) -> float:
    return a["metrics"]["margin_delta_per_day"] * HORIZON_DAYS - a["cost"]


# ------------------------------------------------------------------------------------ pilot
def pilot_plan(lab: Lab, goal: dict, rec: dict) -> dict:
    t = goal["target_category"]
    base = lab.baseline
    p0 = sum(base["cat_units"][t]) / sum(base["visitors"])
    mde = max(5.0, min(goal["uplift_pct"], rec["metrics"]["target_sales"]["pct"])) / 100
    p1 = p0 * (1 + mde)
    n_arm = math.ceil((1.96 + 0.84) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2)
    stores_per_arm = 2
    days = max(14, 7 * math.ceil(n_arm / (stores_per_arm * PER_DAY) / 7))
    return {
        "test_stores": stores_per_arm, "control_stores": stores_per_arm, "duration_days": days,
        "primary_kpi": f"{S.CATS[t]['name']} purchase rate per visitor",
        "secondary_kpis": ["Peak-hour checkout wait", "Average basket value", "Total gross margin"],
        "baseline_rate": p0, "min_detectable_lift_pct": mde * 100, "visitors_needed_per_arm": n_arm,
        "assumptions": f"80% power, 5% significance, ~{PER_DAY:,} visitors per store per day, matched test/control stores.",
        "success_rule": f"Ship if {S.CATS[t]['name'].lower()} rate rises ≥{mde * 100:.0f}% vs control and peak wait stays within "
                        f"+{goal['max_congestion_pct']:g}% of baseline load.",
    }


# ------------------------------------------------------------------------------------ the loop
def run_agent(lab: Lab, text: str, overrides: dict | None = None) -> Iterator[dict]:
    llm = lab.llm
    yield {"type": "mode", "planner": llm.label, "live": llm.live,
           "note": None if llm.live else "Set GEMINI_API_KEY (or Vertex AI env vars) to switch the planner, goal parser and "
                                         "explanations to Gemini. The simulator, validator and numbers are identical either way."}
    goal, parser = parse_goal(llm, text, overrides)
    yield {"type": "goal", "goal": goal, "parser": parser}
    yield {"type": "trace", "tool": "get_store_metrics", "detail": f"{lab.analytics['totals']['journeys']:,} journeys, "
           f"{lab.analytics['totals']['transactions']:,} transactions (synthetic demo data)"}
    yield {"type": "diagnosis", "items": diagnose(lab, goal)}

    attempts: list[dict] = []
    last_critique = ""
    for k in range(MAX_ATTEMPTS):
        aid = chr(ord("A") + k)
        prop, source = None, "offline"
        if llm.live:
            prop = llm_propose(lab, goal, attempts)
            source = "gemini"
            if prop is None:
                yield {"type": "trace", "tool": "planner", "detail": f"Gemini call failed ({llm.error}); using offline planner for this step"}
        if prop is None:
            prop, source = offline_propose(goal, attempts), "offline"
        if prop is None:
            break
        if prop.get("stop") and any(a["verdict"] == "qualifies" for a in attempts) and len(attempts) >= 3:
            break
        crit_prev = prop.get("critique") or last_critique
        changes = prop["changes"]
        sig = _signature(changes)
        yield {"type": "attempt_start", "id": aid, "name": prop["name"], "rationale": prop["rationale"],
               "changes": changes, "source": source, "critique_of_previous": crit_prev}
        a = {"id": aid, "name": prop["name"], "rationale": prop["rationale"], "changes": changes, "signature": sig,
             "source": source, "errors": [], "metrics": None, "cost": 0}
        if any(x["signature"] == sig for x in attempts):
            v = {"ok": False, "errors": [{"code": "duplicate_attempt", "message": "Same as an earlier attempt."}], "cost": 0}
        else:
            yield {"type": "trace", "tool": "validate_experiment", "detail": f"{aid}: {len(changes)} change(s)"}
            v = validate(changes, goal)
        a["errors"], a["cost"] = v["errors"], v["cost"]
        yield {"type": "validation", "id": aid, "ok": v["ok"], "errors": v["errors"], "cost": v["cost"]}
        if not v["ok"]:
            a["verdict"] = "rejected"
        else:
            yield {"type": "trace", "tool": "run_simulation", "detail": f"{aid}: {DAYS * PER_DAY:,} synthetic shoppers"}
            yield {"type": "simulating", "id": aid}
            res = simulate(lab.model, v["layout"], n_paths=N_PATHS)
            a["metrics"] = compare(lab.baseline, res, goal["target_category"])
            a["verdict"], violations, a["meets_goal"] = evaluate(goal, a["cost"], a["metrics"])
            a["violations"] = violations
            a["layout"], a["paths"] = v["layout"], res["paths"]
            a["payback_days"] = (a["cost"] / a["metrics"]["margin_delta_per_day"]) if a["metrics"]["margin_delta_per_day"] > 0 else None
            a["net_value"] = net_value(a)
        a["critique"] = offline_critique(goal, a)
        attempts.append(a)
        yield {"type": "result", "id": aid, "verdict": a["verdict"], "metrics": a["metrics"], "cost": a["cost"],
               "violations": a.get("violations", []), "errors": a["errors"], "layout": a.get("layout"),
               "paths": a.get("paths", [])[:N_PATHS], "payback_days": a.get("payback_days"), "net_value": a.get("net_value"),
               "critique": a["critique"]}
        last_critique = a["critique"]
        if len(attempts) >= 5 and any(x["verdict"] == "qualifies" for x in attempts):
            break

    yield from _recommend(lab, goal, attempts)


def _recommend(lab: Lab, goal: dict, attempts: list[dict]) -> Iterator[dict]:
    yield {"type": "trace", "tool": "compare_experiments", "detail": f"{len(attempts)} attempts ranked by {HORIZON_DAYS}-day net margin"}
    sim_ok = [a for a in attempts if a["metrics"]]
    qualifying = [a for a in sim_ok if a["verdict"] == "qualifies"]
    pool = qualifying or [a for a in sim_ok if a["verdict"] in ("below_goal",)]
    ranking = sorted(qualifying, key=net_value, reverse=True) + [a for a in sim_ok if a not in qualifying]
    if not pool:
        yield {"type": "recommendation", "found": False,
               "message": "No candidate met the goal within the constraints. Try a larger budget, a lower target, or a looser congestion limit.",
               "ranking": [_rank_row(a) for a in ranking]}
        yield {"type": "done"}
        return
    best = max(pool, key=net_value) if qualifying else max(pool, key=lambda a: a["metrics"]["target_sales"]["pct"])
    yield {"type": "trace", "tool": "recommend_physical_test", "detail": f"{best['id']}: {best['name']}"}
    t = goal["target_category"]
    m = best["metrics"]
    facts = {
        "recommended": {"name": best["name"], "changes": best["changes"], "cost": best["cost"]},
        "target_sales_pct": round(m["target_sales"]["pct"], 1), "ci90": [round(m["target_sales"]["lo"], 1), round(m["target_sales"]["hi"], 1)],
        "peak_load_pct": round(m["congestion"]["pct"], 1), "exposure_pct": round(m["target_exposure"]["pct"], 1),
        "total_revenue_pct": round(m["total_revenue"]["pct"], 1),
        "complement_lift": next((c for c in lab.analytics["complement_lift"] if c["category"] == t), None),
        "rejected_or_worse": [{"name": a["name"], "verdict": a["verdict"], "why": a["critique"]} for a in attempts if a is not best],
    }
    narrative, source = None, "offline"
    if lab.llm.live:
        out = lab.llm.json(
            "You explain a retail-store experiment recommendation to a store manager in 3-4 plain sentences, in the language of the "
            "request. Use ONLY the numbers in the JSON; never add numbers. Frame results as simulated estimates to be confirmed in a "
            'physical pilot. Return JSON {"why": "..."}.',
            json.dumps({"request_language": goal.get("language", "en"), "facts": facts}, ensure_ascii=False))
        if out and isinstance(out.get("why"), str) and out["why"].strip():
            narrative, source = out["why"].strip(), "gemini"
    if narrative is None:
        cl = facts["complement_lift"]
        parts = []
        if cl:
            parts.append(f"In historical journeys, shoppers who passed {S.CATS[cl['after']]['name'].lower()} first bought "
                         f"{S.CATS[t]['name'].lower()} {cl['lift']:.1f}× as often.")
        parts.append(f"This design lifts exposure to {S.CATS[t]['name'].lower()} by {facts['exposure_pct']:+.0f}% without removing the existing "
                     f"aisle display, so simulated {S.CATS[t]['name'].lower()} sales rise {facts['target_sales_pct']:+.0f}% "
                     f"(90% CI {facts['ci90'][0]:+.0f}…{facts['ci90'][1]:+.0f}%).")
        parts.append(f"Peak checkout load moves {facts['peak_load_pct']:+.1f}%, within your +{goal['max_congestion_pct']:g}% limit. "
                     f"These are simulated estimates — confirm with the physical pilot below.")
        narrative = " ".join(parts)
    pilot = pilot_plan(lab, goal, best)
    yield {"type": "recommendation", "found": True, "id": best["id"], "name": best["name"], "changes": best["changes"],
           "rationale": best["rationale"], "cost": best["cost"], "metrics": m, "payback_days": best.get("payback_days"),
           "net_value": net_value(best), "horizon_days": HORIZON_DAYS, "why": narrative, "why_source": source,
           "meets_goal": best["verdict"] == "qualifies", "layout": best["layout"], "paths": best["paths"],
           "baseline_paths": lab.baseline["paths"], "pilot": pilot, "ranking": [_rank_row(a) for a in ranking],
           "complement_lift": facts["complement_lift"], "assumptions": {
               "visitors_per_day": PER_DAY, "simulated_days": DAYS,
               "data": "Synthetic demo data — 10,000 journeys. Replace with CCTV-derived journeys + POS."}}
    yield {"type": "done"}


def _rank_row(a: dict) -> dict:
    m = a.get("metrics")
    return {"id": a["id"], "name": a["name"], "verdict": a["verdict"], "cost": a["cost"],
            "target_sales_pct": m["target_sales"]["pct"] if m else None,
            "congestion_pct": m["congestion"]["pct"] if m else None,
            "net_value": net_value(a) if m else None}
