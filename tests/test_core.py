import math

import pytest
from fastapi.testclient import TestClient

from app import store as S
from app.agent import Lab, run_agent
from app.llm import parse_goal_offline
from app.main import app
from app.sim import compare, simulate
from app.validator import validate

GOAL = {"budget": 30000, "fixed_categories": ["beverages"], "max_congestion_pct": 5, "target_category": "snacks", "uplift_pct": 10}


@pytest.fixture(scope="module")
def lab():
    return Lab()


# ------------------------------------------------------------------ validator
def test_validator_accepts_signage():
    v = validate([{"type": "add_signage", "category": "snacks", "to_zone": "promo_bev_end"}], GOAL)
    assert v["ok"] and v["cost"] == S.COSTS["add_signage"]


@pytest.mark.parametrize("change,code", [
    ({"type": "move_display", "category": "beverages", "to_zone": "promo_center"}, "fixed_fixture"),
    ({"type": "add_endcap", "category": "snacks", "to_zone": "promo_center"}, "aisle_width"),
    ({"type": "add_signage", "category": "snacks", "to_zone": "entrance"}, "restricted_zone"),
    ({"type": "add_endcap", "category": "snacks", "to_zone": "checkout"}, "bad_placement"),
    ({"type": "add_endcap", "category": "snacks", "to_zone": "nowhere"}, "bad_zone"),
    ({"type": "teleport", "category": "snacks", "to_zone": "snacks"}, "bad_type"),
])
def test_validator_rejects(change, code):
    v = validate([change], GOAL)
    assert not v["ok"] and code in {e["code"] for e in v["errors"]}


def test_validator_budget_and_capacity():
    v = validate([{"type": "add_endcap", "category": "snacks", "to_zone": "promo_bev_end"}], {**GOAL, "budget": 10000})
    assert "budget" in {e["code"] for e in v["errors"]}
    two = [{"type": "add_endcap", "category": "snacks", "to_zone": "promo_bev_end"},
           {"type": "add_endcap", "category": "coffee", "to_zone": "promo_bev_end"}]
    assert "capacity" in {e["code"] for e in validate(two, {**GOAL, "budget": 99999})["errors"]}


def test_manager_fixed_category_cannot_move():
    v = validate([{"type": "move_display", "category": "coffee", "to_zone": "promo_bakery_exit"}], {**GOAL, "fixed_categories": ["coffee"]})
    assert "fixed_fixture" in {e["code"] for e in v["errors"]}


# ------------------------------------------------------------------ goal parsing
def test_parse_goal_english():
    g = parse_goal_offline("Increase snack sales by at least 10%. Don't move refrigeration. Budget ₱30,000. Don't increase checkout congestion.")
    assert (g["target_category"], g["uplift_pct"], g["budget"], g["max_congestion_pct"]) == ("snacks", 10.0, 30000, 5.0)
    assert "beverages" in g["fixed_categories"]


def test_parse_goal_variants():
    g = parse_goal_offline("Increase coffee sales by 12%, budget P40k, keep checkout queue under 8%")
    assert (g["target_category"], g["uplift_pct"], g["budget"], g["max_congestion_pct"]) == ("coffee", 12.0, 40000, 8.0)


# ------------------------------------------------------------------ model & simulator
def test_learned_effects_close_to_demo_world(lab):
    eff = lab.analytics["learned_effects"]["display"]
    assert eff["register"] < 0.6          # register impulse buys are less likely (world: 0.27)
    assert 1.0 < eff["endcap"] < 2.5      # end-caps help (world: 1.57)


def test_baseline_calibration(lab):
    assert lab.calibration["mean_abs_error_pct"] < 15
    conv = next(r for r in lab.calibration["rows"] if r["metric"] == "Conversion")
    assert abs(conv["error_pct"]) < 15


def test_simulator_is_deterministic(lab):
    a = simulate(lab.model, S.BASE_LAYOUT, days=3, per_day=300, seed=3)
    b = simulate(lab.model, S.BASE_LAYOUT, days=3, per_day=300, seed=3)
    assert a["revenue"] == b["revenue"]


def test_signage_helps_and_register_hurts_congestion(lab):
    base = lab.baseline
    sign = simulate(lab.model, S.apply_changes(S.BASE_LAYOUT, [{"type": "add_signage", "category": "snacks", "to_zone": "promo_bev_end"}]))
    reg = simulate(lab.model, S.apply_changes(S.BASE_LAYOUT, [{"type": "move_display", "category": "snacks", "to_zone": "checkout"}]))
    cs, cr = compare(base, sign, "snacks"), compare(base, reg, "snacks")
    assert cs["target_sales"]["pct"] > 5
    assert cr["congestion"]["pct"] > cs["congestion"]["pct"]


def test_relocating_a_category_loses_sales(lab):
    r = simulate(lab.model, S.apply_changes(S.BASE_LAYOUT, [{"type": "move_display", "category": "snacks", "to_zone": "promo_bev_end"}]))
    assert compare(lab.baseline, r, "snacks")["target_sales"]["pct"] < 0


# ------------------------------------------------------------------ agent
def test_agent_never_recommends_a_violation(lab):
    events = list(run_agent(lab, "Increase snack sales by 10%. Budget ₱30k. Don't increase checkout congestion."))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "mode" and kinds[-1] == "done" and "diagnosis" in kinds
    results = [e for e in events if e["type"] == "result"]
    assert any(r["verdict"] == "rejected" for r in results), "validator should reject at least one idea"
    assert any(r["verdict"] == "violates" for r in results), "simulator should catch a constraint violation"
    rec = next(e for e in events if e["type"] == "recommendation")
    assert rec["found"] and rec["metrics"]["congestion"]["pct"] <= 5 and rec["cost"] <= 30000
    assert rec["pilot"]["duration_days"] >= 14


def test_agent_reports_when_nothing_qualifies(lab):
    events = list(run_agent(lab, "Increase coffee sales by 12%. Budget ₱40k. Keep checkout congestion within 1%."))
    rec = next(e for e in events if e["type"] == "recommendation")
    assert rec["found"] is False


# ------------------------------------------------------------------ API
def test_api():
    with TestClient(app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/api/store").json()["zones"]["coffee"]["name"] == "Coffee"
        a = c.get("/api/analytics").json()
        assert a["synthetic"] and a["totals"]["journeys"] == 10000
        r = c.post("/api/experiments/stream", json={"objective": "Increase snack sales by 10%. Budget ₱30k."})
        lines = [l for l in r.text.splitlines() if l.strip()]
        assert '"type": "recommendation"' in lines[-2] and '"type": "done"' in lines[-1]


# ------------------------------------------------------------------ Gemini path (stubbed client, no network)
class FakeGemini:
    """Stands in for app.llm.LLM with live=True to exercise the Gemini-driven code paths."""
    live = True
    label = "fake-gemini"
    error = None

    def __init__(self):
        self.calls = []
        self.plan = [
            {"name": "Snack signage", "rationale": "cheap", "changes": [{"type": "add_signage", "category": "snacks", "to_zone": "promo_bev_end"}]},
            {"name": "Illegal fridge move", "rationale": "bad", "changes": [{"type": "move_display", "category": "beverages", "to_zone": "promo_center"}]},
            {"name": "Illegal fridge move", "rationale": "dup", "changes": [{"type": "move_display", "category": "beverages", "to_zone": "promo_center"}]},
        ]

    def json(self, system, prompt, temperature=0.4):
        self.calls.append(system[:30])
        if system.startswith("You turn a retail manager"):
            return {"target_category": "snacks", "uplift_pct": 10, "budget": 30000, "max_congestion_pct": 5,
                    "fixed_categories": [], "language": "tl", "summary": "Dagdagan ang benta ng snacks"}
        if system.startswith("Rewrite each retail"):
            items = __import__("json").loads(prompt)["items"]
            return {"items": [{**i, "hypothesis": "Hypothesis: " + i["hypothesis"]} for i in items]}
        if system.startswith("You are StoreLab"):
            exp = self.plan.pop(0) if self.plan else self.plan_last
            self.plan_last = exp
            return {"critique": "learned something", "experiment": exp, "stop": len(self.plan) == 0 and exp["name"] == "Snack signage"}
        if system.startswith("You explain"):
            return {"why": "Gemini says: signage is cheap and works."}
        return None


def test_gemini_driven_loop_is_validated_by_code(lab):
    original = lab.llm
    lab.llm = FakeGemini()
    try:
        events = list(run_agent(lab, "Dagdagan ng 10% ang benta ng snacks"))
    finally:
        fake, lab.llm = lab.llm, original
    goal = next(e for e in events if e["type"] == "goal")
    assert goal["parser"] == "gemini" and "beverages" in goal["goal"]["fixed_categories"]
    assert next(e for e in events if e["type"] == "diagnosis")["items"][0]["hypothesis"].startswith("Hypothesis:")
    verdicts = {e["id"]: e["verdict"] for e in events if e["type"] == "result"}
    assert verdicts["A"] == "qualifies"
    assert verdicts["B"] == "rejected" and verdicts["C"] == "rejected"      # LLM output never bypasses the validator
    rec = next(e for e in events if e["type"] == "recommendation")
    assert rec["why_source"] == "gemini" and rec["name"] == "Snack signage"


def test_llm_failure_falls_back_to_offline(lab):
    class Down(FakeGemini):
        def json(self, *a, **k):
            return None
    original, lab.llm = lab.llm, Down()
    try:
        events = list(run_agent(lab, "Increase snack sales by 10%. Budget ₱30k."))
    finally:
        lab.llm = original
    assert next(e for e in events if e["type"] == "goal")["parser"] == "offline-regex"
    assert next(e for e in events if e["type"] == "recommendation")["found"]
