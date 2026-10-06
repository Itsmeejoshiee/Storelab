"""Gemini wrapper with an honest offline fallback.

* With GEMINI_API_KEY (AI Studio) or Vertex AI env vars, Gemini parses the objective (any language),
  words the diagnosis, and acts as the experiment planner in the agent loop.
* Without credentials, deterministic rule-based stand-ins are used so the whole product still runs.
  The API reports which mode is active; the UI shows it. Nothing pretends to be Gemini when it isn't.
"""
from __future__ import annotations

import json
import os
import re

from . import store as S

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class LLM:
    def __init__(self):
        self.client = None
        self.types = None
        self.error: str | None = None
        self.mode = "offline"
        try:
            from google import genai
            from google.genai import types
            self.types = types
            http = types.HttpOptions(timeout=45_000)
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
                                           http_options=http)
                self.mode = "gemini-api"
            elif os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true") and os.getenv("GOOGLE_CLOUD_PROJECT"):
                self.client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                                           location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
                                           http_options=http)
                self.mode = "vertex-ai"
        except Exception as e:  # SDK missing or bad credentials — stay offline
            self.error = f"{type(e).__name__}: {e}"

    @property
    def live(self) -> bool:
        return self.client is not None

    @property
    def label(self) -> str:
        return f"{GEMINI_MODEL} via {self.mode}" if self.live else "offline heuristic planner (no Gemini credentials)"

    def json(self, system: str, prompt: str, temperature: float = 0.4):
        """Ask Gemini for a JSON object. Returns None on any failure (caller falls back)."""
        if not self.live:
            return None
        try:
            resp = self.client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=self.types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                                        response_mime_type="application/json"))
            text = (resp.text or "").strip()
            text = re.sub(r"^```(?:json)?|```$", "", text).strip()
            return json.loads(text)
        except Exception as e:
            self.error = f"{type(e).__name__}: {str(e)[:200]}"
            return None


# ----------------------------------------------------------------------------- goal parsing
CAT_WORDS = {
    "coffee": ["coffee", "kape", "kopi", "コーヒー", "커피", "café", "cafe"],
    "bakery": ["bakery", "pastry", "pastries", "bread", "pandesal", "tinapay", "roti", "パン", "빵"],
    "snacks": ["snack", "chips", "chichirya", "camilan", "スナック", "과자", "meryenda"],
    "beverages": ["beverage", "drink", "soda", "inumin", "minuman", "飲み物", "음료"],
    "household": ["household", "cleaning", "detergent"],
}

GOAL_SYSTEM = """You turn a retail manager's request (any language) into a JSON goal for a store-layout experiment.
Return ONLY JSON with keys:
 target_category: one of coffee, bakery, snacks, beverages, household
 uplift_pct: number, desired % increase in that category's sales (default 10 if unstated)
 budget: integer amount in PHP, or null if unstated (interpret 30k as 30000)
 max_congestion_pct: number, max tolerated % increase in peak checkout load (default 5; "without worsening congestion" means 5)
 fixed_categories: list of categories the manager says must not be moved ("refrigeration" means beverages)
 language: ISO code of the request language
 summary: one short English sentence restating the goal"""


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_goal_offline(text: str) -> dict:
    t = text.lower()
    cat = next((c for c, words in CAT_WORDS.items() if any(w in t for w in words)), "snacks")
    uplift = 10.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", t):
        before = t[max(0, m.start() - 30):m.start()]
        if re.search(r"congest|queue|wait|checkout", before):
            continue
        uplift = float(m.group(1))
        break
    budget = None
    m = re.search(r"(?:₱|php|\bp\b|budget(?:\s+of)?(?:\s+is)?|up to|max(?:imum)?)\s*[:=]?\s*(?:₱|php|p)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?", t)
    if m:
        budget = _num(m.group(1))
        budget *= {"k": 1_000, "m": 1_000_000}.get(m.group(2) or "", 1)
        budget = int(budget)
    cong = 5.0
    m = re.search(r"(?:congest\w*|queue|wait\w*|checkout load)[^\d%]{0,40}?(\d+(?:\.\d+)?)\s*%", t)
    if m:
        cong = float(m.group(1))
    fixed = []
    if re.search(r"refrigerat|fridge|cooler", t):
        fixed.append("beverages")
    for c, words in CAT_WORDS.items():
        if re.search(r"(?:don'?t|do not|not|never|cannot|can'?t)\s+(?:move|relocate|touch)\s+(?:the\s+)?(?:" +
                     "|".join(map(re.escape, words)) + ")", t) and c not in fixed:
            fixed.append(c)
    return {"target_category": cat, "uplift_pct": uplift, "budget": budget, "max_congestion_pct": cong,
            "fixed_categories": fixed, "language": "en",
            "summary": f"Increase {S.CATS[cat]['name'].lower()} sales by ≥{uplift:g}%"
                       + (f" within a {S.CURRENCY}{budget:,} budget" if budget else "")
                       + f", keeping peak checkout load within +{cong:g}%."}


def parse_goal(llm: LLM, text: str, overrides: dict | None = None) -> tuple[dict, str]:
    """Returns (goal, parser) where parser is 'gemini' or 'offline-regex'."""
    goal, parser = None, "offline-regex"
    raw = llm.json(GOAL_SYSTEM, text, temperature=0.0)
    if isinstance(raw, dict):
        try:
            cat = raw["target_category"]
            if cat in S.CATS:
                goal = {
                    "target_category": cat,
                    "uplift_pct": float(raw.get("uplift_pct") or 10),
                    "budget": int(raw["budget"]) if raw.get("budget") not in (None, "") else None,
                    "max_congestion_pct": float(raw.get("max_congestion_pct") or 5),
                    "fixed_categories": [c for c in (raw.get("fixed_categories") or []) if c in S.CATS],
                    "language": str(raw.get("language") or "en"),
                    "summary": str(raw.get("summary") or text)[:240],
                }
                parser = "gemini"
        except Exception:
            goal = None
    if goal is None:
        goal = parse_goal_offline(text)
    for k, v in (overrides or {}).items():
        if v not in (None, ""):
            goal[k] = v
    goal["fixed_categories"] = sorted(set(goal.get("fixed_categories") or []) | {"beverages"})
    return goal, parser
