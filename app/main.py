"""FastAPI app: JSON endpoints + NDJSON streaming of the agent loop + the static UI."""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store as S
from .agent import HORIZON_DAYS, Lab, run_agent
from .sim import DAYS, PER_DAY

STATIC = Path(__file__).resolve().parent.parent / "static"
STATE: dict = {}
PACE_SECONDS = float(os.getenv("STREAM_PACE_SECONDS", "0.45"))  # UX pacing only; simulations take ~0.3 s each


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE["lab"] = Lab()
    yield


app = FastAPI(title="StoreLab", version="0.1.0", lifespan=lifespan)


class ExperimentRequest(BaseModel):
    objective: str
    budget: int | None = None
    max_congestion_pct: float | None = None


@app.get("/api/health")
def health():
    lab: Lab = STATE["lab"]
    return {"status": "ok", "planner": lab.llm.label, "gemini_live": lab.llm.live, "gemini_error": lab.llm.error}


@app.get("/api/store")
def store_info():
    return {
        "width": S.STORE_W, "height": S.STORE_H, "currency": S.CURRENCY,
        "zones": {k: {"name": v["name"], "rect": v["rect"], "kind": v["kind"], "capacity": v.get("capacity"),
                      "fixed": bool(v.get("fixed")), "restricted": bool(v.get("restricted"))} for k, v in S.ZONES.items()},
        "categories": {k: {"name": v["name"], "price": v["price"], "margin": v["margin"], "fixed": v["fixed_fixture"]}
                       for k, v in S.CATS.items()},
        "layout": S.BASE_LAYOUT, "costs": S.COSTS,
        "assumptions": {"visitors_per_day": PER_DAY, "simulated_days": DAYS, "horizon_days": HORIZON_DAYS,
                        "min_aisle_m": S.MIN_AISLE_M},
    }


@app.get("/api/analytics")
def analytics():
    lab: Lab = STATE["lab"]
    return {**lab.analytics, "calibration": lab.calibration}


@app.post("/api/experiments/stream")
def stream_experiments(req: ExperimentRequest):
    lab: Lab = STATE["lab"]
    overrides = {"budget": req.budget, "max_congestion_pct": req.max_congestion_pct}

    def gen():
        try:
            for ev in run_agent(lab, req.objective, overrides):
                if ev["type"] in ("simulating", "result") and PACE_SECONDS:
                    time.sleep(PACE_SECONDS)
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # surface agent failures to the UI instead of a dead stream
            yield json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
