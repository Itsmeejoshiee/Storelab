# StoreLab — A/B testing for physical retail (MVP)

> Understand the store. Simulate the change. Test what matters.

StoreLab turns anonymous shopper journeys + POS sales into a **behavioural digital twin** of a store.
A manager types a goal ("Increase snack sales by 10%, budget ₱30k, don't worsen checkout congestion"); an agent
proposes layout/merchandising experiments, a **validator** enforces physical rules, a **simulator** estimates the
impact, and the agent revises until it has a ranked shortlist worth validating in a real store pilot.

```
Sense → Twin → Design → Simulate → Pilot → Learn
```

This repo implements **Sense → Generate → Simulate → Recommend** end to end, plus pilot planning.

---

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8080          # open http://localhost:8080
```

It works with **no credentials** (offline planner). To use Gemini:

```bash
export GEMINI_API_KEY=...                  # or Vertex AI: GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=... 
export GEMINI_MODEL=gemini-2.5-flash       # any Gemini model your project can call
```

The header badge shows which planner is active. Gemini then (1) parses the goal in any language,
(2) rewords the diagnosis as hypotheses, (3) acts as the planner/critic in the loop, (4) writes the explanation.
**Gemini never produces a number** — every figure comes from the simulator or the analytics.

```bash
pip install -r requirements-dev.txt && pytest          # 21 tests
python -m app.export                                   # CSVs in the BigQuery table schemas -> data/
python cv/make_demo_video.py && python cv/video_to_journeys.py data/demo_store.avi --truth data/demo_truth.json
scripts/deploy.sh YOUR_PROJECT asia-southeast1         # Cloud Run + Vertex AI
```

## The four screens

| Screen | What it shows |
|---|---|
| 1 Live Store | Floor plan heatmap (traffic / dwell / conversion), anonymous shopper dots, per-zone funnel, flows, journey clusters, simulator accuracy table |
| 2 Ask StoreLab | Plain-language goal, optional budget / congestion override |
| 3 AI Lab | Streamed agent loop: each candidate is validated, simulated, critiqued; constraint violations and losing ideas are visible; recommendation with explanation |
| 4 Experiment Plan | Expected impact with 90% intervals, cost, payback, 2-vs-2 store pilot sized from a power calculation, before/after shopper flow, downloadable brief |

## How it works

```
app/store.py      geometry, categories, layouts, change application, shared exposure features
app/history.py    SYNTHETIC world → 10,000 journeys + POS rows (hidden ground truth, never read by the model)
app/model.py      FITS behaviour from those rows: k-means journey clusters, Plackett-Luce transition
                  weights (MM algorithm), stop model, affinities, dwell model, pooled logistic purchase model
app/sim.py        Monte-Carlo shoppers under a layout; counterfactual re-weighting; single-register queue;
                  common random numbers → tight paired baseline-vs-candidate deltas (90% t-intervals)
app/validator.py  budget · fixed fixtures (refrigeration) · capacity · aisle width ≥ 1.2 m · egress · max 3 changes
app/agent.py      goal → diagnosis → propose → validate → simulate → critique → rank → recommend → pilot plan
app/llm.py        Gemini wrapper (google-genai) + regex goal parser fallback
app/main.py       FastAPI; NDJSON stream of agent events
cv/               video → anonymous tracks → journey_events (OpenCV)
```

Design rules from the spec that are enforced in code:
- The LLM only proposes *structured changes*; `validate()` decides legality and `simulate()` decides impact.
- Ranking = 90-day net margin (extra gross margin × 90 − cost) among candidates that meet the goal **and** the congestion limit.
- A candidate that breaks a constraint is never recommended; if nothing qualifies the app says so.

## What is real and what is not

**Real, running, tested:** the analytics, the fitted model, the simulator, the validator, the agent loop, the streaming API,
the UI, the video→journeys pipeline, the pilot power calculation, and the Gemini integration paths (tested with a stub client).

**Synthetic:** all shopper data. `history.py` generates 10,000 journeys / ~4,900 baskets from a hidden shopper world.
Every screen says so. The model is only as good as its inputs: swap in real journeys + POS and re-fit.

**Honest limitations**
- *Simulator accuracy:* on the demo data the fitted baseline reproduces history within ~8% mean absolute error (table on screen 1).
  Because the demo world and the simulator share the same structural assumptions (including the pull strength `PULL`),
  agreement here is **optimistic**; on real stores expect more error and recalibrate `PULL` and display effects from pilots
  (`physical_experiment_results` in the spec). The number to track is whether simulated rankings match pilot rankings.
- *Absolute lifts are not forecasts.* Use them to rank and screen ideas. The demo-world effects are generous.
- *Congestion* is a single-register FIFO queue; the constrained metric is peak-window register utilisation (linear in demand),
  and peak wait is shown alongside because it is non-linear and larger.
- *Video pipeline:* classical background subtraction on a fixed top-down view. On the bundled synthetic clip it recovers ~66% of stops
  and ~16% of exact zone sequences (ID switches, short stops). Real CCTV needs a person detector (e.g. YOLO) and a camera→floor homography
  (hooks marked in `video_to_journeys.py`). It is not connected to the live app yet: the app serves from the synthetic history.
- *Not built:* BigQuery/BQML in the serving path (CSV export + `load_bigquery.sh` provided), Firestore/Firebase Auth, Agent Builder,
  Pub/Sub, Vector Search, floor-plan upload / zone drawing, multi-store pilots feedback loop, multilingual UI (the *goal parser* is multilingual via Gemini).
- `Dockerfile` and `scripts/deploy.sh` were not executed in the authoring environment (no Docker daemon, no GCP project).
  The same commands (`pip install -r requirements.txt && uvicorn …`) were run and tested.
- The Gemini planner/parser/explainer has never been run against the live API from here — only against a stub that mimics its JSON.
  Failures degrade gracefully to the offline planner and are logged in the agent trace.
- The 0.45 s pause between streamed simulation results is UX pacing (`STREAM_PACE_SECONDS`); a run takes ~1.5 s without it.
- UI was exercised with jsdom against the live API (no JS errors, all flows reachable); it has not been viewed in a real browser from here.

## Mapping to the spec

| Spec capability | Status |
|---|---|
| 1 Store Mapper | Fixed demo floor plan in `store.py` (rect zones). Upload/draw UI not built |
| 2 Shopper Journey Intelligence | Done on synthetic data; video pipeline in `cv/` |
| 3 Retail Intelligence (funnels, POS join) | Done (zone funnel: passed → engaged → purchased) |
| 4 Spatial Diagnosis | Evidence computed deterministically; Gemini rewords as hypotheses |
| 5 Experiment Generator | Gemini JSON proposals (or offline templates) → validator |
| 6 Store Simulator | Done; Gemini is never the simulator |
| 7 AI Optimisation Loop | Done: propose → validate → simulate → critique → revise, up to 6 attempts |
| Pilot planning | Done: test/control stores, duration from power calc, success rule |
| Privacy | Anonymous random track ids; no faces/crops stored; only (id, t, x, y, zone) leave the CV script |

## Demo script (≈3 min)

1. **Live Store** — "High traffic, weak conversion": click Snacks; show the funnel and the beverage → snack lift.
2. **Ask StoreLab** — keep the default goal → *Generate experiments*.
3. **AI Lab** — A (register) breaks the congestion limit, B (relocate) loses sales, C is rejected for aisle width, then the winner; read the lesson boxes.
4. **Experiment Plan** — expected impact, payback, pilot design, before/after shopper flow; *Launch physical pilot*.
5. Example 2 (coffee, 8% congestion allowance) recommends cheap signage because the end-cap breaks the limit. Example 3 (Tagalog, 15% goal) hits the "no safe winner" path — it is parsed by the offline regex fallback here (kape → coffee) and by Gemini when configured.
