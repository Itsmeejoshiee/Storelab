"""StoreLab Sense: top-down video -> anonymous journey events (journey_events schema).

    python cv/video_to_journeys.py data/demo_store.avi --out data/journey_events_from_video.csv --truth data/demo_truth.json

Pipeline: background subtraction -> blobs -> nearest-neighbour tracker -> floor coordinates -> zone dwell events.
Privacy by construction: only (random track id, time, x, y, zone) leave this script. No crops, no
embeddings, no faces are ever stored, and track ids are random and reset every run.

Scope note: this classical tracker is enough for a fixed top-down camera. For angled CCTV swap
`detect()` for a person detector (e.g. YOLO) and add a camera->floor homography in `to_floor()`.
"""
from __future__ import annotations

import argparse
import csv
import json
import secrets
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import store as S  # noqa: E402


def zone_of(x: float, y: float) -> str | None:
    for zid, z in S.ZONES.items():
        zx, zy, w, h = z["rect"]
        if zx <= x <= zx + w and zy <= y <= zy + h:
            return zid
    return None


class Tracker:
    def __init__(self, max_dist: float = 60, max_missing: int = 15):
        self.tracks: dict[int, dict] = {}
        self.done: list[dict] = []
        self.next = 0
        self.max_dist, self.max_missing = max_dist, max_missing

    def update(self, dets: list[tuple[float, float]], frame: int):
        unmatched = set(range(len(dets)))
        for tid, t in sorted(self.tracks.items()):
            best, bd = None, self.max_dist
            for i in unmatched:
                d = float(np.hypot(dets[i][0] - t["pos"][0], dets[i][1] - t["pos"][1]))
                if d < bd:
                    best, bd = i, d
            if best is not None:
                t["pos"] = dets[best]
                t["obs"].append((frame, *dets[best]))
                t["missing"] = 0
                unmatched.discard(best)
            else:
                t["missing"] += 1
        for i in unmatched:
            self.tracks[self.next] = {"pos": dets[i], "obs": [(frame, *dets[i])], "missing": 0}
            self.next += 1
        for tid in [k for k, t in self.tracks.items() if t["missing"] > self.max_missing]:
            self.done.append(self.tracks.pop(tid))

    def finish(self):
        self.done.extend(self.tracks.values())
        self.tracks = {}
        return [t for t in self.done if len(t["obs"]) >= 8]


def detect(mask: np.ndarray, min_area: int = 40) -> list[tuple[float, float]]:
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    return [tuple(cent[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def to_floor(px: float, py: float, px_per_m: float) -> tuple[float, float]:
    return px / px_per_m, py / px_per_m  # identity homography for a calibrated top-down view


def events_from_track(track: dict, fps: float, px_per_m: float, min_dwell: float):
    runs, cur = [], None
    for f, px, py in track["obs"]:
        x, y = to_floor(px, py, px_per_m)
        z = zone_of(x, y)
        if cur and cur["zone"] == z:
            cur["end"] = f
            cur["xs"].append(x)
            cur["ys"].append(y)
        else:
            cur = {"zone": z, "start": f, "end": f, "xs": [x], "ys": [y]}
            runs.append(cur)
    out = []
    for r in runs:
        dwell = (r["end"] - r["start"] + 1) / fps
        if r["zone"] and dwell >= min_dwell:
            out.append({"zone_id": r["zone"], "timestamp": round(r["start"] / fps, 1), "x": round(float(np.mean(r["xs"])), 2),
                        "y": round(float(np.mean(r["ys"])), 2), "event_type": "zone_visit", "dwell_seconds": round(dwell, 1)})
    return out


def score(journeys: list[list[str]], truth: list[dict]) -> dict:
    strip = lambda zs: [z for z in zs if z != "entrance"]
    pool = [strip(j) for j in journeys]
    exact = sum(strip(t["zones"]) in pool for t in truth)
    stops_t = sum(len(strip(t["zones"])) for t in truth)
    hit = 0
    for t in truth:
        best = max((len(set(strip(t["zones"])) & set(p)) for p in pool), default=0)
        hit += best
    return {"truth_journeys": len(truth), "recovered_tracks": len(journeys),
            "exact_zone_sequence_match": exact / max(len(truth), 1), "zone_stop_recall": hit / max(stops_t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="data/journey_events_from_video.csv")
    ap.add_argument("--store-id", default="demo_store_01")
    ap.add_argument("--px-per-m", type=float, default=32.0)
    ap.add_argument("--min-dwell", type=float, default=3.0, help="seconds in a zone before it counts as a stop")
    ap.add_argument("--truth", help="ground-truth json from make_demo_video.py (demo only)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    bgs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    trk, f = Tracker(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # learn the empty store for the first frames, then adapt very slowly so people who stand still stay foreground
        mask = bgs.apply(frame, learningRate=-1 if f < 15 else 0.0003)
        mask = cv2.morphologyEx(cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1], cv2.MORPH_OPEN, kernel)
        if f > 5:
            trk.update(detect(mask), f)
        f += 1
    tracks = trk.finish()

    rows, journeys = [], []
    for t in tracks:
        tid = "anon_" + secrets.token_hex(4)  # random, per-run, not linkable across runs
        evs = events_from_track(t, fps, a.px_per_m, a.min_dwell)
        if not evs:
            continue
        journeys.append([e["zone_id"] for e in evs])
        for e in evs:
            rows.append({"anonymous_track_id": tid, "store_id": a.store_id, **e})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["anonymous_track_id", "timestamp", "store_id", "zone_id", "x", "y", "event_type", "dwell_seconds"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: r[k] for k in cols} for r in rows])
    print(f"{f} frames -> {len(tracks)} tracks -> {len(journeys)} journeys, {len(rows)} zone events -> {a.out}")
    if a.truth:
        print(json.dumps(score(journeys, json.loads(Path(a.truth).read_text())), indent=2))


if __name__ == "__main__":
    main()
