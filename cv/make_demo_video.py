"""Render a synthetic top-down 'CCTV' clip of the demo store, plus ground truth for scoring the tracker.

    python cv/make_demo_video.py --out data/demo_store.avi --truth data/demo_truth.json

Real deployments feed real camera footage into video_to_journeys.py instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import history as H  # noqa: E402
from app import store as S  # noqa: E402

PX = 32  # pixels per metre
FPS = 10


def background() -> np.ndarray:
    img = np.full((int(S.STORE_H * PX), int(S.STORE_W * PX), 3), 235, np.uint8)
    for z in S.ZONES.values():
        x, y, w, h = [int(v * PX) for v in z["rect"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (205, 214, 218), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (150, 160, 168), 1)
    return img


def build_paths(n: int, seed: int):
    hist = H.generate(n_journeys=2000, n_days=10, seed=seed)
    rng = np.random.default_rng(seed)
    js = [j for j in hist["journeys"] if len(j["stops"]) >= 1][:n]
    paths = []
    for k, j in enumerate(js):
        wps = [("entrance", (1.0, 6.0), 0.0)]
        for z, dwell in j["stops"]:
            x, y, w, h = S.ZONES[z]["rect"]
            wps.append((z, (x + w * (0.25 + 0.5 * rng.random()), y + h * (0.3 + 0.4 * rng.random())), float(dwell)))
        wps.append(("checkout" if j["bought"] else "entrance", (18.5, 6.2) if j["bought"] else (1.0, 6.0), 0.0))
        paths.append({"start": 2.0 + k * 3.0, "waypoints": wps})
    return paths


def position_at(path: dict, t: float, speed: float = 1.3):
    """Position (m) at time t (s) or None if not on screen."""
    t -= path["start"]
    if t < 0:
        return None
    pos = np.array(path["waypoints"][0][1])
    for _name, p, dwell in path["waypoints"][1:]:
        p = np.array(p)
        d = float(np.linalg.norm(p - pos))
        travel = d / speed
        if t <= travel:
            return pos + (p - pos) * (t / travel if travel else 1)
        t -= travel
        pos = p
        if t <= dwell:
            return pos
        t -= dwell
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_store.avi")
    ap.add_argument("--truth", default="data/demo_truth.json")
    ap.add_argument("--shoppers", type=int, default=25)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    paths = build_paths(a.shoppers, a.seed)
    end = max(p["start"] + sum(np.linalg.norm(np.array(w2[1]) - np.array(w1[1])) / 1.3 + w2[2]
                               for w1, w2 in zip(p["waypoints"], p["waypoints"][1:])) for p in paths) + 2
    bg = background()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (bg.shape[1], bg.shape[0]))
    rng = np.random.default_rng(1)
    for f in range(int(end * FPS)):
        frame = bg.copy()
        for p in paths:
            pos = position_at(p, f / FPS)
            if pos is not None:
                cv2.circle(frame, (int(pos[0] * PX), int(pos[1] * PX)), 6, (60, 60, 200), -1)
        frame = np.clip(frame.astype(np.int16) + rng.integers(-3, 4, frame.shape), 0, 255).astype(np.uint8)  # sensor noise
        vw.write(frame)
    vw.release()
    truth = [{"zones": [w[0] for w in p["waypoints"] if w[0] not in ("entrance",)], "start": p["start"]} for p in paths]
    Path(a.truth).write_text(json.dumps(truth))
    print(f"wrote {a.out}: {int(end * FPS)} frames @ {FPS} fps, {len(paths)} shoppers; ground truth -> {a.truth}")


if __name__ == "__main__":
    main()
