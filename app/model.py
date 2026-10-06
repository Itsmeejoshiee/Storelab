"""Learned behaviour model + descriptive analytics ("StoreLab Sense / Twin").

Everything the simulator knows is *fitted from journeys + POS rows*, never read from the
ground-truth generator:
  * shopper clusters        k-means over which zones a journey visited (latent "missions")
  * transition model        P(next zone | current zone, cluster)   (baseline-layout days)
  * stop model              P(end shopping | stops so far, cluster)
  * affinity                how much each cluster cares about each category
  * dwell model             log-dwell ~ a + b * (affinity x display strength)
  * purchase model          per-category logistic regression (BigQuery-ML equivalent)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from . import store as S

K = 5
BOUNCE = K  # cluster id for journeys that never touched a core aisle


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _kmeans(X: np.ndarray, k: int, seed: int = 0, restarts: int = 8):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        c = [X[rng.integers(len(X))]]
        for _i in range(1, k):
            d2 = np.min([((X - ci) ** 2).sum(1) for ci in c], axis=0)
            c.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
        C = np.array(c, dtype=float)
        for _it in range(40):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
            newC = np.array([X[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C)
    return best[1]


def _irls(X: np.ndarray, y: np.ndarray, ridge: float = 1e-2, iters: int = 30) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    R = np.eye(X.shape[1]) * ridge
    for _ in range(iters):
        p = _sigmoid(X @ beta)
        W = p * (1 - p) + 1e-9
        H = X.T @ (W[:, None] * X) + R
        g = X.T @ (y - p) - R @ beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-7:
            break
    return beta


FEATS = ["dwell10", "endcap", "signage", "register", "comp_before", "comp_adj"]


class Model:
    def __init__(self, history: dict):
        self.history = history
        self.days = {d["day"]: d for d in history["days"]}
        self.J = history["journeys"]
        self._cluster()
        self._fit_transitions()
        self._fit_affinity()
        self._fit_dwell()
        self._fit_purchase()

    # ------------------------------------------------------------------ clustering
    def _vec(self, j) -> np.ndarray:
        v = np.zeros(len(S.CORE_ZONES))
        for z, _ in j["stops"]:
            if z in S.CORE_ZONES:
                v[S.CORE_ZONES.index(z)] = 1
        return v

    def assign(self, vec: np.ndarray) -> int:
        if vec.sum() == 0:
            return BOUNCE
        return int(np.argmin(((self.centroids - vec) ** 2).sum(1)))

    def _cluster(self):
        X = np.array([self._vec(j) for j in self.J])
        nz = X[X.sum(1) > 0]
        self.centroids = _kmeans(nz, K, seed=1)
        for j, v in zip(self.J, X):
            j["cluster"] = self.assign(v)
        counts = Counter(j["cluster"] for j in self.J)
        self.cluster_share = [counts.get(i, 0) / len(self.J) for i in range(K + 1)]
        self.cluster_label = []
        for i in range(K):
            top = [S.CORE_ZONES[t] for t in np.argsort(-self.centroids[i])[:2] if self.centroids[i][t] > 0.25]
            self.cluster_label.append(" + ".join(S.ZONES[t]["name"].split(" ")[0] for t in top) or "Mixed")
        self.cluster_label.append("Walk-in / bounce")

    # ------------------------------------------------------------------ transitions
    def _clean(self):
        return [j for j in self.J if self.days[j["day"]]["clean"]]

    def _fit_transitions(self, alpha: float = 0.3):
        """Choice weights w(z | cur, cluster) fitted by maximum likelihood.

        A shopper picks the next stop among zones they have *not visited yet*, so raw
        frequencies are biased by the availability set. We fit the Plackett-Luce / conditional-logit
        weights with the MM algorithm instead (a few lines of numpy)."""
        nz = len(S.VISITABLE)
        zi = {z: i for i, z in enumerate(S.VISITABLE)}
        obs = defaultdict(list)  # (cluster, cur) -> [(chosen_idx, availability row)]
        end_n = defaultdict(lambda: [0, 0])  # (cluster, n) -> [ends, continues]
        for j in self._clean():
            c, cur, avail = j["cluster"], "entrance", np.ones(nz)
            for n, (z, _) in enumerate(j["stops"]):
                obs[(c, cur)].append((zi[z], avail.copy()))
                avail[zi[z]] = 0
                end_n[(c, n)][1] += 1
                cur = z
            n = len(j["stops"])
            if n < nz:
                end_n[(c, n)][0] += 1
        self.T = {c: {} for c in range(K + 1)}
        for c in range(K + 1):
            for cur in ["entrance"] + S.VISITABLE:
                rows = obs.get((c, cur), [])
                if not rows:
                    self.T[c][cur] = [1.0] * nz
                    continue
                ch = np.array([r[0] for r in rows])
                A = np.array([r[1] for r in rows])
                cnt = np.bincount(ch, minlength=nz) + alpha
                w = np.ones(nz)
                for _ in range(60):
                    denom = np.maximum(A @ w, 1e-12)
                    exposure = A.T @ (1.0 / denom)
                    w_new = np.where(exposure > 0, cnt / np.maximum(exposure, 1e-12), 1.0)
                    w = w_new / w_new.sum() * nz
                self.T[c][cur] = [float(x) for x in w]
        self.p_end = {c: [] for c in range(K + 1)}
        for c in range(K + 1):
            for n in range(nz + 1):
                e, k = end_n[(c, n)]
                self.p_end[c].append((e + 0.5) / (e + k + 1.0) if n < nz else 1.0)

    # ------------------------------------------------------------------ affinity
    def _fit_affinity(self):
        rate = np.zeros((K + 1, len(S.CAT_LIST)))
        cl_n = Counter()
        for j in self._clean():
            c = j["cluster"]
            visited = {z for z, _ in j["stops"]}
            for ci, cat in enumerate(S.CAT_LIST):
                if cat in visited:  # baseline: category cat lives in zone cat
                    cl_n[(c, ci)] += 1
                    rate[c, ci] += cat in j["bought"]
        for (c, ci), n in cl_n.items():
            rate[c, ci] /= n
        rate[BOUNCE] = 0
        self.aff = {c: {cat: float(rate[c, ci] / rate.max()) for ci, cat in enumerate(S.CAT_LIST)}
                    for c in range(K + 1)}

    def x_dwell(self, cluster: int, zone_displays) -> float:
        return max([self.aff[cluster][c] * S.DWELL_STRENGTH[t] for c, t in zone_displays] or [0.0])

    def _fit_dwell(self):
        xs, ys = [], []
        for j in self.J:
            zc = S.layout_index(self.days[j["day"]]["layout"])
            for z, d in j["stops"]:
                xs.append(self.x_dwell(j["cluster"], zc.get(z, [])))
                ys.append(math.log(d))
        xs, ys = np.array(xs), np.array(ys)
        A = np.stack([np.ones_like(xs), xs], 1)
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        self.dwell_a, self.dwell_b = float(coef[0]), float(coef[1])
        self.dwell_sigma = float((ys - A @ coef).std())

    # ------------------------------------------------------------------ purchase model
    # One pooled logistic regression: category x cluster intercepts, display-type effects shared
    # across categories (endcap / signage / register), complement effects per category.
    def _row(self, cat: str, cluster: int, f: dict) -> list[float]:
        ci = S.CAT_LIST.index(cat)
        base = [0.0] * ((K + 1) * len(S.CAT_LIST))
        base[ci * (K + 1) + cluster] = 1.0
        comp_b = [0.0] * len(S.CAT_LIST)
        comp_a = [0.0] * len(S.CAT_LIST)
        comp_b[ci], comp_a[ci] = float(f["comp_before"]), float(f["comp_adj"])
        return base + [float(f["endcap"]), float(f["signage"]), float(f["register"]), float(f["dwell10"])] + comp_b + comp_a

    def expected_dwell(self, cluster: int, zone_displays) -> float:
        """E[dwell] for a stop: deterministic in (cluster, what is displayed there) so the purchase
        model sees the *layout-driven* part of dwell, not noise correlated with unobserved interest."""
        return math.exp(self.dwell_a + self.dwell_b * self.x_dwell(cluster, zone_displays) + self.dwell_sigma ** 2 / 2)

    def _fit_purchase(self):
        X, y = [], []
        for j in self.J:
            zc = S.layout_index(self.days[j["day"]]["layout"])
            full = [(z, self.expected_dwell(j["cluster"], zc.get(z, []))) for z, _ in j["stops"]] + [("checkout", 4.0)]
            for cat in S.CAT_LIST:
                f = S.purchase_features(full, zc, cat)
                if f:
                    X.append(self._row(cat, j["cluster"], f))
                    y.append(float(cat in j["bought"]))
        X, y = np.array(X), np.array(y)
        self.beta = _irls(X, y, ridge=0.05)
        self.n_train = int(len(y))
        self._nb = (K + 1) * len(S.CAT_LIST)

    def p_buy(self, cat: str, cluster: int, f: dict) -> float:
        z = float(np.dot(self.beta, self._row(cat, cluster, f)))
        return 1.0 / (1.0 + math.exp(-z))

    def effects(self) -> dict:
        """Learned odds multipliers (shared display effects + per-category complement effects)."""
        b, nb, n = self.beta, self._nb, len(S.CAT_LIST)
        out = {"display": {"endcap": round(float(math.exp(b[nb])), 2),
                           "signage": round(float(math.exp(b[nb + 1])), 2),
                           "register": round(float(math.exp(b[nb + 2])), 2),
                           "per_10s_dwell": round(float(math.exp(b[nb + 3])), 2)}, "complement": {}}
        for i, cat in enumerate(S.CAT_LIST):
            if S.COMP[cat]:
                out["complement"][cat] = {
                    "after": S.COMP[cat],
                    "seen_before": round(float(math.exp(b[nb + 4 + i])), 2),
                    "seen_just_before": round(float(math.exp(b[nb + 4 + n + i])), 2)}
        return out

    # ------------------------------------------------------------------ analytics
    def analytics(self) -> dict:
        J, n = self.J, len(self.J)
        zstats = {}
        for z in S.VISITABLE:
            visitors = engaged = 0
            dwell_sum = 0.0
            purchases = 0
            cat = next((c for c, zz in [(d["category"], d["zone"]) for d in S.BASE_LAYOUT] if zz == z), None)
            for j in J:
                for zz, d in j["stops"]:
                    if zz == z:
                        visitors += 1
                        engaged += d >= 15
                        dwell_sum += d
                        purchases += bool(cat and cat in j["bought"])
                        break
            zstats[z] = {"zone": z, "name": S.ZONES[z]["name"], "category": cat, "visitors": visitors,
                         "visit_rate": visitors / n, "engaged": int(engaged),
                         "avg_dwell": dwell_sum / visitors if visitors else 0.0,
                         "purchases": int(purchases),
                         "conversion": purchases / visitors if visitors else 0.0}
        buyers = [j for j in J if j["bought"]]
        items = sum(len(j["bought"]) for j in J)
        revenue = sum(S.CATS[c]["price"] for j in J for c in j["bought"])
        # transitions (empirical, all journeys)
        flow: dict[str, Counter] = defaultdict(Counter)
        for j in J:
            cur = "entrance"
            for z, _ in j["stops"]:
                flow[cur][z] += 1
                cur = z
            flow[cur]["checkout" if j["bought"] else "exit"] += 1
        trans = {cur: {k: v / sum(c.values()) for k, v in c.items()} for cur, c in flow.items()}
        top_flows = sorted(({"from": a, "to": b, "p": p} for a, d in trans.items() for b, p in d.items()
                            if a != "entrance" or True), key=lambda r: -r["p"])
        hours = Counter(int(j["t0"] // 3600) for j in J)
        hourly = [{"hour": h, "journeys": hours.get(h, 0)} for h in range(7, 21)]
        # complementary-category lift (evidence for the diagnosis)
        pairs = []
        for cat in ("coffee", "snacks"):
            a = [0, 0]
            b = [0, 0]
            for j in J:
                zc = S.layout_index(self.days[j["day"]]["layout"])
                f = S.purchase_features(j["stops"] + [("checkout", 4.0)], zc, cat)
                if not f or f["register"]:
                    continue
                t = a if f["comp_before"] else b
                t[0] += cat in j["bought"]
                t[1] += 1
            if a[1] and b[1] and b[0]:
                pairs.append({"category": cat, "after": S.COMP[cat], "with_comp_rate": a[0] / a[1],
                              "without_comp_rate": b[0] / b[1], "lift": (a[0] / a[1]) / (b[0] / b[1]),
                              "n_with": a[1], "n_without": b[1]})
        clusters = []
        for i in range(K + 1):
            js = [j for j in J if j["cluster"] == i]
            clusters.append({"id": i, "label": self.cluster_label[i], "share": self.cluster_share[i],
                             "avg_stops": float(np.mean([len(j["stops"]) for j in js])) if js else 0,
                             "conversion": float(np.mean([bool(j["bought"]) for j in js])) if js else 0})
        clean_sample = [j for j in self._clean() if len(j["stops"]) >= 1][:80]
        return {
            "synthetic": True,
            "totals": {"journeys": n, "transactions": len(buyers), "conversion": len(buyers) / n,
                       "avg_basket_items": items / max(len(buyers), 1),
                       "avg_basket_value": revenue / max(len(buyers), 1), "revenue": revenue,
                       "days": len(self.days), "clean_days": sum(d["clean"] for d in self.days.values())},
            "zones": zstats, "transitions": trans, "top_flows": top_flows[:12], "hourly": hourly,
            "complement_lift": pairs, "clusters": clusters,
            "funnel": {z: {"passed": s["visitors"], "engaged": s["engaged"], "purchased": s["purchases"]}
                       for z, s in zstats.items() if s["category"]},
            "sample_paths": [{"id": j["id"], "stops": [[z, round(d, 1)] for z, d in j["stops"]]}
                             for j in clean_sample],
            "learned_effects": self.effects(),
        }
