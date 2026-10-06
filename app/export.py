"""Write the demo dataset as CSVs in the BigQuery table schemas from the spec.

    python -m app.export            # -> data/*.csv
    scripts/load_bigquery.sh PROJECT DATASET
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import history as H
from . import store as S

OUT = Path(__file__).resolve().parent.parent / "data"


def _write(name: str, rows: list[dict]):
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows):,} rows")


def main():
    hist = H.generate()
    _write("stores.csv", [{"store_id": "demo_store_01", "region": "PH-Central Luzon", "currency": "PHP",
                           "timezone": "Asia/Manila", "floorplan_url": ""}])
    _write("zones.csv", [{"zone_id": k, "store_id": "demo_store_01", "name": v["name"], "category": "",
                          "polygon": json.dumps(v["rect"]), "movable": not v.get("fixed", False) and not v.get("restricted", False)}
                         for k, v in S.ZONES.items()])
    _write("products.csv", [{"sku": f"{c}_std", "category": c, "margin": v["margin"], "current_zone": c, "promotion": ""}
                            for c, v in S.CATS.items()])
    _write("journey_events.csv", H.journey_events(hist))
    _write("transactions.csv", H.transactions(hist))


if __name__ == "__main__":
    main()
