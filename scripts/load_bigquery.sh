#!/usr/bin/env bash
# Load the demo CSVs into BigQuery tables named as in the StoreLab data model.
# Usage: python -m app.export && scripts/load_bigquery.sh PROJECT_ID DATASET
# Not wired into the running app yet (the MVP serves from memory); this is the path for the BigQuery/BQML version.
set -euo pipefail
PROJECT="${1:?project}"; DATASET="${2:-storelab}"
bq --project_id "$PROJECT" mk --dataset --location=asia-southeast1 "$DATASET" 2>/dev/null || true
for t in stores zones products journey_events transactions; do
  bq --project_id "$PROJECT" load --autodetect --replace --source_format=CSV "$DATASET.$t" "data/$t.csv"
done
