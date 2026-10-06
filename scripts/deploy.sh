#!/usr/bin/env bash
# Deploy StoreLab to Cloud Run with Gemini on Vertex AI (no API key in the container).
# Usage: scripts/deploy.sh PROJECT_ID [REGION]
# NOT yet run against a real project from this repo's authoring environment — treat as a starting point.
set -euo pipefail
PROJECT="${1:?usage: deploy.sh PROJECT_ID [REGION]}"
REGION="${2:-asia-southeast1}"
gcloud config set project "$PROJECT"
gcloud services enable run.googleapis.com aiplatform.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy storelab --source . --region "$REGION" --allow-unauthenticated --memory 1Gi --cpu 1 \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=$REGION"
SA=$(gcloud run services describe storelab --region "$REGION" --format 'value(spec.template.spec.serviceAccountName)')
SA=${SA:-$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com}
gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA" --role roles/aiplatform.user >/dev/null
gcloud run services describe storelab --region "$REGION" --format 'value(status.url)'
