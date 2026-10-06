# HalluVision Human-Validation — Infra & Deploy Guide

Serverless backend + static site for the R2 #3 human validation task.

```
Browser (webapp/) ──POST /submit──▶ API Gateway ──▶ Lambda ──▶ S3 (source of truth)
                  ◀─GET /submissions─┘                     └──▶ DynamoDB (resume index)
```

- **S3 `DataBucket`** — one JSON object per `(annotator, doc_id)`; the durable record.
- **DynamoDB `LabelsTable`** — `PK=annotator, SK=doc_id`; single Query powers resume.
- **Lambda** — validates the label enum server-side (`faithful|partial|hallucinated`).
- **S3 `SiteBucket`** — hosts the static SPA (`webapp/`).

Cost: 2 evaluators × ~200 autosaves ≈ a few thousand requests → effectively $0 (Free Tier).

## Prerequisites

- AWS account + credentials configured (`aws configure`), region e.g. `ap-northeast-2`.
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- Python 3 (for the data-prep and merge scripts).

## Deploy

```bash
# 1. Build the blinded labeling payload (from the repo root, workspace prepared)
python analyzer/build_items.py
#    -> writes webapp/data/items.json (100 items, no gold/metrics leaked)

# 1b. (N>=3 annotators) assign items. Ids must match config.js ANNOTATORS.
#     DISTRIBUTED (recommended for many labelers): every item gets r labels,
#     spread evenly. 10 labelers, r=3 -> 30 items each, all 100 triple-rated,
#     agreement = Krippendorff's alpha.
python analyzer/assign_items.py \
    --annotators a,b,c,d,e,f,g,h,i,j --redundancy 3
#     ANCHOR (alt, N=3-4): 52 labeled by everyone (Fleiss base) + rest split.
# python analyzer/assign_items.py --annotators a,b,c,d --anchor 52
#    -> writes webapp/data/assignments.json (each annotator sees only their items)
#    Skip this step entirely to have everyone label all 100 (full overlap).

# 2. Build & deploy the backend + buckets
cd infra
sam build
sam deploy --guided
#    Accept defaults; note the stack Outputs:
#      ApiUrl, DataBucketName, LabelsTableName, SiteBucketName, SiteUrl

# ---------------------------------------------------------
# Key                 ApiUrl
# Description         Base URL ? paste into webapp/config.js as API_BASE_URL
# Value               https://<api-id>.execute-api.ap-northeast-2.amazonaws.com

# Key                 DataBucketName
# Description         S3 bucket holding label submissions (use with merge_human_labels.py)
# Value               halluvision-human-eval-<account-id>

# Key                 LabelsTableName
# Description         DynamoDB resume index (use with merge_human_labels.py)
# Value               halluvision-human-labels

# Key                 SiteBucketName
# Description         Static site bucket ? `aws s3 sync webapp/ s3://<this>`
# Value               halluvision-human-eval-site-<account-id>

# Key                 SiteUrl
# Description         Public URL evaluators open in their browser
# Value               http://halluvision-human-eval-site-<account-id>.s3-website.ap-northeast-2.amazonaws.com
# ---------------------------------------------------------

# 3. Wire the frontend to the API
#    Edit ../webapp/config.js -> API_BASE_URL = "<ApiUrl>"

# 4. Publish the static site
aws s3 sync ../webapp/ s3://halluvision-human-eval-site-<account-id>

# 5. Share <SiteUrl> with both evaluators.
```

### Lock down CORS (recommended after first deploy)

The default `AllowedOrigin` is `*`. Once you know the site URL, redeploy pinned:

```bash
sam deploy --parameter-overrides AllowedOrigin=http://<SiteBucketName>.s3-website.<region>.amazonaws.com
```

## Admin dashboard (`admin.html`)

Open `<SiteUrl>/admin.html` and enter the `AdminKey` you deployed with
(default `halluvision-admin`; it is sent as the `x-admin-key` header).

- **Overview** — per annotator: started / in progress / complete, progress bar,
  labeled vs assigned, label distribution, first + last activity. Click a row to
  see every label they gave (doc_id, label, reasoning, timestamp, revision).
- **By item** — all 100 docs with each rater's label side by side, the consensus
  (majority, severity tie-break), and a *only disagreements* filter.
- **Activity log** — full audit trail, newest first: who scored what, when, and
  every `before → after` change, including admin edits and deletes.
- **Edit / Delete** — change or remove any single label. Both actions are
  recorded in the audit log (`admin_update` / `admin_delete`) and bump the
  submission's `revision`.

Set a different key at deploy time:

```bash
sam deploy --parameter-overrides AdminKey=<your-key>
```

> This is a light gate to prevent accidental edits, **not** real authentication.

### Redeploy after adding the admin API

The admin routes and audit log are new backend resources, so redeploy and
re-upload the site:

```bash
cd infra && sam build && sam deploy
aws s3 sync ../webapp/ s3://halluvision-human-eval-site-<account-id>
```

## Local testing (no AWS)

- Frontend only: leave `webapp/config.js` `API_BASE_URL` empty and serve statically —
  labels persist to `localStorage` and can be exported as JSON on the done screen.

  ```bash
  cd webapp && python -m http.server 8080   # open http://localhost:8080
  ```

- Backend: `sam local start-api` (needs Docker) then point `API_BASE_URL` at
  `http://localhost:3000`.

## Collect results

After both evaluators finish (or at any checkpoint):

```bash
# from repo root — one label_<id> column per annotator
python analyzer/merge_human_labels.py \
  --table <LabelsTableName> --annotators a,b,c,d,e,f,g,h,i,j \
  --out notes/human_labels_2026-07-11.csv

# N >= 3 : Krippendorff's alpha (primary) + Fleiss (if a full anchor exists)
#          + pairwise Cohen + metric-vs-consensus
python analyzer/agreement_multi.py \
  --labels notes/human_labels_2026-07-11.csv \
  --output notes/human_validation.md

# N == 2 : the original 2-rater report (label_a / label_b)
# python analyzer/agreement.py \
#   --labels notes/human_labels_2026-07-11.csv --output notes/human_validation.md
```

Both reports skip items lacking valid labels, so a mid-progress export still
yields a valid (smaller-n) κ. Fleiss' κ uses only the **anchor** items (rated by
all N); pairwise Cohen κ uses each pair's common items; the metric-vs-human
correlation uses the per-item **consensus** (majority) label across all 100.

## Teardown

```bash
aws s3 rm s3://<SiteBucketName> --recursive
aws s3 rm s3://<DataBucketName> --recursive   # export first if you want to keep labels!
sam delete
```
