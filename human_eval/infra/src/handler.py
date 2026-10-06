"""
Lambda handler for the HalluVision human-validation API (R2 #3).

Annotator routes:
  POST /submit              {annotator, doc_id, label, reasoning, ts, name, affil}
                            -> validate, write S3 object (source of truth) + upsert
                               DynamoDB item (resume index) + append audit event.
  GET  /submissions?annotator=a
                            -> {doc_id: {label, reasoning, ts}} for that annotator.

Admin routes (require header ``x-admin-key``):
  GET  /admin/all           -> every submission (for the admin dashboard)
  GET  /admin/log?limit=N   -> audit trail, newest first
  POST /admin/update        {annotator, doc_id, label, reasoning} -> edit a label
  POST /admin/delete        {annotator, doc_id}                   -> remove a label

Audit trail: every create/update/delete (by annotator or admin) appends an event
to the ``_audit`` partition of the same DynamoDB table and to ``audit/`` in S3,
so we always know who scored what, when, and what changed.
"""

import json
import os
import uuid
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.conditions import Key

DATA_BUCKET = os.environ["DATA_BUCKET"]
LABELS_TABLE = os.environ["LABELS_TABLE"]
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")

VALID_LABELS = {"faithful", "partial", "hallucinated"}
AUDIT_PK = "_audit"  # reserved partition key for audit events

s3 = boto3.client("s3")
table = boto3.resource("dynamodb").Table(LABELS_TABLE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_annotator(a: str) -> bool:
    # The paper annotators are "a".."j"; ad-hoc testers come in as "other-<slug>".
    # No auth by design — just keep the reserved audit key out.
    return bool(a) and a != AUDIT_PK


def _resp(status, body):
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json",
            "access-control-allow-origin": ALLOWED_ORIGIN,
            "access-control-allow-headers": "content-type,x-admin-key",
            "access-control-allow-methods": "GET,POST,OPTIONS",
        },
        "body": json.dumps(body, default=str),
    }


def _body(event) -> dict:
    try:
        return json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return {}


def _is_admin(event) -> bool:
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    return bool(ADMIN_KEY) and headers.get("x-admin-key") == ADMIN_KEY


def _audit(action: str, actor: str, target_annotator: str, target_doc: str, before, after) -> None:
    ts = _now()
    item = {
        "annotator": AUDIT_PK,
        "doc_id": f"{ts}#{uuid.uuid4().hex[:8]}",
        "action": action,  # create | update | admin_update | admin_delete
        "actor": actor,  # annotator id, or "admin"
        "target_annotator": target_annotator,
        "target_doc": target_doc,
        "before": json.dumps(before) if before else "",
        "after": json.dumps(after) if after else "",
        "ts": ts,
    }
    table.put_item(Item=item)
    s3.put_object(
        Bucket=DATA_BUCKET,
        Key=f"audit/{ts}_{target_annotator}_{target_doc}.json",
        Body=json.dumps(item).encode("utf-8"),
        ContentType="application/json",
    )


def _get_existing(annotator: str, doc_id: str):
    r = table.get_item(Key={"annotator": annotator, "doc_id": doc_id})
    return r.get("Item")


def _put_submission(annotator, doc_id, label, reasoning, client_ts, name, affil, existing):
    now = _now()
    record = {
        "annotator": annotator,
        "doc_id": doc_id,
        "label": label,
        "reasoning": reasoning,
        "ts": client_ts or now,
        "name": name,
        "affil": affil,
        "created_at": (existing or {}).get("created_at", now),
        "updated_at": now,
        "revision": int((existing or {}).get("revision", 0)) + 1,
    }
    s3.put_object(
        Bucket=DATA_BUCKET,
        Key=f"submissions/{annotator}/{doc_id}.json",
        Body=json.dumps(record).encode("utf-8"),
        ContentType="application/json",
    )
    table.put_item(Item=record)
    return record


def lambda_handler(event, context):
    ctx = event.get("requestContext", {}).get("http", {})
    method = ctx.get("method", "")
    path = ctx.get("path", "")

    if method == "POST" and path.endswith("/submit"):
        return handle_submit(event)
    if method == "GET" and path.endswith("/submissions"):
        return handle_submissions(event)

    # ---- admin ----
    if path.endswith("/admin/all") and method == "GET":
        return _guard(event, handle_admin_all)
    if path.endswith("/admin/log") and method == "GET":
        return _guard(event, handle_admin_log)
    if path.endswith("/admin/update") and method == "POST":
        return _guard(event, handle_admin_update)
    if path.endswith("/admin/delete") and method == "POST":
        return _guard(event, handle_admin_delete)

    return _resp(404, {"error": "not found"})


def _guard(event, fn):
    if not _is_admin(event):
        return _resp(401, {"error": "missing or invalid x-admin-key"})
    return fn(event)


# ---------------- annotator ----------------
def handle_submit(event):
    b = _body(event)
    annotator = str(b.get("annotator", "")).strip().lower()
    doc_id = str(b.get("doc_id", "")).strip()
    label = str(b.get("label", "")).strip().lower()
    reasoning = str(b.get("reasoning", "") or "")
    client_ts = str(b.get("ts", "") or "")
    name = str(b.get("name", "") or "")
    affil = str(b.get("affil", "") or "")

    if not _valid_annotator(annotator):
        return _resp(400, {"error": "invalid annotator"})
    if not doc_id:
        return _resp(400, {"error": "doc_id required"})
    if label not in VALID_LABELS:
        return _resp(400, {"error": f"label must be one of {sorted(VALID_LABELS)}"})

    existing = _get_existing(annotator, doc_id)
    record = _put_submission(annotator, doc_id, label, reasoning, client_ts, name, affil, existing)
    _audit(
        "update" if existing else "create",
        annotator,
        annotator,
        doc_id,
        {"label": existing.get("label"), "reasoning": existing.get("reasoning")} if existing else None,
        {"label": label, "reasoning": reasoning},
    )
    return _resp(200, {"ok": True, "revision": record["revision"]})


def handle_submissions(event):
    params = event.get("queryStringParameters") or {}
    annotator = str(params.get("annotator", "")).strip().lower()
    if not _valid_annotator(annotator):
        return _resp(400, {"error": "invalid annotator"})

    out = {}
    resp = table.query(KeyConditionExpression=Key("annotator").eq(annotator))
    for item in resp.get("Items", []):
        out[item["doc_id"]] = {
            "label": item.get("label", ""),
            "reasoning": item.get("reasoning", ""),
            "ts": item.get("ts", ""),
        }
    return _resp(200, out)


# ---------------- admin ----------------
def handle_admin_all(event):
    """Every submission (excluding audit events). Small: <= N x 100 rows."""
    items = []
    kwargs = {}
    while True:
        r = table.scan(**kwargs)
        for it in r.get("Items", []):
            if it.get("annotator") == AUDIT_PK:
                continue
            items.append(
                {
                    "annotator": it.get("annotator"),
                    "doc_id": it.get("doc_id"),
                    "label": it.get("label", ""),
                    "reasoning": it.get("reasoning", ""),
                    "name": it.get("name", ""),
                    "affil": it.get("affil", ""),
                    "created_at": it.get("created_at", ""),
                    "updated_at": it.get("updated_at", it.get("ts", "")),
                    "revision": int(it.get("revision", 1)),
                }
            )
        if "LastEvaluatedKey" not in r:
            break
        kwargs["ExclusiveStartKey"] = r["LastEvaluatedKey"]
    return _resp(200, {"submissions": items})


def handle_admin_log(event):
    params = event.get("queryStringParameters") or {}
    try:
        limit = min(int(params.get("limit", 200)), 1000)
    except ValueError:
        limit = 200
    r = table.query(
        KeyConditionExpression=Key("annotator").eq(AUDIT_PK),
        ScanIndexForward=False,  # newest first (SK starts with the ISO timestamp)
        Limit=limit,
    )
    events = []
    for it in r.get("Items", []):
        events.append(
            {
                "ts": it.get("ts", ""),
                "action": it.get("action", ""),
                "actor": it.get("actor", ""),
                "target_annotator": it.get("target_annotator", ""),
                "target_doc": it.get("target_doc", ""),
                "before": json.loads(it["before"]) if it.get("before") else None,
                "after": json.loads(it["after"]) if it.get("after") else None,
            }
        )
    return _resp(200, {"events": events})


def handle_admin_update(event):
    b = _body(event)
    annotator = str(b.get("annotator", "")).strip().lower()
    doc_id = str(b.get("doc_id", "")).strip()
    label = str(b.get("label", "")).strip().lower()
    reasoning = str(b.get("reasoning", "") or "")

    if not _valid_annotator(annotator) or not doc_id:
        return _resp(400, {"error": "annotator and doc_id required"})
    if label not in VALID_LABELS:
        return _resp(400, {"error": f"label must be one of {sorted(VALID_LABELS)}"})

    existing = _get_existing(annotator, doc_id)
    if not existing:
        return _resp(404, {"error": "submission not found"})

    _put_submission(
        annotator, doc_id, label, reasoning,
        existing.get("ts", ""), existing.get("name", ""), existing.get("affil", ""), existing,
    )
    _audit(
        "admin_update", "admin", annotator, doc_id,
        {"label": existing.get("label"), "reasoning": existing.get("reasoning")},
        {"label": label, "reasoning": reasoning},
    )
    return _resp(200, {"ok": True})


def handle_admin_delete(event):
    b = _body(event)
    annotator = str(b.get("annotator", "")).strip().lower()
    doc_id = str(b.get("doc_id", "")).strip()
    if not _valid_annotator(annotator) or not doc_id:
        return _resp(400, {"error": "annotator and doc_id required"})

    existing = _get_existing(annotator, doc_id)
    if not existing:
        return _resp(404, {"error": "submission not found"})

    table.delete_item(Key={"annotator": annotator, "doc_id": doc_id})
    try:
        s3.delete_object(Bucket=DATA_BUCKET, Key=f"submissions/{annotator}/{doc_id}.json")
    except Exception:  # object may already be gone; the audit trail still records it
        pass
    _audit(
        "admin_delete", "admin", annotator, doc_id,
        {"label": existing.get("label"), "reasoning": existing.get("reasoning")},
        None,
    )
    return _resp(200, {"ok": True})
