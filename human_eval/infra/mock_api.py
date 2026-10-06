"""
Local mock of the HalluVision Lambda API — for testing the labeling app and the
admin dashboard without deploying to AWS.

Mirrors handler.py exactly (same routes, same audit trail, same revision
counters) but keeps everything in memory. It also serves the static ``webapp/``
files, so one command gives you the whole app.

Usage::

    python infra/mock_api.py                 # seeded demo data, port 8123
    python infra/mock_api.py --empty         # no seed data (fresh start)
    python infra/mock_api.py --port 9000

Then open (note the ``?api=`` override, which points the page at this mock
instead of the deployed API in config.js):

    http://localhost:8123/admin.html?api=http://localhost:8123
    http://localhost:8123/index.html?api=http://localhost:8123

Admin key: ``halluvision-admin`` (same default as the SAM template).
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from urllib.parse import urlparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBAPP_DIR = os.path.join(REPO_ROOT, "webapp")

ADMIN_KEY = "halluvision-admin"
VALID_LABELS = {"faithful", "partial", "hallucinated"}

SUBS = {}  # (annotator, doc_id) -> record
EVENTS = []  # audit trail


def now():
    return datetime.now(timezone.utc).isoformat()


def audit(action, actor, target_annotator, target_doc, before, after, at=None):
    EVENTS.append(
        {
            "ts": at or now(),
            "id": uuid.uuid4().hex[:8],
            "action": action,
            "actor": actor,
            "target_annotator": target_annotator,
            "target_doc": target_doc,
            "before": before,
            "after": after,
        }
    )


def put_submission(annotator, doc_id, label, reasoning, client_ts, name, affil, at=None):
    """``at`` overrides the server clock — used by seed() to lay demo data out
    over realistic past sessions instead of stamping everything with now()."""
    key = (annotator, doc_id)
    existing = SUBS.get(key)
    stamp = at or now()
    rec = {
        "annotator": annotator,
        "doc_id": doc_id,
        "label": label,
        "reasoning": reasoning,
        "ts": client_ts or stamp,
        "name": name,
        "affil": affil,
        "created_at": existing["created_at"] if existing else stamp,
        "updated_at": stamp,
        "revision": (existing["revision"] + 1) if existing else 1,
    }
    SUBS[key] = rec
    return rec, existing


def seed():
    """Demo data exercising every status: complete / in progress / not started.

    Timestamps are laid out over realistic past sessions (not all "now"), so the
    time columns, the activity log ordering, and time-based sorting are actually
    meaningful in the demo.
    """
    path = os.path.join(WEBAPP_DIR, "data", "assignments.json")
    if not os.path.exists(path):
        print("[mock] no assignments.json — skipping seed")
        return
    asg = json.load(open(path))
    labels = ["faithful", "partial", "hallucinated"]

    # annotator -> (how many, when they started, seconds per item, session gap)
    #   a: finished, worked across two sittings 4 and 3 days ago
    #   b: still going, started 2 days ago
    #   c: only just began, an hour ago
    plan = {
        "a": (30, timedelta(days=4, hours=2), 95, 15),
        "b": (12, timedelta(days=2, hours=5), 150, 8),
        "c": (1, timedelta(hours=1), 60, 0),
    }
    base = datetime.now(timezone.utc)

    for ann, (n, start_ago, per_item, session_at) in plan.items():
        t = base - start_ago
        for k, doc in enumerate(asg["assign"].get(ann, [])[:n]):
            # a second sitting the next day partway through
            if session_at and k == session_at:
                t += timedelta(days=1) - timedelta(seconds=per_item * session_at)
            lbl = labels[(k + (0 if ann == "a" else 1)) % 3]
            stamp = t.isoformat()
            rec, _ = put_submission(
                ann, doc, lbl, f"seed reason {k}", stamp, ann.upper(), "", at=stamp
            )
            audit("create", ann, ann, doc, None,
                  {"label": lbl, "reasoning": rec["reasoning"]}, at=stamp)
            t += timedelta(seconds=per_item + (k % 7) * 11)  # jittered pacing

    # a couple of later revisions, so updated_at != created_at and revision > 1
    for ann, idx, new_label, ago in (("a", 3, "hallucinated", timedelta(days=1, hours=2)),
                                     ("b", 5, "faithful", timedelta(hours=6))):
        docs = asg["assign"].get(ann, [])
        if len(docs) > idx:
            doc = docs[idx]
            before = SUBS[(ann, doc)]
            stamp = (base - ago).isoformat()
            rec, _ = put_submission(ann, doc, new_label, before["reasoning"] + " (revised)",
                                    before["ts"], ann.upper(), "", at=stamp)
            audit("update", ann, ann, doc,
                  {"label": before["label"], "reasoning": before["reasoning"]},
                  {"label": new_label, "reasoning": rec["reasoning"]}, at=stamp)

    span = sorted(r["created_at"] for r in SUBS.values())
    print(f"[mock] seeded {len(SUBS)} submissions (a=30 complete, b=12, c=1)")
    print(f"[mock] scored between {span[0][:16]} and {span[-1][:16]} (UTC)")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEBAPP_DIR, **kw)

    # ---- helpers ----
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type,x-admin-key")
        self.send_header("access-control-allow-methods", "GET,POST,OPTIONS")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _is_admin(self):
        return self.headers.get("x-admin-key") == ADMIN_KEY

    def _body(self):
        n = int(self.headers.get("content-length", 0) or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def log_message(self, *a):
        pass  # quiet

    # ---- routes ----
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type,x-admin-key")
        self.send_header("access-control-allow-methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/submissions":
            from urllib.parse import parse_qs

            ann = (parse_qs(p.query).get("annotator") or [""])[0].lower()
            out = {
                d: {"label": r["label"], "reasoning": r["reasoning"], "ts": r["ts"]}
                for (a, d), r in SUBS.items()
                if a == ann
            }
            return self._json(200, out)
        if p.path == "/admin/all":
            if not self._is_admin():
                return self._json(401, {"error": "missing or invalid x-admin-key"})
            return self._json(200, {"submissions": list(SUBS.values())})
        if p.path == "/admin/log":
            if not self._is_admin():
                return self._json(401, {"error": "missing or invalid x-admin-key"})
            ev = sorted(EVENTS, key=lambda e: e["ts"], reverse=True)[:300]
            return self._json(200, {"events": ev})
        return super().do_GET()  # static files

    def do_POST(self):
        p = urlparse(self.path)
        b = self._body()

        if p.path == "/submit":
            ann = str(b.get("annotator", "")).lower()
            doc = str(b.get("doc_id", ""))
            label = str(b.get("label", "")).lower()
            if not ann or not doc:
                return self._json(400, {"error": "annotator and doc_id required"})
            if label not in VALID_LABELS:
                return self._json(400, {"error": "invalid label"})
            rec, existing = put_submission(
                ann, doc, label, str(b.get("reasoning", "")), str(b.get("ts", "")),
                str(b.get("name", "")), str(b.get("affil", "")),
            )
            audit(
                "update" if existing else "create", ann, ann, doc,
                {"label": existing["label"], "reasoning": existing["reasoning"]} if existing else None,
                {"label": label, "reasoning": rec["reasoning"]},
            )
            return self._json(200, {"ok": True, "revision": rec["revision"]})

        # admin mutations
        if not self._is_admin():
            return self._json(401, {"error": "missing or invalid x-admin-key"})
        key = (str(b.get("annotator", "")).lower(), str(b.get("doc_id", "")))

        if p.path == "/admin/update":
            existing = SUBS.get(key)
            if not existing:
                return self._json(404, {"error": "submission not found"})
            label = str(b.get("label", "")).lower()
            if label not in VALID_LABELS:
                return self._json(400, {"error": "invalid label"})
            before = {"label": existing["label"], "reasoning": existing["reasoning"]}
            rec, _ = put_submission(
                key[0], key[1], label, str(b.get("reasoning", "")),
                existing["ts"], existing["name"], existing["affil"],
            )
            audit("admin_update", "admin", key[0], key[1], before,
                  {"label": label, "reasoning": rec["reasoning"]})
            return self._json(200, {"ok": True})

        if p.path == "/admin/delete":
            existing = SUBS.pop(key, None)
            if not existing:
                return self._json(404, {"error": "submission not found"})
            audit("admin_delete", "admin", key[0], key[1],
                  {"label": existing["label"], "reasoning": existing["reasoning"]}, None)
            return self._json(200, {"ok": True})

        return self._json(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser(description="Local mock of the HalluVision API + static server")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--empty", action="store_true", help="start with no seed data")
    args = ap.parse_args()

    if not args.empty:
        seed()

    TCPServer.allow_reuse_address = True
    with TCPServer(("", args.port), Handler) as httpd:
        base = f"http://localhost:{args.port}"
        print(f"[mock] serving {WEBAPP_DIR}")
        print(f"[mock] admin key: {ADMIN_KEY}")
        print(f"[mock] admin  -> {base}/admin.html?api={base}")
        print(f"[mock] labeler -> {base}/index.html?api={base}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
