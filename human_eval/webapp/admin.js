/* ===== HalluVision admin dashboard =====
 * Joins the API's submissions/audit log with the local assignments.json +
 * items.json to answer: who has labeled, how far, what they scored, when, and
 * lets an admin edit or delete any single label (every change is audited).
 * All four tables are sortable by clicking a column header.
 */
(function () {
  "use strict";

  const CFG = window.HALLUVISION_CONFIG || {};
  // ?api=<url> overrides config.js — lets you point at a local mock without
  // editing the deployed config. e.g. admin.html?api=http://localhost:8123
  const QP = new URLSearchParams(location.search);
  const API = (QP.has("api") ? QP.get("api") : CFG.API_BASE_URL || "").replace(/\/+$/, "");
  const ANNOTATORS = CFG.ANNOTATORS || [];
  const LABELS = ["faithful", "partial", "hallucinated"];
  const SEV = { faithful: 0, partial: 1, hallucinated: 2 };

  const state = {
    key: "",
    subs: [],
    events: [],
    items: [],
    assignments: null,
    openAnnotator: null,
    editing: null,
  };

  // Per-table sort state: which column, which direction.
  const SORT = {
    overview: { key: "name", dir: "asc" },
    detail: { key: "idx", dir: "asc" },
    items: { key: "idx", dir: "asc" },
    log: { key: "ts", dir: "desc" },
  };

  const el = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );

  // ---------- sorting ----------
  /** Sort rows by SORT[tableKey]; empty values always sink to the bottom. */
  function sortRows(rows, tableKey) {
    const { key, dir } = SORT[tableKey];
    const sign = dir === "asc" ? 1 : -1;
    return rows.slice().sort((x, y) => {
      const a = x._s[key];
      const b = y._s[key];
      const ea = a === null || a === undefined || a === "";
      const eb = b === null || b === undefined || b === "";
      if (ea && eb) return 0;
      if (ea) return 1;
      if (eb) return -1;
      const r = typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
      return r * sign;
    });
  }

  function applyIndicators(tableKey) {
    const tbl = document.querySelector(`.admin-table[data-table="${tableKey}"]`);
    if (!tbl) return;
    const { key, dir } = SORT[tableKey];
    tbl.querySelectorAll("th.sortable").forEach((th) => {
      th.classList.remove("sorted-asc", "sorted-desc");
      if (th.dataset.sort === key) th.classList.add(dir === "asc" ? "sorted-asc" : "sorted-desc");
    });
  }

  const RENDER = {}; // filled after the render fns are defined

  function wireSort() {
    document.querySelectorAll(".admin-table[data-table]").forEach((tbl) => {
      const tableKey = tbl.dataset.table;
      tbl.querySelectorAll("th.sortable").forEach((th) => {
        th.addEventListener("click", () => {
          const s = SORT[tableKey];
          if (s.key === th.dataset.sort) s.dir = s.dir === "asc" ? "desc" : "asc";
          else {
            s.key = th.dataset.sort;
            s.dir = "asc";
          }
          RENDER[tableKey]();
        });
      });
    });
  }

  // ---------- API ----------
  async function api(path, opts = {}) {
    const res = await fetch(`${API}${path}`, {
      ...opts,
      headers: { "content-type": "application/json", "x-admin-key": state.key, ...(opts.headers || {}) },
    });
    if (!res.ok) throw new Error(`${path} → ${res.status}`);
    return res.json();
  }

  async function loadAll() {
    const [all, log] = await Promise.all([api("/admin/all"), api("/admin/log?limit=300")]);
    state.subs = all.submissions || [];
    state.events = log.events || [];
    el("last-sync").textContent = "synced " + new Date().toLocaleTimeString();
  }

  async function loadLocal() {
    try {
      const r = await fetch("data/items.json", { cache: "no-store" });
      state.items = (await r.json()).items || [];
    } catch (_) {
      state.items = [];
    }
    try {
      const r = await fetch("data/assignments.json", { cache: "no-store" });
      if (r.ok) state.assignments = await r.json();
    } catch (_) {
      state.assignments = null;
    }
  }

  // ---------- helpers ----------
  function assignedCount(id) {
    const a = state.assignments;
    if (a && a.assign && a.assign[id]) return a.assign[id].length;
    return state.items.length || 100;
  }
  const subsOf = (id) => state.subs.filter((s) => s.annotator === id);
  function fmt(ts) {
    if (!ts) return "—";
    const d = new Date(ts);
    return isNaN(d) ? ts : d.toLocaleString();
  }
  function docExcerpt(doc_id) {
    const it = state.items.find((i) => i.doc_id === doc_id);
    return it ? (it.article || "").slice(0, 90) : "";
  }
  function allAnnotatorIds() {
    const ids = ANNOTATORS.map((a) => a.id);
    state.subs.forEach((s) => {
      if (!ids.includes(s.annotator)) ids.push(s.annotator);
    });
    return ids;
  }
  function displayName(id) {
    const cfg = ANNOTATORS.find((a) => a.id === id);
    if (cfg) return `${cfg.name} (${id.toUpperCase()})`;
    const s = subsOf(id)[0];
    return (s && s.name ? s.name : id) + " · ad-hoc";
  }
  const tile = (k, v, sub) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="sub">${sub}</div></div>`;

  // ---------- Overview ----------
  function renderOverview() {
    const ids = allAnnotatorIds();

    const totalLabels = state.subs.length;
    const started = ids.filter((id) => subsOf(id).length > 0).length;
    const done = ids.filter((id) => assignedCount(id) > 0 && subsOf(id).length >= assignedCount(id)).length;
    const docsCovered = new Set(state.subs.map((s) => s.doc_id)).size;
    el("stat-row").innerHTML = [
      tile("Annotators", `${started} / ${ids.length}`, "started"),
      tile("Completed", `${done} / ${ids.length}`, "finished their set"),
      tile("Total labels", String(totalLabels), "submitted"),
      tile("Items covered", `${docsCovered} / ${state.items.length || 100}`, "have ≥1 label"),
    ].join("");

    const rows = ids.map((id) => {
      const mine = subsOf(id);
      const assigned = assignedCount(id);
      const n = mine.length;
      const pct = assigned ? Math.round((n / assigned) * 100) : 0;
      const cls = n === 0 ? "notstarted" : n >= assigned ? "done" : "inprogress";
      const text = n === 0 ? "Not started" : n >= assigned ? "Complete" : "In progress";
      const rank = n === 0 ? 0 : n >= assigned ? 2 : 1;
      const first = mine.map((s) => s.created_at).filter(Boolean).sort()[0] || "";
      const last = mine.map((s) => s.updated_at).filter(Boolean).sort().pop() || "";
      const counts = { faithful: 0, partial: 0, hallucinated: 0 };
      mine.forEach((s) => counts[s.label] !== undefined && counts[s.label]++);

      return {
        _s: { name: displayName(id), status: rank, pct, labeled: n, assigned, first, last },
        html: `<tr class="clickable" data-annotator="${esc(id)}">
          <td class="nowrap"><b>${esc(displayName(id))}</b></td>
          <td class="nowrap"><span class="pill ${cls}">${text}</span></td>
          <td class="nowrap"><span class="mini-track"><span class="mini-fill" style="width:${pct}%"></span></span><span class="pct">${pct}%</span></td>
          <td class="num">${n}</td>
          <td class="num">${assigned}</td>
          <td><span class="dist-mini">${LABELS.map(
            (l) => `<span class="d-${l}" style="width:${(counts[l] / Math.max(1, n)) * 100}%" title="${l}: ${counts[l]}"></span>`
          ).join("")}</span></td>
          <td class="ts">${fmt(first)}</td>
          <td class="ts">${fmt(last)}</td>
          <td><button class="btn-mini" data-open="${esc(id)}">View</button></td>
        </tr>`,
      };
    });

    el("overview-body").innerHTML =
      sortRows(rows, "overview").map((r) => r.html).join("") ||
      `<tr><td colspan="9" class="empty">No annotators</td></tr>`;
    el("overview-body")
      .querySelectorAll("[data-annotator]")
      .forEach((tr) => tr.addEventListener("click", () => openDetail(tr.dataset.annotator)));
    applyIndicators("overview");
  }

  // ---------- Annotator detail ----------
  function openDetail(id) {
    state.openAnnotator = id;
    el("detail-panel").hidden = false;
    renderDetail();
    el("detail-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderDetail() {
    const id = state.openAnnotator;
    if (!id) return;
    // "#" = the order they actually labeled in (created_at ascending)
    const chrono = subsOf(id)
      .slice()
      .sort((a, b) => String(a.created_at || "").localeCompare(String(b.created_at || "")));
    el("detail-title").textContent = `${displayName(id)} — ${chrono.length} / ${assignedCount(id)} labeled`;

    const rows = chrono.map((s, i) => {
      // "Scored at" = when they first submitted it; a later admin/annotator edit
      // is surfaced separately so the original scoring time is never masked.
      const scored = s.created_at || s.updated_at || "";
      const edited = (s.revision || 1) > 1 && s.updated_at && s.updated_at !== s.created_at;
      return {
        _s: {
          idx: i + 1,
          doc_id: Number(s.doc_id),
          label: SEV[s.label],
          reasoning: s.reasoning || "",
          scored,
          revision: s.revision || 1,
        },
        html: `<tr>
        <td class="num">${i + 1}</td>
        <td class="mono">${esc(s.doc_id)}</td>
        <td><span class="pill ${esc(s.label)}">${esc(s.label)}</span></td>
        <td class="excerpt" title="${esc(s.reasoning)}">${esc(s.reasoning) || '<span class="muted">—</span>'}</td>
        <td class="ts">${fmt(scored)}${
          edited ? `<span class="edited" title="last edited ${esc(fmt(s.updated_at))}">edited</span>` : ""
        }</td>
        <td class="num">${s.revision || 1}</td>
        <td><div class="row-actions">
          <button class="btn-mini" data-edit="${esc(s.annotator)}|${esc(s.doc_id)}">Edit</button>
          <button class="btn-mini danger" data-del="${esc(s.annotator)}|${esc(s.doc_id)}">Delete</button>
        </div></td>
      </tr>`,
      };
    });

    el("detail-body").innerHTML =
      sortRows(rows, "detail").map((r) => r.html).join("") ||
      `<tr><td colspan="7" class="empty">No labels submitted yet</td></tr>`;
    wireRowActions(el("detail-body"));
    applyIndicators("detail");
  }

  function wireRowActions(root) {
    root.querySelectorAll("[data-edit]").forEach((b) =>
      b.addEventListener("click", (e) => {
        e.stopPropagation();
        const [ann, doc] = b.dataset.edit.split("|");
        openEdit(ann, doc);
      })
    );
    root.querySelectorAll("[data-del]").forEach((b) =>
      b.addEventListener("click", async (e) => {
        e.stopPropagation();
        const [ann, doc] = b.dataset.del.split("|");
        if (!confirm(`Delete ${ann}'s label for doc ${doc}?\nThis is recorded in the audit log.`)) return;
        await api("/admin/delete", { method: "POST", body: JSON.stringify({ annotator: ann, doc_id: doc }) });
        await refresh();
      })
    );
  }

  // ---------- By item ----------
  function renderItems() {
    const onlyDis = el("only-disagree").checked;
    const byDoc = new Map();
    state.subs.forEach((s) => {
      if (!byDoc.has(s.doc_id)) byDoc.set(s.doc_id, []);
      byDoc.get(s.doc_id).push(s);
    });

    const order = state.items.length ? state.items.map((i) => i.doc_id) : [...byDoc.keys()];
    const rows = [];
    order.forEach((doc_id, i) => {
      const list = byDoc.get(doc_id) || [];
      const labels = list.map((s) => s.label);
      const disagree = new Set(labels).size > 1;
      if (onlyDis && !disagree) return;

      const counts = {};
      labels.forEach((l) => (counts[l] = (counts[l] || 0) + 1));
      const top = Math.max(0, ...Object.values(counts));
      const consensus =
        Object.keys(counts)
          .filter((l) => counts[l] === top)
          .sort((a, b) => SEV[b] - SEV[a])[0] || null;
      const excerpt = docExcerpt(doc_id);

      rows.push({
        _s: {
          idx: i + 1,
          doc_id: Number(doc_id),
          excerpt,
          disagree: disagree ? 1 : 0,
          consensus: consensus ? SEV[consensus] : null,
          n: list.length,
        },
        html: `<tr class="${disagree ? "disagree" : ""}">
          <td class="num">${i + 1}</td>
          <td class="mono">${esc(doc_id)}</td>
          <td class="excerpt" title="${esc(excerpt)}">${esc(excerpt)}</td>
          <td>${
            list.length
              ? list
                  .map(
                    (s) =>
                      `<span class="pill ${esc(s.label)}" title="${esc(s.annotator)} · ${esc(
                        fmt(s.updated_at)
                      )}">${esc(s.annotator)}: ${esc(s.label)}</span>`
                  )
                  .join(" ")
              : '<span class="pill ghost">no labels</span>'
          }</td>
          <td>${consensus ? `<span class="pill ${esc(consensus)}">${esc(consensus)}</span>` : "—"}</td>
          <td class="num">${list.length}</td>
        </tr>`,
      });
    });

    el("items-body").innerHTML =
      sortRows(rows, "items").map((r) => r.html).join("") ||
      `<tr><td colspan="6" class="empty">Nothing to show</td></tr>`;
    applyIndicators("items");
  }

  // ---------- Log ----------
  function renderLog() {
    const rows = state.events.map((e) => {
      const chg =
        e.action === "admin_delete"
          ? `<span class="pill ${esc(e.before?.label)}">${esc(e.before?.label)}</span> → <span class="pill ghost">deleted</span>`
          : e.before
          ? `<span class="pill ${esc(e.before.label)}">${esc(e.before.label)}</span> → <span class="pill ${esc(
              e.after?.label
            )}">${esc(e.after?.label)}</span>`
          : `<span class="pill ${esc(e.after?.label)}">${esc(e.after?.label)}</span>`;
      return {
        _s: {
          ts: e.ts || "",
          action: e.action || "",
          actor: e.actor || "",
          target: `${e.target_annotator || ""} ${String(e.target_doc || "").padStart(6, "0")}`,
        },
        html: `<tr>
          <td class="ts">${fmt(e.ts)}</td>
          <td><span class="pill ${e.action.startsWith("admin") ? "hallucinated" : "inprogress"}">${esc(e.action)}</span></td>
          <td>${esc(e.actor)}</td>
          <td class="mono">${esc(e.target_annotator)} · doc ${esc(e.target_doc)}</td>
          <td>${chg}</td>
        </tr>`,
      };
    });

    el("log-body").innerHTML =
      sortRows(rows, "log").map((r) => r.html).join("") ||
      `<tr><td colspan="5" class="empty">No activity yet</td></tr>`;
    applyIndicators("log");
  }

  RENDER.overview = renderOverview;
  RENDER.detail = renderDetail;
  RENDER.items = renderItems;
  RENDER.log = renderLog;

  // ---------- Edit modal ----------
  function openEdit(annotator, doc_id) {
    const s = state.subs.find((x) => x.annotator === annotator && x.doc_id === doc_id);
    if (!s) return;
    state.editing = { annotator, doc_id, label: s.label, reasoning: s.reasoning || "" };
    el("edit-target").textContent = `${displayName(annotator)} · doc ${doc_id}`;
    el("edit-reasoning").value = s.reasoning || "";
    syncEditButtons();
    el("edit-modal").hidden = false;
  }
  function syncEditButtons() {
    el("edit-labels")
      .querySelectorAll(".lbtn")
      .forEach((b) => b.classList.toggle("selected", b.dataset.label === state.editing.label));
  }
  function closeEdit() {
    el("edit-modal").hidden = true;
    state.editing = null;
  }

  // ---------- boot / refresh ----------
  async function refresh() {
    await loadAll();
    renderOverview();
    renderItems();
    renderLog();
    if (state.openAnnotator) renderDetail();
  }

  function wire() {
    wireSort();

    el("gate-btn").addEventListener("click", async () => {
      state.key = el("admin-key").value.trim();
      if (!API) {
        el("gate-status").textContent = "config.js has no API_BASE_URL — deploy the stack first.";
        return;
      }
      el("gate-status").textContent = "Connecting…";
      try {
        await loadLocal();
        await refresh();
        document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
        el("screen-admin").classList.add("active");
        localStorage.setItem("halluvision:adminkey", state.key);
      } catch (err) {
        el("gate-status").textContent = "Failed: " + err.message + " (wrong key?)";
      }
    });
    el("admin-key").value = localStorage.getItem("halluvision:adminkey") || "";
    el("admin-key").addEventListener("keydown", (e) => e.key === "Enter" && el("gate-btn").click());

    el("refresh-btn").addEventListener("click", refresh);
    el("detail-close").addEventListener("click", () => {
      el("detail-panel").hidden = true;
      state.openAnnotator = null;
    });
    el("only-disagree").addEventListener("change", renderItems);

    document.querySelectorAll(".tab").forEach((t) =>
      t.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((x) => x.classList.remove("active"));
        t.classList.add("active");
        el("tab-" + t.dataset.tab).classList.add("active");
      })
    );

    el("edit-labels")
      .querySelectorAll(".lbtn")
      .forEach((b) =>
        b.addEventListener("click", () => {
          state.editing.label = b.dataset.label;
          syncEditButtons();
        })
      );
    el("edit-cancel").addEventListener("click", closeEdit);
    el("edit-save").addEventListener("click", async () => {
      const { annotator, doc_id, label } = state.editing;
      await api("/admin/update", {
        method: "POST",
        body: JSON.stringify({ annotator, doc_id, label, reasoning: el("edit-reasoning").value.trim() }),
      });
      closeEdit();
      await refresh();
    });
  }

  wire();
})();
