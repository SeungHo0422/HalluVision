"""
Keyboard-driven review GUI for the NER validation sheets (Reviewer 2 #2a).

Bundles the two CSVs produced by ``analyzer.ner_manual_sheet`` into one
self-contained HTML file — no server, no network, no dependencies. Open it in a
browser, judge each entity with the number keys, and export the filled CSVs
(plus a JSON snapshot) when done. The exported CSVs carry the exact columns
``analyzer.ner_manual_score`` expects, so scoring is unchanged.

Progress is written to ``localStorage`` on every keystroke and can also be saved
to / restored from a JSON file, so a closed tab does not cost the session.

Usage::

    python -m final_pipeline.analyzer.ner_review_app \
        --entities notes/csv/manual_ner_70/B_entities_precision.csv \
        --docs notes/csv/manual_ner_70/A_docs_recall.csv \
        --output notes/csv/manual_ner_70/review.html

Then::

    xdg-open notes/csv/manual_ner_70/review.html
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _read_csv(path: str) -> Tuple[List[str], List[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NER gold 수기 검증 — HalluVision R2 #2a</title>
<style>
  :root {
    --bg: #f6f7f9; --card: #fff; --fg: #1a1d21; --muted: #6b7280;
    --line: #e2e5ea; --accent: #2563eb; --ok: #15803d; --warn: #b45309;
    --bad: #b91c1c; --mark: #fef08a; --chip: #eef2f7;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14171c; --card: #1c2027; --fg: #e6e8eb; --muted: #9aa3af;
      --line: #2b313a; --accent: #60a5fa; --ok: #4ade80; --warn: #fbbf24;
      --bad: #f87171; --mark: #78591c; --chip: #262c35;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR", sans-serif;
  }
  header {
    position: sticky; top: 0; z-index: 10; background: var(--card);
    border-bottom: 1px solid var(--line); padding: 10px 18px;
  }
  .bar { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  h1 { font-size: 15px; margin: 0; font-weight: 650; }
  .tabs { display: flex; gap: 6px; }
  .tab {
    border: 1px solid var(--line); background: transparent; color: var(--fg);
    padding: 5px 12px; border-radius: 999px; cursor: pointer; font-size: 13px;
  }
  .tab.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  .spacer { flex: 1; }
  .btn {
    border: 1px solid var(--line); background: transparent; color: var(--fg);
    padding: 5px 11px; border-radius: 7px; cursor: pointer; font-size: 13px;
  }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .prog { height: 5px; background: var(--line); border-radius: 3px; margin-top: 9px; overflow: hidden; }
  .prog > i { display: block; height: 100%; background: var(--accent); width: 0; transition: width .18s; }
  .stats { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .stats b { color: var(--fg); font-weight: 600; }
  main { max-width: 900px; margin: 22px auto 90px; padding: 0 18px; }
  .card {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 12px; padding: 20px 22px; margin-bottom: 14px;
  }
  .meta { font-size: 12.5px; color: var(--muted); display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .chip {
    background: var(--chip); border-radius: 6px; padding: 2px 9px;
    font-size: 12px; color: var(--fg);
  }
  .entity {
    font: 600 27px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace;
    margin: 14px 0 12px; word-break: break-all;
  }
  .badges { display: flex; gap: 7px; flex-wrap: wrap; margin-bottom: 14px; }
  .badge {
    font-size: 11.5px; padding: 2px 9px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted);
  }
  .badge.warn { color: var(--warn); border-color: var(--warn); }
  .badge.bad { color: var(--bad); border-color: var(--bad); }
  .ctx {
    background: var(--chip); border-radius: 9px; padding: 13px 15px;
    font-size: 14px; margin-bottom: 16px; line-height: 1.75;
  }
  mark { background: var(--mark); color: inherit; border-radius: 3px; padding: 0 2px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
  .choice {
    flex: 1; min-width: 120px; border: 1.5px solid var(--line); background: transparent;
    color: var(--fg); padding: 11px 10px; border-radius: 9px; cursor: pointer;
    font-size: 14px; display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  .choice:hover { border-color: var(--accent); }
  .choice.on { border-color: var(--accent); background: var(--accent); color: #fff; }
  .choice.on kbd { background: rgba(255,255,255,.22); color: #fff; border-color: transparent; }
  .choice.sm { font-size: 13px; padding: 8px 9px; }
  kbd {
    font: 11px/1 ui-monospace, monospace; border: 1px solid var(--line);
    background: var(--chip); border-radius: 4px; padding: 3px 5px; color: var(--muted);
  }
  .sub { font-size: 12px; color: var(--muted); margin: 14px 0 6px; }
  .hint { font-size: 12.5px; color: var(--warn); margin: -2px 0 8px; }
  input[type=text] {
    width: 100%; padding: 8px 11px; border: 1px solid var(--line);
    border-radius: 8px; background: var(--bg); color: var(--fg); font-size: 13.5px;
  }
  input[type=text]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  details { margin-top: 14px; }
  summary { cursor: pointer; font-size: 13px; color: var(--muted); }
  .article {
    margin-top: 11px; max-height: 340px; overflow-y: auto; font-size: 13.5px;
    background: var(--chip); border-radius: 9px; padding: 13px 15px; line-height: 1.8;
  }
  .help {
    position: fixed; bottom: 0; left: 0; right: 0; background: var(--card);
    border-top: 1px solid var(--line); padding: 8px 18px; font-size: 12px;
    color: var(--muted); display: flex; gap: 16px; flex-wrap: wrap; justify-content: center;
  }
  .gold { display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0 12px; }
  .lab { font-size: 12px; color: var(--muted); width: 96px; flex-shrink: 0; padding-top: 8px; }
  .missrow { display: flex; gap: 10px; align-items: flex-start; margin-bottom: 8px; }
  .done { color: var(--ok); font-weight: 600; }
  .nav { display: flex; gap: 8px; margin-top: 16px; }
  .toast {
    position: fixed; bottom: 52px; left: 50%; transform: translateX(-50%);
    background: var(--accent); color: #fff; padding: 8px 16px; border-radius: 8px;
    font-size: 13px; opacity: 0; transition: opacity .2s; pointer-events: none;
  }
  .toast.on { opacity: 1; }
  .overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,.45); display: none;
    align-items: center; justify-content: center; z-index: 50;
  }
  .overlay.on { display: flex; }
  .modal {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 26px 30px; max-width: 480px; margin: 0 16px; text-align: center;
    font-size: 14.5px; line-height: 1.75;
  }
  .modal h2 { margin: 0 0 10px; font-size: 19px; }
  .modal .nav { justify-content: center; }
  @media (max-width: 640px) { .grid2 { grid-template-columns: 1fr; } .help { display: none; } }
</style>
</head>
<body>
<header>
  <div class="bar">
    <h1>NER gold 수기 검증</h1>
    <span class="chip" id="who" style="display:none"></span>
    <div class="tabs">
      <button class="tab on" data-mode="precision">1. 엔티티 판정</button>
      <button class="tab" data-mode="recall">2. 누락 엔티티</button>
    </div>
    <span class="spacer"></span>
    <button class="btn" id="btn-jump">미판정으로</button>
    <button class="btn" id="btn-csv">CSV 내보내기</button>
    <button class="btn" id="btn-json">JSON 저장</button>
    <button class="btn" id="btn-load">불러오기</button>
    <input type="file" id="file" accept=".json" hidden>
  </div>
  <div class="prog"><i id="prog"></i></div>
  <div class="stats" id="stats"></div>
</header>
<main id="view"></main>
<div class="help" id="help"></div>
<div class="toast" id="toast"></div>
<div class="overlay" id="doneovl">
  <div class="modal">
    <div id="ovl-body"></div>
    <div class="nav">
      <button class="choice" id="ovl-json">JSON 다시 저장</button>
      <button class="choice sm" id="ovl-close">닫기</button>
    </div>
  </div>
</div>
<script id="payload" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"];
const VERDICTS = [
  { key: '1', v: 'correct',   ko: '정확' },
  { key: '2', v: 'partial',   ko: '부분' },
  { key: '3', v: 'incorrect', ko: '오류' },
];
const ERRTYPES = [
  { key: 'q', e: 'span_boundary', ko: '스팬 잘림' },
  { key: 'w', e: 'wrong_label',   ko: '라벨 오류' },
  { key: 'e', e: 'not_an_entity', ko: '엔티티 아님' },
  { key: 'r', e: 'not_in_text',   ko: '원문에 없음' },
];
const STORE = 'halluvision_ner_review_v1';
/* 2차 검수자 배포 모드: 목표를 앞 20개 문서 분량으로 표시하고 recall 탭을
   숨긴다. κ가 엔티티 판정만 쓰므로 2차 검수자에게 recall은 없는 작업이다. */
const SECOND = __SECOND__;
/* 빌드 시 고정되는 검수자 이름. 화면에 표시되고, state에 실려 progress JSON과
   그 파일명에도 들어간다 (merge 스크립트는 ent/doc만 읽으므로 κ에 무해). */
const ANNOTATOR = __ANNOTATOR__;

let state = { ent: {}, doc: {}, i: 0, di: 0, mode: 'precision' };
try {
  const saved = JSON.parse(localStorage.getItem(STORE) || 'null');
  if (saved && saved.ent) state = Object.assign(state, saved);
} catch (err) { /* corrupt storage — start clean */ }
if (SECOND) state.mode = 'precision';
if (ANNOTATOR) {
  state.annotator = ANNOTATOR;
  const who = document.getElementById('who');
  who.textContent = '검수자: ' + ANNOTATOR;
  who.style.display = '';
}

const docById = {};
DATA.docs.forEach(d => { docById[d.doc_id] = d; });

let storageOK = true;
const save = () => {
  try { localStorage.setItem(STORE, JSON.stringify(state)); }
  catch (err) { storageOK = false; }
};
const esc = s => (s == null ? '' : String(s)).replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const eRec = id => (state.ent[id] = state.ent[id] || { verdict: '', error_type: '', correct_form: '', notes: '' });
const dRec = id => (state.doc[id] = state.doc[id] || { missed: {}, doc_done: '', doc_notes: '' });

/* Matches over the RAW text and escapes each piece separately, so an entity
   containing &, < or " still highlights and nothing lands inside a tag. */
function highlightAll(text, needles, minLen) {
  const uniq = [...new Set(needles)].filter(n => n && n.length >= (minLen || 1))
    .sort((a, b) => b.length - a.length);
  if (!uniq.length) return esc(text);
  const pat = uniq.map(n => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  const re = new RegExp('(' + pat + ')', 'gi');
  let out = '', last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m[0].length === 0) { re.lastIndex++; continue; }
    out += esc(text.slice(last, m.index)) + '<mark>' + esc(m[0]) + '</mark>';
    last = m.index + m[0].length;
  }
  return out + esc(text.slice(last));
}

const highlight = (text, needle) => highlightAll(text, [needle], 1);

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('on');
  clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove('on'), 1600);
}

/* ---------------------------------------------------------------- progress */

function stats() {
  const judged = DATA.entities.filter(e => (state.ent[e.row_id] || {}).verdict).length;
  const tally = { correct: 0, partial: 0, incorrect: 0 };
  DATA.entities.forEach(e => {
    const v = (state.ent[e.row_id] || {}).verdict;
    if (v) tally[v]++;
  });
  const done = DATA.docs.filter(d => (state.doc[d.doc_id] || {}).doc_done).length;
  const warn = storageOK ? '' : ' &nbsp;·&nbsp; <b style="color:var(--bad)">자동 저장 불가 — ' +
      '"JSON 저장"으로 수동 백업하세요</b>';
  if (SECOND) {
    document.getElementById('prog').style.width = Math.min(100, 100 * judged / SUB_COUNT) + '%';
    document.getElementById('stats').innerHTML =
      `엔티티 <b>${judged}</b> / <b>${SUB_COUNT}</b> (2차 검증 목표` +
      `${judged > SUB_COUNT ? `, 전체 ${DATA.entities.length}` : ''}) &nbsp;·&nbsp; ` +
      `정확 <b>${tally.correct}</b> · 부분 <b>${tally.partial}</b> · 오류 <b>${tally.incorrect}</b>` + warn;
    return;
  }
  const total = state.mode === 'precision' ? DATA.entities.length : DATA.docs.length;
  const cur = state.mode === 'precision' ? judged : done;
  document.getElementById('prog').style.width = (100 * cur / total) + '%';
  document.getElementById('stats').innerHTML =
    `엔티티 <b>${judged}</b> / ${DATA.entities.length} &nbsp;·&nbsp; ` +
    `정확 <b>${tally.correct}</b> · 부분 <b>${tally.partial}</b> · 오류 <b>${tally.incorrect}</b>` +
    ` &nbsp;·&nbsp; 문서 확인 <b>${done}</b> / ${DATA.docs.length}` + warn;
}

/* --------------------------------------------------------- precision view */

function renderPrecision() {
  const items = DATA.entities;
  state.i = Math.max(0, Math.min(state.i, items.length - 1));
  const it = items[state.i];
  const rec = eRec(it.row_id);
  const doc = docById[it.doc_id] || {};

  const flagCls = it.span_flag === 'ok' ? '' : (it.span_flag === 'absent' ? 'bad' : 'warn');
  const badges = [
    `<span class="badge ${flagCls}">${esc(it.span_flag)}</span>`,
    `<span class="badge">원문 ${esc(it.n_occurrences)}회</span>`,
    `<span class="badge">NER 목록 ${esc(it.n_in_ner_list)}회</span>`,
  ];
  if (it.also_labeled_as) badges.push(`<span class="badge warn">같은 표면형: ${esc(it.also_labeled_as)}</span>`);
  if (Number(it.weight) > 1) badges.push(`<span class="badge">weight ${esc(it.weight)}</span>`);

  document.getElementById('view').innerHTML = `
    <div class="card">
      <div class="meta">
        <span>#${state.i + 1} / ${items.length}</span>
        <span class="chip">${esc(it.label)}</span>
        <span>doc ${esc(it.doc_id)} · 문서 ${esc(it.review_order)}/${DATA.docs.length}</span>
      </div>
      <div class="entity">${esc(it.entity)}</div>
      <div class="badges">${badges.join('')}</div>
      <div class="ctx">${it.context ? highlight(it.context, it.entity) : '<i>원문에서 찾지 못함 — 전문을 확인하세요</i>'}</div>
      <div class="row">
        ${VERDICTS.map(v => `<button class="choice ${rec.verdict === v.v ? 'on' : ''}" data-v="${v.v}">
          <kbd>${v.key}</kbd> ${v.ko}</button>`).join('')}
      </div>
      <div class="row" id="errrow" ${rec.verdict && rec.verdict !== 'correct' ? '' : 'hidden'}>
        ${ERRTYPES.map(e => `<button class="choice sm ${rec.error_type === e.e ? 'on' : ''}" data-e="${e.e}">
          <kbd>${e.key}</kbd> ${e.ko}</button>`).join('')}
      </div>
      ${rec.verdict && rec.verdict !== 'correct' && !rec.error_type
        ? '<div class="hint">오류 유형을 골라야 다음으로 넘어갑니다</div>' : ''}
      <div class="grid2">
        <div>
          <div class="sub">올바른 형태 / 라벨 <kbd>c</kbd></div>
          <input type="text" id="cf" value="${esc(rec.correct_form)}" placeholder="예: UNC2165 또는 Malware">
        </div>
        <div>
          <div class="sub">메모 <kbd>n</kbd></div>
          <input type="text" id="nt" value="${esc(rec.notes)}" placeholder="판단 근거·애매한 점">
        </div>
      </div>
      <details id="art">
        <summary>원문 전문 보기 <kbd>f</kbd> — ${esc(doc.article_chars || '')}자</summary>
        <div class="article">${highlight(doc.article_text || '', it.entity)}</div>
      </details>
    </div>`;

  document.querySelectorAll('[data-v]').forEach(b =>
    b.onclick = () => setVerdict(b.dataset.v));
  document.querySelectorAll('[data-e]').forEach(b =>
    b.onclick = () => setErr(b.dataset.e));
  document.getElementById('cf').oninput = ev => { rec.correct_form = ev.target.value; save(); };
  document.getElementById('nt').oninput = ev => { rec.notes = ev.target.value; save(); };
  stats();
}

function setVerdict(v) {
  const it = DATA.entities[state.i];
  const rec = eRec(it.row_id);
  rec.verdict = rec.verdict === v ? '' : v;
  if (rec.verdict === 'correct' || !rec.verdict) rec.error_type = '';
  save();
  if (rec.verdict === 'correct') { next(); } else { renderPrecision(); }
  maybeComplete();
}

/* 2차 교차 검증(κ)에는 앞 20개 문서 분량이면 충분하다 (층화 round-robin이라
   앞부분만 해도 표본 대표성이 유지됨). 엔티티 행은 문서 단위로 연속이므로
   앞에서부터 20번째 문서까지의 doc_id 집합이 그 경계다. */
const SUBSET_N = 20;
const SUBSET_IDS = (() => {
  const ids = new Set();
  for (const e of DATA.entities) {
    if (ids.size >= SUBSET_N && !ids.has(e.doc_id)) break;
    ids.add(e.doc_id);
  }
  return ids;
})();
const SUB_COUNT = DATA.entities.filter(e => SUBSET_IDS.has(e.doc_id)).length;

function showDoneModal(bodyHtml, closeLabel) {
  document.getElementById('ovl-body').innerHTML = bodyHtml;
  document.getElementById('ovl-close').textContent = closeLabel;
  document.getElementById('doneovl').classList.add('on');
}

/* 두 지점에서 완료 안내를 띄우고 progress JSON을 자동 저장한다: 앞 20개 문서
   분량을 채웠을 때(여기까지면 충분), 그리고 204개 전부를 채웠을 때. 각 안내는
   한 번만 뜬다 (state.subset_notified / done_notified). 플래그는 내보내는
   JSON에 같이 실리지만 merge 스크립트는 ent/doc만 읽으므로 κ 산출에 무해. */
function maybeComplete() {
  const judged = DATA.entities.filter(e => (state.ent[e.row_id] || {}).verdict).length;
  if (judged === DATA.entities.length) {
    if (state.done_notified) return;
    state.done_notified = 'y';
    save();
    exportJSON();
    showDoneModal(
      '<h2>🎉 엔티티 판정을 모두 완료했습니다</h2>' +
      `<p>진행 상황 파일 <b>${PROG_NAME}</b>이 방금 자동으로 ` +
      '내려받아졌습니다. 다운로드가 보이지 않으면 아래 버튼을 눌러 다시 저장해 주세요.</p>' +
      '<p><b>받으신 JSON 파일을 송승호님께 보내시고, 평가를 완료했다고 알려주세요.</b></p>',
      '닫기');
    return;
  }
  if (state.done_notified || state.subset_notified) return;
  const sub = DATA.entities.filter(e => SUBSET_IDS.has(e.doc_id));
  if (sub.length && sub.every(e => (state.ent[e.row_id] || {}).verdict)) {
    state.subset_notified = 'y';
    save();
    exportJSON();
    showDoneModal(
      `<h2>✅ 문서 ${SUBSET_N}개 분량, 엔티티 ${sub.length}개 판정을 완료했습니다</h2>` +
      '<p>2차 교차 검증에는 <b>여기까지면 충분합니다.</b> 진행 상황 파일 ' +
      `<b>${PROG_NAME}</b>이 방금 자동으로 내려받아졌습니다.</p>` +
      '<p><b>받으신 JSON 파일을 송승호님께 보내시고, 평가를 완료했다고 알려주세요.</b> ' +
      '물론 여유가 되시면 계속 판정하셔도 됩니다. 끝까지 하시면 완료 시점에 ' +
      '다시 안내가 뜹니다.</p>',
      '계속 판정하기');
  }
}

/* 오류 유형은 부분/오류 판정에 딸린 값이다. 판정 없이 먼저 누르거나 '정확'에
   붙이는 것은 무시하고, 판정이 붙어 있을 때만 기록하고 다음으로 넘어간다. */
function setErr(e) {
  const it = DATA.entities[state.i];
  const rec = eRec(it.row_id);
  if (!rec.verdict) { toast('먼저 2(부분) 또는 3(오류)을 누르세요'); return; }
  if (rec.verdict === 'correct') { toast('정확 판정에는 오류 유형이 없습니다'); return; }
  rec.error_type = rec.error_type === e ? '' : e;
  save();
  renderPrecision();
  if (rec.error_type) next();
}

const next = () => { if (state.i < DATA.entities.length - 1) state.i++; save(); renderPrecision(); };
const prev = () => { if (state.i > 0) state.i--; save(); renderPrecision(); };

/* ------------------------------------------------------------ recall view */

function renderRecall() {
  const docs = DATA.docs;
  state.di = Math.max(0, Math.min(state.di, docs.length - 1));
  const d = docs[state.di];
  const rec = dRec(d.doc_id);
  const goldAll = LABELS.flatMap(l => (d['gold_' + l] || '').split('|').map(s => s.trim()).filter(Boolean));

  document.getElementById('view').innerHTML = `
    <div class="card">
      <div class="meta">
        <span>문서 ${state.di + 1} / ${docs.length}</span>
        <span class="chip">doc ${esc(d.doc_id)}</span>
        <span>summary P ${esc(d.summary_precision)} · QA P ${esc(d.qa_precision)}</span>
        <span>gold ${esc(d.n_gold_entities)}개</span>
        ${rec.doc_done ? '<span class="done">확인 완료</span>' : ''}
      </div>
      <div class="sub">NER이 뽑은 엔티티 — 여기 없는 것만 아래에 적습니다</div>
      <div class="gold">
        ${LABELS.map(l => (d['gold_' + l] || '')
          .split('|').map(s => s.trim()).filter(Boolean)
          .map(e => `<span class="chip">${esc(l[0])}: ${esc(e)}</span>`).join('')).join('') || '<i>없음</i>'}
      </div>
      <div class="article" id="artbox">${highlightAll(d.article_text || '', goldAll, 3)}</div>
      <div class="sub">누락 엔티티 — 여러 개면 | 로 구분</div>
      ${LABELS.map((l, idx) => `
        <div class="missrow">
          <span class="lab">${esc(l)}</span>
          <input type="text" data-miss="${esc(l)}" value="${esc(rec.missed[l] || '')}"
                 placeholder="NER이 놓친 ${esc(l)} 엔티티" ${idx === 0 ? 'id="firstmiss"' : ''}>
        </div>`).join('')}
      <div class="missrow">
        <span class="lab">메모</span>
        <input type="text" id="dnotes" value="${esc(rec.doc_notes)}" placeholder="문서 단위 메모">
      </div>
      <div class="nav">
        <button class="choice ${rec.doc_done ? 'on' : ''}" id="bdone"><kbd>d</kbd> 확인 완료 & 다음</button>
        <button class="choice sm" id="bprev"><kbd>←</kbd> 이전</button>
        <button class="choice sm" id="bnext"><kbd>→</kbd> 다음</button>
      </div>
    </div>`;

  document.querySelectorAll('[data-miss]').forEach(inp =>
    inp.oninput = ev => { rec.missed[inp.dataset.miss] = ev.target.value; save(); });
  document.getElementById('dnotes').oninput = ev => { rec.doc_notes = ev.target.value; save(); };
  document.getElementById('bdone').onclick = markDone;
  document.getElementById('bprev').onclick = () => { if (state.di > 0) state.di--; save(); renderRecall(); };
  document.getElementById('bnext').onclick = () => { if (state.di < docs.length - 1) state.di++; save(); renderRecall(); };
  stats();
}

function markDone() {
  const d = DATA.docs[state.di];
  const rec = dRec(d.doc_id);
  rec.doc_done = rec.doc_done ? '' : 'y';
  save();
  if (rec.doc_done && state.di < DATA.docs.length - 1) { state.di++; renderRecall(); }
  else renderRecall();
}

/* ------------------------------------------------------------------ export */

function toCSV(header, rows) {
  const cell = v => {
    const s = v == null ? '' : String(v);
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  return '﻿' + [header.join(',')]
    .concat(rows.map(r => header.map(h => cell(r[h])).join(','))).join('\r\n');
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type: type || 'text/csv;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

function exportCSV() {
  const ents = DATA.entities.map(e => {
    const r = state.ent[e.row_id] || {};
    return Object.assign({}, e, {
      verdict: r.verdict || '', error_type: r.error_type || '',
      correct_form: r.correct_form || '', notes: r.notes || '',
    });
  });
  const docs = DATA.docs.map(d => {
    const r = state.doc[d.doc_id] || { missed: {} };
    const out = Object.assign({}, d, { doc_done: r.doc_done || '', doc_notes: r.doc_notes || '' });
    LABELS.forEach(l => { out['missed_' + l] = (r.missed || {})[l] || ''; });
    return out;
  });
  download('B_entities_precision.csv', toCSV(DATA.entityHeader, ents));
  setTimeout(() => download('A_docs_recall.csv', toCSV(DATA.docHeader, docs)), 350);
  toast('CSV 2개를 내려받았습니다 — 시트 폴더에 덮어쓰세요');
}

const PROG_NAME = 'ner_review_progress' + (ANNOTATOR
  ? '_' + ANNOTATOR.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
  : '') + '.json';

function exportJSON() {
  download(PROG_NAME, JSON.stringify(state, null, 2), 'application/json');
  toast('진행 상황 JSON을 저장했습니다');
}

/* ------------------------------------------------------------------- shell */

function render() {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.dataset.mode === state.mode));
  document.getElementById('help').innerHTML = state.mode === 'precision'
    ? '<span><kbd>1</kbd> 정확 → 바로 다음</span>' +
      '<span><kbd>2</kbd><kbd>3</kbd> 부분·오류 → <kbd>q</kbd><kbd>w</kbd><kbd>e</kbd><kbd>r</kbd> 유형까지 골라야 다음</span>' +
      '<span><kbd>←</kbd><kbd>→</kbd> 이동</span><span><kbd>f</kbd> 전문</span><span><kbd>c</kbd> 형태 <kbd>n</kbd> 메모</span>' +
      '<span><kbd>u</kbd> 판정 취소</span><span><kbd>Esc</kbd> 입력 해제</span>'
    : '<span><kbd>d</kbd> 확인 완료 &amp; 다음</span><span><kbd>←</kbd><kbd>→</kbd> 문서 이동</span>' +
      '<span><kbd>m</kbd> 누락 입력</span><span><kbd>Esc</kbd> 입력 해제</span>';
  if (state.mode === 'precision') renderPrecision(); else renderRecall();
}

document.querySelectorAll('.tab').forEach(t =>
  t.onclick = () => { state.mode = t.dataset.mode; save(); render(); });
if (SECOND) document.querySelector('.tab[data-mode="recall"]').style.display = 'none';
document.getElementById('btn-csv').onclick = exportCSV;
document.getElementById('btn-json').onclick = exportJSON;
document.getElementById('ovl-json').onclick = exportJSON;
document.getElementById('ovl-close').onclick = () =>
  document.getElementById('doneovl').classList.remove('on');
document.getElementById('btn-load').onclick = () => document.getElementById('file').click();
document.getElementById('file').onchange = ev => {
  const f = ev.target.files[0];
  if (!f) return;
  const fr = new FileReader();
  fr.onload = () => {
    try {
      const loaded = JSON.parse(fr.result);
      if (!loaded.ent) throw new Error('bad file');
      state = Object.assign(state, loaded); save(); render();
      toast('진행 상황을 불러왔습니다');
      maybeComplete();
    } catch (err) { toast('불러오기 실패 — 형식이 다릅니다'); }
  };
  fr.readAsText(f);
};
document.getElementById('btn-jump').onclick = () => {
  if (state.mode === 'precision') {
    const idx = DATA.entities.findIndex(e => !(state.ent[e.row_id] || {}).verdict);
    if (idx < 0) { toast('모든 엔티티를 판정했습니다'); return; }
    state.i = idx;
  } else {
    const idx = DATA.docs.findIndex(d => !(state.doc[d.doc_id] || {}).doc_done);
    if (idx < 0) { toast('모든 문서를 확인했습니다'); return; }
    state.di = idx;
  }
  save(); render();
};

document.addEventListener('keydown', ev => {
  const ovl = document.getElementById('doneovl');
  if (ovl.classList.contains('on')) {
    if (ev.key === 'Escape' || ev.key === 'Enter') { ev.preventDefault(); ovl.classList.remove('on'); }
    return;
  }
  const active = document.activeElement || document.body;
  const typing = /^(INPUT|TEXTAREA)$/.test(active.tagName);
  if (ev.key === 'Escape') { active.blur(); return; }
  if (typing) {
    if (ev.key === 'Enter') { active.blur(); }
    return;
  }
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;

  if (state.mode === 'precision') {
    const v = VERDICTS.find(x => x.key === ev.key);
    if (v) { ev.preventDefault(); setVerdict(v.v); return; }
    const e = ERRTYPES.find(x => x.key === ev.key.toLowerCase());
    if (e) { ev.preventDefault(); setErr(e.e); return; }
    if (ev.key === 'ArrowRight' || ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); next(); }
    else if (ev.key === 'ArrowLeft') { ev.preventDefault(); prev(); }
    else if (ev.key === 'f') { ev.preventDefault(); const a = document.getElementById('art'); a.open = !a.open; }
    else if (ev.key === 'c') { ev.preventDefault(); document.getElementById('cf').focus(); }
    else if (ev.key === 'n') { ev.preventDefault(); document.getElementById('nt').focus(); }
    else if (ev.key === 'u') {
      ev.preventDefault();
      const rec = eRec(DATA.entities[state.i].row_id);
      rec.verdict = ''; rec.error_type = ''; save(); renderPrecision();
    }
  } else {
    if (ev.key === 'd') { ev.preventDefault(); markDone(); }
    else if (ev.key === 'ArrowRight') { ev.preventDefault(); if (state.di < DATA.docs.length - 1) state.di++; save(); renderRecall(); }
    else if (ev.key === 'ArrowLeft') { ev.preventDefault(); if (state.di > 0) state.di--; save(); renderRecall(); }
    else if (ev.key === 'm') { ev.preventDefault(); document.getElementById('firstmiss').focus(); }
  }
});

window.addEventListener('beforeunload', save);
render();
</script>
</body>
</html>
"""


#: 판정 입력 열 — --blind가 비우는 대상. row_id 등 식별 열은 건드리지 않으므로
#: 판정이 시작된 뒤에도 blind 사본 재생성은 안전하다 (시트 재생성과 다름).
ENT_JUDGMENT_COLS = ["verdict", "error_type", "correct_form", "notes", "verdict_b"]
DOC_JUDGMENT_COLS = ["missed_Malware", "missed_System", "missed_Indicator",
                     "missed_Vulnerability", "missed_Organization",
                     "doc_done", "doc_notes"]


def build_html(entities_csv: str, docs_csv: str, blind: bool = False,
               second_rater: bool = False, annotator: str = "") -> str:
    ent_header, ent_rows = _read_csv(entities_csv)
    doc_header, doc_rows = _read_csv(docs_csv)

    if blind:
        for row in ent_rows:
            for col in ENT_JUDGMENT_COLS:
                if col in row:
                    row[col] = ""
        for row in doc_rows:
            for col in DOC_JUDGMENT_COLS:
                if col in row:
                    row[col] = ""

    payload = {
        "entityHeader": ent_header,
        "docHeader": doc_header,
        "entities": ent_rows,
        "docs": doc_rows,
    }
    data = json.dumps(payload, ensure_ascii=False)
    # keep the JSON from terminating the host <script> block
    data = data.replace("</", "<\\/")
    return (HTML_TEMPLATE
            .replace("__DATA__", data)
            .replace("__SECOND__", "true" if second_rater else "false")
            .replace("__ANNOTATOR__", json.dumps(annotator or "", ensure_ascii=False)))


def main() -> None:
    p = argparse.ArgumentParser(description="Build the NER review GUI (R2 #2a)")
    p.add_argument("--entities", default="notes/csv/manual_ner_70/B_entities_precision.csv")
    p.add_argument("--docs", default="notes/csv/manual_ner_70/A_docs_recall.csv")
    p.add_argument("--output", default="notes/csv/manual_ner_70/review.html")
    p.add_argument("--blind", action="store_true",
                   help="blank all judgment columns so the page starts empty "
                        "(second annotator / deployment copy)")
    p.add_argument("--second-rater", action="store_true",
                   help="present the first-20-docs subset as the goal and hide "
                        "the recall tab (κ uses entity verdicts only)")
    p.add_argument("--annotator", default="",
                   help="annotator name baked into the page: shown in the "
                        "header and carried in the progress JSON and its filename")
    args = p.parse_args()

    html = build_html(args.entities, args.docs, blind=args.blind,
                      second_rater=args.second_rater, annotator=args.annotator)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[ner_review_app] {len(html) / 1024:.0f} KB → {out}")


if __name__ == "__main__":
    main()
