"""FastAPI app: read/manage API + enqueue jobs + interactive dashboard.

Deployed as a Vercel serverless function (fast endpoints only — the heavy
rank/optimize/track work runs in the separate worker, which the API triggers by
enqueuing a job). Shares the Postgres database with the worker.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import markdown as md_lib
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .. import db, report
from ..config import load_config
from ..db import Store

JobKind = Literal["rank", "optimize", "track"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    from dotenv import load_dotenv

    load_dotenv()  # local .env for CITEWORTHY_DATABASE_URL etc. (no-op if absent)
    app.state.config = load_config()
    store = db.connect()
    try:
        db.ensure_schema(store)
    finally:
        store.close()
    yield


app = FastAPI(title="Citeworthy", lifespan=lifespan)


def get_store():
    store = db.connect()
    try:
        yield store
    finally:
        store.close()


def _md_to_html(markdown_text: str) -> str:
    return md_lib.markdown(markdown_text, extensions=["tables", "fenced_code"])


# --- request models --------------------------------------------------------


class QueryIn(BaseModel):
    id: str
    text: str
    tier: Literal["money", "supporting"] = "money"
    our_url: str | None = None


class JobIn(BaseModel):
    kind: JobKind
    query_id: str | None = None


# --- queries ---------------------------------------------------------------


@app.get("/api/queries")
def list_queries(store: Store = Depends(get_store)):
    return [q.model_dump(mode="json") for q in db.list_queries(store)]


@app.post("/api/queries", status_code=201)
def add_query(body: QueryIn, store: Store = Depends(get_store)):
    from ..models import TargetQuery

    q = TargetQuery(id=body.id, text=body.text, tier=body.tier, our_url=body.our_url)
    db.upsert_query(store, q)
    return q.model_dump(mode="json")


@app.delete("/api/queries/{query_id}")
def remove_query(query_id: str, store: Store = Depends(get_store)):
    if not db.remove_query(store, query_id):
        raise HTTPException(404, f"no such query: {query_id}")
    return {"removed": query_id}


# --- jobs ------------------------------------------------------------------


@app.post("/api/jobs", status_code=201)
def create_job(body: JobIn, store: Store = Depends(get_store)):
    if body.kind in ("rank", "optimize"):
        if not body.query_id:
            raise HTTPException(400, f"{body.kind} requires a query_id")
        if db.get_query(store, body.query_id) is None:
            raise HTTPException(404, f"no such query: {body.query_id}")
    job_id = uuid.uuid4().hex[:12]
    db.enqueue_job(store, job_id, body.kind, body.query_id, datetime.now(timezone.utc))
    return db.get_job(store, job_id)


@app.get("/api/jobs")
def list_jobs(limit: int = 50, store: Store = Depends(get_store)):
    return db.list_jobs(store, limit=limit)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, store: Store = Depends(get_store)):
    job = db.get_job(store, job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id}")
    return job


# --- reports ---------------------------------------------------------------


@app.get("/api/report/{query_id}")
def query_report(query_id: str, store: Store = Depends(get_store)):
    our_domain = app.state.config.our_domain
    try:
        markdown = report.query_report_markdown(store, query_id, our_domain=our_domain)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return {"query_id": query_id, "markdown": markdown, "html": _md_to_html(markdown)}


@app.get("/api/portfolio")
def portfolio(store: Store = Depends(get_store)):
    from ..portfolio import build_portfolio

    config = app.state.config
    rows, correlations = build_portfolio(store, config)
    markdown = report.render_portfolio_report_md(rows, correlations, our_domain=config.our_domain)
    return {"markdown": markdown, "html": _md_to_html(markdown)}


@app.get("/api/costs")
def costs(store: Store = Depends(get_store)):
    rows = db.cost_summary(store)
    total = sum(r["est_usd"] or 0.0 for r in rows)
    return {"by_model": rows, "total_usd": total}


# --- dashboard -------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


DASHBOARD_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Citeworthy</title>
<style>
  :root { color-scheme: light dark; --bd:#8883; --accent:#3b7; }
  * { box-sizing: border-box; }
  body { font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 1.5rem; max-width: 1000px; margin: 0 auto; }
  h1 { margin: 0 0 .25rem; } h2 { margin: 2rem 0 .5rem; border-bottom: 1px solid var(--bd); padding-bottom: .25rem; }
  .sub { opacity: .7; margin: 0 0 1rem; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid var(--bd); vertical-align: top; }
  button { font: inherit; padding: .3rem .6rem; border: 1px solid var(--bd); border-radius: 6px; background: transparent; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  input, select { font: inherit; padding: .35rem; border: 1px solid var(--bd); border-radius: 6px; background: transparent; color: inherit; }
  form.inline { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; margin: .5rem 0 1rem; }
  .status { font-size: .8rem; padding: .1rem .5rem; border-radius: 999px; border: 1px solid var(--bd); }
  .status.done { color: var(--accent); border-color: var(--accent); }
  .status.error { color: #e55; border-color: #e55; }
  .status.running { color: #e90; border-color: #e90; }
  #report { border: 1px solid var(--bd); border-radius: 8px; padding: 1rem 1.25rem; margin-top: 1rem; display: none; }
  #report table { font-size: .9rem; } #report pre { overflow-x: auto; }
  .muted { opacity: .6; }
</style>
</head>
<body>
<h1>Citeworthy</h1>
<p class="sub">AI citation optimizer for fool.com.au — rank, optimize, and track citation share.</p>

<h2>Queries</h2>
<form class="inline" onsubmit="addQuery(event)">
  <input id="q-id" placeholder="slug (best-asx-dividend-shares)" required>
  <input id="q-text" placeholder="query text" required style="flex:1;min-width:180px">
  <select id="q-tier"><option value="money">money</option><option value="supporting">supporting</option></select>
  <input id="q-url" placeholder="our_url (optional)" style="flex:1;min-width:160px">
  <button class="primary" type="submit">Add</button>
</form>
<table id="queries"><tbody></tbody></table>
<p><button onclick="enqueue('track', null)">Run tracker (all queries)</button>
   <button onclick="loadPortfolio()">View portfolio</button>
   <button onclick="loadCosts()">Costs</button></p>

<h2>Jobs <span class="muted" id="jobs-note"></span></h2>
<table id="jobs"><thead><tr><th>Job</th><th>Kind</th><th>Query</th><th>Status</th><th>Report</th></tr></thead><tbody></tbody></table>

<div id="report"></div>

<script>
const api = (p, o) => fetch(p, o).then(r => r.ok ? r.json() : r.json().then(e => Promise.reject(e)));

async function loadQueries() {
  const qs = await api('/api/queries');
  const tb = document.querySelector('#queries tbody');
  tb.innerHTML = qs.length ? '' : '<tr><td class="muted">No queries yet.</td></tr>';
  for (const q of qs) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><strong>${q.id}</strong><br><span class="muted">${q.text}</span></td>
      <td>${q.tier}</td>
      <td>
        <button onclick="enqueue('rank','${q.id}')">Rank</button>
        <button onclick="enqueue('optimize','${q.id}')">Optimize</button>
        <button onclick="loadReport('${q.id}')">Report</button>
        <button onclick="delQuery('${q.id}')">✕</button>
      </td>`;
    tb.appendChild(tr);
  }
}
async function addQuery(e) {
  e.preventDefault();
  await api('/api/queries', {method:'POST', headers:{'content-type':'application/json'},
    body: JSON.stringify({id: qid.value, text: qtext.value, tier: qtier.value, our_url: qurl.value || null})})
    .catch(err => alert(err.detail || 'error'));
  e.target.reset(); loadQueries();
}
const qid=document.getElementById('q-id'), qtext=document.getElementById('q-text'),
      qtier=document.getElementById('q-tier'), qurl=document.getElementById('q-url');
async function delQuery(id) { if(!confirm('Remove '+id+'?')) return; await api('/api/queries/'+id,{method:'DELETE'}); loadQueries(); }
async function enqueue(kind, query_id) {
  await api('/api/jobs', {method:'POST', headers:{'content-type':'application/json'},
    body: JSON.stringify({kind, query_id})}).catch(err => alert(err.detail || 'error'));
  loadJobs();
}
async function loadJobs() {
  const jobs = await api('/api/jobs?limit=20');
  const tb = document.querySelector('#jobs tbody');
  tb.innerHTML = jobs.length ? '' : '<tr><td class="muted" colspan="5">No jobs yet.</td></tr>';
  let active = 0;
  for (const j of jobs) {
    if (j.status === 'queued' || j.status === 'running') active++;
    const tr = document.createElement('tr');
    const rep = j.report_query_id ? `<button onclick="loadReport('${j.report_query_id}')">view</button>`
      : (j.kind === 'track' ? `<button onclick="loadPortfolio()">portfolio</button>` : '');
    tr.innerHTML = `<td class="muted">${j.job_id}</td><td>${j.kind}</td><td>${j.query_id||'—'}</td>
      <td><span class="status ${j.status}">${j.status}</span>${j.error?'<br><span class="muted">'+j.error+'</span>':''}</td>
      <td>${j.status==='done'?rep:''}</td>`;
    tb.appendChild(tr);
  }
  document.getElementById('jobs-note').textContent = active ? `(${active} active — auto-refreshing)` : '';
}
async function loadReport(id) {
  const r = await api('/api/report/'+id).catch(e => ({html:'<p class="muted">'+(e.detail||'no report')+'</p>'}));
  show(r.html);
}
async function loadPortfolio() { const r = await api('/api/portfolio'); show(r.html); }
async function loadCosts() {
  const c = await api('/api/costs');
  const rows = c.by_model.map(m => `<tr><td>${m.model}</td><td>${m.calls}</td><td>$${(m.est_usd||0).toFixed(4)}</td></tr>`).join('');
  show(`<h3>Token spend</h3><table><thead><tr><th>Model</th><th>Calls</th><th>USD</th></tr></thead>
    <tbody>${rows||'<tr><td class="muted">No calls logged.</td></tr>'}</tbody></table>
    <p><strong>Total: $${c.total_usd.toFixed(4)}</strong></p>`);
}
function show(html) { const d = document.getElementById('report'); d.innerHTML = html; d.style.display='block'; d.scrollIntoView({behavior:'smooth'}); }

loadQueries(); loadJobs();
setInterval(loadJobs, 3000);
</script>
</body>
</html>
"""
