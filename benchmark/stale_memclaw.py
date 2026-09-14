#!/usr/bin/env python3
"""Run MemClaw against the STALE benchmark (implicit-conflict memory validity).

Per STALE sample: ingest its haystack_session into MemClaw (per-sample isolated,
strong mode so write-side contradiction/supersession fires), then for each of the
3 probing queries (dim1/dim2/dim3) recall from MemClaw and let an LLM answer using
ONLY the recalled memories. Emits STALE's answer-record schema:
  [{uid, target_model_responses: {dim1_response, dim2_response, dim3_response}}]
which STALE/Evaluation/full_eval_performance.py scores with the Gemini judge.

Answer LLM = OpenAI-compatible (default Gemini via base_url). Env:
  MEMCLAW_BENCH_URL, MEMCLAW_BENCH_TENANT   (memclaw)
  ANSWER_API_KEY, ANSWER_BASE_URL, ANSWER_MODEL   (answer generation)
Usage: MEMCLAW_BENCH_TENANT=<t> python3 stale_memclaw.py <memclaw_key> <data.json> <n> <out.json>
"""
import asyncio, os, sys, json, time
import runner_memclaw as R
from openai import AsyncOpenAI

URL = os.environ["MEMCLAW_BENCH_URL"]; TENANT = os.environ["MEMCLAW_BENCH_TENANT"]
MC_KEY = sys.argv[1]; DATA = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 5
OUT = sys.argv[4] if len(sys.argv) > 4 else "results/stale_answers.json"
TOP_K = int(os.environ.get("STALE_TOP_K", "20"))
# A57 (b1) recency-augmented recall: alongside the cosine top-K, force-include
# the K most-RECENT memories of the question's fleet (recency = haystack
# session index parsed from source_uri, the benchmark's own ordering). 0 = off.
RECENT_K = int(os.environ.get("STALE_RECENT_K", "0"))
# A57 (b1-oracle): force-include the memories created from M_new's session
# (the "NEW" rel). NOT a fix — the experimental CEILING: if dim1 still fails
# with M_new guaranteed in the prompt, the bottleneck is answer-LLM
# inference, not retrieval, and no retrieval fix can win. 0/1.
ORACLE_NEW = os.environ.get("STALE_ORACLE_NEW", "0") == "1"
# A57 (b1-qexp) query-expansion recall: one LLM call reformulates the probing
# query into N "what could have changed this?" search queries; each is
# recalled (k=QEXP_K) and unioned with the base recall. Targets the vocabulary
# gap: the update is phrased in different words than the question. 0 = off.
QEXP_N = int(os.environ.get("STALE_QEXP_N", "0"))
QEXP_K = int(os.environ.get("STALE_QEXP_K", "10"))
# A57 follow-on (dim2 finding): instruct the answer model to CHALLENGE a false
# premise instead of complying with it. dim2 was 0/15 in every retrieval
# config INCLUDING the oracle -- the model happily plans a Portland itinerary
# while holding evidence the user left Portland. Retrieval cannot fix that;
# this prompt line is the candidate lever. 0/1.
PREMISE_CHECK = os.environ.get("STALE_PREMISE_CHECK", "0") == "1"
SHOW_REL = os.environ.get("STALE_SHOW_RELATIONS", "0") == "1"  # (a): surface status/supersession in prompt

# --- Vertex AI auth: service-account -> OAuth token, auto-refreshed (tokens expire ~1h) ---
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "google/gemini-2.5-flash")
VERTEX_SA = os.environ["VERTEX_SA_FILE"]
# No default: this is a public repo and the GCP project id is infrastructure
# detail, not configuration a reader should inherit. Set VERTEX_PROJECT.
VERTEX_PROJECT = os.environ["VERTEX_PROJECT"]
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
if VERTEX_LOCATION == "global":
    VERTEX_BASE = f"https://aiplatform.googleapis.com/v1beta1/projects/{VERTEX_PROJECT}/locations/global/endpoints/openapi"
else:
    VERTEX_BASE = f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/endpoints/openapi"
from google.oauth2 import service_account
import google.auth.transport.requests as _gt
_creds = service_account.Credentials.from_service_account_file(
    VERTEX_SA, scopes=["https://www.googleapis.com/auth/cloud-platform"])
_client = None
_client_tok = None
def get_answer_client():
    global _client, _client_tok
    if not _creds.valid:
        _creds.refresh(_gt.Request())
    if _client is None or _client_tok != _creds.token:
        _client_tok = _creds.token
        _client = AsyncOpenAI(api_key=_creds.token, base_url=VERTEX_BASE)
    return _client

DIMS = ["dim1_query", "dim2_query", "dim3_query"]

# --- REAL memclaw_recall MCP tool over HTTP (vs REST /search) ---------------
import httpx as _httpx
USE_MCP = os.environ.get("STALE_USE_MCP_RECALL", "0") == "1"  # call the actual memclaw_recall tool
_MCP_URL = os.environ["MEMCLAW_BENCH_URL"].rstrip("/") + "/mcp/"
_mcp_http = _httpx.AsyncClient(verify=False, timeout=120)
_mcp_ready = False

def _mcp_headers(agent_id):
    return {"X-API-Key": MC_KEY, "X-Agent-ID": agent_id, "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"}

def _parse_rpc(resp):
    t = resp.text
    if "data:" in t[:64] or t.lstrip().startswith("event:"):
        for line in t.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
    return resp.json()

async def _mcp_init(agent_id):
    global _mcp_ready
    if _mcp_ready:
        return
    await _mcp_http.post(_MCP_URL, headers=_mcp_headers(agent_id), json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "stale-bench", "version": "1"}}})
    await _mcp_http.post(_MCP_URL, headers=_mcp_headers(agent_id),
                         json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
    _mcp_ready = True

async def mcp_recall(query, agent_id, fleet_ids, top_k):
    """Call the ACTUAL memclaw_recall MCP tool; return its results list (MemoryOut dicts)."""
    await _mcp_init(agent_id)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "memclaw_recall",
                       "arguments": {"query": query, "agent_id": agent_id,
                                     "fleet_ids": fleet_ids, "top_k": top_k}}}
    r = await _mcp_http.post(_MCP_URL, headers=_mcp_headers(agent_id), json=body)
    data = _parse_rpc(r)
    text = data.get("result", {}).get("content", [{}])[0].get("text", "{}")
    return json.loads(text).get("results", [])


def build_prompts(history_text, query_text, dim_key):
    # Mirrors STALE/Evaluation/run_target_model.py build_prompts, but history_text
    # is MemClaw's RECALLED memories rather than the full haystack.
    # (a): when relations are surfaced, tell the model to heed the markers.
    note = (" Some memories are annotated: [OUTDATED] / [SUPERSEDED] mean that fact is no longer "
            "true, and [CURRENT] marks a memory that updates an earlier one. Trust the CURRENT "
            "state and disregard outdated/superseded facts.") if SHOW_REL else ""
    if PREMISE_CHECK:
        note += (" Before responding, check whether the question or request rests on an "
                 "assumption about the user's current situation that the memories contradict "
                 "or no longer support (they may imply a change without stating it outright). "
                 "If so, say that the assumption appears outdated and answer for the user's "
                 "actual current situation instead of going along with the premise. "
                 "If the memories do answer the question, answer it — do not abstain merely "
                 "because a memory is older; flag only assumptions the memories actually "
                 "contradict or supersede.")
    if dim_key == "dim3_query":
        sysp = ("You are a helpful assistant. Review the following memory of your past "
                "conversations with the user, then respond to the user's latest query directly." + note)
        usr = f"[Recalled Memory]\n{history_text}\n\n[Latest Query]\n{query_text}"
    else:
        sysp = ("You are a helpful assistant. Review the following memory of your past "
                "conversations with the user, then accurately answer the question." + note)
        usr = f"[Recalled Memory]\n{history_text}\n\n[Question]\n{query_text}"
    return sysp, usr


def format_recall(memories):
    lines = []
    for m in memories:
        ts = m.get("ts_valid_start") or (m.get("metadata") or {}).get("session_date") or ""
        t = f" [{ts[:10]}]" if ts else ""
        tag = ""
        if SHOW_REL:  # (a): annotate status/supersession relations MemClaw computed
            st = m.get("status") or "active"
            marks = []
            if st in ("outdated", "conflicted"):
                marks.append(f"[{st.upper()}]")
            if m.get("supersedes_id"):
                marks.append("[CURRENT]")
            if m.get("superseded_by"):
                marks.append("[SUPERSEDED]")
            if marks:
                tag = " " + " ".join(marks)
        lines.append(f"-{t}{tag} {m.get('content', '')}")
    return "\n".join(lines) if lines else "(no relevant memory found)"


async def answer(sysp, usr):
    for attempt in range(3):
        try:
            r = await get_answer_client().chat.completions.create(
                model=ANSWER_MODEL,
                messages=[{"role": "system", "content": sysp}, {"role": "user", "content": usr}],
                temperature=0.0,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == 2:
                return f"[ANSWER_ERROR: {e}]"
            await asyncio.sleep(2 * (attempt + 1))


import csv as _csv

MODE = os.environ.get("STALE_MODE", "both")           # seed | run | both
TAG = os.environ.get("STALE_TAG", "run")              # names the retrieved/ + csv per config
RESULTS_DIR = os.path.dirname(OUT) or "results"
INPUTS_DIR = os.path.join(RESULTS_DIR, "stale_inputs")           # written ONCE (writes don't change)
RETR_DIR = os.path.join(RESULTS_DIR, "stale_retrieved", TAG)     # per run/config
CSV_PATH = os.path.join(RESULTS_DIR, f"stale_index_{TAG}.csv")


def _tagger(rec):
    io, inw = (rec.get("relevant_session_index") or [None, None])[:2]
    ot, nt = f"-s{io}/", f"-s{inw}/"
    def which(su):
        su = su or ""
        return "OLD" if ot in su else ("NEW" if nt in su else "")
    return which, io, inw


async def seed_one(mc, rec):
    """Ingest one question's haystack into a PERSISTENT per-question fleet (no cleanup) and
    write its inputs .md ONCE (writes don't change across runs)."""
    uid = rec["uid"]; fleet, agent = f"stale-{uid}"[:60], f"staleag-{uid}"[:60]
    which, io, inw = _tagger(rec)
    sessions = rec["haystack_session"]; ids = [f"{uid}-s{i}" for i in range(len(sessions))]
    ing = await R.ingest_test_case(mc, TENANT, sessions, ids, rec.get("timestamps", []),
                                   fleet_override=fleet, agent_override=agent)
    # inputs .md — write once
    os.makedirs(INPUTS_DIR, exist_ok=True)
    path = os.path.join(INPUTS_DIR, f"{uid[:8]}.md")
    if not os.path.exists(path):
        allm = [m for m in await mc.list_memories(TENANT) if m.get("fleet_id") == fleet]
        rel = [m for m in allm if which(m.get("source_uri")) in ("OLD", "NEW")]
        L = [f"# INPUTS — {uid}  (type {rec.get('type')})", "",
             f"**M_old** (session {io}): {rec.get('M_old')}",
             f"**M_new** (session {inw}): {rec.get('M_new')}",
             f"**why:** {rec.get('explanation')}", "",
             "## Probing queries",
             f"- dim1: {rec['probing_queries']['dim1_query']}",
             f"- dim2: {rec['probing_queries']['dim2_query']}",
             f"- dim3: {rec['probing_queries']['dim3_query']}", "",
             f"## Written memories: {len(allm)} total; {len(rel)} from the M_old/M_new sessions (write-side state)", ""]
        for m in rel:
            L.append(f"- [{which(m.get('source_uri'))}] status=`{m.get('status')}` "
                     f"supersedes=`{(m.get('supersedes_id') or '-')}` "
                     f"superseded_by=`{(m.get('superseded_by') or '-')}`  \"{(m.get('title') or '')[:60]}\"")
            L.append(f"    {(m.get('content') or '')[:200]}")
        open(path, "w").write("\n".join(L))
    sup = 0
    allm = [m for m in await mc.list_memories(TENANT) if m.get("fleet_id") == fleet]
    for m in allm:
        if which(m.get("source_uri")) in ("OLD", "NEW") and (m.get("supersedes_id") or m.get("status") in ("outdated", "conflicted")):
            sup += 1
    return {"created": ing["total_created"], "superseded": sup}


async def expand_query(q):
    """One LLM call -> up to QEXP_N alternative search queries phrased to find
    facts that could have UPDATED or contradicted the answer."""
    sysp = ("You write search queries over a personal-memory store. Given a question, "
            "produce alternative queries that would find memories describing a CHANGE "
            "in the user's situation relevant to the question — later facts, moves, "
            "switches, new circumstances — which may use completely different words "
            "than the question. One query per line, no numbering, no commentary.")
    usr = f"Question: {q}\nWrite {QEXP_N} alternative search queries."
    out = await answer(sysp, usr)
    if out.startswith("[ANSWER_ERROR"):
        return []
    return [ln.strip("-• ").strip() for ln in out.splitlines() if ln.strip()][:QEXP_N]


def _session_idx(m):
    import re as _re
    mm = _re.search(r"-s(\d+)/", m.get("source_uri") or "")
    return int(mm.group(1)) if mm else -1


async def eval_one(mc, rec, by_fleet=None):
    """Recall (real memclaw_recall, k=20) + answer per dim on the PERSISTENT seed. Writes the
    per-run retrieved .md and returns the answer record + a CSV row."""
    uid = rec["uid"]; fleet, agent = f"stale-{uid}"[:60], f"staleag-{uid}"[:60]
    which, io, inw = _tagger(rec)
    responses = {}; per = {}
    os.makedirs(RETR_DIR, exist_ok=True)
    L = [f"# RETRIEVED [{TAG}] — {uid} (type {rec.get('type')})", "",
         f"inputs: ../../stale_inputs/{uid[:8]}.md", ""]
    counts = {}
    for dim in DIMS:
        q = rec["probing_queries"][dim]
        if USE_MCP:
            mems = await mcp_recall(q, agent_id=agent, fleet_ids=[fleet], top_k=TOP_K)
        else:
            mems = (await mc.search(tenant_id=TENANT, query=q, top_k=TOP_K))["memories"]
        n_recency = 0
        if QEXP_N:
            alts = await expand_query(q)
            have = {str(m.get("id")) for m in mems}
            for aq in alts:
                if USE_MCP:
                    extra = await mcp_recall(aq, agent_id=agent, fleet_ids=[fleet], top_k=QEXP_K)
                else:
                    extra = (await mc.search(tenant_id=TENANT, query=aq, top_k=QEXP_K))["memories"]
                for m in extra:
                    if str(m.get("id")) not in have:
                        have.add(str(m.get("id")))
                        mems = list(mems) + [m]
                        n_recency += 1
        if ORACLE_NEW and by_fleet is not None:
            news = [m for m in by_fleet.get(fleet, []) if which(m.get("source_uri")) == "NEW"]
            have = {str(m.get("id")) for m in mems}
            added = [m for m in news if str(m.get("id")) not in have]
            n_recency = len(added)
            mems = list(mems) + added
        if RECENT_K and by_fleet is not None:
            # A57 (b1): append the fleet's most-recent memories (by session
            # index) that cosine missed; dedup by id. Order: cosine first,
            # recency tail — the answer LLM sees both vocabularies.
            recents = sorted(by_fleet.get(fleet, []), key=_session_idx, reverse=True)[:RECENT_K]
            have = {str(m.get("id")) for m in mems}
            added = [m for m in recents if str(m.get("id")) not in have]
            n_recency = len(added)
            mems = list(mems) + added
        recalled = [{"rank": j, "rel": which(m.get("source_uri")), "status": m.get("status"),
                     "sim": m.get("similarity"), "src": m.get("source_uri"),
                     "content": (m.get("content") or "")[:180]} for j, m in enumerate(mems)]
        hist = format_recall(mems); sysp, usr = build_prompts(hist, q, dim)
        resp = await answer(sysp, usr)
        responses[dim.replace("_query", "_response")] = resp
        counts[dim] = {"n": len(mems), "new": sum(1 for r in recalled if r["rel"] == "NEW"),
                       "old": sum(1 for r in recalled if r["rel"] == "OLD"),
                       "via_recency": n_recency}
        per[dim] = recalled
        L += [f"## {dim}  (recalled {len(mems)}: NEW={counts[dim]['new']} OLD={counts[dim]['old']})",
              f"**query:** {q}", "", "### exact prompt sent to answer LLM", "```",
              f"[system] {sysp}", f"[user] {usr}", "```", "### recalled memories (rank|rel|status|sim|content)"]
        for r in recalled:
            L.append(f"- {r['rank']:>2} | {r['rel'] or '-':<3} | {r['status']:<9} | {str(r['sim'])[:6]:>6} | {r['content']}")
        L += ["", f"### response\n{resp}", ""]
    open(os.path.join(RETR_DIR, f"{uid[:8]}.md"), "w").write("\n".join(L))
    row = {
        "uid": uid, "type": rec.get("type"),
        "dim1_query": rec["probing_queries"]["dim1_query"],
        "dim2_query": rec["probing_queries"]["dim2_query"],
        "dim3_query": rec["probing_queries"]["dim3_query"],
        "recall_dim1": counts["dim1_query"]["n"], "new_recalled_dim1": counts["dim1_query"]["new"],
        "old_recalled_dim1": counts["dim1_query"]["old"],
        "recency_added_dim1": counts["dim1_query"]["via_recency"],
        "inputs_md": f"stale_inputs/{uid[:8]}.md",
        "retrieved_md": f"stale_retrieved/{TAG}/{uid[:8]}.md",
    }
    return {"uid": uid, "target_model_responses": responses, "row": row,
            "debug": {"uid": uid, "type": rec.get("type"), "dims": per}}


async def main():
    os.environ["MEMCLAW_WRITE_MODE"] = "strong"
    mc = R.MemClawClient(URL, MC_KEY)
    data = json.load(open(DATA))
    records = (data if isinstance(data, list) else data.get("data", []))[:N]
    print(f"STALE x MemClaw | mode={MODE} tag={TAG} | {len(records)} questions | "
          f"recall={'memclaw_recall' if USE_MCP else '/search'} k={TOP_K} recent_k={RECENT_K} oracle_new={ORACLE_NEW} qexp={QEXP_N}x{QEXP_K} | model={ANSWER_MODEL}", flush=True)
    t0 = time.time()

    if MODE in ("seed", "both"):
        for i, rec in enumerate(records):
            try:
                s = await seed_one(mc, rec)
                print(f"[seed {i+1}/{len(records)}] {rec['uid'][:8]} created={s['created']} superseded={s['superseded']} | {int(time.time()-t0)}s", flush=True)
            except Exception as e:
                print(f"[seed {i+1}] ERROR {rec.get('uid')}: {e}", flush=True)

    if MODE == "inputs":
        # Rebuild enriched inputs .md from the PERSISTENT seed (settled RDF/status; no re-ingest):
        # meta + raw-haystack dump + TABLE of every created memory (with subject/predicate/object).
        os.makedirs(INPUTS_DIR, exist_ok=True); os.makedirs(os.path.join(INPUTS_DIR, "haystack"), exist_ok=True)
        allm = await mc.list_memories(TENANT)
        by_fleet = {}
        for m in allm:
            by_fleet.setdefault(m.get("fleet_id"), []).append(m)
        for rec in records:
            uid = rec["uid"]; fleet = f"stale-{uid}"[:60]
            which, io, inw = _tagger(rec)
            mems = sorted(by_fleet.get(fleet, []), key=lambda m: m.get("source_uri", ""))
            json.dump(rec["haystack_session"], open(os.path.join(INPUTS_DIR, "haystack", f"{uid[:8]}.json"), "w"), ensure_ascii=False)
            L = [f"# INPUTS — {uid}  (type {rec.get('type')})", "",
                 f"**M_old** (session {io}): {rec.get('M_old')}",
                 f"**M_new** (session {inw}): {rec.get('M_new')}",
                 f"**why:** {rec.get('explanation')}", "", "## Probing queries",
                 f"- dim1: {rec['probing_queries']['dim1_query']}",
                 f"- dim2: {rec['probing_queries']['dim2_query']}",
                 f"- dim3: {rec['probing_queries']['dim3_query']}", "",
                 f"## Raw haystack: `haystack/{uid[:8]}.json` ({len(rec['haystack_session'])} sessions, {sum(len(s) for s in rec['haystack_session'])} turns)", "",
                 f"## Created memories: {len(mems)} total (fleet `{fleet}`). Supersession needs same subject_entity_id+predicate.", "",
                 "| rel | src | type | status | subject_entity_id | predicate | object_value | supersedes_id | title |",
                 "|---|---|---|---|---|---|---|---|---|"]
            for m in mems:
                su = (m.get("source_uri") or "").replace("longmemeval://", "")
                L.append(f"| {which(m.get('source_uri')) or ''} | {su} | {m.get('memory_type','')} | {m.get('status','')} | "
                         f"{str(m.get('subject_entity_id') or '')[:8]} | {(m.get('predicate') or '')[:24]} | "
                         f"{(m.get('object_value') or '')[:24]} | {str(m.get('supersedes_id') or '')[:8]} | {(m.get('title') or '')[:40]} |")
            L += ["", "## Full text of M_old / M_new memories", ""]
            for m in mems:
                rl = which(m.get("source_uri"))
                if rl:
                    L.append(f"- **[{rl}]** ({(m.get('source_uri') or '').replace('longmemeval://','')}) status=`{m.get('status')}`\n    {m.get('content') or ''}")
            open(os.path.join(INPUTS_DIR, f"{uid[:8]}.md"), "w").write("\n".join(L))
        print(f"rebuilt {len(records)} enriched inputs .md from persistent seed ({len(allm)} memories)", flush=True)

    if MODE in ("run", "both"):
        by_fleet = None
        if RECENT_K or ORACLE_NEW:
            allm = await mc.list_memories(TENANT)
            by_fleet = {}
            for m in allm:
                by_fleet.setdefault(m.get("fleet_id"), []).append(m)
            print(f"recency augmentation ON: k={RECENT_K}, {len(allm)} memories cached across {len(by_fleet)} fleets", flush=True)
        out = []; dbg = []; rows = []
        for i, rec in enumerate(records):
            try:
                r = await eval_one(mc, rec, by_fleet=by_fleet)
                out.append({"uid": r["uid"], "target_model_responses": r["target_model_responses"]})
                dbg.append(r["debug"]); rows.append(r["row"])
                c1 = r["row"]
                print(f"[run {i+1}/{len(records)}] {r['uid'][:8]} recall_dim1={c1['recall_dim1']} new={c1['new_recalled_dim1']} | {int(time.time()-t0)}s", flush=True)
            except Exception as e:
                print(f"[run {i+1}] ERROR {rec.get('uid')}: {e}", flush=True)
            with open(OUT, "w") as f:
                json.dump(out, f, ensure_ascii=False)
            with open(OUT.replace("answers", "debug"), "w") as f:
                json.dump(dbg, f, ensure_ascii=False)
            if rows:
                with open(CSV_PATH, "w", newline="") as f:
                    w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)
        print(f"\nWrote {len(out)} answers -> {OUT}", flush=True)
        print(f"CSV index -> {CSV_PATH} | inputs -> {INPUTS_DIR}/ | retrieved -> {RETR_DIR}/", flush=True)
    await mc.http.aclose()

asyncio.run(main())
