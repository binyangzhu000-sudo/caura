#!/usr/bin/env python3
"""
MemClaw-based LongMemEval Benchmark Runner.

Uses MemClaw purely for retrieval (POST /api/search), then generates answers
using the LongMemEval chain-of-thought prompt with a direct OpenAI call.
This cleanly separates retrieval quality from answer generation quality.

Calls the MemClaw REST API:
  - POST /api/memories  for ingestion
  - POST /api/search    for retrieval (pure — no LLM summarization)

Zero imports from the MemClaw codebase — pure HTTP client.

Usage:
    python runner_memclaw.py -t single-session-user -n 10 --url http://localhost:8000 --api-key mc_xxx
    python runner_memclaw.py -t temporal-reasoning -n 20 --url https://memclaw.net --api-key mc_xxx
    python runner_memclaw.py -t knowledge-update -i 0  # single test at index 0

Requires:
    pip install httpx openai
"""

import argparse
import asyncio
import random
import json
import os
import re
import statistics
import time

from dotenv import load_dotenv

load_dotenv()
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ============================================================================
# Paths
# ============================================================================

DATA_FILE = Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"
RESULTS_DIR = Path(__file__).parent / "results"

# ============================================================================
# Defaults
# ============================================================================

DEFAULT_URL = "http://localhost:8000"
DEFAULT_AGENT_ID = os.environ.get("MEMCLAW_BENCH_AGENT") or "longmemeval-bench"
DEFAULT_DISTANCE_FROM_ANSWER = 0  # 0 = only answer sessions, None = all sessions
TIMEOUT = 120.0
BULK_MAX_ITEMS = 100  # MemClaw hard limit
MAX_CONTENT_LENGTH = 10000  # MemClaw hard limit

# Parallel processing
MAX_CONCURRENT_INGESTIONS = int(os.environ.get("MEMCLAW_BENCH_INGEST_CONCURRENCY", "2"))   # concurrent memory writes per test case (low to avoid agent upsert races; override via env)
MAX_CONCURRENT_TEST_CASES = 5  # concurrent test cases


# ============================================================================
# Date Parsing (same logic as test_single.py)
# ============================================================================

def parse_benchmark_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse LongMemEval date format to timezone-aware UTC datetime.
    Format: '2023/05/20 (Sat) 02:21' -> datetime (UTC)
    """
    if not date_str:
        return None
    try:
        clean_date = re.sub(r'\s*\([^)]*\)\s*', ' ', date_str).strip()
        for fmt in ['%Y/%m/%d %H:%M', '%Y/%m/%d', '%Y-%m-%d %H:%M', '%Y-%m-%d']:
            try:
                naive_dt = datetime.strptime(clean_date, fmt)
                return naive_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None
    except Exception:
        return None


def datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO 8601 string for the MemClaw API."""
    if dt is None:
        return None
    return dt.isoformat()


# ============================================================================
# Haystack Filtering (distance_from_answer)
# ============================================================================

def filter_haystack_by_distance(
    test_case: Dict[str, Any],
    distance: Optional[int],
) -> Tuple[List[List[Dict]], List[str], List[str]]:
    """
    Filter haystack sessions to only include sessions near the answer.

    LongMemEval packs 50+ distractor sessions around the answer session(s).
    `distance_from_answer` controls how many neighboring sessions to keep:
      - 0:    Only the answer session(s) themselves
      - N>0:  Answer session(s) ± N neighboring sessions
      - None: All sessions (no filtering)

    Returns filtered (sessions, session_ids, dates) tuples.
    """
    sessions = test_case.get("haystack_sessions", [])
    session_ids = test_case.get("haystack_session_ids", [])
    dates = test_case.get("haystack_dates", [])
    answer_ids = set(test_case.get("answer_session_ids", []))

    # No filtering requested — return everything
    if distance is None:
        return sessions, session_ids, dates

    # Find indices of answer sessions
    answer_indices = [i for i, sid in enumerate(session_ids) if sid in answer_ids]

    if not answer_indices:
        # No answer sessions marked — fall back to all sessions
        return sessions, session_ids, dates

    # Build set of indices to keep: answer ± distance
    keep = set()
    for ai in answer_indices:
        for d in range(-distance, distance + 1):
            idx = ai + d
            if 0 <= idx < len(sessions):
                keep.add(idx)

    # Filter, preserving order
    filtered_sessions = [sessions[i] for i in sorted(keep)]
    filtered_ids = [session_ids[i] for i in sorted(keep)]
    filtered_dates = [dates[i] for i in sorted(keep)]

    return filtered_sessions, filtered_ids, filtered_dates


# ============================================================================
# Data Loading (same as test_single.py)
# ============================================================================

def load_test_cases(
    file_path: Path,
    question_type: str,
    count: Optional[int] = None,
    start_index: int = 0,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load test cases from the benchmark data file.

    If seed is provided, randomly sample `count` cases (ignoring start_index).
    Otherwise, slice sequentially by start_index + count.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    matching = [item for item in data if item.get('question_type') == question_type]

    if not matching:
        print(f"No test cases found for question_type='{question_type}'")
        print(f"Available types: {sorted(set(item.get('question_type') for item in data))}")
        return []

    if seed is not None and count:
        import random
        random.seed(seed)
        matching = random.sample(matching, min(count, len(matching)))
    elif count:
        matching = matching[start_index:start_index + count]
    elif start_index > 0:
        matching = matching[start_index:start_index + 1]

    return matching


# ============================================================================
# LLM Judge — official LongMemEval task-specific prompts (evaluate_qa.py)
# ============================================================================

def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool = False,
) -> str:
    """Build task-specific judge prompt — mirrors official LongMemEval evaluate_qa.py exactly."""
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        else:
            raise ValueError(f"Unknown question type: {task}")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        return template.format(question, answer, response)


async def evaluate_answer(
    question: str,
    expected_answer: str,
    generated_response: str,
    question_type: str,
    question_id: str,
    openai_api_key: str,
) -> bool:
    """LLM judge using official LongMemEval task-specific prompts."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_api_key)

    abstention = "_abs" in question_id
    prompt = get_anscheck_prompt(
        task=question_type,
        question=question,
        answer=expected_answer,
        response=generated_response,
        abstention=abstention,
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().lower()
        return "yes" in result
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return False


# ============================================================================
# Timing Statistics (same structure as test_single.py)
# ============================================================================

@dataclass
class TimingStats:
    """Collects timing statistics across all test cases.

    Shared buckets (``ingestion_times``, ``test_case_times``) aggregate once
    per case regardless of K.  Per-K buckets segment search / answer-gen /
    eval by the retrieval depth used for that iteration so multi-K runs can
    report latency breakdowns cleanly.

    ``recall_times`` and ``evaluation_times`` are legacy (mirror the first K
    in the list) and kept populated for downstream scripts that haven't yet
    learned about the ``per_k`` structure.
    """
    ingestion_times: List[float] = field(default_factory=list)
    recall_times: List[float] = field(default_factory=list)
    evaluation_times: List[float] = field(default_factory=list)
    test_case_times: List[float] = field(default_factory=list)
    per_k_search_times: Dict[int, List[float]] = field(default_factory=dict)
    per_k_answer_times: Dict[int, List[float]] = field(default_factory=dict)
    per_k_eval_times: Dict[int, List[float]] = field(default_factory=dict)

    def _stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "std": 0, "p95": 0, "p99": 0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def percentile(p):
            k = (n - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < n else f
            return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f]) if c != f else sorted_vals[f]

        return {
            "count": n,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if n > 1 else 0,
            "p95": percentile(95),
            "p99": percentile(99),
        }

    def record_per_k(self, k: int, *, search: float, answer_gen: float, evaluation: float) -> None:
        """Helper: append per-K timings in one call."""
        self.per_k_search_times.setdefault(k, []).append(search)
        self.per_k_answer_times.setdefault(k, []).append(answer_gen)
        self.per_k_eval_times.setdefault(k, []).append(evaluation)

    def get_summary(self) -> Dict[str, Any]:
        all_ks = sorted(
            set(self.per_k_search_times)
            | set(self.per_k_answer_times)
            | set(self.per_k_eval_times)
        )
        per_k: Dict[str, Dict[str, Dict[str, float]]] = {}
        for k in all_ks:
            per_k[str(k)] = {
                "search": self._stats(self.per_k_search_times.get(k, [])),
                "answer_generation": self._stats(self.per_k_answer_times.get(k, [])),
                "evaluation": self._stats(self.per_k_eval_times.get(k, [])),
            }
        return {
            "ingestion": self._stats(self.ingestion_times),
            "recall": self._stats(self.recall_times),
            "evaluation": self._stats(self.evaluation_times),
            "test_case_execution": self._stats(self.test_case_times),
            "per_k": per_k,
        }


# ============================================================================
# MemClaw HTTP Client
# ============================================================================

class MemClawClient:
    """
    Thin HTTP client for the MemClaw REST API.

    Every memory operation goes through the real MemClaw service —
    enrichment, embedding, entity extraction, contradiction detection,
    hybrid search, graph expansion, recall summarization — all server-side.
    """

    def __init__(self, base_url: str, api_key: str | None = None):
        self.api = f"{base_url.rstrip('/')}/api/v1"
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        self.http = httpx.AsyncClient(headers=headers, timeout=TIMEOUT)
        # Per-write write_mode override via env. Accepts "fast" | "strong" | "stm".
        # When set, every POST /api/v1/memories carries this in the body so the
        # server bypasses the tenant default. Used by the regression bench to
        # force strong (synchronous enrichment + embedding) for deterministic
        # post-ingest searches.
        wm = os.environ.get("MEMCLAW_WRITE_MODE", "").strip().lower()
        self.write_mode: Optional[str] = wm if wm in ("fast", "strong", "stm") else None

    async def health_check(self) -> bool:
        try:
            r = await self.http.get(f"{self.api}/health")
            return r.status_code == 200
        except Exception:
            return False

    async def write_single(
        self,
        tenant_id: str,
        agent_id: str,
        item: Dict[str, Any],
        fleet_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        POST /api/memories — write a single memory.
        Triggers MemClaw's full pipeline: LLM enrichment, embedding,
        entity extraction, dedup, contradiction detection.
        """
        body: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            **item,
        }
        if fleet_id:
            body["fleet_id"] = fleet_id
        if self.write_mode and "write_mode" not in body:
            body["write_mode"] = self.write_mode

        # Retry transient throttling (nginx limit_req 429s) and upstream 5xx with
        # exponential backoff + jitter. memclaw.dev fronts the app with an nginx
        # limit_req that rejects sustained write bursts well below the app's own
        # rate limit; without retry those writes are LOST, leaving an incomplete
        # haystack and corrupting recall accuracy. Honors Retry-After when present.
        max_retries = int(os.environ.get("MEMCLAW_BENCH_WRITE_RETRIES", "6"))
        attempt = 0
        while True:
            r = await self.http.post(f"{self.api}/memories", json=body)
            if r.status_code in (200, 201):
                return r.json()
            if r.status_code in (429, 502, 503, 504) and attempt < max_retries:
                ra = r.headers.get("Retry-After", "")
                if ra.isdigit():
                    delay = float(ra)
                else:
                    delay = min(0.5 * (2 ** attempt), 30.0)
                delay += random.uniform(0, 0.5)
                await asyncio.sleep(delay)
                attempt += 1
                continue
            return {
                "_error": f"HTTP {r.status_code}: {r.text[:200]}",
            }

    async def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        valid_at: Optional[str] = None,
        diagnostic: bool = False,
        fleet_ids: Optional[List[str]] = None,
        score_formula: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        POST /api/search — pure retrieval, no LLM summarization.

        Returns: {"memories": [...], "memory_count": N, "search_ms": N}
        """
        body: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "query": query,
            "top_k": top_k,
        }
        if valid_at:
            body["valid_at"] = valid_at
        if diagnostic:
            body["diagnostic"] = True
        if fleet_ids:
            # Scope recall to a fleet (default is tenant-wide). Used by the seed/reuse
            # flow to keep each category's memories isolated to its own fleet.
            body["fleet_ids"] = fleet_ids
        if score_formula is not None:
            # Per-request search-profile override (A/B the ranking formula):
            # 0 = legacy multiplicative boost stack, 1 = unified relevance-dominant.
            body["search_profile"] = {"score_formula": score_formula}

        r = await self.http.post(f"{self.api}/search", json=body)

        if r.status_code == 200:
            data = r.json()
            # /search returns a list of MemoryOut objects directly
            memories = data if isinstance(data, list) else data.get("items", [])
            return {
                "memories": memories,
                "memory_count": len(memories),
                "_raw_response": data,
            }
        else:
            return {
                "memories": [],
                "memory_count": 0,
                "_error": f"HTTP {r.status_code}: {r.text[:200]}",
            }

    async def recall(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 3,
        valid_at: Optional[str] = None,
        diagnostic: bool = False,
    ) -> Dict[str, Any]:
        """
        POST /api/recall — triggers MemClaw's full pipeline:
        embedding, hybrid search (vector + keyword + graph + freshness + recall boost),
        then LLM summarization.

        Returns: {"query", "summary", "memory_count", "memories", "recall_ms"}
        """
        body: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "query": query,
            "top_k": top_k,
        }
        if valid_at:
            body["valid_at"] = valid_at
        if diagnostic:
            body["diagnostic"] = True

        r = await self.http.post(f"{self.api}/recall", json=body)

        if r.status_code == 200:
            return r.json()
        else:
            return {
                "query": query,
                "summary": "",
                "memory_count": 0,
                "memories": [],
                "recall_ms": 0,
                "_error": f"HTTP {r.status_code}: {r.text[:200]}",
            }

    async def list_memories(
        self,
        tenant_id: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        GET /api/memories — list all memories for a tenant.
        Returns lightweight dicts for diagnostics (no embedding vectors).
        Paginates automatically if more than `limit` memories exist.
        """
        all_memories = []
        offset = 0
        while True:
            r = await self.http.get(
                f"{self.api}/memories",
                params={
                    "tenant_id": tenant_id,
                    "limit": limit,
                    "offset": offset,
                    "sort": "created_at",
                    "order": "asc",
                },
            )
            if r.status_code != 200:
                break
            data = r.json()
            items = data.get("items", [])
            if not items:
                break
            all_memories.extend(items)
            # If we got fewer than limit, we've reached the end
            if len(items) < limit:
                break
            offset += limit
        return all_memories

    async def wait_until_embedded(
        self,
        tenant_id: str,
        max_wait: float = 60.0,
        poll_interval: float = 3.0,
    ) -> Dict[str, Any]:
        """Poll until every stored memory for this tenant is embedded.

        Deferred-embedding deployments (e.g. staging, EMBED_ON_HOT_PATH=false)
        embed asynchronously via a worker, so a fixed settle sleep races the
        worker → partial/unstable recall, and the post-test tenant-wide cleanup
        deletes rows whose embed-requests are still queued (worker then logs
        "row deleted between publish and processing" and never embeds them).

        We poll ``metadata.embedding_pending`` (exposed on the list endpoint)
        and return once all rows are embedded, or ``max_wait`` elapses. A row
        with the flag absent (e.g. strong/inline writes) counts as embedded.

        Returns {reason, embedded, total, enrich_pending, elapsed}.
        """
        start = time.time()
        while True:
            mems = await self.list_memories(tenant_id)
            total = len(mems)
            embedded = sum(
                1 for m in mems
                if not (m.get("metadata") or {}).get("embedding_pending", False)
            )
            enrich_pending = sum(
                1 for m in mems
                if (m.get("metadata") or {}).get("enrichment_pending", False)
            )
            elapsed = time.time() - start
            if total > 0 and embedded == total:
                return {"reason": "all_embedded", "embedded": embedded,
                        "total": total, "enrich_pending": enrich_pending, "elapsed": elapsed}
            if elapsed >= max_wait:
                return {"reason": "timeout", "embedded": embedded,
                        "total": total, "enrich_pending": enrich_pending, "elapsed": elapsed}
            await asyncio.sleep(poll_interval)

    async def delete_tenant_memories(self, tenant_id: str):
        """Cleanup: delete all memories for this tenant."""
        await self.http.delete(f"{self.api}/memories", params={"tenant_id": tenant_id})

    async def close(self):
        await self.http.aclose()


# ============================================================================
# Ingestion: LongMemEval sessions → MemClaw single writes
# ============================================================================

def build_memory_items(
    session: List[Dict[str, Any]],
    session_id: str,
    date_str: str,
    session_idx: int,
) -> List[Dict[str, Any]]:
    """
    Convert one LongMemEval session into MemClaw memory item dicts.

    Each user turn becomes a memory. Content includes the speaker role
    for context. Timestamps and metadata are preserved for traceability.
    """
    ts = datetime_to_iso(parse_benchmark_date(date_str))
    items = []

    for turn_idx, msg in enumerate(session):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue

        # Prefix with role so MemClaw's enrichment sees the conversational context
        prefixed = f"{role.capitalize()}: {content}"

        # Truncate if needed (MemClaw MAX_CONTENT_LENGTH = 10000)
        if len(prefixed) > MAX_CONTENT_LENGTH:
            prefixed = prefixed[:MAX_CONTENT_LENGTH]

        item: Dict[str, Any] = {
            "content": prefixed,
            "source_uri": f"longmemeval://sess-{session_id}/turn-{turn_idx}",
            "metadata": {
                "benchmark": "longmemeval",
                "session_id": session_id,
                "session_idx": session_idx,
                "turn_idx": turn_idx,
                "role": role,
                "has_answer": msg.get("has_answer", False),
            },
        }
        if ts:
            item["ts_valid_start"] = ts
            item["reference_datetime"] = ts

        items.append(item)

    return items


async def ingest_test_case(
    client: MemClawClient,
    tenant_id: str,
    haystack_sessions: List[List[Dict]],
    haystack_session_ids: List[str],
    haystack_dates: List[str],
    max_concurrent: int = MAX_CONCURRENT_INGESTIONS,
    fleet_override: Optional[str] = None,
    agent_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest all sessions for one test case into MemClaw via concurrent single writes.

    Returns ingestion summary: {total_created, total_duplicates, total_errors, elapsed_s}
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    # Agent for these writes: a per-question agent (agent_override) in seed mode so
    # every write in this case shares ONE home fleet (no cross-fleet 403); else the
    # global bench agent.
    write_agent = agent_override or DEFAULT_AGENT_ID

    # Build all (item, fleet_id, session_idx) tuples upfront
    all_writes: List[Tuple[Dict[str, Any], str, int]] = []
    for session_idx, (session, session_id, date_str) in enumerate(
        zip(haystack_sessions, haystack_session_ids, haystack_dates)
    ):
        items = build_memory_items(session, session_id, date_str, session_idx)
        for item in items:
            # fleet_override (seed mode) = a per-QUESTION fleet, so this case's memories
            # are isolated to their own fleet and recall can be scoped to exactly them
            # (preserves LongMemEval per-question isolation in a persistent tenant).
            # Else: MEMCLAW_BENCH_FLEET pins one fleet, or per-session fleets locally.
            fleet = fleet_override or os.environ.get("MEMCLAW_BENCH_FLEET") or f"sess-{session_id}"
            all_writes.append((item, fleet, session_idx))

    if not all_writes:
        return {"total_created": 0, "total_duplicates": 0, "total_errors": 0}

    # Write the first memory sequentially to ensure the agent row is created
    # before concurrent writes race on the unique constraint.
    first_item, first_fid, first_sidx = all_writes[0]
    first_result = await client.write_single(
        tenant_id=tenant_id,
        agent_id=write_agent,
        item=first_item,
        fleet_id=first_fid,
    )
    if "_error" in first_result:
        print(f"    Write error (session {first_sidx}) [{first_item.get('source_uri','')}] item={first_item.get('content','')[:90]!r}: {first_result['_error']}")
    results = [first_result]

    # Now parallelize the rest
    if len(all_writes) > 1:
        async def write_one(item: Dict, fleet_id: str, session_idx: int) -> Dict:
            async with semaphore:
                result = await client.write_single(
                    tenant_id=tenant_id,
                    agent_id=write_agent,
                    item=item,
                    fleet_id=fleet_id,
                )
                if "_error" in result:
                    print(f"    Write error (session {session_idx}) [{item.get('source_uri','')}] item={item.get('content','')[:90]!r}: {result['_error']}")
                return result

        rest_results = await asyncio.gather(
            *[write_one(item, fid, sidx) for item, fid, sidx in all_writes[1:]],
            return_exceptions=True,
        )
        results.extend(rest_results)

    total_created = 0
    total_duplicates = 0
    total_errors = 0
    for r in results:
        if isinstance(r, Exception):
            total_errors += 1
        elif "_error" in r:
            total_errors += 1
        else:
            total_created += 1

    return {
        "total_created": total_created,
        "total_duplicates": total_duplicates,
        "total_errors": total_errors,
    }


# ============================================================================
# Answer Generation (LongMemEval-style CoT prompt, runner-side)
# ============================================================================

ANSWER_PROMPT_COT = """\
I will give you several memories from past conversations with a user. \
Please answer the question based on the relevant memories. \
Answer the question step by step: first extract all the relevant information, \
and then reason over the information to get the answer.

IMPORTANT rules:
1. When memories contain CONTRADICTORY or UPDATED information about the same fact, \
ALWAYS prefer the MOST RECENT memory (latest date). Older values are outdated.
2. Verify that the specific entities in the question (names, roles, items, sports, etc.) \
match EXACTLY what appears in the memories. If the question asks about "Dr. Johnson" but \
memories only mention "Dr. Smith", or asks about "football" but memories only mention \
"baseball", the answer is NOT in the memories — say "I don't have enough information to \
answer this question."
3. If no retrieved memory contains information relevant to the question, say \
"I don't have enough information to answer this question."


Memories:

{memories_string}

Current Date: {question_date}
Question: {question}
Answer (step by step):"""


# Advice-seeking prompt for single-session-preference questions.
# These questions are NOT factual QA — they are help/advice requests, and the
# memories carry the user's preferences, constraints, and prior context. The
# model must produce a concrete personalized response that USES those preferences,
# not analyse them. Critically: do NOT abstain just because the question's topic
# (e.g. "Miami hotels") isn't in memory — the memories carry the user's *style*
# preferences that apply to any topic in that domain.
ANSWER_PROMPT_PREFERENCE = """\
The user is asking you for help, advice, or a recommendation. Below are memories \
from past conversations that reveal the user's preferences, constraints, prior \
decisions, and stated interests relevant to this kind of request.

Your job: produce a concrete, helpful response that DIRECTLY answers the user's \
request and is PERSONALIZED based on what the memories tell you about the user.

Critical rules:
1. The memories carry the user's preferences and context — apply them. \
The question's specific subject (a city, a product, a topic) does NOT need to \
appear in memories; what should appear is the user's style/constraints/prior \
context that informs your suggestions.
2. Do NOT say "I don't have enough information" — you have the user's preferences \
in the memories. Use them.
3. Do NOT just analyse the memories or list what you found. Make actual \
suggestions or give actual advice.
4. Reference specifics from the memories (named items, prior decisions, stated \
likes/dislikes) so the response is clearly tailored to this user.
5. If memories contain contradictory or updated preferences, defer to the most \
recent.

Memories:

{memories_string}

Current Date: {question_date}
User's request: {question}
Your personalized response:"""


def format_memories_for_qa(memories: List[Dict[str, Any]]) -> str:
    """Format retrieved MemClaw memories for the QA prompt.

    Sorts chronologically and presents each memory with its date,
    type, title, and content — mirroring LongMemEval's session format.
    """
    # Sort by ts_valid_start (chronological order)
    def sort_key(m: Dict) -> str:
        ts = m.get("ts_valid_start") or ""
        return ts if isinstance(ts, str) else str(ts)

    sorted_mems = sorted(memories, key=sort_key)

    blocks = []
    for i, m in enumerate(sorted_mems, 1):
        ts = m.get("ts_valid_start", "")
        # Extract date portion
        if isinstance(ts, str) and len(ts) >= 10:
            date_str = ts[:10]
        else:
            date_str = "Unknown"

        title = m.get("title") or ""
        content = m.get("content", "")
        mem_type = m.get("memory_type", "")
        status = m.get("status", "active")

        header = f"### Memory {i}:"
        header += f"\nDate: {date_str}"
        if title:
            header += f"\nTitle: {title}"
        if mem_type:
            header += f"\nType: {mem_type}"
        if status and status != "active":
            header += f"\nStatus: {status}"
        header += f"\nContent:\n{content}"

        blocks.append(header)

    return "\n\n".join(blocks)


async def generate_answer(
    question: str,
    memories: List[Dict[str, Any]],
    question_date: Optional[str],
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    question_type: Optional[str] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Generate an answer using LongMemEval-style CoT prompt.

    For single-session-preference, switches to an advice-giving prompt because
    those questions are advice requests, not factual QA — the memories carry
    user preferences to apply, not facts to extract.

    Returns (answer_text, usage_dict).
    """
    from openai import AsyncOpenAI

    if not memories:
        return "No relevant memories found.", None

    memories_string = format_memories_for_qa(memories)

    # Format question_date like LongMemEval: "2023/05/20"
    date_display = "Unknown"
    if question_date:
        dt = parse_benchmark_date(question_date)
        if dt:
            date_display = dt.strftime("%Y/%m/%d")

    if question_type == "single-session-preference":
        prompt_template = ANSWER_PROMPT_PREFERENCE
    else:
        prompt_template = ANSWER_PROMPT_COT
    # A64 control knob: the premise-challenge instruction from the STALE
    # experiments, verbatim. Off by default; used to measure over-hedging on
    # TRUE-premise questions before the instruction can ship in any product
    # prompt. Appended as an extra rule so both templates stay untouched.
    if os.environ.get("PREMISE_CHECK", "0") == "1":
        prompt_template += (
            "\n\nAdditional rule: before responding, check whether the question or "
            "request rests on an assumption about the user's current situation that "
            "the memories contradict or no longer support (they may imply a change "
            "without stating it outright). If so, say that the assumption appears "
            "outdated and answer for the user's actual current situation instead of "
            "going along with the premise. If the memories do answer the question, "
            "answer it — do not abstain merely because a memory is older; flag only "
            "assumptions the memories actually contradict or supersede."
        )
    prompt = prompt_template.format(
        memories_string=memories_string,
        question_date=date_display,
        question=question,
    )

    client = AsyncOpenAI(api_key=openai_api_key)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()
        usage = {
            "model": model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
        return answer, usage
    except Exception as e:
        return f"Answer generation failed: {e}", None


def _normalize_text(s: str) -> str:
    """Lowercase + strip punctuation → whitespace, collapse spaces. For the
    URI-independent content match."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def compute_retrieval_metrics(
    answer_source_uris: List[str],
    retrieved_memories: List[Dict[str, Any]],
    expected_answer: str | None = None,
) -> Dict[str, Any]:
    """Retrieval quality of the top-k.

    ``retrieved_memories`` MUST be in ranked order (index 0 = rank 1).

    Beyond the legacy set-membership ``recall_at_k`` (order-blind — a reranker
    that reorders within top-k leaves it unchanged), we compute:

    - ``gold_rank`` / ``mrr`` / ``hit_at_1`` / ``hit_at_3`` — RANK-SENSITIVE, so
      a reranker moving the gold *up* is actually measured.
    - ``answer_in_content`` — URI-independent presence of the expected answer
      text in any retrieved memory. Robust to MemClaw's write pipeline
      (dedup/enrich/merge) rewriting or collapsing the gold turn's source_uri,
      which makes exact-URI ``recall_at_k`` under-count real retrieval.
      Best-effort (normalized-substring); a lower bound, not exact.
    """
    k = len(retrieved_memories)
    ordered_uris = [m.get("source_uri") for m in retrieved_memories]
    retrieved_uris = {u for u in ordered_uris if u}
    answer_uris = set(answer_source_uris)

    # Rank of the first gold URI in ranked order (None if absent from top-k).
    gold_rank = next((i for i, u in enumerate(ordered_uris, start=1) if u in answer_uris), None)
    mrr = (1.0 / gold_rank) if gold_rank else 0.0
    hit_at_1 = gold_rank == 1
    hit_at_3 = gold_rank is not None and gold_rank <= 3

    # URI-independent content match against the expected answer text.
    answer_in_content: bool | None = None
    if expected_answer:
        norm_ans = _normalize_text(expected_answer)
        if norm_ans:
            answer_in_content = any(
                norm_ans in _normalize_text(m.get("content", "")) for m in retrieved_memories
            )

    base = {
        "k": k,
        "gold_rank": gold_rank,
        "mrr": mrr,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "answer_in_content": answer_in_content,
    }
    if not answer_uris:
        return {"recall_at_k": None, "answer_found": 0, "answer_total": 0, **base}

    found = len(answer_uris & retrieved_uris)
    return {
        "recall_at_k": found / len(answer_uris),
        "answer_found": found,
        "answer_total": len(answer_uris),
        **base,
    }


# ============================================================================
# Test Runner
# ============================================================================

async def run_single_test_case(
    client: MemClawClient,
    test_case: Dict[str, Any],
    test_id: int,
    openai_api_key: str,
    timing_stats: TimingStats,
    question_type: str,
    cleanup: bool = True,
    distance_from_answer: Optional[int] = DEFAULT_DISTANCE_FROM_ANSWER,
    top_k: int = 3,
    top_ks: Optional[List[int]] = None,
    diagnostic: bool = False,
    answer_model: str = "gpt-4o-mini",
    settle_time: float = 0.0,
    reuse_seed: bool = False,
    seed_only: bool = False,
    search_fleet: Optional[str] = None,
    score_formula: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single benchmark test case against MemClaw, at one or more K values.

    If ``top_ks`` is provided, the case is evaluated at each K value in sequence
    — ingestion + cleanup run ONCE; search / answer-gen / LLM-judge run per K.
    If ``top_ks`` is None, falls back to a single evaluation at ``top_k``
    (fully backwards-compatible with single-K callers).

    Output is always structured with ``per_k_results`` (a dict keyed by K).
    Top-level convenience fields (``retrieval_metrics``, ``generated_answer``,
    ``model_answer``, ``top_k``) mirror the HEADLINE K — the largest K in the
    list — so existing scripts reading the old single-K shape still work.
    """
    test_start = time.time()

    # Resolve the list of K values.
    if top_ks is None or len(top_ks) == 0:
        top_ks = [top_k]
    # Canonicalize: dedup + ascending.
    ks = sorted(set(int(k) for k in top_ks))
    headline_k = ks[-1]  # largest K — best-case recall signal for the progress line

    # Unique tenant per test case for isolation.
    # MEMCLAW_BENCH_TENANT override: pin to a single fixed tenant when the API
    # key is tenant-scoped (e.g. memclaw.dev staging). Requires -c 1 so the
    # per-test tenant-wide cleanup doesn't race across concurrent tests.
    tenant_id = os.environ.get("MEMCLAW_BENCH_TENANT") or f"lme-{test_id}-{uuid.uuid4().hex[:8]}"

    # Filter haystack to sessions near the answer
    haystack_sessions, haystack_session_ids, haystack_dates = filter_haystack_by_distance(
        test_case, distance_from_answer,
    )
    total_messages = sum(len(s) for s in haystack_sessions)

    all_sessions = test_case.get("haystack_sessions", [])
    if len(haystack_sessions) < len(all_sessions):
        print(f"  [distance_from_answer={distance_from_answer}] "
              f"Filtered {len(all_sessions)} → {len(haystack_sessions)} sessions "
              f"({sum(len(s) for s in all_sessions)} → {total_messages} messages)")

    # Collect source_uris for turns that contain the answer
    answer_source_uris = []
    answer_session_ids_set = set(str(s) for s in test_case.get("answer_session_ids", []))
    for sid, session in zip(haystack_session_ids, haystack_sessions):
        if str(sid) not in answer_session_ids_set:
            continue
        for tidx, msg in enumerate(session):
            if msg.get("has_answer", False):
                answer_source_uris.append(f"longmemeval://sess-{sid}/turn-{tidx}")

    print(f"\n{'='*60}")
    print(f"Test Case #{test_id}  (K values: {ks})")
    print(f"{'='*60}")
    print(f"Tenant: {tenant_id}")
    print(f"Question: {test_case['question']}")
    print(f"Sessions: {len(haystack_sessions)}, Messages: {total_messages}")

    question_id = test_case.get("question_id", "")

    # Per-question fleet + agent for the seed/reuse flow: isolate each question's ~50
    # sessions to its OWN fleet, so recall over a persistent shared tenant matches the
    # per-case-isolated flow (faithful LongMemEval). agent:fleet is 1:1 so each agent
    # writes only to its home fleet (no cross-fleet 403 / no trust elevation needed).
    # Deterministic from question_id → --seed-only and --reuse-seed derive the SAME fleet.
    q_key = "".join(c if c.isalnum() else "-" for c in (question_id or f"t{test_id}"))
    # Writes ALWAYS use a per-question 1:1 agent:fleet so each agent writes only to
    # its own fleet (no cross-fleet 403 — plain mode's per-session fleets tripped the
    # fleet-scope policy). In plain June-style mode, cleanup isolates each question, so
    # a per-question write-fleet + tenant-wide search reproduces per-case isolation.
    q_fleet = f"q-{q_key}"
    q_agent = f"qa-{q_key}"

    # Per-K result template — populated inside the loop below.
    per_k_results: Dict[str, Dict[str, Any]] = {}

    result: Dict[str, Any] = {
        "test_id": test_id,
        "tenant_id": tenant_id,
        "question_type": question_type,
        "question_id": question_id,
        "question": test_case["question"],
        "expected_answer": test_case["answer"],
        # Top-level convenience fields mirror the HEADLINE K (largest).
        # Populated after the per-K loop for backwards compat with old scripts.
        "generated_answer": None,
        "model_answer": None,
        "session_messages_count": total_messages,
        "memories_created": 0,
        "memories_retrieved": 0,
        "success": False,
        "errors": [],
        "llm_usage": None,
        "retrieval_metrics": None,
        "timing": {
            "test_case_total": 0,
            "ingestion": 0,
            "search": 0,           # mirror of headline K
            "answer_generation": 0,
            "evaluation": 0,
        },
        # Diagnostics for failure analysis
        "answer_session_ids": test_case.get("answer_session_ids", []),
        "answer_source_uris": answer_source_uris,
        "question_date": test_case.get("question_date"),
        "haystack_session_ids": [str(sid) for sid in haystack_session_ids],
        "all_stored_memories": [],  # populated once after ingestion
        "top_k": headline_k,
        "top_ks": ks,
        "per_k_results": per_k_results,
    }

    try:
        # ==== Phase A — once per case ====

        # Step 1: Ingest sessions via POST /api/memories.
        # --reuse-seed: skip ingest entirely — the corpus is already seeded and we
        # only re-run retrieval. This is the whole point of the seed-once flow.
        if reuse_seed:
            print(f"\n[1/4] REUSE-SEED — skipping ingest; querying pre-seeded tenant {tenant_id}")
        else:
            print(f"\n[1/4] Ingesting {total_messages} messages into MemClaw...")
            ingestion_start = time.time()

            ingest_result = await ingest_test_case(
                client=client,
                tenant_id=tenant_id,
                haystack_sessions=haystack_sessions,
                haystack_session_ids=haystack_session_ids,
                haystack_dates=haystack_dates,
                fleet_override=q_fleet,
                agent_override=q_agent,
            )

            ingestion_time = time.time() - ingestion_start
            timing_stats.ingestion_times.append(ingestion_time)
            result["timing"]["ingestion"] = ingestion_time
            result["memories_created"] = ingest_result["total_created"]

            print(f"  Created: {ingest_result['total_created']}, "
                  f"Duplicates: {ingest_result['total_duplicates']}, "
                  f"Errors: {ingest_result['total_errors']} "
                  f"({ingestion_time:.2f}s)")

        # Poll the deferred embedder instead of a blind sleep. On
        # EMBED_ON_HOT_PATH=false deployments (staging) a fixed sleep races the
        # async worker → partial/unstable recall, and the tenant-wide cleanup
        # below deletes rows whose embed-requests are still queued (worker logs
        # "row deleted between publish and processing" and never embeds them).
        # Proceed once every stored memory is embedded, with settle_time as a
        # hard cap. This also delays cleanup until embeds are done.
        if settle_time > 0 and not reuse_seed:
            print(f"\n[1.5/4] Waiting (≤{settle_time:.0f}s) for embeddings to settle...")
            es = await client.wait_until_embedded(tenant_id, max_wait=settle_time, poll_interval=3.0)
            note = "" if es["reason"] == "all_embedded" else "  ⚠ TIMEOUT — recall may be partial"
            print(f"  settle: {es['reason']} embedded={es['embedded']}/{es['total']} "
                  f"enrich_pending={es['enrich_pending']} in {es['elapsed']:.0f}s{note}")

        # Fetch all stored memories once for diagnostics. Skipped in --reuse-seed:
        # the tenant holds the WHOLE seeded corpus (thousands of rows), which is
        # irrelevant to this question's hit@k and would bloat every result row.
        if not reuse_seed:
            all_mems = await client.list_memories(tenant_id)
            result["all_stored_memories"] = [
                {
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "content": m.get("content"),
                    "memory_type": m.get("memory_type"),
                    "status": m.get("status"),
                    "weight": m.get("weight"),
                    "fleet_id": m.get("fleet_id"),
                    "source_uri": m.get("source_uri"),
                    "ts_valid_start": m.get("ts_valid_start"),
                    "entity_links": m.get("entity_links", []),
                    "metadata": {
                        k: v for k, v in (m.get("metadata") or {}).items()
                        if k in ("session_id", "session_idx", "turn_idx", "role", "tags", "summary")
                    },
                }
                for m in all_mems
            ]
            print(f"  Stored memories: {len(result['all_stored_memories'])} total")

        question_date = parse_benchmark_date(test_case.get("question_date"))

        # ==== Phase B — loop per K ====
        # --seed-only: ingest only, skip all search/eval (empty the K-loop).
        if seed_only:
            print("\n[2-4/4] SEED-ONLY — ingest done; skipping search / answer / judge.")
            result["success"] = True

        for idx, k in enumerate([] if seed_only else ks, start=1):
            print(f"\n--- K={k} ({idx}/{len(ks)}) ---")
            k_block: Dict[str, Any] = {
                "requested_top_k": k,
                "memories_retrieved": 0,
                "recalled_memories": [],
                "retrieval_metrics": None,
                "generated_answer": None,
                "model_answer": False,
                "llm_usage": None,
                "errors": [],
                "timing": {"search": 0.0, "answer_generation": 0.0, "evaluation": 0.0},
            }
            per_k_results[str(k)] = k_block

            # Step 2 — search at top_k=k
            print(f"[2/4] Searching MemClaw (top_k={k})...")
            search_start = time.time()
            search_result = await client.search(
                tenant_id=tenant_id,
                query=test_case["question"],
                top_k=k,
                valid_at=question_date.isoformat() if question_date else None,
                diagnostic=diagnostic,
                # Scope recall to this question's fleet (seed/reuse) → only its own ~50
                # sessions, matching per-case isolation. Falls back to --search-fleet.
                # reuse_seed searches a persistent SHARED tenant → must fleet-scope.
                # Plain/June-style isolates via cleanup → search tenant-wide (fleet-
                # scoping a small fleet inside a large shared tenant loses recall; see
                # benchmark-shared-tenant-seed-invalid).
                fleet_ids=([q_fleet] if reuse_seed else ([search_fleet] if search_fleet else None)),
                score_formula=score_formula,
            )
            search_time = time.time() - search_start
            k_block["timing"]["search"] = search_time

            retrieved_memories = search_result.get("memories", [])
            k_block["recalled_memories"] = retrieved_memories
            k_block["memories_retrieved"] = search_result.get("memory_count", len(retrieved_memories))
            if "_error" in search_result:
                k_block["errors"].append(search_result["_error"])

            retrieval_metrics = compute_retrieval_metrics(
                answer_source_uris, retrieved_memories, test_case.get("answer")
            )
            k_block["retrieval_metrics"] = retrieval_metrics
            recall_score = retrieval_metrics.get("recall_at_k")
            recall_display = f"{recall_score:.2f}" if recall_score is not None else "N/A"
            print(f"  Retrieved: {len(retrieved_memories)} ({search_time:.2f}s)  "
                  f"Recall@{k}: {recall_display} "
                  f"({retrieval_metrics['answer_found']}/{retrieval_metrics['answer_total']})")

            # Step 3 + 4 — answer-gen + judge at every K value.
            gen_time = 0.0
            eval_time = 0.0
            generated_answer = ""
            if retrieved_memories and openai_api_key:
                print(f"[3/4] Generating answer (model={answer_model})...")
                gen_start = time.time()
                generated_answer, gen_usage = await generate_answer(
                    question=test_case["question"],
                    memories=retrieved_memories,
                    question_date=test_case.get("question_date"),
                    openai_api_key=openai_api_key,
                    model=answer_model,
                    question_type=question_type,
                )
                gen_time = time.time() - gen_start
                k_block["timing"]["answer_generation"] = gen_time
                k_block["generated_answer"] = generated_answer
                if gen_usage:
                    k_block["llm_usage"] = gen_usage
                preview = generated_answer[:80] + ("..." if len(generated_answer) > 80 else "")
                print(f"  Gen {gen_time:.2f}s: {preview}")

                # Step 4 — LLM judge
                if generated_answer:
                    print(f"[4/4] Evaluating...")
                    eval_start = time.time()
                    is_correct = await evaluate_answer(
                        question=test_case["question"],
                        expected_answer=test_case["answer"],
                        generated_response=generated_answer,
                        question_type=question_type,
                        question_id=question_id,
                        openai_api_key=openai_api_key,
                    )
                    eval_time = time.time() - eval_start
                    k_block["timing"]["evaluation"] = eval_time
                    k_block["model_answer"] = is_correct
                    print(f"  Result: {'CORRECT' if is_correct else 'INCORRECT'} ({eval_time:.2f}s)")
            else:
                k_block["generated_answer"] = ""
                k_block["model_answer"] = False
                print(f"[3/4] Skipped answer generation (no memories or API key)")

            # Record per-K timings
            timing_stats.record_per_k(
                k, search=search_time, answer_gen=gen_time, evaluation=eval_time
            )

        # Mirror headline-K fields back up for backwards compat with old readers.
        # Skipped under --seed-only: no K-loop ran, so per_k_results is empty.
        if not seed_only:
            headline_block = per_k_results[str(headline_k)]
            result["retrieval_metrics"] = headline_block["retrieval_metrics"]
            result["generated_answer"] = headline_block["generated_answer"]
            result["model_answer"] = headline_block["model_answer"]
            result["llm_usage"] = headline_block["llm_usage"]
            result["memories_retrieved"] = headline_block["memories_retrieved"]
            result["recalled_memories"] = headline_block["recalled_memories"]
            result["timing"]["search"] = headline_block["timing"]["search"]
            result["timing"]["answer_generation"] = headline_block["timing"]["answer_generation"]
            result["timing"]["evaluation"] = headline_block["timing"]["evaluation"]
            # Mirror into legacy timing lists (single entry per case, from headline K)
            timing_stats.recall_times.append(headline_block["timing"]["search"])
            timing_stats.evaluation_times.append(headline_block["timing"]["evaluation"])

        result["success"] = True

    except Exception as e:
        result["errors"].append(str(e))
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        test_time = time.time() - test_start
        timing_stats.test_case_times.append(test_time)
        result["timing"]["test_case_total"] = test_time

        # ==== Phase C — once per case ====
        # Never delete in seed/reuse modes — the whole point is a persistent corpus.
        if cleanup and not seed_only and not reuse_seed:
            try:
                await client.delete_tenant_memories(tenant_id)
            except Exception:
                pass

        headline_block = per_k_results.get(str(headline_k)) or {}
        rm = headline_block.get("retrieval_metrics") or result.get("retrieval_metrics") or {}
        r_score = rm.get("recall_at_k")
        metrics_display = f"  R@{headline_k}={r_score:.2f}" if r_score is not None else ""
        print(f"\nTest #{test_id} completed in {test_time:.2f}s - "
              f"{'CORRECT' if result.get('model_answer') else 'INCORRECT'}{metrics_display}")

    return result


# ============================================================================
# Benchmark Orchestrator
# ============================================================================

async def run_benchmark(
    base_url: str,
    api_key: str | None,
    question_type: str,
    count: Optional[int] = None,
    start_index: int = 0,
    data_file: Path = DATA_FILE,
    output_dir: Optional[Path] = None,
    cleanup: bool = True,
    distance_from_answer: Optional[int] = DEFAULT_DISTANCE_FROM_ANSWER,
    seed: Optional[int] = None,
    max_concurrent_tests: int = MAX_CONCURRENT_TEST_CASES,
    top_k: int = 3,
    top_ks: Optional[List[int]] = None,
    diagnostic: bool = False,
    answer_model: str = "gpt-4o-mini",
    settle_time: float = 0.0,
    reuse_seed: bool = False,
    seed_only: bool = False,
    search_fleet: Optional[str] = None,
    score_formula: Optional[int] = None,
) -> Dict[str, Any]:
    """Run benchmark tests against MemClaw and save results.

    If ``top_ks`` is provided, each test case is evaluated at every K value in
    the list (ingest once, search/answer/judge per K).  Otherwise falls back to
    single-K mode at ``top_k``.
    """
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resumable checkpoint: one JSONL line per FINISHED question, flushed as it
    # completes, keyed by the stable LongMemEval ``question_id``. A crash/stop
    # leaves finished questions on disk; a restart with the SAME --output-dir
    # skips them and continues. Per question_type so categories don't collide.
    progress_path = output_dir / f"{question_type}.progress.jsonl"

    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    benchmark_start = time.time()

    # Resolve K values once, here. Downstream always sees a list.
    ks = sorted(set(int(k) for k in (top_ks or [top_k])))
    headline_k = ks[-1]  # largest — drives the progress line

    print("=" * 60)
    print("MEMCLAW LONGMEMEVAL BENCHMARK")
    print("=" * 60)
    print(f"MemClaw URL: {base_url}")
    print(f"Question type: {question_type}")
    print(f"Count: {count or 1}")
    print(f"Start index: {start_index}")
    print(f"Cleanup: {cleanup}")
    print(f"Distance from answer: {distance_from_answer if distance_from_answer is not None else 'all (no filtering)'}")
    print(f"Seed: {seed if seed is not None else 'sequential'}")
    if len(ks) > 1:
        print(f"K values: {ks}  (headline: K={headline_k})")
    else:
        print(f"Top-k: {ks[0]}")
    print(f"Answer model: {answer_model}")
    print(f"Diagnostic: {diagnostic}")
    print(f"Concurrency: {max_concurrent_tests} test cases, {MAX_CONCURRENT_INGESTIONS} ingestions/case")
    print()

    # Health check
    client = MemClawClient(base_url, api_key)
    healthy = await client.health_check()
    if not healthy:
        print(f"ERROR: MemClaw health check failed at {base_url}")
        await client.close()
        return {}
    print(f"MemClaw health: OK")

    # Load test cases
    print(f"\nLoading test cases from {data_file}...")
    test_cases = load_test_cases(data_file, question_type, count, start_index, seed)
    if not test_cases:
        await client.close()
        return {}
    print(f"Loaded {len(test_cases)} test cases")
    print(f"OpenAI judge: {'configured' if openai_api_key else 'NOT CONFIGURED'}")

    # Resume: drop test cases whose question_id is already in the checkpoint.
    # Only fully-finished (success) questions are checkpointed, so hard failures
    # are retried, not skipped. Resumed results are folded back into the final
    # aggregate below so the saved JSON is identical to a fresh run.
    done_results: Dict[str, Dict[str, Any]] = {}
    if progress_path.exists():
        with progress_path.open() as _pf:
            for _line in _pf:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _rec = json.loads(_line)
                except json.JSONDecodeError:
                    continue
                _qid = _rec.get("question_id")
                if _qid:
                    done_results[_qid] = _rec
    if done_results:
        _before = len(test_cases)
        test_cases = [tc for tc in test_cases if tc.get("question_id") not in done_results]
        print(f"Resume: {len(done_results)} question(s) already finished in "
              f"{progress_path.name} — {len(test_cases)}/{_before} remaining.")

    timing_stats = TimingStats()
    completed_count = 0
    correct_count = 0
    lock = asyncio.Lock()

    semaphore = asyncio.Semaphore(max_concurrent_tests)

    async def run_with_sem(idx: int, test_case: Dict) -> Dict:
        nonlocal completed_count, correct_count
        async with semaphore:
            result = await run_single_test_case(
                client=client,
                test_case=test_case,
                test_id=idx,
                openai_api_key=openai_api_key,
                timing_stats=timing_stats,
                question_type=question_type,
                cleanup=cleanup,
                distance_from_answer=distance_from_answer,
                top_k=ks[0],        # for legacy single-K paths
                top_ks=ks,
                diagnostic=diagnostic,
                answer_model=answer_model,
                settle_time=settle_time,
                reuse_seed=reuse_seed,
                seed_only=seed_only,
                search_fleet=search_fleet,
                score_formula=score_formula,
            )

            async with lock:
                completed_count += 1
                # Headline accuracy tracked at the largest K (best-case recall signal).
                headline_block = (result.get("per_k_results") or {}).get(str(headline_k)) or {}
                if headline_block.get("model_answer") is True:
                    correct_count += 1
                print(f"\n[PROGRESS] {completed_count}/{len(test_cases)} completed — "
                      f"K={headline_k} {correct_count}/{completed_count} "
                      f"({100 * correct_count / completed_count:.1f}%)")

                # Checkpoint a FINISHED question (crash-safe + resumable). Only
                # persist completed measurements (success); hard failures are
                # left out so a restart retries them. Inside the lock → no
                # interleaved writes even if -c > 1.
                if result.get("success") and result.get("question_id"):
                    try:
                        with progress_path.open("a") as _pf:
                            _pf.write(json.dumps(result) + "\n")
                            _pf.flush()
                    except Exception as _e:  # never let checkpointing kill the run
                        print(f"  ⚠ checkpoint write failed for {result.get('question_id')}: {_e}")

            return result

    try:
        results_unordered = await asyncio.gather(
            *[run_with_sem(idx, tc) for idx, tc in enumerate(test_cases)],
            return_exceptions=True,
        )
        # Gather preserves order, but handle exceptions
        results = []
        for idx, r in enumerate(results_unordered):
            if isinstance(r, Exception):
                print(f"  Test #{idx} raised exception: {r}")
                results.append({"test_id": idx, "success": False, "errors": [str(r)], "model_answer": False})
            else:
                results.append(r)

    finally:
        await client.close()

    # Fold in questions resumed from the checkpoint so the final aggregate +
    # saved JSON cover the full category, not just this session's remainder.
    if done_results:
        results = list(done_results.values()) + results

    benchmark_time = time.time() - benchmark_start

    # Compute per-K aggregate metrics.
    per_k_summary: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        k_key = str(k)
        k_blocks = [
            (r.get("per_k_results") or {}).get(k_key)
            for r in results
        ]
        k_blocks = [b for b in k_blocks if b is not None]

        recall_scores = [
            b["retrieval_metrics"]["recall_at_k"]
            for b in k_blocks
            if b.get("retrieval_metrics") and b["retrieval_metrics"].get("recall_at_k") is not None
        ]
        avg_recall = statistics.mean(recall_scores) if recall_scores else 0
        perfect = sum(1 for s in recall_scores if s == 1.0)

        # Rank-sensitive metrics (the ones a reranker actually moves) over
        # questions that HAVE a gold answer turn (answer_total > 0).
        gold_rms = [
            b["retrieval_metrics"]
            for b in k_blocks
            if b.get("retrieval_metrics") and b["retrieval_metrics"].get("answer_total", 0) > 0
        ]
        avg_mrr = statistics.mean([rm.get("mrr", 0.0) for rm in gold_rms]) if gold_rms else 0
        hit1 = statistics.mean([1.0 if rm.get("hit_at_1") else 0.0 for rm in gold_rms]) if gold_rms else 0
        hit3 = statistics.mean([1.0 if rm.get("hit_at_3") else 0.0 for rm in gold_rms]) if gold_rms else 0
        found_ranks = [rm["gold_rank"] for rm in gold_rms if rm.get("gold_rank")]
        avg_gold_rank = round(statistics.mean(found_ranks), 2) if found_ranks else None

        # URI-independent content recall (robust to dedup/enrich URI rewrites).
        content_flags = [
            b["retrieval_metrics"]["answer_in_content"]
            for b in k_blocks
            if b.get("retrieval_metrics") and b["retrieval_metrics"].get("answer_in_content") is not None
        ]
        content_recall = (
            round(statistics.mean([1.0 if c else 0.0 for c in content_flags]), 4)
            if content_flags else None
        )

        evaluated = [b for b in k_blocks if b.get("model_answer") is not None]
        correct = sum(1 for b in evaluated if b["model_answer"] is True)
        incorrect = sum(1 for b in evaluated if b["model_answer"] is False)

        per_k_summary[k_key] = {
            "k": k,
            "avg_recall_at_k": round(avg_recall, 4),
            "perfect_recall_count": perfect,
            "total_with_answers": len(recall_scores),
            # rank-sensitive (reranker-visible):
            "avg_mrr": round(avg_mrr, 4),
            "hit_at_1": round(hit1, 4),
            "hit_at_3": round(hit3, 4),
            "avg_gold_rank_when_found": avg_gold_rank,
            # URI-independent content recall:
            "content_recall": content_recall,
            "total_evaluated": len(evaluated),
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": (correct / len(evaluated) * 100) if evaluated else 0,
        }

    # Headline (largest K) mirrored into the legacy top-level keys so existing
    # aggregator scripts keep working.
    headline_summary = per_k_summary[str(headline_k)]

    output_data = {
        "benchmark_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": "MemClaw",
            "memclaw_url": base_url,
            "total_test_cases": len(results),
            "total_benchmark_time": benchmark_time,
            "question_type": question_type,
            "sample_size": len(test_cases),
            "distance_from_answer": distance_from_answer,
            "seed": seed,
            "top_k": headline_k,
            "top_ks": ks,
            "answer_model": answer_model,
            "diagnostic": diagnostic,
        },
        "retrieval_summary": {
            "avg_recall_at_k": headline_summary["avg_recall_at_k"],
            "perfect_recall_count": headline_summary["perfect_recall_count"],
            "total_with_answers": headline_summary["total_with_answers"],
            "k": headline_k,
        },
        "evaluation_summary": {
            "total_evaluated": headline_summary["total_evaluated"],
            "correct": headline_summary["correct"],
            "incorrect": headline_summary["incorrect"],
            "accuracy": headline_summary["accuracy"],
        },
        "per_k_summary": per_k_summary,
        "timing_statistics": timing_stats.get_summary(),
        "results": results,
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{question_type}_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Total time: {benchmark_time:.2f}s ({benchmark_time / 60:.1f} min)")
    print(f"Test cases: {len(results)}")
    print(f"\n--- Per-K summary ---")
    print(f"  {'K':>4}  {'recall':>7}  {'MRR':>6}  {'H@1':>5}  {'H@3':>5}  "
          f"{'rank':>5}  {'cRcl':>6}  {'correct':>7}  {'acc%':>6}")
    for k in ks:
        s = per_k_summary[str(k)]
        gr = s.get("avg_gold_rank_when_found")
        cr = s.get("content_recall")
        print(f"  {k:>4}  {s['avg_recall_at_k']:>7.4f}  {s.get('avg_mrr', 0):>6.3f}  "
              f"{s.get('hit_at_1', 0):>5.2f}  {s.get('hit_at_3', 0):>5.2f}  "
              f"{(f'{gr:.1f}' if gr is not None else '  -'):>5}  "
              f"{(f'{cr:.3f}' if cr is not None else '   -'):>6}  "
              f"{s['correct']:>3}/{s['total_evaluated']:<3}  "
              f"{s['accuracy']:>5.1f}%")
    print(f"\nHeadline (K={headline_k}): accuracy {headline_summary['accuracy']:.1f}%  "
          f"avg_recall {headline_summary['avg_recall_at_k']:.4f}")
    print(f"\nResults saved to: {output_file}")

    return output_data


# ============================================================================
# CLI
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark against MemClaw REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 10 temporal-reasoning tests against local MemClaw
    python runner_memclaw.py -t temporal-reasoning -n 10

    # Run against memclaw.net
    python runner_memclaw.py -t single-session-user -n 20 --url https://memclaw.net --api-key mc_xxx

    # Single test at index 5
    python runner_memclaw.py -t knowledge-update -i 5

    # Keep test data (don't cleanup)
    python runner_memclaw.py -t multi-session -n 5 --no-cleanup
        """,
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"MemClaw API URL (default: {DEFAULT_URL})")
    parser.add_argument("--api-key", default=None, help="MemClaw API key (X-API-Key header)")
    parser.add_argument("--question-type", "-t", default="single-session-user", help="Question type to test")
    parser.add_argument("--count", "-n", type=int, default=None, help="Number of tests to run")
    parser.add_argument("--index", "-i", type=int, default=0, help="Starting index (default: 0)")
    parser.add_argument("--data-path", type=Path, default=DATA_FILE, help=f"Path to longmemeval_s_cleaned.json")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help=f"Output directory (default: {RESULTS_DIR})")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep test data in MemClaw after run")
    parser.add_argument(
        "--distance-from-answer", "-d", type=int, default=DEFAULT_DISTANCE_FROM_ANSWER,
        help="Sessions to keep around answer (0=answer only, omit for default). Use -1 for all sessions.",
    )
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for sampling test cases")
    parser.add_argument(
        "--concurrency", "-c", type=int, default=MAX_CONCURRENT_TEST_CASES,
        help=f"Max concurrent test cases (default: {MAX_CONCURRENT_TEST_CASES}). Use 1 for sequential.",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=3,
        help="Number of memories to retrieve for recall (default: 3). Ignored if --top-ks is set.",
    )
    parser.add_argument(
        "--top-ks", type=str, default=None,
        help="Comma-separated list of K values (e.g. '5,7,10,15,20'). "
             "Each test case is ingested ONCE and evaluated at every K. "
             "Overrides --top-k. Values must be between 1 and 20.",
    )
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="Request diagnostic mode: capture all candidates with score breakdowns and recall prompt",
    )
    parser.add_argument(
        "--answer-model", "-m", type=str, default="gpt-4o-mini",
        help="OpenAI model for answer generation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--settle-time", type=float, default=0.0,
        help="Seconds to wait after ingestion for background tasks (entity extraction, contradiction detection) to complete before searching. Default: 0 (no wait).",
    )
    parser.add_argument(
        "--seed-only", action="store_true",
        help="Seed ONLY: ingest the corpus (cleanup forced OFF), skip search/answer/judge. "
             "Run once to build a persistent tenant, then reuse with --reuse-seed.",
    )
    parser.add_argument(
        "--reuse-seed", action="store_true",
        help="Reuse a pre-seeded tenant: SKIP ingest (cleanup forced OFF), run retrieval+eval only. "
             "The corpus must already be seeded (--seed-only). NOTE: all questions share the seeded "
             "tenant, so retrieval is over the whole category corpus (harder than per-case isolation).",
    )
    parser.add_argument(
        "--search-fleet", type=str, default=os.environ.get("MEMCLAW_BENCH_SEARCH_FLEET") or None,
        help="Scope recall to this fleet (search passes fleet_ids=[value]); default tenant-wide. "
             "Use the category's seed fleet to isolate categories sharing a tenant.",
    )
    parser.add_argument(
        "--score-formula", type=int, choices=[0, 1], default=None,
        help="Per-request ranking-formula A/B: 0=legacy boost stack, 1=unified relevance-dominant. "
             "Omit to use the tenant/server default. Pass with --reuse-seed to compare v0 vs v1.",
    )

    args = parser.parse_args()

    if args.seed_only and args.reuse_seed:
        parser.error("--seed-only and --reuse-seed are mutually exclusive (seed once, then reuse).")

    dist = args.distance_from_answer
    if dist is not None and dist < 0:
        dist = None  # -1 means no filtering

    # Parse --top-ks into a sanitized list. When absent, fall back to --top-k.
    top_ks_list: Optional[List[int]] = None
    if args.top_ks:
        try:
            parsed = [int(x.strip()) for x in args.top_ks.split(",") if x.strip()]
        except ValueError:
            parser.error(f"--top-ks must be a comma-separated list of integers, got: {args.top_ks!r}")
        if not parsed:
            parser.error("--top-ks produced an empty list")
        for k in parsed:
            if not (1 <= k <= 200):
                parser.error(f"--top-ks value {k} out of range; must be between 1 and 200")
        top_ks_list = sorted(set(parsed))

    await run_benchmark(
        base_url=args.url,
        api_key=args.api_key,
        question_type=args.question_type,
        count=args.count,
        start_index=args.index,
        data_file=args.data_path,
        output_dir=args.output_dir,
        cleanup=not args.no_cleanup,
        distance_from_answer=dist,
        seed=args.seed,
        max_concurrent_tests=args.concurrency,
        top_k=args.top_k,
        top_ks=top_ks_list,
        diagnostic=args.diagnostic,
        answer_model=args.answer_model,
        settle_time=args.settle_time,
        reuse_seed=args.reuse_seed,
        seed_only=args.seed_only,
        search_fleet=args.search_fleet,
        score_formula=args.score_formula,
    )


if __name__ == "__main__":
    asyncio.run(main())
