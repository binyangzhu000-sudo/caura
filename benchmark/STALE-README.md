# Running the STALE bench against MemClaw

STALE measures whether a store notices that a fact it holds has been **superseded** — the
implicit-conflict case. `stale_memclaw.py` seeds each STALE sample's haystack into an isolated
fleet, recalls against the probing queries, and emits STALE's own answer-record schema so the
upstream judge can score it.

Everything you need is either in this directory or in the public upstream repo. Nothing is
machine-local.

## 1. Upstream data and judge

The fixtures and the judge are **not ours** — they come from the STALE benchmark (MIT,
Hanxiang Chao):

```bash
git clone https://github.com/icedreamc/STALE.git stale-repo
```

That gives you:

- `stale-repo/data/T1_T2_400_FULL.json` — **400 samples, 200 T1 + 200 T2**
- `stale-repo/STALE/Evaluation/full_eval_performance.py` — the Gemini judge

### Rebuilding the 15-sample T2 slice

The slice used for the 2026-08-05 baseline is simply the **first 15 T2 samples**, in file order:

```python
import json
full = json.load(open("stale-repo/data/T1_T2_400_FULL.json"))
t2 = [x for x in full if x["type"] == "T2"][:15]
json.dump(t2, open("benchmark/stale_T2_15.json", "w"))
```

Verified: those 15 uids are exactly the first 15 `type == "T2"` entries upstream. A T1 slice is the
same expression with `"T1"` — there are 200 to choose from.

## 2. Configuration

`stale_memclaw.py` builds Vertex credentials **at module scope**, so these are required at import
even in `STALE_MODE=seed` where no answer model is ever called. This is the first thing that will
stop you:

| env | required | note |
|---|---|---|
| `VERTEX_SA_FILE` | **yes, at import** | path to a Vertex service-account JSON |
| `VERTEX_PROJECT` | **yes** | GCP project; no default (public repo) |
| `VERTEX_LOCATION` | no | defaults to `global` |
| `MEMCLAW_BENCH_URL` | yes | e.g. `http://localhost:8000` or a staging host |
| `MEMCLAW_BENCH_TENANT` | yes | use a **clean** tenant; residue invalidates the run |
| `ANSWER_MODEL` | no | defaults to `google/gemini-2.5-flash` |
| `STALE_MODE` | no | `seed` \| `run` \| `both` \| `inputs` (default `both`) |
| `STALE_TOP_K` | no | default 20 |

Writes are forced to **strong** mode by the runner (`MEMCLAW_WRITE_MODE=strong`), so the write-side
contradiction path runs inline.

## 3. The mechanism half needs no judge

`superseded` and `new_recalled_dim1` are read from **supersession state** at seed/recall time, not
from judged answers. So the mechanism numbers cost no LLM answer calls and no judge:

```bash
MEMCLAW_BENCH_URL=... MEMCLAW_BENCH_TENANT=<clean-tenant> \
VERTEX_SA_FILE=... VERTEX_PROJECT=... \
STALE_MODE=seed STALE_TAG=<tag> \
python3 -u stale_memclaw.py <api-key> stale_T2_15.json 15 results/<tag>.json
```

Per-sample it prints `created=N superseded=M`. Seeding is the expensive part: ~586 memories per
sample, ~7 min each on a local stack, so 15 samples is roughly 90–110 minutes.

Only the **score** half needs the judge, via `STALE_MODE=run` and then
`full_eval_performance.py`.

## 4. Reading the result honestly

Four things that will produce a wrong number if ignored:

1. **Detection is post-commit async.** Sample immediately after seeding and you under-count
   `superseded`; the bias is one-directional. Wait for the detection queue to drain —
   `queued_ms` on the `path_a_completed` / `path_c_completed` lines should be at zero before you
   read anything.
2. **Count relation-upsert failures first.** `relation_upsert_partial created=N failed=M`
   (added in #1495) means the entity graph is degrading under you; a write-back tally taken while
   that is non-zero is measuring the storage layer, not the product.
3. **Tally the two write-backs separately.** `subject_writeback` and `predicate_writeback` both
   emit `outcome=skipped_ambiguous` and `outcome=set`, while `skipped_no_subject` is subject-only
   and `skipped_no_canonical` is predicate-only. A single `grep -oE "outcome=[a-z_]+"` conflates
   two populations with different denominators — and omits `kept_existing` entirely. Reconcile
   against `SELECT count(*) … WHERE subject_entity_id IS NOT NULL / predicate IS NOT NULL`.
4. **Record the deployed SHA before writing a single memory.** `/api/v1/version` returns the
   release tag, not the SHA, and a tag can predate merges it appears to contain. Use
   `git merge-base --is-ancestor <sha> <tag>`.

## 5. Baseline

| run | date | config | result |
|---|---|---|---|
| T2, 15 samples | **2026-08-05** | judge `google/gemini-2.5-flash`, `num_samples=15` | dim1 4/15 · dim2 0/15 · dim3 7/15 · **overall 11/45 = 0.2444** |

**No code SHA was recorded with that run.** Scores are therefore comparable; a delta is *not*
attributable to any code change. Record the SHA on the next run so the comparison after it is
clean.

A "T1 42%" figure circulates in older notes. There is no artifact behind it that I can find —
treat it as unsourced until someone reproduces it.

## 6. Why the runner is not lint-clean

`ruff check benchmark/` reports ~29 findings in `stale_memclaw.py` /
`runner_memclaw.py` — all cosmetic (`E402` import placement, `E702`/`E401`
statement separators, `F541` f-strings without placeholders). CI does not lint
`benchmark/`, and these are **left alone deliberately**.

The runner is a measurement instrument, and the number in §5 was produced by
this exact code. Reformatting it to satisfy a linter that does not run on this
directory would put a behaviour risk — however small — between the baseline and
the next run, in exchange for tidiness nobody reads. `E402` in particular is
load-bearing here: Vertex credentials are constructed at module scope from
environment read between the imports, so hoisting them is not a no-op.

If this file ever stops being compared against a historical number, clean it
then.
