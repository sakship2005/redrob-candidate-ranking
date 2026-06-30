# Redrob Intelligent Candidate Discovery & Ranking

> Submission for the Redrob Intelligent Candidate Discovery & Ranking Challenge
> (India Runs Hackathon / Hack2Skill).
> Ranks 100,000 synthetic candidate profiles against a single Senior AI Engineer
> job description and returns the top 100, with a transparent score and a
> fact-based reason for every candidate.

---

## TL;DR

- **Input:** 100,000 candidate profiles (`candidates.jsonl`) + one job description.
- **Output:** `team_submission.csv` — top 100 candidates with `candidate_id, rank, score, reasoning`.
- **Runs:** CPU-only, fully offline at ranking time, well under 5 minutes, under 16 GB RAM.
- **Approach:** a transparent, weighted, rule-and-retrieval scoring pipeline — **no LLM at ranking time, no GPU, no external API.**

---

## The problem this solves (and the trap it avoids)

The job description is deliberately loaded with AI buzzwords. A naive system that
ranks by **keyword density in the skills list** is easy to fool — and the dataset
includes profiles engineered to exploit exactly that: non-technical people
(HR, marketing, operations, sales) who stuff "AI", "ML", "NLP" into their skills
list and headline while their actual work history is brand design, accounting,
or warehouse operations.

Our central design principle is therefore:

> **Rank people by what they actually *did* (their career history), not by what
> they *listed* (their skills).**

Every weighting and guard in the pipeline follows from that principle.

---

## How a candidate is scored

Each surviving candidate gets a 0–100 score from five weighted signals:

| Signal | Weight | What it measures | Source |
|---|---:|---|---|
| **Career relevance** | 0.35 | How well the candidate's actual career-history text matches the JD (BM25 retrieval). | `career_history` titles + descriptions |
| **Production / ownership** | 0.25 | Whether they *owned and shipped* real systems vs. were *exposed to / assisted with* them. | ownership-language detection + production-context BM25 |
| **Trajectory** | 0.20 | Whether seniority grew over time (without unfairly punishing IC moves or career breaks). | sequence of job titles + durations |
| **Skill match** | 0.15 | Genuine skill overlap with the JD — deliberately the **lowest** weight because skill-keyword matching is the trap. | skills + summary |
| **Hireability** | 0.05 | Real platform behavior: responsiveness, interview completion, offer acceptance. | `redrob_signals` |

A weighted sum produces the final score. Weights are the single source of truth
in the config cell and sum to exactly 1.0 (asserted at load time).

### Two modifiers on top of the weighted sum

- **Consistency penalty** (on skill match): if a candidate self-rates a skill
  "advanced" but the platform's own `skill_assessment_scores` measure them far
  lower, that skill's contribution is discounted. This uses an independent
  measured signal most approaches ignore.

- **Non-technical-profile guard** (on career relevance): if a candidate's actual
  **career-history work** is dominated by non-technical activity (brand design,
  accounting, sales) with essentially no technical work, their career-relevance
  score is crushed — so buzzword-stuffed non-technical profiles cannot reach the
  top 100, regardless of how many AI terms appear in their skills list.

### A gate that runs before any scoring

- **Honeypot / impossibility gate:** candidates with physically impossible
  histories are dropped entirely (never scored, never ranked) — e.g. experience
  that implies starting work years before finishing a degree, overlapping
  full-time roles, or a job ending before the degree even began. The gate flags
  only *genuine impossibilities*; it does **not** drop people merely for not
  listing their early jobs.

### JD-alignment penalties (encoding what the JD explicitly rejects)

The job description is unusually direct about who is **not** a fit. We encode
those explicit statements as score down-weights — and deliberately do **not** add
keyword "boosts" for the JD's "want" signals, because the BM25 career-relevance
score already captures retrieval/ranking *depth* semantically, and keyword-presence
boosts would re-introduce the very keyword trap the challenge penalizes. Each
penalty maps to a line in the JD:

- **All-services careers** — candidates whose entire history is at IT-services /
  consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, …) with
  no product-company experience. (JD: "only worked at consulting firms … not a fit.")
- **Availability** — candidates who haven't logged in for months *and* barely
  respond to recruiters are, for hiring purposes, not actually available, and are
  strongly down-weighted. (JD: "hasn't logged in for 6 months and has a 5% response
  rate is … not actually available. Down-weight them.")
- **Non-India without visa sponsorship** — (JD: "Outside India: case-by-case, but
  we don't sponsor work visas.")
- **Title-chasing / job-hopping** — switching companies roughly every 1.5 years.
  (JD: "switching companies every 1.5 years … we're not a fit.")
- **Long notice period** — softened, not a reject. (JD: "30+ day notice … the bar
  gets higher.")

Penalties only ever *reduce* a score (multiplier capped at ≤ 1.0); they never
inflate one. This keeps the core ranking driven by demonstrated capability, with
the JD's hard "do-not-want" signals applied as a disciplined filter on top.

---

## Why BM25 (and not embeddings)

We initially prototyped semantic embeddings for career-relevance matching, but on
the challenge's CPU-only / 5-minute constraints, embedding 100k profiles did not
fit the time budget. We therefore use **BM25** — the ranking function that powers
production search engines (Elasticsearch, Lucene) — which:

- runs in **seconds** on 100k documents on CPU,
- needs **no model download, no GPU, no internet** at ranking time,
- handles document length and term saturation better than plain TF-IDF.

Because BM25 scores on short text are tiny in absolute terms, we **percentile-
normalize** the retrieval-derived signals across the candidate pool so they span
the full 0–100 range and carry their intended weight in the final sum.

This was a deliberate, measured constraint-driven tradeoff, not a shortcut.

---

## Reasoning is fact-based, never generated

The `reasoning` field for each candidate is assembled directly from the computed
score breakdown — **not** written by an LLM. It therefore cannot hallucinate
claims the data doesn't support, and every phrase maps to a real number. Example:

```
9y exp, currently Staff Machine Learning Engineer; strong role-fit to the JD;
clear production ownership; upward trajectory
```

The pipeline also surfaces honest red flags rather than only positives, e.g.
`self-rated skills exceed measured assessments` appears on candidates whose
self-ratings diverge from their measured assessment scores — even when they still
rank highly on the strength of their track record.

---

## Validation & testing

This pipeline was validated at every stage, not just at the end:

- **Per-signal unit tests** — each sub-score (career relevance, production,
  trajectory, skill match, hireability) was tested against contrasting fixtures
  (e.g. a genuine ML engineer vs. a data engineer who only dabbled; a clear
  owner vs. a candidate hedging with "exposure" / "transitioning") to confirm
  the intended candidate scores higher.

- **Honeypot gate calibration** — an early version over-flagged ~18–33% of
  *realistic* candidates (people who simply didn't list early jobs). We diagnosed
  the cause, rewrote the gate to flag only true impossibilities, and re-verified:
  realistic profiles now pass at ~0% false-drop while genuine impossibilities are
  still caught.

- **Speed / efficiency work** — we measured runtime explicitly and optimized
  toward the 5-minute budget: from per-candidate embedding (over budget) → batched
  embedding → one-text-per-candidate → BM25 retrieval (seconds). Bulk,
  vectorized similarity replaced per-candidate loops.

- **Output-quality validation against real profiles** — we inspected the actual
  profiles behind the top 100 (not just the scores). This is how we discovered
  that buzzword-stuffed non-technical profiles were leaking in despite healthy-
  looking scores, and added the non-technical-profile guard. After the fix, the
  top 100 contains **zero** non-technical leaks — confirmed by a full title-
  distribution audit.

- **Submission-format self-validation** — before writing the CSV, the pipeline
  asserts exactly 100 rows, contiguous ranks 1..100, non-increasing scores,
  non-empty reasoning, and unique candidate IDs. A malformed submission fails
  loudly here rather than after upload.

- **Robust data loading** — the loader streams JSONL, tolerates and reports
  malformed lines instead of crashing, auto-detects `.json` / `.jsonl` / `.csv`
  (and `.gz` variants), and guards against loading a partially-uploaded file.

---

## Repository structure

```
.
├── README.md                  # this file
├── requirements.txt           # dependencies (rank-bm25, numpy, scikit-learn)
├── submission_metadata.yaml   # challenge submission metadata
├── ranker.py                  # standalone, runnable pipeline (CLI)
├── notebook/
│   └── Redrob_Ranker.ipynb    # Colab demo notebook (sandbox link)
└── output/
    └── team_submission.csv    # the top-100 ranking
```

---

## How to run

### Option A — standalone script (recommended for reproduction)

```bash
pip install -r requirements.txt
python ranker.py --candidates candidates.jsonl --jd job_description.txt --out team_submission.csv
```

The script prints timing and run statistics (candidates processed, honeypots
dropped, runtime) and writes the validated CSV.

### Option B — Colab notebook (sandbox demo)

Open `notebook/Redrob_Ranker.ipynb` in Google Colab, upload `candidates.jsonl`
(and your JD), and **Runtime → Run all**. The notebook is the live demo link for
the submission.

---

## Constraints compliance

| Constraint | How we meet it |
|---|---|
| CPU only | BM25 + numpy + lexical rules; no GPU anywhere. |
| No internet at ranking time | No API calls; all scoring is local. |
| Under 5 minutes for 100k | BM25 retrieval + vectorized similarity; measured well within budget. |
| Under 16 GB RAM | Streaming JSONL load; sparse BM25 index. |
| ≤ 10% honeypots in top 100 | Honeypots gated out *before* scoring; top 100 audited to contain none. |

---

## Design philosophy (one line)

Identify the people who would actually succeed in the role by reading the work
they've done — transparently, defensibly, and within hard production constraints —
rather than rewarding whoever packed the most keywords into a profile.