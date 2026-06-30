#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Discovery & Ranking
=================================================
Ranks candidate profiles against a job description and writes the top-N to CSV.

Design principle: rank people by what they actually DID (career history), not by
what they LISTED (skills). CPU-only, offline, fits the 5-minute / 16GB budget.

Usage:
    python ranker.py --candidates candidates.jsonl --jd job_description.txt --out team_submission.csv
"""

import argparse
import csv
import gzip
import json
import re
import time
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi



# ============================================================
# JD-ALIGNMENT PENALTIES
# Encode the JD's explicit "do not want" signals as score down-weights.
# We deliberately do NOT add keyword "boosts" for the JD's "want" signals --
# the BM25 career-relevance score already captures retrieval/ranking DEPTH
# semantically, and adding keyword-presence boosts would re-introduce the
# very keyword trap the challenge penalizes. Every rule below maps to an
# explicit line in the job description.
# ============================================================
_SERVICES_INDUSTRIES = {"IT Services", "Consulting"}
_SERVICES_COMPANIES = {"tcs", "infosys", "wipro", "accenture", "cognizant",
                       "capgemini", "tech mahindra", "hcl", "ltimindtree"}


def jd_penalty_multiplier(cand, now_str="2026-06-30"):
    """Returns (multiplier in [0.3, 1.0], notes). Down-weights candidates the
    JD explicitly says are not a fit. Never boosts above 1.0."""
    mult, notes = 1.0, []
    profile = cand.get("profile", {}) or {}
    sig = cand.get("redrob_signals", {}) or {}
    history = cand.get("career_history", []) or []

    # All-services career (JD: "only worked at consulting firms ... not a fit")
    if history:
        all_ind = all(h.get("industry") in _SERVICES_INDUSTRIES for h in history)
        comps = [(h.get("company", "") or "").lower() for h in history]
        all_co = bool(comps) and all(any(s in c for s in _SERVICES_COMPANIES) for c in comps)
        if all_ind or all_co:
            mult *= 0.45
            notes.append("all-services career")

    # Non-India, no visa sponsorship (JD: "Outside India ... we don't sponsor visas")
    country = profile.get("country", "India")
    if country and country != "India":
        mult *= 0.55
        notes.append("non-India (no visa sponsorship)")

    # Job-hopping / title-chasing (JD: "switching companies every 1.5 years ... not a fit")
    non_cur = [h for h in history if not h.get("is_current")]
    if len(non_cur) >= 3:
        avg = sum(h.get("duration_months", 0) or 0 for h in non_cur) / len(non_cur)
        if avg < 18:
            mult *= 0.85
            notes.append("job-hopping")

    # Long notice -- softened (JD: "30+ day notice ... the bar gets higher", not a reject)
    notice = sig.get("notice_period_days")
    if isinstance(notice, (int, float)) and notice > 90:
        mult *= 0.95
        notes.append("long notice")

    # Availability (JD: "hasn't logged in 6 months + 5% response rate = not available")
    rr = sig.get("recruiter_response_rate")
    rr = rr if isinstance(rr, (int, float)) else 1.0
    la = sig.get("last_active_date")
    try:
        di = (datetime.strptime(now_str, "%Y-%m-%d") - datetime.strptime(la, "%Y-%m-%d")).days
    except Exception:
        di = 0
    if rr < 0.15 and di > 90:
        mult *= 0.5
        notes.append("unavailable")
    elif rr < 0.15:
        mult *= 0.7
        notes.append("low responsiveness")
    elif di > 150:
        mult *= 0.88
        notes.append("inactive")

    return max(0.3, min(1.0, mult)), notes


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "weights": {
        "career_relevance": 0.35,
        "production_signal": 0.25,
        "trajectory": 0.20,
        "skill_match": 0.15,
        "hireability": 0.05,
    },
    "consistency_penalty": {
        "enabled": True,
        "gap_threshold": 20,
        "mild_penalty_multiplier": 0.9,
        "severe_gap_threshold": 35,
        "severe_penalty_multiplier": 0.5,
    },
    "proficiency_scale": {"beginner": 25, "intermediate": 55, "advanced": 85},
    "honeypot_rules": {
        "check_overlapping_roles": True,
        "max_allowed_simultaneous_current_roles": 1,
    },
    "top_n": 100,
    "output_columns": ["candidate_id", "rank", "score", "reasoning"],
}
_ws = round(sum(CONFIG["weights"].values()), 4)
assert _ws == 1.0, f"Weights must sum to 1.0, got {_ws}"

DATE_FMT = "%Y-%m-%d"
PRODUCTION_REFERENCE_TEXT = (
    "Owned and shipped production systems end to end. Deployed and scaled "
    "services serving real traffic. On-call ownership of reliability and "
    "incidents. Took systems from design to production."
)


# ============================================================
# DATA LOADING
# ============================================================
def _open_path(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "rt", encoding="utf-8")


def load_candidates(path):
    base = (path[:-3] if path.endswith(".gz") else path).lower()
    records, bad = [], 0
    if base.endswith(".csv"):
        with _open_path(path) as f:
            records = [dict(r) for r in csv.DictReader(f)]
    elif base.endswith(".json"):
        with _open_path(path) as f:
            parsed = json.load(f)
        records = parsed if isinstance(parsed, list) else [parsed]
    else:  # jsonl (default)
        with _open_path(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
    if bad:
        print(f"  (skipped {bad} malformed line(s))")
    if not records:
        raise RuntimeError(f"{path} parsed to 0 records.")
    return records


def load_jd(path):
    base = (path[:-3] if path.endswith(".gz") else path).lower()
    with _open_path(path) as f:
        text = f.read()
    if base.endswith(".json"):
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = parsed.get("description") or parsed.get("job_description") or text
    return text


# ============================================================
# HONEYPOT GATE (only true impossibilities)
# ============================================================
def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, DATE_FMT)
    except (ValueError, TypeError):
        return None


def run_honeypot_gate(cand, config):
    reasons = []
    profile = cand.get("profile", {}) or {}
    years_exp = profile.get("years_of_experience")
    history = cand.get("career_history", []) or []
    education = cand.get("education", []) or []

    # impossible experience
    if years_exp is not None:
        if years_exp > 60:
            reasons.append("experience exceeds human maximum")
        else:
            degree_ends = [e.get("end_year") for e in education if e.get("end_year")]
            if degree_ends:
                implied_start = datetime.now().year - years_exp
                if implied_start < min(degree_ends) - 6:
                    reasons.append("experience implies working years before degree completion")

    # overlapping non-current roles
    if config["honeypot_rules"].get("check_overlapping_roles"):
        intervals = []
        for h in history:
            if h.get("is_current"):
                continue
            s, e = _parse_date(h.get("start_date")), _parse_date(h.get("end_date"))
            if s and e:
                intervals.append((s, e))
        intervals.sort()
        for i in range(len(intervals) - 1):
            if (intervals[i][1] - intervals[i + 1][0]).days > 31:
                reasons.append("overlapping full-time roles")
                break

    # too many simultaneous current roles
    cur = sum(1 for h in history if h.get("is_current"))
    if cur > config["honeypot_rules"].get("max_allowed_simultaneous_current_roles", 1):
        reasons.append("multiple simultaneous current roles")

    # job ending before degree started
    degree_starts = [e.get("start_year") for e in education if e.get("start_year")]
    if degree_starts and history:
        ends = [datetime.now() if h.get("is_current") else _parse_date(h.get("end_date")) for h in history]
        ends = [e for e in ends if e]
        if ends and max(ends).year < min(degree_starts) - 1:
            reasons.append("job ends before degree starts")

    return (len(reasons) == 0), reasons


# ============================================================
# SUB-SCORE HELPERS
# ============================================================
OWNERSHIP_VERBS = [
    "owned", "own ", "led", "leading", "built and shipped", "shipped", "deployed",
    "architected", "designed and built", "drove", "spearheaded", "on-call", "on call",
    "end to end", "end-to-end", "from scratch", "took ownership", "responsible for",
    "maintained", "scaled",
]
HEDGE_WORDS = [
    "exposure", "exposed to", "assisted", "supported the", "helped the", "familiar with",
    "worked closely with", "adjacent", "some exposure", "involved in", "contributed to",
    "learning", "building competence", "interested in", "transitioning",
]


def _ownership_language_score(text):
    if not text:
        return 0.0
    tl = text.lower()
    own = sum(1 for v in OWNERSHIP_VERBS if v in tl)
    hedge = sum(1 for h in HEDGE_WORDS if h in tl)
    return float(max(0.0, min(100.0, 40 + 15 * (own - 1.5 * hedge))))


SENIORITY_LEVELS = [
    (6, ["chief", "cto", "vp ", "vice president", "head of", "director"]),
    (5, ["principal", "staff", "distinguished"]),
    (4, ["lead", "manager", "founding", "founder"]),
    (3, ["senior", "sr.", "sr "]),
    (2, ["engineer", "developer", "scientist", "analyst", "specialist"]),
    (1, ["junior", "jr.", "jr ", "intern", "trainee", "associate", "graduate"]),
]


def _title_seniority(title):
    if not title:
        return 2
    tl = title.lower()
    for level, kws in SENIORITY_LEVELS:
        if any(k in tl for k in kws):
            return level
    return 2


def score_trajectory(cand):
    history = cand.get("career_history", []) or []
    if not history:
        return 0.0
    ordered = sorted(history, key=lambda r: _parse_date(r.get("start_date")) or datetime.min)
    levels = [_title_seniority(r.get("title")) for r in ordered]
    first, last, peak = levels[0], levels[-1], max(levels)
    net = last - first
    peak_score = (peak - 1) / 5.0 * 100.0
    if net >= 2:
        growth = 100.0
    elif net == 1:
        growth = 80.0
    elif net == 0:
        growth = 40.0 + (last - 1) / 5.0 * 40.0
    else:
        growth = max(35.0, peak_score - 25.0)
    return 0.5 * peak_score + 0.5 * growth


def _consistency_multiplier(cand, config):
    cfg = config["consistency_penalty"]
    if not cfg.get("enabled"):
        return 1.0, []
    scale = config["proficiency_scale"]
    measured = (cand.get("redrob_signals", {}) or {}).get("skill_assessment_scores", {}) or {}
    penalties, ev = [], []
    for s in cand.get("skills", []) or []:
        name, prof = s.get("name"), (s.get("proficiency") or "").lower()
        if name in measured and prof in scale:
            gap = scale[prof] - measured[name]
            if gap >= cfg["severe_gap_threshold"]:
                penalties.append(cfg["severe_penalty_multiplier"])
                ev.append(f"{name}: self={prof} vs measured={measured[name]:.0f}")
            elif gap >= cfg["gap_threshold"]:
                penalties.append(cfg["mild_penalty_multiplier"])
                ev.append(f"{name}: self={prof} vs measured={measured[name]:.0f}")
    return (min(penalties) if penalties else 1.0), ev


def score_hireability(cand):
    sig = cand.get("redrob_signals", {}) or {}
    comp = {"recruiter_response_rate": 0.40, "interview_completion_rate": 0.35, "offer_acceptance_rate": 0.25}
    wsum, wused = 0.0, 0.0
    for f, w in comp.items():
        v = sig.get(f)
        if isinstance(v, (int, float)):
            wsum += v * 100.0 * w
            wused += w
    return 50.0 if wused == 0 else wsum / wused


# Non-technical-profile guard: judge by CAREER WORK, not stuffed skills
TECH_ROLE_TERMS = {
    "machine learning", "deep learning", "ml engineer", "ai engineer", "data scientist",
    "nlp", "computer vision", "neural network", "trained", "fine-tuned", "fine-tuning",
    "deployed model", "inference", "recommendation system", "ranking model", "llm",
    "model training", "ml model", "ml models", "ml pipeline", "data pipeline",
    "built model", "machine learning model", "predictive model", "classifier",
    "research engineer", "applied scientist", "ml infrastructure", "feature engineering",
    "embeddings", "transformer", "pytorch", "tensorflow", "scikit", "mlops",
    "software engineer", "backend engineer", "data engineer", "ml systems",
}
NONTECH_WORK_TERMS = {
    "brand identity", "logo", "packaging design", "creative direction", "typography",
    "month-end close", "financial reporting", "gaap", "tax filings", "statutory compliance",
    "accounting", "warehouse", "fulfillment", "enterprise sales", "arr quota", "quota",
    "stakeholder communication", "business diagnostics", "process re-engineering",
    "customer support", "graphic design", "marketing campaign", "recruitment", "payroll",
}


def _career_only_text(cand):
    parts = []
    for r in cand.get("career_history", []) or []:
        parts.append(r.get("title", "") or "")
        parts.append(r.get("description", "") or "")
    return " ".join(parts).lower()


def _is_nontechnical_profile(cand):
    txt = _career_only_text(cand)
    tech = sum(1 for t in TECH_ROLE_TERMS if t in txt)
    nontech = sum(1 for t in NONTECH_WORK_TERMS if t in txt)
    return nontech > tech and tech <= 1


def _combined_text(cand):
    p = cand.get("profile", {}) or {}
    summary = p.get("summary", "") or ""
    roles = [(r.get("title", "") or "") + ". " + (r.get("description", "") or "")
             for r in cand.get("career_history", []) or []]
    skills = ", ".join(s.get("name", "") for s in (cand.get("skills", []) or []) if s.get("name"))
    return " ".join([summary] + roles + ["Skills: " + skills]).strip()


def _jd_keywords(jd):
    words = re.findall(r"[a-zA-Z][a-zA-Z+#.]{2,}", jd.lower())
    stop = {"the", "and", "for", "are", "but", "with", "you", "your", "our", "who", "that",
            "this", "have", "has", "not", "will", "can", "far", "more", "than", "long",
            "list", "about", "into", "real", "want", "looking", "senior", "team"}
    return set(w for w in words if w not in stop)


def _skill_overlap_score(cand, jd_kw):
    summary = (cand.get("profile", {}) or {}).get("summary", "") or ""
    text = (summary + " " + " ".join(s.get("name", "") for s in (cand.get("skills", []) or []))).lower()
    ck = set(re.findall(r"[a-zA-Z][a-zA-Z+#.]{2,}", text))
    return float(min(100.0, len(ck & jd_kw) / 6.0 * 100.0)) if jd_kw else 0.0


def _percentile_normalize(scores):
    s = np.asarray(scores, dtype=float)
    n = len(s)
    return np.full(n, 50.0) if n <= 1 else s.argsort().argsort() / (n - 1) * 100.0


def _build_reasoning(cand, cr, ps, tr, penalties):
    p = cand.get("profile", {}) or {}
    years, title = p.get("years_of_experience"), p.get("current_title", "")
    parts = []
    if years is not None and title:
        parts.append(f"{years:.0f}y exp, currently {title}")
    elif title:
        parts.append(f"currently {title}")
    parts.append("strong role-fit to the JD" if cr >= 60 else "moderate role-fit" if cr >= 40 else "limited role-fit (background differs from the role)")
    if ps >= 65:
        parts.append("clear production ownership")
    elif ps <= 35:
        parts.append("mostly supporting/exposure-level work")
    if tr >= 70:
        parts.append("upward trajectory")
    elif tr <= 35:
        parts.append("flat trajectory")
    if penalties:
        parts.append("self-rated skills exceed measured assessments")
    r = "; ".join(parts)
    return r[0].upper() + r[1:] if r else "Scored on available signals."


# ============================================================
# PIPELINE
# ============================================================
def rank(candidates, jd_text, config):
    t0 = time.time()

    survivors, disq = [], 0
    for c in candidates:
        if run_honeypot_gate(c, config)[0]:
            survivors.append(c)
        else:
            disq += 1
    print(f"  honeypot gate: {len(survivors):,} survive, {disq:,} dropped ({100*disq/len(candidates):.1f}%)")

    texts = [_combined_text(c) for c in survivors]
    tok = lambda t: [w for w in re.findall(r"[a-zA-Z][a-zA-Z+#.]{1,}", t.lower()) if len(w) > 1]
    index = BM25Okapi([tok(t) for t in texts])
    cr_raw = np.asarray(index.get_scores(tok(jd_text)))
    prod_raw = np.asarray(index.get_scores(tok(PRODUCTION_REFERENCE_TEXT)))
    cr_scores = _percentile_normalize(cr_raw)
    prod_ctx = _percentile_normalize(prod_raw)
    print(f"  BM25 retrieval + normalization done ({time.time()-t0:.1f}s)")

    jd_kw = _jd_keywords(jd_text)
    w = config["weights"]
    scored = []
    for i, c in enumerate(survivors):
        cr = float(cr_scores[i])
        if _is_nontechnical_profile(c):
            cr = min(cr, 10.0)
        hist = c.get("career_history", []) or []
        summ = (c.get("profile", {}) or {}).get("summary", "") or ""
        prod_text = " ".join([summ] + [(r.get("description", "") or "") for r in hist])
        ps = 0.70 * _ownership_language_score(prod_text) + 0.30 * float(prod_ctx[i])
        tr = score_trajectory(c)
        mult, penalties = _consistency_multiplier(c, config)
        sk = _skill_overlap_score(c, jd_kw) * mult
        hi = score_hireability(c)
        final = w["career_relevance"]*cr + w["production_signal"]*ps + w["trajectory"]*tr + w["skill_match"]*sk + w["hireability"]*hi
        jd_mult, jd_notes = jd_penalty_multiplier(c)
        final *= jd_mult
        scored.append({
            "candidate_id": c.get("candidate_id"),
            "score": round(final, 2),
            "reasoning": _build_reasoning(c, cr, ps, tr, penalties),
        })

    scored.sort(key=lambda r: r["score"], reverse=True)
    top = scored[:config["top_n"]]
    for i, r in enumerate(top, 1):
        r["rank"] = i
    print(f"  ranked {len(scored):,} candidates, took top {len(top)} ({time.time()-t0:.1f}s total)")
    return top


def validate(rows, config):
    n = config["top_n"]
    assert len(rows) == n, f"expected {n} rows, got {len(rows)}"
    assert [r["rank"] for r in rows] == list(range(1, n + 1)), "ranks not 1..N"
    for i in range(1, len(rows)):
        assert rows[i]["score"] <= rows[i - 1]["score"], "score increases"
    assert all(r["reasoning"].strip() for r in rows), "empty reasoning"
    assert len({r["candidate_id"] for r in rows}) == n, "duplicate ids"


def write_csv(rows, path, config):
    validate(rows, config)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=config["output_columns"], extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in config["output_columns"]})
    print(f"  wrote {len(rows)} rows to {path} (validated)")


def main():
    ap = argparse.ArgumentParser(description="Redrob candidate ranker")
    ap.add_argument("--candidates", required=True, help="path to candidates .jsonl/.json/.csv (.gz ok)")
    ap.add_argument("--jd", required=True, help="path to job description .txt/.json")
    ap.add_argument("--out", default="team_submission.csv", help="output CSV path")
    args = ap.parse_args()

    print("Loading data...")
    candidates = load_candidates(args.candidates)
    jd_text = load_jd(args.jd)
    print(f"  {len(candidates):,} candidates, JD {len(jd_text)} chars")

    print("Ranking...")
    top = rank(candidates, jd_text, CONFIG)

    print("Writing submission...")
    write_csv(top, args.out, CONFIG)
    print("Done.")


if __name__ == "__main__":
    main()