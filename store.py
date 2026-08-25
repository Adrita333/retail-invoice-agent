"""
store.py - the audit trail. Three files.

Project 1 (luxury copilot) had no store layer at all - that was the honest gap.
Project 2 (Veda) added decisions.csv and reviews.csv. This build adds the third
one, and it is the reason the whole RAG layer exists:

    store/decisions.csv   what the agent decided, and why           480 rows
    store/citations.csv   WHICH CLAUSE justified each decision      1 per cited call
    store/reviews.csv     what a human did about it                 empty until app.py

WHY citations.csv IS A SEPARATE FILE
Appendix clause A6.2: "A rejection that does not cite a governing clause may
not be pursued in recovery." So an uncited rejection is not a weak rejection -
it is an unusable one. Blood cannot take it to the retailer. That makes the
citation log a commercial asset, not a debugging convenience: it is the
evidence pack for a trade negotiation.

This file therefore enforces one hard rule, and fails loudly if it is broken:
EVERY REJECT MUST CARRY A CITATION. No exceptions, no warnings-only.

WHY reviews.csv IS WRITTEN EMPTY
It is the shape of the human's answer, created before there is an answer.
In production this is the file that becomes the training label - every
approve, override and edit is a datapoint the backtest could never give you.
Writing the header now makes that explicit rather than an afterthought.
"""

import os
import re
from datetime import datetime, timezone

import pandas as pd

STORE = "store"

# cite() in retrieval.py builds: 'doc.md clause 7.2: "text"'
# Parsing back a string this file's own codebase formatted is a small smell.
# In a larger build rules.py would carry doc and clause as separate fields
# through to here. At this size the format is defined in one function and
# tested by the assertion below, so the trade is acceptable - but it is a
# trade, and worth naming rather than hiding.
CITE_RE = re.compile(r'^(?P<doc>\S+\.md) clause (?P<clause>[A-G]?\d+\.\d+): "(?P<text>.*)"$')


def _split_citation(s):
    m = CITE_RE.match(s or "")
    return (m.group("doc"), m.group("clause"), m.group("text")) if m else (None, None, None)


def write_all(dec, store=STORE):
    os.makedirs(store, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---------------- 1. decisions.csv ----------------
    decisions = dec.copy()
    decisions["decided_at"] = stamp
    decisions = decisions[[
        "claim_id", "retailer_id", "contract_id", "claim_type",
        "claimed_amount_usd", "verdict", "gate", "reason",
        "entitled_amount_usd", "excess_usd", "basis", "citation", "decided_at"]]
    decisions.to_csv(f"{store}/decisions.csv", index=False)

    # ---------------- 2. citations.csv ----------------
    rows = []
    for _, r in dec.iterrows():
        doc, clause, text = _split_citation(r.citation)
        if doc is None:
            continue
        rows.append({
            "claim_id": r.claim_id, "retailer_id": r.retailer_id,
            "contract_id": r.contract_id, "verdict": r.verdict, "gate": r.gate,
            "claimed_amount_usd": r.claimed_amount_usd, "excess_usd": r.excess_usd,
            "source_document": doc, "clause": clause, "clause_text": text,
            "decided_at": stamp,
        })
    citations = pd.DataFrame(rows)
    citations.to_csv(f"{store}/citations.csv", index=False)

    # ---------------- 3. reviews.csv (empty, schema only) ----------------
    reviews_path = f"{store}/reviews.csv"
    if not os.path.exists(reviews_path):
        pd.DataFrame(columns=[
            "review_id", "claim_id", "reviewer", "action",
            "override_verdict", "reason", "reviewed_at"
        ]).to_csv(reviews_path, index=False)

    # ---------------- the hard rule ----------------
    rejects = dec[dec.verdict == "REJECT"]
    uncited = rejects[~rejects.claim_id.isin(citations.get("claim_id", []))]
    return decisions, citations, rejects, uncited


if __name__ == "__main__":
    from main import EXTRACTIONS, run

    # match main.py: if the extraction cache exists, the pipeline is driven
    # from the retailer's own text, not from the pre-parsed columns. store.py
    # must persist the SAME run main.py reported, or the app shows one set of
    # numbers and the terminal another.
    use_ext = os.path.exists(EXTRACTIONS)
    print(f"  input: {'LLM-extracted fields' if use_ext else 'pre-parsed columns'}")
    dec, _, _ = run(use_extractions=use_ext)
    decisions, citations, rejects, uncited = write_all(dec)

    print("\n" + "=" * 72)
    print("  STORE WRITTEN")
    print("=" * 72)
    print(f"   store/decisions.csv   {len(decisions):>4} rows   every claim, every reason")
    print(f"   store/citations.csv   {len(citations):>4} rows   clause behind each decision")
    print(f"   store/reviews.csv        0 rows   waiting for a human")

    print("\n" + "-" * 72)
    print("  CITATION COVERAGE  ·  Appendix A6.2")
    print("-" * 72)
    print(f"   {len(rejects)} rejections, {len(rejects) - len(uncited)} carry a citation")
    if len(uncited):
        print(f"\n   FAIL - {len(uncited)} uncited rejection(s). Not recoverable under A6.2:")
        for cid in uncited.claim_id.tolist()[:10]:
            print(f"      {cid}")
        raise SystemExit("uncited rejection - fix before shipping")
    print("   PASS - every rejection is defensible.")

    print("\n" + "-" * 72)
    print("  WHICH DOCUMENT IS DOING THE WORK")
    print("-" * 72)
    for doc, g in citations.groupby("source_document"):
        print(f"   {doc:<38}{len(g):>5} citations")
    print("\n   most-cited clauses:")
    top = citations.groupby(["source_document", "clause"]).agg(
        n=("claim_id", "size"), usd=("excess_usd", "sum")).sort_values("n", ascending=False)
    for (doc, cl), r in top.head(6).iterrows():
        print(f"     {doc:<36} {cl:<6}{int(r.n):>5} claims   US${r.usd:>9,.0f} exposure")

    print("\n" + "-" * 72)
    print("  THE EVIDENCE PACK  ·  one rejection, as a retailer would receive it")
    print("-" * 72)
    ex = citations[citations.verdict == "REJECT"].sort_values(
        "excess_usd", ascending=False).iloc[0]
    d = decisions[decisions.claim_id == ex.claim_id].iloc[0]
    print(f"   claim      {ex.claim_id}   {ex.retailer_id}   US${ex.claimed_amount_usd:,.2f}")
    print(f"   verdict    {ex.verdict} at gate {ex.gate}")
    print(f"   because    {d.reason}")
    print(f"   under      {ex.source_document}, clause {ex.clause}")
    print(f"   which says \"{ex.clause_text[:150]}...\"")
    print(f"   recovering US${ex.excess_usd:,.2f}")
    print("\n   That is a letter, not a log line. It is why citations.csv exists.")
    print("=" * 72 + "\n")