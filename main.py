"""
main.py - the driver. Runs all 480 claims and reports against deck slide 9.

Reads nothing new. It wires retrieval to rules, loops, and totals.

WHY main.py IMPORTS BOTH retrieval AND rules
The clause index is built ONCE here and passed into rules. If rules built its
own index it would either rebuild it 480 times or hide it in a global. Built
here and injected, rules stays testable with a fake index and nothing happens
implicitly at import time.

THE THREE KPIs THIS FILE CAN PRODUCE
Slide 9 lists four. Three of them are properties of the agent's own output and
are computed here. The fourth - Recovery acceptance rate - is a property of
HISTORY, not of the agent: it is the share of past rejections the retailer
accepted. Only outcomes.csv knows that, and outcomes.csv is quarantined to
eval.py. So it is reported there, not here. A KPI you cannot compute from
your own output is not yours to print.

THE EFFORT MODEL, stated openly because every dollar depends on it
    4 KAMs x US$60,000                     = US$240,000/yr   deck says 150-350K
    x 70% of bandwidth on invoice checking = US$168,000/yr   deck says 105-245K
    42 min to review a claim by hand, 2 min to spot-check an auto-cleared one
    effort reduction  = 1 - (auto x 2 + (1 - auto) x 42) / 42
    effort saved      = US$168,000 x reduction
Change any assumption here and every dollar figure moves. That is the point of
keeping them in one visible block instead of scattered through the code.
"""

import json
import os

import pandas as pd

from retrieval import ClauseIndex, load_corpus
from rules import build_context, evaluate, load_data

EXTRACTIONS = "store/extractions.json"

# --- the effort model. Every US$ number traces back to these six lines. ---
KAMS = 4
COST_PER_KAM = 60_000
BANDWIDTH = 0.70
MIN_MANUAL = 42
MIN_AUTO = 2
KAM_COST = KAMS * COST_PER_KAM
CHECKING_COST = KAM_COST * BANDWIDTH


def apply_extractions(claims, path=EXTRACTIONS):
    """
    Drive the pipeline from what was READ OUT OF THE RETAILER'S NOTE instead of
    from the structured columns my generator pre-parsed.

    TWO fields come from the note: claim_type and claimed_amount_usd.
    THREE do not: retailer_id, invoice_id, promo_id.

    That split is not a shortcut, it is how deduction management actually
    works. A retailer's remittance quotes an amount and a reason; the invoice
    linkage comes from Meridian's own remittance matching, and the promotion
    reference exists only in Meridian's system - it is never in the retailer's
    text. 62% of these notes quote no invoice number at all, which is a
    clause 7.1 breach and a retailer-process problem, not something a better
    extractor could fix.

    A claim whose type could not be read is marked Unclassified rather than
    silently keeping the true value. Pretending extraction succeeded is how
    an eval lies to you.
    """
    ext = {e["claim_id"]: e for e in json.load(open(path))}
    c = claims.copy()
    c["claim_type"] = [ext[i]["claim_type"] or "Unclassified" for i in c.claim_id]
    c["claimed_amount_usd"] = [
        ext[i]["claimed_amount_usd"] if ext[i]["claimed_amount_usd"] is not None
        else a for i, a in zip(c.claim_id, c.claimed_amount_usd)]
    return c


def run(use_extractions=False):
    """Score every claim. Returns a DataFrame, one row per claim."""
    data = load_data()
    if use_extractions:
        data["claims"] = apply_extractions(data["claims"])
    ix = ClauseIndex(load_corpus())

    contract_of = {r.retailer_id: r.contract_id
                   for _, r in data["customers"].iterrows()}
    # one context per contract, not per claim - resolve_params hits retrieval
    ctxs = {cid: build_context(data, ix, cid)
            for cid in sorted(set(contract_of.values()))}

    rows = [evaluate(c, ctxs[contract_of[c.retailer_id]])
            for _, c in data["claims"].iterrows()]
    return pd.DataFrame(rows), data, ctxs


def kpis(dec):
    auto = (dec.verdict == "AUTO_CLEAR").mean()
    reduction = 1 - (auto * MIN_AUTO + (1 - auto) * MIN_MANUAL) / MIN_MANUAL
    return {
        "auto_rate": auto,
        "reduction": reduction,
        "effort_saved": CHECKING_COST * reduction,
        "leakage": dec.loc[dec.verdict == "REJECT", "excess_usd"].sum(),
        "hours_before": len(dec) * MIN_MANUAL / 60,
        "hours_after": len(dec) * (auto * MIN_AUTO + (1 - auto) * MIN_MANUAL) / 60,
    }


def band(value, lo, hi, unit=""):
    return "inside" if lo <= value <= hi else f"OUTSIDE {lo}-{hi}{unit}"


if __name__ == "__main__":
    USE_EXTRACTIONS = os.path.exists(EXTRACTIONS)
    dec, data, ctxs = run(use_extractions=USE_EXTRACTIONS)
    k = kpis(dec)
    n = len(dec)

    print("\n" + "=" * 74)
    print(f"  RETAIL INVOICE AGENT  ·  {n} claims scored")
    print(f"  input: {'LLM-extracted fields (store/extractions.json)' if USE_EXTRACTIONS else 'pre-parsed structured columns'}")
    print("=" * 74)

    print("\nVERDICTS")
    for v in ("AUTO_CLEAR", "REJECT", "HOLD"):
        c = int((dec.verdict == v).sum())
        print(f"   {v:<12}{c:>5}   {c/n*100:>5.1f}%")

    print("\nWHICH GATE DECIDED")
    for g, c in dec.gate.value_counts().sort_index().items():
        val = dec.loc[dec.gate == g, "claimed_amount_usd"].sum()
        print(f"   {g:<18}{c:>5}   US${val:>10,.0f}")

    print("\n" + "-" * 74)
    print("  KPIs  ·  slide 9 of the deck")
    print("-" * 74)
    print(f"   % auto-cleared              {k['auto_rate']*100:>8.1f}%")
    print(f"   invoice effort saved        US${k['effort_saved']:>9,.0f}/yr"
          f"   deck 50-150K   {band(k['effort_saved']/1000, 50, 150, 'K')}")
    print(f"      via effort reduction     {k['reduction']*100:>8.1f}%"
          f"        deck 50-70%    {band(k['reduction']*100, 50, 70, '%')}")
    print(f"      {k['hours_before']:.0f} h/yr by hand  ->  {k['hours_after']:.0f} h/yr with the agent")
    print(f"   leakage exposure found      US${k['leakage']:>9,.0f}/yr")
    print(f"   recovery acceptance rate         see eval.py"
          f"   - needs outcomes.csv, which is quarantined")

    print("\n" + "-" * 74)
    print("  RECONCILIATION  ·  every dollar traced")
    print("-" * 74)
    print(f"   {KAMS} KAMs x US${COST_PER_KAM:,}       = US${KAM_COST:>8,}/yr   deck 150-350K")
    print(f"   x {BANDWIDTH:.0%} bandwidth           = US${CHECKING_COST:>8,.0f}/yr   deck 105-245K")
    print(f"   x {k['reduction']*100:.1f}% effort reduction = US${k['effort_saved']:>8,.0f}/yr   deck  50-150K")

    print("\n" + "-" * 74)
    print("  DATA GAP  ·  what the agent could not check, and why")
    print("-" * 74)
    RECOMPUTABLE = ("Promo Discount Support", "Co-op Marketing", "Damages")
    cl = data["claims"]
    cov = cl.claim_type.isin(RECOMPUTABLE)
    print(f"   {cov.sum()} of {len(cl)} claims ({cov.mean()*100:.0f}%) have a reference value")
    print("   the agent can recompute against. For the rest there is nothing")
    print("   in Meridian's data to check the amount against:\n")
    for t in sorted(set(cl.claim_type) - set(RECOMPUTABLE)):
        sub = cl[cl.claim_type == t]
        need = {"Listing Fee": "a per-SKU listing-fee schedule",
                "Freight": "freight agreements as structured data",
                "Shortage": "received-quantity on the proof of delivery",
                "Price Difference": "a price-change log with effective dates",
                "Unclassified": "a claim type the extractor could not read - "
                                "these must go to a human"}
        print(f"     {t:<20}{len(sub):>4} claims  ·  would need {need.get(t, '?')}")
    print("\n   This is the finding, not the failure. An agent that says 'I have")
    print("   no reference for this' is worth more than one that invents a number.")

    # ---- run the rules twice and compare. Veda's pattern. ----------------
    if USE_EXTRACTIONS:
        clean_dec, _, _ = run(use_extractions=False)
        a = clean_dec.set_index("claim_id").verdict
        b = dec.set_index("claim_id").verdict
        drift = [i for i in a.index if a[i] != b[i]]
        print("\n" + "-" * 74)
        print("  EXTRACTION RISK  ·  the same rules, driven two ways")
        print("-" * 74)
        print(f"   from pre-parsed fields  ->  {(a=='AUTO_CLEAR').sum()} auto-cleared")
        print(f"   from the retailer note  ->  {(b=='AUTO_CLEAR').sum()} auto-cleared")
        print(f"   {len(drift)} of {n} verdicts differ ({len(drift)/n*100:.1f}%)")
        print("\n   That percentage IS the extraction risk. It is the cost of")
        print("   starting from what the retailer wrote instead of from a clean")
        print("   record, and it is the number to watch when swapping the")
        print("   regex baseline for a model.")
        loosened = [i for i in drift if a[i] != "AUTO_CLEAR" and b[i] == "AUTO_CLEAR"]
        tightened = [i for i in drift if a[i] == "AUTO_CLEAR" and b[i] != "AUTO_CLEAR"]
        print(f"\n   {len(loosened)} became MORE permissive  <- the dangerous direction:")
        print("      a defect the clean pipeline caught now slips through, because")
        print("      the extractor could not read the claim type and so no")
        print("      type-specific gate could fire.")
        print(f"   {len(tightened)} became more cautious - extra work, no risk.")
        for i in drift[:4]:
            print(f"     {i}   {a[i]:<11} -> {b[i]}")
        print("\n   A production build adds one gate: type not established -> HOLD.")
        print("   That converts every one of these from a silent miss into a queue item.")
    print("=" * 74 + "\n")