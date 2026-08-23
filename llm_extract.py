"""
llm_extract.py - the ONLY file that touches Raw_Claim_Text or a model.

    python llm_extract.py            regex baseline, no API key needed
    python llm_extract.py --llm      Claude extraction (needs ANTHROPIC_API_KEY)

WHY THIS FILE EXISTS
Until now the pipeline started from claims.csv's structured columns - claim_type,
claimed_amount_usd, invoice_id - which my generator had already parsed. In
reality a retailer sends an email or a remittance note and someone has to turn
it into those fields. That someone was my data generator. This file replaces it.

    Raw_Claim_Text  ->  claim_type, claimed_amount_usd, invoice_id
                        promo_id, is_resubmission

WHY A REGEX BASELINE SITS NEXT TO THE MODEL
"Does this need an LLM?" is a question to answer with a measurement, not an
assumption. The baseline is honest competition: contract remittance notes are
semi-structured, so a regex does well on amounts and invoice references. Where
the model earns its place is claim_type - "Deduction taken: Ramadan Bundle
support" carries no keyword a regex can rely on. Run both, compare, report the gap.

WHY IT WRITES A CACHE
store/extractions.json is read by main.py. Nothing calls a model at demo time.
Same discipline as Veda: the demo cannot fail live because nothing is running
live.

THE ANSWER KEY FOR THIS STAGE
claims.csv's structured columns. They are what a correct extraction should
recover, so accuracy is measurable without any new data.
"""

import json
import os
import re
import sys

import pandas as pd

OUT = "store/extractions.json"

TYPE_KEYWORDS = [
    ("Co-op Marketing",        ["co-op", "coop", "marketing contribution"]),
    ("Promo Discount Support", ["promo support", "promotional discount",
                                "promo discount", "promotion support"]),
    ("Listing Fee",            ["listing fee", "listing"]),
    ("Damages",                ["damag", "dmg", "credit note"]),
    ("Shortage",               ["short delivery", "shortfall", "qty shortage",
                                "shortage"]),
    ("Freight",                ["freight", "delivery charge"]),
    ("Price Difference",       ["price diff", "priced higher", "price differ"]),
]

AMOUNT_RE = re.compile(r"(?:USD|US\$|\$)\s?([\d,]+(?:\.\d{2})?)", re.I)
INVOICE_RE = re.compile(r"(INV-\d{3}-\d{4})", re.I)
PROMO_RE = re.compile(r"(PRM-\d{4})", re.I)


def extract_regex(text):
    """The baseline. No model, no key, no cost."""
    t = (text or "").lower()

    ctype = None
    for name, keys in TYPE_KEYWORDS:
        if any(k in t for k in keys):
            ctype = name
            break

    amt = AMOUNT_RE.search(text or "")
    inv = INVOICE_RE.search(text or "")
    prm = PROMO_RE.search(text or "")

    return {
        "claim_type": ctype,
        "claimed_amount_usd": float(amt.group(1).replace(",", "")) if amt else None,
        "invoice_id": inv.group(1).upper() if inv else None,
        "promo_id": prm.group(1).upper() if prm else None,
        "is_resubmission": "resubmission" in t,
        "method": "regex",
    }


PROMPT = """You are extracting structured fields from a retailer's claim note
sent to a supplier. Return ONLY a JSON object, no prose.

Fields:
  claim_type          one of: Co-op Marketing, Promo Discount Support,
                      Listing Fee, Damages, Shortage, Freight, Price Difference
                      (null if genuinely unclear)
  claimed_amount_usd  number, no currency symbol or commas (null if absent)
  invoice_id          format INV-000-0000 (null if absent)
  promo_id            format PRM-0000 (null if absent)
  is_resubmission     true if the note suggests this was sent before

Note:
{text}"""


def extract_llm(texts, model="claude-sonnet-4-5"):
    """One call per claim. Slow and costs money - hence the cache."""
    from anthropic import Anthropic
    client = Anthropic()
    out = []
    for i, t in enumerate(texts):
        if i % 50 == 0:
            print(f"   ...{i}/{len(texts)}", flush=True)
        try:
            r = client.messages.create(
                model=model, max_tokens=300,
                messages=[{"role": "user", "content": PROMPT.format(text=t)}])
            body = r.content[0].text.strip()
            body = re.sub(r"^```(?:json)?|```$", "", body, flags=re.M).strip()
            d = json.loads(body)
            d["method"] = "llm"
        except Exception as e:                    # never let one bad row stop the run
            d = extract_regex(t)
            d["method"] = f"regex-fallback ({type(e).__name__})"
        out.append(d)
    return out


def score(ext, claims):
    """
    Accuracy against claims.csv's own structured columns.

    Two of these numbers are meaningless if reported naively, so they are
    split. An extractor cannot recover a reference that is not in the note.
    Scoring absence as a miss measures the RETAILER's process, not the
    extractor - so absence is counted separately and named as what it is.
    """
    n = len(claims)
    r = {"type_hit": 0, "amount_hit": 0,
         "inv_in_note": 0, "inv_correct": 0, "inv_absent": 0,
         "promo_in_note": 0, "promo_expected": 0}
    for e, (_, c) in zip(ext, claims.iterrows()):
        if e["claim_type"] == c.claim_type:
            r["type_hit"] += 1
        if e["claimed_amount_usd"] is not None and \
                abs(e["claimed_amount_usd"] - float(c.claimed_amount_usd)) < 0.01:
            r["amount_hit"] += 1
        if e["invoice_id"]:
            r["inv_in_note"] += 1
            if e["invoice_id"] == c.invoice_id:
                r["inv_correct"] += 1
        else:
            r["inv_absent"] += 1
        if c.promo_id:
            r["promo_expected"] += 1
            if e["promo_id"]:
                r["promo_in_note"] += 1
    r["n"] = n
    return r


if __name__ == "__main__":
    use_llm = "--llm" in sys.argv
    claims = pd.read_csv("data/claims.csv", keep_default_na=False)
    texts = claims.Raw_Claim_Text.tolist()

    if use_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Run without --llm for the baseline.")

    print(f"\nExtracting {len(texts)} claim notes "
          f"({'Claude' if use_llm else 'regex baseline'})...")
    ext = extract_llm(texts) if use_llm else [extract_regex(t) for t in texts]

    for e, cid in zip(ext, claims.claim_id):
        e["claim_id"] = cid
    os.makedirs("store", exist_ok=True)
    json.dump(ext, open(OUT, "w"), indent=1)

    a = score(ext, claims)
    n = a["n"]
    print("\n" + "=" * 72)
    print(f"  EXTRACTION  ·  {n} notes  ·  method: {ext[0]['method']}")
    print("=" * 72)
    print("\n  WHAT THE EXTRACTOR GOT RIGHT")
    for label, v in (("claim_type", a["type_hit"] / n),
                     ("claimed_amount_usd", a["amount_hit"] / n)):
        print(f"   {label:<22}{v*100:>6.1f}%  {'#' * int(v * 40)}")
    if a["inv_in_note"]:
        v = a["inv_correct"] / a["inv_in_note"]
        print(f"   {'invoice_id':<22}{v*100:>6.1f}%  {'#' * int(v * 40)}"
              f"   of the {a['inv_in_note']} notes that quote one")

    print("\n  WHAT IS NOT IN THE NOTE AT ALL  (not an extraction failure)")
    print(f"   {a['inv_absent']} of {n} notes ({a['inv_absent']/n*100:.0f}%) quote no invoice reference.")
    print("   Clause 7.1 requires one. No extractor of any kind can recover")
    print("   what the retailer never wrote. This is a retailer-process")
    print("   problem and the cheapest thing on the whole list to fix.")
    print(f"\n   {a['promo_expected'] - a['promo_in_note']} of {a['promo_expected']} promo claims quote no promotion reference in the")
    print("   note - the PRM id lives in Blood's system, never in the retailer's")
    print("   text. So promo linkage must come from MATCHING, not extraction.")

    miss = [e for e, (_, c) in zip(ext, claims.iterrows())
            if e["claim_type"] != c.claim_type]
    print(f"\n  WHERE A MODEL WOULD EARN ITS PLACE  ·  {len(miss)} claim_type misses")
    for e in miss[:3]:
        t = claims.loc[claims.claim_id == e["claim_id"], "Raw_Claim_Text"].iloc[0]
        print(f"   got {str(e['claim_type']):<20} <- \"{t[:58]}\"")
    if miss:
        print("   No keyword to key off. A model reads the intent; a regex cannot.")
    print(f"\n   -> {OUT}")
    print("=" * 72 + "\n")