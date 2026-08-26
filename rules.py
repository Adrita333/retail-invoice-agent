"""
rules.py - the decision engine. Seven gates, in severity order.

The division of labour, and the third RAG lesson of this build:

    RETRIEVAL supplies the PARAMETER.   "ninety (90) days"
    CODE      supplies the PROCEDURE.   claim_date - invoice_date > 90

The engine never asks a model for a verdict. It asks the corpus for a number,
pulls that number out of the quoted clause with a regex, and then decides in
plain Python. If no retrieved clause yields a usable number, the answer is
"not established" and the claim escalates to a human - guideline G5.2. It is
never guessed.

GATES, in order. First one to fire wins.

  1  CLAIM WINDOW        clause 7.2   expired -> REJECT   (outranks everything)
  2  MATERIALITY         A2.3         under threshold -> AUTO_CLEAR
  3  CONTRACT CLAUSE     5.2 / 9.4    no promo ref, over allowance -> REJECT
  4  ARITHMETIC          G4.1/A2.1    claimed > recomputed + tolerance -> REJECT
  5  DUPLICATE           A5.1/A5.2    -> REJECT
  6  SUBSTANTIATION      G3.1         no evidence, >= threshold -> HOLD
  7  AUTHORITY           A3.2         above threshold -> HOLD (never automated)

Gate 1 is first because an expired claim is not payable whatever else is true
about it - checking the arithmetic on a claim that lapsed two months ago is
wasted work.

Gate 3 sits above gate 4 for a reason discovered by running it: both catch a
promo claim with no reference, but gate 4 says "claimed US$1,850 vs recomputed
US$0.00" and gate 3 says clause 5.2. Only one of those is an argument you can
put in front of a retailer. Order the gates by the quality of the citation
they produce, not just by what they catch.

Gate 7 is last because it is not a finding, it is an escalation: the claim may
be perfectly valid and still need a signature.
"""

import re

import pandas as pd

from retrieval import QUERIES, ClauseIndex, cite, load_corpus

DATA = "data"


# ----------------------------------------------------------------------
# PARAMETERS - pulled out of retrieved clause text, never hard-coded
# ----------------------------------------------------------------------
# A2.3 and G3.1 both mention US$150 but say different things. The generic
# substantiation query finds G3.1; materiality needs its own.
Q_MATERIALITY = ("claims below this value are settled without individual "
                 "substantiation review cost of review exceeds the exposure")


def _first(ix, query, contract_id, pattern, must_contain=None, k=4):
    """
    Search, then walk the top-k looking for a clause that actually contains a
    usable number. Top-1 is not trusted: 'payment terms are ninety (90) days
    from invoice date' scores well against a claim-window query and is the
    wrong clause. must_contain is the guard that rejects it.
    """
    for h in ix.search(query, contract_id=contract_id, k=k):
        if must_contain and must_contain.lower() not in h["text"].lower():
            continue
        m = re.search(pattern, h["text"])
        if m:
            return m, h
    return None, None


def resolve_params(ix, contract_id):
    """Everything the gates need, with the clause each number came from."""
    p, src = {}, {}

    m, h = _first(ix, QUERIES["claim_window"], contract_id, r"\((\d+)\)\s+days",
                  must_contain="submitted within")
    p["window_days"], src["window_days"] = (int(m.group(1)), h) if m else (None, None)

    m, h = _first(ix, QUERIES["damages"], contract_id, r"\((\d+(?:\.\d+)?)%\)",
                  must_contain="damages allowance")
    p["damages_pct"], src["damages_pct"] = (float(m.group(1)) / 100, h) if m else (None, None)

    m, h = _first(ix, QUERIES["tolerance"], contract_id, r"\((\d+)%\).*?US\$(\d+)",
                  must_contain="tolerance")
    if m:
        p["tol_pct"], p["tol_abs"] = int(m.group(1)) / 100, float(m.group(2))
        src["tolerance"] = h
    else:
        p["tol_pct"], p["tol_abs"], src["tolerance"] = None, None, None

    m, h = _first(ix, Q_MATERIALITY, contract_id, r"US\$([\d,]+)",
                  must_contain="settled without")
    p["materiality"], src["materiality"] = (float(m.group(1).replace(",", "")), h) if m else (None, None)

    m, h = _first(ix, QUERIES["authority"], contract_id, r"above US\$([\d,]+)",
                  must_contain="approval")
    p["authority"], src["authority"] = (float(m.group(1).replace(",", "")), h) if m else (None, None)

    p["_src"] = src
    return p


# ----------------------------------------------------------------------
def load_data(d=DATA):
    r = {n: pd.read_csv(f"{d}/{n}.csv", keep_default_na=False)
         for n in ["claims", "customers", "invoices", "shipments",
                   "trade_promotions", "price_list"]}
    for c in ["claimed_amount_usd"]:
        r["claims"][c] = pd.to_numeric(r["claims"][c])
    return r


def recompute(claim, ctx):
    """
    Independent recomputation of what Meridian actually owes - guideline G4.1.

    Returns (entitled, basis) or (None, reason) when there is no reference
    value in the data to recompute against. That second case is not a
    failure of the agent. It is a gap in Meridian's master data, and the agent
    saying so is more useful than the agent inventing a number.
    """
    t = claim.claim_type
    inv = ctx["inv_by_id"].get(claim.invoice_id)

    if t in ("Promo Discount Support", "Co-op Marketing"):
        if not claim.promo_id:
            return 0.0, "no promotion reference"
        pr = ctx["promo_by_id"].get(claim.promo_id)
        if pr is None:
            return 0.0, "promotion reference not found"
        return float(pr["agreed_support_usd"]), f"agreed support on {claim.promo_id}"

    if t == "Damages":
        if inv is None:
            return None, "invoice not found"
        sh = ctx["ship_by_inv"].get(claim.invoice_id)
        if sh is None:
            return None, "no shipment record"
        dmg_value = float(sh["units_damaged"]) * float(inv["unit_price_usd"])
        cap = float(inv["gross_invoice_usd"]) * ctx["params"]["damages_pct"]
        return round(min(dmg_value, cap), 2), (
            f"{int(sh['units_damaged'])} damaged units x US${float(inv['unit_price_usd']):.2f}, "
            f"capped at {ctx['params']['damages_pct']*100:.1f}% of invoice")

    # Listing Fee, Freight, Shortage, Price Difference: nothing in the data
    # to recompute against. See the data-gap report in main.py.
    return None, f"no reference value available for {t}"


# ----------------------------------------------------------------------
def evaluate(claim, ctx):
    """Run the gates in order. First to fire decides. Always returns a dict."""
    p = ctx["params"]
    src = p["_src"]
    amt = float(claim.claimed_amount_usd)
    inv = ctx["inv_by_id"].get(claim.invoice_id)

    def out(verdict, gate, reason, citation=None, entitled=None, excess=0.0,
            basis=""):
        return {"claim_id": claim.claim_id, "retailer_id": claim.retailer_id,
                "contract_id": ctx["contract_id"], "claim_type": claim.claim_type,
                "claimed_amount_usd": round(amt, 2), "verdict": verdict,
                "gate": gate, "reason": reason,
                "citation": citation or "", "entitled_amount_usd": entitled,
                "excess_usd": round(excess, 2), "basis": basis}

    # --- GATE 1: claim window. Outranks everything. -------------------
    if p["window_days"] is None:
        return out("HOLD", "1-window", "claim window not established in the corpus")
    if inv is None:
        return out("HOLD", "0-data", "invoice not found")
    age = (pd.Timestamp(claim.claim_date) - pd.Timestamp(inv["invoice_date"])).days
    if age > p["window_days"]:
        return out("REJECT", "1-window",
                   f"submitted {age} days after invoice, window is {p['window_days']} days",
                   cite(src["window_days"]), entitled=0.0, excess=amt)

    # --- GATE 2: materiality. Cheaper to pay than to review. ----------
    if p["materiality"] is not None and amt < p["materiality"]:
        return out("AUTO_CLEAR", "2-materiality",
                   f"US${amt:,.2f} below the US${p['materiality']:,.0f} review threshold",
                   cite(src["materiality"]), entitled=amt)

    entitled, basis = recompute(claim, ctx)
    recomputable = entitled is not None

    # --- GATE 3: contract clause (the RAG-heavy gate) -----------------
    # This runs BEFORE the arithmetic gate, deliberately. Both would catch a
    # promo claim with no reference - but gate 4 rejects it as "claimed
    # US$1,850 vs recomputed US$0.00", while this one rejects it citing
    # clause 5.2. The first is a calculation. The second is an argument you
    # can put in front of a retailer. Same verdict, very different letter.
    if claim.claim_type in ("Promo Discount Support", "Co-op Marketing") \
            and not claim.promo_id:
        return out("REJECT", "3-clause",
                   "promotional support claimed with no pre-approved promotion reference",
                   cite(ctx["promo_clause"]), entitled=0.0, excess=amt)

    if claim.claim_type == "Damages" and recomputable:
        cap = float(inv["gross_invoice_usd"]) * p["damages_pct"]
        tol = max(cap * p["tol_pct"], p["tol_abs"])
        if amt > cap + tol:
            return out("REJECT", "3-clause",
                       f"damages claimed at {amt/float(inv['gross_invoice_usd'])*100:.2f}% "
                       f"of gross invoice, allowance is {p['damages_pct']*100:.1f}%",
                       cite(src["damages_pct"]), entitled=round(cap, 2),
                       excess=amt - cap, basis=basis)

    # --- GATE 4: independent recomputation ----------------------------
    if recomputable:
        tol = max(entitled * p["tol_pct"], p["tol_abs"])
        if amt > entitled + tol:
            return out("REJECT", "4-arithmetic",
                       f"claimed US${amt:,.2f} vs recomputed US${entitled:,.2f} "
                       f"(tolerance US${tol:,.2f})",
                       cite(src["tolerance"]), entitled=entitled,
                       excess=amt - entitled, basis=basis)

    # --- GATE 5: duplicate --------------------------------------------
    key = (claim.retailer_id, claim.invoice_id, claim.claim_type, round(amt, 2))
    first_id, first_date = ctx["dupe_index"].get(key, (None, None))
    if first_id and first_id != claim.claim_id:
        return out("REJECT", "5-duplicate",
                   f"same retailer, invoice, type and amount as {first_id} ({first_date})",
                   cite(ctx["dupe_clause"]), entitled=0.0, excess=amt)

    # --- GATE 6: substantiation ---------------------------------------
    if not claim.supporting_doc_ref:
        return out("HOLD", "6-substantiation",
                   "no supporting document and above the materiality threshold",
                   cite(ctx["subst_clause"]), entitled=entitled, basis=basis)

    # --- GATE 7: authority. Not a finding - an escalation. ------------
    if p["authority"] is not None and amt > p["authority"]:
        return out("HOLD", "7-authority",
                   f"US${amt:,.2f} exceeds the US${p['authority']:,.0f} "
                   f"KAM settlement authority",
                   cite(src["authority"]), entitled=entitled, basis=basis)

    return out("AUTO_CLEAR", "0-clear",
               "passed every gate" + ("" if recomputable else f" ({basis})"),
               entitled=entitled, basis=basis)


# ----------------------------------------------------------------------
def build_context(data, ix, contract_id):
    """Everything a gate needs, resolved once per contract, not per claim."""
    inv = data["invoices"]
    sh = data["shipments"]
    pr = data["trade_promotions"]
    params = resolve_params(ix, contract_id)

    dupe = {}
    c = data["claims"].sort_values(["claim_date", "claim_id"], kind="stable")
    for _, r in c.iterrows():
        key = (r.retailer_id, r.invoice_id, r.claim_type,
               round(float(r.claimed_amount_usd), 2))
        if key not in dupe:
            dupe[key] = (r.claim_id, r.claim_date)

    return {
        "contract_id": contract_id,
        "params": params,
        "inv_by_id": {r.invoice_id: r for _, r in inv.iterrows()},
        "ship_by_inv": {r.invoice_id: r for _, r in sh.iterrows()},
        "promo_by_id": {r.promo_id: r for _, r in pr.iterrows()},
        "dupe_index": dupe,
        "dupe_clause": ix.search(QUERIES["duplicate"], contract_id, k=1)[0],
        "promo_clause": ix.search(QUERIES["promo_backing"], contract_id, k=1)[0],
        "subst_clause": ix.search(QUERIES["substantiation"], contract_id, k=1)[0],
    }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    ix = ClauseIndex(load_corpus())

    print("=" * 76)
    print("PARAMETERS RESOLVED FROM THE CORPUS (not hard-coded)")
    print("=" * 76)
    for cid in ("CONTRACT_ALPHA", "CONTRACT_BETA"):
        p = resolve_params(ix, cid)
        print(f"\n  {cid}")
        print(f"     claim window     {p['window_days']} days"
              f"        <- {p['_src']['window_days']['doc']} cl {p['_src']['window_days']['clause']}")
        print(f"     damages allowance {p['damages_pct']*100:.1f}%"
              f"           <- {p['_src']['damages_pct']['doc']} cl {p['_src']['damages_pct']['clause']}")
        print(f"     tolerance        {p['tol_pct']*100:.0f}% or US${p['tol_abs']:.0f}"
              f"   <- {p['_src']['tolerance']['doc']} cl {p['_src']['tolerance']['clause']}")
        print(f"     materiality      US${p['materiality']:.0f}"
              f"          <- {p['_src']['materiality']['doc']} cl {p['_src']['materiality']['clause']}")
        print(f"     authority        US${p['authority']:,.0f}"
              f"        <- {p['_src']['authority']['doc']} cl {p['_src']['authority']['clause']}")

    print("\n" + "=" * 76)
    print("FIVE CLAIMS, END TO END")
    print("=" * 76)
    cust = {r.retailer_id: r for _, r in data["customers"].iterrows()}
    ctxs = {cid: build_context(data, ix, cid)
            for cid in ("CONTRACT_ALPHA", "CONTRACT_BETA")}
    shown = set()
    for _, claim in data["claims"].iterrows():
        ctx = ctxs[cust[claim.retailer_id].contract_id]
        d = evaluate(claim, ctx)
        if d["gate"] in shown:
            continue
        shown.add(d["gate"])
        print(f"\n  {d['claim_id']}  {d['retailer_id']}  {d['claim_type']}"
              f"  US${d['claimed_amount_usd']:,.2f}")
        print(f"     {d['verdict']:<11} gate {d['gate']}")
        print(f"     reason: {d['reason']}")
        if d["citation"]:
            print(f"     cites : {d['citation'][:130]}...")
        if len(shown) >= 6:
            break
    print("\n" + "=" * 76)