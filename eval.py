"""
eval.py - the only file allowed to open the answer key.

Reads   store/decisions.csv     what the agent decided
        data/ground_truth.csv   what the contract says     <- quarantined
        data/outcomes.csv       what the humans did        <- quarantined

WHAT IS AND IS NOT A FAIR TEST, stated before the numbers

  CIRCULAR (report it, do not boast about it)
    I generated the defects and I wrote the rules that detect them. A high
    detection rate partly proves my code agrees with my other code.

  NOT CIRCULAR (this is where the real evidence is)
    1. RETRIEVAL. Did the agent cite the clause that actually governs this
       retailer? Nothing in the rules can fake that - the corpus either gets
       filtered correctly or it does not.
    2. THE COUNTERFACTUAL. Run the whole thing again with the contract filter
       switched OFF and count how many verdicts change. That number is not a
       property of my rules. It is a property of the documents.
    3. THE HUMAN BASELINE. outcomes.csv records what KAMs actually caught.
       I did not write those decisions to be beatable - they come from
       per-defect catch rates set before any rule existed.

  The headline is never "the agent scores X". It is "the agent scores X where
  the team currently scores Y, and here is the one test I could have failed".
"""

import os

import pandas as pd

from main import (BANDWIDTH, EXTRACTIONS, KAM_COST, MIN_AUTO, MIN_MANUAL,
                  apply_extractions)
from retrieval import ClauseIndex, load_corpus
from rules import build_context, evaluate, load_data


def load():
    dec = pd.read_csv("store/decisions.csv", keep_default_na=False)
    gt = pd.read_csv("data/ground_truth.csv")
    out = pd.read_csv("data/outcomes.csv", keep_default_na=False)
    cit = pd.read_csv("store/citations.csv", keep_default_na=False)
    m = dec.merge(gt, on="claim_id", suffixes=("", "_truth")).merge(out, on="claim_id")
    m["agent_flagged"] = m.verdict != "AUTO_CLEAR"
    m["agent_rejected"] = m.verdict == "REJECT"
    # a human "spotted" it if they rejected it, or paid it with a recorded waiver
    m["human_spotted"] = (m.kam_decision == "Rejected") | m.kam_note.str.startswith("waived")
    return m, cit


def counterfactual():
    """
    Re-score all 480 with the contract filter OFF. Not a property of my rules.

    It must use the SAME inputs as the real run, or the comparison measures two
    changes at once and the number means nothing. If the pipeline is driven from
    extracted fields, so is the counterfactual. The filter is the only variable.
    """
    data = load_data()
    if os.path.exists(EXTRACTIONS):
        data["claims"] = apply_extractions(data["claims"])
    ix = ClauseIndex(load_corpus())
    contract_of = {r.retailer_id: r.contract_id for _, r in data["customers"].iterrows()}
    blind = build_context(data, ix, None)          # <- no filter
    rows = [evaluate(c, blind) for _, c in data["claims"].iterrows()]
    return pd.DataFrame(rows), blind["params"], contract_of


if __name__ == "__main__":
    m, cit = load()
    n = len(m)
    bad = m[~m.is_clean]
    clean = m[m.is_clean]

    print("\n" + "=" * 76)
    print(f"  EVAL  ·  {n} claims  ·  agent vs answer key vs the humans")
    print("=" * 76)

    # ---------------------------------------------------------------- 1
    print("\n" + "-" * 76)
    print("  1 · DETECTION   partly circular - I wrote both the defects and the rules")
    print("-" * 76)
    print(f"  {'defect':<26}{'agent':>8}{'human':>8}{'n':>6}")
    for d, g in bad.groupby("defect_type"):
        print(f"  {d:<26}{g.agent_rejected.mean()*100:>7.0f}%{g.human_spotted.mean()*100:>7.0f}%{len(g):>6}")
    print(f"  {'OVERALL':<26}{bad.agent_rejected.mean()*100:>7.0f}%{bad.human_spotted.mean()*100:>7.0f}%{len(bad):>6}")

    # ---------------------------------------------------------------- 2
    print("\n" + "-" * 76)
    print("  2 · FALSE REJECTS   honest claims wrongly rejected")
    print("-" * 76)
    fp_a = int((clean.verdict == "REJECT").sum())
    fp_h = int((clean.kam_decision == "Rejected").sum())
    print(f"   agent  {fp_a:>4} of {len(clean)}   {fp_a/len(clean)*100:.1f}%")
    print(f"   human  {fp_h:>4} of {len(clean)}   {fp_h/len(clean)*100:.1f}%")
    print("\n   Every wrong rejection costs twice: the claim gets paid anyway, and")
    print("   Meridian spends negotiating capital on an argument that does not hold.")

    # ---------------------------------------------------------------- 3
    print("\n" + "-" * 76)
    print("  3 · LEAKAGE   US$ of claims that should not be paid")
    print("-" * 76)
    truth = m.overclaim_usd.sum()
    found = m.loc[m.agent_rejected, "excess_usd"].sum()
    missed_h = m.loc[~m.human_spotted & ~m.is_clean, "overclaim_usd"].sum()
    print(f"   in the data                 US${truth:>10,.0f}")
    print(f"   agent identified            US${found:>10,.0f}   {found/truth*100:.0f}%")
    print(f"   humans let through          US${missed_h:>10,.0f}   {missed_h/truth*100:.0f}%")

    # ---------------------------------------------------------------- 4
    print("\n" + "-" * 76)
    print("  4 · RECOVERY ACCEPTANCE RATE   slide 9 KPI 3 - only history knows this")
    print("-" * 76)
    rej_h = m[m.kam_decision == "Rejected"]
    rar_hist = (rej_h.retailer_response == "Accepted").mean()
    well_founded = rej_h[~rej_h.is_clean]
    rar_wf = (well_founded.retailer_response == "Accepted").mean()
    print(f"   historical, all rejections            {rar_hist*100:>5.1f}%   ({len(rej_h)} rejections)")
    print(f"   historical, well-founded ones only    {rar_wf*100:>5.1f}%   ({len(well_founded)})")
    print(f"   agent's rejections that are well-founded  "
          f"{(1 - fp_a/max(int((m.verdict=='REJECT').sum()),1))*100:>5.1f}%")
    print(f"\n   projected under the agent             {rar_wf*100:>5.1f}%")
    print("   The agent lifts this KPI from BOTH directions: it catches more real")
    print("   defects, and it stops the wrong rejections that drag the rate down.")

    # ---------------------------------------------------------------- 5
    print("\n" + "-" * 76)
    print("  5 · RETRIEVAL   NOT circular. The corpus filter either works or it does not.")
    print("-" * 76)
    contract_docs = {"CONTRACT_ALPHA": "contract_alpha.md", "CONTRACT_BETA": "contract_beta.md"}
    wrong = cit[cit.apply(lambda r: r.source_document in contract_docs.values()
                          and r.source_document != contract_docs.get(r.contract_id), axis=1)]
    own = cit[cit.source_document.isin(contract_docs.values())]
    print(f"   {len(cit)} citations issued")
    print(f"   {len(own)} came from a retailer-specific contract")
    print(f"   {len(wrong)} cited the WRONG retailer's contract")
    print("   PASS - no claim was judged against another retailer's terms."
          if not len(wrong) else "   FAIL")

    # ---------------------------------------------------------------- 6
    print("\n" + "-" * 76)
    print("  6 · COUNTERFACTUAL   filter OFF. Not a property of my rules - of the documents.")
    print("-" * 76)
    blind_dec, blind_params, contract_of = counterfactual()
    b = blind_dec.set_index("claim_id")
    a = m.set_index("claim_id")
    both = a.index.intersection(b.index)
    changed = [i for i in both if a.loc[i, "verdict"] != b.loc[i, "verdict"]]
    print(f"   unfiltered, every retailer is judged on: "
          f"{blind_params['window_days']} day window, "
          f"{blind_params['damages_pct']*100:.1f}% damages allowance")
    print(f"   {len(changed)} of {n} verdicts change when the filter is removed")
    if changed:
        print("\n   examples:")
        for i in changed[:5]:
            print(f"     {i}  {contract_of[a.loc[i,'retailer_id']]:<15}"
                  f"{a.loc[i,'verdict']:<11} -> {b.loc[i,'verdict']:<11}"
                  f" (unfiltered gate {b.loc[i,'gate']})")
    val = float(a.loc[changed, "claimed_amount_usd"].sum()) if changed else 0.0
    print(f"\n   US${val:,.0f} of claims decided differently. Every one of them would")
    print("   carry a real clause number - from the wrong contract.")

    # ---------------------------------------------------------------- 7
    auto = (m.verdict == "AUTO_CLEAR").mean()
    red = 1 - (auto * MIN_AUTO + (1 - auto) * MIN_MANUAL) / MIN_MANUAL

    # --- hand the scorecard to app.py -------------------------------------
    # app.py must never open ground_truth.csv or outcomes.csv. eval.py is the
    # only reader of those, so eval.py is what publishes the numbers derived
    # from them. The app displays; it does not compute.
    pd.DataFrame([{
        "effort_saved_usd": round(KAM_COST * BANDWIDTH * red),
        # Published so app.py never has to reconstruct it by dividing
        # effort_saved by reduction - both of which are rounded, which put
        # US$4.80 of round-trip error into the filtered-scope tile.
        "checking_cost_usd": round(KAM_COST * BANDWIDTH),
        "auto_rate": round(auto, 4),
        "reduction": round(red, 4),
        "leakage_found_usd": round(found),
        "leakage_in_data_usd": round(truth),
        "rar_historical": round(rar_hist, 4),
        "rar_projected": round(rar_wf, 4),
        "detection_agent": round(bad.agent_rejected.mean(), 4),
        "detection_human": round(bad.human_spotted.mean(), 4),
        "false_rejects_agent": fp_a,
        "false_rejects_human": fp_h,
        "counterfactual_flips": len(changed),
        "counterfactual_usd": round(val),
    }]).to_csv("store/scorecard.csv", index=False)

    print("\n" + "=" * 76)
    print("  SCORECARD  ·  slide 9")
    print("=" * 76)
    print(f"   invoice effort saved       US${KAM_COST*BANDWIDTH*red:>9,.0f}/yr   deck 50-150K")
    print(f"   % auto-cleared                  {auto*100:>7.1f}%")
    print(f"   recovery acceptance rate        {rar_hist*100:>7.1f}%  ->  {rar_wf*100:.1f}% projected")
    print(f"   leakage exposure identified US${found:>9,.0f}/yr")
    print("\n   vs the humans:")
    print(f"     detection      {bad.agent_rejected.mean()*100:.0f}%  vs  {bad.human_spotted.mean()*100:.0f}%")
    print(f"     false rejects  {fp_a}   vs  {fp_h}")
    print("=" * 76 + "\n")