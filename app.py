"""
app.py - the exception queue a KAM would actually work in.

    streamlit run app.py        (NOT python app.py)

READS ONLY WHAT IS ALREADY ON DISK
    store/decisions.csv   the verdicts
    store/citations.csv   the clause behind each one
    store/scorecard.csv   the KPIs, published by eval.py
    store/reviews.csv     what humans have decided so far
    data/claims.csv       the original claim text
    data/customers.csv    retailer names

It imports neither rules nor retrieval. Nothing is computed while the demo is
running, so nothing can fail while the demo is running. Same discipline as
Veda: no model call at demo time.

It also never opens ground_truth.csv or outcomes.csv. eval.py is the only
reader of those and publishes scorecard.csv for this file to display.

KPI SCOPE
A radio in the sidebar switches the four tiles between all 480 claims and the
current filter. Three of them recompute. Recovery acceptance rate cannot - it
comes from outcomes.csv via eval.py and this file may never open that, so it
always covers all claims and the caption says so.

A NOTE ON DOLLAR SIGNS
Streamlit renders markdown, and two "$" in one string pair up as LaTeX math
delimiters - "US$50-150K ... US$117,000" silently becomes an italic equation.
Every markdown string below with more than one dollar sign escapes them as \\$.
"""

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Blood · Retail Invoice Agent",
                   page_icon="🧾", layout="wide")

STORE, DATA = "store", "data"
REVIEWS = f"{STORE}/reviews.csv"


@st.cache_data
def load():
    dec = pd.read_csv(f"{STORE}/decisions.csv", keep_default_na=False)
    cit = pd.read_csv(f"{STORE}/citations.csv", keep_default_na=False)
    card = pd.read_csv(f"{STORE}/scorecard.csv").iloc[0]
    claims = pd.read_csv(f"{DATA}/claims.csv", keep_default_na=False)
    cust = pd.read_csv(f"{DATA}/customers.csv")
    dec = dec.merge(claims[["claim_id", "claim_date", "invoice_id",
                            "Raw_Claim_Text", "supporting_doc_ref"]], on="claim_id")
    dec = dec.merge(cust[["retailer_id", "retailer_name", "market", "kam_owner"]],
                    on="retailer_id")
    return dec, cit, card


def read_reviews():
    if os.path.exists(REVIEWS):
        return pd.read_csv(REVIEWS, keep_default_na=False)
    return pd.DataFrame(columns=["review_id", "claim_id", "reviewer", "action",
                                 "override_verdict", "reason", "reviewed_at"])


def append_review(row):
    df = read_reviews()
    df.loc[len(df)] = row
    df.to_csv(REVIEWS, index=False)


dec, cit, card = load()
reviews = read_reviews()

RECOMPUTABLE = ("Promo Discount Support", "Co-op Marketing", "Damages")
cov = dec.claim_type.isin(RECOMPUTABLE)

# ----------------------------------------------------------------- header
st.title("Retail Invoice Checking Intelligence")
st.caption("Blood · use case 5 of 6 · scorecard 4.30, act-now quadrant · "
           f"{len(dec)} claims scored against 4 contract documents")

# ----------------------------------------------------------------- sidebar
# defined BEFORE the KPI row, because the tiles now depend on the filter
with st.sidebar:
    st.header("Filter the queue")
    verdicts = st.multiselect("Verdict", ["REJECT", "HOLD", "AUTO_CLEAR"],
                              default=["REJECT", "HOLD"])
    retailers = st.multiselect("Retailer", sorted(dec.retailer_name.unique()),
                               default=list(sorted(dec.retailer_name.unique())))
    gates = st.multiselect("Gate", sorted(dec.gate.unique()),
                           default=sorted(dec.gate.unique()))
    min_val = st.slider("Minimum claim value (US$)", 0, 6000, 0, step=50)

    st.divider()
    # Recovery acceptance rate can never follow the filter - it comes from
    # outcomes.csv via eval.py and this file may not open that. So the scope
    # is an explicit choice and the one tile that cannot follow says so.
    scope = st.radio("KPI scope", ["All claims", "Current filter"], index=0,
                     help="Recovery acceptance rate always covers all claims - "
                          "it is a historical fact, not a property of a subset.")

    st.divider()
    st.subheader("Agent vs the team")
    st.write(f"**Detection** {card.detection_agent*100:.0f}% "
             f"vs {card.detection_human*100:.0f}% today")
    st.write(f"**False rejects** {int(card.false_rejects_agent)} "
             f"vs {int(card.false_rejects_human)} today")
    st.caption("A wrong rejection costs twice — the claim gets paid anyway, and "
               "Blood spends negotiating capital on an argument that collapses.")

    st.divider()
    st.subheader("Why the contract filter matters")
    st.write(f"**{int(card.counterfactual_flips)} verdicts** change if retrieval "
             f"searches all four documents instead of filtering to the governing "
             f"contract first — **US\${card.counterfactual_usd:,.0f}** decided wrongly.")
    st.caption("Every one of them would cite a real clause number. From the "
               "wrong retailer's contract.")

# ----------------------------------------------------------------- filter
q = dec[dec.verdict.isin(verdicts) & dec.retailer_name.isin(retailers)
        & dec.gate.isin(gates) & (dec.claimed_amount_usd >= min_val)]
q = q.sort_values("claimed_amount_usd", ascending=False)

# ----------------------------------------------------------------- KPI row
# slide 9's four KPIs, in slide 9's order. Nothing else.
MIN_MANUAL, MIN_AUTO = 42, 2
# mirrors main.py: 4 KAMs x US$60,000 x 70% bandwidth. Derived from the
# scorecard so the two can never drift apart.
CHECKING_COST = card.effort_saved_usd / card.reduction

if scope == "All claims":
    t_effort, t_red = card.effort_saved_usd, card.reduction
    t_auto, t_n = card.auto_rate, len(dec)
    t_leak, t_leak_base = card.leakage_found_usd, card.leakage_in_data_usd
else:
    t_auto = (q.verdict == "AUTO_CLEAR").mean() if len(q) else 0.0
    t_red = 1 - (t_auto * MIN_AUTO + (1 - t_auto) * MIN_MANUAL) / MIN_MANUAL
    t_effort = CHECKING_COST * t_red * (len(q) / len(dec))
    t_n = len(q)
    t_leak = q.loc[q.verdict == "REJECT", "excess_usd"].sum()
    t_leak_base = card.leakage_in_data_usd

k1, k2, k3, k4 = st.columns(4)
k1.metric("Invoice effort saved", f"US${t_effort:,.0f}/yr",
          f"{t_red*100:.1f}% less manual effort")
k2.metric("% auto-cleared", f"{t_auto*100:.1f}%",
          f"{int(t_auto*t_n)} of {t_n} claims")
k3.metric("Recovery acceptance rate", f"{card.rar_projected*100:.1f}%",
          f"{(card.rar_projected-card.rar_historical)*100:+.1f} pts vs today")
k4.metric("Leakage exposure identified", f"US${t_leak:,.0f}/yr",
          f"{t_leak/t_leak_base*100:.0f}% of what is there")

st.caption(
    (f"**Scope: all {len(dec)} claims.**" if scope == "All claims" else
     f"**Scope: the {len(q)} claims currently filtered.**")
    + " Recovery acceptance rate always covers all claims — it is a historical"
      " fact about past rejections, not a property of a subset.  \n"
    f"Deck slide 9 claims US\$50–150K from productivity. This lands at "
    f"US\${card.effort_saved_usd:,.0f} using the deck's own formula — 4 KAMs × US\$60K × 70% "
    f"bandwidth × {card.reduction*100:.1f}% reduction. The leakage figure is "
    f"**revenue protection on top**, which slide 9 never quantified."
)

st.divider()

left, right = st.columns([1.15, 1])

with left:
    st.subheader(f"Exception queue · {len(q)} claims · "
                 f"US${q.claimed_amount_usd.sum():,.0f}")
    st.caption("Biggest exposure first — **this section always responds to the "
               "filters**. Everything not listed was cleared without a human "
               "touching it.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Claims in view", f"{len(q)}",
              f"{len(q)/len(dec)*100:.0f}% of {len(dec)}")
    m2.metric("Value in view", f"US${q.claimed_amount_usd.sum():,.0f}",
              f"{q.claimed_amount_usd.sum()/dec.claimed_amount_usd.sum()*100:.0f}% of book")
    m3.metric("Recoverable in view", f"US${q.excess_usd.sum():,.0f}",
              f"{int((q.verdict=='REJECT').sum())} rejections")

    st.dataframe(
        q[["claim_id", "retailer_name", "claim_type", "claimed_amount_usd",
           "verdict", "gate"]].rename(columns={
               "claim_id": "Claim", "retailer_name": "Retailer",
               "claim_type": "Type", "claimed_amount_usd": "US$",
               "verdict": "Verdict", "gate": "Gate"}),
        hide_index=True, width="stretch", height=420,
        column_config={"US$": st.column_config.NumberColumn(format="%.2f")})

with right:
    st.subheader("Evidence pack")
    if not len(q):
        st.info("Nothing matches those filters.")
        st.stop()

    pick = st.selectbox("Claim", q.claim_id.tolist(),
                        format_func=lambda c: f"{c} · "
                        f"US${float(q.loc[q.claim_id==c,'claimed_amount_usd'].iloc[0]):,.2f}")
    r = q[q.claim_id == pick].iloc[0]

    tone = {"REJECT": "error", "HOLD": "warning", "AUTO_CLEAR": "success"}[r.verdict]
    getattr(st, tone)(f"**{r.verdict}** at gate {r.gate} — {r.reason}")

    a, b = st.columns(2)
    a.write(f"**Retailer** {r.retailer_name} ({r.market})")
    a.write(f"**Contract** `{r.contract_id}`")
    a.write(f"**KAM** {r.kam_owner}")
    b.write(f"**Claimed** US${r.claimed_amount_usd:,.2f}")
    b.write(f"**Entitled** {'—' if r.entitled_amount_usd == '' else f'US${float(r.entitled_amount_usd):,.2f}'}")
    b.write(f"**Recoverable** US${float(r.excess_usd):,.2f}")

    st.write("**What the retailer sent** — this is the text llm_extract.py read")
    st.code(r.Raw_Claim_Text or "(no text)", language=None)
    if not r.supporting_doc_ref:
        st.caption("⚠ no supporting document attached")

    if r.basis:
        st.write(f"**How the entitlement was recomputed** — {r.basis}")

    st.write("**Governing clause**")
    c = cit[cit.claim_id == pick]
    if len(c):
        c = c.iloc[0]
        st.info(f"**{c.source_document} · clause {c.clause}**\n\n"
                f"> {c.clause_text}")
    else:
        st.warning("No clause cited. Under Appendix A6.2 this cannot be pursued "
                   "in recovery — it must go to a human.")

    # ------------------------------------------------- human decision
    st.divider()
    st.write("**Your decision**")
    prior = reviews[reviews.claim_id == pick]
    if len(prior):
        p = prior.iloc[-1]
        st.success(f"Already reviewed by {p.reviewer}: **{p.action}**"
                   + (f" — {p.reason}" if p.reason else ""))

    with st.form(f"review_{pick}", clear_on_submit=True):
        reviewer = st.text_input("Reviewer", value="K. Tan")
        action = st.radio("Action",
                          ["Approve the agent", "Override", "Request evidence"],
                          horizontal=True)
        override = st.selectbox("Override to", ["", "AUTO_CLEAR", "REJECT", "HOLD"])
        reason = st.text_input("Reason")
        submitted = st.form_submit_button("Record decision")

    if submitted:
        # An override without a reason is how an audit trail becomes useless.
        if action != "Approve the agent" and not reason.strip():
            st.error("A reason is required for anything other than approval.")
        elif action == "Override" and not override:
            st.error("Choose what you are overriding it to.")
        else:
            append_review({
                "review_id": f"REV-{len(reviews):04d}", "claim_id": pick,
                "reviewer": reviewer, "action": action,
                "override_verdict": override, "reason": reason,
                "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
            st.cache_data.clear()
            st.rerun()

# ----------------------------------------------------------------- footer
st.divider()
f1, f2 = st.columns(2)
with f1:
    st.subheader("Accepted unchanged")
    if len(reviews):
        acc = (reviews.action == "Approve the agent").mean()
        st.metric("of reviewed claims", f"{acc*100:.0f}%", f"{len(reviews)} reviewed")
    else:
        st.metric("of reviewed claims", "—", "0 reviewed")
    st.caption("The metric that decides whether this graduates from shadow mode. "
               "Every review written here is a label the backtest could not give us.")
with f2:
    st.subheader("What the agent could not check")
    # computed, not hard-coded - it moved when the extraction layer went in
    st.write(f"**{int(cov.sum())} of {len(dec)} claims ({cov.mean()*100:.0f}%)** have a "
             f"reference value to recompute against. For the rest there is nothing in "
             f"Blood's data to check the amount against:")
    st.caption("Freight → needs freight agreements as structured data · "
               "Listing Fee → a per-SKU fee schedule · "
               "Price Difference → a price-change log · "
               "Shortage → received quantity on the POD · "
               "Unclassified → the extractor could not read the type")