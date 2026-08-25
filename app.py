"""
app.py - the exception queue a KAM would actually work in.

    streamlit run app.py

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
reader of those and publishes scorecard.csv for this file to display. The
app shows numbers; it does not derive them from the answer key.

THE SCREEN IS AN ARGUMENT
Three header tiles that state the population, then slide 9's four KPIs in
slide 9's order - not a dashboard of everything measurable. Below them, the
claims a human still has to look at, biggest exposure first, each carrying the
clause that justifies it. A reviewer can Approve, Override or Ask for evidence,
and an override demands a reason. That is "AI proposes, teams approve" as a
form field.
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

# ----------------------------------------------------------------- header
st.title("Retail Invoice Checking Intelligence")
st.caption("Blood · use case 5 of 6 · scorecard 4.30, act-now quadrant · "
           f"{len(dec)} claims scored against 4 contract documents")

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    # THE TWO COMPACT CONTROLS SIT AT THE TOP, ON PURPOSE.
    # The multiselects render as tall stacks of chips, which used to push the
    # scope radio and the value slider so far down the sidebar that nobody
    # found them. The controls you reach for most should not be the ones you
    # have to scroll to.
    st.header("KPI scope")
    scope = st.radio("The four tiles cover",
                     [f"All {len(dec)} scored claims", "Current filter only"],
                     index=0, label_visibility="collapsed",
                     help="Current filter scopes the tiles by KAM, retailer and "
                          "claim value only. Verdict and gate are the agent's own "
                          "output, so scoping a rate to them would be circular. "
                          "Recovery acceptance rate always covers all claims.")

    min_val = st.slider("Minimum claim value (US$)", 0, 6000, 0, step=50)

    st.divider()
    st.header("Filter the queue")
    verdicts = st.multiselect("Verdict", ["REJECT", "HOLD", "AUTO_CLEAR"],
                              default=["REJECT", "HOLD"])
    # Slide 9's business problem is KAM bandwidth, so the first question the
    # app has to answer is "which of these are MINE". A KAM does not work a
    # book-wide queue; they work their own accounts.
    kams = st.multiselect("KAM", sorted(dec.kam_owner.unique()),
                          default=sorted(dec.kam_owner.unique()))
    retailers = st.multiselect("Retailer", sorted(dec.retailer_name.unique()),
                               default=list(sorted(dec.retailer_name.unique())))
    gates = st.multiselect("Gate", sorted(dec.gate.unique()),
                           default=sorted(dec.gate.unique()))

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


# ----------------------------------------------------------------- queue
# An EMPTY multiselect means "no filter on this field", not "exclude
# everything". Streamlit hands back [] when you clear a box, and isin([])
# matches zero rows - so clearing one filter silently blanks the whole screen
# mid-demo. Nobody clears a filter meaning "show me nothing".
def keep(col, chosen):
    return col.isin(chosen) if chosen else pd.Series(True, index=col.index)


q = dec[keep(dec.verdict, verdicts) & keep(dec.retailer_name, retailers)
        & keep(dec.kam_owner, kams) & keep(dec.gate, gates)
        & (dec.claimed_amount_usd >= min_val)]
q = q.sort_values("claimed_amount_usd", ascending=False)

# ----------------------------------------------------------------- KPI row
# slide 9's four KPIs, in slide 9's order. Nothing else.
MIN_MANUAL, MIN_AUTO = 42, 2
# 4 KAMs x US$60,000 x 70% bandwidth.
#
# It is NOT derived as effort_saved / reduction. Both of those are rounded on
# the way into the scorecard, so dividing one by the other reconstructs
# US$168,007 instead of US$168,000 - and that US$7 became a US$5 disagreement
# between the two scope settings. Never rebuild an input by dividing two
# rounded outputs.
#
# eval.py publishes it. If the scorecard predates that change, fall back to the
# constants rather than crashing - the app should degrade, not die, when an
# upstream file is one version behind.
KAMS, COST_PER_KAM, BANDWIDTH = 4, 60_000, 0.70
CHECKING_COST = (float(card["checking_cost_usd"])
                 if "checking_cost_usd" in card.index
                 else KAMS * COST_PER_KAM * BANDWIDTH)

# THE KPI POPULATION IS NOT THE QUEUE, AND THIS IS THE IMPORTANT LINE.
#
# Verdict and gate are the agent's OWN OUTPUT. Scoping a rate to a set that was
# selected on that rate is circular. The Verdict filter defaults to REJECT+HOLD,
# so scoping "% auto-cleared" to the queue asks "of the claims that were not
# auto-cleared, how many were auto-cleared?" - definitionally 0%, and 0% then
# drags effort saved to US$0. Filtering to gate 1-window does the same thing,
# because every auto-clear lives in gate 0-clear.
#
# So the tiles scope by properties of the BOOK - KAM, retailer, claim value -
# and ignore the two filters that are properties of the ANSWER.
kpi_pop = dec[keep(dec.retailer_name, retailers) & keep(dec.kam_owner, kams)
              & (dec.claimed_amount_usd >= min_val)]

if scope.startswith("All"):
    t_effort, t_red = card.effort_saved_usd, card.reduction
    t_auto, t_n = card.auto_rate, len(dec)
    t_leak, t_leak_base = card.leakage_found_usd, card.leakage_in_data_usd
else:
    t_n = len(kpi_pop)
    t_auto = (kpi_pop.verdict == "AUTO_CLEAR").mean() if t_n else 0.0
    t_red = 1 - (t_auto * MIN_AUTO + (1 - t_auto) * MIN_MANUAL) / MIN_MANUAL
    t_effort = CHECKING_COST * t_red * (t_n / len(dec))
    t_leak = pd.to_numeric(
        kpi_pop.loc[kpi_pop.verdict == "REJECT", "excess_usd"]).sum()
    t_leak_base = card.leakage_in_data_usd

# ------------------------------------------------------------- header row
# THE ONE LINE THAT STOPS EVERY "BUT I ONLY SEE 129" QUESTION.
# The agent processed all 480. Clearing 351 of them IS the work - it is not
# "only doing 129". The queue shows 129 because the Verdict filter defaults to
# REJECT+HOLD, not because the other 351 were skipped.
#
# These three follow the SAME scope radio as the four tiles below, and are
# computed from the SAME population (kpi_pop). Two populations on one screen is
# how a dashboard starts contradicting itself: the header said 480 while the
# caption underneath said 91.
if scope.startswith("All"):
    h_n, h_auto = len(dec), int((dec.verdict == "AUTO_CLEAR").sum())
    h_label = "Claims scored"
    h_note = "every claim, all 12 months"
else:
    h_n, h_auto = len(kpi_pop), int((kpi_pop.verdict == "AUTO_CLEAR").sum())
    h_label = "Claims in scope"
    h_note = f"of {len(dec)} — KAM, retailer and value filters"
h_exc = h_n - h_auto

h1, h2, h3 = st.columns(3)
h1.metric(h_label, f"{h_n}", h_note)
h2.metric("Cleared by the agent", f"{h_auto}",
          f"{h_auto/h_n*100:.1f}% — no human touched them" if h_n else "—")
h3.metric("Left for a KAM", f"{h_exc}",
          f"{h_exc/h_n*100:.1f}% — the queue below" if h_n else "—")
st.caption(
    (f"Every KPI on this page is measured across all {h_n} scored claims, "
     f"because clearing {h_auto} of them without a human IS the saving. The "
     f"{h_exc} in the queue are what is left over, not the whole job."
     if scope.startswith("All") else
     f"Scoped to {h_n} claims. These three and the four tiles below share one "
     f"population, so they always agree. Verdict and Gate are excluded from it "
     f"on purpose — they are the agent's own output.")
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Invoice effort saved", f"US${t_effort:,.0f}/yr",
          f"{t_red*100:.1f}% less manual effort")
k2.metric("% auto-cleared", f"{t_auto*100:.1f}%",
          f"{round(t_auto*t_n)} of {t_n} claims")
# Recovery acceptance rate NEVER scopes, in either mode. Say so on the tile
# rather than letting it look broken. delta_color="off" drops the green arrow
# so it does not read as a change that just failed to happen.
k3.metric("Recovery acceptance rate", f"{card.rar_projected*100:.1f}%",
          f"{(card.rar_projected-card.rar_historical)*100:+.1f} pts vs today"
          + ("" if scope.startswith("All") else "  ·  whole book, never scoped"),
          delta_color="normal" if scope.startswith("All") else "off")
k4.metric("Leakage exposure identified", f"US${t_leak:,.0f}/yr",
          f"{t_leak/t_leak_base*100:.0f}% of what is there")

if scope.startswith("All"):
    scope_line = (
        f"**Scope: all {len(dec)} scored claims.** The four tiles show the business case "
        f"for the whole book, so they do NOT follow the filters — switch the KPI "
        f"scope to *Current filter* if you want them to. The exception queue below "
        f"always follows every filter, in both modes.  \n")
else:
    scope_line = (
        f"**Scope: {t_n} claims — filtered by KAM, retailer and claim value.** "
        f"The tiles ignore the Verdict and Gate filters even here: those are the "
        f"agent's own output, and a rate measured over a set selected on that rate "
        f"is circular.  \n")

st.caption(
    scope_line
    + "**Recovery acceptance rate never scopes, in either mode.** It comes from "
      "what retailers did about 56 past rejections. Split that by retailer and "
      "you get 7 to 13 each — one changed mind swings the rate 14 points, so a "
      "filtered figure would be noise dressed as a KPI.  \n"
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
    st.caption(f"Biggest exposure first — **this section responds to the filters**. "
               f"{len(q)} of {len(dec)} claims, US\${q.claimed_amount_usd.sum():,.0f} "
               f"of US\${dec.claimed_amount_usd.sum():,.0f}. Everything not listed "
               f"was cleared without a human touching it.")
    # These three DO respond to every filter. The four tiles above are the
    # business case for the whole book; these are the working view. Keeping
    # them visually separate is the point - a KPI that moves when you drag a
    # slider is an exploration tool, not a KPI.
    m1, m2, m3 = st.columns(3)
    m1.metric("Claims in view", f"{len(q)}",
              f"{len(q)/len(dec)*100:.0f}% of {len(dec)}")
    m2.metric("Value in view", f"US${q.claimed_amount_usd.sum():,.0f}",
              f"{q.claimed_amount_usd.sum()/dec.claimed_amount_usd.sum()*100:.0f}% of book")
    m3.metric("Recoverable in view", f"US${q.excess_usd.sum():,.0f}",
              f"{int((q.verdict=='REJECT').sum())} rejections")

    with st.expander("Workload by KAM  ·  the bandwidth story slide 9 is about"):
        w = (dec.assign(exception=dec.verdict != "AUTO_CLEAR")
             .groupby("kam_owner")
             .agg(claims=("claim_id", "size"),
                  exceptions=("exception", "sum"),
                  value_usd=("claimed_amount_usd", "sum")))
        w["auto_cleared_%"] = ((1 - w.exceptions / w.claims) * 100).round(0)
        w["hrs_yr_today"] = (w.claims * MIN_MANUAL / 60).round(0)
        w["hrs_yr_agent"] = ((w.exceptions * MIN_MANUAL
                              + (w.claims - w.exceptions) * MIN_AUTO) / 60).round(0)
        st.dataframe(w.reset_index().rename(columns={"kam_owner": "KAM"}),
                     hide_index=True, width="stretch")
        st.caption("Slide 9 says the KAM team spends ~70% of its bandwidth "
                   "checking invoices. This is where that bandwidth goes, and "
                   "what is left after the agent clears the routine ones.")

    st.dataframe(
        q[["claim_id", "retailer_name", "kam_owner", "claim_type",
           "claimed_amount_usd", "verdict", "gate"]].rename(columns={
               "kam_owner": "KAM",
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

    st.write("**What the retailer sent**")
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
        # Slide 9: the KAM is the person doing the checking. The reviewer is
        # not a separate role - it defaults to the KAM who owns this retailer.
        # In production this comes from SSO, not a text box: a free-text name
        # in an audit trail is a weakness, and worth naming before it is found.
        reviewer = st.text_input("Reviewing KAM", value=r.kam_owner)
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