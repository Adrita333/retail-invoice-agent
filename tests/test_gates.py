"""
Tests for the claim-scoring gates.

These are invariant tests, not golden-number tests. Asserting that 351 claims
auto-clear would break every time the generator seed or a threshold changed,
and would prove nothing about whether the agent is sound - only that it still
does what it did last week.

What is asserted here are the properties the README claims and a reviewer
would want held: that no verdict is ever issued without a clause behind it,
that no clause is ever borrowed from a contract that does not govern the
retailer, that an unsubstantiated claim is held rather than cleared, and that
the window gate cannot be talked past by a later gate.

Run with:  python -m pytest -q
"""

import pandas as pd
import pytest

import main

# The corpus files every retailer is subject to, whichever contract they hold.
SHARED_SOURCES = {
    "claim_substantiation_guidelines.md",
    "trade_terms_appendix.md",
}

# Contract-specific corpus files, and the contract each one governs.
CONTRACT_SOURCES = {
    "contract_alpha.md": "CONTRACT_ALPHA",
    "contract_beta.md": "CONTRACT_BETA",
}

VERDICTS = {"AUTO_CLEAR", "HOLD", "REJECT"}


@pytest.fixture(scope="session")
def scored():
    """Score every claim once and share it across the module."""
    decisions, data, ctxs = main.run()
    return decisions, data, ctxs


def source_file(citation):
    """The corpus file a citation came from: the first token."""
    return str(citation).split()[0] if str(citation).strip() else ""


# --- coverage -------------------------------------------------------------

def test_every_claim_receives_exactly_one_verdict(scored):
    decisions, data, _ = scored
    assert len(decisions) == len(data["claims"])
    assert decisions.claim_id.is_unique


def test_verdicts_come_from_the_known_set(scored):
    decisions, _, _ = scored
    assert set(decisions.verdict) <= VERDICTS


def test_every_claim_records_which_gate_decided_it(scored):
    decisions, _, _ = scored
    assert decisions.gate.astype(str).str.strip().ne("").all()


# --- the citation contract ------------------------------------------------

def test_no_rejection_is_issued_without_a_clause(scored):
    """
    The README's central claim: a verdict with no clause behind it is not a
    verdict. If this fails, the agent is asserting rather than citing.
    """
    decisions, _, _ = scored
    rejects = decisions[decisions.verdict == "REJECT"]
    uncited = rejects[rejects.citation.astype(str).str.strip() == ""]
    assert uncited.empty, (
        f"{len(uncited)} rejection(s) carry no clause: "
        f"{uncited.claim_id.tolist()[:5]}"
    )


def test_no_citation_comes_from_a_non_governing_contract(scored):
    """
    Filtering to the governing contract happens BEFORE retrieval ranks
    anything. This is the test that would catch it if that order were ever
    reversed - a TF-IDF search over the whole corpus will happily return a
    persuasive clause from the wrong retailer's contract.
    """
    decisions, _, _ = scored
    cited = decisions[decisions.citation.astype(str).str.strip() != ""].copy()
    cited["source"] = cited.citation.map(source_file)

    foreign = cited[
        cited.source.isin(CONTRACT_SOURCES)
        & (cited.source.map(CONTRACT_SOURCES) != cited.contract_id)
    ]
    assert foreign.empty, (
        f"{len(foreign)} citation(s) from a contract that does not govern "
        f"the retailer: {foreign[['claim_id', 'contract_id', 'source']].head().to_dict('records')}"
    )


def test_every_citation_comes_from_a_known_corpus_file(scored):
    decisions, _, _ = scored
    cited = decisions[decisions.citation.astype(str).str.strip() != ""]
    sources = set(cited.citation.map(source_file))
    assert sources <= SHARED_SOURCES | set(CONTRACT_SOURCES)


# --- holds are not silent clears -----------------------------------------

def test_nothing_unsubstantiated_is_auto_cleared(scored):
    """
    A claim the agent cannot substantiate must be HELD for a human, never
    cleared. The failure mode this guards against is a gate quietly falling
    through to the AUTO_CLEAR at the end of evaluate().
    """
    decisions, _, _ = scored
    substantiation_holds = decisions[decisions.gate == "6-substantiation"]
    assert (substantiation_holds.verdict == "HOLD").all()


def test_rejections_record_the_amount_at_stake(scored):
    decisions, _, _ = scored
    rejects = decisions[decisions.verdict == "REJECT"]
    assert (rejects.excess_usd > 0).all(), \
        "a rejection with nothing at stake is a rejection with no reason to exist"


# --- gate ordering --------------------------------------------------------

def test_the_window_gate_outranks_every_later_gate(scored):
    """
    Take a claim that clears today, age it far past the contractual window,
    and confirm the window gate decides it. Gate order is the whole design:
    a claim raised too late is out of time regardless of how good its
    arithmetic is.
    """
    decisions, data, ctxs = scored
    from rules import evaluate

    cleared = decisions[decisions.verdict == "AUTO_CLEAR"].claim_id.iloc[0]
    claims = data["claims"]
    row = claims[claims.claim_id == cleared].iloc[0].copy()

    contract_of = {r.retailer_id: r.contract_id
                   for _, r in data["customers"].iterrows()}
    ctx = ctxs[contract_of[row.retailer_id]]

    row["claim_date"] = str(
        pd.Timestamp(row.claim_date) + pd.Timedelta(days=3650)
    )[:10]

    verdict = evaluate(row, ctx)
    assert verdict["verdict"] == "REJECT"
    assert verdict["gate"] == "1-window"
    assert verdict["citation"], "the window rejection must cite its clause too"


# --- reproducibility ------------------------------------------------------

def test_scoring_the_same_data_twice_gives_the_same_answer(scored):
    """
    The README says every figure reproduces exactly. Retrieval ranks by
    TF-IDF; if any tie-break ever became dependent on dict or set ordering,
    this is where it would surface.
    """
    first, _, _ = scored
    second, _, _ = main.run()
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True),
        second.reset_index(drop=True),
    )
