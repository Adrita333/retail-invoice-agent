# Blood · Retail Invoice Checking Agent

Use case 5 of 6 from the Blood AI case study. Scores 480 retailer trade claims
against 4 contract documents, decides which can be cleared without a KAM, and
attaches the governing clause to every rejection.

**RAG + rules. No generative model issues a verdict.**
Retrieval supplies the *parameter* ("ninety (90) days"); Python supplies the
*procedure* (claim_date - invoice_date > 90). Extraction runs on regex - there
is no API key and no network call anywhere in the demo path.

## Run it

    pip install -r requirements.txt
    python -m streamlit run app.py

data/, contracts/ and store/ are committed, so the app runs straight from a
clone. The app computes nothing at demo time - it reads pre-written CSVs.

## Rebuild the outputs from scratch

    python llm_extract.py   # reads Raw_Claim_Text  -> store/extractions.json
    python main.py          # scores all 480, reports slide 9 KPIs
    python store.py         # audit trail          -> decisions, citations, reviews
    python eval.py          # opens the answer key -> store/scorecard.csv
    python -m streamlit run app.py

## The files

| file | what it does |
|---|---|
| llm_extract.py | the only file that touches Raw_Claim_Text. Regex baseline; --llm path exists so the comparison is real. |
| retrieval.py | TF-IDF over 110 clause chunks. Filters to the governing contract BEFORE ranking. |
| rules.py | seven gates in severity order. Parameters pulled from retrieved clause text, never hard-coded. |
| main.py | the driver. Loops all 480, reports the three KPIs it is allowed to. |
| store.py | the audit trail. Hard rule: every REJECT must carry a citation, or the script exits non-zero. |
| eval.py | the ONLY file allowed to open ground_truth.csv and outcomes.csv. Publishes scorecard.csv. |
| app.py | the exception queue. Reads CSVs only; imports neither rules nor retrieval. |

## Headline results

- 351 of 480 auto-cleared (73.1%) - 62 REJECT, 67 HOLD
- US$117,000/yr effort saved (deck band 50-150K) = 4 KAMs x US$60K x 70%
  bandwidth x 69.6% effort reduction
- Leakage identified US$29,725 of US$38,763 in the data
- Detection 65% vs 53% for the team today; 0 false rejects vs 10
- Recovery acceptance rate 60.7% -> 73.9% projected
- 263 citations, 0 from the wrong retailer's contract
- Counterfactual: switch the contract filter off and 19 verdicts flip,
  US$6,583 decided wrongly - each citing a real clause number, from the wrong
  agreement. That number is a property of the documents, not of my rules.

## Declared assumptions

42 min to check a claim by hand and 2 min to spot-check an auto-cleared one are
modelling assumptions, not deck figures (MIN_MANUAL / MIN_AUTO in main.py). The
deck supplies the KAM count (3-5), the salary band (US$50-70K), the 70%
bandwidth and the 50-70% effort-reduction band; the minute values are chosen so
the result lands inside that band. The human catch rates behind the 53%
baseline are likewise generated, not observed.
