# Blood · Customer Feedback & Enquiry Intelligence Agent

Use case 6 of 6 from the Blood AI case study. Scores 900 customer enquiries
across five languages (English, Singlish, Malay, Bahasa Indonesia, Mandarin)
and decides which can be answered from a pre-approved library without a human.

**No LLM.** Keyword lists per language plus a regex for order references. The
Gemini version hit a 20-request-per-day free-tier wall, so this reads the same
900 messages for free, instantly, identically on every run - and eval.py prices
exactly what that costs in accuracy.

## Run it

    pip install -r requirements.txt
    streamlit run app.py

data/ and store/ are committed, so the app runs straight from a clone. The app
computes nothing at demo time - it reads pre-written CSVs.

## Rebuild the outputs from scratch

    python extract.py    # reads the 900 messages   -> store/extractions.json
    python store.py      # runs main + rules, 3 hard rules -> decisions, drafts, reviews
    python eval.py       # opens the answer key     -> store/scorecard.csv
    streamlit run app.py

## The files

| file | what it does |
|---|---|
| extract.py | the only file that reads a customer's words. Keyword lists, five languages. Reports, never decides. |
| rules.py | six gates in severity order. Safety first and unconditional. |
| main.py | the driver. Loops all 900, computes the two KPIs it is allowed to. |
| store.py | the audit trail. Three hard rules; exits non-zero if any breaks. |
| eval.py | the ONLY file allowed to open ground_truth.csv and outcomes.csv. Publishes scorecard.csv. |
| app.py | the inbox. Reads CSVs only; imports neither rules nor main. |

## Headline results

- 43.9% deflected · 395 of 900 answered with no human
- US$11.11 -> US$4.29 per enquiry · US$73,625/yr (deck band 50-100K)
- First response 3.2 -> 1.9 hrs · CSAT 3.74 -> 3.99 (MODELLED, labelled as such)
- Health-complaint recall 100%, 0 auto-answered - and still 0 with the safety
  gate deleted, because "Health complaint" has no row in approved_answers.csv
  in any language. The guardrail is a missing row.
- Topic accuracy 94.8% overall - English 100%, Malay 83%. That 17-point gap is
  the priced case for a language model.
