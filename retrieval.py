"""
retrieval.py - the RAG layer.

Four documents. ~110 clause-level chunks. TF-IDF vectors and a cosine
similarity in numpy. No vector database, no embedding API, no LangChain.

Three decisions worth defending out loud:

1. TF-IDF, not neural embeddings. Contract language is keyword-dense and the
   exact terms matter - "ninety days", "damages allowance", "promotion
   reference". Semantic paraphrase is not the problem here. TF-IDF is also
   deterministic, free, offline and reproducible in a demo.

2. A crude stemmer. Legal prose says "submitted", the query says "submit".
   Without stemming those are different tokens and clause 7.2 never surfaces.
   Twelve lines of suffix-stripping beats installing NLTK for this.

3. THE FILTER RUNS BEFORE THE SEARCH. Both agreements use clause 7.2 and 9.4
   with different terms, and each contains a decoy carrying the other one's
   number. No embedding model fixes that - a better embedding retrieves the
   wrong clause more confidently. The only fix is to narrow the candidate set
   to the governing contract BEFORE computing similarity.

       search(q, contract_id="CONTRACT_ALPHA")   correct
       search(q) then filter the results         too late, already ranked
"""

import glob
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS_DIR = "contracts"

CLAUSE_RE = re.compile(r"^([A-G]?\d+\.\d+)\s+(.+)")
HEAD_RE = re.compile(r"^#{1,3}\s+(.*)")

_SUFFIXES = ("ations", "ation", "ements", "ement", "ingly", "ing", "ions",
             "ion", "ies", "ed", "es", "s")


def stem(w):
    """Deliberately crude. Good enough for contract English."""
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def analyzer(text):
    """Unigrams + bigrams over stemmed tokens."""
    toks = [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def _frontmatter(text):
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return meta, text
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


def load_corpus(corpus_dir=CORPUS_DIR):
    """
    One chunk = one clause, carrying its section heading and its contract_id.

    Line-by-line rather than regex-splitting the whole document, because a
    "## 7. Claim Submission" heading sits on its own line between clauses and
    a split-based parser attaches it to the END of the previous clause. That
    loses the section, and the section is half the retrieval signal.
    """
    chunks = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        meta, body = _frontmatter(open(path).read())
        contract_id = meta.get("contract_id", "SHARED")
        doc = os.path.basename(path)
        section, clause, buf = "", None, []

        def flush():
            if clause and buf:
                text = re.sub(r"\s+", " ", " ".join(buf)).replace("**", "").strip()
                if len(text) >= 30:
                    chunks.append({
                        "doc": doc, "contract_id": contract_id,
                        "section": section, "clause": clause, "text": text,
                        "searchable": f"clause {clause} {section} {text}",
                    })

        for line in body.splitlines():
            line = line.rstrip()
            h = HEAD_RE.match(line)
            if h:
                flush()
                clause, buf = None, []
                section = h.group(1).strip()
                continue
            c = CLAUSE_RE.match(line)
            if c:
                flush()
                clause, buf = c.group(1), [c.group(2)]
            elif line.strip() and clause:
                buf.append(line.strip())      # clause continued on next line
        flush()
    return chunks


class ClauseIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
        self.M = np.asarray(
            self.vec.fit_transform([c["searchable"] for c in chunks]).todense())

    def search(self, query, contract_id=None, k=3):
        """
        contract_id=None  -> search everything. WRONG mode for a real claim.
                             It exists so the demo can show why.
        contract_id=<id>  -> keep that agreement plus SHARED docs, THEN rank.
        """
        if contract_id is None:
            idx = np.arange(len(self.chunks))
        else:
            idx = np.array([i for i, c in enumerate(self.chunks)
                            if c["contract_id"] in (contract_id, "SHARED")])

        q = np.asarray(self.vec.transform([query]).todense())[0]
        scores = self.M[idx] @ q
        out = []
        for o in np.argsort(-scores)[:k]:
            c = dict(self.chunks[idx[o]])
            c["score"] = float(scores[o])
            out.append(c)
        return out


def cite(hit):
    """A citation is document + clause + quoted text. Never a paraphrase."""
    return f'{hit["doc"]} clause {hit["clause"]}: "{hit["text"]}"'


# The queries the rules engine will actually issue. Named, not improvised,
# so retrieval behaviour is reproducible and testable.
QUERIES = {
    "claim_window":   "claims must be submitted within days of the supplier invoice or they lapse",
    "damages":        "damages allowance per cent of gross invoice value",
    "promo_backing":  "promotional support promotion reference pre-approved not payable",
    "duplicate":      "duplicate claim same invoice same amount resubmission",
    "tolerance":      "tolerance five per cent or fifty dollars whichever greater not challenged",
    "authority":      "claims above this value require Commercial Manager approval before settlement",
    "substantiation": "missing evidence claim held not rejected asked once",
}

# NOTE for rules.py: never trust top-1 blindly. Each gate takes top-k and
# pulls the number it needs out of the clause text by regex. If no retrieved
# clause contains a usable number, the answer is "not established" and the
# claim escalates to a human - guideline G5.2. Retrieval informs, rules decide.


# ----------------------------------------------------------------------
if __name__ == "__main__":
    chunks = load_corpus()
    ix = ClauseIndex(chunks)

    print("=" * 74)
    print(f"CORPUS: {len(chunks)} clause chunks from "
          f"{len(set(c['doc'] for c in chunks))} documents")
    for cid in sorted(set(c["contract_id"] for c in chunks)):
        print(f"   {cid:<18}{sum(1 for c in chunks if c['contract_id']==cid):>4} chunks")
    print(f"   TF-IDF matrix: {ix.M.shape[0]} x {ix.M.shape[1]}")

    print("\n" + "=" * 74)
    print("THE TRAP - same question, and the filter decides the answer")
    print("=" * 74)
    q = QUERIES["claim_window"]
    print(f'query: "{q}"\n')

    print("--- NO FILTER (what a naive RAG does) ---")
    for h in ix.search(q, contract_id=None, k=4):
        print(f"  {h['score']:.3f}  {h['doc']:<32} clause {h['clause']:<5} {h['text'][:60]}...")

    for cid in ("CONTRACT_ALPHA", "CONTRACT_BETA"):
        print(f"\n--- FILTERED TO {cid} ---")
        for h in ix.search(q, contract_id=cid, k=2):
            print(f"  {h['score']:.3f}  {h['doc']:<32} clause {h['clause']:<5} {h['text'][:60]}...")

    print("""
  Read that unfiltered block again. Top hit is ALPHA 7.2 - ninety days.
  It wins for EVERY retailer, including the three on CONTRACT_BETA whose
  window is 120 days. So unfiltered, Selera Retail, Bulan Grocers and
  Pasar Digital all get told 90 days, and every claim they file between
  day 91 and day 120 is rejected with a real clause number attached to it.

  Well-formed. Well-cited. Wrong. And a retailer will win that dispute.""")

    print("\n" + "=" * 74)
    print("TWO RETAILERS, SAME QUESTION, TWO DIFFERENT CORRECT ANSWERS")
    print("=" * 74)
    for key in ("claim_window", "damages"):
        print(f"\n  {key}:")
        for cid in ("CONTRACT_ALPHA", "CONTRACT_BETA"):
            top = ix.search(QUERIES[key], contract_id=cid, k=1)[0]
            print(f"    {cid:<16} {top['doc']:<20} cl {top['clause']:<5} {top['text'][:66]}...")

    print("\n" + "=" * 74)
    print("EVERY GATE QUERY RESOLVES TO A CLAUSE")
    print("=" * 74)
    for key, qq in QUERIES.items():
        top = ix.search(qq, contract_id="CONTRACT_ALPHA", k=1)[0]
        print(f"  {key:<16} -> {top['doc']:<34} clause {top['clause']:<5} ({top['score']:.2f})")
    print("=" * 74)