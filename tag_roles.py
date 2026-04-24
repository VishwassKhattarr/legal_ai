import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
CHUNKS_PATH = "data/chunks/chunks.csv"
OUTPUT_PATH = "data/chunks/chunks_tagged.csv"

# ── Improved Rule-based Tagger ───────────────────────────────────────────────

def assign_role(text):
    t = text.lower()

    # ── RPC (Final ruling) — very specific phrases ──
    if any(w in t for w in [
        "appeal is dismissed", "appeal is allowed",
        "bail is granted", "bail is rejected",
        "we hereby order", "the order is set aside",
        "accordingly dismissed", "accordingly allowed",
        "petition is dismissed", "petition is allowed",
        "the conviction is", "acquitted of", "sentence of",
        "is sentenced to", "we allow the appeal",
        "we dismiss the appeal"
    ]):
        return "RPC"

    # ── RATIO (Reasoning) — court's own conclusion logic ──
    if any(w in t for w in [
        "we are of the view that",
        "we are of the opinion that",
        "it is well settled that",
        "the principle that",
        "thus we hold",
        "we therefore hold",
        "for the foregoing reasons",
        "in our considered opinion",
        "we find no merit",
        "we find merit",
        "having regard to the facts",
        "in the light of the above"
    ]):
        return "RATIO"

    # ── ISSUE — legal questions framed by court ──
    if any(w in t for w in [
        "the question that arises",
        "the only question",
        "the short question",
        "the question for consideration",
        "points that arise for consideration",
        "the issue that arises",
        "whether the accused",
        "whether the appellant",
        "the point for determination"
    ]):
        return "ISSUE"

    # ── STA — statute references ──
    if any(w in t for w in [
        "under section", "under article",
        "as per section", "u/s ",
        "section 302", "section 304", "section 376",
        "section 420", "section 498",
        "ipc,", "ipc.", "crpc,", "crpc.",
        "the indian penal code",
        "the constitution of india",
        "the code of criminal procedure"
    ]):
        return "STA"

    # ── PRE_RELIED — precedents court agrees with ──
    if any(w in t for w in [
        "relied upon by us",
        "we place reliance on",
        "as held by this court in",
        "following the ratio in",
        "in view of the decision in",
        "this court in the case of",
        "the judgment in the case of",
        "as laid down in"
    ]):
        return "PRE_RELIED"

    # ── PRE_NOT_RELIED — precedents court rejects ──
    if any(w in t for w in [
        "is not applicable to the facts",
        "cannot be relied upon",
        "is distinguishable",
        "the said judgment is not applicable",
        "does not help the",
        "is of no assistance"
    ]):
        return "PRE_NOT_RELIED"

    # ── ARG_PETITIONER ──
    if any(w in t for w in [
        "learned counsel for the petitioner submitted",
        "learned counsel for the appellant submitted",
        "it is argued on behalf of the petitioner",
        "the petitioner contends",
        "the appellant contends",
        "on behalf of the petitioner it is submitted",
        "mr. counsel for the petitioner"
    ]):
        return "ARG_PETITIONER"

    # ── ARG_RESPONDENT ──
    if any(w in t for w in [
        "learned counsel for the respondent submitted",
        "learned counsel for the state submitted",
        "it is argued on behalf of the respondent",
        "the respondent contends",
        "on behalf of the respondent it is submitted",
        "the state submits",
        "the prosecution submits"
    ]):
        return "ARG_RESPONDENT"

    # ── RLC — lower court ruling ──
    if any(w in t for w in [
        "the trial court held",
        "the high court held",
        "the sessions court held",
        "the lower court held",
        "the tribunal held",
        "the magistrate held",
        "the division bench held",
        "the high court observed"
    ]):
        return "RLC"

    # ── FAC — facts of the case ──
    if any(w in t for w in [
        "fir was registered",
        "fir was filed",
        "first information report",
        "it is alleged that",
        "the complainant stated",
        "the accused was arrested",
        "the incident occurred",
        "on the date of",
        "the deceased was",
        "the prosecution case is"
    ]):
        return "FAC"

    # ── ANALYSIS — court discussion ──
    if any(w in t for w in [
        "the court notes that",
        "the court observes that",
        "upon perusal of the record",
        "on examination of the evidence",
        "after considering the submissions",
        "having considered the material",
        "on a careful reading of"
    ]):
        return "ANALYSIS"

    # ── PREAMBLE — only document headers ──
    if any(w in t for w in [
        "in the supreme court of india",
        "in the high court of",
        "before the hon'ble",
        "criminal appeal no.",
        "civil appeal no.",
        "writ petition no.",
        "special leave petition"
    ]):
        return "PREAMBLE"

    return "NONE"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading chunks...")
    df = pd.read_csv(CHUNKS_PATH)
    print(f"Total chunks: {len(df)}")

    print("\nTagging rhetorical roles...")
    tqdm.pandas()
    df["role"] = df["text"].progress_apply(assign_role)

    print("\nRole distribution:")
    print(df["role"].value_counts())
    print(f"\nNONE percentage: {(df['role']=='NONE').mean()*100:.1f}%")

    os.makedirs("data/chunks", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()