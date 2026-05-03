#!/usr/bin/env python3
"""
Rigorous audit and filter of narratives.jsonl.

For every record:
  1. Clean the narrative (strip ``` bleed + commentary markers)
  2. Check for residual bleed patterns that cannot appear in a real therapy session
  3. Verify Y=1 records contain an explicit depression signal
  4. Verify Y=0 records do not contain unambiguous depression language
  5. Verify minimum length

Writes only verified records to out/reliable_narratives.jsonl.
Prints exact per-failure counts (records can fail multiple criteria).
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Step 1: Bleed-through cleaning ──────────────────────────────────────────
# Text is truncated at the first occurrence of any marker (case-insensitive).

BLEED_MARKERS = [
    # Direct rewrite announcements
    "here is your rewritten",
    "here is the rewritten",
    "here is a rewritten",
    "here's a rewritten",
    "here's the rewritten",
    "here is a revised",
    "here's a revised",
    "rewritten version",
    "revised transcript",
    "rewrite response",
    # Model self-commentary
    "the rewritten response",
    "the original response",
    "note that the",
    "note that your",
    "the response provided",
    "i have rewritten",
    "the provided text",
    "the ai response",
    "the original prompt was followed",
    # "Please revise / rewrite / rate / proceed / provide" family
    "please revise",
    "please rewrite",
    "please rate",
    "please generate another",
    "please write another version",
    "please proceed",
    "please provide",
    "please continue with",
    "please finish",
    "please complete",
    "i revised nothing",
    "i didn't change anything",
    "i have not changed",
    "another version:",
    "another monologue",
    "another transcript",
    "the following critique",
    "based on the critique",
    "here is the generated text",
    "generate another version",
    "generate a new transcript",
    "new attempt at writing",
    "however, i would like you to generate",
    "i would like you to generate",
    "however, the task requires",
    "note that the task",
    "note that your task",
    "the task requires",
    "following the same rules",
    "following the same profile",
    "keeping the same rules",
    "keeping the same profile",
    "same rules and",
    "same guidelines",
    # Common sign-off phrases
    "the best answer is",
    "the final answer is",
    "as requested",
    "let me know if you need",
    "please let me know",
    "please modify",
    "i hope this meets",
    "i hope it meets",
    "no further action is needed",
    "the response has already",
    # Checklist / requirements commentary
    "meets all the requirements",
    "meets all the instructions",
    "meets most of the requirements",
    "for a patient with the profile",
    "this response captures",
    "this narrative captures",
    "pressing distress without",
    "note:",
    # Noise / generation meta
    "the noise variables",
    # Structural analysis markers
    "## step",
    "### ",
    "step-by-step analysis",
    "step 1:",
    "step 2:",
    "step 3:",
    "understanding the patient",
    "patient profile alignment",
    "first-person monologue",
    "avoidance of clinical",
    "no explicit causal",
    # LaTeX / code markers
    "$\\boxed",
    "boxed{",
    "```python",
    # Meta-instruction fragments
    "monologue of approximately",
    "approximately 200 words",
    "word count",
    "ensure the narrative",
    "ensure it fully",
    "while adhering",
    "while ensuring",
    "adhering to the",
    "to adhere to the",
    "given the patient",
    "given that the patient",
    "above response",
    "above transcript",
    "in summary,\n",
    "let me analyze",
    "let me break",
]


def clean_narrative(text: str) -> str:
    """Strip ``` markers, all known commentary patterns, and trailing artifacts."""
    text = text.split("```")[0]
    lower = text.lower()
    cutoff = len(text)
    for marker in BLEED_MARKERS:
        idx = lower.find(marker)
        if idx != -1:
            cutoff = min(cutoff, idx)
    text = text[:cutoff].strip()
    # Remove stray trailing non-content characters left by partial marker cuts
    # e.g. "(Note:..." → "(", or "---\n\nPlease modify..." → "---"
    import re
    text = re.sub(r'[\s\(\[\{\-—=\*#"\']*$', '', text)
    return text.strip()


# ── Step 2: Residual bleed patterns ─────────────────────────────────────────
# These patterns cannot naturally occur in a therapy patient's monologue.
# They catch bleed text that survived clean_narrative() because it appeared
# before the first ``` and before all BLEED_MARKERS in the text.

RESIDUAL_BLEED = [
    "## ",              # markdown headers
    "\\boxed",          # latex
    "the revised response",
    "the original response",
    "meets all the requirements",
    "meets all the instructions",
    "step-by-step analysis",
    "word count",
    "avoidance of clinical",
    "patient profile alignment",
    "please let me know",
    "please modify",
    "i hope this meets",
    "i hope it meets",
    "the ai response",
    "monologue of approximately",
    "approximately 200 words",
    "given the patient",
    "given that the patient",
    "ensure the narrative",
    "while adhering to",
    "while ensuring",
    "above response",
    "above transcript",
    "this narrative is written",
    "this response is written",
    "for the purposes of this",
    "as per the instructions",
    "as per the rules",
    "as specified",
    "following the rules",
    "following the guidelines",
]


def has_residual_bleed(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in RESIDUAL_BLEED)


# ── Step 3: Y=1 depression signal ───────────────────────────────────────────
# At least ONE of these must appear in a Y=1 cleaned narrative.
# Covers the range of phrasing the model used across the 780 records.

DEPRESSION_SIGNALS = [
    "depressed",
    # "down" variants
    "feeling really down",
    "feel really down",
    "been really down",
    "felt really down",
    "feeling so down",
    "feel so down",
    "felt so down",
    "just been really down",
    "really down",          # catches "feeling... really down" after normalization
    "feeling down lately",
    "feel down lately",
    "been feeling down",
    # "low" variants
    "feeling really low",
    "feel really low",
    "been really low",
    "been feeling low",
    "felt really low",
    "feeling very low",
    "feel very low",
    "feeling low",
    "feel low",
    "persistently low",
    "persistent low",
    "persistent gloom",
    # sadness / cloud / heaviness
    "constant sadness",
    "persistent sadness",
    "sense of sadness",
    "constant cloud",
    "heavy cloud",
    "under this cloud",     # "waking up under this cloud every morning"
    "feels so heavy",       # "everything feels so heavy"
    "feel so heavy",
    "feeling so heavy",
    "everything feels heavy",
    "heavy feeling",
    "this heaviness",
    # emptiness
    "this emptiness",
    "into emptiness",
    "feel so empty",        # "I just feel so empty" (also covers "feel so... empty" after normalization)
    "feel empty",
    "feeling empty",
    # darkness / pointlessness
    "this darkness",
    "the darkness",
    "feels pointless",
    "feel pointless",
    "nothing brings me joy",
    "nothing seems to bring",
    "nothing feels right",
    "nothing seems right",
    "walking through quicksand",
    "constant weight",
    "this rut",
    "really struggling to find any joy",
    "struggling to find joy",
]


def _normalize(text: str) -> str:
    """Collapse ellipsis variants to a space so '...' doesn't break phrase matching."""
    import re
    text = text.replace("…", " ").replace("...", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.lower()


def has_depression_signal(text: str) -> bool:
    normalized = _normalize(text)
    return any(s in normalized for s in DEPRESSION_SIGNALS)


# ── Step 4: Y=0 disqualifiers ───────────────────────────────────────────────
# These phrases unambiguously signal depression and must NOT appear in Y=0 records.
# "feeling low" alone is too generic (could be energy/spirits), so only used above.

DEPRESSION_DISQUALIFIERS = [
    "depressed",
    "persistently low",
    "constant sadness",
    "constant cloud",
    "heavy cloud",
    "nothing brings me joy",
    "nothing seems to bring me joy",
    "this emptiness",
    "into this emptiness",
    "persistent sadness",
    "persistent gloom",
]


def has_false_depression(text: str) -> bool:
    lower = text.lower()
    return any(d in lower for d in DEPRESSION_DISQUALIFIERS)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    base = Path(__file__).parent
    narratives_path = base / "out" / "narratives.jsonl"
    output_path     = base / "out" / "reliable_narratives.jsonl"

    records = []
    with narratives_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    n_y1  = sum(1 for r in records if r["response_value"] == 1)
    n_y0  = total - n_y1
    print(f"Loaded {total} records  (Y=1: {n_y1}  |  Y=0: {n_y0})")
    print()

    fail_short      = []   # too short after cleaning
    fail_residual   = []   # residual bleed survived clean_narrative
    fail_no_signal  = []   # Y=1 but no depression keyword
    fail_false_dep  = []   # Y=0 but contains depression language
    reliable        = []

    for r in records:
        raw   = r["narrative"]
        clean = clean_narrative(raw)
        r["_clean"] = clean

        failures = set()

        if len(clean) < 200:
            failures.add("too_short")

        if has_residual_bleed(clean):
            failures.add("residual_bleed")

        if r["response_value"] == 1 and not has_depression_signal(clean):
            failures.add("no_depression_signal")

        if r["response_value"] == 0 and has_false_depression(clean):
            failures.add("false_depression")

        if not failures:
            reliable.append(r)
        else:
            if "too_short"           in failures: fail_short.append(r["doc_id"])
            if "residual_bleed"      in failures: fail_residual.append(r["doc_id"])
            if "no_depression_signal" in failures: fail_no_signal.append(r["doc_id"])
            if "false_depression"    in failures: fail_false_dep.append(r["doc_id"])

    removed = total - len(reliable)
    n_rel_y1 = sum(1 for r in reliable if r["response_value"] == 1)
    n_rel_y0 = len(reliable) - n_rel_y1

    print("=" * 52)
    print(f"  Reliable records : {len(reliable):>4}  (Y=1: {n_rel_y1}  Y=0: {n_rel_y0})")
    print(f"  Removed total    : {removed:>4}")
    print("=" * 52)
    print()
    print("Failure breakdown (one record may fail multiple criteria):")
    print(f"  Too short after cleaning (<200 chars) : {len(fail_short):>4}")
    print(f"  Residual bleed survived cleaning      : {len(fail_residual):>4}")
    print(f"  Y=1 — no depression signal found      : {len(fail_no_signal):>4}")
    print(f"  Y=0 — contains depression language    : {len(fail_false_dep):>4}")
    print()

    if fail_short:
        print(f"  too_short      → {fail_short}")
    if fail_residual:
        print(f"  residual_bleed → {fail_residual[:20]}{'...' if len(fail_residual) > 20 else ''}")
    if fail_no_signal:
        print(f"  no_signal (Y=1)→ {fail_no_signal}")
    if fail_false_dep:
        print(f"  false_dep (Y=0)→ {fail_false_dep}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in reliable:
            out = {k: v for k, v in r.items() if not k.startswith("_")}
            out["narrative"] = r["_clean"]
            f.write(json.dumps(out) + "\n")

    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
