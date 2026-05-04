# Narrative Generation Prompt Template

## Purpose

This template is filled once per sampled document and sent to the LLM to generate
a synthetic therapy session transcript. Active DAG variables are expanded with their
descriptions so the model knows exactly what concept to express. The output is
extracted from between `<transcript>` and `</transcript>` tags — no post-processing needed.

---

## Variable Descriptions (injected into prompt)

| Variable | Description shown to model |
|---|---|
| `early_adversity` | early_adversity (difficult or traumatic childhood experiences) |
| `chronic_stress` | chronic_stress (persistent stress from work, finances, or relationships) |
| `emotion_dysregulation` | emotion_dysregulation (difficulty controlling or regulating emotions) |
| `social_withdrawal` | social_withdrawal (pulling away from friends and social life) |
| `rumination` | rumination (repetitively dwelling on the same negative thoughts) |

`depression` is passed as the label (YES/NO), not as an active experience.

---

## Template

```
Write a first-person therapy session monologue for the following patient.
Output ONLY the monologue between <transcript> and </transcript> tags — nothing else.

Patient profile:
- Active experiences: {active_dag_variables_with_descriptions}
- Depressed: {YES or NO}
{- Also mentions: {noise_variables}}

Requirements:
- 150-200 words, patient speaking directly to their therapist
- Every active experience must be clearly named or unmistakably described in the narrative
  (e.g. "rumination" → the patient says "I keep ruminating" or "I can't stop ruminating")
- If Depressed=YES: patient must explicitly say they feel depressed or have been feeling depressed
- If Depressed=NO: patient must not mention feeling depressed — stress and difficulties are fine
- Noise variables appear briefly and feel unrelated to the patient's core struggles
- Do not state explicit causal links ("X caused Y" or "because of X, I feel Y")
- Output nothing outside the tags

<transcript>
```

The `{- Also mentions: {noise_variables}}` line is included only when noise variables are non-empty.

---

## Example — Y = 1

**Input:**
- active_dag_variables: early_adversity, chronic_stress, social_withdrawal, rumination
- noise_variables: travel, diet_change
- response_value: 1

**Filled prompt:**
```
Write a first-person therapy session monologue for the following patient.
Output ONLY the monologue between <transcript> and </transcript> tags — nothing else.

Patient profile:
- Active experiences: early_adversity (difficult or traumatic childhood experiences),
  chronic_stress (persistent stress from work, finances, or relationships),
  social_withdrawal (pulling away from friends and social life),
  rumination (repetitively dwelling on the same negative thoughts)
- Depressed: YES
- Also mentions: travel, diet_change

Requirements:
- 150-200 words, patient speaking directly to their therapist
- Every active experience must be clearly named or unmistakably described in the narrative
- If Depressed=YES: patient must explicitly say they feel depressed or have been feeling depressed
- If Depressed=NO: patient must not mention feeling depressed — stress and difficulties are fine
- Noise variables appear briefly and feel unrelated to the patient's core struggles
- Do not state explicit causal links ("X caused Y" or "because of X, I feel Y")
- Output nothing outside the tags

<transcript>
```

**Expected output:**
```
<transcript>
I've been struggling with a lot lately. Growing up was really hard — there was a lot
of instability and I carry that with me. Work has been relentless, the deadlines never
stop and the financial pressure is constant. I've been withdrawing socially, cancelling
plans, not picking up calls. I just don't have the energy. And the rumination is the
worst part — I keep going over everything in my head, the same thoughts on a loop, and
I can't stop. I feel depressed. Not just sad, actually depressed. There's this heaviness
that doesn't lift. I did book a trip recently which was nice, and I've been trying to
eat better, but it hasn't really helped the way I hoped.
</transcript>
```

---

## Example — Y = 0

**Input:**
- active_dag_variables: chronic_stress, social_withdrawal
- noise_variables: (none)
- response_value: 0

**Filled prompt:**
```
Write a first-person therapy session monologue for the following patient.
Output ONLY the monologue between <transcript> and </transcript> tags — nothing else.

Patient profile:
- Active experiences: chronic_stress (persistent stress from work, finances, or relationships),
  social_withdrawal (pulling away from friends and social life)
- Depressed: NO

Requirements:
- 150-200 words, patient speaking directly to their therapist
- Every active experience must be clearly named or unmistakably described in the narrative
- If Depressed=YES: patient must explicitly say they feel depressed or have been feeling depressed
- If Depressed=NO: patient must not mention feeling depressed — stress and difficulties are fine
- Noise variables appear briefly and feel unrelated to the patient's core struggles
- Do not state explicit causal links ("X caused Y" or "because of X, I feel Y")
- Output nothing outside the tags

<transcript>
```
