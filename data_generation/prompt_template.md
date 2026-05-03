# Narrative Generation Prompt Template

## Purpose

This template is filled once per sampled document and sent to the LLM to generate
a synthetic therapy session transcript. The active DAG variables and noise variables
come from the Stage 1+2 SCM sampler. The LLM's only job is to wrap them in
realistic, implicit, first-person speech.

---

## Template

```
You are generating a realistic therapy session transcript for a causal inference
research study on mental health.

Patient profile:
- Relevant experiences: {active_dag_variables}
- Depression: {YES or NO}
{- Also mentions: {noise_variables}}

Write a first-person monologue of approximately 200 words, as if the patient is
speaking to their therapist during a session.

Rules:
- Do not use clinical or technical language — express everything through personal
  experiences, feelings, and daily life events
- Do not state causal relationships explicitly (avoid "X caused Y" or "because of
  X, I feel Y")
- If depression is YES, the patient must clearly state in their own words that they
  have been feeling depressed or persistently low — as a single unified experience,
  not a list of separate symptoms
- If depression is NO, the patient must not describe feeling depressed or
  persistently low — difficulties and stress may be present but the patient is coping
- Noise variables should appear naturally and briefly, feeling unrelated to the
  patient's core emotional struggles
- Write only the patient's words — no therapist dialogue, no labels, no headings

Transcript:
```

The line `{- Also mentions: {noise_variables}}` is included only when noise
variables are non-empty. When there are no noise variables, that line is omitted
entirely.

---

## Example — Y = 1

**Input vector:**
- active_dag_variables: early_adversity, chronic_stress, emotion_dysregulation,
  social_withdrawal, rumination
- noise_variables: travel, diet_change
- depression: YES

**Filled prompt:**
```
You are generating a realistic therapy session transcript for a causal inference
research study on mental health.

Patient profile:
- Relevant experiences: early_adversity, chronic_stress, emotion_dysregulation,
  social_withdrawal, rumination
- Depression: YES
- Also mentions: travel, diet_change

Write a first-person monologue of approximately 200 words, as if the patient is
speaking to their therapist during a session.

Rules:
- Do not use clinical or technical language — express everything through personal
  experiences, feelings, and daily life events
- Do not state causal relationships explicitly (avoid "X caused Y" or "because of
  X, I feel Y")
- If depression is YES, the patient must clearly state in their own words that they
  have been feeling depressed or persistently low — as a single unified experience,
  not a list of separate symptoms
- If depression is NO, the patient must not describe feeling depressed or
  persistently low — difficulties and stress may be present but the patient is coping
- Noise variables should appear naturally and briefly, feeling unrelated to the
  patient's core emotional struggles
- Write only the patient's words — no therapist dialogue, no labels, no headings

Transcript:
```

---

## Example — Y = 0

**Input vector:**
- active_dag_variables: chronic_stress, social_withdrawal
- noise_variables: (none)
- depression: NO

**Filled prompt:**
```
You are generating a realistic therapy session transcript for a causal inference
research study on mental health.

Patient profile:
- Relevant experiences: chronic_stress, social_withdrawal
- Depression: NO

Write a first-person monologue of approximately 200 words, as if the patient is
speaking to their therapist during a session.

Rules:
- Do not use clinical or technical language — express everything through personal
  experiences, feelings, and daily life events
- Do not state causal relationships explicitly (avoid "X caused Y" or "because of
  X, I feel Y")
- If depression is YES, the patient must clearly state in their own words that they
  have been feeling depressed or persistently low — as a single unified experience,
  not a list of separate symptoms
- If depression is NO, the patient must not describe feeling depressed or
  persistently low — difficulties and stress may be present but the patient is coping
- Noise variables should appear naturally and briefly, feeling unrelated to the
  patient's core emotional struggles
- Write only the patient's words — no therapist dialogue, no labels, no headings

Transcript:
```
