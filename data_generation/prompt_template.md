# Narrative Generation Prompt

This is the exact prompt sent to the model for each patient record.
Placeholders in `{curly braces}` are filled per patient — a fully filled example is at the bottom.

---

## Prompt

<!-- TEMPLATE_START -->
```
You are writing synthetic therapy session transcripts for a causal inference research study.

TASK
Write a first-person monologue (150-200 words) as a patient speaking to their therapist.

PATIENT PROFILE
- Core experiences: {active_dag_variables}
{noise_line}- Depressed: {depression_label}

CAUSAL STRUCTURE TO ENCODE
The narrative must imply the following causal relationships through the flow of the story:
{active_edges}
Use phrases like "when X happens, I find myself Y-ing", "ever since X, I notice Y",
"X tends to leave me with Y" to suggest direction without stating it explicitly.

WHAT THE NARRATIVE MUST CONTAIN
Each experience must be expressed in natural language as a real patient would say it:
    early_adversity       → "my childhood was really rough / unstable / traumatic"
                            "growing up there was a lot of chaos and instability at home"
                            "I had a really hard upbringing"
                            DON'T say "early adversity" — no one says this
    chronic_stress        → "constant stress", "the pressure never lets up"
                            "I've been under so much stress with work / money / relationships"
                            "stress" is natural — use it freely
    emotion_dysregulation → "I can't control my emotions", "my emotions are all over the place"
                            "I break down easily", "I can't keep it together emotionally"
                            DON'T say "emotion dysregulation" — no one says this
    social_withdrawal     → "I've been pulling away from everyone", "I stopped seeing friends"
                            "I've been isolating myself", "I go quiet and stop reaching out"
                            DON'T say "social withdrawal" — no one says this
    rumination            → "I can't stop going over the same thoughts"
                            "my mind keeps looping on the same things"
                            "I keep replaying every conversation and mistake"
                            DON'T say "I am ruminating" — no one says this
    depression            → "I feel depressed", "I've been depressed", "I am depressed"
                            People DO say this — use it directly if Depressed=YES
- If Depressed=NO: patient must not say they feel depressed or persistently low.
- Noise variables appear in one sentence only, feel unrelated to the patient's core struggles.

DON'T
- Don't use the technical variable names for early_adversity, emotion_dysregulation,
  social_withdrawal, or rumination — they sound robotic and unnatural.
- Don't state explicit causal links ("X caused Y", "because of X, I feel Y").
- Don't mention depression in any form if Depressed=NO.
- Don't write therapist dialogue, headings, labels, or anything outside the transcript tags.
- Don't describe noise variables as emotionally significant or causally related to anything.

EXAMPLE
Profile: chronic_stress → social_withdrawal → rumination → depression | Depressed=YES | Noise: travel
<transcript>
Work has been relentless — the constant stress just doesn't let up, the deadlines, the
financial pressure, all of it piling on. When things get this bad I just pull away from
everyone, stop returning calls, cancel plans, go quiet. I know I should reach out but I
genuinely can't make myself do it. And being alone with it all just makes everything worse
because I can't stop going over the same thoughts — every mistake, every conversation,
the same loops endlessly, and I cannot shut it off. It leaves me completely hollowed out.
I've been depressed. Not just tired — actually depressed, and it has been like this for
weeks. I did take a trip to Japan last year which was a nice escape, but right now
nothing helps.
</transcript>

Now write the transcript for this patient. Output ONLY between the tags.

<transcript>
```
<!-- TEMPLATE_END -->

---

## Placeholder Reference

### `{active_dag_variables}`

Each active variable is expanded using this mapping:

| Variable | Shown to model |
|---|---|
| `early_adversity` | early_adversity (difficult or traumatic childhood experiences) |
| `chronic_stress` | chronic_stress (persistent stress from work, finances, or relationships) |
| `emotion_dysregulation` | emotion_dysregulation (difficulty controlling or regulating emotions) |
| `social_withdrawal` | social_withdrawal (pulling away from friends and social life) |
| `rumination` | rumination (repetitively dwelling on the same negative thoughts) |

### `{active_edges}`

Computed per patient by intersecting their active variables (+ `depression` if Y=1)
with the full ground truth DAG:

```
early_adversity       → chronic_stress
early_adversity       → emotion_dysregulation
early_adversity       → depression
chronic_stress        → emotion_dysregulation
chronic_stress        → social_withdrawal
emotion_dysregulation → rumination
social_withdrawal     → rumination
rumination            → depression
```

Only edges where both endpoints are active for that patient are injected.

### `{noise_line}`

`- Noise (mention briefly, unrelated to struggles): travel, diet_change`
Line is omitted entirely if the patient has no noise variables.

---

## Fully Filled Example

**Patient record:**
- active_dag_variables: `early_adversity`, `chronic_stress`, `social_withdrawal`, `rumination`
- noise_variables: `travel`
- response_value: `1`

**Active edges computed:**
```
early_adversity   → chronic_stress
chronic_stress    → social_withdrawal
social_withdrawal → rumination
rumination        → depression
```

**Exact prompt sent to model:**

```
You are writing synthetic therapy session transcripts for a causal inference research study.

TASK
Write a first-person monologue (150-200 words) as a patient speaking to their therapist.

PATIENT PROFILE
- Core experiences: early_adversity (difficult or traumatic childhood experiences),
  chronic_stress (persistent stress from work, finances, or relationships),
  social_withdrawal (pulling away from friends and social life),
  rumination (repetitively dwelling on the same negative thoughts)
- Noise (mention briefly, unrelated to struggles): travel
- Depressed: YES

CAUSAL STRUCTURE TO ENCODE
The narrative must imply the following causal relationships through the flow of the story:
  early_adversity   → chronic_stress
  chronic_stress    → social_withdrawal
  social_withdrawal → rumination
  rumination        → depression
Use phrases like "when X happens, I find myself Y-ing", "ever since X, I notice Y",
"X tends to leave me with Y" to suggest direction without stating it explicitly.

WHAT THE NARRATIVE MUST CONTAIN
Each experience must be expressed in natural language as a real patient would say it:
    early_adversity       → "my childhood was really rough / unstable / traumatic"
                            "growing up there was a lot of chaos and instability at home"
                            "I had a really hard upbringing"
                            DON'T say "early adversity" — no one says this
    chronic_stress        → "constant stress", "the pressure never lets up"
                            "I've been under so much stress with work / money / relationships"
                            "stress" is natural — use it freely
    emotion_dysregulation → "I can't control my emotions", "my emotions are all over the place"
                            "I break down easily", "I can't keep it together emotionally"
                            DON'T say "emotion dysregulation" — no one says this
    social_withdrawal     → "I've been pulling away from everyone", "I stopped seeing friends"
                            "I've been isolating myself", "I go quiet and stop reaching out"
                            DON'T say "social withdrawal" — no one says this
    rumination            → "I can't stop going over the same thoughts"
                            "my mind keeps looping on the same things"
                            "I keep replaying every conversation and mistake"
                            DON'T say "I am ruminating" — no one says this
    depression            → "I feel depressed", "I've been depressed", "I am depressed"
                            People DO say this — use it directly if Depressed=YES
- If Depressed=NO: patient must not say they feel depressed or persistently low.
- Noise variables appear in one sentence only, feel unrelated to the patient's core struggles.

DON'T
- Don't use the technical variable names for early_adversity, emotion_dysregulation,
  social_withdrawal, or rumination — they sound robotic and unnatural.
- Don't state explicit causal links ("X caused Y", "because of X, I feel Y").
- Don't mention depression in any form if Depressed=NO.
- Don't write therapist dialogue, headings, labels, or anything outside the transcript tags.
- Don't describe noise variables as emotionally significant or causally related to anything.

EXAMPLE
Profile: chronic_stress → social_withdrawal → rumination → depression | Depressed=YES | Noise: travel
<transcript>
Work has been relentless — the constant stress just doesn't let up, the deadlines, the
financial pressure, all of it piling on. When things get this bad I just pull away from
everyone, stop returning calls, cancel plans, go quiet. I know I should reach out but I
genuinely can't make myself do it. And being alone with it all just makes everything worse
because I can't stop going over the same thoughts — every mistake, every conversation,
the same loops endlessly, and I cannot shut it off. It leaves me completely hollowed out.
I've been depressed. Not just tired — actually depressed, and it has been like this for
weeks. I did take a trip to Japan last year which was a nice escape, but right now
nothing helps.
</transcript>

Now write the transcript for this patient. Output ONLY between the tags.

<transcript>
```
