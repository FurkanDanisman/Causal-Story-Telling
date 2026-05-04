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
Each core experience must appear ONCE as a single clear, dedicated expression —
specific enough that anyone reading it could unambiguously identify the concept.
Do not scatter vague references throughout — one precise expression per concept.

Required proxies per variable:
    early_adversity       → ONE of: "my childhood was really unstable and difficult"
                                   "growing up was traumatic — there was a lot of chaos at home"
                                   "I had a really hard and unstable upbringing"
                            Must clearly signal: difficult childhood. Not just "things were hard".
                            DON'T say "early adversity".

    chronic_stress        → ONE of: "I've been under constant stress — work, money, all of it"
                                   "the pressure has been relentless and it never lets up"
                            Must clearly signal: ongoing, persistent stress. Not just "I'm tired".
                            "stress" is natural — use it directly.

    emotion_dysregulation → ONE of: "I can't control my emotions when it gets bad"
                                   "my emotions are completely all over the place"
                                   "I break down and can't pull myself together emotionally"
                            Must clearly signal: inability to regulate emotions. Not just "I feel things".
                            DON'T say "emotion dysregulation".

    social_withdrawal     → ONE of: "I've been pulling away from everyone and isolating myself"
                                   "I stopped reaching out — I cut people off and go quiet"
                                   "I've been isolating, cancelling everything, seeing no one"
                            Must clearly signal: deliberate social isolation. Not just "I'm busy".
                            DON'T say "social withdrawal".

    rumination            → ONE of: "I can't stop going over the same thoughts again and again"
                                   "my mind keeps looping on the same mistakes and conversations"
                                   "I keep replaying everything endlessly and can't shut it off"
                            Must clearly signal: repetitive stuck thinking. Not just "I worry".
                            DON'T say "I am ruminating".

    depression            → "I feel depressed" / "I've been depressed" / "I am depressed"
                            Use directly and explicitly if Depressed=YES.
                            DON'T say "depressed" in any form if Depressed=NO.

- Noise variables appear in one sentence only, briefly, and feel unrelated to core struggles.

DON'T
- Don't use technical variable names (early_adversity, emotion_dysregulation,
  social_withdrawal, rumination) — they are unnatural.
- Don't be vague or spread a concept loosely — one clear proxy per concept.
- Don't state causal links explicitly ("X caused Y", "because of X I feel Y").
- Don't write therapist dialogue, headings, labels, or anything outside the transcript tags.
- Don't describe noise variables as emotionally significant or causally related to anything.

EXAMPLE
Profile: chronic_stress → social_withdrawal → rumination → depression | Depressed=YES | Noise: travel
<transcript>
Work has been relentless — I've been under constant stress, the deadlines and financial
pressure just never let up. When it gets this bad I pull away from everyone and isolate
myself, stop returning calls, cancel plans, go completely quiet. I know I should reach out
but I can't make myself do it. And being alone with it all just makes it worse — I can't
stop going over the same thoughts again and again, every mistake, every conversation, the
same loops and I cannot shut them off. It leaves me completely hollowed out. I've been
depressed. Not just tired or stressed — actually depressed, and it has been like this for
weeks now. I did take a trip to Japan last year which was a nice change, but right now
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
Each core experience must appear ONCE as a single clear, dedicated expression —
specific enough that anyone reading it could unambiguously identify the concept.
Do not scatter vague references throughout — one precise expression per concept.

Required proxies per variable:
    early_adversity       → ONE of: "my childhood was really unstable and difficult"
                                   "growing up was traumatic — there was a lot of chaos at home"
                                   "I had a really hard and unstable upbringing"
                            Must clearly signal: difficult childhood. Not just "things were hard".
                            DON'T say "early adversity".

    chronic_stress        → ONE of: "I've been under constant stress — work, money, all of it"
                                   "the pressure has been relentless and it never lets up"
                            Must clearly signal: ongoing, persistent stress. Not just "I'm tired".
                            "stress" is natural — use it directly.

    emotion_dysregulation → ONE of: "I can't control my emotions when it gets bad"
                                   "my emotions are completely all over the place"
                                   "I break down and can't pull myself together emotionally"
                            Must clearly signal: inability to regulate emotions. Not just "I feel things".
                            DON'T say "emotion dysregulation".

    social_withdrawal     → ONE of: "I've been pulling away from everyone and isolating myself"
                                   "I stopped reaching out — I cut people off and go quiet"
                                   "I've been isolating, cancelling everything, seeing no one"
                            Must clearly signal: deliberate social isolation. Not just "I'm busy".
                            DON'T say "social withdrawal".

    rumination            → ONE of: "I can't stop going over the same thoughts again and again"
                                   "my mind keeps looping on the same mistakes and conversations"
                                   "I keep replaying everything endlessly and can't shut it off"
                            Must clearly signal: repetitive stuck thinking. Not just "I worry".
                            DON'T say "I am ruminating".

    depression            → "I feel depressed" / "I've been depressed" / "I am depressed"
                            Use directly and explicitly if Depressed=YES.
                            DON'T say "depressed" in any form if Depressed=NO.

- Noise variables appear in one sentence only, briefly, and feel unrelated to core struggles.

DON'T
- Don't use technical variable names (early_adversity, emotion_dysregulation,
  social_withdrawal, rumination) — they are unnatural.
- Don't be vague or spread a concept loosely — one clear proxy per concept.
- Don't state causal links explicitly ("X caused Y", "because of X I feel Y").
- Don't write therapist dialogue, headings, labels, or anything outside the transcript tags.
- Don't describe noise variables as emotionally significant or causally related to anything.

EXAMPLE
Profile: chronic_stress → social_withdrawal → rumination → depression | Depressed=YES | Noise: travel
<transcript>
Work has been relentless — I've been under constant stress, the deadlines and financial
pressure just never let up. When it gets this bad I pull away from everyone and isolate
myself, stop returning calls, cancel plans, go completely quiet. I know I should reach out
but I can't make myself do it. And being alone with it all just makes it worse — I can't
stop going over the same thoughts again and again, every mistake, every conversation, the
same loops and I cannot shut them off. It leaves me completely hollowed out. I've been
depressed. Not just tired or stressed — actually depressed, and it has been like this for
weeks now. I did take a trip to Japan last year which was a nice change, but right now
nothing helps.
</transcript>

Now write the transcript for this patient. Output ONLY between the tags.

<transcript>
```
