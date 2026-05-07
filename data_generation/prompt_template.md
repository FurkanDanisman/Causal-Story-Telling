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
{noise_line}{inactive_line}- Depressed: {depression_label}

CAUSAL STRUCTURE TO ENCODE
The narrative must imply the following causal relationships through the flow of the story:
{active_edges}
Use phrases like "when X happens, I find myself Y-ing", "ever since X, I notice Y",
"X tends to leave me with Y" to suggest direction without stating it explicitly.

WHAT THE NARRATIVE MUST CONTAIN
Each core experience must appear at least ONCE as a single clear, dedicated expression —
specific enough that anyone reading it could unambiguously identify the concept without breaking the causal direction.
Do not scatter vague references throughout — one precise expression per concept.
If only one or two core experiences are listed, write with more depth and specificity about
each one — concrete detail, how it manifests day to day, how it feels — do not pad with
vague reflections or new concepts not in the profile.

Required proxies for this patient's active experiences:
{active_proxies}

- Noise variables must appear as a plain factual statement embedded naturally in the narrative —
  no emotional evaluation, no suggestion that they help or hurt anything, no causal direction.
  Good: "Things are fine on the outside — I even got a promotion at work recently — but this is what I keep coming back to."
        "I started eating better lately, not that it changes any of this."
  Bad: "Changing my diet has helped a bit." ← implies the noise caused improvement.
  Bad: "Moving was really stressful." ← implies the noise caused a negative state.

DON'T
- Do NOT introduce any psychological concept, symptom, or experience not listed in the
  patient profile. Write ONLY about what is listed. If stress is not listed, do not mention
  stress. If trust issues are not listed, do not mention trust. If withdrawal is not listed,
  do not mention withdrawal. The profile is exhaustive.
- Don't use technical variable names (early_adversity, emotion_dysregulation,
  social_withdrawal, rumination) — they are unnatural.
- Don't be vague or spread a concept loosely — one clear proxy per concept.
- Don't state causal links explicitly ("X caused Y", "because of X I feel Y").
- Don't write therapist dialogue, headings, labels, or anything outside the transcript tags.
- Don't give noise variables any emotional weight, positive or negative.
- If Depressed=NO: do not mention depression at all — not to deny it, not to contrast against it,
  not to say "I'm not depressed but...". The active variables are the complete story.

EXAMPLE 1 — multiple active variables, Depressed=YES, Noise: travel
Profile: chronic_stress → social_withdrawal → rumination → depression
<transcript>
Work has been relentless — I've been under constant stress, the deadlines and financial
pressure just never let up. When it gets this bad I pull away from everyone and isolate
myself, stop returning calls, cancel plans, go completely quiet. I know I should reach out
but I can't make myself do it. And being alone with it all just makes it worse — I can't
stop going over the same thoughts again and again, every mistake, every conversation, the
same loops and I cannot shut them off. It leaves me completely hollowed out. I've been
depressed. Not just tired or stressed — actually depressed, and it has been like this for
weeks now. Things look fine on the outside — I even took a trip to Japan last year — but
I cannot get back to feeling like myself.
</transcript>

EXAMPLE 2 — sparse profile, Depressed=NO, Noise: work_promotion
Profile: early_adversity
<transcript>
Growing up was traumatic — there was a lot of chaos at home and no real stability, ever.
I learned early to stay quiet and read every room before I could relax. What I notice now
is that I explain myself too carefully in conversations, like I still need to prove that
what happened was serious enough to count. I carry that hypervigilance into everything —
meetings, friendships, just being at home. I find myself bracing for things to fall apart
even when nothing is wrong. There are whole parts of that period I have never said out
loud, and I am not sure I have the language for them yet. Things are otherwise okay —
I even got a promotion at work recently, not that it touches any of this. I just want to
understand that time clearly, without either dramatising it or brushing past it.
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
Each core experience must appear at least ONCE as a single clear, dedicated expression —
specific enough that anyone reading it could unambiguously identify the concept.
Do not scatter vague references throughout — one precise expression per concept.
If only one or two core experiences are listed, write with more depth and specificity about
each one — concrete detail, how it manifests day to day, how it feels — do not pad with
vague reflections or new concepts not in the profile.

Required proxies for this patient's active experiences:
    early_adversity       → Examples: "my childhood was really unstable and difficult"
                                      "growing up was traumatic — there was a lot of chaos at home"
                                      "I had a really hard and unstable upbringing"
                            Must clearly signal: difficult childhood. Not just "things were hard".
                            DON'T say "early adversity".

    chronic_stress        → Examples: "I've been under constant stress — work, money, all of it"
                                      "the pressure has been relentless and it never lets up"
                            Must clearly signal: ongoing, persistent stress. Not just "I'm tired".
                            "stress" is natural — use it directly.

    social_withdrawal     → Examples: "I've been pulling away from everyone and isolating myself"
                                      "I stopped reaching out — I cut people off and go quiet"
                                      "I've been isolating, cancelling everything, seeing no one"
                            Must clearly signal: deliberate social isolation. Not just "I'm busy".
                            DON'T say "social withdrawal".

    rumination            → Examples: "I can't stop going over the same thoughts again and again"
                                      "my mind keeps repeating the same mistakes and conversations"
                                      "I keep replaying everything endlessly and can't shut it off"
                            Must clearly signal: repetitive stuck thinking. Not just "I worry".
                            DON'T say "I am ruminating".

    depression            → "I feel depressed" / "I've been depressed" / "I am depressed"
                            Use one of these exactly. State it plainly — do not elaborate,
                            qualify, or follow it with metaphors like "like I'm stuck" or
                            "like I'm in a fog". Just say it and move on.
                            DON'T say "depressed" in any form if Depressed=NO.

- Noise variables must appear as a plain factual statement embedded naturally in the narrative —
  no emotional evaluation, no suggestion that they help or hurt anything, no causal direction.
  Good: "Things are fine on the outside — I even got a promotion at work recently — but this is what I keep coming back to."
        "I started eating better lately, not that it changes any of this."
  Bad: "Changing my diet has helped a bit." ← implies the noise caused improvement.
  Bad: "Moving was really stressful." ← implies the noise caused a negative state.

DON'T
- Do NOT introduce any psychological concept, symptom, or experience not listed in the
  patient profile. Write ONLY about what is listed. If stress is not listed, do not mention
  stress. If trust issues are not listed, do not mention trust. If withdrawal is not listed,
  do not mention withdrawal. The profile is exhaustive.
- Don't use technical variable names (early_adversity, emotion_dysregulation,
  social_withdrawal, rumination) — they are unnatural.
- Don't be vague or spread a concept loosely — one clear proxy per concept.
- Don't state causal links explicitly ("X caused Y", "because of X I feel Y").
- Don't write therapist dialogue, headings, labels, or anything outside the transcript tags.
- Don't give noise variables any emotional weight, positive or negative.
- If Depressed=NO: do not mention depression at all — not to deny it, not to contrast against it,
  not to say "I'm not depressed but...". The active variables are the complete story.

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
