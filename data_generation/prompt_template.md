# Narrative Generation Prompt Template

## Purpose

One prompt per patient record. Active DAG variables are expanded with descriptions.
The active edges from the ground truth DAG are computed and injected so the narrative
encodes causal directionality, not just variable presence.
Output is extracted from `<transcript>...</transcript>` — no post-processing needed.

---

## Template

```
You are writing synthetic therapy session transcripts for a causal inference research study.

TASK
Write a first-person monologue (150-200 words) as a patient speaking to their therapist.

PATIENT PROFILE
- Core experiences: {active_dag_variables_with_descriptions}
- Noise (mention briefly, unrelated to struggles): {noise_variables}   ← omit line if empty
- Depressed: {YES or NO}

CAUSAL STRUCTURE TO ENCODE
The narrative must imply the following causal relationships through the flow of the story:
{active_edges}
The patient should speak in a way that naturally suggests these directions — for example
using phrases like "when X happens, I find myself Y-ing", "X tends to make me",
"ever since X, I notice Y", "X leaves me with Y". The causal direction must be
recoverable from the text, but must not be stated as an explicit fact.

WHAT THE NARRATIVE MUST CONTAIN
- Every core experience must appear by its exact name or an unmistakable close variant:
    rumination            → "I keep ruminating" / "the rumination" / "I can't stop ruminating"
    chronic_stress        → "the chronic stress" / "chronic stress at work"
    social_withdrawal     → "I've been withdrawing socially" / "social withdrawal"
    emotion_dysregulation → "I can't regulate my emotions" / "emotion dysregulation"
    early_adversity       → "early adversity" / "childhood adversity" / "adversity growing up"
- If Depressed=YES: patient must explicitly say "I feel depressed" / "I've been depressed" / "I am depressed".
- If Depressed=NO: patient must not say they feel depressed or persistently low.
- Noise variables appear in one sentence only and feel unrelated to the patient's core struggles.

DON'T
- Don't replace core experience names with vague metaphors only — use the actual terms.
- Don't say "X caused Y" or "because of X" — imply direction through narrative flow, not explicit statements.
- Don't mention depression in any form if Depressed=NO.
- Don't write therapist dialogue, headings, labels, or anything outside the transcript tags.
- Don't describe noise variables as emotionally significant or causally connected to anything.

EXAMPLE
Profile: chronic_stress → social_withdrawal → rumination → depression | Depressed=YES | Noise: travel
<transcript>
Work has been relentless — the chronic stress just does not let up, the deadlines, the
financial pressure, all of it piling on. When the stress gets this bad I find myself
withdrawing socially, cancelling plans, not picking up calls, just going quiet. I know I
should reach out but I genuinely can't make myself do it, and being alone just feeds the
rumination. I keep ruminating on every conversation, every mistake, the same loops over
and over, and I cannot shut it off. The rumination leaves me feeling completely hollowed
out. I feel depressed. Not just tired — actually depressed, and it has been like this for
weeks. I did book a trip last month which was a nice distraction, but it hasn't changed anything.
</transcript>

Now write the transcript for this patient. Output ONLY between the tags.

<transcript>
```

---

## How active edges are computed

For each patient record, `build_prompt()` intersects the patient's active variables
(plus `depression` if Y=1) with the full ground truth edge list:

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

Only edges where both endpoints are active for that patient are injected into the prompt.
