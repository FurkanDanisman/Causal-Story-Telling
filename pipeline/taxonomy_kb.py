"""
Knowledge-base taxonomy for LLM-guided variable extraction.

Source: Beck & Bredemeier (2016), "A Unified Model of Depression:
        Integrating Clinical, Cognitive, Biological, and Evolutionary Perspectives"

This module defines the taxonomy as pure clinical knowledge — construct definitions,
diagnostic criteria, and intensity levels derived from the theoretical framework.
No regex patterns. No dataset-specific phrasings.

The taxonomy is used to build a structured prompt that guides an LLM to apply
the clinical framework to a narrative text. The LLM handles linguistic variation;
the knowledge (what to look for and how to classify it) comes from this module.

Architecture:
  taxonomy_kb.py          <- clinical knowledge (this file)
  step1_kb_extract.py     <- LLM execution engine + prompt builder
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class IntensityLevel:
    level: int
    label: str
    clinical_criteria: str


@dataclass
class TaxonomyVar:
    name: str
    category: str       # "Distal Vulnerability" | "Precipitation" | "Outcome" | "Maintenance" | "Reversal"
    subcategory: str
    clinical_definition: str
    clinical_notes: str             # nuances, distinctions, common confusions
    prototypical_expressions: List[str]  # drawn from clinical literature, NOT dataset-specific
    intensity_levels: List[IntensityLevel]  # levels 0–3


# ── Taxonomy ───────────────────────────────────────────────────────────────────

TAXONOMY: List[TaxonomyVar] = [

    # ══════════════════════════════════════════════════════════════════════════
    # 1. DISTAL VULNERABILITY FACTORS (Confounders / Root Causes)
    # ══════════════════════════════════════════════════════════════════════════

    TaxonomyVar(
        name="early_life_trauma",
        category="Distal Vulnerability",
        subcategory="Early Life Experiences",
        clinical_definition=(
            "Adverse childhood experiences that elevate baseline vulnerability to depression. "
            "Includes parental loss or separation during childhood, maltreatment, physical or "
            "emotional abuse, neglect, or a chronically unstable home environment. These are "
            "distal causes: they shape how an individual responds to later stressors, but do "
            "not directly cause depression on their own."
        ),
        clinical_notes=(
            "Distinguish from adult stressors (those belong to Precipitation). "
            "The reference must anchor the adversity in childhood or early life. "
            "A vague mention of 'difficult times' without developmental anchoring scores L1 at most. "
            "Severity follows the intensity and specificity of the reported experience."
        ),
        prototypical_expressions=[
            "I had a very difficult childhood",
            "I lost my mother when I was young",
            "I was abused as a child",
            "Growing up, the environment at home was chaotic and unsafe",
            "My parents were absent / neglectful / violent",
            "I experienced a lot of hardship growing up",
            "I was always told I wasn't good enough as a kid",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No mention of childhood history or early adversity."),
            IntensityLevel(1, "Low",
                "Vague or mild reference to childhood difficulties without specific loss or abuse "
                "(e.g., 'things were hard growing up', 'my childhood wasn't easy')."),
            IntensityLevel(2, "Moderate",
                "Clear evidence of significant early loss (e.g., parental death or separation), "
                "recurring abuse, neglect, or a markedly unstable home environment."),
            IntensityLevel(3, "High",
                "Explicit severe trauma: sexual or physical abuse, chronic maltreatment, or "
                "pervasive early identity damage (e.g., 'I've always been worthless since childhood')."),
        ],
    ),

    TaxonomyVar(
        name="genetic_risk",
        category="Distal Vulnerability",
        subcategory="Genetic Risk",
        clinical_definition=(
            "References to biological predisposition to depression, including family history of "
            "depression or mood disorders, or explicit mention of genetic or neurobiological "
            "vulnerability (e.g., serotonin transporter gene polymorphisms, 5-HTTLPR)."
        ),
        clinical_notes=(
            "A general mention of 'family' without explicit mental health history does not qualify. "
            "Must involve explicit reference to depression, mood disorder, or mental illness in "
            "biological relatives, or to biological/genetic predisposition."
        ),
        prototypical_expressions=[
            "Depression runs in my family",
            "My mother / father struggled with depression",
            "Mental illness is in my family",
            "I've been told I have a biological predisposition to depression",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No mention of family history or biological risk."),
            IntensityLevel(1, "Low",
                "Vague reference suggesting familial emotional difficulty without explicit diagnosis "
                "(e.g., 'everyone in my family tends to be anxious or sad')."),
            IntensityLevel(2, "Moderate",
                "Clear mention of a first-degree relative (parent, sibling) with diagnosed depression "
                "or mood disorder."),
            IntensityLevel(3, "High",
                "Multiple affected relatives, or explicit reference to genetic or neurobiological "
                "vulnerability (e.g., mentions 5-HTTLPR, 'I was told my brain chemistry makes me prone')."),
        ],
    ),

    TaxonomyVar(
        name="info_processing_biases",
        category="Distal Vulnerability",
        subcategory="Information Processing Biases",
        clinical_definition=(
            "Systematic cognitive tendencies that bias information processing toward the negative. "
            "Includes (a) selective attention: habitually focusing on negative stimuli while "
            "discounting positive information; (b) memory biases: difficulty recalling specific "
            "positive autobiographical memories, overgeneralization of negative events, or "
            "enhanced recall of failures and losses."
        ),
        clinical_notes=(
            "This is a trait-level cognitive style, not a momentary reaction. "
            "Distinguish from situational rumination (which is a maintenance factor). "
            "Here the bias applies across contexts and time, not just to the current stressor. "
            "A key marker is overgeneralization: jumping from a specific failure to a global "
            "conclusion about oneself or one's history ('I always fail', 'nothing ever works out')."
        ),
        prototypical_expressions=[
            "I always focus on what goes wrong, never what goes right",
            "I can't remember things ever being truly good",
            "I tend to catastrophize — one bad thing means everything is ruined",
            "I've never been able to see the positive side",
            "My mind automatically goes to the worst interpretation",
            "I remember every failure but can't recall my successes",
            "When something bad happens I assume it will always be that way",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No evidence of systematic negative cognitive style."),
            IntensityLevel(1, "Low",
                "Occasional pessimistic interpretation, acknowledged as temporary or situational."),
            IntensityLevel(2, "Moderate",
                "Clear pattern of selective negative attention or memory bias across situations "
                "(e.g., dismissing positives, selectively recalling failures)."),
            IntensityLevel(3, "High",
                "Pervasive overgeneralization from specific events to global negative self-concept "
                "or history (e.g., 'I've always been a failure', 'nothing has ever worked out for me')."),
        ],
    ),

    TaxonomyVar(
        name="sociotropy",
        category="Distal Vulnerability",
        subcategory="Personality Vulnerability",
        clinical_definition=(
            "A personality dimension characterized by high investment in interpersonal relationships "
            "as the primary source of self-worth. Sociotropic individuals are hypersensitive to "
            "rejection, criticism, abandonment, or loss of approval from others. Their self-esteem "
            "is contingent on being loved and accepted."
        ),
        clinical_notes=(
            "Distinguish from normal concern for relationships. The key marker is that the person's "
            "identity and self-worth are fundamentally contingent on others' approval. "
            "A sociotropic person doesn't just want connection — they feel they cannot function "
            "or have value without it."
        ),
        prototypical_expressions=[
            "I need people to approve of me to feel okay",
            "When someone criticizes me, I fall apart",
            "I can't stand the idea of someone being angry at me",
            "My whole sense of self depends on my relationships",
            "Being rejected is the worst thing that can happen to me",
            "I constantly worry about what others think of me",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No evidence of interpersonal dependency as self-worth source."),
            IntensityLevel(1, "Low",
                "Heightened concern about others' opinions, but self-worth remains largely independent."),
            IntensityLevel(2, "Moderate",
                "Clear evidence that self-esteem is significantly contingent on approval or acceptance; "
                "strong fear of rejection or abandonment."),
            IntensityLevel(3, "High",
                "Identity completely fused with interpersonal relationships; perception of being "
                "unable to exist or have value without the love or approval of others."),
        ],
    ),

    TaxonomyVar(
        name="autonomy_vulnerability",
        category="Distal Vulnerability",
        subcategory="Personality Vulnerability",
        clinical_definition=(
            "A personality dimension characterized by high investment in independence, achievement, "
            "mastery, and control as the primary source of self-worth. Autonomous individuals are "
            "hypersensitive to failure, loss of control, or any diminishment of their independence "
            "or competence."
        ),
        clinical_notes=(
            "Distinguish from healthy ambition or competence. The key marker is that failure or "
            "loss of control is catastrophically threatening to the person's identity — not just "
            "disappointing. The person's entire self-concept rests on being capable, in control, "
            "and achieving."
        ),
        prototypical_expressions=[
            "Failing at something is unbearable — it means I'm worthless",
            "I can't accept not being in control",
            "My sense of self depends entirely on my accomplishments",
            "Needing help from others feels like a complete failure",
            "If I can't do it independently, I'm not worthy",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No evidence of autonomy as primary self-worth source."),
            IntensityLevel(1, "Low",
                "Values independence and achievement but can tolerate failure without identity collapse."),
            IntensityLevel(2, "Moderate",
                "Significant sensitivity to failure, loss of control, or dependence on others; "
                "these events are experienced as deeply threatening to self-image."),
            IntensityLevel(3, "High",
                "Complete identity collapse in response to failure or loss of control; "
                "catastrophic self-devaluation from a single performance failure."),
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # 2. PRECIPITATION: TRIGGERING EVENTS & APPRAISALS
    # ══════════════════════════════════════════════════════════════════════════

    TaxonomyVar(
        name="nature_of_stressor",
        category="Precipitation",
        subcategory="Triggering Event",
        clinical_definition=(
            "The objective nature of the precipitating stressor. Beck & Bredemeier identify key "
            "stressor types: loss of a relationship (separation, divorce, death), loss of status "
            "or role (job loss, demotion), humiliation or public failure, rejection, or chronic "
            "stressors (financial strain, marital discord, caretaking burden). The transition from "
            "stressor to depression depends on appraisal — the same event affects people differently."
        ),
        clinical_notes=(
            "Score the nature and severity of the OBJECTIVE stressor, not the person's reaction to it "
            "(that is captured by cognitive_appraisal). A manageable daily hassle scores L1; "
            "losing one's job or primary relationship scores L2; total collapse of life structure scores L3. "
            "Chronic stressors (e.g., years of financial hardship) are considered more severe than acute ones."
        ),
        prototypical_expressions=[
            "I lost my job recently",
            "My marriage ended / I went through a divorce",
            "Someone close to me died",
            "I've been under financial strain for years",
            "I was rejected / humiliated publicly",
            "I have been a caretaker for a sick family member",
            "The pressure at work has been relentless",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No identifiable precipitating stressor mentioned."),
            IntensityLevel(1, "Low",
                "Minor or manageable stressor — a setback, a difficult period, without involving "
                "major loss of a vital resource."),
            IntensityLevel(2, "Moderate",
                "Clear loss or threat to a vital resource: job loss, relationship breakdown, "
                "bereavement, chronic financial or relational stressor."),
            IntensityLevel(3, "High",
                "Multiple simultaneous losses, or a catastrophic, life-defining stressor "
                "(e.g., total financial ruin, loss of a child, severe chronic illness)."),
        ],
    ),

    TaxonomyVar(
        name="cognitive_appraisal",
        category="Precipitation",
        subcategory="Appraisal of Loss",
        clinical_definition=(
            "The individual's subjective interpretation of the stressor, which determines whether "
            "it triggers the depression program. Three appraisal dimensions are critical: "
            "(a) vital resource investment — the degree to which the lost resource was central "
            "to identity and goals; (b) irreversibility — the perception that the loss is "
            "permanent and beyond personal control; (c) global devaluation — generalizing from "
            "a specific loss to a catastrophic conclusion about the self ('I lost my job' → 'I am worthless')."
        ),
        clinical_notes=(
            "Appraisal is the cognitive mediator between stressor and depression. "
            "The same event can be appraised as manageable (L0-1) or as a total, irreversible "
            "self-defining failure (L3). Look for language indicating permanence, uncontrollability, "
            "and self-generalization. Distinguish from the stressor itself (nature_of_stressor) — "
            "this variable captures what the person MAKES of the event."
        ),
        prototypical_expressions=[
            "I'll never get over this",
            "This defines who I am as a failure",
            "There's nothing I can do to change this",
            "My whole identity was tied to that job / relationship",
            "I can't see any way forward",
            "This loss has made me realize I'm fundamentally worthless",
            "It's permanent — things will never go back to how they were",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent / Manageable",
                "The event is not mentioned, or is explicitly framed as temporary and manageable."),
            IntensityLevel(1, "Specific Loss",
                "The loss is acknowledged as real but confined to one domain; "
                "the person does not yet generalize it to their core identity."),
            IntensityLevel(2, "Vital Resource Loss",
                "The lost resource (job, partner, health) is described as central to the person's "
                "identity or goals; the loss feels significant and hard to recover from."),
            IntensityLevel(3, "Catastrophic / Irreversible",
                "The loss is appraised as total, permanent, and beyond control; "
                "the person has moved to global self-devaluation "
                "(e.g., 'I am worthless', 'I will never recover', 'I have nothing left')."),
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # 3. THE DEPRESSION PROGRAM (Outcome)
    # ══════════════════════════════════════════════════════════════════════════

    TaxonomyVar(
        name="anergia",
        category="Outcome",
        subcategory="Energy Conservation",
        clinical_definition=(
            "Physical fatigue, psychomotor retardation, and reduced activity level consistent with "
            "Beck & Bredemeier's 'sickness behavior' model: the organism conserves energy following "
            "a perceived vital loss. Distinct from sadness — this is the somatic, behavioral "
            "expression of energy shutdown. Includes physical exhaustion, slowing of movement "
            "and speech, and difficulty initiating or sustaining activities."
        ),
        clinical_notes=(
            "Distinguish from anhedonia (loss of pleasure) and social withdrawal (interpersonal "
            "disengagement). Anergia is primarily a physical, somatic complaint: the body is "
            "shutting down to conserve resources. Key markers: 'I can't get out of bed', "
            "'everything requires enormous effort', 'I move and think more slowly than usual'."
        ),
        prototypical_expressions=[
            "I have no energy for anything",
            "Getting out of bed is a struggle",
            "Everything takes enormous effort",
            "My body feels heavy and slow",
            "I can barely function physically",
            "Simple tasks exhaust me completely",
            "I can't seem to get going no matter how hard I try",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No physical fatigue or energy impairment reported."),
            IntensityLevel(1, "Mild",
                "Mild fatigue or reduced energy — still able to function, though with effort."),
            IntensityLevel(2, "Moderate",
                "Clear somatic exhaustion; activities require disproportionate effort; "
                "significant slowdown in daily functioning."),
            IntensityLevel(3, "Severe",
                "Psychomotor retardation; inability to initiate basic activities (get out of bed, "
                "maintain hygiene); near-complete physical shutdown."),
        ],
    ),

    TaxonomyVar(
        name="anhedonia",
        category="Outcome",
        subcategory="Energy Conservation",
        clinical_definition=(
            "Loss of interest or pleasure in previously valued activities, goals, relationships, "
            "food, or sex. Beck & Bredemeier frame this as an adaptive energy-conservation "
            "mechanism: the organism disengages from goals that feel unattainable given the "
            "perceived resource loss. Clinically it manifests as emotional flatness, indifference "
            "to previously meaningful activities, and reduced motivation."
        ),
        clinical_notes=(
            "Distinguish from temporary boredom or loss of interest in a specific activity. "
            "Anhedonia is pervasive and represents a qualitative change from the person's baseline. "
            "Key marker: activities that used to bring pleasure no longer do. "
            "Often expressed as 'nothing feels worth it anymore' or 'I don't enjoy anything'."
        ),
        prototypical_expressions=[
            "I've lost interest in things I used to love",
            "Nothing feels enjoyable anymore",
            "I don't get pleasure from anything",
            "I used to love going out / cooking / exercising — now I can't bring myself to do it",
            "Everything feels flat and pointless",
            "I've stopped caring about the things that used to matter to me",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No loss of pleasure or interest reported."),
            IntensityLevel(1, "Mild",
                "Reduced enjoyment in some activities; still able to experience pleasure occasionally."),
            IntensityLevel(2, "Moderate",
                "Clear loss of interest in most previously valued activities; "
                "difficulty finding anything rewarding."),
            IntensityLevel(3, "Severe",
                "Pervasive, total anhedonia — nothing brings pleasure; "
                "complete withdrawal from all previously valued goals and activities."),
        ],
    ),

    TaxonomyVar(
        name="social_withdrawal",
        category="Outcome",
        subcategory="Energy Conservation",
        clinical_definition=(
            "Active disengagement from social relationships and activities, understood here as an "
            "energy-conservation behavior in the depression program. The person reduces or "
            "eliminates social contact to limit the metabolic and emotional cost of interaction. "
            "Distinct from introversion or social anxiety — this is a change from the person's "
            "baseline social behavior."
        ),
        clinical_notes=(
            "Important: in Beck's model, social withdrawal as an OUTCOME is driven by energy "
            "conservation (sickness behavior), not primarily by fear or shame. It is a behavioral "
            "shutdown, not an avoidance strategy. The person often knows they should reach out "
            "but cannot muster the energy or motivation to do so."
        ),
        prototypical_expressions=[
            "I've been isolating myself from everyone",
            "I stopped returning calls and messages",
            "I've cancelled all my plans and see no one",
            "I've pulled away from my friends and family",
            "I don't have the energy to be around people",
            "I used to be social — now I avoid everyone",
            "Being with other people feels impossible right now",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No change in social engagement reported."),
            IntensityLevel(1, "Mild",
                "Slight reduction in social activity; still maintains some connections."),
            IntensityLevel(2, "Moderate",
                "Active withdrawal: cancelling plans, not responding to contacts, "
                "significant reduction in social participation."),
            IntensityLevel(3, "Severe",
                "Complete social isolation; no contact with friends or family; "
                "total withdrawal from social life."),
        ],
    ),

    TaxonomyVar(
        name="hypersomnia_appetite",
        category="Outcome",
        subcategory="Atypical Strategies",
        clinical_definition=(
            "Atypical energy-replenishment behaviors: excessive sleep (hypersomnia) and/or "
            "increased appetite, understood as the organism's attempt to restore energy reserves "
            "following the perceived vital loss. These are the opposite of classic insomnia and "
            "appetite loss — they represent a distinct subtype of the depression program."
        ),
        clinical_notes=(
            "Distinguish from insomnia/agitation (protective vigilance). "
            "Here the person sleeps too much and/or eats more. Often co-occurs with anergia. "
            "In the evolutionary model, hypersomnia and increased appetite are homeostatic "
            "responses to the energy debt created by the depression program."
        ),
        prototypical_expressions=[
            "I sleep much more than usual but still feel exhausted",
            "I've been eating more to cope",
            "I can't stop sleeping — I sleep 12, 14 hours and it's not enough",
            "I've been eating everything in sight without feeling satisfied",
            "Food is the only thing that brings me any comfort",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No hypersomnia or increased appetite reported."),
            IntensityLevel(1, "Mild",
                "Slightly increased sleep duration or appetite without significant impairment."),
            IntensityLevel(2, "Moderate",
                "Clear hypersomnia (sleeping significantly more than usual) and/or "
                "notable increase in appetite or food intake."),
            IntensityLevel(3, "Severe",
                "Extreme hypersomnia (sleeping most of the day) and/or compulsive eating; "
                "these behaviors are impairing daily functioning."),
        ],
    ),

    TaxonomyVar(
        name="protective_vigilance",
        category="Outcome",
        subcategory="Protective Vigilance",
        clinical_definition=(
            "Paradoxical activation within the depression program: psychomotor agitation, "
            "irritability, insomnia, and hypervigilance that coexist with exhaustion. "
            "Beck & Bredemeier interpret this as an evolutionary protective mechanism: "
            "the organism maintains environmental monitoring (to detect further threats) "
            "even while conserving energy. Clinically: racing thoughts, inability to sleep "
            "or relax, restlessness, irritability."
        ),
        clinical_notes=(
            "The co-occurrence of exhaustion AND agitation/insomnia is the hallmark of this "
            "construct. The person is tired but cannot rest. Distinguish from anxiety disorder — "
            "here the vigilance is specifically linked to the depression program and the perceived "
            "threat following the vital loss."
        ),
        prototypical_expressions=[
            "I'm exhausted but I can't sleep",
            "My mind won't stop — I can't turn it off even when I want to rest",
            "I feel restless and on edge all the time",
            "I snap at people even when I have no energy",
            "I'm wired and exhausted at the same time",
            "I lie awake for hours even though I'm completely drained",
            "I can't concentrate — my mind keeps racing",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No agitation, insomnia, or hypervigilance reported."),
            IntensityLevel(1, "Mild",
                "Some difficulty relaxing or mild sleep disruption; manageable restlessness."),
            IntensityLevel(2, "Moderate",
                "Clear insomnia or psychomotor agitation coexisting with exhaustion; "
                "irritability; difficulty concentrating due to racing thoughts."),
            IntensityLevel(3, "Severe",
                "Severe and persistent insomnia despite extreme fatigue; "
                "marked agitation; inability to rest or concentrate at all."),
        ],
    ),

    TaxonomyVar(
        name="cognitive_triad_self",
        category="Outcome",
        subcategory="Negative Cognitive Triad",
        clinical_definition=(
            "The self-directed arm of Beck's negative cognitive triad: viewing oneself as "
            "fundamentally defective, incompetent, unlovable, or a burden to others. "
            "This is not situational self-criticism but a global, stable, negative self-concept "
            "that the person treats as a fact rather than a distorted belief."
        ),
        clinical_notes=(
            "Distinguish from situational guilt or shame about a specific action. "
            "The triad self-view is global and characterological: 'I AM worthless / incompetent / "
            "unlovable' not 'I did a bad thing'. In severe cases, this belief can extend to "
            "suicidal ideation: if one is fundamentally defective and a burden, death may feel "
            "like a logical conclusion."
        ),
        prototypical_expressions=[
            "I am fundamentally worthless",
            "I've never been good enough",
            "I'm a burden to everyone around me",
            "No one could truly love someone like me",
            "I am a failure as a person",
            "There's something deeply wrong with me",
            "I don't deserve to be happy or loved",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No global negative self-concept; self-criticism is situational."),
            IntensityLevel(1, "Mild",
                "Occasional self-doubt or negative self-appraisal, but not pervasive or global."),
            IntensityLevel(2, "Moderate",
                "Dominant negative self-view; frequent self-criticism; "
                "self-described as incompetent, unlovable, or a burden."),
            IntensityLevel(3, "Severe",
                "Absolute belief in own worthlessness or defectiveness; "
                "possible suicidal ideation or wish to disappear."),
        ],
    ),

    TaxonomyVar(
        name="cognitive_triad_world",
        category="Outcome",
        subcategory="Negative Cognitive Triad",
        clinical_definition=(
            "The world-directed arm of Beck's negative cognitive triad: perceiving the social "
            "and physical environment as hostile, rejecting, unfair, or a permanent source of "
            "deprivation. The world is experienced as fundamentally unsafe or unrewarding."
        ),
        clinical_notes=(
            "Distinguish from realistic assessments of specific social problems. "
            "The triad world-view is global and generalized: 'the world / people / life IS "
            "hostile/unfair' not 'this specific person treated me badly'. "
            "Key markers: cynicism, generalized mistrust, perception that the environment "
            "will always deprive or harm."
        ),
        prototypical_expressions=[
            "Nobody really cares about anyone",
            "The world is a hostile and unfair place",
            "People are fundamentally selfish and untrustworthy",
            "No matter what I do, the environment works against me",
            "Life is set up to make people like me fail",
            "I can never get a fair chance",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No global negative worldview; criticism is situation-specific."),
            IntensityLevel(1, "Mild",
                "Some cynicism or perception of unfairness in specific areas."),
            IntensityLevel(2, "Moderate",
                "Generalized view of the world as hostile, rejecting, or depriving; "
                "pervasive distrust of others."),
            IntensityLevel(3, "Severe",
                "Absolute conviction that the world is irredeemably hostile; "
                "complete sense of being trapped in a threatening environment with no escape."),
        ],
    ),

    TaxonomyVar(
        name="cognitive_triad_future",
        category="Outcome",
        subcategory="Negative Cognitive Triad",
        clinical_definition=(
            "The future-directed arm of Beck's negative cognitive triad: hopelessness and "
            "helplessness. The person anticipates that suffering will continue indefinitely "
            "and that no action can change the outcome. This is the most dangerous component "
            "of the triad as it is the strongest predictor of suicidal behavior."
        ),
        clinical_notes=(
            "Hopelessness (nothing will ever get better) and helplessness (I can do nothing "
            "to change it) are distinct but often co-occur. Both must be assessed. "
            "The severity of this construct is among the most important clinical indicators: "
            "severe hopelessness in the context of global self-devaluation is a major suicide risk factor."
        ),
        prototypical_expressions=[
            "Things will never get better",
            "There's no point in trying",
            "I can't see any future for myself",
            "No matter what I do, nothing will change",
            "I have no hope left",
            "The future looks completely dark",
            "I'll never feel okay again",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "Future is viewed with some hope or uncertainty, not despair."),
            IntensityLevel(1, "Mild",
                "Some worry about the future; pessimistic expectations but not absolute."),
            IntensityLevel(2, "Moderate",
                "Clear hopelessness or helplessness; difficulty imagining a positive future; "
                "sense that effort is pointless."),
            IntensityLevel(3, "Severe",
                "Absolute hopelessness and helplessness; no conceivable positive future; "
                "possibly expressed as a wish to not exist."),
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # 4. MAINTENANCE vs. REVERSAL FACTORS (Moderators)
    # ══════════════════════════════════════════════════════════════════════════

    TaxonomyVar(
        name="rumination",
        category="Maintenance",
        subcategory="Maintenance Factors",
        clinical_definition=(
            "Repetitive, self-focused negative thinking in response to distress — a key "
            "maintenance mechanism that prolongs and deepens depression. "
            "Rumination involves passively and repetitively dwelling on one's symptoms, "
            "their causes, and their consequences, without moving toward active problem-solving. "
            "Susan Nolen-Hoeksema's Response Styles Theory identifies rumination as the primary "
            "cognitive mechanism that transforms low mood into clinical depression."
        ),
        clinical_notes=(
            "Distinguish from normal reflection or grief. Rumination is characterized by: "
            "(a) repetitiveness — the same thoughts recur without resolution; "
            "(b) passivity — the person does not problem-solve; "
            "(c) self-focus — attention is directed inward to symptoms and failings. "
            "Also distinguish from info_processing_biases (trait-level cognitive style): "
            "rumination is a state-level process triggered by the current depressive episode."
        ),
        prototypical_expressions=[
            "I keep going over the same things in my head without getting anywhere",
            "The same thoughts come back again and again — I can't stop them",
            "I replay conversations and mistakes over and over",
            "My mind gets stuck in loops I can't break out of",
            "I lie awake thinking about everything that went wrong",
            "I keep asking myself why this happened to me",
            "I analyze my problems endlessly but never find an answer",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No pattern of repetitive negative self-focused thinking."),
            IntensityLevel(1, "Mild",
                "Some repetitive worrying or dwelling on problems; manageable and occasionally resolvable."),
            IntensityLevel(2, "Moderate",
                "Clear pattern of uncontrolled repetitive negative thinking; "
                "thoughts return repeatedly without resolution; "
                "the person is aware they are stuck but cannot stop."),
            IntensityLevel(3, "Severe",
                "Consuming, uncontrollable thought loops; severe interference with daily functioning; "
                "the person feels completely trapped in their own mind."),
        ],
    ),

    TaxonomyVar(
        name="avoidant_coping",
        category="Maintenance",
        subcategory="Maintenance Factors",
        clinical_definition=(
            "Behavioral or cognitive strategies aimed at escaping or avoiding the distress "
            "associated with the depression program, rather than addressing its causes. "
            "Avoidant coping maintains depression by preventing the corrective experiences "
            "and problem-solving that would terminate the depression program. "
            "Includes behavioral avoidance (not dealing with the problem), "
            "distraction (alcohol, screens, overwork), and emotional suppression."
        ),
        clinical_notes=(
            "Distinguish from adaptive coping (problem-solving, help-seeking). "
            "Avoidant coping is characterized by the absence of engagement with the problem "
            "and often provides short-term relief at the cost of long-term maintenance. "
            "Social withdrawal as avoidance differs from social withdrawal as sickness behavior: "
            "here the motivation is escaping distress, not conserving energy."
        ),
        prototypical_expressions=[
            "I know I should deal with it but I keep putting it off",
            "I drink / watch TV / stay busy to avoid thinking about it",
            "I've been avoiding anything that reminds me of it",
            "I push the feelings down and try not to think about them",
            "I distract myself as much as possible",
            "I've stopped doing the things I know would help because I can't face them",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No avoidant coping pattern identified."),
            IntensityLevel(1, "Mild",
                "Occasional procrastination or distraction, but generally engages with problems."),
            IntensityLevel(2, "Moderate",
                "Clear pattern of avoiding the core problem; using distraction or suppression "
                "as primary coping; aware of the avoidance but unable to stop."),
            IntensityLevel(3, "Severe",
                "Complete behavioral and emotional disengagement from the problem; "
                "possibly using substances or compulsive behaviors to escape; "
                "avoidance is comprehensive and impairing."),
        ],
    ),

    TaxonomyVar(
        name="social_support",
        category="Reversal",
        subcategory="Reversal Factors",
        clinical_definition=(
            "The presence and quality of social support as a buffering and recovery factor. "
            "Beck & Bredemeier identify social support as a key reversal mechanism: "
            "the warmth, validation, and practical help of others can counteract feelings "
            "of worthlessness, modify the cognitive triad, and restore a sense of vital resources. "
            "Includes support from friends, family, romantic partners, or mental health professionals."
        ),
        clinical_notes=(
            "Assess both the PRESENCE of support and the person's ability to RECEIVE it. "
            "Someone may have support available but be unable to accept it (due to shame, "
            "sociotropy, or autonomy-based vulnerability). The buffering effect requires "
            "both availability and receptiveness."
        ),
        prototypical_expressions=[
            "My friends have really been there for me",
            "I have a therapist I trust",
            "My family has been incredibly supportive",
            "Talking to someone who understands has helped",
            "I feel less alone because people care about me",
            "My partner has been a huge source of support",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No social support mentioned or available."),
            IntensityLevel(1, "Low",
                "Some social presence but superficial or unreliable support."),
            IntensityLevel(2, "Moderate",
                "Clear and active support from at least one person (friend, family, therapist); "
                "the person can receive and benefit from this support."),
            IntensityLevel(3, "High",
                "Strong, multi-source support network actively engaged in the person's recovery; "
                "including professional therapeutic support."),
        ],
    ),

    TaxonomyVar(
        name="cognitive_reappraisal",
        category="Reversal",
        subcategory="Reversal Factors",
        clinical_definition=(
            "Active cognitive work to correct distorted appraisals and modify the negative "
            "cognitive triad. Beck & Bredemeier identify cognitive reappraisal as the primary "
            "therapeutic reversal mechanism: helping the person recognize that their appraisals "
            "of loss as catastrophic, irreversible, and self-defining are distortions, not facts. "
            "Includes both spontaneous reappraisal and therapist-guided cognitive restructuring."
        ),
        clinical_notes=(
            "Distinguish from superficial positive thinking or denial. "
            "Genuine reappraisal involves examining the evidence, questioning global conclusions, "
            "and developing more balanced, realistic interpretations. "
            "It does NOT require abandoning all negative feelings — just correcting "
            "disproportionate or distorted interpretations."
        ),
        prototypical_expressions=[
            "I'm trying to see it differently",
            "My therapist is helping me challenge my automatic thoughts",
            "I've started questioning whether my interpretations are accurate",
            "I'm learning to separate what happened from what it means about me",
            "I realized my thinking was distorted — I was catastrophizing",
            "I'm working on not defining myself by this failure",
        ],
        intensity_levels=[
            IntensityLevel(0, "Absent", "No evidence of cognitive reappraisal or restructuring."),
            IntensityLevel(1, "Low",
                "Tentative attempts to think differently; not yet sustained or effective."),
            IntensityLevel(2, "Moderate",
                "Active engagement in reappraisal, either spontaneously or through therapy; "
                "some success in challenging distorted beliefs."),
            IntensityLevel(3, "High",
                "Sustained and effective cognitive restructuring; "
                "clear evidence of modified beliefs and more balanced interpretations."),
        ],
    ),
]

# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_taxonomy_block() -> str:
    """
    Render the full taxonomy as a structured text block for inclusion in an LLM prompt.
    Groups variables by category.
    """
    from collections import defaultdict
    by_category: dict = defaultdict(list)
    for var in TAXONOMY:
        by_category[var.category].append(var)

    lines = []
    for category, vars_in_cat in by_category.items():
        lines.append(f"\n## {category.upper()}")
        for var in vars_in_cat:
            lines.append(f"\n### {var.name}  [{var.subcategory}]")
            lines.append(f"Definition: {var.clinical_definition}")
            lines.append(f"Clinical notes: {var.clinical_notes}")
            lines.append("Intensity levels:")
            for lvl in var.intensity_levels:
                lines.append(f"  Level {lvl.level} - {lvl.label}: {lvl.clinical_criteria}")
    return "\n".join(lines)


# Convenience lookup
TAXONOMY_BY_NAME: dict[str, TaxonomyVar] = {v.name: v for v in TAXONOMY}
TAXONOMY_VAR_NAMES: list[str] = [v.name for v in TAXONOMY]