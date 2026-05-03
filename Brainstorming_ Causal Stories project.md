Brainstorming: “Causal Stories” project

- Application 1: Subsidy design for poverty  
- Application 2: Personalized mentorship

Estimating causal effects of interventions (e.g., poverty, education) on societies from narrative testimonies (storytelling). The question is whether reliable causal estimates can be obtained from such data, given the large number of confounding factors.

Several problems in this approach:

- the large amount of unmeasured confounders (including time-varying confounding)  
- Identification under unconfoundedness: conditional on variables extracted from the text, is treatment assignment as good as random?  
- selection bias: we could correct for it with inverse probability of participation weighting, but only if you have auxiliary data on the target population to model participation.  
- No counterfactual structure  
- might be more…

Let’s say we estimated with no significant bias a causal effect. Then how can we guarantee this effect is easily generalizable? → transportability across population

1) **Objective: (24 April)**

Suppose we made all these assumptions and simplify the problem

* Input:   
  * Narrative (someone explaining story / therapy sessions) \* 1 millions of story  
  * Response: Depression  
* Output:  
  *  Causal Representation / DAG 

Example

* Input: A therapy transcript and target outcome: depression.  
    
* Output: Job loss → Financial stress → Sleep disruption → Depressive symptoms 

2) **Next meeting’s goal: How do we do this? (31 April)**  
   1) **We both come up with ideas and argue against each other's points.** 

**Causal Representation**

1) **Extract any candidate variable from the text.**

**Definition**. A candidate variable is any text-grounded concept that describes a state, event, behavior, condition, attribute, or action that could vary across units, people, or contexts.

**Rule**. Extract a text-grounded concept c as a candidate variable if there exists at least one plausible comparison case in which c is absent, different in kind, or different in intensity.

**Idea**: Counterfactual variation across cases.

**Example**: I have been feeling exhausted for the past few weeks. Work has been very stressful, and I keep staying late to finish everything. When I get home, I usually skip dinner and scroll on my phone until late at night. I have also stopped answering messages from my friends because I do not feel like talking.

| Text Span | Candidate Variable  | Why extracted |
| :---- | :---- | :---- |
| feeling exhausted | exhaustion | could be absent or vary in intensity |
| for the past few weeks | duration of exhaustion | could vary in length |
| “work has been very stressful” | work stress | could be absent or vary in intensity |
| staying late | overtime work | could be absent or vary in frequency |
| skip dinner | skipped meals | could be absent or vary in frequency |
| scroll on my phone until late at night | late-night phone use | could be absent or vary in duration |
| stopped answering messages | reduced social responsiveness | could be absent or vary in degree |
| do not feel like talking | low desire for social interaction | could be absent or vary in intensity |

2) **Construct edge probabilities given text** 

Given an unstructured text D, suppose the first stage extracts a set of candidate variables:

C \= {C1, C2, …, CK}.

The next step is to construct a directed edge-probability matrix. For every ordered pair (Ci, Cj), we estimate:

Pij(D) \= P(Ci → Cj | D),

where Pij(D) represents the probability, based only on the information in the text, that Ci has a directed causal relation toward Cj. The diagonal entries are set to zero, Pii(D) \= 0, since self-edges are not considered.

3) **Computation of edge probabilities using open source models**

For each ordered pair (Ci, Cj), create a prompt like:

`Document:`

`[D]`

`Candidate variables:`

`[C1, C2, ..., CK]`

`Target directed edge:`

`Ci → Cj`

`Other candidate variables that could lie between Ci and Cj:`

`[C \ {Ci, Cj}]`

`Task:`

`Decide whether the document supports a direct causal edge from Ci to Cj.`

`Definition of direct edge:`

`A direct edge Ci → Cj is supported only if the document suggests that Ci affects, changes, produces, worsens, improves, triggers, or contributes to Cj without requiring another listed candidate variable as an intermediate step.`

`Do not answer Yes if the document only supports an indirect pathway such as:`

`Ci → Ck → Cj`

`Do not answer Yes only because Ci happens before Cj or is associated with Cj.`

`Question:`

`Does the document support a direct causal edge Ci → Cj?`

`Answer with one word only: Yes or No.`

`Answer:`

Then read the next-token logits for `"Yes"` and `"No"`.

Compute:

P(Yes | D, claim \= Ci → Cj,C \\ {Ci, Cj},U)

where U represents the randomness in the evaluation procedure, such as prompt variation, sampling seed, few-shot examples, or model decoding conditions. 

Using softmax over the two logits:

P(Yes | D, claim \= Ci → Cj,C \\ {Ci, Cj},U) \= exp(logit(Yes)) / \[exp(logit(Yes)) \+ exp(logit(No))\]

Then separately ask the reverse claim:

Claim:

The text supports the directed causal relation: Cj → Ci.

and compute:

P(Yes | D, claim \= Cj → Ci, C \\ {Ci, Cj}, U)

For each directed claim Ci → Cj, we estimate the probability that the claim is supported by the text using the model’s Yes/No logits.

Let the edge probability be defined as the expected model support for the directed claim:

Pij(D) \= E\[ P(Yes | D, claim \= Ci → Cj, C \\ {Ci, Cj}, U)\],

Since this expectation is not available analytically, we approximate it by Monte Carlo. For b \= 1, …, B, draw an independent evaluation condition Ub and compute:

Pij,b(D) \= P(Yes | D, claim \= Ci → Cj, C \\ {Ci, Cj}, Ub).

Then the Monte Carlo approximation is:

Pij(D) ≈ (1/B) sum from b \= 1 to B Pij,b(D).

4) ### **Iteration Across Documents**

We repeat the full procedure for each document D^1, D^2, …, D^N. For each document D^n, we extract the candidate variables, construct the directed edge-probability matrix. After this is done for all documents, we construct a common dataset-level variable set:

C^all \= union of C^1, C^2, …, C^N.

This set contains all non-duplicated candidate variables.

Next, for each document D^n, we expand its edge-probability matrix so that it is defined over the common variable set C^all. If a variable appears in C but does not appear in document D^m, then all edge probabilities involving that variable are set to zero for that document. In other words, missing variables are treated as unsupported by that specific text.

Once every document has an edge-probability matrix over the same variable set C, we average the edge probabilities across documents. For each ordered pair (Ci, Cj), the dataset-level edge probability is defined as:

Pij(dataset) \= average of Pij(D^n) across n \= 1, …, N.

This dataset-level probability measures how consistently the directed relation Ci → Cj is supported across the full set of documents.

5) ### **Thresholding the Dataset-Level Edge Probabilities**

We then classify each dataset-level edge probability using two threshold parameters, tau\_low and tau\_high, where tau\_low \< tau\_high.

* If Pij(dataset) \>= tau\_high, classify Pij(dataset) as high.  
* If Pij(dataset) \<= tau\_low, classify Pij(dataset) as low.  
* If tau\_low \< Pij(dataset) \< tau\_high, classify Pij(dataset) as uncertain.

For each pair (Ci, Cj), we interpret the two directed probabilities Pij(dataset) and Pji(dataset) jointly:

* Pij high, Pji low → directed relation Ci → Cj.  
* Pij low, Pji high → directed relation Cj → Ci.  
* Pij low, Pji low → no supported connection.  
* Pij high, Pji high → bidirectional or mutually reinforcing relation.  
* Otherwise → uncertain relation.

The result is a dataset-level probabilistic directed causal representation. Unlike an individual graph, which reflects the causal story in one document, this graph summarizes causal relations that are repeatedly supported across the full dataset.

6) **Identify all variables that have a directed path into Y**

Given the full text-grounded graph, the response variable Y is selected from the set of candidate variables. Formally, let Y be one element of C. After constructing the directed edge-probability matrix, we obtain a graph over all candidate variables. This graph may contain many relations that are not directly relevant to Y. Therefore, we extract the response-specific subgraph by identifying all variables that have a directed path into Y.

A candidate variable Ci is treated as a possible cause of Y if there exists a directed path from Ci to Y in the constructed graph. This path may be direct, such as Ci → Y, or indirect, such as Ci → Cj → Y. The set of all such variables forms the ancestor set of Y.

Anc(Y) \= {Ci : there exists a directed path from Ci to Y}.

The response-specific causal representation is then defined as the subgraph containing Y, the variables in Anc(Y), and the directed paths connecting these variables to Y.

Remove everything else.

7) **Problem / Next steps**

We can first evaluate the empirical performance of this framework using simulation studies. In these simulations, we can control the true variables, the true graph, and the text-generation process, which allows us to assess whether the method correctly recovers candidate variables and directed relations and under what conditions (vary n, nodes, dimension etc.)

The main limitation of the current setup is that it still implicitly relies on an unconfoundedness-type assumption. That is, when we interpret an extracted edge Ci → Cj as a causal relation, we are assuming that the text contains enough information to rule out alternative explanations, such as omitted common causes or reverse causality.

The next step is therefore to relax this assumption. Instead of treating each extracted edge as directly causal, the framework should explicitly account for possible unobserved confounding, missing variables, and ambiguity in causal direction. This could be done by adding a later stage that identifies potential hidden common causes, flags edges that may be confounded, and reports whether each relation should be interpreted as causal, partially supported, or only hypothesis-generating.


IDEA 1 : Do IPW for disproportionate response variables. Have a weighted average. 
Example: Y = 0 (80%) Y = 1 (20%)
Inverse weighted average! 
IDEA 2: We have to normalize as much as possible for C^is :) 
