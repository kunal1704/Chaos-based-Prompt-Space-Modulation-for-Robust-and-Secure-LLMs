# Paper Story

## Working title

Chaos-Inspired Feature Expansion as a Nonlinear Lens on Frozen Transformer Embeddings

## Recommended framing

The safest and most publishable near-term framing is not "Chaos improves LLM robustness" and not "a new LLM architecture." The strongest framing is:

> We introduce a deterministic chaos-inspired feature expansion as a diagnostic intervention for frozen transformer embedding spaces, and test whether its effects are distinguishable from matched stochastic perturbations and standard representation-analysis baselines.

This story makes the current proof-of-concept scientifically defensible because it treats ChaosFEX as a representation-analysis lens first, not as a performance-improvement mechanism.

## Core claim to test

ChaosFEX may induce structured, deterministic nonlinear perturbations of token embeddings that differ from Gaussian noise and preserve or reveal aspects of local geometry in ways measurable by representation similarity and neighborhood metrics.

This is a hypothesis, not yet a result.

## Ranked research stories

| Rank | Story | Recommendation | Novelty | Risk | Complexity | Publication potential |
|---:|---|---|---:|---:|---:|---:|
| 1 | Story D: Chaos as a nonlinear dynamical lens for studying transformer representations | Primary | 8/10 | Medium | Medium | Strong workshop, plausible arXiv |
| 2 | Story B: Chaos-inspired representation probing for LLMs | Secondary | 7/10 | Medium | Medium | Strong workshop if probes are rigorous |
| 3 | Story C: Chaos-inspired geometry analysis of transformer embeddings | Supporting angle | 6/10 | Low-medium | Low | Good as analysis section |
| 4 | Story A: Chaos-inspired robustness mechanism for frozen LLMs | Defer | 5/10 | High | Medium-high | Risky without task wins |
| 5 | Story E: ChaosFEX as deterministic reservoir-style random feature map for frozen LLMs | Backup framing | 7/10 | Medium | Medium | Good ML workshop angle |

## Story A: Chaos-inspired robustness mechanism for frozen LLMs

Novelty score: 5/10.

Risk level: high.

Experimental complexity: medium to high.

Publication potential: weak for ACL/EMNLP main; possible for safety/robustness workshops if results are strong.

Strongest venues: BlackboxNLP, NLP4ConvAI safety workshops, NeurIPS/ICLR robustness or trustworthy ML workshops.

Reviewer concerns:

- Current code does not evaluate robustness.
- Additive embedding perturbation could degrade semantics rather than improve robustness.
- Gaussian/noise/random-feature controls are mandatory.
- Prompt-injection robustness claims require careful threat models and task metrics.

Verdict: do not lead with this in a 10-15 day arXiv sprint.

## Story B: Chaos-inspired representation probing for LLMs

Novelty score: 7/10.

Risk level: medium.

Experimental complexity: medium.

Publication potential: good workshop potential if framed as a probing method with controls.

Strongest venues: BlackboxNLP, ACL/EMNLP findings only if experiments are strong, NeurIPS interpretability workshops.

Reviewer concerns:

- Probes can memorize; selectivity and control tasks are required.
- ChaosFEX may be just another nonlinear feature map.
- Need simple baselines: linear probe, MLP probe, random Fourier features, polynomial features.

Verdict: viable, but only after showing ChaosFEX has distinctive behavior beyond generic nonlinearity.

## Story C: Chaos-inspired geometry analysis of transformer embeddings

Novelty score: 6/10.

Risk level: low to medium.

Experimental complexity: low.

Publication potential: good as a section, too thin alone unless linked to clear hypotheses.

Strongest venues: NLP interpretability workshops, representation learning workshops.

Reviewer concerns:

- PCA/t-SNE/UMAP plots are suggestive but not evidence.
- Need quantitative geometry metrics, not only visualization.
- Need multiple models and datasets to avoid cherry-picking.

Verdict: useful as the empirical backbone of the paper.

## Story D: Chaos as a nonlinear dynamical lens for studying transformer representations

Novelty score: 8/10.

Risk level: medium.

Experimental complexity: medium.

Publication potential: strongest near-term story.

Strongest venues: BlackboxNLP, NeurIPS interpretability workshops, ICLR mechanistic interpretability workshops, Complex Systems + ML venues.

Reviewer concerns:

- Avoid vague dynamical-systems metaphors.
- Make the dynamical object precise: the skew-tent trajectory used as a deterministic coordinate-wise feature generator.
- Compare to reservoir computing and random features.
- Demonstrate that chaos-specific parameters affect measured representation properties.

Verdict: best primary story.

## Story E: Stronger discovered framing

Proposed framing: ChaosFEX as a deterministic reservoir-style feature map for frozen LLM representation analysis.

Novelty score: 7/10.

Risk level: medium.

Experimental complexity: medium.

Publication potential: good for ML workshops and as a bridge to journal extension.

Reviewer concerns:

- Must distinguish from reservoir computing, random kitchen sinks, and kernel methods.
- Must not overclaim biological inspiration.
- Needs ablations over map parameters, trajectory length, and reduction method.

Verdict: use this as theoretical positioning inside Story D.

## Recommended paper outline

1. Introduction: frozen LLM embeddings are high-dimensional, structured, and hard to analyze; deterministic nonlinear probes may expose geometry not captured by linear methods.
2. Background: Transformer representation analysis, probing, geometry metrics, ChaosFEX/Neurochaos Learning, reservoir computing.
3. Method: extract frozen embeddings; normalize; apply ChaosFEX; reduce; compare representation shifts and optional probes.
4. Baselines: Gaussian noise, random projection, random Fourier features, shuffled ChaosFEX, nonchaotic map.
5. Experiments: geometry, representation similarity, prompt perturbation sensitivity, small probing task.
6. Results: report only if chaos-specific differences survive controls.
7. Limitations: no training, no robustness claim unless supported, small models, embedding-level intervention only.

## ArXiv feasibility in 10-15 days

Feasible if the paper is positioned as an exploratory representation-analysis study with honest limitations. Not feasible if positioned as a new robust LLM method without compelling task evidence.

## Go / no-go recommendation

Recommendation: PIVOT.

Continue the current technical direction, but pivot the story from "Chaos-based prompt modulation for robustness/security" to "Chaos-inspired nonlinear representation analysis of frozen transformer embeddings."

Rationale:

- The integration is novel enough to explore.
- Robustness/security claims are not yet supported.
- Representation analysis can be tested quickly and defensibly.
- Reviewers will accept negative or mixed results more readily under an analysis framing than under a method-improves-LLMs framing.
- A journal extension could later add broader models, mathematical analysis of the feature map, and downstream task studies.
