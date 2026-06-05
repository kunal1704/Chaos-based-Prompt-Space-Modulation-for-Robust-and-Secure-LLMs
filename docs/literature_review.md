# Literature Review

This review focuses on the defensible research question for the current repository:

Can a ChaosFEX/Neurochaos feature expansion serve as a meaningful nonlinear lens or intervention for frozen transformer embedding spaces, beyond standard noise and representation-analysis baselines?

## Transformer representation analysis

### Tenney, Das, and Pavlick, 2019. "BERT Rediscovers the Classical NLP Pipeline." ACL.

Source: https://aclanthology.org/P19-1452/

Summary: Uses edge probing to localize linguistic information across BERT layers. The paper finds a rough progression from lower-level syntactic information to higher-level semantic/coreference information across depth.

Relevance: Establishes the standard style of asking where information is represented inside a frozen transformer.

Novelty comparison: Our project does not yet localize linguistic information across layers. A chaos-inspired method would need to show that ChaosFEX exposes or perturbs representational information differently from ordinary probes.

### Clark, Khandelwal, Levy, and Manning, 2019. "What Does BERT Look At? An Analysis of BERT's Attention." BlackboxNLP.

Source: https://aclanthology.org/W19-4828/

Summary: Analyzes BERT attention heads and identifies syntactic/coreference patterns in some heads.

Relevance: Useful contrast: attention analysis studies internal routing, while this project currently studies embedding-level transformations.

Novelty comparison: ChaosFEX is not an attention interpretability method. If the project claims interpretability, it should connect effects to representation geometry or downstream probing, not attention-head semantics.

### Rogers, Kovaleva, and Rumshisky, 2020. "A Primer in BERTology." TACL.

Source: https://aclanthology.org/2020.tacl-1.54/

Summary: Surveys what was known about BERT's linguistic knowledge, attention patterns, probing, and interpretability limitations.

Relevance: Frames reviewer expectations: probing results are easy to overinterpret and require controls.

Novelty comparison: ChaosFEX must be positioned as a controlled diagnostic lens, not a vague "BERTology" plot generator.

### Hewitt and Manning, 2019. "A Structural Probe for Finding Syntax in Word Representations." NAACL.

Source: https://nlp.stanford.edu/pubs/hewitt2019structural.pdf

Summary: Learns a linear transformation under which squared distances encode syntactic tree distances.

Relevance: Canonical example of geometry-based probing.

Novelty comparison: ChaosFEX is nonlinear and deterministic, but the burden is to show it reveals properties missed by linear probes.

### Hewitt and Liang, 2019. "Designing and Interpreting Probes with Control Tasks." EMNLP-IJCNLP.

Source: https://aclanthology.org/D19-1275/

Summary: Introduces control tasks and selectivity to avoid mistaking probe capacity for encoded linguistic knowledge.

Relevance: Essential for any Story B probing claim.

Novelty comparison: A ChaosFEX probe must include selectivity, label-randomization controls, and capacity-matched baselines.

## Representation geometry and similarity

### Jolliffe, 2002. "Principal Component Analysis." Springer.

Source: https://link.springer.com/doi/10.1007/978-3-642-04898-2_455

Summary: Standard reference for PCA as a linear variance-preserving projection.

Relevance: PCA is a first-pass diagnostic for anisotropy, dominant directions, and spectral concentration in embeddings.

Novelty comparison: ChaosFEX cannot claim geometry insight from PCA plots alone; PCA should be a baseline visualization and spectrum summary.

### van der Maaten and Hinton, 2008. "Visualizing Data using t-SNE." JMLR.

Source: https://www.jmlr.org/papers/v9/vandermaaten08a.html

Summary: Introduces t-SNE for nonlinear visualization of high-dimensional data while addressing crowding problems.

Relevance: Useful visualization for token/prompt clusters after chaos modulation.

Novelty comparison: t-SNE can mislead. It should be paired with quantitative neighbor preservation and repeated seeds.

### McInnes, Healy, Saul, and Grossberger, 2018. "UMAP: Uniform Manifold Approximation and Projection." JOSS.

Source: https://joss.theoj.org/papers/10.21105/joss.00861

Summary: Presents UMAP as a manifold-learning method for dimension reduction and visualization.

Relevance: UMAP is a stronger visualization baseline than t-SNE for neighborhood/manifold comparisons.

Novelty comparison: Any chaos-geometry claim should use UMAP only as visualization, not proof.

### Raghu, Gilmer, Yosinski, and Sohl-Dickstein, 2017. "SVCCA." NeurIPS.

Source: https://papers.neurips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability

Summary: Proposes Singular Vector Canonical Correlation Analysis for comparing neural representations.

Relevance: Useful for comparing original, chaos-modulated, and noise-modulated representations across layers.

Novelty comparison: ChaosFEX novelty depends on producing similarity patterns not explained by norm-matched perturbations.

### Kornblith, Norouzi, Lee, and Hinton, 2019. "Similarity of Neural Network Representations Revisited." ICML.

Source: https://arxiv.org/abs/1905.00414

Summary: Introduces CKA as a representation similarity measure with advantages over some CCA variants.

Relevance: CKA is a core metric for measuring how much chaos modulation changes embeddings and hidden states.

Novelty comparison: CKA is a baseline metric, not a competing method. It can make the chaos intervention measurable.

## Dynamical systems views of transformers

### Katharopoulos, Vyas, Pappas, and Fleuret, 2020. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML.

Source: https://proceedings.mlr.press/v119/katharopoulos20a

Summary: Shows that linear attention admits a recurrent formulation for autoregressive transformers.

Relevance: Supports the broader idea that transformer computation can be examined through sequence/dynamics formalisms.

Novelty comparison: This is architectural/theoretical, whereas our current project is an external embedding-level feature transformation.

### Fernando and Guitchounts, 2025. "Transformer Dynamics: A neuroscientific approach to interpretability of large language models."

Source: https://arxiv.org/abs/2502.12131

Summary: Proposes studying LLMs with a dynamical-systems and neuroscience-inspired interpretability framework.

Relevance: Very close in spirit to Story D and raises the bar for precise definitions.

Novelty comparison: Our project should avoid broad "transformer dynamics" claims unless the dynamical object is exactly the ChaosFEX skew-tent feature generator and its induced representation shift.

### Engels et al., 2024. "Transformers Represent Belief State Geometry in their Residual Stream." NeurIPS.

Source: https://papers.nips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html

Summary: Studies residual-stream geometry and belief-state structure, including nontrivial geometries.

Relevance: Shows that transformer residual streams can have interpretable geometry beyond surface-level visualization.

Novelty comparison: Our project currently operates at input embeddings, not residual streams. Extending analysis across layers would make the story stronger.

## Chaos theory in machine learning and Neurochaos Learning

### Yadav, Venkatesh, and Nagaraj, 2019. "ChaosNet: A Chaos based Artificial Neural Network Architecture for Classification."

Source: https://arxiv.org/abs/1910.02423

Summary: Introduces ChaosNet, inspired by chaotic neuronal firing, for classification using chaos-derived features and simple decision rules.

Relevance: Foundational prior for the repository's ChaosFEX dependency.

Novelty comparison: ChaosNet is a classifier architecture for classical tasks, not a transformer embedding analysis method.

### Yadav, Venkatesh, and Nagaraj, 2019. "A Novel Chaos Theory Inspired Neuronal Architecture."

Source: https://arxiv.org/abs/1905.12601

Summary: Early statement of the chaos-inspired neural architecture concept.

Relevance: Background for the biological/dynamical motivation of ChaosFEX.

Novelty comparison: The LLM embedding bridge appears distinct, but the paper's biological inspiration should be cited carefully rather than overextended.

### NB, Harikrishnan, and Nagaraj, 2020. "A Neurochaos Learning Architecture for Genome Classification."

Source: https://arxiv.org/abs/2010.10995

Summary: Applies ChaosFEX+SVM to synthetic and genome classification settings, emphasizing low-sample regimes.

Relevance: Shows ChaosFEX as a feature expansion that can improve linear separability.

Novelty comparison: Strong prior for classical feature expansion. The novelty gap is applying ChaosFEX to frozen LLM embeddings and representation analysis rather than tabular/genome classification.

### Sethi et al., 2023. "Neurochaos feature transformation for Machine Learning." Integration.

Source: https://www.sciencedirect.com/science/article/abs/pii/S0167926023000202

Summary: Evaluates ChaosFEX/Neurochaos features with classical ML algorithms across benchmark datasets.

Relevance: The main empirical prior that ChaosFEX can act as a useful feature transformer.

Novelty comparison: Reviewers may say our method is "ChaosFEX applied to embeddings." We need transformer-specific hypotheses and controls.

### Harikrishnan et al., 2022. "Neurochaos Feature Transformation and Classification for Imbalanced Learning."

Source: https://arxiv.org/abs/2205.06742

Summary: Studies neurochaos feature transformation under imbalanced classification.

Relevance: Indicates existing ChaosFEX work already studies robustness-like data regimes in classical ML.

Novelty comparison: Do not claim general robustness novelty unless tested in LLM-specific conditions.

### Henry and Nagaraj, 2025. "Hyperparameter-Free Neurochaos Learning Algorithm for Classification."

Source: https://arxiv.org/abs/2508.01478

Summary: Proposes AutochaosNet to reduce hyperparameter tuning in Neurochaos Learning.

Relevance: Shows active development of Neurochaos methods and highlights hyperparameter sensitivity as a recognized issue.

Novelty comparison: Our experiments must vary ChaosFEX parameters or justify a fixed configuration.

## Reservoir computing and echo state networks

### Jaeger, 2001. "The echo state approach to analysing and training recurrent neural networks."

Source: cited in https://www.sciencedirect.com/science/article/pii/S1574013709000173

Summary: Introduces Echo State Networks, using a fixed recurrent reservoir and trained readout.

Relevance: Conceptual analogy: ChaosFEX can be framed as a deterministic feature map/reservoir-like expansion, though not recurrent over token sequences in the current code.

Novelty comparison: The project must avoid claiming to invent reservoir-style feature expansion.

### Maass, Natschlager, and Markram, 2002. "Real-time computing without stable states." Neural Computation.

Source: https://pubmed.ncbi.nlm.nih.gov/12433288/

Summary: Introduces Liquid State Machines, where transient dynamics provide computation.

Relevance: Supports dynamical feature-expansion framing.

Novelty comparison: ChaosFEX is coordinate-wise and deterministic; it is not a liquid-state sequence processor unless redesigned.

### Lukosevicius and Jaeger, 2009. "Reservoir computing approaches to recurrent neural network training." Computer Science Review.

Source: https://www.sciencedirect.com/science/article/pii/S1574013709000173

Summary: Survey of reservoir generation/adaptation and readout training.

Relevance: Necessary prior if Story E is used.

Novelty comparison: Our contribution should be a transformer-representation application and evaluation, not reservoir computing theory.

## Prior work combining ChaosFEX/ChaosNet with transformers or LLM embeddings

Searches for combinations of `ChaosFEX`, `ChaosNet`, `Neurochaos Learning`, `transformer embeddings`, and `LLM embeddings` did not reveal a direct prior applying ChaosFEX to GPT/BERT/LLM embedding spaces.

Closest adjacent work:

- Transformer-based forecasting/control for chaotic dynamical systems, e.g. "Chaos Meets Attention" and transformer control of chaos. These use transformers to model chaotic systems, not chaos features to analyze transformer embeddings.
- Dynamical-systems interpretability of transformers, which studies LLM internals but does not use ChaosFEX.
- Neurochaos Learning applied to tabular, genome, imbalanced, graph/linked-data, and classical classification tasks.

Current novelty hypothesis:

The novel contribution is likely the use of ChaosFEX as a deterministic nonlinear feature expansion/intervention on frozen transformer embeddings, evaluated against noise/random-feature/probing baselines.

Weakness:

This novelty is application-level unless the experiments show a transformer-specific representational phenomenon.
