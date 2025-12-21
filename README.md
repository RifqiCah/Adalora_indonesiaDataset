Abstract—The rapid scaling of Pre-trained Language Models
(PLMs) has made full fine-tuning computationally prohibitive,
necessitating the adoption of Parameter-Efficient Fine-Tuning
(PEFT) methods. While techniques such as Low-Rank Adaptation (LoRA) have proven effective, they typically employ a
fixed-rank allocation strategy that ignores the heterogeneous
importance of different model layers. This paper investigates
the efficacy of Adaptive Low-Rank Adaptation (AdaLoRA) for
fine-tuning compact Transformer models on Indonesian language
tasks, a domain currently underrepresented in PEFT research.
We evaluate the proposed method across three Natural Language
Processing (NLP) domains: Natural Language Understanding
(NLU) using bert-tiny, Question Answering (QA) using
bert-tiny on SQuAD-id, and Natural Language Generation
(NLG) using mt5-small on WikiLingua. Our experiments
demonstrate extreme parameter efficiency, updating as few as
0.17% to 0.31% of total parameters. Results show mixed performance: while the model achieves high accuracy on tokenlevel tasks like NER (87.33%), it suffers from low recall due to
class imbalance. Similarly, for sentence-level tasks like Sentiment
Analysis, it reaches a moderate accuracy of 54.40%, but faces
convergence challenges in complex generative tasks. This study
concludes that adaptive rank allocation is a viable strategy
for deploying efficient NLP models on resource-constrained
edge devices, though larger parameter budgets are required for
complex reasoning tasks.
