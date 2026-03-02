# 🧠 SemEval-2026 Task 11

Welcome to the official repository for **SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in Large Language Models**.

Official website: <https://sites.google.com/view/semeval-2026-task-11>

---

### Content Biases in LLMs

A major challenge for Large Language Models (LLMs) is their tendency to confuse formal logical validity with the content of arguments. This phenomenon, known as **content effect**, means LLMs can:

* Overestimate the validity of arguments that align with common knowledge.

* Underestimate the validity of arguments that seem implausible, even if they are logically sound.

* Exhibit biases towards validity assessment depending on the content of the argument (including concrete entities, terms, and languages)

This issue highlights a fundamental problem: the pre-training process inherently entangles reasoning with content, limiting the reliability and application of LLMs in critical real-world scenarios. While various methods have been proposed to address this, a truly effective solution remains elusive, especially across different languages.

--- 

### A Multilingual Evaluation of Content Effect on Reasoning

SemEval-2026 Task 11 aims to tackle this challenge by focusing on **multilingual syllogistic reasoning**. Participants will build models that can assess the formal validity of logical arguments, completely independent of their plausibility, across a variety of languages.

To achieve this, we will release a novel, large-scale dataset of syllogistic arguments. This dataset will help us measure not only a model's accuracy but also how the content effect manifests and varies across different languages.

We encourage participants to explore solutions based on natively multilingual open-source or open-weight models that offer insights into the internal reasoning mechanisms.

---

### Timeline

- Training data ready  (1 September 2025)
- Evaluation kit release, information on languages, and output formats  (31 October 2025)
- Evaluation data ready  & Codabench competition (1 December 2025)
- Practice Phase (10 December 2025 - 31 December 2025)
- Evaluation Phase (10 January 2026 - 31 January 2026)
- Paper submission (February 2026)
- Notification to authors (March 2026)
- Camera-ready (April 2026)
- SemEval workshop Summer 2026 (co-located with a major NLP conference)

---

### Task Overview

This task consists of four subtasks that build on each other, moving from a simple English setting to a complex multilingual one with distracting information. The competition will be hosted on **Codabench**.

#### Training Data

The training set is exclusively in English to simulate a low-resource setting. Arguments are categorized by both `validity` (true/false) and `plausibility` (true/false), though the goal is always to predict validity.

**Example from the training set:**

```json
{
    "id": "0",
    "syllogism": "Not all canines are aquatic creatures known as fish. It is certain that no fish belong to the class of mammals. Therefore, every canine falls under the category of mammals.",
    "validity": false,
    "plausibility": true
}
```

* **Note:** The model must correctly predict `validity: false`, ignoring the `plausibility: true` (which is based on world knowledge).
  
 ---
## Subtasks

In `evaluation_kit`, you can find the Python scrips that will be used to evaluate the systems. The scripts include mock examples to show participants the expected JSON output format. 

---

### Subtask 1: Syllogistic Reasoning in English (Binary Classification)

**Goal:** Determine the formal validity of syllogisms in English.

### Subtask 1 Metrics: Validity and Content Effect

| Metric | Definition | Purpose |
| :--- | :--- | :--- |
| **Overall Accuracy** ($\text{ACC}$) | Percentage of correct validity predictions across all items. | Measures basic logical competence. |
| **Total Content Effect** ($\text{TCE}$) | A composite score measuring the average accuracy difference due to plausibility across all four logical-plausibility conditions in English. **A lower TCE indicates higher logical integrity.** | Measures the model's overall susceptibility to content bias. |
| **Primary Ranking Metric** | $$\frac{\text{ACC}}{1 + \ln(1 + \text{TCE})}$$ | **The official ranking metric.** This metric rewards high accuracy and smoothly penalizes content bias, favoring models that are both correct and robust. |

---

### Subtask 2: Syllogistic Reasoning with Irrelevant Premises in English (Retrieval + Classification)

**Goal:** The task is to jointly predict the validity of the syllogism and the set of relevant premises that entail the conclusion.

**Note:**  In this task, the set of relevant premises is the set of statements that are necessary and sufficient to entail the conclusion. This means that **only "valid" syllogisms** will have relevant premises.

### Subtask 2 Metrics: Retrieval, Validity, and Bias

| Metric | Definition | Purpose |
| :--- | :--- | :--- |
| $\text{F1}$ premises | Macro-averaged F1-Score for correctly identifying the subset of relevant premises out of all available premises. | Measures the model's ability to filter relevant information. |
| **Combined Performance** ($\text{Avg}$) | $\text{Avg}(\text{ACC}, \text{F1})$ | Equates the weight given to the core reasoning task and the retrieval task. |
| **Primary Ranking Metric** | $$\frac{\text{Avg}}{1 + \ln(1 + \text{TCE})}$$ | **The official ranking metric.** It applies the content bias penalty to the average performance metric. |

---

### Subtask 3: Multilingual Syllogistic Reasoning (Multilingual Binary Classification)

**Goal:** Extend binary classification to multiple languages.

### Subtask 3 Metrics: Multilingual Validity and Content Effect

| Metric | Definition | Purpose |
| :--- | :--- | :--- |
| **Multilingual Accuracy** ($\text{Acc}$) | The average accuracy across all evaluated languages. | Measures average logical competence across languages. |
| **Multilingual Content Effect** ($\text{TCE}$) | A composite score that measures content effect within each target language and the difference in TCE between each target language and English (cross-lingual stability penalty). **A lower TCE indicates higher cross-lingual robustness.** | Measures the model's overall susceptibility to content bias and its stability across languages. |
| **Primary Ranking Metric** | $$\frac{\text{Acc}}{1 + \ln(1 + \text{TCE})}$$ | **The official ranking metric.** This metric rewards high average accuracy and smoothly penalizes multilingual content bias, favoring robust models. |

---

### Subtask 4: Multilingual Syllogistic Reasoning with Irrelevant Premises (Multilingual Retrieval + Classification)

**Goal:** Handle noisy, irrelevant premises in multiple languages. The task is to jointly predict the validity of the syllogism and the set of relevant premises that entail the conclusion.

**Note:**  In this task, the set of relevant premises is the set of statements that are necessary and sufficient to entail the conclusion. This means that **only "valid" syllogisms** will have relevant premises.

### Subtask 4 Metrics: Multilingual Retrieval, Validity, and Bias

| Metric | Definition | Purpose |
| :--- | :--- | :--- |
| $\text{F1}$ premises | Macro-averaged F1-Score for correctly identifying the subset of relevant premises out of all available premises, averaged across all languages. | Measures the model's multilingual ability to filter relevant information. |
| **Multilingual Combined Performance** ($\text{Avg}$) | $\text{Avg}(\text{Acc}, \text{F1})$ | Equates the weight given to the core reasoning task and the retrieval task, averaged multilngually. |
| **Multilingual Content Effect** ($\text{TCE}$) | Same as Subtask 3. | Measures the model's overall susceptibility to content bias and its stability across languages. |
| **Primary Ranking Metric** | $$\frac{\text{Avg}}{1 + \ln(1 + \text{TCE})}$$ | **The official ranking metric.** This metric rewards high average accuracy and smoothly penalizes multilingual content bias, favoring robust models. |

---

### Languages

The sutasks will include the following languages:

**Subtask 1 & 2**:

- English (en)

**Subtask 3 & 4**

- German (de)
- Spanish (es)
- French (fr)
- Italian (it)
- Dutch (nl)
- Portuguese (pt)
- Russian (ru)
- Chinese (zh)
- Swahili (sw)
- Bengali (bn)
- Telugu (te)

---

### Organizers

* Marco Valentino

* Leonardo Ranaldi

* Giulia Pucci

* Federico Ranaldi

* Andre Freitas

---

### References

If you found this repository useful, please consider citing our work: 

```
@inproceedings{valentino-etal-2026-semeval,
    title = "SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in Large Language Models",
    author = { Valentino, Marco  and
               Ranaldi, Leonardo  and
               Pucci, Giulia and
               Ranaldi, Federico and
               Freitas, Andr{\'e}},
    booktitle = "Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)",
    year = "2026"
}
```

### Other Relevant References:

Valentino, M., Kim, G., Dalal, D., Zhao, Z., & Freitas, A. (2025). Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering. arXiv preprint arXiv:2505.12189. 

Ranaldi, L., Valentino, M., and Freitas, A. (2025). Improving chain-of-thought reasoning via quasi-symbolic abstractions. ACL 2025.

Kim, G., Valentino, M. and Freitas, A. (2024). Reasoning circuits in language models: A mechanistic interpretation of syllogistic inference . Findings of ACL 2025.

Wysocka, M., Carvalho, D., Wysocki, O., Valentino, M., and Freitas, A. (2024). Syllobio-NLI: Evaluating large language models on biomedical syllogistic reasoning. NAACL 2025.

Seals, T. and Shalin, V. (2024). Evaluating the deductive competence of large language models. NAACL 2024.

Ozeki, K., Ando, R., Morishita, T., Abe, H., Mineshima, K., and Okada, M. (2024). Exploring reasoning biases in large language models through syllogism: Insights from the neubaroco dataset. Findings of ACL 2024.

Dasgupta, I., Lampinen, A. K., Chan, S. C. Y., Sheahan, H. R., Creswell, A., Kumaran, D., McClelland, J. L., and Hill, F. (2022). Language models show human-like content effects on reasoning tasks. arXiv preprint arXiv:2207.07051.

Bertolazzi, L., Gatt, A., and Bernardi, R. (2024). A systematic analysis of large language models as soft reasoners: The case of syllogistic inferences. EMNLP 2024.

Eisape, T., Tessler, M., Dasgupta, I., Sha, F., Steenkiste, S., and Linzen, T. (2024). A systematic comparison of syllogistic reasoning in humans and language models. NAACL 2024.

Quan, X., Valentino, M., Dennis, L., and Freitas, A. (2024). Verification and refinement of natural language explanations through llm-symbolic theorem proving. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing.

Lyu, Q., Havaldar, S., Stein, A., Zhang, L., Rao, D., Wong, E., Apidianaki, M., and Callison-Burch, C. (2023). Faithful chain-of-thought reasoning. AACL 2023.

Xu, J., Fei, H., Pan, L., Liu, Q., Lee, M., and Hsu, W. (2024). Faithful logical reasoning via symbolic chain-of-thought. arXiv preprint arXiv:2405.18357.
