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

#### Subtask 1: Syllogistic Reasoning in English

* **Goal:** Determine the formal validity of syllogisms in English.

* **Metrics:**

    * **Accuracy:** The percentage of correct validity predictions.
    * **Intra-Plausibility Content Effect:** The average difference in accuracy between valid and invalid arguments given a plausibility value (measures biases towards a specific validity label). 
    * **Cross-Plausibility Content Effect:** The average difference in accuracy between plausible and implausible arguments given a formal validity value (measures biases towards the plausibility value). 
    * **Total Content Effect:** The average between intra and cross-plausibility content effect. A lower content effect is preferable, as it indicates that the model is relying on logical structure rather than real-world content or biases.
 
Ranking will be based on the ratio of accuracy to total content effect. A higher ratio indicates a model that is both accurate and robust against content bias.

---

#### Subtask 2: Syllogistic Reasoning with Irrelevant Premises in English

* **Goal:** Determine validity while filtering out "noisy" or irrelevant premises in English.

* **Metrics:**

  * **Binary Prediction:** Same accuracy and content effect metrics as Subtask 1.

  * **Premise Selection:** An F1 score to measure the model's ability to identify relevant premises.
    
* **Ranking:** Based on the ratio of Accuracy and F1 to Content Effect. A higher ratio indicates a more robust model.
---

#### Subtask 3: Multilingual Syllogistic Reasoning

* **Goal:** Extend binary classification to multiple languages.

* **Metrics:**

  * **Accuracy:** Percentage of correct validity predictions in each language.

  * **Multilingual Content Effect:** A composite score that measures both content effect within a target language and the difference in content effect between that language and English.

* **Ranking:** Based on the ratio of Accuracy to Multilingual Content Effect.

---

#### Subtask 4: Multilingual Syllogistic Reasoning with Irrelevant Premises

* **Goal:** Handle noisy, irrelevant premises in multiple languages.

* **Metrics:**

  * **Binary Prediction:** Same accuracy and content effect metrics as Subtask 3.

  * **Premise Selection:** F1 score for identifying relevant premises.

* **Ranking:** Based on the ratio of Accuracy and F1 to Multilingual Content Effect.

---

### Organizers

* Marco Valentino

* Leonardo Ranaldi

* Giulia Pucci

* Federico Ranaldi

* Andre Freitas

---

### References

* \[1\] Seals, T. and Shalin, V. (2024). Evaluating the deductive competence of large language models. NAACL 2024.

* \[2\] Kim, G., Valentino, M. and Freitas, A. (2024). Reasoning circuits in language models: A mechanistic interpretation of syllogistic inference. Findings of ACL 2025.

* \[3\] Wysocka, M., Carvalho, D., Wysocki, O., Valentino, M., and Freitas, A. (2024). Syllobio-NLI: Evaluating large language models on biomedical syllogistic reasoning. NAACL 2025.

* \[4\] Ozeki, K., Ando, R., Morishita, T., Abe, H., Mineshima, K., and Okada, M. (2024). Exploring reasoning biases in large language models through syllogism: Insights from the neubaroco dataset. Findings of ACL 2024.

* \[5\] Dasgupta, I., Lampinen, A. K., Chan, S. C. Y., Sheahan, H. R., Creswell, A., Kumaran, D., McClelland, J. L., and Hill, F. (2022). Language models show human-like content effects on reasoning tasks. arXiv preprint arXiv:2207.07051.

* \[6\] Bertolazzi, L., Gatt, A., and Bernardi, R. (2024). A systematic analysis of large language models as soft reasoners: The case of syllogistic inferences. EMNLP 2024.

* \[7\] Eisape, T., Tessler, M., Dasgupta, I., Sha, F., Steenkiste, S., and Linzen, T. (2024). A systematic comparison of syllogistic reasoning in humans and language models. NAACL 2024.

* \[8\] Quan, X., Valentino, M., Dennis, L., and Freitas, A. (2024). Verification and refinement of natural language explanations through llm-symbolic theorem proving. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing.

* \[9\] Lyu, Q., Havaldar, S., Stein, A., Zhang, L., Rao, D., Wong, E., Apidianaki, M., and Callison-Burch, C. (2023). Faithful chain-of-thought reasoning. AACL 2023.

* \[10\] Xu, J., Fei, H., Pan, L., Liu, Q., Lee, M., and Hsu, W. (2024). Faithful logical reasoning via symbolic chain-of-thought. arXiv preprint arXiv:2405.18357.

* \[11\] Ranaldi, L., Valentino, M., Polonsky, A., and Freitas, A. (2025). Improving chain-of-thought reasoning via quasi-symbolic abstractions. ACL 2025.

* \[12\] Valentino, M., Kim, G., Dalal, D., Zhao, Z., & Freitas, A. (2025). Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering. arXiv preprint arXiv:2505.12189.
