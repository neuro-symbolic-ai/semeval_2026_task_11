"""
LLM-based normalizer for syllogistic arguments.

Takes a full syllogism string as input and rewrites the three sentences
into a canonical categorical form using a prompt template.
"""

from typing import List, Dict, Callable
from tqdm import tqdm


class SyllogismNormalizer:
    """
    Normalizes syllogistic arguments using an LLM.

    Mirrors the structure of SyllogismPredictor, but instead of predicting
    validity, it rewrites the syllogism text while keeping the original
    JSON schema (id + syllogism).
    """

    def __init__(
        self,
        llm_client,
        prompt_template: Callable[[str], str],
    ):
        """
        Args:
            llm_client: Initialized LLM client (e.g. OpenRouterClient)
            prompt_template: Prompt function taking a syllogism string
                             and returning a prompt
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template

    def normalize_one(self, example: Dict) -> Dict:
        """
        Normalize a single syllogism example.

        Args:
            example: Dictionary containing at least "id" and "syllogism"

        Returns:
            Dictionary with the same schema as input:
            {
                "id": ...,
                "syllogism": "normalized text"
            }
        """
        original_syllogism = example["syllogism"]

        prompt = self.prompt_template(original_syllogism)

        # LLM returns plain text (normalized syllogism)
        normalized_syllogism = self.llm_client.generate_text(prompt)

        return {
            "id": example.get("id"),
            "syllogism": normalized_syllogism.strip(),
        }

    def normalize_batch_as_dicts(
        self,
        test_examples: List,
        show_progress: bool = False,
    ) -> List[Dict]:
        """
        Normalize a batch of syllogism examples.

        Args:
            test_examples: List of input examples (TestExample objects or dicts)
            show_progress: Whether to show a progress bar

        Returns:
            List of normalized syllogism dictionaries
        """
        results = []
        iterator = tqdm(test_examples) if show_progress else test_examples

        for example in iterator:
            # Convert TestExample to dict if necessary
            if hasattr(example, 'model_dump'):  # Pydantic v2
                example_dict = example.model_dump()
            elif hasattr(example, 'dict'):  # Pydantic v1
                example_dict = example.dict()
            else:
                example_dict = example  # Already a dict
            
            results.append(self.normalize_one(example_dict))

        return results
