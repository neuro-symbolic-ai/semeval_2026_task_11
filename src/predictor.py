"""
Prediction pipeline for syllogistic reasoning.
"""
from typing import List, Callable
from tqdm import tqdm

from .models import TestExample, ValidityPrediction, LLMResponse
from .llm_client import OpenRouterClient


class SyllogismPredictor:
    """
    Predictor for syllogism validity using LLMs.
    """

    def __init__(
        self,
        llm_client: OpenRouterClient,
        prompt_template: Callable[[str], str],
    ):
        """
        Initialize the predictor.

        Args:
            llm_client: OpenRouter client for API calls
            prompt_template: Function that formats the prompt
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template

    def predict_single(self, syllogism: str, example_id: str) -> ValidityPrediction:
        """
        Predict validity for a single syllogism.

        Args:
            syllogism: The syllogism text
            example_id: Unique identifier for the example

        Returns:
            ValidityPrediction with id and validity
        """
        # Format the prompt
        prompt = self.prompt_template(syllogism)

        # Get structured response from LLM
        try:
            response = self.llm_client.generate_structured(
                prompt=prompt,
                response_model=LLMResponse,
                temperature=0.0,  # Deterministic for consistency
            )

            return ValidityPrediction(
                id=example_id,
                validity=response.validity
            )

        except Exception as e:
            print(f"Error predicting for {example_id}: {e}")
            # Default to False on error (conservative approach)
            return ValidityPrediction(
                id=example_id,
                validity=False
            )

    def predict_batch(
        self,
        test_examples: List[TestExample],
        show_progress: bool = True
    ) -> List[ValidityPrediction]:
        """
        Predict validity for a batch of syllogisms.

        Args:
            test_examples: List of test examples
            show_progress: Whether to show progress bar

        Returns:
            List of ValidityPrediction objects
        """
        predictions = []

        # Use tqdm for progress bar if requested
        iterator = tqdm(test_examples, desc="Predicting") if show_progress else test_examples

        for example in iterator:
            prediction = self.predict_single(
                syllogism=example.syllogism,
                example_id=example.id
            )
            predictions.append(prediction)

        return predictions

    def predict_batch_as_dicts(
        self,
        test_examples: List[TestExample],
        show_progress: bool = True
    ) -> List[dict]:
        """
        Predict validity and return as list of dictionaries for JSON serialization.

        Args:
            test_examples: List of test examples
            show_progress: Whether to show progress bar

        Returns:
            List of prediction dictionaries with 'id' and 'validity' keys
        """
        predictions = self.predict_batch(test_examples, show_progress)
        return [pred.model_dump() for pred in predictions]
