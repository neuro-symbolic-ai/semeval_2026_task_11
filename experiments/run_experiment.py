"""
Main experiment runner for SemEval 2026 Task 11 - Subtask 1.

Usage:
    python experiments/run_experiment.py \\
        --model "anthropic/claude-3.5-sonnet" \\
        --prompt "direct" \\
        --input "train_data/subtask 1/train_data.json" \\
        --output "predictions/test_predictions.json" \\
        --evaluate \\
        --reference "train_data/subtask 1/train_data.json"

    Results are automatically saved to CSV with filename:
        experiments/results_{model}_{prompt}.csv

    Or specify custom path:
        --results-csv "experiments/my_results.csv"
"""
import argparse
import sys
import csv
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import OpenRouterClient
from src.data_loader import load_test_data, load_training_data, save_predictions
from src.prompts import get_prompt_template, PROMPT_TEMPLATES
from src.predictor import SyllogismPredictor
from src.evaluator import evaluate_predictions


def save_results_to_csv(args, results, csv_path):
    """
    Save experiment results to CSV file.

    Args:
        args: Parsed command-line arguments
        results: Dictionary of evaluation results
        csv_path: Path to save the CSV file
    """
    # Prepare row data
    row = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'prompt_template': args.prompt,
        'input_file': args.input,
        'output_file': args.output,
        'reference_file': args.reference,
        'num_examples': args.limit if args.limit else results['total_predictions'],

        # Main metrics
        'accuracy': results['accuracy'],
        'content_effect': results['content_effect'],
        'combined_score': results['combined_score'],
        'correct_predictions': results['correct_predictions'],
        'total_predictions': results['total_predictions'],

        # Subgroup accuracies
        'acc_plausible_valid': results['subgroup_accuracies']['plausible_valid'],
        'acc_implausible_valid': results['subgroup_accuracies']['implausible_valid'],
        'acc_plausible_invalid': results['subgroup_accuracies']['plausible_invalid'],
        'acc_implausible_invalid': results['subgroup_accuracies']['implausible_invalid'],

        # Content effect breakdown
        'content_effect_intra_validity': results['content_effect_breakdown']['intra_validity'],
        'content_effect_inter_validity': results['content_effect_breakdown']['inter_validity'],
    }

    # Check if file exists to determine if we need to write headers
    file_exists = Path(csv_path).exists()

    # Write to CSV
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run syllogism validity prediction experiments"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3.5-sonnet",
        help="OpenRouter model identifier (default: anthropic/claude-3.5-sonnet)"
    )

    # Prompt template
    parser.add_argument(
        "--prompt",
        type=str,
        default="direct",
        choices=list(PROMPT_TEMPLATES.keys()),
        help=f"Prompt template to use. Options: {', '.join(PROMPT_TEMPLATES.keys())}"
    )

    # Input/Output paths
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file (test data)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save predictions JSON file"
    )

    # Evaluation options
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate predictions against reference data"
    )

    parser.add_argument(
        "--reference",
        type=str,
        help="Path to reference/ground truth JSON file (required if --evaluate is set)"
    )

    # Optional: limit number of examples (for testing)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)"
    )

    # API key (optional, can also use .env)
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenRouter API key (optional, can use .env file)"
    )

    # Results CSV (optional)
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Path to save experiment results CSV (default: auto-generated from model and prompt)"
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Validate evaluation arguments
    if args.evaluate and not args.reference:
        parser.error("--reference is required when --evaluate is set")

    print("="*60)
    print("SemEval 2026 Task 11 - Syllogism Validity Prediction")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Prompt Template: {args.prompt}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    if args.limit:
        print(f"Limit: {args.limit} examples")
    print("="*60 + "\n")

    # Initialize LLM client
    print("Initializing OpenRouter client...")
    try:
        llm_client = OpenRouterClient(
            api_key=args.api_key,
            model=args.model
        )
        print("Client initialized successfully.\n")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENROUTER_API_KEY in .env file or use --api-key argument")
        sys.exit(1)

    # Load prompt template
    print(f"Loading prompt template: {args.prompt}...")
    prompt_template = get_prompt_template(args.prompt)
    print("Prompt template loaded.\n")

    # Load test data
    print(f"Loading test data from: {args.input}...")
    try:
        test_examples = load_test_data(args.input)
        if args.limit:
            test_examples = test_examples[:args.limit]
        print(f"Loaded {len(test_examples)} test examples.\n")
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)

    # Initialize predictor
    predictor = SyllogismPredictor(
        llm_client=llm_client,
        prompt_template=prompt_template
    )

    # Run predictions
    print("Running predictions...")
    print("This may take a while depending on the number of examples...\n")

    try:
        predictions = predictor.predict_batch_as_dicts(
            test_examples=test_examples,
            show_progress=True
        )
    except Exception as e:
        print(f"\nError during prediction: {e}")
        sys.exit(1)

    # Save predictions
    print(f"\nSaving predictions to: {args.output}...")
    try:
        # Create output directory if it doesn't exist
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_predictions(predictions, args.output)
        print(f"Predictions saved successfully.\n")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        sys.exit(1)

    # Evaluate if requested
    if args.evaluate:
        print("Evaluating predictions...")
        try:
            results = evaluate_predictions(
                ground_truth_path=args.reference,
                predictions_path=args.output,
                output_path=None,  # Don't save evaluation results to separate file
                verbose=True
            )

            # Save results to CSV
            if args.results_csv:
                csv_path = args.results_csv
            else:
                # Auto-generate CSV filename from model and prompt
                model_safe = args.model.replace('/', '_').replace('\\', '_')
                csv_path = f"experiments/results_{model_safe}_{args.prompt}.csv"

            # Create experiments directory if it doesn't exist
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

            save_results_to_csv(args, results, csv_path)
            print(f"Results appended to: {csv_path}")

        except Exception as e:
            print(f"Error during evaluation: {e}")
            sys.exit(1)

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
