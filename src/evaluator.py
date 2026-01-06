"""
Evaluation wrapper for SemEval 2026 Task 11.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add evaluation_kit to path to import the evaluation script
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation_kit" / "task 1 & 3"))

try:
    from evaluation_script import (
        calculate_accuracy,
        calculate_subgroup_accuracy,
        calculate_content_effect_bias,
        calculate_smooth_combined_metric,
        run_full_scoring
    )
except ImportError as e:
    print(f"Warning: Could not import evaluation script: {e}")
    print("Evaluation functions may not be available.")


def evaluate_predictions(
    ground_truth_path: str,
    predictions_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth.

    Args:
        ground_truth_path: Path to ground truth JSON file
        predictions_path: Path to predictions JSON file
        output_path: Optional path to save evaluation results as JSON
        verbose: Whether to print detailed results

    Returns:
        Dictionary with evaluation metrics:
        - accuracy: Overall accuracy percentage
        - content_effect: Total content effect score
        - combined_score: Combined metric for ranking
    """
    # Load data
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # Check coverage
    gt_ids = set(example["id"] for example in ground_truth)
    pred_ids = set(example["id"] for example in predictions)
    missing = gt_ids - pred_ids

    if missing:
        print(f"Warning: {len(missing)} examples missing from predictions")
        if verbose:
            print(f"Missing IDs: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    # Create ID map
    gt_map = {item['id']: item for item in ground_truth}

    # Calculate overall accuracy
    overall_acc, correct, total = calculate_accuracy(
        ground_truth_list=ground_truth,
        predictions_list=predictions,
        metric_name='validity',
        prediction_key='validity',
        plausibility_filter=None
    )

    # Calculate subgroup accuracies
    acc_plausible_valid, _, _ = calculate_subgroup_accuracy(
        gt_map, predictions, gt_validity=True, gt_plausibility=True
    )
    acc_implausible_valid, _, _ = calculate_subgroup_accuracy(
        gt_map, predictions, gt_validity=True, gt_plausibility=False
    )
    acc_plausible_invalid, _, _ = calculate_subgroup_accuracy(
        gt_map, predictions, gt_validity=False, gt_plausibility=True
    )
    acc_implausible_invalid, _, _ = calculate_subgroup_accuracy(
        gt_map, predictions, gt_validity=False, gt_plausibility=False
    )

    conditional_accuracies = {
        'acc_plausible_valid': acc_plausible_valid,
        'acc_implausible_valid': acc_implausible_valid,
        'acc_plausible_invalid': acc_plausible_invalid,
        'acc_implausible_invalid': acc_implausible_invalid
    }

    # Calculate content effect
    bias_metrics = calculate_content_effect_bias(conditional_accuracies)
    tot_content_effect = bias_metrics['tot_content_effect']

    # Calculate combined score
    combined_score = calculate_smooth_combined_metric(overall_acc, tot_content_effect)

    # Prepare results
    results = {
        'accuracy': round(overall_acc, 4),
        'content_effect': round(tot_content_effect, 4),
        'combined_score': round(combined_score, 4),
        'correct_predictions': correct,
        'total_predictions': total,
        'subgroup_accuracies': {
            'plausible_valid': round(acc_plausible_valid, 2),
            'implausible_valid': round(acc_implausible_valid, 2),
            'plausible_invalid': round(acc_plausible_invalid, 2),
            'implausible_invalid': round(acc_implausible_invalid, 2),
        },
        'content_effect_breakdown': {
            'intra_validity': round(bias_metrics['content_effect_intra_validity_label'], 4),
            'inter_validity': round(bias_metrics['content_effect_inter_validity_label'], 4),
        }
    }

    # Print results if verbose
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {results['accuracy']:.2f}% ({correct}/{total})")
        print(f"Content Effect: {results['content_effect']:.2f}")
        print(f"Combined Score: {results['combined_score']:.2f}")
        print("\nSubgroup Accuracies:")
        print(f"  Plausible & Valid:     {results['subgroup_accuracies']['plausible_valid']:.2f}%")
        print(f"  Implausible & Valid:   {results['subgroup_accuracies']['implausible_valid']:.2f}%")
        print(f"  Plausible & Invalid:   {results['subgroup_accuracies']['plausible_invalid']:.2f}%")
        print(f"  Implausible & Invalid: {results['subgroup_accuracies']['implausible_invalid']:.2f}%")
        print("\nContent Effect Breakdown:")
        print(f"  Intra-validity: {results['content_effect_breakdown']['intra_validity']:.2f}")
        print(f"  Inter-validity: {results['content_effect_breakdown']['inter_validity']:.2f}")
        print("="*60 + "\n")

    # Save to file if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Results saved to: {output_path}")

    return results
