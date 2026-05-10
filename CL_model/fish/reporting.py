from __future__ import annotations


def print_section_title(title: str):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def print_search_result(model_name: str, step_name: str, best_params: dict, best_score: float):
    print(f"[{model_name}] Completed tuning step: {step_name}")
    print(f"[{model_name}] Best parameters: {best_params}")
    print(f"[{model_name}] Best score: {best_score:.6f}")


def print_classification_results(results: dict):
    print(f"Test Set Accuracy: {results['test_accuracy']:.6f}")
    print(f"Test Set Sensitivity: {results['test_sensitivity']}")
    print(f"Test Set Specificity: {results['test_specificity']}")
    print(f"Test Set F1 Score: {results['test_f1_macro']:.6f}")
    print(f"Cross-Validation Specificity: {results['cross_validation_specificity']:.6f}")
    print(f"Cross-Validation Accuracy: {results['cross_validation_accuracy']:.6f}")
    print(f"Cross-Validation Recall Macro: {results['cross_validation_recall_macro']:.6f}")
    print(f"Cross-Validation F1 Macro: {results['cross_validation_f1_macro']:.6f}")


def print_regression_results(results: dict):
    print(f"Test Set R2 Score: {results['test_r2']:.6f}")
    print(f"Test Set MSE: {results['test_mse']:.6f}")
    print(f"Train Set R2 Score: {results['train_r2']:.6f}")
    print(f"Train Set MSE: {results['train_mse']:.6f}")
    print(f"Cross-Validation R2 Score Mean: {results['cross_validation_r2']:.6f}")
    print(f"Average Cross-Validation MSE Score: {results['cross_validation_mse']:.6f}")
