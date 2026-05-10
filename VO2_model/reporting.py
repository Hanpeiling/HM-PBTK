"""Reporting and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List



def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path



def save_json(data: Dict[str, Any] | List[Dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)



def print_tuning_results(tuning_results: List[Dict[str, Any]]) -> None:
    print("Hyperparameter search results:")
    for item in tuning_results:
        print(f"- Parameter: {item['searched_parameter']}")
        print(f"  Best params: {item['best_params']}")
        print(f"  Best CV R2: {item['best_score']:.6f}")



def print_metrics(metrics: Dict[str, float]) -> None:
    print("Final model evaluation:")
    print(f"Test Set R2 Score: {metrics['test_r2']:.6f}")
    print(f"Test Set MSE: {metrics['test_mse']:.6f}")
    print(f"Train Set R2 Score: {metrics['train_r2']:.6f}")
    print(f"Train Set MSE: {metrics['train_mse']:.6f}")
    print(f"Cross-Validation R2 Score Mean: {metrics['cv_r2_mean']:.6f}")
    print(f"Average Cross-Validation MSE Score: {metrics['cv_mse_mean']:.6f}")
