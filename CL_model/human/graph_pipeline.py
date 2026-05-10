from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from common import ensure_dir
from graph_dataset import build_data_and_save_for_split, load_graph_from_csv_bin_for_split
from graph_model import MGA
from graph_training import (
    EarlyStopping,
    collate_molgraphs,
    pos_weight,
    run_a_train_epoch_heterogeneous,
    run_an_eval_epoch_heterogeneous,
    set_random_seed,
)


def _build_graph_args(base_dir: Path) -> dict[str, Any]:
    args: dict[str, Any] = {}
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    args["atom_data_field"] = "atom"
    args["bond_data_field"] = "etype"
    args["classification_metric_name"] = "roc_auc"
    args["regression_metric_name"] = "r2"

    args["num_epochs"] = 1000
    args["patience"] = 50
    args["batch_size"] = 256
    args["mode"] = "higher"
    args["in_feats"] = 40
    args["rgcn_hidden_feats"] = [128, 128]
    args["classifier_hidden_feats"] = 64
    args["rgcn_drop_out"] = 0.2
    args["drop_out"] = 0.1
    args["lr"] = 3
    args["weight_decay"] = 5
    args["loop"] = True

    args["task_name"] = "graph"
    args["data_name"] = "graph"
    args["times"] = 10

    args["select_task_list"] = ["graph"]
    args["select_task_index"] = []
    args["classification_num"] = 0
    args["regression_num"] = 0
    args["all_task_list"] = ["graph"]

    for index, task in enumerate(args["all_task_list"]):
        if task in args["select_task_list"]:
            args["select_task_index"].append(index)

    for task in args["select_task_list"]:
        if task in ["class01"]:
            args["classification_num"] += 1
        if task in ["graph"]:
            args["regression_num"] += 1

    if args["classification_num"] != 0 and args["regression_num"] != 0:
        args["task_class"] = "classification_regression"
    elif args["classification_num"] != 0 and args["regression_num"] == 0:
        args["task_class"] = "classification"
    else:
        args["task_class"] = "regression"

    data_output_dir = ensure_dir(base_dir / "data")
    model_output_dir = ensure_dir(base_dir / "model")

    args["input_csv"] = str(base_dir / "graph.csv")
    args["bin_path"] = str(data_output_dir / f"{args['data_name']}.bin")
    args["group_path"] = str(data_output_dir / f"{args['data_name']}_group.csv")
    args["early_stop_path"] = str(model_output_dir / f"{args['task_name']}_early_stop.pth")
    args["summary_path"] = str(base_dir / "graph_result.csv")
    args["all_runs_path"] = str(base_dir / "graph_result_all_runs.csv")
    return args


def run_graph_pipeline(base_dir: str | Path) -> dict[str, Any]:
    base_dir = Path(base_dir).resolve()
    args = _build_graph_args(base_dir)

    print(
        f"Classification tasks: {args['classification_num']}, "
        f"Regression tasks: {args['regression_num']}"
    )

    print("Building graph dataset files from graph.csv ...")
    build_data_and_save_for_split(
        origin_path=args["input_csv"],
        save_path=args["bin_path"],
        group_path=args["group_path"],
        task_list_selected=None,
    )

    result_columns = (
        args["select_task_list"]
        + ["group"]
        + args["select_task_list"]
        + ["group"]
        + args["select_task_list"]
        + ["group"]
    )
    result_pd = pd.DataFrame(columns=result_columns)

    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []

    for time_id in range(args["times"]):
        set_random_seed(2020 + time_id)
        print("*" * 100)
        print(f"{args['task_name']}, run {time_id + 1}/{args['times']}")
        print("*" * 100)

        train_set, val_set, test_set, task_number = load_graph_from_csv_bin_for_split(
            bin_path=args["bin_path"],
            group_path=args["group_path"],
            select_task_index=args["select_task_index"],
        )
        print("Molecule graph generation is complete.")

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=args["batch_size"],
            shuffle=True,
            collate_fn=collate_molgraphs,
        )
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=args["batch_size"],
            shuffle=True,
            collate_fn=collate_molgraphs,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args["batch_size"],
            collate_fn=collate_molgraphs,
        )

        pos_weight_np = pos_weight(train_set, classification_num=args["classification_num"])
        loss_criterion_c = torch.nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=pos_weight_np.to(args["device"])
        )
        loss_criterion_r = torch.nn.MSELoss(reduction="none")

        model = MGA(
            in_feats=args["in_feats"],
            rgcn_hidden_feats=args["rgcn_hidden_feats"],
            n_tasks=task_number,
            rgcn_drop_out=args["rgcn_drop_out"],
            classifier_hidden_feats=args["classifier_hidden_feats"],
            dropout=args["drop_out"],
            loop=args["loop"],
        )
        optimizer = Adam(
            model.parameters(),
            lr=10 ** (-args["lr"]),
            weight_decay=10 ** (-args["weight_decay"]),
        )
        stopper = EarlyStopping(
            patience=args["patience"],
            task_name=args["task_name"],
            mode=args["mode"],
            filename=args["early_stop_path"],
        )
        model.to(args["device"])

        for epoch in range(args["num_epochs"]):
            run_a_train_epoch_heterogeneous(
                args,
                epoch,
                model,
                train_loader,
                loss_criterion_c,
                loss_criterion_r,
                optimizer,
            )

            validation_result = run_an_eval_epoch_heterogeneous(args, model, val_loader)
            train_score = run_an_eval_epoch_heterogeneous(args, model, train_loader)
            test_score = run_an_eval_epoch_heterogeneous(args, model, test_loader)
            val_score = float(np.mean(validation_result))
            early_stop = stopper.step(val_score, model)

            print(
                f"Epoch {epoch + 1}/{args['num_epochs']}, validation {val_score:.4f}, "
                f"best validation {stopper.best_score:.4f}",
                "validation result:",
                validation_result,
            )

            if early_stop:
                break

        stopper.load_checkpoint(model)
        test_score = run_an_eval_epoch_heterogeneous(args, model, test_loader)
        train_score = run_an_eval_epoch_heterogeneous(args, model, train_loader)
        val_score = run_an_eval_epoch_heterogeneous(args, model, val_loader)

        result = train_score + ["training"] + val_score + ["valid"] + test_score + ["test"]
        result_pd.loc[time_id] = result

        all_times_train_result.append(train_score)
        all_times_val_result.append(val_score)
        all_times_test_result.append(test_score)

        print(f"Completed results for run {time_id + 1}")
        print("Training result:", train_score)
        print("Validation result:", val_score)
        print("Test result:", test_score)

    result_pd.to_csv(args["all_runs_path"], index=False)

    mean_train = np.mean(np.array(all_times_train_result), axis=0).tolist()
    mean_val = np.mean(np.array(all_times_val_result), axis=0).tolist()
    mean_test = np.mean(np.array(all_times_test_result), axis=0).tolist()

    summary_df = pd.DataFrame(
        [mean_train + ["training"] + mean_val + ["valid"] + mean_test + ["test"]],
        columns=result_columns,
    )
    summary_df.to_csv(args["summary_path"], index=False)

    print("Saved graph summary to:", args["summary_path"])
    print(summary_df)

    return {
        "all_runs_path": args["all_runs_path"],
        "summary_path": args["summary_path"],
        "mean_train": [float(x) for x in mean_train],
        "mean_val": [float(x) for x in mean_val],
        "mean_test": [float(x) for x in mean_test],
    }
