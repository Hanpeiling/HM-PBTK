from __future__ import annotations

import random

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, precision_recall_curve, r2_score, roc_auc_score, auc
from torch.utils.data import DataLoader


def set_random_seed(seed: int = 10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pos_weight(train_set, classification_num):
    smiles, graphs, labels, mask = map(list, zip(*train_set))
    labels = np.array(labels)
    task_pos_weight_list = []
    for task in range(classification_num):
        num_pos = 0
        num_neg = 0
        for value in labels[:, task]:
            if value == 1:
                num_pos += 1
            if value == 0:
                num_neg += 1
        task_pos_weight_list.append(num_neg / (num_pos + 1e-8))
    return torch.tensor(task_pos_weight_list)


class Meter:
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.sigmoid(torch.cat(self.y_pred, dim=0))
        y_true = torch.cat(self.y_true, dim=0)

        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))
        return scores

    def r2(self):
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores

    def compute_metric(self, metric_name: str):
        if metric_name == "roc_auc":
            return self.roc_auc_score()
        if metric_name == "r2":
            return self.r2()
        raise ValueError(f"Unsupported metric: {metric_name}")


def collate_molgraphs(data):
    smiles, graphs, labels, mask = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.tensor(labels)
    mask = torch.tensor(mask)
    return smiles, bg, labels, mask


def run_a_train_epoch_heterogeneous(
    args,
    epoch,
    model,
    data_loader,
    loss_criterion_c,
    loss_criterion_r,
    optimizer,
    task_weight=None,
):
    model.train()
    train_meter_c = Meter()
    train_meter_r = Meter()

    if task_weight is not None:
        task_weight = task_weight.float().to(args["device"])

    for batch_data in data_loader:
        smiles, bg, labels, mask = batch_data
        mask = mask.float().to(args["device"])
        atom_feats = bg.ndata.pop(args["atom_data_field"]).float().to(args["device"])
        bond_feats = bg.edata.pop(args["bond_data_field"]).long().to(args["device"])
        bg = bg.to(args["device"])
        logits = model(bg, atom_feats, bond_feats, norm=None)
        labels = labels.type_as(logits).to(args["device"])

        if args["task_class"] == "classification_regression":
            logits_c = logits[:, : args["classification_num"]]
            labels_c = labels[:, : args["classification_num"]]
            mask_c = mask[:, : args["classification_num"]]

            logits_r = logits[:, args["classification_num"] :]
            labels_r = labels[:, args["classification_num"] :]
            mask_r = mask[:, args["classification_num"] :]

            if task_weight is None:
                loss = (loss_criterion_c(logits_c, labels_c) * (mask_c != 0).float()).mean() + (
                    loss_criterion_r(logits_r, labels_r) * (mask_r != 0).float()
                ).mean()
            else:
                task_weight_c = task_weight[: args["classification_num"]]
                task_weight_r = task_weight[args["classification_num"] :]
                loss = (
                    torch.mean(loss_criterion_c(logits_c, labels_c) * (mask_c != 0).float(), dim=0) * task_weight_c
                ).mean() + (
                    torch.mean(loss_criterion_r(logits_r, labels_r) * (mask_r != 0).float(), dim=0) * task_weight_r
                ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter_c.update(logits_c, labels_c, mask_c)
            train_meter_r.update(logits_r, labels_r, mask_r)

        elif args["task_class"] == "classification":
            if task_weight is None:
                loss = (loss_criterion_c(logits, labels) * (mask != 0).float()).mean()
            else:
                loss = (
                    torch.mean(loss_criterion_c(logits, labels) * (mask != 0).float(), dim=0) * task_weight
                ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter_c.update(logits, labels, mask)

        else:
            if task_weight is None:
                loss = (loss_criterion_r(logits, labels) * (mask != 0).float()).mean()
            else:
                loss = (
                    torch.mean(loss_criterion_r(logits, labels) * (mask != 0).float(), dim=0) * task_weight
                ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter_r.update(logits, labels, mask)

    if args["task_class"] == "classification_regression":
        train_score = np.mean(
            train_meter_c.compute_metric(args["classification_metric_name"])
            + train_meter_r.compute_metric(args["regression_metric_name"])
        )
        print(f"Epoch {epoch + 1}/{args['num_epochs']}, training r2+auc {train_score:.4f}")
    elif args["task_class"] == "classification":
        train_score = np.mean(train_meter_c.compute_metric(args["classification_metric_name"]))
        print(f"Epoch {epoch + 1}/{args['num_epochs']}, training {args['classification_metric_name']} {train_score:.4f}")
    else:
        train_score = np.mean(train_meter_r.compute_metric(args["regression_metric_name"]))
        print(f"Epoch {epoch + 1}/{args['num_epochs']}, training {args['regression_metric_name']} {train_score:.4f}")


def run_an_eval_epoch_heterogeneous(args, model, data_loader):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()

    with torch.no_grad():
        for batch_data in data_loader:
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(args["device"])
            mask = mask.float().to(args["device"])
            atom_feats = bg.ndata.pop(args["atom_data_field"]).float().to(args["device"])
            bond_feats = bg.edata.pop(args["bond_data_field"]).long().to(args["device"])
            bg = bg.to(args["device"])
            logits = model(bg, atom_feats, bond_feats, norm=None)
            labels = labels.type_as(logits).to(args["device"])

            if args["task_class"] == "classification_regression":
                logits_c = logits[:, : args["classification_num"]]
                labels_c = labels[:, : args["classification_num"]]
                mask_c = mask[:, : args["classification_num"]]
                logits_r = logits[:, args["classification_num"] :]
                labels_r = labels[:, args["classification_num"] :]
                mask_r = mask[:, args["classification_num"] :]
                eval_meter_c.update(logits_c, labels_c, mask_c)
                eval_meter_r.update(logits_r, labels_r, mask_r)
            elif args["task_class"] == "classification":
                eval_meter_c.update(logits, labels, mask)
            else:
                eval_meter_r.update(logits, labels, mask)

    if args["task_class"] == "classification_regression":
        return eval_meter_c.compute_metric(args["classification_metric_name"]) + eval_meter_r.compute_metric(
            args["regression_metric_name"]
        )
    if args["task_class"] == "classification":
        return eval_meter_c.compute_metric(args["classification_metric_name"])
    return eval_meter_r.compute_metric(args["regression_metric_name"])


class EarlyStopping:
    def __init__(self, mode: str = "higher", patience: int = 10, filename: str | None = None, task_name: str = "None"):
        if filename is None:
            filename = f"model/{task_name}_early_stop.pth"

        assert mode in ["higher", "lower"]
        self.mode = mode
        self._check = self._check_higher if self.mode == "higher" else self._check_lower
        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save({"model_state_dict": model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename, map_location=torch.device("cpu"))["model_state_dict"])
