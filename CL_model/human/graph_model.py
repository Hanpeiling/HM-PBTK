from __future__ import annotations

import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import RelGraphConv
from dgl.readout import sum_nodes
from torch import nn


class WeightAndSum(nn.Module):
    def __init__(self, in_feats, task_num: int = 1, attention: bool = True, return_weight: bool = False):
        super().__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight = return_weight
        self.atom_weighting_specific = nn.ModuleList(
            [self.atom_weight(self.in_feats) for _ in range(self.task_num)]
        )
        self.shared_weighting = self.atom_weight(self.in_feats)

    def forward(self, bg, feats):
        feat_list = []
        atom_list = []

        for i in range(self.task_num):
            with bg.local_scope():
                bg.ndata["h"] = feats
                weight = self.atom_weighting_specific[i](feats)
                bg.ndata["w"] = weight
                specific_feats_sum = sum_nodes(bg, "h", "w")
                atom_list.append(bg.ndata["w"])
            feat_list.append(specific_feats_sum)

        with bg.local_scope():
            bg.ndata["h"] = feats
            bg.ndata["w"] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, "h", "w")

        if self.attention:
            if self.return_weight:
                return feat_list, atom_list
            return feat_list
        return shared_feats_sum

    def atom_weight(self, in_feats):
        return nn.Sequential(nn.Linear(in_feats, 1), nn.Sigmoid())


class RGCNLayer(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_rels: int = 64 * 21,
        activation=F.relu,
        loop: bool = False,
        residual: bool = True,
        batchnorm: bool = True,
        rgcn_drop_out: float = 0.5,
    ):
        super().__init__()
        self.activation = activation
        self.graph_conv_layer = RelGraphConv(
            in_feats,
            out_feats,
            num_rels=num_rels,
            regularizer="basis",
            num_bases=None,
            bias=True,
            activation=activation,
            self_loop=loop,
            dropout=rgcn_drop_out,
        )
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats, etype, norm=None):
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        return new_feats


class BaseGNN(nn.Module):
    def __init__(
        self,
        gnn_out_feats,
        n_tasks,
        rgcn_drop_out: float = 0.5,
        return_mol_embedding: bool = False,
        return_weight: bool = False,
        classifier_hidden_feats: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.task_num = n_tasks
        self.gnn_layers = nn.ModuleList()
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(
            gnn_out_feats, self.task_num, return_weight=self.return_weight
        )
        self.fc_in_feats = gnn_out_feats
        self.return_mol_embedding = return_mol_embedding

        self.fc_layers1 = nn.ModuleList(
            [self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)]
        )
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)]
        )
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)]
        )
        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)]
        )

    def forward(self, bg, node_feats, etype, norm=None):
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats, etype, norm)

        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        for i in range(self.task_num):
            mol_feats = feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predict = self.output_layer1[i](h3)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)

        if self.return_mol_embedding:
            return feats_list[0]

        if self.return_weight:
            return prediction_all, atom_weight_list, node_feats
        return prediction_all

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
        )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(nn.Linear(hidden_feats, out_feats))


class MGA(BaseGNN):
    def __init__(
        self,
        in_feats,
        rgcn_hidden_feats,
        n_tasks,
        return_weight: bool = False,
        classifier_hidden_feats: int = 128,
        loop: bool = False,
        return_mol_embedding: bool = False,
        rgcn_drop_out: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__(
            gnn_out_feats=rgcn_hidden_feats[-1],
            n_tasks=n_tasks,
            classifier_hidden_feats=classifier_hidden_feats,
            return_mol_embedding=return_mol_embedding,
            return_weight=return_weight,
            rgcn_drop_out=rgcn_drop_out,
            dropout=dropout,
        )

        for out_feats in rgcn_hidden_feats:
            self.gnn_layers.append(RGCNLayer(in_feats, out_feats, loop=loop, rgcn_drop_out=rgcn_drop_out))
            in_feats = out_feats
