from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dgl import DGLGraph
from dgl.data.graph_serialize import load_graphs, save_graphs
from rdkit import Chem
from rdkit.Chem import MolFromSmiles


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} is not in the allowable set {allowable_set}.")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H: bool = False, use_chirality: bool = True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            "B",
            "C",
            "N",
            "O",
            "F",
            "Si",
            "P",
            "S",
            "Cl",
            "As",
            "Se",
            "Br",
            "Te",
            "I",
            "At",
            "other",
        ],
    ) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + [
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
    ] + one_of_k_encoding_unk(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            "other",
        ],
    ) + [atom.GetIsAromatic()]

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"]) + [
                atom.HasProp("_ChiralityPossible")
            ]
        except Exception:
            results = results + [False, False] + [atom.HasProp("_ChiralityPossible")]

    return np.array(results)


def one_of_k_atompair_encoding(x, allowable_set):
    for atompair in allowable_set:
        if x in atompair:
            x = atompair
            break
        if atompair == allowable_set[-1]:
            x = allowable_set[-1]
    return [x == s for s in allowable_set]


def etype_features(bond, use_chirality: bool = True, atompair: bool = True):
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
    ]
    for i, matched in enumerate(bond_feats_1):
        if matched is True:
            a = i

    bond_feats_2 = bond.GetIsConjugated()
    b = 1 if bond_feats_2 is True else 0

    # Important: this intentionally preserves the notebook behavior.
    # The original notebook used `bond.IsInRing` instead of `bond.IsInRing()`.
    bond_feats_3 = bond.IsInRing
    c = 1 if bond_feats_3 is True else 0

    index = a * 1 + b * 4 + c * 8

    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
        )
        for i, matched in enumerate(bond_feats_4):
            if matched is True:
                d = i
        index = index + d * 16

    if atompair is True:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        bond_feats_5 = one_of_k_atompair_encoding(
            atom_pair_str,
            [
                ["CC"],
                ["CN", "NC"],
                ["ON", "NO"],
                ["CO", "OC"],
                ["CS", "SC"],
                ["SO", "OS"],
                ["NN"],
                ["SN", "NS"],
                ["CCl", "ClC"],
                ["CF", "FC"],
                ["CBr", "BrC"],
                ["others"],
            ],
        )
        for i, matched in enumerate(bond_feats_5):
            if matched is True:
                e = i
        index = index + e * 64

    return index


def construct_rgcn_bigraph_from_smiles(smiles: str):
    graph = DGLGraph()
    mol = MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    graph.add_nodes(num_atoms)

    atom_features_all = []
    for atom in mol.GetAtoms():
        atom_features_all.append(atom_features(atom).tolist())
    graph.ndata["atom"] = torch.tensor(atom_features_all)

    src_list = []
    dst_list = []
    etype_feature_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        etype_feature = etype_features(bond)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_feature_all.append(etype_feature)
        etype_feature_all.append(etype_feature)

    graph.add_edges(src_list, dst_list)
    normal_all = []
    for value in etype_feature_all:
        normal = etype_feature_all.count(value) / len(etype_feature_all)
        normal = round(normal, 1)
        normal_all.append(normal)

    graph.edata["etype"] = torch.tensor(etype_feature_all)
    graph.edata["normal"] = torch.tensor(normal_all)
    return graph


def build_mask(labels_list, mask_value: int = 100):
    mask = []
    for value in labels_list:
        mask.append(0 if value == mask_value else 1)
    return mask


def multi_task_build_dataset(dataset_smiles: pd.DataFrame, labels_list, smiles_name: str):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_list]
    split_index = dataset_smiles["group"]
    smiles_list = dataset_smiles[smiles_name]
    molecule_number = len(smiles_list)

    for i, smiles in enumerate(smiles_list):
        try:
            graph = construct_rgcn_bigraph_from_smiles(smiles)
            mask = build_mask(labels.loc[i], mask_value=123456)
            molecule = [smiles, graph, labels.loc[i], mask, split_index.loc[i]]
            dataset_gnn.append(molecule)
            print(f"{i + 1}/{molecule_number} molecules transformed.")
        except Exception:
            print(f"Failed to transform molecule: {smiles}")
            molecule_number = molecule_number - 1
            failed_molecule.append(smiles)

    print(f"Failed molecules: {failed_molecule} (count={len(failed_molecule)})")
    return dataset_gnn


def build_data_and_save_for_split(
    origin_path: str,
    save_path: str,
    group_path: str,
    task_list_selected=None,
):
    data_origin = pd.read_csv(origin_path)
    smiles_name = "smiles"
    data_origin = data_origin.fillna(123456)
    labels_list = [x for x in data_origin.columns if x not in ["smiles", "group"]]
    if task_list_selected is not None:
        labels_list = task_list_selected

    dataset_gnn = multi_task_build_dataset(
        dataset_smiles=data_origin,
        labels_list=labels_list,
        smiles_name=smiles_name,
    )

    smiles, graphs, labels, mask, split_index = map(list, zip(*dataset_gnn))
    graph_labels = {
        "labels": torch.tensor(labels),
        "mask": torch.tensor(mask),
    }

    split_index_pd = pd.DataFrame(columns=["smiles", "group"])
    split_index_pd.smiles = smiles
    split_index_pd.group = split_index
    split_index_pd.to_csv(group_path, index=None, columns=None)

    print("Molecule graphs saved.")
    save_graphs(save_path, graphs, graph_labels)


def split_dataset_according_index(dataset, train_index, val_index, test_index, data_type: str = "np"):
    if data_type == "pd":
        return (
            pd.DataFrame(dataset[train_index]),
            pd.DataFrame(dataset[val_index]),
            pd.DataFrame(dataset[test_index]),
        )
    return dataset[train_index], dataset[val_index], dataset[test_index]


def load_graph_from_csv_bin_for_split(
    bin_path: str,
    group_path: str,
    select_task_index=None,
):
    smiles = pd.read_csv(group_path, index_col=None).smiles.values
    group = pd.read_csv(group_path, index_col=None).group.to_list()
    graphs, detailed_information = load_graphs(bin_path)
    labels = detailed_information["labels"]
    mask = detailed_information["mask"]

    if select_task_index is not None:
        labels = labels[:, select_task_index]
        mask = mask[:, select_task_index]

    notuse_mask = torch.mean(mask.float(), 1).numpy().tolist()
    not_use_index = []
    for index, notuse in enumerate(notuse_mask):
        if notuse == 0:
            not_use_index.append(index)

    train_index = []
    val_index = []
    test_index = []

    for index, group_index in enumerate(group):
        if group_index == "training" and index not in not_use_index:
            train_index.append(index)
        if group_index == "valid" and index not in not_use_index:
            val_index.append(index)
        if group_index == "test" and index not in not_use_index:
            test_index.append(index)

    graphs_np = np.array(list(graphs))
    train_smiles, val_smiles, test_smiles = split_dataset_according_index(smiles, train_index, val_index, test_index)
    train_labels, val_labels, test_labels = split_dataset_according_index(
        labels.numpy(), train_index, val_index, test_index, data_type="pd"
    )
    train_mask, val_mask, test_mask = split_dataset_according_index(
        mask.numpy(), train_index, val_index, test_index, data_type="pd"
    )
    train_graph, val_graph, test_graph = split_dataset_according_index(graphs_np, train_index, val_index, test_index)

    task_number = train_labels.values.shape[1]

    train_set = []
    val_set = []
    test_set = []

    for i in range(len(train_index)):
        train_set.append([train_smiles[i], train_graph[i], train_labels.values[i], train_mask.values[i]])
    for i in range(len(val_index)):
        val_set.append([val_smiles[i], val_graph[i], val_labels.values[i], val_mask.values[i]])
    for i in range(len(test_index)):
        test_set.append([test_smiles[i], test_graph[i], test_labels.values[i], test_mask.values[i]])

    print(len(train_set), len(val_set), len(test_set), task_number)
    return train_set, val_set, test_set, task_number
