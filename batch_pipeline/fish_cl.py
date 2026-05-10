from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing

from .ad_domain import NSG


def _predict_fish_bin(work_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    xtrain = pd.read_excel(work_dir / 'cL/Xtrain_sy_fish.xlsx').iloc[:, 1:]
    _ = pd.read_excel(work_dir / 'cL/Xtest_sy_fish.xlsx').iloc[:, 1:]
    x_standard_train = xtrain.iloc[:, 0:35]
    scaler = preprocessing.StandardScaler().fit(x_standard_train)

    bin2 = pd.read_excel(work_dir / 'data.xlsx', sheet_name='fish bin')
    bin_st = bin2.iloc[:, 3:38]
    bin_sp = bin2.iloc[:, 38:]
    bin1 = pd.DataFrame(scaler.transform(bin_st), columns=bin_st.columns)
    new_data1 = pd.concat([bin1, bin_sp], axis=1)

    loaded_model = joblib.load(work_dir / 'cL/GB_cls_to pre fishbin final_122.pkl')
    predictions = pd.DataFrame(loaded_model.predict(new_data1))
    return predictions, bin2


def _calculate_query_ad_scores(bin2: pd.DataFrame, work_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(work_dir / 'cL/nosmote.xlsx')
    smiles_to_id: dict[str, int] = {}
    current_id = 1

    def get_id(smiles):
        nonlocal current_id
        if smiles not in smiles_to_id:
            smiles_to_id[smiles] = current_id
            current_id += 1
        return smiles_to_id[smiles]

    df['CmpdID'] = df['smiles'].apply(get_id)
    df_training = df[df['group'] == 'training'].copy()

    nsg = NSG(df_training, smiCol='smiles', yCol='P')
    nsg.calcPairwiseSimilarityWithFp('MACCS_keys')

    df_smiles = bin2.copy().rename(columns={'SMILES': 'smiles'})
    p_list, i_list = [], []
    for smiles in df_smiles['smiles']:
        df_vad = pd.DataFrame({'CmpdID': ['query'], 'smiles': [smiles]})
        qtsm = nsg.genQTSM(df_vad, smiCol='smiles')
        ad_metric = nsg.queryADMetrics(qtsm)
        p_list.append(ad_metric.simiDensity.values[0])
        i_list.append(ad_metric.simiWtLD_w.values[0])
    df_smiles['P'] = p_list
    df_smiles['I'] = i_list
    return df_smiles[['P', 'I']]


def predict_fish_cl(df: pd.DataFrame, fish_bin_data: pd.DataFrame, work_dir: Path, original_input_excel: Path) -> pd.DataFrame:
    file1 = df.copy()
    bin_predictions, bin2 = _predict_fish_bin(work_dir)
    pi_scores = _calculate_query_ad_scores(bin2, work_dir)

    contat = pd.concat([bin_predictions, pi_scores], axis=1)
    contat['AD'] = contat.apply(lambda row: 1 if row['P'] > 1 and row['I'] < 1 else 0, axis=1)
    contat = contat.drop(contat.columns[[1, 2]], axis=1)
    contat.columns = ['Bin3 ', contat.columns[1]]

    logd_sheet = pd.read_excel(work_dir / 'data.xlsx', sheet_name='logD')
    contatt = pd.concat([contat, logd_sheet], axis=1)
    contatt['AD'] = pd.to_numeric(contatt['AD'], errors='coerce').fillna(0).astype(int)
    contatt['AD_LogD'] = pd.to_numeric(contatt['AD_LogD'], errors='coerce').fillna(0).astype(int)
    contatt['AD_Clint'] = pd.to_numeric(contatt['AD_Clint'], errors='coerce').fillna(0).astype(int)
    contatt['AD_final'] = (contatt['AD'] & contatt['AD_LogD'] & contatt['AD_Clint']).astype(int)
    contatt_final = contatt[['Bin3 ', 'AD_final']].rename(columns={'AD_final': 'AD'})

    bb = pd.concat([fish_bin_data, contatt_final], axis=1)
    a = bb[[
        'species', 'family', 'Clhuman', 'LogD55_pred', 'LogD74_pred', 'BLI', 'X0Av', 'VE3sign_X', 'J_Dz(v)',
        'ATSC8e', 'Eig13_EA(dm)', 'EE_G', 'VE3sign_G', 'VE1_RG', 'VE3sign_RG', 'VE3sign_G/D', 'TDB01i',
        'Mor04u', 'Mor15u', 'Mor22u', 'Mor23u', 'Mor12m', 'Mor06v', 'Mor10v', 'Mor22v', 'Mor04s', 'Mor08s', 'Mor12s',
        'Mor17s', 'Mor28s', 'E1u', 'E1s', 'H6u', 'HATS2u', 'R4u', 'R5u', 'R4s', 'Family_Adrianichthyidae',
        'Family_Anabantidae', 'Family_Anguillidae', 'Family_Channichthyidae', 'Family_Cichlidae', 'Family_Cyprinidae',
        'Family_Danionidae', 'Family_Engraulidae', 'Family_Ictaluridae', 'Family_Labridae', 'Family_Lateolabracidae', 'Family_Leuciscidae', 'Family_Moronidae', 'Family_Mullidae', 'Family_Nototheniidae', 'Family_Petromyzontidae', 'Family_Pleuronectidae', 'Family_Poeciliidae', 'Family_Salmonidae', 'Family_Sparidae', 'Environment_Freshwater', 'Environment_Marine', 'Environment_Marine; freshwater', 'Bin3 '
    ]]

    xtrain = pd.read_excel(work_dir / 'cL/Xtrain_sy_eu.xlsx')
    x_standard_train = xtrain.iloc[:, 4:]
    scaler = preprocessing.StandardScaler().fit(x_standard_train)
    a_st = pd.DataFrame(a.iloc[:, 2:-24])
    a_sp = a.iloc[:, -24:]
    xt = pd.DataFrame(scaler.transform(a_st), columns=a_st.columns)
    input_fish = pd.concat([xt, a_sp], axis=1)

    xtrainmodel = pd.read_excel(work_dir / 'cL/Xtrain_std_eu.xlsx').iloc[:, 1:80]
    xtestmodel = pd.read_excel(work_dir / 'cL/Xtest_std_eu.xlsx').iloc[:, 1:80]
    loaded_model = joblib.load(work_dir / 'cL/eu de fish model_122.pkl')
    predictions = pd.DataFrame(loaded_model.predict(input_fish))

    v1y_pred = pd.read_excel(work_dir / 'cL/VOTE_test_pre.xlsx')
    _ = pd.read_excel(work_dir / 'cL/VOTE_test_obs.xlsx')
    _ = pd.read_excel(work_dir / 'cL/VOTE_train_obs.xlsx')
    _ = pd.DataFrame(loaded_model.predict(xtrainmodel))

    tx = np.transpose(xtrainmodel)
    xtx = np.dot(tx, xtrainmodel)
    invxtx = np.linalg.inv(xtx)
    xsettest = input_fish
    txexample = np.transpose(xsettest)
    hat0 = np.dot(xsettest, invxtx)
    hat = np.dot(hat0, txexample)
    leverage = np.diagonal(hat)

    predictions.to_excel(work_dir / 'fish cl pred.xlsx')
    existing_excel = pd.read_excel(work_dir / 'fish cl pred.xlsx')
    existing_excel.insert(2, 'AD', pd.DataFrame(leverage))
    existing_excel.to_excel(work_dir / 'fish cl pred.xlsx', index=False)

    pred_df = pd.read_excel(work_dir / 'fish cl pred.xlsx')
    pred_df['AD'] = (pred_df['AD'] <= 0.818).astype(int)
    pred_df = pred_df.drop(columns=pred_df.columns[0])
    pred_df = pred_df.rename(columns={pred_df.columns[0]: 'Cl'})
    pred_df['Cl'] = ((10 ** pred_df['Cl']) * 24 * 60 * 510 / 1000).round(2)
    pred_df['AD2'] = contatt_final['AD'].astype(int)
    pred_df['AD'] = pred_df['AD'].astype(int)
    pred_df['AD_Cl'] = (pred_df['AD'] & pred_df['AD2']).astype(int)
    pred_df = pred_df.drop(columns=pred_df.columns[1])
    pred_df = pred_df.drop(columns=pred_df.columns[1])

    file1['Cl'] = pred_df['Cl']
    file1['AD_Cl'] = pred_df['AD_Cl']
    file1.rename(columns={'acid i=1,base i=-1': 'aa'}, inplace=True)
    file1.rename(columns={'Total time ': 'stop'}, inplace=True)

    original_df = pd.read_excel(original_input_excel).rename(columns={'Species': 'species'})
    file1['species'] = original_df['species']

    ionizable_rows = file1[
        (pd.to_numeric(file1['pKa_a_pred'], errors='coerce').notna()) |
        (pd.to_numeric(file1['pKa_b_pred'], errors='coerce').notna())
    ]
    neutral_rows = file1[
        (pd.to_numeric(file1['pKa_a_pred'], errors='coerce').isna()) &
        (pd.to_numeric(file1['pKa_b_pred'], errors='coerce').isna())
    ]

    with pd.ExcelWriter(work_dir / '1.xlsx', engine='xlsxwriter') as excel_writer:
        ionizable_rows.to_excel(excel_writer, sheet_name='dl', index=False)
        neutral_rows.to_excel(excel_writer, sheet_name='zx', index=False)

    excel_data = pd.ExcelFile(work_dir / '1.xlsx')
    with pd.ExcelWriter(work_dir / '11.xlsx', engine='openpyxl') as writer:
        for sheet_name in excel_data.sheet_names:
            sheet_df = pd.read_excel(work_dir / '1.xlsx', sheet_name=sheet_name)
            sheet_df.insert(0, 'num', range(len(sheet_df)))
            sheet_df['pKa_a_pred'].fillna(0, inplace=True)
            sheet_df['pKa_b_pred'].fillna(0, inplace=True)
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    return file1
