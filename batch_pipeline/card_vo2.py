from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import load
from sklearn import preprocessing


def predict_fcard(df: pd.DataFrame, work_dir: Path) -> pd.DataFrame:
    file1 = df.copy()
    xtrain = pd.read_excel(work_dir / 'fcard/Xtrain_sy_card.xlsx')
    x_standard_train = xtrain.iloc[:, 3:8]
    scaler = preprocessing.StandardScaler().fit(x_standard_train)

    a = pd.read_excel(work_dir / 'card.xlsx')
    aa = a.iloc[:, 7:10]
    aa_sp = a.iloc[:, 10:]
    xt1 = pd.DataFrame(scaler.transform(aa), columns=aa.columns)
    car = pd.concat([xt1, aa_sp], axis=1)

    xtrainmodel = pd.read_excel(work_dir / 'fcard/Xtrain_std_card.xlsx').iloc[:, 1:20]
    xtestmodel = pd.read_excel(work_dir / 'fcard/Xtest_std_card.xlsx').iloc[:, 1:20]
    ytrain = pd.read_excel(work_dir / 'fcard/Ytrain_sy_card.xlsx').iloc[:, 1]
    ytest = pd.read_excel(work_dir / 'fcard/Ytest_sy_card.xlsx').iloc[:, 1]
    card_model = joblib.load(work_dir / 'fcard/cardiac et_model2.pkl')

    predictions = pd.DataFrame(card_model.predict(car))
    predictions.to_excel(work_dir / 'card pred.xlsx')

    v1y_pred = pd.read_excel(work_dir / 'fcard/et_test_pre_card.xlsx').iloc[:, 0]
    ytest_exp = pd.read_excel(work_dir / 'fcard/et_test_obs_card.xlsx')
    ytrain_exp = pd.read_excel(work_dir / 'fcard/et_train_obs_card.xlsx')
    v1y0_pred = pd.DataFrame(card_model.predict(xtrainmodel))

    _ = ytrain_exp - v1y0_pred
    _ = ytest_exp - pd.DataFrame(v1y_pred)

    tx = np.transpose(xtrainmodel)
    xtx = np.dot(tx, xtrainmodel)
    invxtx = np.linalg.inv(xtx)
    xsettest = car
    txexample = np.transpose(xsettest)
    hat0 = np.dot(xsettest, invxtx)
    hat = np.dot(hat0, txexample)
    leverage = np.diagonal(hat)

    existing_excel = pd.read_excel(work_dir / 'card pred.xlsx')
    existing_excel.insert(2, 'AD_Fcard', pd.DataFrame(leverage))
    existing_excel.to_excel(work_dir / 'card pred.xlsx', index=False)

    pred_df = pd.read_excel(work_dir / 'card pred.xlsx')
    pred_df['AD_Fcard'] = (pred_df['AD_Fcard'] <= 0.195).astype(int)
    pred_df = pred_df.drop(columns=pred_df.columns[0])
    pred_df = pred_df.rename(columns={pred_df.columns[0]: 'Fcard'})
    pred_df['Fcard'] = np.exp(pred_df['Fcard']).round(2)

    file1['Fcard'] = pred_df['Fcard']
    file1['AD_Fcard'] = pred_df['AD_Fcard']
    return file1


def predict_vo2(df: pd.DataFrame, work_dir: Path) -> pd.DataFrame:
    file1 = df.copy()
    xtrain = pd.read_excel(work_dir / 'vo2/Xtrain_sy_vo2.xlsx')
    x_standard_train = xtrain.iloc[:, 1:4]
    scaler = preprocessing.StandardScaler().fit(x_standard_train)

    a = pd.read_excel(work_dir / 'VO2.xlsx')
    a.rename(columns={'Length [cm]': 'Length (cm)'}, inplace=True)
    a.rename(columns={'Environment_Marine; freshwater': 'Environment_Marine; Freshwater'}, inplace=True)
    o2 = a.iloc[:, 6:9]
    o2_sp = a.iloc[:, 9:]
    xt1 = pd.DataFrame(scaler.transform(o2), columns=o2.columns)
    o2all = pd.concat([xt1, o2_sp], axis=1)

    xtrainmodel = pd.read_excel(work_dir / 'vo2/Xtrain_std_vo2.xlsx').iloc[:, 1:55]
    xtestmodel = pd.read_excel(work_dir / 'vo2/Xtest_std_vo2.xlsx').iloc[:, 1:55]
    ytrain = pd.read_excel(work_dir / 'vo2/Ytrain_sy_vo2.xlsx').iloc[:, 1]
    ytest = pd.read_excel(work_dir / 'vo2/Ytest_sy_vo2.xlsx').iloc[:, 1]
    loaded_model = load(work_dir / 'vo2/o2 rf_model.joblib')

    predictions = pd.DataFrame(loaded_model.predict(o2all))
    predictions.to_excel(work_dir / 'o2 pred.xlsx')

    y_pred_extra = pd.read_excel(work_dir / 'vo2/rf_test_pre_vo2.xlsx')
    ytest_exp = pd.read_excel(work_dir / 'vo2/rf_test_obs_vo2.xlsx')
    ytrain_exp = pd.read_excel(work_dir / 'vo2/rf_train_obs_vo2.xlsx')
    y_pred_extra0 = pd.DataFrame(loaded_model.predict(xtrainmodel))

    _ = ytrain_exp - y_pred_extra0
    _ = ytest_exp - y_pred_extra

    tx = np.transpose(xtrainmodel)
    xtx = np.dot(tx, xtrainmodel)
    invxtx = np.linalg.inv(xtx)
    xsettest = o2all
    txexample = np.transpose(xsettest)
    hat0 = np.dot(xsettest, invxtx)
    hat = np.dot(hat0, txexample)
    leverage = np.diagonal(hat)

    existing_excel = pd.read_excel(work_dir / 'o2 pred.xlsx')
    existing_excel.insert(2, 'AD_VO2', pd.DataFrame(leverage))
    existing_excel.to_excel(work_dir / 'o2 pred.xlsx', index=False)

    pred_df = pd.read_excel(work_dir / 'o2 pred.xlsx')
    pred_df['AD_VO2'] = (pred_df['AD_VO2'] <= 0.075).astype(int)
    pred_df = pred_df.drop(columns=pred_df.columns[0])
    pred_df = pred_df.rename(columns={pred_df.columns[0]: 'VO2'})
    pred_df['VO2'] = np.exp(pred_df['VO2']).round(2)
    pred_df.to_excel(work_dir / 'o2 pred.xlsx', index=False)

    file1['VO2'] = pred_df['VO2']
    file1['AD_VO2'] = pred_df['AD_VO2']
    file1.to_excel(work_dir / '1.xlsx')
    return file1
