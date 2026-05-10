from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    input_excel: str = 'Batch prediction.xlsx'
    biochemical_excel: str = 'Biochemical_parameters.xlsx'
    physiological_excel: str = 'Physiological_parameters.xlsx'
    output_intermediate_excel: str = '1.xlsx'
    output_final_excel: str = '11.xlsx'


REQUIRED_MODEL_INPUTS = [
    'Biochemical_parameters.xlsx',
    'Physiological_parameters.xlsx',
    'fcard/Xtrain_sy_card.xlsx',
    'fcard/Xtest_sy_card.xlsx',
    'fcard/Ytrain_sy_card.xlsx',
    'fcard/Ytest_sy_card.xlsx',
    'fcard/Xtrain_std_card.xlsx',
    'fcard/Xtest_std_card.xlsx',
    'fcard/cardiac et_model2.pkl',
    'fcard/et_test_pre_card.xlsx',
    'fcard/et_test_obs_card.xlsx',
    'fcard/et_train_obs_card.xlsx',
    'fcard/et_train_pre_card.xlsx',
    'vo2/Xtrain_sy_vo2.xlsx',
    'vo2/Xtest_sy_vo2.xlsx',
    'vo2/Ytrain_sy_vo2.xlsx',
    'vo2/Ytest_sy_vo2.xlsx',
    'vo2/Xtrain_std_vo2.xlsx',
    'vo2/Xtest_std_vo2.xlsx',
    'vo2/o2 rf_model.joblib',
    'vo2/rf_test_pre_vo2.xlsx',
    'vo2/rf_test_obs_vo2.xlsx',
    'vo2/rf_train_obs_vo2.xlsx',
    'vo2/rf_train_pre_vo2.xlsx',
    'cL/Xtrain_sy_fish.xlsx',
    'cL/Xtest_sy_fish.xlsx',
    'cL/Ytrain_sy_fish.xlsx',
    'cL/Ytest_sy_fish.xlsx',
    'cL/Xtrain_std_fish.xlsx',
    'cL/Xtest_std_fish.xlsx',
    'cL/GB_cls_to pre fishbin final_122.pkl',
    'cL/nosmote.xlsx',
    'cL/Xtrain_sy_eu.xlsx',
    'cL/Xtest_sy_eu.xlsx',
    'cL/Ytrain_sy_eu.xlsx',
    'cL/Ytest_sy_eu.xlsx',
    'cL/Xtrain_std_eu.xlsx',
    'cL/Xtest_std_eu.xlsx',
    'cL/eu de fish model_122.pkl',
    'cL/VOTE_test_pre.xlsx',
    'cL/VOTE_test_obs.xlsx',
    'cL/VOTE_train_obs.xlsx',
    'cL/VOTE_train_pre.xlsx',
]
