from __future__ import annotations

from pathlib import Path

from .card_vo2 import predict_fcard, predict_vo2
from .config import REQUIRED_MODEL_INPUTS
from .fish_cl import predict_fish_cl
from .preprocessing import (
    build_feature_workbooks,
    calculate_pka_features,
    fill_physiological_parameters,
    load_input_dataframe,
    merge_biochemical_parameter_sheet,
    merge_logd_and_estimate_body_length,
    populate_smiles_family_environment,
)
from .utils import ensure_required_files


def run_pipeline(work_dir: str | Path = '.', input_excel: str = 'Batch prediction.xlsx') -> Path:
    work_dir = Path(work_dir).resolve()
    input_path = work_dir / input_excel
    ensure_required_files(work_dir, [input_excel, *REQUIRED_MODEL_INPUTS])

    df = load_input_dataframe(input_path)
    df = populate_smiles_family_environment(df, work_dir / 'Biochemical_parameters.xlsx', work_dir / 'Physiological_parameters.xlsx')
    df = fill_physiological_parameters(df, work_dir / 'Physiological_parameters.xlsx')
    df = merge_biochemical_parameter_sheet(df, work_dir / 'Biochemical_parameters.xlsx')
    df = merge_logd_and_estimate_body_length(df, work_dir / 'Biochemical_parameters.xlsx', work_dir / 'Physiological_parameters.xlsx')
    df = calculate_pka_features(df)
    df, fish_bin_data = build_feature_workbooks(df, work_dir, work_dir / 'Biochemical_parameters.xlsx')
    df = predict_fcard(df, work_dir)
    df = predict_vo2(df, work_dir)
    _ = predict_fish_cl(df, fish_bin_data, work_dir, input_path)
    return work_dir / '11.xlsx'
