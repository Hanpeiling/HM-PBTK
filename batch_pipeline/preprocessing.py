from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import merge_columns_by_smiles

warnings.filterwarnings('ignore')

SPECIAL_SPECIES = ['Oncorhynchus mykiss', 'Pimephales promelas']

INITIAL_EXTRA_COLUMNS = [
    'SMILES', 'AD_FUB', 'AD_LogD', 'AD_Clint', 'Fcard', 'AD_Fcard', 'VO2',
    'AD_VO2', 'Cl', 'AD_Cl', 'pKa_a_pred', 'pKa_b_pred', 'pKa', 'acid i=1,base i=-1',
    'fngill', 'fnfish', 'log_Kowion', 'Dowgill', 'Dow', 'sc_blood', 'sc_liver',
    'sc_gonads', 'sc_fat', 'sc_GIT', 'sc_brain', 'sc_kidney', 'sc_skin',
    'sc_rpt', 'sc_ppt', 'liver_frac', 'gonads_frac', 'GIT_frac', 'fat_frac',
    'brain_frac', 'kidney_frac', 'skin_frac', 'rpt_frac', 'ppt_frac',
    'water_liver', 'water_brain', 'water_gonads', 'water_fat', 'water_skin',
    'water_GIT', 'water_kidney', 'water_rpt', 'water_ppt', 'lipids_liver',
    'lipids_brain', 'lipids_gonads', 'lipids_fat', 'lipids_skin', 'lipids_GIT',
    'lipids_kidney', 'lipids_rpt', 'lipids_ppt'
]

FAMILY_COLUMNS = [
    'Family _Acipenseridae', 'Family _Adrianichthyidae', 'Family _Anguillidae', 'Family _Bagridae', 'Family _Balistidae', 'Family _Carangidae',
    'Family _Catostomidae', 'Family _Centrarchidae', 'Family _Chanidae', 'Family _Channidae', 'Family _Cichlidae', 'Family _Clariidae', 'Family _Clupeidae',
    'Family _Coryphaenidae', 'Family _Cottidae', 'Family _Cyprinidae', 'Family _Danionidae', 'Family _Eleginopidae', 'Family _Esocidae', 'Family _Gadidae', 'Family _Gasterosteidae', 'Family _Heteropneustidae',
    'Family _Ictaluridae', 'Family _Leuciscidae', 'Family _Lotidae', 'Family _Moronidae', 'Family _Mugilidae', 'Family _Myctophidae', 'Family _Myxinidae',
    'Family _Osphronemidae', 'Family _Percidae', 'Family _Pleuronectidae', 'Family _Poeciliidae', 'Family _Polypteridae',
    'Family _Pomacentridae', 'Family _Salmonidae', 'Family _Scombridae', 'Family _Scophthalmidae', 'Family _Scorpaenidae',
    'Family _Scyliorhinidae', 'Family _Serrasalmidae', 'Family _Sparidae', 'Family _Squalidae', 'Family _Synbranchidae', 'Family _Syngnathidae', 'Family _Trachinidae',
    'Family_Anabantidae', 'Family_Anguillidae', 'Family_Channichthyidae', 'Family_Cichlidae', 'Family_Cyprinidae',
    'Family_Adrianichthyidae', 'Family_Danionidae', 'Family_Engraulidae', 'Family_Ictaluridae', 'Family_Labridae', 'Family_Lateolabracidae', 'Family_Leuciscidae', 'Family_Moronidae', 'Family_Mullidae', 'Family_Nototheniidae', 'Family_Petromyzontidae', 'Family_Pleuronectidae', 'Family_Poeciliidae', 'Family_Salmonidae', 'Family_Sparidae'
]

ENVIRONMENT_COLUMNS = ['Environment_Freshwater', 'Environment_Marine', 'Environment_Marine; freshwater']


def load_input_dataframe(input_excel: Path) -> pd.DataFrame:
    df = pd.read_excel(input_excel)
    new_column_names = {
        'Chemicals': 'chemicals',
        'Species': 'species',
        'Gender': 'gender',
        'Total time,day': 'stop',
        'Exposure time,day': 'time_final_dose',
        'LogKow': 'logKow',
        'Water pH': 'pH',
        'Body weight,g': 'BW',
        'Body length,cm': 'BL',
        'Water temperature,℃': 'TC_c',
        'Exposure dose,μg/mL': 'Dose_water',
        'Plasma fraction unbound': 'Unbound_fraction',
    }
    df = df.rename(columns=new_column_names)
    if 'species' in df.columns:
        df['species'] = df['species'].astype(str).str.strip().str.replace(' ', '', regex=False)
    for column in INITIAL_EXTRA_COLUMNS:
        df[column] = pd.NA
    return df


def populate_smiles_family_environment(df: pd.DataFrame, biochemical_excel: Path, physiological_excel: Path) -> pd.DataFrame:
    file1 = df.copy()
    file2 = pd.read_excel(biochemical_excel, sheet_name='chemicals')
    file2['CAS_RN'] = file2['CAS_RN'].str.replace(r'^CAS_RN:\s*', '', regex=True)
    file1['SMILES'] = ''

    for idx, chemical in file1['chemicals'].items():
        pattern = r'\b' + re.escape(str(chemical)) + r'\b'
        matched_rows = file2.apply(lambda row: row.astype(str).str.contains(pattern, na=False), axis=1).any(axis=1)
        smiles_values = file2.loc[matched_rows, 'SMILES']
        if not smiles_values.empty:
            file1.at[idx, 'SMILES'] = smiles_values.iloc[0]

    species_lookup = pd.read_excel(physiological_excel, sheet_name='list')
    file1['family'] = ''
    file1['environment'] = ''
    for idx, species in file1['species'].items():
        matched_rows = species_lookup[species_lookup['species'] == species]
        if matched_rows.empty:
            continue
        file1.at[idx, 'family'] = matched_rows['family'].values[0]
        file1.at[idx, 'environment'] = matched_rows['environment'].values[0]
    return file1


def _resolve_physiology_sheet(df: pd.DataFrame, row_idx: int, physiological_excel: Path) -> pd.DataFrame:
    species_info = df.at[row_idx, 'species']
    try:
        return pd.read_excel(physiological_excel, sheet_name=species_info)
    except (KeyError, ValueError):
        family_info = df.at[row_idx, 'family']
        try:
            return pd.read_excel(physiological_excel, sheet_name=family_info)
        except (KeyError, ValueError):
            environment_info = df.at[row_idx, 'environment']
            return pd.read_excel(physiological_excel, sheet_name=environment_info)


def fill_physiological_parameters(df: pd.DataFrame, physiological_excel: Path) -> pd.DataFrame:
    file1 = df.copy()

    for i, species_info in enumerate(file1['species']):
        try:
            phys_params = _resolve_physiology_sheet(file1, i, physiological_excel)
            frac_data = phys_params['_frac']
            if len(frac_data) < 9:
                raise ValueError(f"Not enough data in '_frac' column for {species_info}.")
            frac_data = frac_data.iloc[:9].copy()
            frac_data.index = ['fat_frac', 'brain_frac', 'GIT_frac', 'gonads_frac', 'kidney_frac', 'liver_frac', 'skin_frac', 'ppt_frac', 'rpt_frac']
            for idx, col_name in enumerate(frac_data.index):
                file1.loc[i, col_name] = frac_data.iloc[idx]
        except (KeyError, ValueError) as exc:
            print(f'An error occurred for species {species_info}: {exc}')

    for i, (species_info, gender_info) in enumerate(zip(file1['species'], file1['gender'])):
        try:
            phys_params = _resolve_physiology_sheet(file1, i, physiological_excel) if species_info not in SPECIAL_SPECIES else pd.read_excel(physiological_excel, sheet_name=species_info)
            if str(gender_info).lower() == 'male':
                data_column = phys_params.iloc[:, 4]
            elif str(gender_info).lower() == 'female':
                data_column = phys_params.iloc[:, 5]
            else:
                raise ValueError(f'Invalid gender {gender_info} for species {species_info}')
            if len(data_column) < 9:
                raise ValueError(f'Not enough lipid data for {species_info}')
            data_column = data_column.iloc[:9].copy()
            data_column.index = ['lipids_fat', 'lipids_brain', 'lipids_GIT', 'lipids_gonads', 'lipids_kidney', 'lipids_liver', 'lipids_skin', 'lipids_ppt', 'lipids_rpt']
            for idx, col_name in enumerate(data_column.index):
                file1.loc[i, col_name] = data_column.iloc[idx]
        except (KeyError, ValueError) as exc:
            print(f'An error occurred for species {species_info}: {exc}')

    for i, (species_info, gender_info) in enumerate(zip(file1['species'], file1['gender'])):
        try:
            phys_params = _resolve_physiology_sheet(file1, i, physiological_excel) if species_info not in SPECIAL_SPECIES else pd.read_excel(physiological_excel, sheet_name=species_info)
            if str(gender_info).lower() == 'male':
                data_column = phys_params.iloc[:, 6]
            elif str(gender_info).lower() == 'female':
                data_column = phys_params.iloc[:, 7]
            else:
                raise ValueError(f'Invalid gender {gender_info} for species {species_info}')
            if len(data_column) < 9:
                raise ValueError(f'Not enough water data for {species_info}')
            data_column = data_column.iloc[:9].copy()
            data_column.index = ['water_fat', 'water_brain', 'water_GIT', 'water_gonads', 'water_kidney', 'water_liver', 'water_skin', 'water_ppt', 'water_rpt']
            for idx, col_name in enumerate(data_column.index):
                file1.loc[i, col_name] = data_column.iloc[idx]
        except (KeyError, ValueError) as exc:
            print(f'An error occurred for species {species_info}: {exc}')

    for i, (species_info, gender_info) in enumerate(zip(file1['species'], file1['gender'])):
        try:
            phys_params = _resolve_physiology_sheet(file1, i, physiological_excel) if species_info not in SPECIAL_SPECIES else pd.read_excel(physiological_excel, sheet_name=species_info)
            if str(gender_info).lower() == 'male':
                data_column = phys_params.iloc[:, 2]
            elif str(gender_info).lower() == 'female':
                data_column = phys_params.iloc[:, 3]
            else:
                raise ValueError(f'Invalid gender {gender_info} for species {species_info}')
            if len(data_column) < 10:
                raise ValueError(f'Not enough sc data for {species_info}')
            data_column = data_column.iloc[:10].copy()
            data_column.index = ['sc_fat', 'sc_brain', 'sc_GIT', 'sc_gonads', 'sc_kidney', 'sc_liver', 'sc_skin', 'sc_ppt', 'sc_rpt', 'sc_blood']
            for idx, col_name in enumerate(data_column.index):
                file1.loc[i, col_name] = data_column.iloc[idx]
        except (KeyError, ValueError) as exc:
            print(f'An error occurred for species {species_info}: {exc}')

    return file1


def merge_biochemical_parameter_sheet(df: pd.DataFrame, biochemical_excel: Path) -> pd.DataFrame:
    merged = merge_columns_by_smiles(df, pd.read_excel(biochemical_excel, sheet_name='para'))
    for idx, row in merged.iterrows():
        if pd.notna(row['Unbound_fraction']):
            merged.at[idx, 'AD_FUB'] = 'input'
        else:
            merged.at[idx, 'Unbound_fraction'] = merged.at[idx, 'Unbound_fraction_pred']
            merged.at[idx, 'AD_FUB'] = merged.at[idx, 'AD_FUB_pred']
        if pd.isna(row['logKow']):
            merged.at[idx, 'logKow'] = merged.at[idx, 'logKow_pred']
    return merged


def merge_logd_and_estimate_body_length(df: pd.DataFrame, biochemical_excel: Path, physiological_excel: Path) -> pd.DataFrame:
    file1 = merge_columns_by_smiles(df, pd.read_excel(biochemical_excel, sheet_name='logD'))
    file_list = pd.read_excel(physiological_excel, sheet_name='list')
    for i, species in enumerate(file1['species']):
        try:
            if not pd.isnull(file1.loc[i, 'BL']) and np.isfinite(file1.loc[i, 'BL']):
                continue
            matching_row = file_list[file_list['species'] == species]
            if matching_row.empty:
                raise ValueError(f'No matching species found for {species}')
            a = matching_row['a'].values[0]
            b = matching_row['b'].values[0]
            bw = file1.loc[i, 'BW']
            if bw <= 0 or a <= 0 or b <= 0:
                raise ValueError(f'Invalid values for calculation: BW={bw}, a={a}, b={b}')
            bl = 10 ** ((np.log10(bw) - np.log10(a)) / b)
            file1.loc[i, 'BL'] = round(bl, 1)
        except (KeyError, ValueError) as exc:
            print(f'An error occurred for species {species}: {exc}')
    return file1


def calculate_pka_features(df: pd.DataFrame) -> pd.DataFrame:
    file1 = df.copy()
    file1['pKa_a_pred'] = pd.to_numeric(file1['pKa_a_pred'], errors='coerce')
    file1['pKa_b_pred'] = pd.to_numeric(file1['pKa_b_pred'], errors='coerce')
    file1['pKa'] = pd.NA
    file1['acid i=1,base i=-1'] = pd.NA

    for i, (pka_a, pka_b) in enumerate(zip(file1['pKa_a_pred'], file1['pKa_b_pred'])):
        if pd.notna(pka_a) and pd.isna(pka_b):
            file1.loc[i, 'pKa'] = pka_a
            file1.loc[i, 'acid i=1,base i=-1'] = 1
        elif pd.isna(pka_a) and pd.notna(pka_b):
            file1.loc[i, 'pKa'] = pka_b
            file1.loc[i, 'acid i=1,base i=-1'] = -1
        elif pd.notna(pka_a) and pd.notna(pka_b):
            file1.loc[i, 'pKa'] = pka_a
            file1.loc[i, 'acid i=1,base i=-1'] = 1
        else:
            file1.loc[i, 'pKa'] = pd.NA
            file1.loc[i, 'acid i=1,base i=-1'] = pd.NA

    pH = 7.4
    file1['fngill'] = np.nan
    file1['fnfish'] = np.nan
    file1['log_Kowion'] = np.nan
    file1['Dowgill'] = np.nan
    file1['Dow'] = np.nan

    for i, row in file1.iterrows():
        if pd.isna(row['pKa']):
            continue
        pka = row['pKa']
        acid_base = row['acid i=1,base i=-1']
        log_kow = float(row['logKow'])
        fngill = 1 / (1 + (10 ** (acid_base * (pH - pka))))
        fnfish = 1 / (1 + (10 ** (acid_base * (7.4 - pka))))
        log_kowion = log_kow - 3.5
        dowgill = fngill * (10 ** log_kow) + (1 - fngill) * (10 ** log_kowion)
        dow = fnfish * (10 ** log_kow) + (1 - fnfish) * (10 ** log_kowion)
        file1.loc[i, 'fngill'] = fngill
        file1.loc[i, 'fnfish'] = fnfish
        file1.loc[i, 'log_Kowion'] = log_kowion
        file1.loc[i, 'Dowgill'] = dowgill
        file1.loc[i, 'Dow'] = round(dow, 1)
    return file1


def build_feature_workbooks(df: pd.DataFrame, work_dir: Path, biochemical_excel: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    file1 = df.copy()
    file1['Weight [g^0.75]'] = file1['BW'] ** 0.75
    file1['Length [cm]'] = file1['BL']
    file1['Temperature [k^-1]'] = 1000 / (273 + file1['TC_c'])

    for family in FAMILY_COLUMNS:
        family_name = family.split('_')[1]
        file1[family] = (file1['family'] == family_name).astype(int)

    for env in ENVIRONMENT_COLUMNS:
        file1[env] = 0

    for i, row in file1.iterrows():
        env = row['environment']
        if pd.isna(env):
            continue
        if 'Marine; Freshwater' in str(env):
            file1.loc[i, 'Environment_Marine; freshwater'] = 1
            file1.loc[i, 'Environment_Freshwater'] = 0
            file1.loc[i, 'Environment_Marine'] = 0
        else:
            if 'Freshwater' in str(env):
                file1.loc[i, 'Environment_Freshwater'] = 1
            if 'Marine' in str(env):
                file1.loc[i, 'Environment_Marine'] = 1

    card = file1[[
        'SMILES', 'chemicals', 'species', 'family', 'environment', 'gender', 'Weight [g^0.75]', 'Length [cm]', 'Temperature [k^-1]',
        'Family _Acipenseridae', 'Family _Anguillidae', 'Family _Catostomidae', 'Family _Centrarchidae',
        'Family _Coryphaenidae', 'Family _Cyprinidae', 'Family _Danionidae', 'Family _Eleginopidae',
        'Family _Ictaluridae', 'Family _Leuciscidae', 'Family _Moronidae', 'Family _Salmonidae',
        'Environment_Freshwater', 'Environment_Marine', 'Environment_Marine; freshwater'
    ]]
    card.to_excel(work_dir / 'card.xlsx')

    vo2 = file1[[
        'SMILES', 'species', 'family', 'environment', 'gender', 'Length [cm]', 'Weight [g^0.75]', 'Temperature [k^-1]',
        'Family _Acipenseridae', 'Family _Adrianichthyidae', 'Family _Anguillidae', 'Family _Bagridae', 'Family _Balistidae', 'Family _Carangidae',
        'Family _Catostomidae', 'Family _Centrarchidae', 'Family _Chanidae', 'Family _Channidae', 'Family _Cichlidae', 'Family _Clariidae',
        'Family _Clupeidae', 'Family _Coryphaenidae', 'Family _Cottidae', 'Family _Cyprinidae', 'Family _Esocidae', 'Family _Gadidae',
        'Family _Gasterosteidae', 'Family _Heteropneustidae', 'Family _Ictaluridae', 'Family _Leuciscidae', 'Family _Lotidae', 'Family _Mugilidae',
        'Family _Myctophidae', 'Family _Myxinidae', 'Family _Osphronemidae', 'Family _Percidae',
        'Family _Pleuronectidae', 'Family _Poeciliidae', 'Family _Polypteridae', 'Family _Pomacentridae', 'Family _Salmonidae',
        'Family _Scombridae', 'Family _Scophthalmidae', 'Family _Scorpaenidae', 'Family _Scyliorhinidae', 'Family _Serrasalmidae',
        'Family _Sparidae', 'Family _Squalidae', 'Family _Synbranchidae', 'Family _Syngnathidae', 'Family _Trachinidae',
        'Environment_Freshwater', 'Environment_Marine', 'Environment_Marine; freshwater'
    ]]
    vo2.to_excel(work_dir / 'VO2.xlsx')

    fish_ready = merge_columns_by_smiles(file1, pd.read_excel(biochemical_excel, sheet_name='fish bin'))
    fish_bin_data = fish_ready[[
        'SMILES', 'species', 'family', 'Clhuman', 'LogD55_pred', 'LogD74_pred', 'BLI', 'X0Av', 'VE3sign_X', 'J_Dz(v)',
        'ATSC8e', 'Eig13_EA(dm)', 'EE_G', 'VE3sign_G', 'VE1_RG', 'VE3sign_RG', 'VE3sign_G/D', 'TDB01i',
        'Mor04u', 'Mor15u', 'Mor22u', 'Mor23u', 'Mor12m', 'Mor06v', 'Mor10v', 'Mor22v', 'Mor04s', 'Mor08s', 'Mor12s',
        'Mor17s', 'Mor28s', 'E1u', 'E1s', 'H6u', 'HATS2u', 'R4u', 'R5u', 'R4s', 'Family_Adrianichthyidae',
        'Family_Anabantidae', 'Family_Anguillidae', 'Family_Channichthyidae', 'Family_Cichlidae', 'Family_Cyprinidae',
        'Family_Danionidae', 'Family_Engraulidae', 'Family_Ictaluridae', 'Family_Labridae', 'Family_Lateolabracidae', 'Family_Leuciscidae', 'Family_Moronidae', 'Family_Mullidae', 'Family_Nototheniidae', 'Family_Petromyzontidae', 'Family_Pleuronectidae', 'Family_Poeciliidae', 'Family_Salmonidae', 'Family_Sparidae', 'Environment_Freshwater', 'Environment_Marine', 'Environment_Marine; freshwater'
    ]]
    fish_cl_data = fish_ready[[
        'species', 'family', 'Clhuman', 'LogD55_pred', 'LogD74_pred', 'BLI', 'X0Av', 'VE3sign_X', 'J_Dz(v)',
        'ATSC8e', 'Eig13_EA(dm)', 'EE_G', 'VE3sign_G', 'VE1_RG', 'VE3sign_RG', 'VE3sign_G/D', 'TDB01i',
        'Mor04u', 'Mor15u', 'Mor22u', 'Mor23u', 'Mor12m', 'Mor06v', 'Mor10v', 'Mor22v', 'Mor04s', 'Mor08s', 'Mor12s',
        'Mor17s', 'Mor28s', 'E1u', 'E1s', 'H6u', 'HATS2u', 'R4u', 'R5u', 'R4s', 'Family_Adrianichthyidae',
        'Family_Anabantidae', 'Family_Anguillidae', 'Family_Channichthyidae', 'Family_Cichlidae', 'Family_Cyprinidae',
        'Family_Danionidae', 'Family_Engraulidae', 'Family_Ictaluridae', 'Family_Labridae', 'Family_Lateolabracidae', 'Family_Leuciscidae', 'Family_Moronidae', 'Family_Mullidae', 'Family_Nototheniidae', 'Family_Petromyzontidae', 'Family_Pleuronectidae', 'Family_Poeciliidae', 'Family_Salmonidae', 'Family_Sparidae', 'Environment_Freshwater', 'Environment_Marine', 'Environment_Marine; freshwater'
    ]]
    logd_data = fish_ready[['SMILES', 'AD_LogD', 'AD_Clint']]
    with pd.ExcelWriter(work_dir / 'data.xlsx') as writer:
        fish_bin_data.to_excel(writer, sheet_name='fish bin', index=False)
        logd_data.to_excel(writer, sheet_name='logD', index=False)
        fish_cl_data.to_excel(writer, sheet_name='fish cl', index=False)
    return file1, fish_bin_data
