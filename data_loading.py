import re
import pandas as pd
import numpy as np
from typing import Dict, Union
# ydata profiling imports
try:
    from ydata_profiling import ProfileReport
except:
    has_ydata = False
else:
    has_ydata = True
import os
from tqdm import tqdm
import warnings

#from datenimport_aicare.data_preprocessing import tumorDataset 

def aggregation_function(x: pd.Series) -> pd.Series:
    '''
    Aggregation function for the aggregation of data depending on data type.
    
    Parameters:
        x (pd.Series): Series to aggregate
    '''
    x = x.dropna()
    if len(x) == 0:
        return pd.NA
    if x.dtype == np.float64:
        return x.mean()
    elif x.dtype == pd.StringDtype():
        return ", ".join(x.unique())
    elif x.dtype == pd.Timestamp:
        return x.max()
    return x.iloc[0]

def convert_array_type(col:pd.Series, target) -> pd.Series:
    '''
    Function to split multiple values in Series 
    and convert them to chosen datatype with convert_single_col

    Parameters: 
    col (pd.Series): Column to convert
    target (type): Target data type
    '''
    column = []

    for v in range(len(col)):
        if not pd.isnull(col.iloc[v]):
            values = col.iloc[v].split(';')
            if target == int:
            #    value = [int(v) for v in value]

                values = np.array(values, dtype=int)
            else:
                values = pd.Series(values, dtype=target)
        else:
            values = col.iloc[v]

        #value = pd.Series(value, dtype=target)
        
        column.append(values)
    if len(column) > 0:    
        column = pd.Series(column)
    else:
        column = pd.Series(column, dtype='object')

    return column


def convert_single_col(col:pd.Series, target) -> pd.Series:
    '''
    Converts a column to a specific data type.

    Parameters:
        col (pd.Series): Column to convert
        target (type): Target data type
    '''
    if target == pd.StringDtype:
        col = col.astype(pd.StringDtype())

    if target == pd.CategoricalDtype:
        if col.dtype == pd.StringDtype():
            col = col.str.upper().str.strip()
        col = col.astype(pd.CategoricalDtype(ordered=True))
    elif target == pd.Timestamp:	
        col = pd.to_datetime(col, format="%Y-%m-%d")
    elif target == pd.Int32Dtype:
        col = col.astype(pd.Int32Dtype())

    return col

def convert_all_cols(cols:pd.DataFrame, date_cols:list=[], cat_cols=[], num_cols=[], array_cols=[]):
    new_df = pd.DataFrame()
    if len(date_cols) > 0:
        for date_col in date_cols:
            new_df.loc[:,date_col] = convert_single_col(cols[date_col], pd.Timestamp).copy()
    if len(cat_cols) > 0:
        for cat_col in cat_cols:
            new_df.loc[:,cat_col] = convert_single_col(cols[cat_col], pd.CategoricalDtype).copy()
    if len(num_cols) > 0:
        for num_col in num_cols:
            new_df.loc[:,num_col] = convert_single_col(cols[num_col], pd.Int32Dtype).copy()
    if len(array_cols) > 0:
        for array_col in range(len(array_cols['name'])):
            new_df.loc[:,array_cols['name'][array_col]] = convert_array_type(cols[array_cols['name'][array_col]], array_cols['type'][array_col]).copy()
   
        other_cols = [col for col in cols.columns if col not in date_cols and 
                    col not in cat_cols and col not in num_cols and col not in array_cols['name']]
    else:
        other_cols = [col for col in cols.columns if col not in date_cols and 
                    col not in cat_cols and col not in num_cols]
    new_df.loc[:,other_cols] = cols.loc[:,other_cols].copy()
    
    # print(new_df)
    return new_df

def get_histology_grouping_lung(morpho_short: pd.Series):
    """
    Function to get the histology grouping for lung cancer.
    """
    histo_groups = []
    for code in morpho_short:
        if pd.isna(code):
            histo_groups.append(np.nan)
        else:
            code = int(code)
            if code in range(8010, 8577):
                if code in [*range(8050, 8079), *range(8083,8085)]:
                    histo_groups.append('Squamous cell carcinoma')
                elif code in [8140, 8211, 8230, 8231, *range(8250, 8261), 8323, *range(8480, 8491), 
                            *range(8550, 8553), *range(8570, 8575), 8576]:
                    histo_groups.append('Adenocarcinoma')
                elif code in range(8041,8046):
                    histo_groups.append('Small cell carcinoma')
                elif code in [*range(8010, 8013), *range(8014, 8032), 8035, 8310]:
                    histo_groups.append('Large cell carcinoma')
                else:
                    histo_groups.append('Other carcinoma')
            elif code in [*range(8800, 8812), 8830, *range(8840, 8922), 8990, 8991, *range(9040, 9045),
                        *range(9120, 9134), 9150, *range(9540, 9582)]:
                histo_groups.append('Sarcoma')
            elif code in range(8000, 8005):
                histo_groups.append('Unspecified neoplasm')
            else:
                histo_groups.append('Other neoplasm')
    return histo_groups

def get_histology_grouping_breast(morpho_short: pd.Series):
    
    histo_groups = []
    for code in morpho_short:
        if pd.isna(code):
            histo_groups.append(np.nan)
        else:

            code = int(code)
            if code in range(8500, 8544):
                histo_groups.append('Ductal/lobular carcinoma')
            elif code in range(8140, 8385):
                histo_groups.append('Adenoma or adenocarcinoma')
            elif code in [*range(8800, 8812), 8830, *range(8840, 8922), 8990, 8991, *range(9040, 9045),
                        *range(9120, 9134), 9150, *range(9540, 9582)]:
                histo_groups.append('Sarcoma')
            else:
                histo_groups.append('Other neoplasm')
    return histo_groups

def get_histology_grouping_thyroid(morpho_short: pd.Series):
    histo_groups = []
    for code in morpho_short:
        if pd.isna(code):
            histo_groups.append(np.nan)
        else:
            code = int(code)
            if code in range(8010, 8577):
                if code in [8290, *range(8330, 8336)]:
                    histo_groups.append('Follicular carcinoma')
                elif code in [8050, 8260, 8350, *range(8340, 8345), *range(8450, 8461)]:
                    histo_groups.append('Papillary carcinoma')
                elif code in [8345, *range(8510, 8514)]: 
                    histo_groups.append('Medullary carcinoma')
                elif code in [8020, 8035]:
                    histo_groups.append('Anaplastic carcinoma')
                else:
                    histo_groups.append('Other carcinoma')
            elif code in [*range(8800, 8812), 8830, *range(8840, 8922), 8990, 8991, *range(9040, 9045),
                        *range(9120, 9134), 9150, *range(9540, 9582)]:
                histo_groups.append('Sarcoma')
            elif code in range(8000, 8005):
                histo_groups.append('Unspecified neoplasm')
            else:
                histo_groups.append('Other neoplasm')
    return histo_groups

def get_histology_grouping_non_hodgkin_lymphoma(morpho_short: pd.Series):
    histo_groups = []
    for code in morpho_short:
        if pd.isna(code):
            histo_groups.append(np.nan)
        else:
            code = int(code)
            if code in [9597, *range(9670, 9700), 9712, 9728, *range(9731, 9739),
                        *range(9761, 9765), *range(9811, 9819), *range(9823, 9827), 
                        9833, 9836, 9940]:
                histo_groups.append('B-cell lymphoma')
            elif code in [*range(9700, 9727), 9729, *range(9827, 9832), 9834, 9837, 9948]:
                histo_groups.append('T/NK-cell lymphoma')
            elif code in [9840, *range(9860, 9932), 9945, 9946, 9950, *range(9960, 9965),
                        9975, *range(9980, 9990), *range(9991, 9993)]:
                histo_groups.append('Myeloid neoplasm')
            elif code == 9591:
                histo_groups.append('Non-Hodgkin lymphoma NOS')
            else:
                histo_groups.append('Other neoplasm')
    return histo_groups


def get_morpho_short(morpho_codes: pd.Series):
    """
    Function to get the short morphology codes from the long morphology codes.

    Parameters:
        morpho_codes (pd.Series): Series of long morphology codes.
    """
    morpho_short = []
    for code in morpho_codes:
        if not pd.isna(code):
            morpho_short.append(code.split('/')[0])
        else:
            morpho_short.append(pd.NA)
    return morpho_short

def get_histology_grouping(morpho_short: pd.Series, tumor_entity: str):

    if tumor_entity == "lung":
        histo_groups = get_histology_grouping_lung(morpho_short)
    elif tumor_entity == "breast":
        histo_groups = get_histology_grouping_breast(morpho_short)
    elif tumor_entity == "thyroid":
        histo_groups = get_histology_grouping_thyroid(morpho_short)
    elif tumor_entity == "non_hodgkin_lymphoma":
        histo_groups = get_histology_grouping_non_hodgkin_lymphoma(morpho_short)
    return histo_groups


def get_tumor_icds(tumor_entity:str):
    """
    Returns fitting icds for tumor entity.

    Parameters: 
        tumor_entity (str): Tumor entity to filter the data for
    """

    icds = []
    if tumor_entity != None:
        if tumor_entity == "lung":
            icds = ["C34.0", "C34.1", "C34.2", "C34.3", "C34.8", "C34.9", "C34"]
        elif tumor_entity == "breast":
            icds = ["C50.0", "C50.1", "C50.2", "C50.3", "C50.4", "C50.5", "C50.6", "C50.8", "C50.9", "C50"]
        elif tumor_entity == "thyroid":
            icds = ["C73"]
        elif tumor_entity == "non_hodgkin_lymphoma":
            icds = ["C82","C82.0","C82.1","C82.2","C82.3","C82.4","C82.5","C82.6","C82.7","C82.9",
                    "C83","C83.0","C83.1","C83.3","C83.5","C83.7","C83.8","C83.9",
                    "C84","C84.0","C84.1","C84.4",
                    "C84.5","C84.6","C84.7","C84.8","C84.9","C85","C85.1","C85.2","C85.7","C85.9",
                    "C86","C86.0","C86.1","C86.2","C86.3","C86.4","C86.5","C86.6",
                    "C88","C88.0","C88.2","C88.4","C88.7"]
        elif tumor_entity == "all":
            icds = ["C34.0", "C34.1", "C34.2", "C34.3", "C34.8", "C34.9", "C34",
                    "C50.0", "C50.1", "C50.2", "C50.3", "C50.4", "C50.5", "C50.6", "C50.8", "C50.9", "C50",
                    "C73",
                    "C82","C82.0","C82.1","C82.2","C82.3","C82.4","C82.5","C82.6","C82.7","C82.9",
                    "C83","C83.0","C83.1","C83.3","C83.5","C83.7","C83.8","C83.9",
                    "C84","C84.0","C84.1","C84.4",
                    "C84.5","C84.6","C84.7","C84.8","C84.9","C85","C85.1","C85.2","C85.7","C85.9",
                    "C86","C86.0","C86.1","C86.2","C86.3","C86.4","C86.5","C86.6",
                    "C88","C88.0","C88.2","C88.4","C88.7"]
        else:
            warnings.warn('Warning Message: Your input of tumor entity does not match the valid options: lung, breast, thyroid, non_hodgkin_lymphoma.')

    return icds

def gather_tnm_values(row: pd.Series, tumor_entity):
    """
    Gather tnm values, pTNM is preferred.
    If pTNM is not available, then cTNM is chosen.
    If none of them is available, then NaN is chosen.

    Parameters:
        row (pd.Series): Current row of Dataframe.
    """

    row = row.copy(deep=True)
    # row = row.replace(to_replace=pd.NA, value='NA')
    # row = row.replace(to_replace='', value='NA')

    # if tumor_entity == 'lung' and row['pTNM_T'] == 'X': 
    #     row = row.replace(to_replace='X', value='NA')
    #     row['pTNM_T'] = 'X'
    # elif tumor_entity == 'lung' and row['cTNM_T']== 'X':
    #     row = row.replace(to_replace='X', value='NA')
    #     row['cTNM_T'] = 'X'
    # else:
    #     row = row.replace(to_replace='X', value='NA')

    # print(row['pTNM_T'], row['pTNM_N'], row['pTNM_M'])
    # print(row['cTNM_T'], row['cTNM_N'], row['cTNM_M'])

    if (row["pTNM_y"] == "y") or \
       ((pd.isna(row["pTNM_T"]) or row["pTNM_T"] == 'X') and (pd.isna(row["pTNM_N"]) or row["pTNM_N"] == 'X') and (pd.isna(row["pTNM_M"]) or row["pTNM_M"] == 'X' or row["pTNM_M"] == '0')):
        return row['cTNM_T'], row["cTNM_N"], row["cTNM_M"], row["cTNM_Version"]
    else:
        return row['pTNM_T'], row['pTNM_N'], row['pTNM_M'], row['pTNM_Version']

    # if ("NA" in row['pTNM_T'] or "X" in row['pTNM_T']) and ("NA" in row['pTNM_N']) and ("NA" in row['pTNM_M']):
    #     return row['cTNM_T'], row["cTNM_N"], row["cTNM_M"], row["cTNM_Version"]
    # else:
    #     return row['pTNM_T'], row['pTNM_N'], row['pTNM_M'], row['pTNM_Version']

def determine_uicc_stage(tumor_df: pd.Series, tumor_entity: str, patient_df: pd.Series):
    """
    Determines the uicc stage based on the TNM classification.
    If both p_TNM and c_TNM is available, then p_TNM is preferred.

    Parameters:
        tumor_df (pd.Series): Filtered dataframe for chosen tumor entity.
        tumor_entity (str): Chosen tumor entity.
    """
    if tumor_entity == "non_hodgkin_lymphoma":
        tumor_df["TNM_T"] = ""
        tumor_df["TNM_N"] = ""
        tumor_df["TNM_M"] = ""
        tumor_df["UICC"] = ""
        tumor_df["Ann_Arbor"] = tumor_df.apply(extract_ann_arbor_stage, axis=1)
        return tumor_df
    tnm_m, tnm_n, tnm_t, versions = [], [], [], []
    uicc = []

    for row in tqdm(range(tumor_df.shape[0]), desc="Generating tnm values and uicc stage..."):

        # print(tumor_df.iloc[row]['pTNM_T'], tumor_df.iloc[row]['pTNM_N'], tumor_df.iloc[row]['pTNM_M'])
        # print(tumor_df.iloc[row]['cTNM_T'], tumor_df.iloc[row]['cTNM_N'], tumor_df.iloc[row]['cTNM_M'])

        t, n, m, version = gather_tnm_values(row=tumor_df.iloc[row, :], tumor_entity=tumor_entity)
        if m == 'x' or m == 'X':
            m = '0'
        tnm_t.append(t)
        tnm_m.append(m)
        tnm_n.append(n)

        versions.append(version)

        if tumor_entity == 'breast':
            uicc.append(determine_uicc_breast(tnm_t[row], tnm_n[row], tnm_m[row], version=versions[row]))
        elif tumor_entity == 'lung':
            uicc.append(determine_uicc_lung(tnm_t[row], tnm_n[row], tnm_m[row], version=versions[row]))
        elif tumor_entity == 'thyroid':
            uicc.append(determine_uicc_thyroid(tnm_t[row], tnm_n[row], tnm_m[row],
                                               icd = tumor_df.iloc[row]['Primaertumor_Morphologie_ICD_O'], 
                                               age=tumor_df.iloc[row]['Alter_bei_Diagnose'],
                                               version = versions[row])
                                               )
        #print(uicc[-1])
    # append filtered TNM categories
    for tnm_str, tnm_col in ("TNM_T",tnm_t), ("TNM_N",tnm_n), ("TNM_M",tnm_m), ("TNM_Version",versions):
        tumor_df.loc[:, tnm_str] = tnm_col

        # tnm_ser.replace("NA", pd.NA, inplace=True)
       
        tumor_df[tnm_str] = convert_single_col(tumor_df.loc[:, tnm_str], target=pd.CategoricalDtype)
        # print(tumor_df.loc[:, tnm_str].unique())

    tumor_df["UICC"] = uicc
    tumor_df["UICC"] = convert_single_col(tumor_df.loc[:, "UICC"], target=pd.CategoricalDtype)

    tumor_df["Ann_Arbor"] = ""
    # print(tumor_df.loc[:, "UICC"].unique())
    return tumor_df

# Function to extract Ann-Arbor or ANN-ARBOR-STADIUM stage
def extract_ann_arbor_stage(row):
    if pd.isna(row["Weitere_Klassifikation_Name"]) or pd.isna(row["Weitere_Klassifikation_Stadium"]):
        return pd.NA
    # Split names and stages
    names = row["Weitere_Klassifikation_Name"].split(";")
    stages = row["Weitere_Klassifikation_Stadium"].split(";")
    
    # Pair names with stages
    classification_stages = dict(zip(names, stages))
    ann_arbor_pattern = re.compile(r"ann[\s\-_]?arbor[\s\-_]?stadium?|ann[\s\-_]?arbor", re.IGNORECASE)
    
    # Find the stage for matching classifications
    for name, stage in classification_stages.items():
        if ann_arbor_pattern.search(name):  # Check if the name matches the regex
            return stage
    return pd.NA
    # # Find the stage for Ann-Arbor or ANN-ARBOR-STADIUM
    # for name, stage in classification_stages.items():
    #     if name.lower() in ["ann-arbor", "ann-arbor-stadium", "ann arbor", "ann_arbor"]:
    #         return stage
    # return None

def determine_uicc_thyroid(tnm_t, tnm_n, tnm_m, icd, age, version = '8'):
    """
    Determines uicc stages for breast cancer for each patient.

    Parameters:
    tnm_t: Classification of tumour.
    tnm_m: Missing or occurency of metastasis.
    tnm_n: Missing or occurency of metastasis in lymph nodes.
    icd: ICD of the main tumour, to determine, whether tumour is pappilary, follicular, medullary or anaplastic.
    geb_dat: Birth date of patient.
    diag_dat: Diagnosis date of patient. Both dates serves to compute the age of the patient.
    version: Version of the UICC classification.
    """

    cancer_type = ""

    if pd.isna(icd):
        return pd.NA
    else:
        # split /3 of string
        icd = icd.split('/')[0]
    
    if icd in ['8290', '8330', '8331', '8332', '8333', '8334', '8335', '8337', '8339']:
        cancer_type = 'follicular'
    elif (icd in ['8050', '8260', '8350']) or (icd in [str(x) for x in list(range(8340, 8345))]) or (icd in [str(x) for x in list(range(8450, 8461))]): 
        cancer_type = 'papillary'
    elif (icd in ['8345', '8346']) or (icd in [str(x) for x in list(range(8510, 8514))]):
        cancer_type = 'medullary'
    elif icd in [str(x) for x in list(range(8020, 8036))]:
        cancer_type = 'anaplastic'
    else:
        #warnings.warn('No cancer type could be detected.')
        return pd.NA
    if version == '7':
        if (cancer_type in ['follicular', 'papillary']) and (age < 45):
            if '0' == tnm_m:
                return 'I' 
            elif '1' == tnm_m:
                return 'II'
            else:
                return pd.NA
        elif (cancer_type in ['follicular', 'papillary']) and (age >= 45):
            if ((tnm_t in ['1A', '1B']) and ('0' == tnm_n) and ('0' == tnm_m)):
                return 'I'
            elif (tnm_t == '2') and ('0' == tnm_n) and ('0' == tnm_m):
                return 'II'
            elif ((('3' == tnm_t) and ('0' == tnm_n) and ('0' == tnm_m)) or ((tnm_t in ['1', '2', '3']) and ('1a' == tnm_n) and ('0' == tnm_m))):
                return 'III'
            elif (('4A' == tnm_t) and ('0' == tnm_m)) or ((tnm_t in ['1', '2', '3']) and ('1b' == tnm_n) and ('0' == tnm_m)):
                return 'IVA'
            elif (('4B' == tnm_t) and ('0' == tnm_m)):
                return 'IVB'
            elif '1' == tnm_m:
                return 'IVC'
            else:
                return pd.NA
            
        elif cancer_type == 'medullary':
            if ((tnm_t in ['1A', '1B']) and ('0' == tnm_n) and ('0' == tnm_m)):
                return 'I'
            elif ((tnm_t in ['2', '3']) and ('0' == tnm_n) and ('0' == tnm_m)):
                return 'II'
            elif ((tnm_t in ['1', '2', '3']) and ('1A' == tnm_n) and ('0' == tnm_m)):
                return 'III'
            elif (((tnm_t in ['1', '2', '3']) and ('1B' == tnm_n) and ('0' == tnm_m)) or \
                (('4A' == tnm_t) and ('0' == tnm_m))):
                return 'IVA'
            elif (('4B' == tnm_t) and ('0' == tnm_m)):
                return 'IVB'
            elif '1' == tnm_m:
                return 'IVC'
            else:
                return pd.NA
        elif cancer_type == 'anaplastic':
            if (('4A' == tnm_t) and ('0' == tnm_m)):
                return 'IVA'
            elif (('4B' == tnm_t) and ('0' == tnm_m)):
                return 'IVB'
            
            elif '1' == tnm_m:
                return 'IVC'
            else:
                return pd.NA
        else:
                return pd.NA
            
    elif version == '8':
        if (cancer_type in ['follicular', 'papillary']) and (age < 55):
            if '0' == tnm_m:
                return 'I' 
            elif '1' == tnm_m:
                return 'II'
            else:
                return pd.NA
        elif (cancer_type in ['follicular', 'papillary']) and (age >= 55):
            if ((tnm_t in ['1A', '1B', '2']) and ('0' == tnm_n) and ('0' == tnm_m)):
                return 'I'
            elif ((('3' == tnm_t) and ('0' == tnm_n) and ('0' == tnm_m)) or ((tnm_t in ['1', '2', '3']) and ('1' == tnm_n) and ('0' == tnm_m))):
                return 'II'
            elif (('4A' == tnm_t) and ('0' == tnm_m)):
                return 'III'
            elif (('4B' == tnm_t) and ('0' == tnm_m)):
                return 'IVA'
            elif '1' == tnm_m:
                return 'IVB'
            else:
                return pd.NA
        elif cancer_type == 'medullary':
            if ((tnm_t in ['1A', '1B']) and ('0' == tnm_n) and ('0' == tnm_m)):
                return 'I'
            elif ((tnm_t in ['2', '3']) and ('0' == tnm_n) and ('0' == tnm_m)):
                return 'II'
            elif ((tnm_t in ['1', '2', '3']) and ('1A' == tnm_n) and ('0' == tnm_m)):
                return 'III'
            elif (((tnm_t in ['1', '2', '3']) and ('1B' == tnm_n) and ('0' == tnm_m)) or \
                (('4A' == tnm_t) and ('0' == tnm_m))):
                return 'IVA'
            elif (('4B' == tnm_t) and ('0' == tnm_m)):
                return 'IVB'
            elif '1' == tnm_m:
                return 'IVC'
            else:
                return pd.NA
        elif cancer_type == 'anaplastic':
            if ((tnm_t in ['1', '2', '3A']) and ('0' == tnm_n) and ('0' == tnm_m)):
                return 'IVA'
            elif ((tnm_t in ['1', '2', '3A']) and ('1' == tnm_n) and ('0' == tnm_m)):
                return 'IVB'
            elif ((tnm_t in ['3B', '4A', '4B']) and (tnm_n in ['0', '1']) and ('0' == tnm_m)):
                return 'IVB'
            elif '1' == tnm_m:
                return 'IVC'
            else:
                return pd.NA
        else:
                return pd.NA
    else:
        return pd.NA

def determine_uicc_breast(tnm_t, tnm_n, tnm_m, version = '8'):
    """
    Determines uicc stages for breast cancer for each patient.

    Parameters:
    tnm_t: Classification of tumour.
    tnm_m: Missing or occurency of metastasis.
    tnm_n: Missing or occurency of metastasis in lymph nodes.
    version: Version of the UICC classification.
    """
    if version == '7':
        if ("0" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "0"
        elif (tnm_t in ['1', '1mi']) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IA"
        elif (tnm_t in ['0', '1', '1mi']) and ("1mi" == tnm_n) and ("0" == tnm_m):
            return "IB"
        elif ((tnm_t in ['0', '1', '1mi']) and ("1" == tnm_n) and ("0" == tnm_m)) or \
             (("2" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m)):
            return "IIA"
        elif (("2" == tnm_t) and ("1" == tnm_n) and ("0" == tnm_m)) or \
             (("3" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m)):
            return "IIB"
        elif ((tnm_t in ["0", "1", "1mi", "2"]) and ("2" == tnm_n) and ("0" == tnm_m) or \
                (("3" == tnm_t) and (tnm_n in ["1", "2"]) and ("0" == tnm_m))):
            return "IIIA"
        elif (("4" == tnm_t) and (tnm_n in ["0", "1", "2"]) and ("0" == tnm_m)):
            return "IIIB"
        elif (("3" == tnm_n) and ("0" == tnm_m)):
            return "IIIC"
        elif ("1" == tnm_m):
            return "IV"
        else:
            return pd.NA
        
    elif version == '8':
        # determine uicc stage
        if ("IS" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "0"
        elif ("1" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IA"
        elif (tnm_t in ["0", "1"]) and ("1MI" == tnm_n) and ("0" == tnm_m):
            return "IB"
        elif ((tnm_t in ["0", "1"]) and ("1" == tnm_n) and ("0" == tnm_m)) \
                or (("2" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m)):
            return "IIA"
        elif (("2" == tnm_t) and ("1" == tnm_n) and ("0" == tnm_m)) or \
                (("3" == tnm_t) and ("0" == tnm_n) and "0" == tnm_m):
            return "IIB"
        elif ((tnm_t in ["0", "1", "2"]) and ("2" == tnm_n) and ("0" == tnm_m) or \
                (("3" == tnm_t) and (tnm_n in ["1", "2"]) and ("0" == tnm_m))):
            return "IIIA"
        elif (("4" == tnm_t) and (tnm_n in ["0", "1", "2"]) and ("0" == tnm_m)):
            return "IIIB"
        elif (("3" == tnm_n) and ("0" == tnm_m)):
            return "IIIC"
        elif ("1" == tnm_m):
            return "IV"
        # y Fälle, Primärtumor bei OP nicht mehr vorhanden, weil vorher erfolgreich therapiert --> Ausschließen von UICC
        elif (("0" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m)):
            return pd.NA
        else:
            return pd.NA
    else:
        return pd.NA
    
def determine_uicc_lung(tnm_t, tnm_n, tnm_m, version = '8'):
    """
    Determines uicc stages for lung cancer for each patient.

    Parameters:
    tnm_t: Classification of tumour.
    tnm_m: Missing or occurency of metastasis.
    tnm_n: Missing or occurency of metastasis in lymph nodes.
    version: Version of the UICC classification.
    """
    # determine uicc stage
    if version == '7':
        if ("X" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "OKK"
        elif ("IS" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "0"
        elif ("1" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IA"
        elif (tnm_t in ["1", "1A", "1B"]) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IA"
        elif ("2A" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IB"
        elif (("2B" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m)) or \
             ((tnm_t in ["1", "1A", "1B", "2A"]) and ("1" == tnm_n) and ("0" == tnm_m)):
            return "IIA"
        elif (((tnm_t == "2B") and ("1" == tnm_n) and ("0" == tnm_m)) or \
             (("3" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m))):
            return "IIB"
        elif ((tnm_t in ["1", "1A", "1B", "2", "2A", "2B"]) and ("2" == tnm_n) and ("0" == tnm_m)) or \
                (("3" == tnm_t) and (tnm_n in ["1", "2"]) and ("0" == tnm_m)) or \
                (("4" == tnm_t) and (tnm_n in ["0", "1"]) and ("0" == tnm_m)):
            return "IIIA"
        elif ((("3" == tnm_n) and ("0" == tnm_m)) or \
             (("4" == tnm_t) and ("2" == tnm_n) and ("0" == tnm_m))):
            return "IIIB"

        elif (tnm_m in ["1", "1A", "1B"]):
            return "IV"

        # y Fälle, Primärtumor bei OP nicht mehr vorhanden, weil vorher erfolgreich therapiert --> Ausschließen von UICC
        elif (("0" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m)):
            return pd.NA
        else:
            return pd.NA
    elif version == '8':
        if ("X" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "OKK"
        elif ("IS" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "0"
        elif ("1" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IA"
        elif (tnm_t in ["1A", "1MI"]) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IA1"
        elif ("1B" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IA2"
        elif ("1C" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IA3"
        elif ("2A" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IB"
        elif ("2B" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m):
            return "IIA"
        elif (((tnm_t in ["1", "1A", "1B", "1C", "2", "2A", "2B"]) and ("1" == tnm_n) and ("0" == tnm_m)) or \
                (("3" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m))):
            return "IIB"
        elif ((tnm_t in ["1", "1A", "1B", "1C", "2", "2A", "2B"]) and ("2" == tnm_n) and ("0" == tnm_m)) or \
                (("3" == tnm_t) and ("1" == tnm_n) and ("0" == tnm_m)) or \
                (("4" == tnm_t) and (tnm_n in ["0", "1"]) and ("0" == tnm_m)):
            return "IIIA"
        elif (((tnm_t in ["1", "1A", "1B", "1C", "2", "2A", "2B"]) and ("3" == tnm_n) and ("0" == tnm_m)) or \
            ((tnm_t in ["3", "4"]) and ("2" == tnm_n) and ("0" == tnm_m))):
            return "IIIB"
        elif ((tnm_t in ["3", "4"]) and ("3" == tnm_n) and ("0" == tnm_m)):
            return "IIIC"
        elif ("1" == tnm_m):
            return "IV"
        elif ("1A" == tnm_m or "1B" == tnm_m):
            return "IVA"
        elif ("1C" == tnm_m):
            return "IVB"
        # y Fälle, Primärtumor bei OP nicht mehr vorhanden, weil vorher erfolgreich therapiert --> Ausschließen von UICC
        elif (("0" == tnm_t) and ("0" == tnm_n) and ("0" == tnm_m)):
            return pd.NA
        else:
            return pd.NA
    else:
        return pd.NA



def import_aicare(path:str, tumor_entity:str, registry:Union[str, list]=None):
    '''
    Imports data from csv files and returns a dictionary containing the dataframes.

    Parameters:
        path (str): Path to the data folder
        tumor_entity (str): Tumor entity to filter the data for
        registry (str or list): RegistryID to filter the data for (according to https://plattform65c.atlassian.net/wiki/spaces/P6/pages/2228301/Lieferregister+Typ)
    '''

    dataset_dict = {}
    for table in ["patient", "tumor", "strahlentherapie", "systemtherapie", "op", "modul_mamma", "fe"]:
        if os.path.isfile(f"{path}{table}.csv"):
            csv = pd.read_csv(f"{path}{table}.csv", header=0, sep=",",quotechar='"',dtype=pd.StringDtype())
            #check if there is any column besides index and 'V1'
            if len(csv.columns) > 2:
                # strip blank space from column names and rows, also transform NA's
                csv.columns = csv.columns.str.strip()
                for col in csv:
                    csv[col] = csv[col].str.strip().replace([np.nan, 'NA', ''], pd.NA)
                if registry != None and registry != "all":
                    if type(registry) == str:
                        csv = csv[csv["Register_ID_FK"] == registry]
                    elif type(registry) == list:
                        csv = csv[csv["Register_ID_FK"].isin(registry)]
                dataset_dict[table] = csv
        else:
            print(f"No {table}.csv in {path}!")

    # get valid icd for tumor entity
    icds = get_tumor_icds(tumor_entity=tumor_entity)
    
    # filter pandas series for specific tumor entities
    dataset_dict["tumor"]["Patient_ID_unique"] = dataset_dict["tumor"]["Patient_ID_FK"].astype(str) + "_" + dataset_dict["tumor"]["Register_ID_FK"].astype(str)
    dataset_dict["tumor"] = dataset_dict["tumor"][dataset_dict["tumor"]["Primaertumor_ICD"].isin(icds)]
    patid_subset = dataset_dict["tumor"]["Patient_ID_unique"]
    dataset_dict["patient"]["Patient_ID_unique"] = dataset_dict["patient"]["Patient_ID"].astype(str) + "_" + dataset_dict["patient"]["Register_ID_FK"].astype(str)
    dataset_dict["patient"] = dataset_dict["patient"][dataset_dict["patient"]["Patient_ID_unique"].isin(patid_subset)]
    tumor_id = dataset_dict["tumor"]["Tumor_ID"]
    for table in ["strahlentherapie", "systemtherapie", "op", "modul_mamma", "fe"]:
        if table in dataset_dict:
            dataset_dict[table]["Patient_ID_unique"] = dataset_dict[table]["Patient_ID_FK"].astype(str) + "_" + dataset_dict[table]["Register_ID_FK"].astype(str)
            
            dataset_dict[table] = dataset_dict[table][dataset_dict[table]["Tumor_ID_FK"].isin(tumor_id)]

    for i in dataset_dict.keys():
        if i == 'patient':
            date_cols = ["Geburtsdatum","Datum_Vitalstatus"]
            cat_cols = ["Geschlecht","Verstorben","Todesursache_Grundleiden","Todesursache_Grundleiden_Version"]
            array_cols = [] #{'name': ["Weitere_Todesursachen", "Weitere_Todesursachen_Version"], 'type': [pd.StringDtype(), pd.StringDtype()]}
            # convert datatypes in dataframe patient
            dataset_dict[i] = convert_all_cols(dataset_dict[i], date_cols=date_cols, cat_cols=cat_cols, array_cols=array_cols)
            dataset_dict[i]["Verstorben"] = dataset_dict[i]["Verstorben"].map({'N': 0, 'J': 1})
            print("Patient Data complete!")
            
        elif i == 'tumor':
            date_cols = ["Diagnosedatum"]
            cat_cols = ["Inzidenzort","Diagnosesicherung","Primaertumor_ICD","Primaertumor_ICD_Version","Primaertumor_Topographie_ICD_O","Primaertumor_Topographie_ICD_O_Version",
                        "Primaertumor_Morphologie_ICD_O","Primaertumor_Morphologie_ICD_O_Version","Primaertumor_Grading","cTNM_Version",
                        "cTNM_y","cTNM_r","cTNM_a","cTNM_praefix_T","cTNM_T","cTNM_praefix_N","cTNM_N","cTNM_praefix_M","cTNM_M","c_m_Symbol","c_L.Kategorie","c_V.Kategorie","c_Pn.Kategorie",
                        "c_S.Kategorie","cTNM_UICC_Stadium","Seitenlokalisation","pTNM_Version","pTNM_y","pTNM_r","pTNM_a","pTNM_praefix_T","pTNM_T",
                        "pTNM_praefix_N","pTNM_N","pTNM_praefix_M","pTNM_M","p_m_Symbol","p_L.Kategorie","p_V.Kategorie","p_Pn.Kategorie","p_S.Kategorie","pTNM_UICC_Stadium","Primaertumor_DCN",
                        "Primaerdiagnose_Menge_FM","Weitere_Klassifikation_UICC","Weitere_Klassifikation_Name","Weitere_Klassifikation_Stadium"]
            num_cols = ["Anzahl_Tage_Diagnose_Tod", "Primaertumor_LK_untersucht","Primaertumor_LK_befallen", "Anzahl_Monate_Diagnose_Zensierung", "Alter_bei_Diagnose"]
            # convert datatypes in dataframe tumor
            dataset_dict[i] = convert_all_cols(dataset_dict[i], date_cols=date_cols, cat_cols=cat_cols, num_cols=num_cols)
            

            # add uicc column
            dataset_dict[i] = determine_uicc_stage(tumor_df=dataset_dict[i], tumor_entity=tumor_entity, patient_df=dataset_dict['patient'])
            dataset_dict[i]["morpho_short"] = get_morpho_short(dataset_dict[i]["Primaertumor_Morphologie_ICD_O"])
            dataset_dict[i]["morpho_short"] = dataset_dict[i]["morpho_short"].astype(pd.CategoricalDtype())
            dataset_dict[i]["histo_group"] = get_histology_grouping(dataset_dict[i]["morpho_short"], tumor_entity=tumor_entity)
            dataset_dict[i]["histo_group"] = dataset_dict[i]["histo_group"].astype(pd.CategoricalDtype())
            print("Tumor Data complete!")

        elif i == 'strahlentherapie':
            date_cols = ['Beginn_Bestrahlung']
            cat_cols = ['Intention_st', 'Stellung_OP', 'Applikationsart', 'Zielgebiet_CodeVersion', 'Zielgebiet_Code', 'Seite_Zielgebiet']
            num_cols = ['Anzahl_Tage_Diagnose_ST', 'Anzahl_Tage_ST']
            array_cols = [] #{'name': ['Applikationsspezifikation'], 'type': [pd.StringDtype()]}
            dataset_dict[i] = convert_all_cols(dataset_dict[i], date_cols=date_cols, cat_cols=cat_cols, num_cols=num_cols, array_cols=array_cols)
            print("Strahlentherapie Data complete!")

        elif i == 'systemtherapie':
            date_cols = ['Beginn_SYST']
            cat_cols = ['Intention_sy', 'Stellung_OP', 'Therapieart']
            array_cols = [] #{'name': ['Substanzen', 'Protokolle'], 'type': [pd.StringDtype(), pd.StringDtype()]}
            num_cols = ['Anzahl_Tage_Diagnose_SYST', 'Anzahl_Tage_SYST']
            dataset_dict[i] = convert_all_cols(dataset_dict[i], date_cols=date_cols, cat_cols=cat_cols, num_cols=num_cols, array_cols=array_cols)
            print("Systemtherapie Data complete!")

        elif i == 'op':
            date_cols = ['Datum_OP']
            cat_cols = ['Intention', 'Beurteilung_Residualstatus']
            num_cols = ['Anzahl_Tage_Diagnose_OP']
            array_cols = [] #{'name': ['Menge_OPS_code', 'Menge_OPS_version'], 'type': [pd.StringDtype(), pd.StringDtype()]}
            dataset_dict[i] = convert_all_cols(dataset_dict[i], date_cols=date_cols, cat_cols=cat_cols, num_cols=num_cols, array_cols=array_cols)
            print("OP Data complete!")

        elif i == 'modul_mamma':
            cat_cols = ['Praetherapeutischer_Menopausenstatus', 'HormonrezeptorStatus_Oestrogen', 'HormonrezeptorStatus_Progesteron', 'Her2neuStatus']
            num_cols = ['TumorgroesseInvasiv', 'TumorgroesseDCIS']
            dataset_dict[i] = convert_all_cols(dataset_dict[i], cat_cols=cat_cols, num_cols=num_cols)
            print("Modul Data complete!")

        elif i == 'fe':
            cat_cols = ["Folgeereignis_TNM_Version","Folgeereignis_y_Symbol","Folgeereignis_r_Symbol","Folgeereignis_a_Symbol",\
                        "Folgeereignis_praefix_T","Folgeereignis_TNM_T","Folgeereignis_praefix_N","Folgeereignis_TNM_N","Folgeereignis_praefix_M",\
                        "Folgeereignis_TNM_M","Folgeereignis_m_Symbol","Folgeereignis_L_Kategorie","Folgeereignis_V_Kategorie",\
                        "Folgeereignis_Pn_Kategorie","Folgeereignis_S_Kategorie","Folgeereignis_TNM_UICC","Gesamtbeurteilung_Tumorstatus",\
                        "Verlauf_Lokaler_Tumorstatus","Verlauf_Tumorstatus_Lymphknoten", "Verlauf_Tumorstatus_Fernmetastasen"]
            num_cols = []
            date_cols = ["Datum_Folgeereignis"]
            array_cols = [] #{'name':["Folgeereignis_Menge_weitere_Klassifikationen_Name","Folgeereignis_Menge_weitere_Klassifikationen_Stadium", "Menge_FM"], 'type': [pd.StringDtype(), pd.StringDtype(), pd.StringDtype()]}

            dataset_dict[i] = convert_all_cols(dataset_dict[i], date_cols=date_cols, cat_cols=cat_cols, num_cols=num_cols, array_cols=array_cols)
            
            print("FE Data complete!")

    return dataset_dict



def import_vonko(path: str, oncotree_data: bool = False, processed_data: bool = False, extra_features: bool = False,
                 aggregate_therapies: bool = False, simplify: bool = True, histo_dichotomy: bool = True) -> Dict:
    '''
    Imports data from csv files and returns a dictionary containing the dataframes.
    
    Parameters:
        path (str): Path to the data folder
        oncotree_data (bool): If True, the oncotree data is used
        processed_data (bool): If True, the processed data is used
        extra_features (bool): If True, additional features are added to the data
        aggregate_therapies (bool): If True, therapies are aggregated
        simplify (bool): If True, the data is simplified (UICC and TNM stages are simplified)
        histo_dichotomy (bool): If True, the histology is dichotomized for SCLC and NSCLC
    '''

    vonko = {}

    # Create path to Tumoren file based on parameters
    tumoren_relative_path = "Tumoren"
    if processed_data:
        tumoren_relative_path += "_aufbereitet"
    if oncotree_data:
        tumoren_relative_path += "_oncotree"
    tumoren_relative_path += ".csv"

    # Read data from csv files
    diag_data = pd.read_csv(path + tumoren_relative_path, header=0, sep=";", dtype=pd.StringDtype())
    vonko["MET_PT"] = pd.read_csv(path + "MET_PT.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype())
    vonko["MET_VM"] = pd.read_csv(path + "MET_Verlauf.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype())
    vonko["OP"] = pd.read_csv(path + "OP.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype(), na_values=[""])
    vonko["ST"] = pd.read_csv(path + "ST.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype())
    vonko["SYS"] = pd.read_csv(path + "SY.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype())
    vonko["VM"] = pd.read_csv(path + "VM.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype())

    # Define categorical and numeric columns
    cat_cols = ['geschl', 'diag_sich', 'diag_seite', 'diag_icd', 'topo_icdo',
                'morpho_icdo', 'morpho_icdo_version', 'grading', 'tnm_r',
                'c_tnm_version', 'c_tnm_r', 'c_tnm_t', 'c_tnm_n', 'c_tnm_m', 'c_tnm_l', 'c_tnm_v', 'c_tnm_pn', 'c_uicc',
                'p_tnm_version', 'p_tnm_r', 'p_tnm_t', 'p_tnm_n', 'p_tnm_m', 'p_tnm_l', 'p_tnm_v', 'p_tnm_pn', 'p_uicc',
                'tnm_version', 'tnm_r', 'tnm_t', 'tnm_n', 'tnm_m', 'uicc']
    num_cols = ['alter', 'tod_alter', 'lk_befall', 'lk_unters', 'sentinel_befall', 'sentinel_unters', 'vit_status']

    # Add oncotree column if oncotree conversion has been used
    if oncotree_data:
        cat_cols.append('oncotree')

    # Add processed histology column if processed data has been used
    if processed_data:
        cat_cols.append('histo_gr')
        cat_cols.append("morpho_kurz")

    # Exclude Data without vital status and date
    #print(diag_data.shape , '1')

    diag_data = diag_data[~diag_data["vitdat"].isna()]
    diag_data = diag_data[~diag_data["vit_status"].isna()]

    #print(diag_data.shape, '2')
    
    # Convert date columns to datetime
    diag_data["vitdat"] = pd.to_datetime(diag_data["vitdat"], format="%m.%Y")
    diag_data["diagdat"] = pd.to_datetime(diag_data["diagdat"], format="%m.%Y")
    # Add 14 days to the vitdat and diagdat
    diag_data["vitdat"] = diag_data["vitdat"] + pd.to_timedelta(14, unit="d")
    diag_data["diagdat"] = diag_data["diagdat"] + pd.to_timedelta(14, unit="d")

    #print(diag_data.shape, '3')

    vonko["ST"]["stdat_beginn"] = pd.to_datetime(vonko["ST"]["stdat_beginn"], format="%d.%m.%Y")
    vonko["ST"]["stdat_ende"] = pd.to_datetime(vonko["ST"]["stdat_ende"], format="%d.%m.%Y")
    vonko["SYS"]["sydat_beginn"] = pd.to_datetime(vonko["SYS"]["sydat_beginn"], format="%d.%m.%Y")
    vonko["SYS"]["sydat_ende"] = pd.to_datetime(vonko["SYS"]["sydat_ende"], format="%d.%m.%Y")
    vonko["OP"]["opdat"] = pd.to_datetime(vonko["OP"]["opdat"], format="%d.%m.%Y")
    vonko["VM"]["vmdat"] = pd.to_datetime(vonko["VM"]["vmdat"], format="%d.%m.%Y")

    #print(diag_data.shape, '3.5')

    # Plausibility check
    diag_data = diag_data[diag_data["vitdat"] >= diag_data["diagdat"]]
    #print(diag_data.shape, '4')

    # Add extra features which are not in the original data
    if extra_features:

        diag_data_stmerge = diag_data.merge(vonko["ST"], left_on="tunr", right_on="tunr", how="left")
        filtered_st = diag_data_stmerge[(diag_data_stmerge["stdat_beginn"] - diag_data_stmerge["diagdat"]).dt.days < 180]
        diag_data["has_ST"] = diag_data["tunr"].isin(filtered_st["tunr"]).astype(int)
        
        therapy_intention = []
        for tunr in diag_data["tunr"]:
            detail = filtered_st[filtered_st["tunr"] == tunr]["st_int"].values
            therapy_intention.append(detail[0] if len(detail)>0 else "no")
        diag_data["ST_Intention"] = therapy_intention
        
        
        diag_data_sysmerge = diag_data.merge(vonko["SYS"], left_on="tunr", right_on="tunr", how="left")
        filtered_sys = diag_data_sysmerge[(diag_data_sysmerge["sydat_beginn"] - diag_data_sysmerge["diagdat"]).dt.days
                                          < 180]
        diag_data["has_SYS"] = diag_data["tunr"].isin(filtered_sys["tunr"]).astype(int)
        therapy_detail = []
        therapy_intention = []
        for tunr in diag_data["tunr"]:
            sys_pat = filtered_sys[filtered_sys["tunr"] == tunr]
            
            therapy_intention.append(sys_pat["sy_int"].values[0] if len(sys_pat["sy_int"].values)>0 else "no")
            therapy_detail.append(sys_pat["sy_art"].values[0] if len(sys_pat["sy_art"].values)>0 else "no")
        
        diag_data["SY_Intention"] = therapy_intention
        diag_data["SY_Art"] = therapy_detail
        

        diag_data_opmerge = diag_data.merge(vonko["OP"], left_on="tunr", right_on="tunr", how="left")
        filtered_op = diag_data_opmerge[(diag_data_opmerge["opdat"] - diag_data_opmerge["diagdat"]).dt.days < 180]
        diag_data["has_OP"] = diag_data["tunr"].isin(filtered_op["tunr"]).astype(int)
        therapy_detail = []
        therapy_intention = []
        for tunr in diag_data["tunr"]:
            op_pat = filtered_op[filtered_op["tunr"] == tunr]
            
            therapy_intention.append(op_pat["op_int"].values[0] if len(op_pat["op_int"].values)>0 else "no")
            therapy_detail.append(op_pat["op_ops"].values[0] if len(op_pat["op_ops"].values)>0 else "no")
        diag_data["OP_Intention"] = therapy_intention
        diag_data["OP_OPS"] = therapy_detail
        # diag_data['has_ST'] = diag_data['tunr'].isin(vonko['ST']['tunr']).astype(np.int8())
        # diag_data['has_SYS'] = diag_data['tunr'].isin(vonko['SYS']['tunr']).astype(np.int8())
        # diag_data['has_OP'] = diag_data['tunr'].isin(vonko['OP']['tunr']).astype(np.int8())
        cat_cols.extend(['has_ST', 'has_SYS', 'has_OP', "ST_Intention", "SY_Intention", "SY_Art", 'OP_Intention', 'OP_OPS'])

    # Aggregate data if start date is the same
    if aggregate_therapies:
        vonko["ST"] = vonko["ST"].groupby(["tunr", "stdat_beginn"], as_index=False).agg(lambda x: aggregation_function(x))
        vonko["SYS"] = vonko["SYS"].groupby(["tunr", "sydat_beginn"], as_index=False).agg(lambda x: aggregation_function(x))

    if simplify:
        # Convert categoricals to categorical data type
        # Remove trailing characters from uicc
        
        diag_data["uicc"] = diag_data["uicc"].str.split("[^IV]", regex=True).str[0].str.strip().replace(r'^\s*$', pd.NA, regex=True)

        for tnm_col in ["tnm_t", "tnm_n", "tnm_m"]:
            diag_data[tnm_col] = diag_data[tnm_col].str.split("[^0-4]", regex=True).str[0].str.strip().replace(r'^\s*$', pd.NA, regex=True)

    if histo_dichotomy:

        def lambda_histo_dichtomy(x):
            if x == '3':
                return 0
            elif x in ['1', '2', '4', '5', '6', '7']:
                return 1
            else:
                return 2
        histo_subset = diag_data["histo_gr"].apply(lambda_histo_dichtomy)
        diag_data.__setitem__("histo_gr", histo_subset.values)

        # exclude patients with sarcoma
        diag_data = diag_data[diag_data["histo_gr"] != 2]
    else:
       def lambda_histo_encoding(x):
           return int(x)
       diag_data["histo_gr"] = diag_data["histo_gr"].apply(lambda_histo_encoding)


    for cat_col in cat_cols:
        if diag_data[cat_col].dtype == pd.StringDtype():
            diag_data[cat_col] = diag_data[cat_col].str.upper().str.strip()
        diag_data[cat_col] = diag_data[cat_col].astype(pd.CategoricalDtype(ordered=True))

    # Convert numerics to numeric data type
    for num_col in num_cols:
        diag_data[num_col] = pd.to_numeric(diag_data[num_col])

    # Exclude patients with age > 90
    # diag_data = diag_data[diag_data["alter"] < 90]

    # Convert zustand to unified format
    karnofsky_dict = dict.fromkeys(['100%', '90%'], '0')
    karnofsky_dict.update(dict.fromkeys(['80%', '70%'], '1'))
    karnofsky_dict.update(dict.fromkeys(['60%', '50%'], '2'))
    karnofsky_dict.update(dict.fromkeys(['40%', '30%'], '3'))
    karnofsky_dict.update(dict.fromkeys(['20%', '10%'], '4'))
    karnofsky_dict.update(dict.fromkeys(['U', pd.NA], np.nan))
    diag_data['zustand'] = [karnofsky_dict.get(zustand, zustand) for zustand in diag_data['zustand']]
    diag_data['zustand'] = diag_data['zustand'].astype(pd.CategoricalDtype(ordered=True))
    # diag_data['zustand'] = diag_data['zustand'].astype(np.int8())
    vonko["Tumoren"] = diag_data

    return vonko
if has_ydata:
    def generate_report(data: pd.DataFrame, name: str, config_file: str):
        profile = ProfileReport(data, title=name+" Report", explorative=True, config_file=config_file)
        profile.to_file(name+"_report.html")


if __name__ == "__main__":
    #data = import_vonko("./data/raw/", processed_data=True, extra_features=True, aggregate_therapies=False)
    #print(data["Tumoren"].head(n=15))
    #print(data["Tumoren"]["uicc"].value_counts())
    #print(data["ST"].head(n=15))
    #print(data['Tumoren'].shape)

    # generate for every csv file a report with y profiling
    #data_list = list(data.keys())

    #for i in data_list:
    #    generate_report(data[str(i)], "./reports/"+str(i), "./minimal_config.yaml")    
    data = import_aicare("./aicare/aicare_gesamt/", "lung", registry=None)    
    print(data["tumor"].head(n=15))
    data['tumor'][['Register_ID_FK','TNM_Version', "cTNM_T", "cTNM_N", "cTNM_M", 'cTNM_UICC_Stadium',
                   "pTNM_T", "pTNM_N", "pTNM_M", 'pTNM_UICC_Stadium',
                   'TNM_T','TNM_N', 'TNM_M', 'UICC']].to_csv('df_uicc.csv', index=False)
    #data = import_aicare("./data/aicare_nds/", "thyroid")
    print(data["patient"].head(n=15))

    #data['op'].to_csv('df_op.csv', index=False)

    """
    data_list = list(data.keys())
    for i in data_list:
        generate_report(data[str(i)], "./data/reports/"+str(i)+"_nds", "datenimport_aicare/minimal_config.yaml")
    """
