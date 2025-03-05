import sys
import pandas as pd
import sklearn.impute as impute
try:
    import torch
    import torch.utils.data
except ImportError:
    _has_torch = False
else:
    _has_torch = True
from typing import List, Tuple
import numpy as np
import sklearn
import sklearn.preprocessing as preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
sys.path.append('./')
import requests
import os
from tqdm import tqdm

# sklearn.set_config(transform_output="pandas")


class Encoder():
    '''Encodes categorical variables to numerical variables'''
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.encodeTable = {}
        self.dataframe = dataframe

    def label_encoding(self, col_name: str, na_sentinel: bool = True) -> pd.DataFrame:

        if pd.api.types.is_categorical_dtype(self.dataframe[col_name]):
            self.dataframe[col_name], self.encodeTable[col_name] = pd.factorize(self.dataframe[col_name],
                                                                                sort=True,
                                                                                use_na_sentinel=na_sentinel)
            # self.dataframe[col_name] = self.dataframe[col_name].cat.codes
        
        return self.dataframe

    def one_hot_encoding(self, col_name: str) -> pd.DataFrame:
        if pd.api.types.is_categorical_dtype(self.dataframe[col_name]):
            one_hot_encoding = pd.get_dummies(self.dataframe[col_name], prefix=col_name)
            self.dataframe = pd.concat([self.dataframe.drop(columns=[col_name]), one_hot_encoding], axis=1)
        return self.dataframe
    
    
def dataframe_to_strings(df):
    """
    Converts a dataframe to a list of strings
    """
    result = []
    for _, row in df.iterrows():
        row_string = ', '.join([f"{col} ist {val}" for col, val in row.items()])
        result.append(row_string)
    return result


def encode_selected_variables(dataframe: pd.DataFrame, selected_variables: List[str], na_sentinel=True) -> Tuple[pd.DataFrame, Encoder]:
    '''Converts selected categorical variables to numerical variables'''
    encoder = Encoder(dataframe=dataframe)
    for feature in selected_variables:
        dataframe = encoder.label_encoding(feature, na_sentinel=na_sentinel)
    return dataframe, encoder



def imputation(X:pd.DataFrame, imputation_method:str, one_hot, logger, random_state, imputation_features=None, selected_features=None):
    # Impute missing values
    if imputation_features is not None:
        X = X[imputation_features].copy()
    #logger.info(f"Length before Imputation: {len(X)}")
    #logger.info(f"Imputation Method: {imputation_method}")
    
    
    if imputation_method == "none":
        pass
    elif imputation_method == "KNNImputer":
        imputer = KNNImputer(missing_values=-1, n_neighbors=1, weights="uniform")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, copy=True)
    elif imputation_method == "SimpleImputer":
        imputer = SimpleImputer(missing_values=-1, strategy="most_frequent")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, copy=True)
    elif imputation_method == "MissForest":
        # Simple MissForest emulated with a RDF combined with the Iterative Imputer
        iterative_imputer = IterativeImputer(missing_values=-1, max_iter=100,
                                        estimator= RandomForestClassifier(
                                                n_estimators=20,
                                                max_depth=10,
                                                bootstrap=True,
                                                max_samples=0.5,
                                                n_jobs=2,
                                                random_state=0,
                                        ),
                                        initial_strategy='most_frequent', random_state=random_state)
        X = pd.DataFrame(iterative_imputer.fit_transform(X), columns=X.columns, copy=True)
    else:
        raise ValueError("Imputation method not found")
    #logger.info(f"Length after Imputation: {len(X)}")
    if selected_features is not None:
        X = X[selected_features]
    if one_hot:
        if "tnm_t" in imputation_features:
            X = pd.get_dummies(X, columns=["geschl", "histo_gr", "tnm_t", "tnm_n", "tnm_m"])
        else:
            X = pd.get_dummies(X, columns=["geschl", "histo_gr", "uicc"])

    return X

def calculate_survival_time_multiregistry(dataframe: pd.DataFrame,
                                          registry_col: str = "Register_ID_FK",
                                          diag_col: str = "Diagnosedatum",
                                          vit_col: str = "Datum_Vitalstatus",
                                          deceased_col: str = "Verstorben",
                                          days_diag_to_death_col: str = "Anzahl_Tage_Diagnose_Tod",
                                          months_diag_to_censorship_col: str = "Anzahl_Monate_Diagnose_Zensierung") -> pd.DataFrame:
    """
    Calculate survival time for a dataframe with multiple registries based on 'Register_ID'.
    For registry "9", calculate survival time using alternative logic.
    For other registries, calculate the difference between 'Diagnosedatum' and 'Datum_Vitalstatus'.

    Parameters:
        dataframe (pd.DataFrame): Input dataframe with necessary columns.
        registry_col (str): Column name for registry IDs.
        diag_col (str): Column name for diagnosis date.
        vit_col (str): Column name for vital status date.
        deceased_col (str): Column name indicating if the patient is deceased.
        days_diag_to_death_col (str): Column name for days between diagnosis and death.
        months_diag_to_censorship_col (str): Column name for months between diagnosis and censorship.

    Returns:
        pd.Series: A series containing the survival time.
    """
    def calculate_row(row):
        if row[registry_col] == "9":
            if row[deceased_col] == 1:
                return row[days_diag_to_death_col]
            else:
                return round(row[months_diag_to_censorship_col] * 30.5)
        else:
            if pd.isnull(row[days_diag_to_death_col]):
                return (row[vit_col] - row[diag_col]).days
            else:
                return row[days_diag_to_death_col]

    # Apply the function row-wise
    survival_time = dataframe.apply(calculate_row, axis=1)
    return survival_time

def get_survival_time_alternative(Verstorben, Anzahl_Tage_Diagnose_Tod, Anzahl_Monate_Diagnose_Zensierung):
    """
    Survival time is the Anzahl_Tage_Diagnose_Tod if the patient is dead, otherwise it is Anzahl_Monate_Diagnose_Zensierung (special case for one registry)
    """
    if Verstorben == 1:
        return Anzahl_Tage_Diagnose_Tod
    else:
        return round(Anzahl_Monate_Diagnose_Zensierung * 30.5)


def calculate_survival_time(dataframe: pd.DataFrame, vit_col:str = "vitdat", diag_col:str = "diagdat") -> pd.DataFrame:
    """
    Calculate the survival time as time between diagnosis and last vital state
    """
    survival_time = dataframe.loc[:, vit_col] - dataframe.loc[:, diag_col]
    return survival_time.dt.components.days.values


def calculate_outcome_in_X_years(dataframe: pd.DataFrame, years: int) -> pd.DataFrame:
    """
    Calculate long time survival
    """
    dead = (dataframe["survival_time"] < 365 * years) & (dataframe["vit_status"] == 1)
    dataframe["DeadInXYears"] = dead.values
    alive = dataframe["survival_time"] >= 365 * years
    
    dataframe = dataframe[dead | alive]
    
    return dataframe

if _has_torch:
    class tumorDataset(torch.utils.data.Dataset):
        """
        Torch Dataset Wrapper.
        """
        def __init__(self, selected_features: pd.DataFrame, target: pd.DataFrame, events: pd.DataFrame = None) -> None:
            super().__init__()
            selected_features = preprocessing.MinMaxScaler().fit_transform(selected_features)
            self.selected_features = torch.tensor(selected_features, dtype=torch.float32)
            self.target = torch.tensor(np.asarray(target), dtype=torch.float32)
            if events is not None:
                self.events = torch.tensor(np.asarray(events), dtype=torch.int16)
            else:
                # If no events are given, we assume that all patients are uncensored.
                self.events = torch.ones(len(self.target), dtype=torch.int16)

        def __len__(self) -> int:
            return len(self.selected_features)

        def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
            return self.selected_features[index], self.target[index], self.events[index]


def get_morphology_groups(morpho_codes: pd.Series, basepath:str, entity:str, ontoserver_url:str):
    parentCodes = []
    code_dict = {}
    if os.path.isfile(f"{basepath}/morpho_groups/morpho_groups_{entity}.csv"):
        code_dict = pd.read_csv(f"{basepath}/morpho_groups/morpho_groups_{entity}.csv", index_col=0).to_dict(orient="index")
    
    for code in morpho_codes:
        if not pd.isna(code):
            if code in code_dict.keys():
                parentCode = code_dict[code]['parentCode']
                parentCodes.append(parentCode)
            else:
                lookup_url = f"{ontoserver_url}/CodeSystem/$lookup?system=http://terminology.hl7.org/CodeSystem/icd-o-3&code={code}"
                lookup = requests.get(lookup_url).json()
                
                if lookup['resourceType'] == 'Parameters':
                    parentCode = lookup['parameter'][5]['part'][1]['valueCode']
                    nameCode = lookup['parameter'][1]['valueString']
                    
                    parent_lookup_url = f"{ontoserver_url}/CodeSystem/$lookup?system=http://terminology.hl7.org/CodeSystem/icd-o-3&code={parentCode}"
                    parent_lookup = requests.get(parent_lookup_url).json()
                    
                    parentName = parent_lookup['parameter'][1]['valueString']
                    print(f"code: {code} name: {nameCode} parentCode: {parentCode} parentName: {parentName}")
                    
                    code_dict[code] = {'parentCode': parentCode, 'parentName': parentName, 'nameCode': nameCode}
                    parentCodes.append(parentCode)
                else:
                    code_modified = f"{str.split(code, '/')[0]}/3"
                    lookup_url = f"{ontoserver_url}/CodeSystem/$lookup?system=http://terminology.hl7.org/CodeSystem/icd-o-3&code={code_modified}"
                    lookup = requests.get(lookup_url).json()
                    
                    if lookup['resourceType'] == 'Parameters':
                        parentCode = lookup['parameter'][5]['part'][1]['valueCode']
                        nameCode = lookup['parameter'][1]['valueString']
                        
                        parent_lookup_url = f"{ontoserver_url}/CodeSystem/$lookup?system=http://terminology.hl7.org/CodeSystem/icd-o-3&code={parentCode}"
                        parent_lookup = requests.get(parent_lookup_url).json()
                        
                        parentName = parent_lookup['parameter'][1]['valueString']
                        print(f"code: {code} name: {nameCode} parentCode: {parentCode} parentName: {parentName}")
                        
                        code_dict[code] = {'parentCode': parentCode, 'parentName': parentName, 'nameCode': nameCode}
                        parentCodes.append(parentCode)
                    else:
                        parentCodes.append(pd.NA)
        else:
            parentCodes.append(pd.NA)
    
    dataframe = pd.DataFrame.from_dict(code_dict).T
    dataframe.to_csv(f"{basepath}/morpho_groups/morpho_groups_{entity}.csv")
    return parentCodes, dataframe

def create_timeline_for_patient(data, entity, registry):

    """
    function for creating timeline for each patient and tumor
    data: data dict with tumor, op, systemtherapie, strahlentherapie and patient data
    """

    time_line = pd.DataFrame(columns=["Patient_ID", "Tumor_ID", "Primaertumor_ICD", "Ereignis_Typ", "Datum", "Geburtsdatum", "Verstorben", "Todesursache", "Intention", "Beurteilung_Residualstatus", "Gesamtbeurteilung_Tumorstatus", "Anzahl_Tage_Diagnose", "Menge_OPS_Code", "Stellung_OP", "Therapieart"])

    for tumor_id in tqdm(data["tumor"]["Tumor_ID"], "Creating timeline of patients..."):

        # search for folgeereignis for patient
        patient_fe = data["fe"][data["fe"]["Tumor_ID_FK"]==tumor_id].copy()#.copy(deep=True)
        # update time_line with new patients data

        patient_data = data["patient"][data["patient"]["Patient_ID"]==data["tumor"][data["tumor"]["Tumor_ID"]==tumor_id]["Patient_ID_FK"].item()].copy()#.copy(deep=True)
        
        tumor_data = data["tumor"][data["tumor"]["Tumor_ID"]==tumor_id].copy()#.copy(deep=True)
        op = data["op"][data["op"]["Tumor_ID_FK"]==tumor_id].copy()#.copy(deep=True)
        sys = data["systemtherapie"][data["systemtherapie"]["Tumor_ID_FK"]==tumor_id].copy()#.copy(deep=True)
        strahl = data["strahlentherapie"][data["strahlentherapie"]["Tumor_ID_FK"]==tumor_id].copy()#.copy(deep=True)


        patient_time_line = pd.DataFrame(columns=["Patient_ID", "Tumor_ID", "Primaertumor_ICD", "Ereignis_Typ", "Datum", "Geburtsdatum", "Verstorben", "Todesursache", "Intention", "Beurteilung_Residualstatus", "Gesamtbeurteilung_Tumorstatus", "Anzahl_Tage_Diagnose", "Menge_OPS_Code", "Stellung_OP", "Therapieart"])
        patient_time_line["Patient_ID"] = tumor_data["Patient_ID_FK"]
        patient_time_line["Tumor_ID"] = tumor_id
        patient_time_line["Datum"] = tumor_data["Diagnosedatum"]
        patient_time_line["Primaertumor_ICD"] = tumor_data["Primaertumor_ICD"].item()
        patient_time_line["Ereignis_Typ"] = "DIAG"
        patient_time_line["Gesamtbeurteilung_Tumorstatus"] = pd.NA
        patient_time_line["Beurteilung_Residualstatus"] = pd.NA
        patient_time_line["Intention"] = pd.NA
        patient_time_line["Anzahl_Tage_Diagnose"] = pd.NA
        patient_time_line["Menge_OPS_Code"] = pd.NA
        patient_time_line["Stellung_OP"] = pd.NA
        patient_time_line["Therapieart"] = pd.NA
        patient_time_line["Geburtsdatum"] = patient_data["Geburtsdatum"].item()
        patient_time_line["Verstorben"] = "0"
        patient_time_line["Todesursache"] = pd.NA


        # append last vit date
        last_vit_date = pd.DataFrame(columns=["Patient_ID", "Tumor_ID", "Primaertumor_ICD", "Ereignis_Typ", "Datum", "Geburtsdatum", "Verstorben", "Todesursache", "Intention", "Beurteilung_Residualstatus", "Gesamtbeurteilung_Tumorstatus", "Anzahl_Tage_Diagnose", "Menge_OPS_Code", "Stellung_OP", "Therapieart"])
        last_vit_date["Patient_ID"] = patient_data["Patient_ID"]
        last_vit_date["Tumor_ID"] = tumor_id
        last_vit_date["Datum"] = patient_data["Datum_Vitalstatus"]
        last_vit_date["Primaertumor_ICD"] = tumor_data["Primaertumor_ICD"].item()
        last_vit_date["Ereignis_Typ"] = "LAST_VIT"
        last_vit_date["Gesamtbeurteilung_Tumorstatus"] = pd.NA
        last_vit_date["Beurteilung_Residualstatus"] = pd.NA
        last_vit_date["Intention"] = pd.NA
        last_vit_date["Anzahl_Tage_Diagnose"] = pd.NA
        last_vit_date["Menge_OPS_Code"] = pd.NA
        last_vit_date["Stellung_OP"] = pd.NA
        last_vit_date["Therapieart"] = pd.NA
        last_vit_date["Geburtsdatum"] = patient_data["Geburtsdatum"].item()
        last_vit_date["Verstorben"] = patient_data["Verstorben"]
        last_vit_date["Todesursache"] = patient_data["Todesursache_Grundleiden"]

        patient_time_line = pd.concat([patient_time_line, last_vit_date], ignore_index=True)
        # sort dates ascending
        patient_time_line.sort_values(by=['Datum'], inplace=True, ascending=True, ignore_index=True)


        # append following events
        if not patient_fe.empty:
            new_fe = pd.DataFrame(columns=["Patient_ID", "Tumor_ID", "Primaertumor_ICD", "Ereignis_Typ", "Datum", "Geburtsdatum", "Verstorben", "Todesursache", "Intention", "Beurteilung_Residualstatus", "Gesamtbeurteilung_Tumorstatus", "Anzahl_Tage_Diagnose", "Menge_OPS_Code", "Stellung_OP", "Therapieart"])
            new_fe["Patient_ID"] = patient_fe["Patient_ID_FK"]
            new_fe["Tumor_ID"] = patient_fe["Tumor_ID_FK"]
            new_fe["Primaertumor_ICD"] = tumor_data["Primaertumor_ICD"].item()
            new_fe["Datum"] = patient_fe["Datum_Folgeereignis"]
            new_fe["Ereignis_Typ"] = "FE"
            new_fe["Gesamtbeurteilung_Tumorstatus"] = patient_fe["Gesamtbeurteilung_Tumorstatus"]
            new_fe["Beurteilung_Residualstatus"] = pd.NA
            new_fe["Intention"] = pd.NA
            new_fe["Anzahl_Tage_Diagnose"] = pd.NA
            new_fe["Menge_OPS_Code"] = pd.NA
            new_fe["Stellung_OP"] = pd.NA
            new_fe["Therapieart"] = pd.NA
            new_fe["Geburtsdatum"] = patient_data["Geburtsdatum"].item()
            new_fe["Verstorben"] = "0"
            new_fe["Todesursache"] = pd.NA

            patient_time_line = pd.concat([patient_time_line, new_fe], ignore_index=True)
            # sort dates ascending
            patient_time_line.sort_values(by=['Datum'], inplace=True, ascending=True, ignore_index=True)


        # arange op data and append to patient time line
        if not op.empty:
            new_op = pd.DataFrame(columns=["Patient_ID", "Tumor_ID", "Primaertumor_ICD", "Ereignis_Typ", "Datum", "Geburtsdatum", "Verstorben", "Todesursache", "Intention", "Beurteilung_Residualstatus", "Gesamtbeurteilung_Tumorstatus", "Anzahl_Tage_Diagnose", "Menge_OPS_Code", "Stellung_OP", "Therapieart"])
            new_op["Patient_ID"] = op["Patient_ID_FK"]
            new_op["Tumor_ID"] = op["Tumor_ID_FK"]
            new_op["Primaertumor_ICD"] = tumor_data["Primaertumor_ICD"].item()
            new_op["Datum"] = op["Datum_OP"]
            new_op["Ereignis_Typ"] = "OP"
            new_op["Beurteilung_Residualstatus"] = op["Beurteilung_Residualstatus"]
            new_op["Intention"] = op["Intention"]
            new_op["Anzahl_Tage_Diagnose"] = op["Anzahl_Tage_Diagnose_OP"]
            new_op["Menge_OPS_Code"] = op["Menge_OPS_code"]
            new_op["Gesamtbeurteilung_Tumorstatus"] = pd.NA
            new_op["Stellung_OP"] = pd.NA
            new_op["Therapieart"] = pd.NA
            new_op["Geburtsdatum"] = patient_data["Geburtsdatum"].item()
            new_op["Verstorben"] = "0"
            new_op["Todesursache"] = pd.NA

            patient_time_line = pd.concat([patient_time_line, new_op], ignore_index=True)
            # sort dates ascending
            patient_time_line.sort_values(by=['Datum'], inplace=True, ascending=True, ignore_index=True)

        # append radiotherapy
        if not strahl.empty:
            new_strahl = pd.DataFrame(columns=["Patient_ID", "Tumor_ID", "Primaertumor_ICD", "Ereignis_Typ", "Datum", "Geburtsdatum", "Verstorben", "Todesursache", "Intention", "Beurteilung_Residualstatus", "Gesamtbeurteilung_Tumorstatus", "Anzahl_Tage_Diagnose", "Menge_OPS_Code", "Stellung_OP", "Therapieart"])
            new_strahl["Patient_ID"] = strahl["Patient_ID_FK"]
            new_strahl["Tumor_ID"] = strahl["Tumor_ID_FK"]
            new_strahl["Primaertumor_ICD"] = tumor_data["Primaertumor_ICD"].item()
            new_strahl["Ereignis_Typ"] = "STR"
            new_strahl["Datum"] = strahl["Beginn_Bestrahlung"]
            new_strahl["Beurteilung_Residualstatus"] = pd.NA
            new_strahl["Intention"] = strahl["Intention_st"]
            new_strahl["Anzahl_Tage_Diagnose"] = strahl["Anzahl_Tage_Diagnose_ST"]
            new_strahl["Menge_OPS_Code"] = pd.NA
            new_strahl["Gesamtbeurteilung_Tumorstatus"] = pd.NA
            new_strahl["Stellung_OP"] = strahl["Stellung_OP"]
            new_strahl["Therapieart"] = pd.NA
            new_strahl["Geburtsdatum"] = patient_data["Geburtsdatum"].item()
            new_strahl["Verstorben"] = "0"
            new_strahl["Todesursache"] = pd.NA

            patient_time_line = pd.concat([patient_time_line, new_strahl], ignore_index=True)
            # sort dates ascending
            patient_time_line.sort_values(by=['Datum'], inplace=True, ascending=True, ignore_index=True)

        # append system therapy
        if not sys.empty:
            new_sys = pd.DataFrame(columns=["Patient_ID", "Tumor_ID", "Primaertumor_ICD", "Ereignis_Typ", "Datum", "Geburtsdatum", "Verstorben", "Todesursache", "Intention", "Beurteilung_Residualstatus", "Gesamtbeurteilung_Tumorstatus", "Anzahl_Tage_Diagnose", "Menge_OPS_Code", "Stellung_OP", "Therapieart"])
            new_sys["Patient_ID"] = sys["Patient_ID_FK"]
            new_sys["Tumor_ID"] = sys["Tumor_ID_FK"]
            new_sys["Primaertumor_ICD"] = tumor_data["Primaertumor_ICD"].item()
            new_sys["Ereignis_Typ"] = "SYS"
            new_sys["Datum"] = sys["Beginn_SYST"]
            new_sys["Beurteilung_Residualstatus"] = pd.NA
            new_sys["Intention"] = sys["Intention_sy"]
            new_sys["Anzahl_Tage_Diagnose"] = sys["Anzahl_Tage_Diagnose_SYST"]
            new_sys["Menge_OPS_Code"] = pd.NA
            new_sys["Gesamtbeurteilung_Tumorstatus"] = pd.NA
            new_sys["Stellung_OP"] = sys["Stellung_OP"]
            new_sys["Therapieart"] = sys["Therapieart"]
            new_sys["Geburtsdatum"] = patient_data["Geburtsdatum"].item()
            new_sys["Verstorben"] = "0"
            new_sys["Todesursache"] = pd.NA

            patient_time_line = pd.concat([patient_time_line, new_sys], ignore_index=True)
            # sort dates ascending
            patient_time_line.sort_values(by=['Datum'], inplace=True, ascending=True, ignore_index=True)

        # append patient time line to time line of all patients
        time_line = pd.concat([time_line, patient_time_line], ignore_index=True)

    # save time line dataframe as csv file
    time_line.to_csv(f"time_line_{entity}_{registry}.csv")

def split_operations(operations:pd.DataFrame, entity:str):
    if entity == "breast":
        relevant_ops = {
            #Tumortherapeutische Operation
            "5-870": "Partielle (brusterhaltende) Exzision der Mamma und Destruktion von Mammagewebe",
            "5-872": "(Modifizierte radikale) Mastektomie",
            "5-874": "Erweiterte (radikale) Mastektomie mit Resektion an den Mm. pectorales majores et minores und Thoraxwandteilresektion",
            "5-877": "Subkutane Mastektomie und hautsparende Mastektomieverfahren",
            "5-882": "Operationen an der Brustwarze",
            
            #Lymphknotenexzision
            "5-401.1": "Exzision einzelner Lymphknoten und Lymphgefäße, Axillär",
            "5-406.1": "Regionale Lymphadenektomie (Ausräumung mehrerer Lymphknoten einer Region) im Rahmen einer anderen Operation, Axillär",
            "5-407.0": "Radikale (systematische) Lymphadenektomie im Rahmen einer anderen Operation, Axillär",

            #Lymphknotenexzision als selbstständiger Eingriff

            "5-402.1": "Regionale Lymphadenektomie (Ausräumung mehrerer Lymphknoten einer Region) als selbständiger Eingriff, Axillär",
            "5-404.0": "Radikale (systematische) Lymphadenektomie als selbständiger Eingriff, Axillär"

        }

        # menge_op = operations["Menge_OPS_code"].str.split(";")
        def create_binary_vector(ops_string):
            ops_list = ops_string.split(";")  # Split OPS codes
            return [1 if any(code.startswith(prefix) for code in ops_list) else 0 for prefix in relevant_ops]
        # Apply function to create binary vectors
        operations["binary_vector"] = operations["Menge_OPS_code"].apply(create_binary_vector)
    return operations