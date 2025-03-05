# %%
# Import
import sys
sys.path.append('../')
from datenimport_aicare.data_loading import import_vonko, import_aicare
from datenimport_aicare.data_preprocessing import calculate_survival_time, impute, encode_selected_variables
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

# %%
#Load data
vonko=False
config = yaml.safe_load(Path("./config.yaml").read_text())
base_path = config["base_path"]

if vonko:
    data = import_vonko(path=f"{base_path}/aicare/raw/", oncotree_data=False,
                            processed_data=True, extra_features=False, simplify=True)
    X = data["Tumoren"].copy()
    X["survival_time"] = calculate_survival_time(X, "vitdat", "diagdat")
else:
    data = import_aicare(path=f"{base_path}/aicare/aicare_gesamt/", tumor_entity="lung", registry="all")
    X = pd.merge(data["patient"], data["tumor"], how="left", left_on="Patient_ID_unique", right_on="Patient_ID_unique")
    X["survival_time"] = calculate_survival_time(X, "Datum_Vitalstatus", "Diagnosedatum")
    X.rename(columns={"Verstorben": "vit_status"}, inplace=True)
    X = X[X["survival_time"] > 0]
    for tnm_col in ["TNM_T", "TNM_N", "TNM_M"]:
        X[tnm_col] = X[tnm_col].str.split("[^0-4]", regex=True).str[0].str.strip().replace(r'^\s*$', np.nan, regex=True).astype(pd.CategoricalDtype(ordered=True))



          
#X, encoder = encode_selected_variables(X, imputation_features, na_sentinel=True)
# X = X.replace(-1, np.nan, inplace=False)
# X = X.dropna(axis=0, how="any", subset=selected_features)

y = pd.DataFrame({'vit_status': X['vit_status'].astype(bool),
                'survival_time': X['survival_time']})
y = Surv.from_dataframe("vit_status", "survival_time", y)

# %%
#Plot Kaplan-Meier curve
def compare_categories(column: str):
    #print(encoder.encodeTable[column].values)
    #print(encoder.encodeTable[column].codes)
    categories = list(X[column].cat.categories)
    categories.append("nan")
    print(categories)
    for category in categories:#zip(encoder.encodeTable[column].values, encoder.encodeTable[column].codes):
        print(category)#, code)
        mask = X[column] == category
        if str(category) == "nan":
            mask = X[column].isna()
        if not mask.any() or mask.values.sum() < 20:
            continue
       
        time, survival_prob, conf_int = kaplan_meier_estimator(
            y["vit_status"][mask],
            y["survival_time"][mask],
            conf_type="log-log")
        #plt.step(time, survival_prob, where="post")
        plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
        plt.step(time, survival_prob, where="post",
                 label=(column + " = %s") % category)

    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")
    
# %%
def count_values(row):
    
    values = ["PUL", "OSS", "HEP", "BRA", "LYM", "MAR", "PLE", "PER", "ADR", "SKI", "OTH", "GEN"]
    selection = row['Primaerdiagnose_Menge_FM']
    if pd.isna(selection):
        return {value: "0" for value in values}
    split_row = selection.split(';')
    value_dict =  {value: (str(split_row.count(value)) if split_row.count(value) <=3 else ">3") for value in values}

    return value_dict

# List of values to count

# Apply the function to each row and create a DataFrame from the resulting dictionary
counts_df = X.apply(count_values, axis=1).apply(pd.Series)

# Add the counts as new columns to the original DataFrame
df_with_counts = pd.concat([X, counts_df], axis=1)
for value in ["PUL", "OSS", "HEP", "BRA", "LYM", "MAR", "PLE", "PER", "ADR", "SKI", "OTH", "GEN"]:
    df_with_counts[value] = df_with_counts[value].astype(pd.CategoricalDtype(ordered=True))

# %%
X = df_with_counts
# %%
column = "PUL"
compare_categories(column)

plt.savefig(f"{base_path}/results/{column}.png", dpi=500)

plt.show()
# %%
