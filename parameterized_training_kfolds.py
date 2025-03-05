import sys

from sklearn.ensemble import RandomForestClassifier
import torch
import datetime
import pickle
from pathlib import Path

import evaluation
import numpy as np
import optuna
import pandas as pd
import yaml
from datenimport_aicare.data_loading import import_vonko, import_aicare
from datenimport_aicare.data_preprocessing import (calculate_survival_time,
                                                   encode_selected_variables,
                                                   imputation,
                                                   get_morphology_groups,
                                                   tumorDataset)
from evaluation import PartialLogLikelihood, PartialMSE
from sklearn.model_selection import KFold, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.util import Surv
from training_survival_analysis import train_model
from models import TabNetSurvivalRegressor
import argparse
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imputation_method", type=str, default="none", help="Which imputation method to use")
    parser.add_argument("--model", type=str, default="rsf", help="Which model to use")
    parser.add_argument("--deep_surv_model", type=str, default="none", help="If model is deep_surv, which specific deep_surv model to use")
    parser.add_argument("--tnm", action="store_true", help="Use TNM instead of UICC")
    parser.add_argument("--one-hot", action="store_true", help="Use one-hot encoding instead of label encoding")
    parser.add_argument("--loss", type=str, default="pll", help="Which loss function to use for Tabnet and Deep_Surv")
    parser.add_argument("--dataset", type=str, default="vonko", help="Which dataset to use")
    parser.add_argument("--imputation_before", action="store_true", help="Impute the data before splitting or afterwards")
    args = parser.parse_args()
    model = args.model
    if model not in ["rsf", "cox", "deep_surv", "tabnet"]:
        raise ValueError("Model not implemented")
    if args.imputation_method not in ["none", "KNNImputer", "SimpleImputer", "MissForest"]:
        raise ValueError("Imputation method not implemented")
    if args.loss == "pll":
        loss_fn = PartialLogLikelihood
    elif args.loss == "mse":
        loss_fn = PartialMSE
    else:
        raise ValueError("Loss function not implemented")
    if args.dataset =="vonko":
        imputation_features = ["geschl", "alter", "uicc", "histo_gr", "vit_status", "survival_time"]
        selected_features = ["geschl", "alter", "histo_gr", "uicc"]  
        subset = "uicc"
        if args.tnm:
            imputation_features = ["geschl", "alter", "tnm_t", "tnm_n", "tnm_m", "histo_gr", "vit_status", "survival_time"]
            selected_features = ["geschl", "alter", "histo_gr", "tnm_t", "tnm_n", "tnm_m"]
            subset = "tnm"
    elif args.dataset == "aicare":
        imputation_features = ["Geschlecht", "Alter_bei_Diagnose", "Morphologie_Gruppe", "UICC", "vit_status", "survival_time"]
        selected_features = ["Geschlecht", "Alter_bei_Diagnose", "Morphologie_Gruppe", "UICC"]
        subset = "uicc"

        if args.tnm:
            imputation_features = ["Geschlecht", "Alter_bei_Diagnose", "Morphologie_Gruppe", "TNM_T", "TNM_N", "TNM_M", "vit_status", "survival_time"]
            selected_features = ["Geschlecht", "Alter_bei_Diagnose", "Morphologie_Gruppe", "TNM_T", "TNM_N", "TNM_M"]
            subset = "tnm"

    
    config = yaml.safe_load(Path("./config.yaml").read_text())
    base_path = config["base_path"]
    study_name=f"{args.dataset}/{subset}/missings_imputed_with_{args.imputation_method}"
    if args.one_hot:
        study_name = study_name + "_onehot"
    if args.imputation_before:
        study_name = study_name + "_imputation_before_splitting"
    
    study_db="sqlite:///optuna.db"
    #If folder does not exist, create it
    if model == "deep_surv":
        dsmodel = args.deep_surv_model
        if dsmodel not in ["minimalistic_network"]:
            raise ValueError("DeepSurv model not implemented")
    Path(f"{base_path}/results/{study_name}").mkdir(parents=True, exist_ok=True)
    #Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = f"{base_path}/results/{study_name}/log_study_{model}"

    if args.dataset == "aicare":
        log_path += f"_registry{config['registry']}"
        log_path += f"_entity_{config['entity']}"
    
    if model=="deep_surv":
        log_path += f"_{dsmodel}_{args.loss}.txt"    
        
    elif model=="tabnet":
        log_path += f"_{args.loss}.txt"

    else:
        log_path += ".txt"

    logger.addHandler(logging.FileHandler(log_path, mode="w"))
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    

    random_state = 42
    np.random.seed(random_state)


    
    config["rsf"]["max_features"]["max"] = len(selected_features)

    # Import Dataset + Preprocessing
    if args.dataset == "vonko":
        vonko = import_vonko(f"{base_path}/aicare/raw/", oncotree_data=False,
                            processed_data=True, extra_features=True, simplify=True)
        
        X = vonko["Tumoren"].copy()
        X["survival_time"] = calculate_survival_time(X, "vitdat", "diagdat")
    elif args.dataset == "aicare":
        aicare_dataset = import_aicare(f"{base_path}/aicare/aicare_gesamt/", tumor_entity=config["entity"], registry=config["registry"])
        X = pd.merge(aicare_dataset["patient"], aicare_dataset["tumor"], how="left", left_on="Patient_ID", right_on="Patient_ID_FK")
        X["survival_time"] = calculate_survival_time(X, "Datum_Vitalstatus", "Diagnosedatum")
        morphology_groups, morpho_df = get_morphology_groups(X["Primaertumor_Morphologie_ICD_O"], basepath=f"{base_path}/aicare", entity = config["entity"], ontoserver_url=config["ontoserver_url"])
        X["Morphologie_Gruppe"] = morphology_groups
        X["Morphologie_Gruppe"] = X.loc[:, "Morphologie_Gruppe"].astype(pd.CategoricalDtype(ordered=True))
        X = X[X["survival_time"]>=0]
        X["Alter_bei_Diagnose"] = (X['Diagnosedatum'] - X['Geburtsdatum']).dt.days // 365.25
        X.rename(columns={"Verstorben": "vit_status"}, inplace=True)


    
    X, encoder = encode_selected_variables(X, imputation_features, na_sentinel=True)

    y = pd.DataFrame({'vit_status': X['vit_status'].astype(bool),
                    'survival_time': X['survival_time']})
    y = Surv.from_dataframe("vit_status", "survival_time", y)


    if args.imputation_method == "none":
        # for each column in selected_features, replace -1 with number of categories + 1
        for feature in selected_features:
            X[feature] = X[feature].replace(-1, len(X[feature].unique())-1)
    
    if args.imputation_before:
        X = imputation(X, imputation_features=imputation_features, selected_features=selected_features, imputation_method=args.imputation_method, one_hot=args.one_hot,
                       logger=logger, random_state=random_state)

    
        

    # Select Features for training
    
    # Split between Test and Training for Hyperparameter Tuning
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    if not args.imputation_before:
        X_test = imputation(X_test, imputation_features=imputation_features, selected_features=selected_features, imputation_method=args.imputation_method, one_hot=args.one_hot,
                            logger=logger, random_state=random_state)

    # create k folds and save them
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    folds = kfold.split(X_train, y_train)
    for i, (train_fold, test_fold) in enumerate(folds):
        # save folds as yaml file
        with open(f"{base_path}/results/{study_name}/folds_{i}.yaml", "w") as f:
            fold_data = {"Training": train_fold.tolist(), "Test": test_fold.tolist()}
            yaml.dump(fold_data, f)
    
    #Random Search
    
    def objective_rsf(trial: optuna.Trial):
        '''Objective function for Random Survival Forests'''
        params = {
            'n_estimators': trial.suggest_int('n_estimators',
                                              config["rsf"]["n_estimators"]["min"],
                                              config["rsf"]["n_estimators"]["max"]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf',
                                                  config["rsf"]["min_samples_leaf"]["min"],
                                                  config["rsf"]["min_samples_leaf"]["max"]),
            'max_features': trial.suggest_int('max_features', 
                                              config["rsf"]["max_features"]["min"],
                                              len(selected_features)),
            'max_depth': trial.suggest_int('max_depth', 
                                           config["rsf"]["max_depth"]["min"],
                                           config["rsf"]["max_depth"]["max"]),
            #'max_samples': trial.suggest_float('max_samples', 0.5, 1.0)
        }

        
        scores = []
        for train_fold, _ in kfold.split(X_train, y_train):
            model = RandomSurvivalForest(**params, random_state=random_state, n_jobs=8)
            if not args.imputation_before:
                X_train_fold = imputation(X_train.iloc[train_fold],imputation_features=imputation_features, selected_features=selected_features,
                                            one_hot=args.one_hot, imputation_method=args.imputation_method, logger=logger, random_state=random_state)
            else:
                X_train_fold = X_train.iloc[train_fold]
            model.fit(X_train_fold, y_train[train_fold])
            scores.append(model.score(X_test, y_test))
        trial_nr = trial.number
        
        logger.info(f"Trial {trial_nr}: {scores}")
        score = np.mean(scores)
        return score  # s['c_index'] #scores['c_index'], scores['ibs'], scores['mean_auc']
    
    def objective_cox(trial: optuna.Trial):
        '''Objective function for Cox PH'''
        params = {
            'alpha': trial.suggest_float('alpha',
                                         config["cox"]["alpha"]["min"],
                                         config["cox"]["alpha"]["max"]),
            'tol': trial.suggest_float('tol',
                                       config["cox"]["tol"]["min"],
                                       config["cox"]["tol"]["max"]),
            'ties': trial.suggest_categorical('ties',
                                              config["cox"]["ties"])
        }
        scores = []
        for train_fold, _ in kfold.split(X_train, y_train):
            model = CoxPHSurvivalAnalysis(**params)
            if not args.imputation_before:
                X_train_fold = imputation(X_train.iloc[train_fold],imputation_features=imputation_features, selected_features=selected_features,
                                            one_hot=args.one_hot, imputation_method=args.imputation_method, logger=logger, random_state=random_state)
            else:
                X_train_fold = X_train.iloc[train_fold]
            model.fit(X_train_fold, y_train[train_fold])
            scores.append(model.score(X_test, y_test))
        trial_nr = trial.number
        logger.info(f"Trial {trial_nr}: {scores}")
        score = np.mean(scores)
        return score
    
    def objective_deep_surv(trial: optuna.Trial, stable_params):
        '''Objective function for DeepSurv'''
        
        flexible_params = {
            "batch_size": trial.suggest_categorical("batch_size",
                                                    config["deep_surv"]["batch_size"]),
            "inner_dim": trial.suggest_categorical("inner_dim", 
                                                   config["deep_surv"]["inner_dim"]),
            "lr": trial.suggest_categorical("lr", config["deep_surv"]["lr"]),
            "weight_decay": trial.suggest_categorical("weight_decay",
                                                      config["deep_surv"]["weight_decay"])
        }
        params = {**stable_params, **flexible_params}
        logger.info(f"Trial {trial.number}: {flexible_params}")
        scores = []
        dataset_test = torch.Tensor(X_test.values)
        for foldnr, (train_fold, _) in enumerate(kfold.split(X_train, y_train)):
            if not args.imputation_before:
                X_train_fold = imputation(X_train.iloc[train_fold],imputation_features=imputation_features, selected_features=selected_features,
                                            one_hot=args.one_hot, imputation_method=args.imputation_method, logger=logger, random_state=random_state)
            else:
                X_train_fold = X_train.iloc[train_fold]
            dataset_train = tumorDataset(X_train_fold, y_train["vit_status"][train_fold], y_train["survival_time"][train_fold])
            model, losses, test_eval = train_model(dataset_train, params, trial=trial)
            model.eval()
            y_pred = model(dataset_test.to(params["device"])).detach().cpu().numpy()
            y_pred = y_pred + np.random.random(y_pred.shape) * 1e-7
            scores.append(concordance_index_censored(y_test["vit_status"], y_test["survival_time"], np.squeeze(y_pred))[0])
        trial_nr = trial.number
        logger.info(f"Trial {trial_nr}: {scores}")
        score = np.mean(scores)
        return score

        
    def objective_tabnet(trial: optuna.Trial):
        params = {
            "n_d": trial.suggest_int("n_d",
                                     config["tabnet"]["n_d"]["min"],
                                     config["tabnet"]["n_d"]["max"]),
            "n_steps": trial.suggest_int("n_steps",
                                         config["tabnet"]["n_steps"]["min"],
                                         config["tabnet"]["n_steps"]["max"]),
            "gamma": trial.suggest_categorical("gamma",
                                               config["tabnet"]["gamma"]),
            "optimizer_params": {
                "lr": trial.suggest_categorical("lr", config["tabnet"]["lr"]),
                "weight_decay": trial.suggest_categorical("weight_decay",
                                                          config["tabnet"]["weight_decay"]),
            },
            "mask_type": trial.suggest_categorical("mask_type",
                                                   config["tabnet"]["mask_type"]),
        }  
        scores= []
        for foldnr, (train_fold, _) in enumerate(kfold.split(X_train, y_train)):
            y_train_numpy = np.stack((y_train["vit_status"][train_fold], y_train["survival_time"][train_fold]), axis=-1) #np.expand_dims(y_train["vit_status"][train_fold],1) # 
            if not args.imputation_before:
                X_train_fold = imputation(X_train.iloc[train_fold],imputation_features=imputation_features, selected_features=selected_features,
                                            one_hot=args.one_hot, imputation_method=args.imputation_method, logger=logger, random_state=random_state)
            else:
                X_train_fold = X_train.iloc[train_fold]

            if args.dataset == "vonko":
                cat_dims = [len(pd.unique(X[feature])) for feature in selected_features if feature != "alter"]
            elif args.dataset == "aicare":
                cat_dims = [len(pd.unique(X[feature])) for feature in selected_features if feature != "Alter_bei_Diagnose"]
            
            cat_idxs = [0,2,3]
            if subset == "tnm":
                cat_idxs = [0,2,3,4,5]
            tabnet = TabNetSurvivalRegressor(seed=random_state, device_name=config["device"], 
                                     n_a=params["n_d"], cat_idxs=cat_idxs, 
                                     cat_dims=cat_dims, **params)
            
            tabnet.fit(
                X_train_fold.values, y_train_numpy,
                loss_fn=loss_fn,
                max_epochs=100
            )
            
            with torch.no_grad():
                y_pred = tabnet.predict(X_test.values)
                
                y_pred = y_pred + np.random.random(y_pred.shape) * 1e-7
                print(y_pred.shape)
                scores.append(concordance_index_censored(y_test["vit_status"], y_test["survival_time"], np.squeeze(y_pred))[0])
        trial_nr = trial.number
        logger.info(f"Trial {trial_nr}: {scores}")
        score = np.mean(scores)
        return score
        


    # Create Study with objective "Maximize C-Index"
    study = optuna.create_study(study_name=study_name+"_"+model+str(datetime.datetime.now()),
                                storage=study_db,
                                direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state)
                                ) 
    if model == "rsf":
        study.optimize(objective_rsf, n_trials=50, n_jobs=2)
    elif model == "cox":
        study.optimize(objective_cox, n_trials=50, n_jobs=2)
    elif model == "deep_surv":
        stable_params = {
            "device": config["device"],
            "model": dsmodel,
            "epochs": 300,
            "input_dim": len(selected_features),
            "loss_fn" : loss_fn
        }
        study.optimize(lambda trial: objective_deep_surv(trial, stable_params), n_trials=50, n_jobs=2)
    elif model == "tabnet":
        study.optimize(objective_tabnet, n_trials=50, n_jobs=1)

    best_params = study.best_trial.params
    logger.info("RESULTS")
    logger.info(f"Best params: {best_params}")
    logger.info(f"Best value: {study.best_value}")
    logger.info("Evaluating best model on test set")
    for i, (train_fold , val_fold) in enumerate(kfold.split(X_train, y_train)):
        if not args.imputation_before:
            X_train_fold = imputation(X_train.iloc[train_fold],imputation_features=imputation_features, selected_features=selected_features,
                                            one_hot=args.one_hot, imputation_method=args.imputation_method, logger=logger, random_state=random_state)
            X_val_fold = imputation(X_train.iloc[val_fold], imputation_features=imputation_features, selected_features=selected_features,
                                        imputation_method=args.imputation_method, one_hot=args.one_hot,
                                        logger=logger, random_state=random_state)
        else:
            X_train_fold = X_train.iloc[train_fold]
            X_val_fold = X_train.iloc[val_fold]
        if model == "rsf":
            best_model = RandomSurvivalForest(**best_params, random_state=random_state, n_jobs=32)
            best_model.fit(X_train_fold, y_train[train_fold])
            scores = evaluation.evaluate_survival_model(best_model, X_val_fold, y_train[train_fold],
                                                    y_train[val_fold])
            pickle.dump(best_model, open(f"{base_path}/results/{study_name}/model_{i}.pkl", "wb"))
        elif model == "cox":
            best_model = CoxPHSurvivalAnalysis(**best_params)
            best_model.fit(X_train_fold, y_train[train_fold])
            scores = evaluation.evaluate_survival_model(best_model, X_val_fold, y_train[train_fold],
                                                    y_train[val_fold])
        elif model == "deep_surv":
            dataset_train = tumorDataset(X_train_fold, y_train["vit_status"][train_fold], y_train["survival_time"][train_fold])
            best_model, losses, test_eval = train_model(dataset_train, {**stable_params, **best_params})
            best_model.eval()
            y_pred = best_model(torch.Tensor(X_val_fold.values).to(stable_params["device"])).detach().cpu().numpy()
            scores = evaluation.evaluate_survival_model(best_model, X_val_fold.values, y_train[train_fold],
                                                        y_train[val_fold])
        elif model == "tabnet":
            optimizer_params = {
                "lr": best_params["lr"],
                "weight_decay": best_params["weight_decay"]
            }
            best_params["optimizer_params"] = optimizer_params
            best_params_subset = best_params.copy()
            best_params_subset.pop("lr")
            best_params_subset.pop("weight_decay")
            if args.dataset == "vonko":
                cat_dims = [len(pd.unique(X[feature])) for feature in selected_features if feature != "alter"]
            elif args.dataset == "aicare":
                cat_dims = [len(pd.unique(X[feature])) for feature in selected_features if feature != "Alter_bei_Diagnose"]

            cat_idxs = [0,2,3]
            if subset == "tnm":
                cat_idxs = [0,2,3,4,5]
            best_model = TabNetSurvivalRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims, seed=random_state,
                                         device_name=config["device"], 
                                         n_a=best_params["n_d"], **best_params_subset)
            y_train_numpy = np.stack((y_train["vit_status"][train_fold], y_train["survival_time"][train_fold]), axis=-1) #np.expand_dims(y_train["vit_status"][train_fold],1) 
            best_model.fit(
                X_train_fold.values, y_train_numpy,
                loss_fn=loss_fn
            )
            scores = evaluation.evaluate_survival_model(best_model, X_val_fold.values, y_train[train_fold],
                                                        y_train[val_fold])
        logger.info(f"Fold {i}:")
        logger.info(scores)
        hyperparameter_path = f"{base_path}/results/{study_name}/parameters_study_{model}"
        if args.dataset == "aicare":
            hyperparameter_path += f"_registry{config['registry']}"
            hyperparameter_path += f"_entity_{config['entity']}"
        if model=="deep_surv":
            hyperparameter_path += f"_{dsmodel}_{args.loss}.yaml"
        elif model=="tabnet":
            hyperparameter_path += f"_{args.loss}.yaml"
        else:
            hyperparameter_path += ".yaml"
        Path(hyperparameter_path).write_text(yaml.dump(best_params))

        
        
