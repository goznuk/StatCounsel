from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score,\
    cumulative_dynamic_auc, integrated_brier_score
import numpy as np
from typing import Dict
import torch
import torch.nn
from scipy.stats import norm
from sksurv.nonparametric import kaplan_meier_estimator
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.abstract_model import TabModel
from torch.utils.data import Dataset
import pandas as pd

def evaluate_survival_model(model, X_test, y_train_numpy, y_test_numpy) -> Dict:
    """
    Evaluate a survival model on the test set with multiple metrics.
    model: Survival model (can be CoxPH, RSF, Pytorch Model or TabNet)
    X_test: Matrix with the variables used for prediction
    y_train_numpy: Survival times for training data to estimate the censoring distribution from. A structured array containing the binary event indicator as first field, and time of event or time of censoring as second field. #첫줄 01, 둘째줄 시간
    y_test_numpy: Survival times of test data. A structured array containing the binary event indicator as first field, and time of event or time of censoring as second field. 
    """

    test_min = float(y_test_numpy['survival_time'].min())
    test_max = float(y_test_numpy['survival_time'].max())

    # 수정
    # 기존 코드 (학습 데이터의 quantile로 생성)
    # times = np.unique(np.quantile(y_train_numpy['survival_time'][y_train_numpy['vit_status'] == 1],
    #                               np.linspace(0.1, 1, 20)).astype(int))

    # 수정: 테스트 데이터 범위 내에서 균등하게 time points 생성 (최대값은 미포함)
    times = np.linspace(test_min, test_max, 20, endpoint=False)


    if isinstance(model, torch.nn.Module):
        model.eval()
        model.cpu()
        with torch.no_grad():
            y_pred = np.squeeze(model(torch.Tensor(np.array(X_test))).detach().cpu().numpy())
          
    else:
        y_pred = np.squeeze(model.predict(X_test))
    c_index = concordance_index_censored(y_test_numpy['vit_status'], y_test_numpy['survival_time'], y_pred)[0]
   
   
    
    
    #for the AUC and IBS we need the survival time to be less than the maximum survival time in the training set 
    test_selection = np.where(y_test_numpy["survival_time"]<= y_train_numpy["survival_time"].max())
    y_test_numpy = y_test_numpy[test_selection]
    y_pred = y_pred[test_selection]
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.iloc[test_selection]
    else:
        X_test = X_test[test_selection]
    times = times[times < int(y_test_numpy['survival_time'].max())]
    auc, mean_auc = cumulative_dynamic_auc(y_train_numpy, y_test_numpy, y_pred, times=times)

    if isinstance(model, torch.nn.Module) or isinstance(model, TabNetRegressor) or isinstance(model, TabModel):
        # These models do not return a survival function, so we have to calculate it ourselves
        baseline_hazard_times, baseline_hazard = kaplan_meier_estimator(y_test_numpy["vit_status"], y_test_numpy["survival_time"])
        print(np.mean(y_pred))
        max_value = np.log(np.log(np.finfo(np.float32).max )) 
        clipped_y_pred = np.clip(y_pred, -max_value, max_value)
        rates = np.array([baseline_hazard ** np.exp(clipped_y_pred_i) for clipped_y_pred_i in clipped_y_pred])
        surv_prob = []
        for time in times:
            kaplan_index = np.argmax(baseline_hazard_times > time)
            surv_prob.append(rates[:, kaplan_index])
        surv_prob = np.column_stack(surv_prob)
        ibs = integrated_brier_score(y_train_numpy, y_test_numpy, surv_prob, times) 
    else:
        surv_prob = np.row_stack([
            fn(times)
            for fn in model.predict_survival_function(X_test)
        ])
        ibs = integrated_brier_score(y_train_numpy, y_test_numpy, surv_prob, times)


    return {
        "c_index": c_index,
        "mean_auc": mean_auc,
        "ibs": ibs
    }



def onePair(x0, x1):
    c = np.log(2.)
    m = torch.nn.LogSigmoid() 
    return 1 + m(x1-x0) / c
  
def rank_loss(logits, times, fail_indicator):
    N = logits.size(0)
    allPairs = onePair(logits.view(N,1), logits.view(1,N))

    temp0 = times.view(1, N) - times.view(N, 1)
    # indices based on times time
    temp1 = temp0>0
    # indices of event-event or event-censor pair
    temp2 = fail_indicator.view(1, N) + fail_indicator.view(N, 1)
    temp3 = temp2>0
    # indices of events
    temp4 = fail_indicator.view(N, 1) * torch.ones(1, N, device = logits.device)
    # selected indices
    final_ind = temp1 * temp3 * temp4
    out = allPairs * final_ind
    return out.sum() / final_ind.sum()

# Convert survival time to risk score
def survival_time_to_risk_score(survival_time, max_time=2106):
    return 1 - survival_time / max_time

def PartialMSE(logits, fail_indicator, times=None):
    '''
    Implementation of a partial MSE loss function
    '''
    # risks = survival_time_to_risk_score(times)
    loss = torch.nn.MSELoss(reduction='none')
    logits = torch.squeeze(logits)
    mse = torch.mean(fail_indicator * loss(logits, times))
    penalty = torch.mean((logits < times) * (1-fail_indicator) * loss(logits, times))
    return mse + penalty - rank_loss(logits, times, fail_indicator) 






def PartialLogLikelihood(logits, fail_indicator, times=None, ties="noties"):
    '''
    Implementation of partial log-likelihood loss function
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model 
    ties: 'noties'
    '''
    logL = 0
    
    logits = logits.squeeze()
    
    # get time order for prediction and fails
    if times is None:
        times = torch.arange(logits.shape[0]).to(logits.device)
    time_index = torch.argsort(-times)
    logits = logits[time_index]
    fail_indicator = fail_indicator[time_index]
    times = times[time_index]
    if ties == 'noties':
        
        log_risk = torch.logcumsumexp(logits, 0)
        likelihood = logits - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * fail_indicator
        logL = -torch.mean(uncensored_likelihood)
    else: 
        raise NotImplementedError("Method not implemented")

    return logL