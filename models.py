import torch
import math
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.utils import filter_weights
from evaluation import PartialLogLikelihood
import numpy as np
import pandas as pd

def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        module.weight.data = torch.nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(module.bias, -bound, bound)

class MinimalisticNetwork(torch.nn.Module):
    '''
    This is a simple network for DeepSurv using Dense Layers.
    '''

    def __init__(self, input_dim=15, inner_dim=128) -> None:
        '''
        Initialize the network with the input and output dimensions.
        @param input_dim: The input dimension of the network.
        @param inner_dim: The inner dimension of the network.
        '''
        super().__init__()
        self.network = torch.nn.Sequential(
            
            torch.nn.Linear(input_dim, inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(inner_dim),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(inner_dim, inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(inner_dim),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(inner_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, x, *args, **kwargs):
        '''
        Forward pass of the network.
        @param x: The input tensor.
        @return: The output tensor.
        '''
        x = self.network(x)

        return x 
    
    def predict(self, X):
        if type(X) is pd.DataFrame:
            X = X.values
        X = torch.tensor(X, dtype=torch.float32, device=next(self.parameters()).device)
        risk_score = self.forward(X).detach().numpy().squeeze()
        return risk_score
    
    def fit(self, X,y):
        """
        Not in function, needed for the permutation importances check
        """
        pass

    def score(self, X, y):
        """Returns the concordance index of the prediction.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        cindex : float
            Estimated concordance index.
        """
        from sksurv.metrics import concordance_index_censored

        name_event, name_time = y.dtype.names
        risk_score = self.predict(X)

        

        result = concordance_index_censored(y[name_event], y[name_time], risk_score)
        return result[0]
    
class TabNetSurvivalRegressor(TabModel):
    def __post_init__(self):
        super(TabNetSurvivalRegressor, self).__post_init__()
        self._task = 'regression'
        self._default_loss = PartialLogLikelihood
        self._default_metric = 'pll'

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true[:,0], y_true[:,1])

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights
    ):
        # if len(y_train.shape) != 2:
        #     msg = "Targets should be 2D : (n_samples, n_regression) " + \
        #           f"but y_train.shape={y_train.shape} given.\n" + \
        #           "Use reshape(-1, 1) for single regression."
        #     raise ValueError(msg)
        self.output_dim = 1
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score
    
    def score(self, X, y):
        """Returns the concordance index of the prediction.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        cindex : float
            Estimated concordance index.
        """
        from sksurv.metrics import concordance_index_censored

        name_event, name_time = y.dtype.names

        risk_score = self.predict(X.values).squeeze()

        result = concordance_index_censored(y[name_event], y[name_time], risk_score)
        return result[0]