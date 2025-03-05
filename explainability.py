import shap
import shap.maskers
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from models import TabNetSurvivalRegressor, MinimalisticNetwork
import torch


class SHAP():
    def __init__(self, model, X_test, feature_names=None) -> None:
        X_test = X_test.astype(int)
        if type(model) is RandomForestClassifier:
            self.model = "rdf"
            self.explainer = shap.TreeExplainer(model=model, data=X_test, model_output="probability")
            self.shap_values = self.explainer(X_test)
            self.shap_values = self.shap_values[:, :, 1]
            self.data = X_test
        elif type(model) is RandomSurvivalForest or type(model) is CoxPHSurvivalAnalysis or type(model) is TabNetSurvivalRegressor or type(model) is MinimalisticNetwork:
            self.model = "rsf"
            background = shap.maskers.Independent(X_test)
            #wrapped_predict = lambda x: np.expand_dims(model.predict(x), axis=1) # add
            #self.explainer = shap.Explainer(model=wrapped_predict, masker=background) # add
            self.explainer = shap.Explainer(model=model.predict, masker=background)  # data=X_test is sometimes needed
            self.shap_values = self.explainer(X_test)
            self.data = X_test

        elif isinstance(model, torch.nn.Module):
            self.model = "pytorch"
            self.data = torch.tensor(X_test.values, dtype=torch.float32)
            print(self.data.shape)
            self.explainer = shap.DeepExplainer(model=model, data=self.data)
            self.explainer.feature_names = feature_names
            self.shap_values = self.explainer.shap_values(self.data)
            
    
    def plot_violin(self):
        return shap.summary_plot(shap_values=self.shap_values, plot_type="violin", show=False)

    def plot_beeswarm(self):
        return shap.plots.beeswarm(shap_values=self.shap_values, show=False)

    def plot_waterfall(self, index):
        shap.plots.waterfall(self.shap_values[index])

    def plot_force(self, index):
        if self.model == "rdf":
            shap.plots.force(self.explainer.expected_value[1], self.shap_values.values[index],
                             self.data.iloc[index:index + 1], matplotlib=True)
        elif self.model == "rsf":
            shap.force_plot(self.shap_values[index], self.shap_values.base_values, matplotlib=True)

    def plot_decision(self, index):
        if self.model == "rdf":
            shap.decision_plot(self.explainer.expected_value[1], self.shap_values.values[index],
                               self.data.iloc[index:index + 1], highlight=0)
        elif self.model == "rsf":
            shap.decision_plot(self.shap_values.base_values[1], self.shap_values.values[index],
                               self.data.iloc[index:index + 1], highlight=0)

    def plot_heatmap(self):
        shap.plots.heatmap(self.shap_values)

    def plot_scatter(self, category, color):
        shap.plots.scatter(self.shap_values[:, category], color=self.shap_values[:, color])
