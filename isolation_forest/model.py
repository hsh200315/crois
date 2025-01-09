import torch
import joblib
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class Isolationforest:
    def __init__(self, n_estimators, random_state):
        super(Isolationforest, self).__init__()
        self.preprocessor = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state)

    def fit(self, x):
        return self.model.fit(x)
    
    def predict(self, x):
        return self.model.predict(x)

    def save(self, model_path, preprocessor_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)

