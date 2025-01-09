import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import mlflow.pyfunc
import joblib
import json
from utils import Preprocessor


class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRURegressor, self).__init__()
        self.params = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
        }
        self.gru = nn.GRU(
            self.params["input_size"],
            self.params["hidden_size"],
            self.params["num_layers"],
        )
        self.fc = nn.Linear(self.params["hidden_size"], self.params["output_size"])
        self.preprocessor = Preprocessor()

    def forward(self, x):
        # GRU output
        out, _ = self.gru(x)
        # Fully connected output using the last time step
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, features):
        """원본 Feature 데이터를 입력받아 예측 결과를 반환합니다."""
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            scaled_features = self.preprocessor.transform(features)
            x = torch.FloatTensor(scaled_features)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            x = x.to(device)
            pred = self.forward(x)
            return pred.cpu().numpy()

    def save(self, model_path, preprocessor_path):
        """모델, 전처리기, 하이퍼파라미터 저장"""
        torch.save(
            {
                "model_state": self.state_dict(),
                "model_params": self.params,
            },
            model_path,
        )
        joblib.dump(self.preprocessor, preprocessor_path)

    @classmethod
    def load(cls, model_path, preprocessor_path):
        """저장된 모델과 Preprocessor을 불러옵니다."""
        checkpoint = torch.load(model_path)
        model = cls(**checkpoint["model_params"])
        model.load_state_dict(checkpoint["model_state"])
        model.preprocessor = joblib.load(preprocessor_path)
        return model


class GRUWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model=None):
        self.model = model

    def load_context(self, context):
        model_path = context.artifacts["model"]
        preprocessor_path = context.artifacts["preprocessor"]
        self.model = GRURegressor.load(model_path, preprocessor_path)

    def predict(self, context, inputs):
        return self.model.predict(inputs)


class SVRRegressor:
    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1):
        self.params = {"kernel": kernel, "C": C, "epsilon": epsilon}
        self.model = SVR(**self.params)
        self.preprocessor = Preprocessor()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, features):
        """원본 Feature 데이터를 입력받아 예측 결과를 반환합니다."""
        scaled_features = self.preprocessor.transform(features)
        return self.model.predict(scaled_features)

    def save(self, model_path, preprocessor_path):
        """
        모델, 전처리기, 하이퍼파라미터 저장
        """
        # 모델 저장
        joblib.dump(self.model, model_path)

        # 전처리기와 하이퍼파라미터 저장
        checkpoint = {
            "preprocessor": self.preprocessor,
            "params": self.params,
        }
        joblib.dump(checkpoint, preprocessor_path)

    @classmethod
    def load(cls, model_path, preprocessor_path):
        """
        저장된 모델, 전처리기, 하이퍼파라미터 불러오기
        """
        checkpoint = joblib.load(preprocessor_path)
        preprocessor = checkpoint["preprocessor"]
        params = checkpoint["params"]

        instance = cls(**params)
        model = joblib.load(model_path)
        instance.model = model
        instance.preprocessor = preprocessor

        return instance


class SVRWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model=None):
        self.model = model

    def load_context(self, context):
        model_path = context.artifacts["model"]
        preprocessor_path = context.artifacts["preprocessor"]
        self.model = SVRRegressor.load(model_path, preprocessor_path)

    def predict(self, context, inputs):
        return self.model.predict(inputs)


class RFRRegressor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }
        self.model = RandomForestRegressor(**self.params)
        self.preprocessor = Preprocessor()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, features):
        """원본 Feature 데이터를 입력받아 예측 결과를 반환합니다."""
        scaled_features = self.preprocessor.transform(features)
        return self.model.predict(scaled_features)

    def save(self, model_path, preprocessor_path):
        """
        모델, 전처리기, 하이퍼파라미터 저장
        """
        # 모델 저장
        joblib.dump(self.model, model_path)

        # 전처리기와 하이퍼파라미터 저장
        checkpoint = {
            "preprocessor": self.preprocessor,
            "params": self.params,
        }
        joblib.dump(checkpoint, preprocessor_path)

    @classmethod
    def load(cls, model_path, preprocessor_path):
        """
        저장된 모델, 전처리기, 하이퍼파라미터 불러오기
        """
        checkpoint = joblib.load(preprocessor_path)
        preprocessor = checkpoint["preprocessor"]
        params = checkpoint["params"]

        instance = cls(**params)
        instance.model = joblib.load(model_path)
        instance.preprocessor = preprocessor
        return instance


class RFRWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model=None):
        self.model = model

    def load_context(self, context):
        model_path = context.artifacts["model"]
        preprocessor_path = context.artifacts["preprocessor"]
        self.model = RFRRegressor.load(model_path, preprocessor_path)

    def predict(self, context, inputs):
        return self.model.predict(inputs)


class CatboostRegressor:
    def __init__(
        self, iterations, learning_rate, depth, loss_function, task_type, verbose
    ):
        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "loss_function": loss_function,
            "task_type": task_type,
            "verbose": verbose,
        }
        self.model = CatBoostRegressor(**self.params)
        self.preprocessor = Preprocessor()

    def fit(self, x, y, eval_set=None, verbose=True):
        self.model.fit(x, y, eval_set=eval_set, verbose=verbose)

    def predict(self, x):
        """원본 데이터를 입력받아 예측을 수행 (모델 클래스에서 사용)"""
        scaled_x = self.preprocessor.transform(x)
        return self.model.predict(scaled_x)

    def save(self, model_path, preprocessor_path):
        """
        모델, 전처리기, 하이퍼파라미터 저장
        """
        # 모델 저장
        self.model.save_model(model_path)

        # 전처리기와 하이퍼파라미터 저장
        checkpoint = {
            "preprocessor": self.preprocessor,
            "params": self.params,
        }
        joblib.dump(checkpoint, preprocessor_path)

    @classmethod
    def load(cls, model_path, preprocessor_path):
        """
        저장된 모델, 전처리기, 하이퍼파라미터 불러오기
        """
        checkpoint = joblib.load(preprocessor_path)
        preprocessor = checkpoint["preprocessor"]
        params = checkpoint["params"]

        instance = cls(**params)
        instance.model = CatBoostRegressor(instance.params)
        instance.model.load_model(model_path)
        instance.preprocessor = preprocessor

        return instance


class CatboostWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model=None):
        self.model = model

    def load_context(self, context):
        model_path = context.artifacts["model"]
        preprocessor_path = context.artifacts["preprocessor"]
        self.model = CatboostRegressor.load(model_path, preprocessor_path)

    def predict(self, context, inputs):
        return self.model.predict(inputs)


class XGboostRegressor:
    def __init__(
        self,
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=8,
        objective="reg:squarederror",
        eval_metric="rmse",
        task_type="GPU",
        verbosity=1,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": objective,
            "eval_metric": eval_metric,
            "task_type": task_type,
            "verbosity": verbosity,
        }
        self.model = XGBRegressor(**self.params)
        self.preprocessor = Preprocessor()

    def fit(self, X, y, eval_set=None, verbose=None):
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

    def predict(self, features):
        """원본 Feature 데이터를 입력받아 예측 결과를 반환합니다."""
        scaled_features = self.preprocessor.transform(features)
        return self.model.predict(scaled_features)

    def save(self, path, preprocessor_path):
        """
        모델, 전처리기, 하이퍼파라미터 저장
        """
        # 모델 저장
        joblib.dump(self.model, path)

        # 전처리기와 하이퍼파라미터 저장
        checkpoint = {"preprocessor": self.preprocessor, "params": self.params}
        joblib.dump(checkpoint, preprocessor_path)

    @classmethod
    def load(cls, model_path, preprocessor_path):
        """
        저장된 모델, 전처리기, 하이퍼파라미터 불러오기
        """
        checkpoint = joblib.load(preprocessor_path)
        preprocessor = checkpoint["preprocessor"]
        params = checkpoint["params"]

        instance = cls(**params)
        model = joblib.load(model_path)
        instance.model = model
        instance.preprocessor = preprocessor

        return instance


class XGboostWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        model_path = context.artifacts["model"]
        preprocessor_path = context.artifacts["preprocessor"]

        # 모델 불러오기
        self.model = XGboostRegressor.load(model_path, preprocessor_path)

    def predict(self, context, inputs):
        return self.model.predict(inputs)
