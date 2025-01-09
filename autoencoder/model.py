import torch
import joblib
import mlflow
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.preprocessor = StandardScaler()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        """전처리된 데이터 통과"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


    def save(self, model_path, preprocessor_path):
        torch.save(self.state_dict(), model_path)
        joblib.dump(self.preprocessor, preprocessor_path)


    @classmethod
    def load(cls, model_path, preprocessor_path):
        """저장된 모델과 preprocessor를 불러옵니다.
            함수 호출과 동시에 인스턴스를 생성해야하기 때문에 @classmethod 사용"""
        model = cls(input_dim=1)
        model.load_state_dict(torch.load(model_path))
        model.preprocessor = joblib.load(preprocessor_path)
        return model


class AutoEncoderWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model=None):
        self.model = model
        
    def load_context(self, context):
        model_path = context.artifacts["model"]
        preprocessor_path = context.artifacts["preprocessor"]
        self.model = Autoencoder.load(model_path, preprocessor_path)

        