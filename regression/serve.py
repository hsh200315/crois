from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import mlflow
import os

load_dotenv()

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

#입력 데이터 모델
class ModelInput(BaseModel):
    features: list[float]

#출력 데이터 모델  
class ModelOutput(BaseModel):
    prediction: float
 
#Metric 기반으로 best model 선택 및 해당 run의 id 반환   
def best_model_run_id(weights: Dict[str, float] = None):
    
    #해당 이름을 가진 experiment 불러오기
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")
    
    #해당 experiment의 모든 run 불러오기
    runs = mlflow.search_runs(experiment.experiment_id)
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{EXPERIMENT_NAME}'.")
    
    metric_cols = [
        "metrics.mae",
        "metrics.evs",
        "metrics.mape",
        "metrics.rmse",
        "metrics.r2",
    ]
    
    #copy안하면 수정 시, 원본 데이터도 수정됨
    metric_df = runs[metric_cols].copy()
    
    # 작을 수록 좋은 metric은 -1, 클수록 좋은 metric은 1로 direction 설정
    metric_directions = {}
    for col in metric_cols:
        if any(x in col.lower() for x in ["mae", "rmse", "mape"]):  # 낮을수록 좋은 지표
            metric_directions[col] = -1
        else:  # 높을수록 좋은 지표
            metric_directions[col] = 1
        
    #column별로 가장 큰 값을 1, 가장 작은 값을 0으로 scaling
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(metric_df.values)
    scaled_metrics_df = pd.DataFrame(scaled_values, columns=metric_cols, index=metric_df.index)
        
    #작을 수록 좋은 값들은 최대 최소 값 바꿈
    for col, direction in metric_directions.items():
        if direction == -1:
            scaled_metrics_df[col] = 1 - scaled_metrics_df[col]
    
    # 가중치 설정 (기본: 동일 가중치)
    if weights is None:
        weights = {name: 1 / len(metric_cols) for name in metric_cols}

    # 5개 metric에 대한 가중합 계산
    scaled_metrics_df["aggregated_score"] = sum(
        scaled_metrics_df[name] * weights[name] for name in metric_cols
    )
    
    best_run_idx = scaled_metrics_df["aggregated_score"].idxmax()
    best_run = runs.loc[best_run_idx]
    best_run_id = best_run["run_id"]
    best_model_name = best_run["tags.model"]
    
    return best_run_id, best_model_name


def load_model():
    run_id, model_name = best_model_run_id()
    model_path = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_path)
    return model, run_id, model_name
    

def cleaning_result(prediction):
    if isinstance(prediction, (list, np.ndarray)):
        while isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
            prediction = prediction[0]
            
    return prediction

async def lifespan(app: FastAPI):
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        global model, run_id, model_name
        model, run_id, model_name = load_model()
        print(f"Model '{model_name}' loaded successfully")
    except Exception as e:
        Exception(f"Failed to load the model: {str(e)}")
        raise e
    yield


#FastAPI instance 생성
app = FastAPI(title="Regression Model Serving API", version="1.0", lifespan=lifespan)


@app.post("/predict", response_model=ModelInput)
async def predict(input_data: ModelOutput):
    """
    입력 데이터를 받아 예측 결과 반환.
    features라는 key를 가진 JSON 형태로 입력 데이터를 받음
    """
    try:
        input_data = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(input_data)
        prediction = cleaning_result(prediction)
        return ModelOutput(prediction=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """
    현재 로드된(서빙되고 있는) 모델의 정보를 반환
    """
    return {
        "model_run_id": run_id,
        "model_name": model_name,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)