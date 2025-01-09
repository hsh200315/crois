import os
import torch
import mlflow
import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                n_epochs=30,
                early_stop_rounds=10):
    """Autoencoder 모델을 학습하고 검증합니다.

    Args:
        model (nn.Module): 학습할 Autoencoder 모델
        train_loader (DataLoader): 훈련 데이터 로더
        val_loader (DataLoader): 검증 데이터 로더
        optimizer (torch.optim.Optimizer): 옵티마이저
        criterion (nn.Module): 손실 함수
        n_epochs (int): 총 학습 에폭 수 (기본값: 30)
        early_stop_rounds (int): 조기 종료를 위한 검증 손실 허용 에폭 수 (기본값: 10)

    Returns:
        tuple: 훈련 손실 목록과 검증 손실 목록
    """
    model.to(device)

    best_loss = float('inf')
    train_losses, val_losses = [], []
    stop_counter = 0

    for epoch in range(n_epochs):
        if stop_counter >= early_stop_rounds:
            print("Early stopping triggered.")
            break

        # Training loop
        model.train()
        train_loss_sum = 0
        for batch in train_loader:
            batch_x = batch[0].cuda()
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / len(train_loader)
        train_losses.append(train_loss)
        mlflow.log_metric("train_loss", train_losses[-1], step=epoch)

        # Validation loop
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch[0].cuda()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                val_loss_sum += loss.item()
        val_loss = val_loss_sum / len(val_loader)
        val_losses.append(val_loss)
        mlflow.log_metric("val_loss", val_losses[-1], step=epoch)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            stop_counter = 0
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (Best model)")
        else:
            stop_counter += 1
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model, train_losses, val_losses


def evaluate_model(model, test_data, th=0.5, dir_path=None, test_tnum=None):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        reconstruction_errors_T1 = torch.mean((outputs - test_data) ** 2, dim=1).cpu().numpy()

    anomalies = reconstruction_errors_T1 > th
    anomaly_indices = np.where(anomalies)[0]
    
    state_t5 = health_index(reconstruction_errors_T1, th, increase_step=1, decrease_step=0.1)
    predicted_labels = np.where(np.array(state_t5)>150, 1, 0)

    # 데이터 길이에 맞게 tool wear 값 interpolate & 라벨링
    toolwear_df = pd.read_csv(os.path.join(dir_path, test_tnum, f'{test_tnum}_all_labels.csv'))

    index = toolwear_df.index
    toolwear = toolwear_df['Tool Wear in (µm)'].values

    f_nearest = interp1d(index, toolwear, kind='nearest')

    index_new = np.linspace(index.min(), index.max(), len(reconstruction_errors_T1))
    toolwear_new = f_nearest(index_new)

    # tool wear 값 250 이상이면 anomaly
    labels = np.where(toolwear_new>=250, 1, 0)
    
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)
    conf_matrix = confusion_matrix(labels, predicted_labels)
    
    

    return (reconstruction_errors_T1, 
            anomaly_indices,
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": conf_matrix
            })


def health_index(scores, th, increase_step=1, decrease_step=0.1):
    """주어진 점수에 기반하여 헬스 지수를 계산합니다. 센서 데이터를 분석하여 건강 상태를 수치로 표현

    Args:
        scores (list): 입력 점수 목록
        th (float): 임계값
        increase_step (float): 점수가 임계값을 초과할 때 증가하는 단계 크기 (기본값: 1)
        decrease_step (float): 점수가 임계값 이하일 때 감소하는 단계 크기 (기본값: 0.1)

    Returns:
        list: 계산된 헬스 지수 목록
    """
    state = 0
    state_values = []
    for score in scores:
        if score > th:
            state += increase_step
        else:
            state = max(0, state - decrease_step)  # Ensure state doesn't go below 0
        state_values.append(state)
    return state_values
