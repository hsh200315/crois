import numpy as np

from model import *
from sklearn.metrics import accuracy_score

def train_model(train_data, n_estimators, random_state, skf):
    
    val_scores = []
    best_clf = None
    best_score = 0
    
    for train_index, val_index in skf.split(train_data, np.zeros(len(train_data))):
        X_train, X_val = train_data[train_index], train_data[val_index]
        
        clf = Isolationforest(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X_train)
        
        y_pred = clf.predict(X_val)
        y_val = np.ones(len(y_pred))
        y_pred = np.where(y_pred == 1, 1, 0)
        
        score = accuracy_score(y_val, y_pred)
        val_scores.append(score)
        
        if score > best_score:
            best_score = score
            best_clf = clf
    
    return best_clf, val_scores

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