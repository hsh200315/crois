from ultralytics import YOLO

# 모델 로드
model = YOLO('runs/detect/yolo11_custom_train/weights/best.pt')  # 전이 학습용으로 사전 학습된 모델로드

# 훈련된 모델로 테스트
results = model.val(data='data.yaml')

# 예측 결과 출력
print(results)
