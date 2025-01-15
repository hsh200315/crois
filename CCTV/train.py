from ultralytics import YOLO

# 1. YOLO 모델 생성 (사전 학습된 모델 사용)
model = YOLO('yolo11n.pt')  # 전이 학습용으로 사전 학습된 모델로드

# 1.5 (optional)
'''
필요시 freeze 기능 추가
'''

# 2. 모델 훈련
model.train(
    data='data.yaml',          # 데이터셋 설정 파일 경로
    epochs=50,                 # 학습 에포크 수
    batch=16,                  # 배치 크기
    imgsz=640,                 # 입력 이미지 크기
    name='yolo11_custom_train' # 실험 이름 (저장 경로로 사용)
)

'''
data.yaml 내부

train: dataset/images/train/  # 학습 데이터 경로
val: dataset/images/val/      # 검증 데이터 경로

nc: 1                         # 클래스 수 (예: 사람만 학습한다면 1)
names: ['person']             # 클래스 이름 리스트


폴더 구조

dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── val/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── data.yaml

'''