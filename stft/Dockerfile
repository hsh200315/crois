# 베이스 이미지로 
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 심볼릭 링크 생성
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# pip 업그레이드
RUN python -m pip install --upgrade pip

# pip가 정상적으로 설치되었는지 확인
RUN python -m pip --version

# 기본적인 Python 패키지 설치
RUN pip install --no-cache-dir \
    wheel \
    setuptools

# Python 패키지 설치를 위한 requirements.txt 파일 생성
COPY requirements.txt requirements.txt

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY serve.py app.py
COPY model.py model.py
COPY utils.py utils.py

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV EXPERIMENT_NAME=tool_wear_stft
ENV MLFLOW_TRACKING_URI=http://mlflow-server:8080
ENV MLFLOW_S3_ENDPOINT_URL=http://mlflow-artifact-store:9000
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=miniostorage

# 포트 설정
EXPOSE 8000

# 실행 명령어
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]