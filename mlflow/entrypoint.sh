#!/bin/sh

while ! mc alias set mlflowminio http://mlflow-artifact-store:9000 minio miniostorage; do
  echo "Waiting for MinIO to be ready..."
  sleep 5
done

# mc config host add mlflowminio http://mlflow-artifact-store:9000 mlflowuser mlflowpassword
mc mb --ignore-existing mlflowminio/mlflow
echo "Created bucket mlflow"

# 정책 추가
# mc admin policy create mlflowminio mlflow-policy /mlflow/bucket-policy.json

# # 사용자 추가
# mc admin user add mlflowminio mlflowuser mlflowpassword

# # 정책 사용자에게 할당
# mc admin policy attach mlflowminio mlflow-policy --user mlflowuser

# 환경 변수 설정
# export MLFLOW_S3_ENDPOINT_URL="http://mlflow-artifact-store:9000"
# export AWS_ACCESS_KEY_ID="minio"
# export AWS_SECRET_ACCESS_KEY="miniostorage"

mkdir -p /mlflow/logs

# 모델 서빙 배포
# python3 /mlflow/code/serve.py > /mlflow/logs/serve.log 2>&1 &

# # 모델 재학습 배포
# python3 /mlflow/code/retrain.py > /mlflow/logs/retrain.log 2>&1 &

# MLFlow 서버 실행
mlflow server \
    --backend-store-uri postgresql://mlflowuser:mlflowpassword@mlflow-backend-store/mlflowdb \
    --artifacts-destination s3://mlflow \
    --default-artifact-root s3://mlflow \
    --host 0.0.0.0 \
    --port 8080

