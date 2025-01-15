import cv2
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('yolo11n.pt')  # YOLOv11n 모델.

# RTSP URL
cctv_user = "admin"
cctv_password = "12345678!"
rtsp_url = f"rtsp://{cctv_user}:{cctv_password}@192.168.30.99:554"

# RTSP 스트림 열기
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    frame = cv2.resize(frame, (640, 360))

    # YOLO 모델로 감지 수행
    results = model(frame, conf=0.5, classes=[0], verbose=False)  # confidence threshold 설정 가능 (기본값 0.25), 사람 class만 결과 표시, 세부내용 로그(verbose) False

    # results 객체 내부 bbox, cls, inference 속도, confidence 등 다양한 정보 내포

    # 감지 결과를 프레임에 표시
    result_frame = results[0].plot()

    # 화면에 출력
    cv2.imshow("YOLO Detection", result_frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
